#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
连接池性能监控系统

实现连接池优化、性能监控和自动调优功能，确保生产环境的最佳性能。
"""

import os
import sys
import time
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import queue
import json
from contextlib import contextmanager

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor
from db.clickhouse_db import get_clickhouse_db, ClickHouseDbmanager
from config.unified_config_manager import get_config
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class ConnectionPoolMetrics:
    """连接池性能指标"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.total_connections_created = 0
        self.total_connections_destroyed = 0
        self.current_active_connections = 0
        self.current_idle_connections = 0
        self.peak_connections = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_wait_time = 0.0
        self.max_wait_time = 0.0
        self.average_connection_lifetime = 0.0
        self.connection_errors = []
        self.performance_samples = []
        
    def add_performance_sample(self, operation: str, duration: float, success: bool):
        """添加性能样本"""
        sample = {
            'timestamp': time.time(),
            'operation': operation,
            'duration': duration,
            'success': success
        }
        self.performance_samples.append(sample)
        
        # 保留最近1000个样本
        if len(self.performance_samples) > 1000:
            self.performance_samples = self.performance_samples[-1000:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.performance_samples:
            return {}
        
        durations = [s['duration'] for s in self.performance_samples if s['success']]
        success_rate = len([s for s in self.performance_samples if s['success']]) / len(self.performance_samples)
        
        return {
            'total_connections_created': self.total_connections_created,
            'total_connections_destroyed': self.total_connections_destroyed,
            'current_active_connections': self.current_active_connections,
            'current_idle_connections': self.current_idle_connections,
            'peak_connections': self.peak_connections,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': success_rate * 100,
            'average_duration': np.mean(durations) if durations else 0,
            'median_duration': np.median(durations) if durations else 0,
            'p95_duration': np.percentile(durations, 95) if durations else 0,
            'p99_duration': np.percentile(durations, 99) if durations else 0,
            'max_wait_time': self.max_wait_time,
            'total_errors': len(self.connection_errors)
        }


class EnhancedConnectionPool:
    """增强的连接池"""
    
    def __init__(self, max_size: int = 20, min_size: int = 5, 
                 acquire_timeout: float = 30.0, max_lifetime: float = 3600.0):
        self.max_size = max_size
        self.min_size = min_size
        self.acquire_timeout = acquire_timeout
        self.max_lifetime = max_lifetime
        
        self._pool = queue.Queue(maxsize=max_size)
        self._all_connections = {}
        self._lock = threading.RLock()
        self._created_count = 0
        self._metrics = ConnectionPoolMetrics()
        
        # 创建初始连接
        self._initialize_pool()
        
        # 启动维护线程
        self._start_maintenance_thread()
        
    def _initialize_pool(self):
        """初始化连接池"""
        for _ in range(self.min_size):
            try:
                conn = self._create_connection()
                self._pool.put_nowait(conn)
            except Exception as e:
                logger.error(f"初始化连接池失败: {e}")
    
    def _create_connection(self):
        """创建新连接"""
        with self._lock:
            conn_id = f"conn_{self._created_count}_{int(time.time())}"
            self._created_count += 1
            
            try:
                db_conn = get_clickhouse_db()
                connection_info = {
                    'id': conn_id,
                    'connection': db_conn,
                    'created_at': time.time(),
                    'last_used': time.time(),
                    'use_count': 0
                }
                
                self._all_connections[conn_id] = connection_info
                self._metrics.total_connections_created += 1
                self._metrics.current_idle_connections += 1
                
                if len(self._all_connections) > self._metrics.peak_connections:
                    self._metrics.peak_connections = len(self._all_connections)
                
                logger.debug(f"创建新连接: {conn_id}")
                return connection_info
                
            except Exception as e:
                logger.error(f"创建连接失败: {e}")
                self._metrics.connection_errors.append({
                    'timestamp': time.time(),
                    'error': str(e),
                    'operation': 'create_connection'
                })
                raise
    
    def _destroy_connection(self, conn_info: Dict[str, Any]):
        """销毁连接"""
        with self._lock:
            conn_id = conn_info['id']
            if conn_id in self._all_connections:
                del self._all_connections[conn_id]
                self._metrics.total_connections_destroyed += 1
                self._metrics.current_idle_connections -= 1
                logger.debug(f"销毁连接: {conn_id}")
    
    @contextmanager
    def acquire_connection(self):
        """获取连接"""
        start_time = time.time()
        connection_info = None
        
        try:
            self._metrics.total_requests += 1
            
            # 尝试从池中获取连接
            try:
                connection_info = self._pool.get(timeout=self.acquire_timeout)
            except queue.Empty:
                # 如果没有可用连接且未达到最大连接数，创建新连接
                with self._lock:
                    if len(self._all_connections) < self.max_size:
                        connection_info = self._create_connection()
                    else:
                        raise RuntimeError("连接池已满，无法获取连接")
            
            # 检查连接是否过期
            if time.time() - connection_info['created_at'] > self.max_lifetime:
                self._destroy_connection(connection_info)
                connection_info = self._create_connection()
            
            # 更新连接使用信息
            with self._lock:
                connection_info['last_used'] = time.time()
                connection_info['use_count'] += 1
                self._metrics.current_active_connections += 1
                self._metrics.current_idle_connections -= 1
            
            wait_time = time.time() - start_time
            self._metrics.total_wait_time += wait_time
            if wait_time > self._metrics.max_wait_time:
                self._metrics.max_wait_time = wait_time
            
            yield connection_info['connection']
            
            self._metrics.successful_requests += 1
            self._metrics.add_performance_sample('acquire_connection', wait_time, True)
            
        except Exception as e:
            self._metrics.failed_requests += 1
            self._metrics.add_performance_sample('acquire_connection', time.time() - start_time, False)
            self._metrics.connection_errors.append({
                'timestamp': time.time(),
                'error': str(e),
                'operation': 'acquire_connection'
            })
            logger.error(f"获取连接失败: {e}")
            raise
        finally:
            # 归还连接
            if connection_info:
                try:
                    self._pool.put_nowait(connection_info)
                    with self._lock:
                        self._metrics.current_active_connections -= 1
                        self._metrics.current_idle_connections += 1
                except queue.Full:
                    # 队列已满，销毁连接
                    self._destroy_connection(connection_info)
    
    def _start_maintenance_thread(self):
        """启动维护线程"""
        def maintenance_task():
            while True:
                try:
                    time.sleep(60)  # 每分钟执行一次维护
                    self._cleanup_expired_connections()
                    self._ensure_min_connections()
                except Exception as e:
                    logger.error(f"连接池维护任务错误: {e}")
        
        maintenance_thread = threading.Thread(target=maintenance_task, daemon=True)
        maintenance_thread.start()
    
    def _cleanup_expired_connections(self):
        """清理过期连接"""
        current_time = time.time()
        expired_connections = []
        
        with self._lock:
            for conn_id, conn_info in self._all_connections.items():
                if (current_time - conn_info['last_used'] > 300 and  # 5分钟未使用
                    len(self._all_connections) > self.min_size):
                    expired_connections.append(conn_info)
        
        for conn_info in expired_connections:
            try:
                # 尝试从队列中移除
                temp_queue = queue.Queue()
                while not self._pool.empty():
                    item = self._pool.get_nowait()
                    if item['id'] != conn_info['id']:
                        temp_queue.put(item)
                
                # 重新放入队列
                while not temp_queue.empty():
                    self._pool.put_nowait(temp_queue.get_nowait())
                
                self._destroy_connection(conn_info)
                
            except Exception as e:
                logger.warning(f"清理过期连接失败: {e}")
    
    def _ensure_min_connections(self):
        """确保最小连接数"""
        with self._lock:
            current_count = len(self._all_connections)
            if current_count < self.min_size:
                for _ in range(self.min_size - current_count):
                    try:
                        conn = self._create_connection()
                        self._pool.put_nowait(conn)
                    except Exception as e:
                        logger.error(f"创建最小连接失败: {e}")
                        break
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取连接池指标"""
        return self._metrics.get_statistics()
    
    def close(self):
        """关闭连接池"""
        with self._lock:
            # 销毁所有连接
            for conn_info in list(self._all_connections.values()):
                self._destroy_connection(conn_info)
            
            # 清空队列
            while not self._pool.empty():
                try:
                    self._pool.get_nowait()
                except queue.Empty:
                    break


class ConnectionPoolPerformanceMonitor:
    """连接池性能监控器"""
    
    def __init__(self):
        self.connection_pool = None
        self.monitoring_active = False
        self.performance_data = []
        self.system_metrics = []
        self.monitoring_thread = None
        
    @exception_handler(reraise=True)
    def initialize_optimized_pool(self, max_size: int = 20, min_size: int = 5) -> bool:
        """初始化优化的连接池"""
        try:
            logger.info(f"初始化优化连接池: max_size={max_size}, min_size={min_size}")
            
            self.connection_pool = EnhancedConnectionPool(
                max_size=max_size,
                min_size=min_size,
                acquire_timeout=30.0,
                max_lifetime=3600.0
            )
            
            # 测试连接池
            with self.connection_pool.acquire_connection() as conn:
                result = conn.query("SELECT 1 as test")
                if not result.empty:
                    logger.info("✅ 优化连接池初始化成功")
                    return True
                
        except Exception as e:
            logger.error(f"❌ 初始化优化连接池失败: {e}")
            return False
        
        return False
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def test_connection_pool_performance(self, num_threads: int = 10, 
                                       queries_per_thread: int = 50) -> Dict[str, Any]:
        """测试连接池性能"""
        if not self.connection_pool:
            raise RuntimeError("连接池未初始化")
        
        result = {
            'num_threads': num_threads,
            'queries_per_thread': queries_per_thread,
            'total_queries': num_threads * queries_per_thread,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_time': 0.0,
            'query_times': [],
            'connection_metrics': {},
            'thread_results': []
        }
        
        def execute_queries(thread_id: int) -> Dict[str, Any]:
            """执行查询的线程函数"""
            thread_result = {
                'thread_id': thread_id,
                'successful_queries': 0,
                'failed_queries': 0,
                'query_times': [],
                'errors': []
            }
            
            for i in range(queries_per_thread):
                query_start_time = time.time()
                
                try:
                    with self.connection_pool.acquire_connection() as conn:
                        # 执行简单查询
                        query = f"""
                        SELECT COUNT(*) as count
                        FROM stock.stock_info
                        WHERE level = '日线'
                        LIMIT 1
                        """
                        
                        query_result = conn.query(query)
                        query_time = time.time() - query_start_time
                        
                        if not query_result.empty:
                            thread_result['successful_queries'] += 1
                            thread_result['query_times'].append(query_time)
                        else:
                            thread_result['failed_queries'] += 1
                            
                except Exception as e:
                    thread_result['failed_queries'] += 1
                    thread_result['errors'].append(str(e))
                    logger.warning(f"线程 {thread_id} 查询 {i+1} 失败: {e}")
            
            return thread_result
        
        start_time = time.time()
        
        try:
            # 启动性能监控
            self.start_monitoring()
            
            # 执行并发查询
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(execute_queries, i) for i in range(num_threads)]
                
                for future in as_completed(futures):
                    try:
                        thread_result = future.result()
                        result['thread_results'].append(thread_result)
                        result['successful_queries'] += thread_result['successful_queries']
                        result['failed_queries'] += thread_result['failed_queries']
                        result['query_times'].extend(thread_result['query_times'])
                        
                    except Exception as e:
                        logger.error(f"线程执行异常: {e}")
                        result['failed_queries'] += queries_per_thread
            
            result['total_time'] = time.time() - start_time
            
            # 获取连接池指标
            result['connection_metrics'] = self.connection_pool.get_metrics()
            
            # 停止监控
            self.stop_monitoring()
            
            # 计算性能统计
            if result['query_times']:
                result['average_query_time'] = np.mean(result['query_times'])
                result['median_query_time'] = np.median(result['query_times'])
                result['p95_query_time'] = np.percentile(result['query_times'], 95)
                result['p99_query_time'] = np.percentile(result['query_times'], 99)
                result['queries_per_second'] = len(result['query_times']) / result['total_time']
            
            logger.info(f"✅ 连接池性能测试完成: {result['successful_queries']}/{result['total_queries']} 成功")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 连接池性能测试失败: {e}")
            result['total_time'] = time.time() - start_time
            return result
        finally:
            self.stop_monitoring()
    
    def start_monitoring(self) -> None:
        """开始性能监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.performance_data = []
        self.system_metrics = []
        
        def monitoring_task():
            """监控任务"""
            while self.monitoring_active:
                try:
                    # 收集系统指标
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    
                    system_metric = {
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_used_gb': memory.used / (1024**3),
                        'memory_available_gb': memory.available / (1024**3)
                    }
                    
                    self.system_metrics.append(system_metric)
                    
                    # 收集连接池指标
                    if self.connection_pool:
                        pool_metrics = self.connection_pool.get_metrics()
                        pool_metrics['timestamp'] = time.time()
                        self.performance_data.append(pool_metrics)
                    
                    time.sleep(1)  # 每秒采集一次
                    
                except Exception as e:
                    logger.warning(f"监控任务异常: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitoring_task, daemon=True)
        self.monitoring_thread.start()
        logger.info("✅ 性能监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止性能监控"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        logger.info("✅ 性能监控已停止")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=300.0)
    def test_concurrent_load(self, max_concurrent_connections: int = 50) -> Dict[str, Any]:
        """测试并发负载能力"""
        if not self.connection_pool:
            raise RuntimeError("连接池未初始化")
        
        result = {
            'max_concurrent_connections': max_concurrent_connections,
            'successful_connections': 0,
            'failed_connections': 0,
            'connection_times': [],
            'peak_active_connections': 0,
            'total_time': 0.0
        }
        
        def test_connection():
            """测试单个连接"""
            connection_start_time = time.time()
            
            try:
                with self.connection_pool.acquire_connection() as conn:
                    # 模拟一些工作负载
                    query_result = conn.query("SELECT 1 as test")
                    time.sleep(0.1)  # 模拟处理时间
                    
                    connection_time = time.time() - connection_start_time
                    return {'success': True, 'time': connection_time}
                    
            except Exception as e:
                connection_time = time.time() - connection_start_time
                return {'success': False, 'time': connection_time, 'error': str(e)}
        
        start_time = time.time()
        
        try:
            self.start_monitoring()
            
            # 启动并发连接测试
            with ThreadPoolExecutor(max_workers=max_concurrent_connections) as executor:
                futures = [executor.submit(test_connection) for _ in range(max_concurrent_connections)]
                
                for future in as_completed(futures):
                    try:
                        conn_result = future.result()
                        result['connection_times'].append(conn_result['time'])
                        
                        if conn_result['success']:
                            result['successful_connections'] += 1
                        else:
                            result['failed_connections'] += 1
                            
                    except Exception as e:
                        result['failed_connections'] += 1
                        logger.error(f"并发连接测试异常: {e}")
            
            result['total_time'] = time.time() - start_time
            
            # 获取峰值连接数
            if self.performance_data:
                result['peak_active_connections'] = max(
                    data.get('current_active_connections', 0) for data in self.performance_data
                )
            
            self.stop_monitoring()
            
            logger.info(f"✅ 并发负载测试完成: {result['successful_connections']}/{max_concurrent_connections} 成功")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 并发负载测试失败: {e}")
            result['total_time'] = time.time() - start_time
            return result
        finally:
            self.stop_monitoring()
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=1800.0)
    def run_comprehensive_monitoring_test(self) -> Dict[str, Any]:
        """运行综合监控测试"""
        start_time = datetime.now()
        
        comprehensive_result = {
            'test_start_time': start_time.isoformat(),
            'test_end_time': None,
            'total_test_time': 0.0,
            'pool_initialization': {},
            'basic_performance_test': {},
            'concurrent_load_test': {},
            'system_performance': {},
            'optimization_recommendations': [],
            'overall_grade': 'Unknown'
        }
        
        try:
            logger.info("🚀 开始连接池性能监控综合测试...")
            
            # 1. 初始化优化连接池
            logger.info("📝 Step 1: 初始化优化连接池...")
            pool_init_success = self.initialize_optimized_pool(max_size=20, min_size=5)
            comprehensive_result['pool_initialization'] = {
                'success': pool_init_success,
                'max_size': 20,
                'min_size': 5
            }
            
            if not pool_init_success:
                comprehensive_result['optimization_recommendations'].append("连接池初始化失败，需要检查数据库配置")
                return comprehensive_result
            
            # 2. 基础性能测试
            logger.info("📝 Step 2: 执行基础性能测试...")
            comprehensive_result['basic_performance_test'] = self.test_connection_pool_performance(
                num_threads=5, queries_per_thread=20
            )
            
            # 3. 并发负载测试
            logger.info("📝 Step 3: 执行并发负载测试...")
            comprehensive_result['concurrent_load_test'] = self.test_concurrent_load(
                max_concurrent_connections=30
            )
            
            # 4. 系统性能分析
            logger.info("📝 Step 4: 分析系统性能...")
            comprehensive_result['system_performance'] = self._analyze_system_performance()
            
            # 5. 生成优化建议
            logger.info("📝 Step 5: 生成优化建议...")
            self._generate_optimization_recommendations(comprehensive_result)
            
            end_time = datetime.now()
            comprehensive_result['test_end_time'] = end_time.isoformat()
            comprehensive_result['total_test_time'] = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ 连接池监控综合测试完成，总耗时: {comprehensive_result['total_test_time']:.1f}秒")
            logger.info(f"📊 综合评级: {comprehensive_result['overall_grade']}")
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"❌ 综合监控测试失败: {e}")
            comprehensive_result['optimization_recommendations'].append(f"测试过程异常: {str(e)}")
            return comprehensive_result
        finally:
            # 清理资源
            if self.connection_pool:
                self.connection_pool.close()
    
    def _analyze_system_performance(self) -> Dict[str, Any]:
        """分析系统性能"""
        if not self.system_metrics:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.system_metrics]
        memory_values = [m['memory_percent'] for m in self.system_metrics]
        
        return {
            'cpu_usage': {
                'average': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'p95': np.percentile(cpu_values, 95)
            },
            'memory_usage': {
                'average': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'p95': np.percentile(memory_values, 95)
            },
            'samples_collected': len(self.system_metrics)
        }
    
    def _generate_optimization_recommendations(self, result: Dict[str, Any]) -> None:
        """生成优化建议"""
        recommendations = []
        score = 0
        
        # 分析基础性能测试
        basic_test = result.get('basic_performance_test', {})
        if basic_test:
            success_rate = (basic_test.get('successful_queries', 0) / 
                          basic_test.get('total_queries', 1)) * 100
            
            if success_rate >= 99:
                score += 30
                recommendations.append("基础性能测试表现优秀")
            elif success_rate >= 95:
                score += 25
                recommendations.append("基础性能测试表现良好")
            else:
                score += 10
                recommendations.append("基础性能测试需要优化，建议检查连接池配置")
            
            # 分析查询性能
            avg_query_time = basic_test.get('average_query_time', 0)
            if avg_query_time < 0.1:
                score += 20
            elif avg_query_time < 0.5:
                score += 15
                recommendations.append("查询性能可以进一步优化")
            else:
                score += 5
                recommendations.append("查询性能较慢，建议优化查询或增加连接池大小")
        
        # 分析并发负载测试
        concurrent_test = result.get('concurrent_load_test', {})
        if concurrent_test:
            success_rate = (concurrent_test.get('successful_connections', 0) / 
                          concurrent_test.get('max_concurrent_connections', 1)) * 100
            
            if success_rate >= 95:
                score += 30
                recommendations.append("并发负载处理能力优秀")
            elif success_rate >= 85:
                score += 25
                recommendations.append("并发负载处理能力良好")
            else:
                score += 10
                recommendations.append("并发负载处理能力需要提升，建议增加连接池大小")
        
        # 分析系统性能
        system_perf = result.get('system_performance', {})
        if system_perf:
            cpu_avg = system_perf.get('cpu_usage', {}).get('average', 0)
            memory_avg = system_perf.get('memory_usage', {}).get('average', 0)
            
            if cpu_avg < 50 and memory_avg < 70:
                score += 20
                recommendations.append("系统资源使用合理")
            elif cpu_avg < 80 and memory_avg < 85:
                score += 15
                recommendations.append("系统资源使用正常，可继续监控")
            else:
                score += 5
                recommendations.append("系统资源使用较高，建议优化或增加硬件资源")
        
        # 确定综合评级
        if score >= 85:
            result['overall_grade'] = 'A+'
        elif score >= 75:
            result['overall_grade'] = 'A'
        elif score >= 65:
            result['overall_grade'] = 'B+'
        elif score >= 55:
            result['overall_grade'] = 'B'
        else:
            result['overall_grade'] = 'C'
        
        result['optimization_recommendations'] = recommendations
    
    def generate_report(self, result: Dict[str, Any]) -> str:
        """生成监控报告"""
        report_lines = [
            "=" * 80,
            "连接池性能监控综合报告",
            "=" * 80,
            f"测试时间: {result.get('test_start_time', 'Unknown')}",
            f"测试耗时: {result.get('total_test_time', 0):.1f}秒",
            f"综合评级: {result.get('overall_grade', 'Unknown')}",
            "",
            "📊 详细测试结果:",
            "-" * 40,
        ]
        
        # 连接池初始化
        pool_init = result.get('pool_initialization', {})
        if pool_init:
            status = "✅ 成功" if pool_init.get('success', False) else "❌ 失败"
            report_lines.extend([
                f"连接池初始化: {status}",
                f"  最大连接数: {pool_init.get('max_size', 0)}",
                f"  最小连接数: {pool_init.get('min_size', 0)}"
            ])
        
        # 基础性能测试
        basic_test = result.get('basic_performance_test', {})
        if basic_test:
            report_lines.extend([
                "",
                "📈 基础性能测试:",
                f"  总查询数: {basic_test.get('total_queries', 0)}",
                f"  成功查询: {basic_test.get('successful_queries', 0)}",
                f"  失败查询: {basic_test.get('failed_queries', 0)}",
                f"  平均查询时间: {basic_test.get('average_query_time', 0):.3f}秒",
                f"  P95查询时间: {basic_test.get('p95_query_time', 0):.3f}秒",
                f"  查询TPS: {basic_test.get('queries_per_second', 0):.1f}"
            ])
        
        # 并发负载测试
        concurrent_test = result.get('concurrent_load_test', {})
        if concurrent_test:
            report_lines.extend([
                "",
                "⚡ 并发负载测试:",
                f"  最大并发连接: {concurrent_test.get('max_concurrent_connections', 0)}",
                f"  成功连接: {concurrent_test.get('successful_connections', 0)}",
                f"  失败连接: {concurrent_test.get('failed_connections', 0)}",
                f"  峰值活跃连接: {concurrent_test.get('peak_active_connections', 0)}"
            ])
        
        # 系统性能
        system_perf = result.get('system_performance', {})
        if system_perf:
            cpu_usage = system_perf.get('cpu_usage', {})
            memory_usage = system_perf.get('memory_usage', {})
            
            report_lines.extend([
                "",
                "💻 系统性能:",
                f"  CPU使用率: 平均 {cpu_usage.get('average', 0):.1f}%, 最大 {cpu_usage.get('max', 0):.1f}%",
                f"  内存使用率: 平均 {memory_usage.get('average', 0):.1f}%, 最大 {memory_usage.get('max', 0):.1f}%"
            ])
        
        # 优化建议
        if result.get('optimization_recommendations'):
            report_lines.extend([
                "",
                "💡 优化建议:"
            ])
            for i, rec in enumerate(result['optimization_recommendations'], 1):
                report_lines.append(f"{i}. {rec}")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    print("🚀 启动连接池性能监控综合测试...")
    
    monitor = ConnectionPoolPerformanceMonitor()
    
    try:
        # 运行综合监控测试
        result = monitor.run_comprehensive_monitoring_test()
        
        # 生成报告
        report = monitor.generate_report(result)
        print(report)
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"connection_pool_performance_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        
        # 返回测试结果
        if result.get('overall_grade') in ['A+', 'A']:
            print("🎉 连接池性能监控测试通过！")
            return 0
        else:
            print("⚠️  建议根据优化建议改进连接池配置。")
            return 1
            
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 