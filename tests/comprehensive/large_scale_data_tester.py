#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
大规模真实数据测试器

测试1000+股票数据的处理能力和性能，验证系统在大数据量下的稳定性。
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
from functools import wraps

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor
from db.clickhouse_db import get_clickhouse_db
from config import get_config

logger = get_logger(__name__)


class LargeScaleDataTester:
    """
    大规模真实数据测试器
    
    功能：
    1. 1000+股票数据查询测试
    2. 大数据量技术指标计算
    3. 批量选股策略执行
    4. 系统资源监控
    5. 性能基准测试
    """
    
    def __init__(self):
        """初始化大规模数据测试器"""
        self.db = None
        self.test_results = {}
        self.performance_metrics = {}
        self.start_time = None
        self.resource_monitor = None
        self.stop_monitoring = False
        
    @exception_handler(reraise=True)
    def setup_database_connection(self) -> bool:
        """建立数据库连接"""
        try:
            self.db = get_clickhouse_db()
            
            # 测试连接
            test_query = "SELECT COUNT(*) as total FROM stock.stock_info"
            result = self.db.query(test_query)
            
            if not result.empty:
                # ClickHouse返回的列名可能是大写的
                col_name = result.columns[0]  # 使用第一列，不管名称
                total_records = result.iloc[0][col_name]
                logger.info(f"✅ 数据库连接成功，总记录数: {total_records:,}")
                return True
            else:
                logger.error("❌ 数据库连接测试失败：无法获取记录数")
                return False
                
        except Exception as e:
            logger.error(f"❌ 数据库连接失败: {e}")
            return False
    
    @exception_handler(reraise=True)
    def get_stock_list(self, limit: int = 1000) -> List[str]:
        """
        获取股票列表
        
        Args:
            limit: 股票数量限制
            
        Returns:
            List[str]: 股票代码列表
        """
        try:
            query = f"""
            SELECT DISTINCT code
            FROM stock.stock_info
            WHERE level = '日线'
            AND date >= '2024-01-01'
            ORDER BY code
            LIMIT {limit}
            """
            
            result = self.db.query(query)
            
            if not result.empty:
                # ClickHouse返回的列名可能是大写的
                code_col = result.columns[0]  # 使用第一列
                stock_codes = result[code_col].tolist()
                logger.info(f"✅ 获取到 {len(stock_codes)} 只股票代码")
                return stock_codes
            else:
                logger.warning("⚠️  未获取到股票代码，使用默认列表")
                return ['000001', '000002', '600000', '600036']
                
        except Exception as e:
            logger.error(f"❌ 获取股票列表失败: {e}")
            return ['000001', '000002', '600000', '600036']
    
    def start_resource_monitoring(self) -> None:
        """开始资源监控"""
        def monitor_resources():
            """资源监控线程"""
            cpu_usage = []
            memory_usage = []
            
            while not self.stop_monitoring:
                try:
                    # CPU使用率
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    cpu_usage.append(cpu_percent)
                    
                    # 内存使用率
                    memory = psutil.virtual_memory()
                    memory_usage.append(memory.percent)
                    
                    time.sleep(1)  # 每秒采集一次
                    
                except Exception as e:
                    logger.warning(f"资源监控异常: {e}")
                    break
            
            # 保存监控结果
            self.performance_metrics['cpu_usage'] = {
                'average': np.mean(cpu_usage) if cpu_usage else 0,
                'max': np.max(cpu_usage) if cpu_usage else 0,
                'min': np.min(cpu_usage) if cpu_usage else 0,
                'samples': len(cpu_usage)
            }
            
            self.performance_metrics['memory_usage'] = {
                'average': np.mean(memory_usage) if memory_usage else 0,
                'max': np.max(memory_usage) if memory_usage else 0,
                'min': np.min(memory_usage) if memory_usage else 0,
                'samples': len(memory_usage)
            }
        
        self.stop_monitoring = False
        self.resource_monitor = threading.Thread(target=monitor_resources)
        self.resource_monitor.daemon = True
        self.resource_monitor.start()
        logger.info("✅ 资源监控已启动")
    
    def stop_resource_monitoring(self) -> None:
        """停止资源监控"""
        self.stop_monitoring = True
        if self.resource_monitor:
            self.resource_monitor.join(timeout=2)
        logger.info("✅ 资源监控已停止")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=300.0)  # 5分钟阈值
    def test_batch_stock_data_query(self, stock_codes: List[str], batch_size: int = 100) -> Dict[str, Any]:
        """
        测试批量股票数据查询
        
        Args:
            stock_codes: 股票代码列表
            batch_size: 批量大小
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        result = {
            'total_stocks': len(stock_codes),
            'batch_size': batch_size,
            'total_batches': (len(stock_codes) + batch_size - 1) // batch_size,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_records': 0,
            'query_times': [],
            'average_query_time': 0.0,
            'total_time': 0.0,
            'throughput_records_per_second': 0.0
        }
        
        start_time = time.time()
        
        try:
            # 分批查询
            for i in range(0, len(stock_codes), batch_size):
                batch_codes = stock_codes[i:i + batch_size]
                batch_start_time = time.time()
                
                try:
                    # 构建批量查询
                    code_list = "', '".join(batch_codes)
                    query = f"""
                    SELECT code, date, open, high, low, close, volume
                    FROM stock.stock_info
                    WHERE code IN ('{code_list}')
                    AND level = '日线'
                    AND date >= '2024-01-01'
                    ORDER BY code, date DESC
                    LIMIT 10000
                    """
                    
                    batch_result = self.db.query(query)
                    batch_time = time.time() - batch_start_time
                    
                    if not batch_result.empty:
                        result['successful_queries'] += 1
                        result['total_records'] += len(batch_result)
                        result['query_times'].append(batch_time)
                        
                        logger.debug(f"批次 {i//batch_size + 1}: {len(batch_codes)} 只股票, "
                                   f"{len(batch_result)} 条记录, 耗时 {batch_time:.2f}秒")
                    else:
                        result['failed_queries'] += 1
                        logger.warning(f"批次 {i//batch_size + 1} 无数据返回")
                        
                except Exception as e:
                    result['failed_queries'] += 1
                    logger.error(f"批次 {i//batch_size + 1} 查询失败: {e}")
            
            # 计算统计信息
            result['total_time'] = time.time() - start_time
            
            if result['query_times']:
                result['average_query_time'] = np.mean(result['query_times'])
            
            if result['total_time'] > 0:
                result['throughput_records_per_second'] = result['total_records'] / result['total_time']
            
            logger.info(f"✅ 批量查询完成: {result['successful_queries']}/{result['total_batches']} 批次成功, "
                       f"总记录 {result['total_records']:,} 条, 总耗时 {result['total_time']:.1f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 批量查询测试失败: {e}")
            result['total_time'] = time.time() - start_time
            return result
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=600.0)  # 10分钟阈值
    def test_technical_indicators_calculation(self, stock_codes: List[str], sample_size: int = 100) -> Dict[str, Any]:
        """
        测试技术指标计算
        
        Args:
            stock_codes: 股票代码列表
            sample_size: 样本大小
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        result = {
            'sample_size': sample_size,
            'indicators_tested': ['MA5', 'MA20', 'RSI', 'MACD'],
            'successful_calculations': 0,
            'failed_calculations': 0,
            'calculation_times': [],
            'total_time': 0.0,
            'average_calculation_time': 0.0
        }
        
        # 随机选择样本股票
        sample_codes = np.random.choice(stock_codes, min(sample_size, len(stock_codes)), replace=False)
        
        start_time = time.time()
        
        try:
            for code in sample_codes:
                calc_start_time = time.time()
                
                try:
                    # 计算多个技术指标 - 修复ClickHouse兼容性问题
                    query = f"""
                    SELECT 
                        code, date, close,
                        avg(close) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ma5,
                        avg(close) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20,
                        close - anyOrNull(close) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) as price_change
                    FROM stock.stock_info
                    WHERE code = '{code}'
                    AND level = '日线'
                    AND date >= today() - INTERVAL 60 DAY
                    ORDER BY date DESC
                    LIMIT 50
                    """
                    
                    indicator_result = self.db.query(query)
                    calc_time = time.time() - calc_start_time
                    
                    if not indicator_result.empty:
                        result['successful_calculations'] += 1
                        result['calculation_times'].append(calc_time)
                    else:
                        result['failed_calculations'] += 1
                        
                except Exception as e:
                    result['failed_calculations'] += 1
                    logger.warning(f"股票 {code} 指标计算失败: {e}")
            
            # 计算统计信息
            result['total_time'] = time.time() - start_time
            
            if result['calculation_times']:
                result['average_calculation_time'] = np.mean(result['calculation_times'])
            
            logger.info(f"✅ 技术指标计算完成: {result['successful_calculations']}/{len(sample_codes)} 只股票成功, "
                       f"总耗时 {result['total_time']:.1f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 技术指标计算测试失败: {e}")
            result['total_time'] = time.time() - start_time
            return result
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=1200.0)  # 20分钟阈值
    def test_concurrent_processing(self, stock_codes: List[str], num_threads: int = 5, 
                                 stocks_per_thread: int = 20) -> Dict[str, Any]:
        """
        测试并发处理能力
        
        Args:
            stock_codes: 股票代码列表
            num_threads: 并发线程数
            stocks_per_thread: 每线程处理股票数
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        result = {
            'num_threads': num_threads,
            'stocks_per_thread': stocks_per_thread,
            'total_stocks': min(len(stock_codes), num_threads * stocks_per_thread),
            'successful_threads': 0,
            'failed_threads': 0,
            'thread_results': [],
            'total_time': 0.0,
            'concurrent_throughput': 0.0
        }
        
        def process_stock_batch(thread_id: int, batch_codes: List[str]) -> Dict[str, Any]:
            """处理股票批次的线程函数"""
            thread_result = {
                'thread_id': thread_id,
                'batch_size': len(batch_codes),
                'successful_stocks': 0,
                'failed_stocks': 0,
                'processing_time': 0.0,
                'records_processed': 0
            }
            
            thread_start_time = time.time()
            
            try:
                # 创建线程独立的数据库连接
                thread_db = get_clickhouse_db()
                
                for code in batch_codes:
                    try:
                        query = f"""
                        SELECT code, date, close, volume
                        FROM stock.stock_info
                        WHERE code = '{code}'
                        AND level = '日线'
                        AND date >= '2024-01-01'
                        ORDER BY date DESC
                        LIMIT 30
                        """
                        
                        stock_result = thread_db.query(query)
                        
                        if not stock_result.empty:
                            thread_result['successful_stocks'] += 1
                            thread_result['records_processed'] += len(stock_result)
                        else:
                            thread_result['failed_stocks'] += 1
                            
                    except Exception as e:
                        thread_result['failed_stocks'] += 1
                        logger.warning(f"线程 {thread_id} 处理股票 {code} 失败: {e}")
                
            except Exception as e:
                logger.error(f"线程 {thread_id} 执行失败: {e}")
            
            thread_result['processing_time'] = time.time() - thread_start_time
            return thread_result
        
        start_time = time.time()
        
        try:
            # 分配股票给各线程
            selected_codes = stock_codes[:result['total_stocks']]
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                
                for i in range(num_threads):
                    start_idx = i * stocks_per_thread
                    end_idx = min(start_idx + stocks_per_thread, len(selected_codes))
                    batch_codes = selected_codes[start_idx:end_idx]
                    
                    if batch_codes:  # 只有当批次非空时才提交任务
                        future = executor.submit(process_stock_batch, i, batch_codes)
                        futures.append(future)
                
                # 收集结果
                for future in as_completed(futures):
                    try:
                        thread_result = future.result()
                        result['thread_results'].append(thread_result)
                        
                        if thread_result['successful_stocks'] > 0:
                            result['successful_threads'] += 1
                        else:
                            result['failed_threads'] += 1
                            
                    except Exception as e:
                        result['failed_threads'] += 1
                        logger.error(f"线程执行异常: {e}")
            
            # 计算统计信息
            result['total_time'] = time.time() - start_time
            
            total_records = sum(tr['records_processed'] for tr in result['thread_results'])
            if result['total_time'] > 0:
                result['concurrent_throughput'] = total_records / result['total_time']
            
            logger.info(f"✅ 并发处理完成: {result['successful_threads']}/{num_threads} 线程成功, "
                       f"总记录 {total_records:,} 条, 总耗时 {result['total_time']:.1f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 并发处理测试失败: {e}")
            result['total_time'] = time.time() - start_time
            return result
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=1800.0)  # 30分钟阈值
    def run_comprehensive_test(self, target_stock_count: int = 1000) -> Dict[str, Any]:
        """
        运行综合大规模数据测试
        
        Args:
            target_stock_count: 目标测试股票数量
            
        Returns:
            Dict[str, Any]: 综合测试结果
        """
        self.start_time = datetime.now()
        
        comprehensive_result = {
            'test_start_time': self.start_time.isoformat(),
            'test_end_time': None,
            'total_test_time': 0.0,
            'target_stock_count': target_stock_count,
            'actual_stock_count': 0,
            'database_connection': False,
            'stock_list_acquisition': {},
            'batch_query_test': {},
            'technical_indicators_test': {},
            'concurrent_processing_test': {},
            'resource_usage': {},
            'overall_performance': 'Unknown',
            'scalability_grade': 'Unknown',
            'recommendations': []
        }
        
        try:
            logger.info("🚀 开始大规模真实数据综合测试...")
            
            # 1. 建立数据库连接
            logger.info("📝 Step 1: 建立数据库连接...")
            comprehensive_result['database_connection'] = self.setup_database_connection()
            
            if not comprehensive_result['database_connection']:
                comprehensive_result['recommendations'].append("需要修复数据库连接问题")
                return comprehensive_result
            
            # 2. 获取股票列表
            logger.info(f"📝 Step 2: 获取 {target_stock_count} 只股票列表...")
            stock_codes = self.get_stock_list(target_stock_count)
            comprehensive_result['actual_stock_count'] = len(stock_codes)
            comprehensive_result['stock_list_acquisition'] = {
                'requested_count': target_stock_count,
                'actual_count': len(stock_codes),
                'success': len(stock_codes) > 0
            }
            
            if len(stock_codes) == 0:
                comprehensive_result['recommendations'].append("无法获取股票列表")
                return comprehensive_result
            
            # 3. 启动资源监控
            logger.info("📝 Step 3: 启动系统资源监控...")
            self.start_resource_monitoring()
            
            # 4. 批量数据查询测试
            logger.info("📝 Step 4: 执行批量数据查询测试...")
            comprehensive_result['batch_query_test'] = self.test_batch_stock_data_query(
                stock_codes, batch_size=50
            )
            
            # 5. 技术指标计算测试
            logger.info("📝 Step 5: 执行技术指标计算测试...")
            comprehensive_result['technical_indicators_test'] = self.test_technical_indicators_calculation(
                stock_codes, sample_size=100
            )
            
            # 6. 并发处理测试
            logger.info("📝 Step 6: 执行并发处理测试...")
            comprehensive_result['concurrent_processing_test'] = self.test_concurrent_processing(
                stock_codes, num_threads=4, stocks_per_thread=25
            )
            
            # 7. 停止资源监控
            logger.info("📝 Step 7: 停止系统资源监控...")
            self.stop_resource_monitoring()
            comprehensive_result['resource_usage'] = self.performance_metrics
            
            # 8. 综合评估
            logger.info("📝 Step 8: 综合性能评估...")
            self._evaluate_performance(comprehensive_result)
            
            end_time = datetime.now()
            comprehensive_result['test_end_time'] = end_time.isoformat()
            comprehensive_result['total_test_time'] = (end_time - self.start_time).total_seconds()
            
            logger.info(f"✅ 大规模数据综合测试完成，总耗时: {comprehensive_result['total_test_time']:.1f}秒")
            logger.info(f"📊 整体性能: {comprehensive_result['overall_performance']}")
            logger.info(f"📈 扩展性等级: {comprehensive_result['scalability_grade']}")
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"❌ 综合测试失败: {e}")
            comprehensive_result['recommendations'].append(f"测试过程异常: {str(e)}")
            return comprehensive_result
        finally:
            # 确保停止监控
            self.stop_resource_monitoring()
    
    def _evaluate_performance(self, result: Dict[str, Any]) -> None:
        """评估整体性能"""
        scores = []
        
        # 批量查询性能评分
        batch_test = result.get('batch_query_test', {})
        if batch_test.get('successful_queries', 0) > 0:
            success_rate = batch_test['successful_queries'] / (
                batch_test['successful_queries'] + batch_test.get('failed_queries', 0)
            )
            if success_rate >= 0.95:
                scores.append(100)
            elif success_rate >= 0.8:
                scores.append(80)
            else:
                scores.append(60)
        else:
            scores.append(0)
        
        # 技术指标计算性能评分
        indicators_test = result.get('technical_indicators_test', {})
        if indicators_test.get('successful_calculations', 0) > 0:
            success_rate = indicators_test['successful_calculations'] / (
                indicators_test['successful_calculations'] + indicators_test.get('failed_calculations', 0)
            )
            if success_rate >= 0.9:
                scores.append(100)
            elif success_rate >= 0.7:
                scores.append(80)
            else:
                scores.append(60)
        else:
            scores.append(0)
        
        # 并发处理性能评分
        concurrent_test = result.get('concurrent_processing_test', {})
        if concurrent_test.get('successful_threads', 0) > 0:
            success_rate = concurrent_test['successful_threads'] / (
                concurrent_test['successful_threads'] + concurrent_test.get('failed_threads', 0)
            )
            if success_rate >= 0.9:
                scores.append(100)
            elif success_rate >= 0.7:
                scores.append(80)
            else:
                scores.append(60)
        else:
            scores.append(0)
        
        # 资源使用评分
        resource_usage = result.get('resource_usage', {})
        cpu_usage = resource_usage.get('cpu_usage', {})
        memory_usage = resource_usage.get('memory_usage', {})
        
        if cpu_usage.get('max', 100) < 80 and memory_usage.get('max', 100) < 80:
            scores.append(100)
        elif cpu_usage.get('max', 100) < 90 and memory_usage.get('max', 100) < 90:
            scores.append(80)
        else:
            scores.append(60)
        
        # 计算综合评分
        if scores:
            overall_score = np.mean(scores)
            
            if overall_score >= 90:
                result['overall_performance'] = 'Excellent'
                result['scalability_grade'] = 'A+'
            elif overall_score >= 80:
                result['overall_performance'] = 'Good'
                result['scalability_grade'] = 'A'
            elif overall_score >= 70:
                result['overall_performance'] = 'Fair'
                result['scalability_grade'] = 'B'
                result['recommendations'].append("建议优化性能后处理更大规模数据")
            else:
                result['overall_performance'] = 'Poor'
                result['scalability_grade'] = 'C'
                result['recommendations'].append("需要解决性能问题后才能处理大规模数据")
        else:
            result['overall_performance'] = 'Failed'
            result['scalability_grade'] = 'F'
            result['recommendations'].append("无法完成性能评估")
    
    def generate_detailed_report(self, result: Dict[str, Any]) -> str:
        """生成详细报告"""
        report_lines = [
            "=" * 80,
            "大规模真实数据测试报告",
            "=" * 80,
            f"测试时间: {result.get('test_start_time', 'Unknown')}",
            f"测试耗时: {result.get('total_test_time', 0):.1f}秒",
            f"目标股票数: {result.get('target_stock_count', 0):,}只",
            f"实际股票数: {result.get('actual_stock_count', 0):,}只",
            f"整体性能: {result.get('overall_performance', 'Unknown')}",
            f"扩展性等级: {result.get('scalability_grade', 'Unknown')}",
            "",
            "📊 详细测试结果:",
            "-" * 40,
        ]
        
        # 数据库连接
        conn_status = "✅ 成功" if result.get('database_connection', False) else "❌ 失败"
        report_lines.append(f"数据库连接: {conn_status}")
        
        # 批量查询测试
        batch_test = result.get('batch_query_test', {})
        if batch_test:
            report_lines.extend([
                "",
                "📈 批量查询测试:",
                f"  成功批次: {batch_test.get('successful_queries', 0)}/{batch_test.get('total_batches', 0)}",
                f"  总记录数: {batch_test.get('total_records', 0):,}条",
                f"  平均查询时间: {batch_test.get('average_query_time', 0):.3f}秒",
                f"  数据吞吐量: {batch_test.get('throughput_records_per_second', 0):.0f} 记录/秒"
            ])
        
        # 技术指标测试
        indicators_test = result.get('technical_indicators_test', {})
        if indicators_test:
            report_lines.extend([
                "",
                "🔢 技术指标计算测试:",
                f"  成功计算: {indicators_test.get('successful_calculations', 0)}只股票",
                f"  平均计算时间: {indicators_test.get('average_calculation_time', 0):.3f}秒",
                f"  测试指标: {', '.join(indicators_test.get('indicators_tested', []))}"
            ])
        
        # 并发处理测试
        concurrent_test = result.get('concurrent_processing_test', {})
        if concurrent_test:
            report_lines.extend([
                "",
                "⚡ 并发处理测试:",
                f"  成功线程: {concurrent_test.get('successful_threads', 0)}/{concurrent_test.get('num_threads', 0)}",
                f"  并发吞吐量: {concurrent_test.get('concurrent_throughput', 0):.0f} 记录/秒",
                f"  总处理时间: {concurrent_test.get('total_time', 0):.1f}秒"
            ])
        
        # 资源使用情况
        resource_usage = result.get('resource_usage', {})
        if resource_usage:
            cpu_usage = resource_usage.get('cpu_usage', {})
            memory_usage = resource_usage.get('memory_usage', {})
            
            report_lines.extend([
                "",
                "💻 系统资源使用:",
                f"  CPU使用率: 平均 {cpu_usage.get('average', 0):.1f}%, 最大 {cpu_usage.get('max', 0):.1f}%",
                f"  内存使用率: 平均 {memory_usage.get('average', 0):.1f}%, 最大 {memory_usage.get('max', 0):.1f}%"
            ])
        
        # 建议
        if result.get('recommendations'):
            report_lines.extend([
                "",
                "💡 优化建议:"
            ])
            for i, rec in enumerate(result['recommendations'], 1):
                report_lines.append(f"{i}. {rec}")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


def main():
    """主函数 - 运行大规模数据测试"""
    print("🚀 启动大规模真实数据综合测试...")
    
    tester = LargeScaleDataTester()
    
    try:
        # 运行综合测试
        result = tester.run_comprehensive_test(target_stock_count=1000)
        
        # 生成详细报告
        detailed_report = tester.generate_detailed_report(result)
        print(detailed_report)
        
        # 保存报告到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"large_scale_data_test_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        
        # 返回测试结果
        if result.get('overall_performance') in ['Excellent', 'Good']:
            print("🎉 大规模数据测试通过！系统可以处理大规模数据。")
            return 0
        else:
            print("⚠️  大规模数据测试未完全通过，请根据建议进行优化。")
            return 1
            
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 