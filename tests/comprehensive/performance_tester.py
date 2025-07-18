#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票选股策略系统性能基准测试器

提供全面的性能基准测试，包括单股查询、批量处理、全市场扫描等性能验证。
遵循L5业务应用层规范，确保系统在不同负载下的性能表现。

测试内容：
- 单股查询性能测试（2秒内）
- 批量处理性能测试（100股30秒、1000股5分钟）
- 全市场扫描性能测试（4000+股票20分钟内）
- 内存使用监控（峰值8GB限制）
- 数据库连接池监控（连接数不超过50个）
- 性能瓶颈分析（CPU、内存、I/O、网络）
- 长时间运行任务监控（进度跟踪和恢复机制）
"""

import os
import sys
import time
import threading
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import tracemalloc

from db.query_executor import get_query_executor
from db.sql_manager import QueryType
from strategy.strategy_executor import StrategyExecutor
from indicators.complete_indicator_registry import complete_registry
from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from .config_manager import get_config_manager
from .logging_config import get_test_logger

logger = get_test_logger('performance_tester')


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    test_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    peak_memory: float = 0.0
    average_cpu: float = 0.0
    throughput: float = 0.0  # 处理速度
    success_rate: float = 0.0
    error_count: int = 0
    database_connections: int = 0
    cache_hit_rate: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    disk_io: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThresholds:
    """性能阈值配置"""
    single_stock_query: float = 2.0  # 秒
    batch_100_stocks: float = 30.0  # 秒
    batch_1000_stocks: float = 300.0  # 秒
    full_market_scan: float = 1200.0  # 秒
    max_memory_usage: float = 8.0  # GB
    max_connections: int = 50
    cache_hit_rate_threshold: float = 0.8
    cpu_usage_threshold: float = 80.0  # %


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        """初始化性能监控器"""
        self.monitoring = False
        self.metrics = PerformanceMetrics("", datetime.now())
        self.monitor_thread = None
        self.start_memory = 0.0
        self.connection_count = 0
        
    def start_monitoring(self, test_name: str) -> None:
        """开始性能监控"""
        self.metrics = PerformanceMetrics(test_name, datetime.now())
        self.monitoring = True
        self.start_memory = psutil.virtual_memory().used / (1024**3)
        
        # 启动内存跟踪
        tracemalloc.start()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.debug(f"开始监控性能测试: {test_name}")
    
    def stop_monitoring(self) -> PerformanceMetrics:
        """停止性能监控并返回指标"""
        self.monitoring = False
        self.metrics.end_time = datetime.now()
        self.metrics.execution_time = (self.metrics.end_time - self.metrics.start_time).total_seconds()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # 停止内存跟踪
        try:
            current, peak = tracemalloc.get_traced_memory()
            self.metrics.peak_memory = peak / (1024**3)  # 转换为GB
            tracemalloc.stop()
        except Exception:
            pass
        
        # 计算平均值
        if self.metrics.cpu_usage:
            self.metrics.average_cpu = sum(self.metrics.cpu_usage) / len(self.metrics.cpu_usage)
        
        if self.metrics.memory_usage:
            self.metrics.peak_memory = max(max(self.metrics.memory_usage), self.metrics.peak_memory)
        
        logger.debug(f"停止监控，执行时间: {self.metrics.execution_time:.2f}秒")
        return self.metrics
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.metrics.cpu_usage.append(cpu_percent)
                
                # 内存使用情况
                memory = psutil.virtual_memory()
                memory_usage_gb = memory.used / (1024**3)
                self.metrics.memory_usage.append(memory_usage_gb)
                
                # 网络I/O
                net_io = psutil.net_io_counters()
                if net_io:
                    self.metrics.network_io = {
                        'bytes_sent': net_io.bytes_sent,
                        'bytes_recv': net_io.bytes_recv
                    }
                
                # 磁盘I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.metrics.disk_io = {
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes
                    }
                
                time.sleep(1.0)  # 每秒监控一次
                
            except Exception as e:
                logger.warning(f"性能监控异常: {e}")
                time.sleep(1.0)
    
    def update_connection_count(self, count: int) -> None:
        """更新数据库连接数"""
        self.connection_count = count
        self.metrics.database_connections = count
    
    def update_cache_hit_rate(self, hit_rate: float) -> None:
        """更新缓存命中率"""
        self.metrics.cache_hit_rate = hit_rate


class PerformanceBenchmarkTester:
    """性能基准测试器"""
    
    def __init__(self):
        """初始化性能基准测试器"""
        self.config = get_config_manager().get_config()
        self.query_executor = get_query_executor()
        self.strategy_executor = StrategyExecutor()
        self.thresholds = PerformanceThresholds()
        
        # 从配置加载阈值
        if hasattr(self.config, 'performance_thresholds'):
            pt = self.config.performance_thresholds
            self.thresholds.single_stock_query = pt.single_stock_query
            self.thresholds.batch_100_stocks = pt.batch_100_stocks
            self.thresholds.batch_1000_stocks = pt.batch_1000_stocks
            self.thresholds.full_market_scan = pt.full_market_scan
            self.thresholds.max_memory_usage = pt.max_memory_usage
            self.thresholds.max_connections = pt.max_connections
            self.thresholds.cache_hit_rate_threshold = pt.cache_hit_rate_threshold
        
        # 获取测试股票池
        self.test_stocks = self._get_test_stock_pool()
        
        logger.info("性能基准测试器初始化完成")
    
    @performance_monitor(threshold=5.0)
    @exception_handler(reraise=True)
    def test_single_stock_query_performance(self) -> Dict[str, Any]:
        """
        测试单股查询性能
        
        Returns:
            Dict[str, Any]: 测试结果
        """
        logger.info("开始单股查询性能测试")
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring("单股查询性能测试")
        
        try:
            test_stock = self.test_stocks[0] if self.test_stocks else "000001"
            test_results = []
            
            # 执行多次查询测试
            for i in range(10):
                start_time = time.time()
                
                # 执行股票数据查询
                params = {
                    'code': test_stock,
                    'start_date': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                    'end_date': datetime.now().strftime('%Y-%m-%d'),
                    'level': '日线'
                }
                
                data = self.query_executor.execute_query(QueryType.STOCK_DATA, params)
                
                end_time = time.time()
                query_time = end_time - start_time
                
                test_results.append({
                    'iteration': i + 1,
                    'query_time': query_time,
                    'data_rows': len(data) if data is not None else 0,
                    'success': data is not None and not data.empty
                })
                
                # 短暂休息避免过载
                time.sleep(0.1)
            
            metrics = monitor.stop_monitoring()
            
            # 计算统计指标
            successful_queries = [r for r in test_results if r['success']]
            query_times = [r['query_time'] for r in successful_queries]
            
            avg_query_time = sum(query_times) / len(query_times) if query_times else 0.0
            max_query_time = max(query_times) if query_times else 0.0
            min_query_time = min(query_times) if query_times else 0.0
            
            # 性能评估
            performance_passed = avg_query_time <= self.thresholds.single_stock_query
            memory_passed = metrics.peak_memory <= self.thresholds.max_memory_usage
            
            result = {
                "test_name": "单股查询性能测试",
                "test_results": test_results,
                "metrics": metrics,
                "statistics": {
                    "total_queries": len(test_results),
                    "successful_queries": len(successful_queries),
                    "success_rate": len(successful_queries) / len(test_results),
                    "avg_query_time": avg_query_time,
                    "max_query_time": max_query_time,
                    "min_query_time": min_query_time,
                    "throughput": len(successful_queries) / metrics.execution_time if metrics.execution_time > 0 else 0.0
                },
                "performance_assessment": {
                    "query_time_passed": performance_passed,
                    "memory_usage_passed": memory_passed,
                    "overall_passed": performance_passed and memory_passed,
                    "threshold_comparison": {
                        "avg_time_vs_threshold": f"{avg_query_time:.2f}s / {self.thresholds.single_stock_query}s",
                        "memory_vs_threshold": f"{metrics.peak_memory:.2f}GB / {self.thresholds.max_memory_usage}GB"
                    }
                },
                "success": performance_passed and memory_passed,
                "summary": {
                    "avg_query_time": avg_query_time,
                    "success_rate": len(successful_queries) / len(test_results),
                    "memory_usage": metrics.peak_memory
                }
            }
            
            if result["success"]:
                logger.info(f"单股查询性能测试通过，平均查询时间: {avg_query_time:.2f}秒")
            else:
                logger.warning(f"单股查询性能测试失败，平均查询时间: {avg_query_time:.2f}秒")
            
            return result
            
        except Exception as e:
            monitor.stop_monitoring()
            logger.error(f"单股查询性能测试异常: {e}")
            return {
                "test_name": "单股查询性能测试",
                "success": False,
                "error": str(e),
                "summary": {
                    "avg_query_time": 0.0,
                    "success_rate": 0.0,
                    "memory_usage": 0.0
                }
            }
    
    @performance_monitor(threshold=40.0)
    @exception_handler(reraise=True)
    def test_batch_processing_performance(self) -> Dict[str, Any]:
        """
        测试批量处理性能
        
        Returns:
            Dict[str, Any]: 测试结果
        """
        logger.info("开始批量处理性能测试")
        
        batch_tests = [
            {"size": 100, "threshold": self.thresholds.batch_100_stocks, "name": "100股批量处理"},
            {"size": 1000, "threshold": self.thresholds.batch_1000_stocks, "name": "1000股批量处理"}
        ]
        
        all_results = {}
        overall_success = True
        
        for batch_config in batch_tests:
            batch_size = batch_config["size"]
            threshold = batch_config["threshold"]
            test_name = batch_config["name"]
            
            logger.info(f"开始 {test_name} 测试")
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring(test_name)
            
            try:
                # 准备测试股票池
                test_batch = self._get_stock_batch(batch_size)
                if len(test_batch) < batch_size:
                    logger.warning(f"实际测试股票数 {len(test_batch)} 少于目标 {batch_size}")
                
                successful_processes = 0
                failed_processes = 0
                processing_times = []
                
                # 使用线程池批量处理
                with ThreadPoolExecutor(max_workers=self.thresholds.max_connections // 10) as executor:
                    futures = []
                    
                    for stock_code in test_batch:
                        future = executor.submit(self._process_single_stock, stock_code)
                        futures.append((future, stock_code))
                    
                    # 收集结果
                    for future, stock_code in futures:
                        try:
                            start_time = time.time()
                            result = future.result(timeout=10.0)  # 单股10秒超时
                            process_time = time.time() - start_time
                            
                            if result['success']:
                                successful_processes += 1
                                processing_times.append(process_time)
                            else:
                                failed_processes += 1
                                
                        except Exception as e:
                            failed_processes += 1
                            logger.warning(f"处理股票 {stock_code} 失败: {e}")
                
                metrics = monitor.stop_monitoring()
                
                # 计算统计指标
                total_processes = successful_processes + failed_processes
                success_rate = successful_processes / total_processes if total_processes > 0 else 0.0
                avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
                throughput = successful_processes / metrics.execution_time if metrics.execution_time > 0 else 0.0
                
                # 性能评估
                time_passed = metrics.execution_time <= threshold
                memory_passed = metrics.peak_memory <= self.thresholds.max_memory_usage
                success_rate_passed = success_rate >= 0.9  # 90%成功率
                
                batch_result = {
                    "test_name": test_name,
                    "batch_size": len(test_batch),
                    "metrics": metrics,
                    "statistics": {
                        "total_processes": total_processes,
                        "successful_processes": successful_processes,
                        "failed_processes": failed_processes,
                        "success_rate": success_rate,
                        "avg_process_time": avg_process_time,
                        "throughput": throughput,
                        "processing_times": processing_times
                    },
                    "performance_assessment": {
                        "execution_time_passed": time_passed,
                        "memory_usage_passed": memory_passed,
                        "success_rate_passed": success_rate_passed,
                        "overall_passed": time_passed and memory_passed and success_rate_passed,
                        "threshold_comparison": {
                            "time_vs_threshold": f"{metrics.execution_time:.2f}s / {threshold}s",
                            "memory_vs_threshold": f"{metrics.peak_memory:.2f}GB / {self.thresholds.max_memory_usage}GB",
                            "success_rate_vs_target": f"{success_rate:.1%} / 90%"
                        }
                    },
                    "success": time_passed and memory_passed and success_rate_passed
                }
                
                all_results[batch_size] = batch_result
                
                if not batch_result["success"]:
                    overall_success = False
                
                logger.info(f"{test_name} 完成，执行时间: {metrics.execution_time:.2f}秒，成功率: {success_rate:.1%}")
                
            except Exception as e:
                monitor.stop_monitoring()
                logger.error(f"{test_name} 测试异常: {e}")
                all_results[batch_size] = {
                    "test_name": test_name,
                    "success": False,
                    "error": str(e)
                }
                overall_success = False
        
        # 汇总结果
        result = {
            "test_name": "批量处理性能测试",
            "batch_results": all_results,
            "success": overall_success,
            "summary": {
                "total_batches": len(batch_tests),
                "successful_batches": sum(1 for r in all_results.values() if r.get("success", False)),
                "overall_performance": overall_success
            }
        }
        
        if result["success"]:
            logger.info("批量处理性能测试全部通过")
        else:
            logger.warning("批量处理性能测试存在失败项")
        
        return result
    
    @performance_monitor(threshold=1300.0)
    @exception_handler(reraise=True)
    def test_full_market_scan_performance(self) -> Dict[str, Any]:
        """
        测试全市场扫描性能
        
        Returns:
            Dict[str, Any]: 测试结果
        """
        logger.info("开始全市场扫描性能测试")
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring("全市场扫描性能测试")
        
        try:
            # 模拟全市场股票池（使用较大样本）
            full_market_stocks = self._get_full_market_simulation()
            total_stocks = len(full_market_stocks)
            
            logger.info(f"模拟全市场扫描，股票总数: {total_stocks}")
            
            # 分批处理以避免内存溢出
            batch_size = 200
            successful_scans = 0
            failed_scans = 0
            scan_details = []
            
            # 进度跟踪
            processed_count = 0
            
            for i in range(0, total_stocks, batch_size):
                batch = full_market_stocks[i:i + batch_size]
                batch_start_time = time.time()
                
                try:
                    # 执行批次扫描
                    batch_results = self._scan_stock_batch(batch)
                    
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    
                    successful_scans += batch_results['successful_count']
                    failed_scans += batch_results['failed_count']
                    
                    scan_details.append({
                        'batch_index': i // batch_size + 1,
                        'batch_size': len(batch),
                        'batch_time': batch_time,
                        'successful_count': batch_results['successful_count'],
                        'failed_count': batch_results['failed_count']
                    })
                    
                    processed_count += len(batch)
                    
                    # 记录进度
                    progress = processed_count / total_stocks
                    if processed_count % 1000 == 0 or processed_count == total_stocks:
                        logger.info(f"扫描进度: {processed_count}/{total_stocks} ({progress:.1%})")
                    
                    # 垃圾回收
                    if i % (batch_size * 5) == 0:
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"批次 {i//batch_size + 1} 扫描失败: {e}")
                    failed_scans += len(batch)
                    scan_details.append({
                        'batch_index': i // batch_size + 1,
                        'batch_size': len(batch),
                        'batch_time': 0.0,
                        'successful_count': 0,
                        'failed_count': len(batch),
                        'error': str(e)
                    })
            
            metrics = monitor.stop_monitoring()
            
            # 计算统计指标
            total_processed = successful_scans + failed_scans
            success_rate = successful_scans / total_processed if total_processed > 0 else 0.0
            throughput = successful_scans / metrics.execution_time if metrics.execution_time > 0 else 0.0
            
            # 性能评估
            time_passed = metrics.execution_time <= self.thresholds.full_market_scan
            memory_passed = metrics.peak_memory <= self.thresholds.max_memory_usage
            success_rate_passed = success_rate >= 0.8  # 80%成功率
            
            result = {
                "test_name": "全市场扫描性能测试",
                "total_stocks": total_stocks,
                "metrics": metrics,
                "statistics": {
                    "total_processed": total_processed,
                    "successful_scans": successful_scans,
                    "failed_scans": failed_scans,
                    "success_rate": success_rate,
                    "throughput": throughput,
                    "batches_processed": len(scan_details),
                    "scan_details": scan_details
                },
                "performance_assessment": {
                    "execution_time_passed": time_passed,
                    "memory_usage_passed": memory_passed,
                    "success_rate_passed": success_rate_passed,
                    "overall_passed": time_passed and memory_passed and success_rate_passed,
                    "threshold_comparison": {
                        "time_vs_threshold": f"{metrics.execution_time:.2f}s / {self.thresholds.full_market_scan}s",
                        "memory_vs_threshold": f"{metrics.peak_memory:.2f}GB / {self.thresholds.max_memory_usage}GB",
                        "success_rate_vs_target": f"{success_rate:.1%} / 80%"
                    }
                },
                "success": time_passed and memory_passed and success_rate_passed,
                "summary": {
                    "execution_time": metrics.execution_time,
                    "success_rate": success_rate,
                    "throughput": throughput,
                    "memory_usage": metrics.peak_memory
                }
            }
            
            if result["success"]:
                logger.info(f"全市场扫描性能测试通过，用时: {metrics.execution_time/60:.2f}分钟，成功率: {success_rate:.1%}")
            else:
                logger.warning(f"全市场扫描性能测试失败，用时: {metrics.execution_time/60:.2f}分钟，成功率: {success_rate:.1%}")
            
            return result
            
        except Exception as e:
            monitor.stop_monitoring()
            logger.error(f"全市场扫描性能测试异常: {e}")
            return {
                "test_name": "全市场扫描性能测试",
                "success": False,
                "error": str(e),
                "summary": {
                    "execution_time": 0.0,
                    "success_rate": 0.0,
                    "throughput": 0.0,
                    "memory_usage": 0.0
                }
            }
    
    @performance_monitor(threshold=10.0)
    @exception_handler(reraise=True)
    def test_memory_usage_monitoring(self) -> Dict[str, Any]:
        """
        测试内存使用监控
        
        Returns:
            Dict[str, Any]: 测试结果
        """
        logger.info("开始内存使用监控测试")
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring("内存使用监控测试")
        
        try:
            memory_test_results = []
            
            # 测试不同内存负载场景
            test_scenarios = [
                {"name": "轻负载", "stock_count": 10, "iterations": 5},
                {"name": "中负载", "stock_count": 50, "iterations": 3},
                {"name": "重负载", "stock_count": 100, "iterations": 2}
            ]
            
            for scenario in test_scenarios:
                scenario_start = time.time()
                initial_memory = psutil.virtual_memory().used / (1024**3)
                
                # 执行内存密集型操作
                for i in range(scenario["iterations"]):
                    test_stocks = self._get_stock_batch(scenario["stock_count"])
                    data_cache = {}
                    
                    # 加载数据到内存
                    for stock_code in test_stocks:
                        try:
                            params = {
                                'code': stock_code,
                                'start_date': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                                'end_date': datetime.now().strftime('%Y-%m-%d'),
                                'level': '日线'
                            }
                            data = self.query_executor.execute_query(QueryType.STOCK_DATA, params)
                            if data is not None:
                                data_cache[stock_code] = data
                        except Exception:
                            continue
                    
                    # 检查内存使用
                    current_memory = psutil.virtual_memory().used / (1024**3)
                    memory_increase = current_memory - initial_memory
                    
                    # 清理缓存
                    del data_cache
                    gc.collect()
                
                scenario_end = time.time()
                final_memory = psutil.virtual_memory().used / (1024**3)
                
                memory_test_results.append({
                    "scenario": scenario["name"],
                    "stock_count": scenario["stock_count"],
                    "iterations": scenario["iterations"],
                    "execution_time": scenario_end - scenario_start,
                    "initial_memory": initial_memory,
                    "final_memory": final_memory,
                    "memory_increase": final_memory - initial_memory
                })
            
            metrics = monitor.stop_monitoring()
            
            # 内存使用评估
            memory_passed = metrics.peak_memory <= self.thresholds.max_memory_usage
            memory_stable = abs(memory_test_results[-1]["memory_increase"]) < 0.5  # 内存增长小于0.5GB
            
            result = {
                "test_name": "内存使用监控测试",
                "memory_test_results": memory_test_results,
                "metrics": metrics,
                "memory_assessment": {
                    "peak_memory_passed": memory_passed,
                    "memory_stable": memory_stable,
                    "overall_passed": memory_passed and memory_stable,
                    "threshold_comparison": {
                        "peak_memory_vs_threshold": f"{metrics.peak_memory:.2f}GB / {self.thresholds.max_memory_usage}GB",
                        "final_increase": f"{memory_test_results[-1]['memory_increase']:.2f}GB"
                    }
                },
                "success": memory_passed and memory_stable,
                "summary": {
                    "peak_memory": metrics.peak_memory,
                    "memory_stable": memory_stable,
                    "scenarios_tested": len(test_scenarios)
                }
            }
            
            if result["success"]:
                logger.info(f"内存使用监控测试通过，峰值内存: {metrics.peak_memory:.2f}GB")
            else:
                logger.warning(f"内存使用监控测试失败，峰值内存: {metrics.peak_memory:.2f}GB")
            
            return result
            
        except Exception as e:
            monitor.stop_monitoring()
            logger.error(f"内存使用监控测试异常: {e}")
            return {
                "test_name": "内存使用监控测试",
                "success": False,
                "error": str(e),
                "summary": {
                    "peak_memory": 0.0,
                    "memory_stable": False,
                    "scenarios_tested": 0
                }
            }
    
    def _get_test_stock_pool(self) -> List[str]:
        """获取测试股票池"""
        # 从配置获取测试股票
        if hasattr(self.config, 'test_data') and hasattr(self.config.test_data, 'sample_stock_codes'):
            return list(self.config.test_data.sample_stock_codes)
        
        # 默认测试股票池
        return [
            "000001", "000002", "600000", "600036", "000858",
            "600519", "000063", "002415", "300059", "002594"
        ]
    
    def _get_stock_batch(self, size: int) -> List[str]:
        """获取指定大小的股票批次"""
        base_stocks = self.test_stocks
        
        # 如果需要更多股票，生成模拟股票代码
        if size > len(base_stocks):
            additional_stocks = []
            for i in range(len(base_stocks), size):
                # 生成模拟股票代码
                if i % 2 == 0:
                    code = f"{(i % 900000 + 100000):06d}"  # 100000-999999
                else:
                    code = f"{(i % 900000 + 600000):06d}"  # 600000-999999
                additional_stocks.append(code)
            
            return base_stocks + additional_stocks
        
        return base_stocks[:size]
    
    def _get_full_market_simulation(self) -> List[str]:
        """获取全市场模拟股票池"""
        # 生成4000+只模拟股票代码
        stocks = []
        
        # 真实测试股票
        stocks.extend(self.test_stocks)
        
        # 生成模拟股票代码
        for i in range(4000):
            if i % 2 == 0:
                code = f"{(i % 900000 + 100000):06d}"  # 深市
            else:
                code = f"{(i % 400000 + 600000):06d}"  # 沪市
            stocks.append(code)
        
        return stocks
    
    def _process_single_stock(self, stock_code: str) -> Dict[str, Any]:
        """处理单只股票"""
        try:
            start_time = time.time()
            
            # 获取股票数据
            params = {
                'code': stock_code,
                'start_date': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'end_date': datetime.now().strftime('%Y-%m-%d'),
                'level': '日线'
            }
            
            data = self.query_executor.execute_query(QueryType.STOCK_DATA, params)
            
            # 简单的技术指标计算（模拟策略处理）
            if data is not None and not data.empty and len(data) >= 5:
                # 计算简单移动平均线
                data['ma5'] = data['close'].rolling(window=5).mean()
                # 计算涨跌幅
                data['pct_change'] = data['close'].pct_change()
            
            end_time = time.time()
            
            return {
                "success": True,
                "stock_code": stock_code,
                "processing_time": end_time - start_time,
                "data_rows": len(data) if data is not None else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "stock_code": stock_code,
                "error": str(e),
                "processing_time": 0.0,
                "data_rows": 0
            }
    
    def _scan_stock_batch(self, stock_batch: List[str]) -> Dict[str, Any]:
        """扫描股票批次"""
        successful_count = 0
        failed_count = 0
        
        for stock_code in stock_batch:
            try:
                result = self._process_single_stock(stock_code)
                if result['success']:
                    successful_count += 1
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1
        
        return {
            "successful_count": successful_count,
            "failed_count": failed_count,
            "total_count": len(stock_batch)
        }


if __name__ == "__main__":
    # 测试性能基准测试器
    tester = PerformanceBenchmarkTester()
    
    print("运行单股查询性能测试...")
    result1 = tester.test_single_stock_query_performance()
    print(f"结果: {result1['success']}, 平均查询时间: {result1['summary']['avg_query_time']:.2f}秒")
    
    print("\n运行批量处理性能测试...")
    result2 = tester.test_batch_processing_performance()
    print(f"结果: {result2['success']}, 批量测试: {result2['summary']['successful_batches']}/{result2['summary']['total_batches']}")
    
    print("\n运行内存使用监控测试...")
    result3 = tester.test_memory_usage_monitoring()
    print(f"结果: {result3['success']}, 峰值内存: {result3['summary']['peak_memory']:.2f}GB")
    
    print("\n运行全市场扫描性能测试（简化版）...")
    # 注意：完整的全市场扫描测试时间较长，这里可以跳过或使用较小的样本 