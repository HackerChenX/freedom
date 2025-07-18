#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大规模股票处理优化器

专门针对4000+股票的批处理策略、内存高效数据流和连接池优化
复用现有的性能基准测试和数据访问组件
"""

import time
import gc
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Iterator, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque
import psutil
import weakref

from utils.logger import getLogger
from .data_access_adapter import get_data_access_adapter, DataAccessAdapter
from .cache_manager import get_cache_manager
from .performance_monitor import get_performance_monitor
from .parallel_test_executor import ParallelTestExecutor, TaskDistribution, ResourceLimits, ExecutionMode
from .full_market_performance_benchmark import PerformanceMetrics

logger = getLogger(__name__)


@dataclass
class BatchProcessingConfig:
    """批处理配置"""
    batch_size: int = 1000              # 批处理大小
    max_concurrent_batches: int = 5     # 最大并发批次
    memory_limit_mb: int = 6144         # 内存限制(6GB)
    gc_frequency: int = 10              # 垃圾回收频率(每N个批次)
    streaming_enabled: bool = True      # 启用数据流处理
    connection_pool_size: int = 50      # 连接池大小


@dataclass
class MemoryOptimization:
    """内存优化配置"""
    chunk_size: int = 100               # 数据块大小
    lazy_loading: bool = True           # 延迟加载
    weak_references: bool = True        # 使用弱引用
    data_compression: bool = True       # 数据压缩
    auto_cleanup: bool = True           # 自动清理


@dataclass
class ProcessingStats:
    """处理统计信息"""
    total_stocks: int = 0
    processed_stocks: int = 0
    successful_stocks: int = 0
    failed_stocks: int = 0
    total_batches: int = 0
    completed_batches: int = 0
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    throughput_stocks_per_sec: float = 0.0


class MemoryEfficientDataStream:
    """内存高效数据流处理器"""
    
    def __init__(self, 
                 stock_codes: List[str],
                 chunk_size: int = 100,
                 data_access: Optional[DataAccessAdapter] = None):
        """
        初始化数据流处理器
        
        Args:
            stock_codes: 股票代码列表
            chunk_size: 数据块大小
            data_access: 数据访问适配器
        """
        self.stock_codes = stock_codes
        self.chunk_size = chunk_size
        self.data_access = data_access or get_data_access_adapter()
        self.current_index = 0
        self.cache = {}
        self.weak_cache = weakref.WeakValueDictionary()
        
    def __iter__(self) -> Iterator[Tuple[str, pd.DataFrame]]:
        """迭代器接口"""
        return self
    
    def __next__(self) -> Tuple[str, pd.DataFrame]:
        """获取下一个股票数据"""
        if self.current_index >= len(self.stock_codes):
            raise StopIteration
        
        stock_code = self.stock_codes[self.current_index]
        self.current_index += 1
        
        # 尝试从缓存获取
        if stock_code in self.weak_cache:
            return stock_code, self.weak_cache[stock_code]
        
        # 从数据源获取
        try:
            stock_data = self.data_access.get_stock_info(stock_code)
            if stock_data is not None:
                # 使用弱引用缓存
                self.weak_cache[stock_code] = stock_data
                return stock_code, stock_data
            else:
                # 返回空DataFrame
                return stock_code, pd.DataFrame()
        except Exception as e:
            logger.debug(f"获取股票 {stock_code} 数据失败: {e}")
            return stock_code, pd.DataFrame()
    
    def get_batch(self, batch_size: int) -> List[Tuple[str, pd.DataFrame]]:
        """获取批次数据"""
        batch = []
        try:
            for _ in range(batch_size):
                batch.append(next(self))
        except StopIteration:
            pass
        return batch
    
    def reset(self):
        """重置流"""
        self.current_index = 0
        self.cache.clear()
        self.weak_cache.clear()


class ConnectionPoolManager:
    """连接池管理器"""
    
    def __init__(self, pool_size: int = 50):
        """
        初始化连接池管理器
        
        Args:
            pool_size: 连接池大小
        """
        self.pool_size = pool_size
        self.active_connections = 0
        self.connection_lock = threading.Lock()
        self.connection_semaphore = threading.Semaphore(pool_size)
        
    def acquire_connection(self) -> bool:
        """获取连接"""
        acquired = self.connection_semaphore.acquire(blocking=False)
        if acquired:
            with self.connection_lock:
                self.active_connections += 1
        return acquired
    
    def release_connection(self):
        """释放连接"""
        with self.connection_lock:
            if self.active_connections > 0:
                self.active_connections -= 1
        self.connection_semaphore.release()
    
    def get_stats(self) -> Dict[str, int]:
        """获取连接池统计"""
        with self.connection_lock:
            return {
                'pool_size': self.pool_size,
                'active_connections': self.active_connections,
                'available_connections': self.pool_size - self.active_connections
            }


class LargeScaleStockProcessor:
    """大规模股票处理器"""
    
    def __init__(self, 
                 batch_config: Optional[BatchProcessingConfig] = None,
                 memory_config: Optional[MemoryOptimization] = None):
        """
        初始化大规模股票处理器
        
        Args:
            batch_config: 批处理配置
            memory_config: 内存优化配置
        """
        self.batch_config = batch_config or BatchProcessingConfig()
        self.memory_config = memory_config or MemoryOptimization()
        
        # 复用现有组件
        self.data_access = get_data_access_adapter()
        self.cache_manager = get_cache_manager()
        self.performance_monitor = get_performance_monitor()
        self.performance_metrics = PerformanceMetrics()
        
        # 连接池管理
        self.connection_pool = ConnectionPoolManager(self.batch_config.connection_pool_size)
        
        # 并行执行器
        task_distribution = TaskDistribution(
            mode=ExecutionMode.HYBRID,
            max_workers=self.batch_config.max_concurrent_batches,
            batch_size=self.batch_config.batch_size
        )
        
        resource_limits = ResourceLimits(
            max_memory_mb=self.batch_config.memory_limit_mb,
            max_cpu_percent=80.0,
            max_execution_time=300
        )
        
        self.parallel_executor = ParallelTestExecutor(task_distribution, resource_limits)
        
        # 状态管理
        self.stats = ProcessingStats()
        self.is_processing = False
        self.memory_monitor_thread = None
        self.stop_monitoring = threading.Event()
        
        logger.info(f"大规模股票处理器初始化完成 - 批大小: {self.batch_config.batch_size}, 内存限制: {self.batch_config.memory_limit_mb}MB")
    
    async def process_stocks(self, 
                           stock_codes: List[str],
                           processing_func: callable,
                           **kwargs) -> Dict[str, Any]:
        """
        处理大规模股票数据
        
        Args:
            stock_codes: 股票代码列表
            processing_func: 处理函数
            **kwargs: 传递给处理函数的参数
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        self.is_processing = True
        start_time = time.time()
        
        # 初始化统计
        self.stats = ProcessingStats(total_stocks=len(stock_codes))
        
        logger.info(f"开始处理 {len(stock_codes)} 只股票...")
        
        try:
            # 启动内存监控
            self._start_memory_monitoring()
            
            # 创建数据流
            data_stream = MemoryEfficientDataStream(
                stock_codes, 
                self.memory_config.chunk_size,
                self.data_access
            )
            
            # 批处理执行
            results = await self._process_in_batches(data_stream, processing_func, **kwargs)
            
            # 更新统计信息
            self.stats.processing_time = time.time() - start_time
            self.stats.throughput_stocks_per_sec = (
                self.stats.successful_stocks / self.stats.processing_time 
                if self.stats.processing_time > 0 else 0
            )
            
            logger.info(f"股票处理完成: {self.stats.successful_stocks}/{self.stats.total_stocks} 成功")
            logger.info(f"处理时间: {self.stats.processing_time:.2f}秒")
            logger.info(f"吞吐量: {self.stats.throughput_stocks_per_sec:.2f} 股票/秒")
            
            return {
                'results': results,
                'stats': self.stats,
                'performance_summary': self._generate_performance_summary()
            }
            
        except Exception as e:
            logger.error(f"大规模股票处理失败: {e}")
            raise
        finally:
            self.is_processing = False
            self._stop_memory_monitoring()
    
    async def _process_in_batches(self, 
                                data_stream: MemoryEfficientDataStream,
                                processing_func: callable,
                                **kwargs) -> Dict[str, Any]:
        """批处理执行"""
        results = {}
        batch_count = 0
        
        # 计算总批次数
        total_batches = (self.stats.total_stocks + self.batch_config.batch_size - 1) // self.batch_config.batch_size
        self.stats.total_batches = total_batches
        
        while True:
            # 获取批次数据
            batch_data = data_stream.get_batch(self.batch_config.batch_size)
            if not batch_data:
                break
            
            batch_count += 1
            logger.debug(f"处理批次 {batch_count}/{total_batches}: {len(batch_data)} 只股票")
            
            # 检查内存使用
            if not self._check_memory_limits():
                logger.warning("内存使用超限，触发优化")
                self._optimize_memory_usage()
            
            # 处理批次
            batch_results = await self._process_single_batch(
                batch_data, processing_func, batch_count, **kwargs
            )
            
            # 合并结果
            results.update(batch_results)
            
            # 更新统计
            self.stats.completed_batches += 1
            self.stats.processed_stocks += len(batch_data)
            
            # 垃圾回收
            if batch_count % self.batch_config.gc_frequency == 0:
                self._perform_garbage_collection()
        
        return results
    
    async def _process_single_batch(self, 
                                  batch_data: List[Tuple[str, pd.DataFrame]],
                                  processing_func: callable,
                                  batch_id: int,
                                  **kwargs) -> Dict[str, Any]:
        """处理单个批次"""
        batch_results = {}
        
        # 创建批次任务
        tasks = []
        for stock_code, stock_data in batch_data:
            if not stock_data.empty:
                task = {
                    'name': f'process_stock_{stock_code}',
                    'func': processing_func,
                    'kwargs': {
                        'stock_code': stock_code,
                        'stock_data': stock_data,
                        **kwargs
                    }
                }
                tasks.append(task)
        
        # 使用并行执行器处理任务
        try:
            with self.parallel_executor:
                execution_results = await self.parallel_executor.execute_tasks(
                    tasks, f"batch_{batch_id}"
                )
            
            # 处理执行结果
            for task_name, task_result in execution_results['results'].items():
                stock_code = task_name.replace('process_stock_', '')
                
                if task_result.get('success', False):
                    batch_results[stock_code] = task_result['result']
                    self.stats.successful_stocks += 1
                else:
                    logger.debug(f"股票 {stock_code} 处理失败: {task_result.get('error', 'Unknown error')}")
                    self.stats.failed_stocks += 1
            
        except Exception as e:
            logger.error(f"批次 {batch_id} 处理失败: {e}")
            self.stats.failed_stocks += len(batch_data)
        
        return batch_results
    
    def _start_memory_monitoring(self):
        """启动内存监控"""
        self.stop_monitoring.clear()
        self.memory_monitor_thread = threading.Thread(
            target=self._memory_monitor_loop,
            daemon=True
        )
        self.memory_monitor_thread.start()
    
    def _stop_memory_monitoring(self):
        """停止内存监控"""
        if self.memory_monitor_thread:
            self.stop_monitoring.set()
            self.memory_monitor_thread.join(timeout=5)
    
    def _memory_monitor_loop(self):
        """内存监控循环"""
        while not self.stop_monitoring.is_set():
            try:
                # 获取内存使用情况
                memory_info = psutil.virtual_memory()
                current_memory_mb = memory_info.used / (1024 ** 2)
                
                # 更新统计
                self.stats.memory_usage_mb = current_memory_mb
                self.stats.peak_memory_mb = max(self.stats.peak_memory_mb, current_memory_mb)
                
                # 检查内存限制
                if current_memory_mb > self.batch_config.memory_limit_mb:
                    logger.warning(f"内存使用超限: {current_memory_mb:.1f}MB > {self.batch_config.memory_limit_mb}MB")
                    self._optimize_memory_usage()
                
                time.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                logger.error(f"内存监控错误: {e}")
                time.sleep(5)
    
    def _check_memory_limits(self) -> bool:
        """检查内存限制"""
        try:
            memory_info = psutil.virtual_memory()
            current_memory_mb = memory_info.used / (1024 ** 2)
            return current_memory_mb <= self.batch_config.memory_limit_mb
        except Exception:
            return True
    
    def _optimize_memory_usage(self):
        """优化内存使用"""
        try:
            logger.info("开始内存优化...")
            
            # 强制垃圾回收
            self._perform_garbage_collection()
            
            # 清理缓存
            self.cache_manager.clear_expired()
            
            # 减少批次大小
            if self.batch_config.batch_size > 100:
                old_size = self.batch_config.batch_size
                self.batch_config.batch_size = max(100, self.batch_config.batch_size // 2)
                logger.info(f"批次大小优化: {old_size} -> {self.batch_config.batch_size}")
            
            # 减少并发数
            if self.batch_config.max_concurrent_batches > 2:
                old_concurrent = self.batch_config.max_concurrent_batches
                self.batch_config.max_concurrent_batches = max(2, self.batch_config.max_concurrent_batches // 2)
                logger.info(f"并发数优化: {old_concurrent} -> {self.batch_config.max_concurrent_batches}")
            
        except Exception as e:
            logger.error(f"内存优化失败: {e}")
    
    def _perform_garbage_collection(self):
        """执行垃圾回收"""
        try:
            before_memory = psutil.virtual_memory().used / (1024 ** 2)
            
            # 执行垃圾回收
            collected = gc.collect()
            
            after_memory = psutil.virtual_memory().used / (1024 ** 2)
            freed_memory = before_memory - after_memory
            
            if freed_memory > 0:
                logger.debug(f"垃圾回收: 释放 {freed_memory:.1f}MB 内存, 回收 {collected} 个对象")
            
        except Exception as e:
            logger.error(f"垃圾回收失败: {e}")
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """生成性能摘要"""
        return {
            'processing_stats': {
                'total_stocks': self.stats.total_stocks,
                'successful_stocks': self.stats.successful_stocks,
                'failed_stocks': self.stats.failed_stocks,
                'success_rate': self.stats.successful_stocks / self.stats.total_stocks if self.stats.total_stocks > 0 else 0,
                'processing_time': self.stats.processing_time,
                'throughput_stocks_per_sec': self.stats.throughput_stocks_per_sec
            },
            'batch_stats': {
                'total_batches': self.stats.total_batches,
                'completed_batches': self.stats.completed_batches,
                'avg_batch_size': self.batch_config.batch_size,
                'max_concurrent_batches': self.batch_config.max_concurrent_batches
            },
            'memory_stats': {
                'peak_memory_mb': self.stats.peak_memory_mb,
                'memory_limit_mb': self.batch_config.memory_limit_mb,
                'memory_efficiency': (self.batch_config.memory_limit_mb - self.stats.peak_memory_mb) / self.batch_config.memory_limit_mb if self.batch_config.memory_limit_mb > 0 else 0
            },
            'connection_stats': self.connection_pool.get_stats(),
            'optimization_config': {
                'batch_size': self.batch_config.batch_size,
                'streaming_enabled': self.batch_config.streaming_enabled,
                'lazy_loading': self.memory_config.lazy_loading,
                'data_compression': self.memory_config.data_compression
            }
        }
    
    def get_processing_stats(self) -> ProcessingStats:
        """获取处理统计信息"""
        return self.stats


# 全局大规模股票处理器实例
_large_scale_processor = None


def get_large_scale_processor(batch_config: Optional[BatchProcessingConfig] = None,
                            memory_config: Optional[MemoryOptimization] = None) -> LargeScaleStockProcessor:
    """
    获取全局大规模股票处理器实例
    
    Args:
        batch_config: 批处理配置
        memory_config: 内存优化配置
        
    Returns:
        LargeScaleStockProcessor: 大规模股票处理器实例
    """
    global _large_scale_processor
    if _large_scale_processor is None:
        _large_scale_processor = LargeScaleStockProcessor(batch_config, memory_config)
    return _large_scale_processor


async def main():
    """测试大规模股票处理器"""
    print("测试大规模股票处理器...")
    
    # 生成测试股票代码
    test_stocks = [f"{i:06d}" for i in range(1000, 1100)]  # 100只股票测试
    
    # 定义处理函数
    def process_stock(stock_code: str, stock_data: pd.DataFrame, **kwargs):
        """测试股票处理函数"""
        time.sleep(0.01)  # 模拟处理时间
        return {
            'stock_code': stock_code,
            'data_rows': len(stock_data),
            'processed_at': datetime.now().isoformat()
        }
    
    # 配置处理器
    batch_config = BatchProcessingConfig(
        batch_size=20,
        max_concurrent_batches=5,
        memory_limit_mb=2048
    )
    
    memory_config = MemoryOptimization(
        chunk_size=10,
        lazy_loading=True,
        data_compression=True
    )
    
    # 执行处理
    processor = LargeScaleStockProcessor(batch_config, memory_config)
    results = await processor.process_stocks(test_stocks, process_stock)
    
    print(f"处理完成:")
    print(f"总股票: {results['stats'].total_stocks}")
    print(f"成功: {results['stats'].successful_stocks}")
    print(f"失败: {results['stats'].failed_stocks}")
    print(f"处理时间: {results['stats'].processing_time:.2f}秒")
    print(f"吞吐量: {results['stats'].throughput_stocks_per_sec:.2f} 股票/秒")
    print(f"峰值内存: {results['stats'].peak_memory_mb:.1f}MB")


if __name__ == "__main__":
    asyncio.run(main())