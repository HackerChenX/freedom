from utils.container import container
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
内存管理优化器

实现高效的内存管理和数据分块处理，确保内存使用<4GB：
1. 智能数据分块策略
2. 内存池管理和回收机制
3. 数据压缩和序列化优化
4. 内存监控和预警系统
"""

import os
import gc
import sys
import psutil
import pickle
import zlib
import threading
import weakref
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import OrderedDict
from functools import wraps
import numpy as np
import pandas as pd

from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

@dataclass
class MemoryTarget:
    """内存使用目标"""
    max_memory_gb: float = 4.0  # 最大内存使用(GB)
    warning_threshold: float = 3.2  # 警告阈值(GB)
    critical_threshold: float = 3.8  # 紧急阈值(GB)
    cleanup_threshold: float = 3.5  # 清理阈值(GB)

@dataclass
class ChunkConfig:
    """分块配置"""
    base_chunk_size: int = 500  # 基础分块大小
    max_chunk_size: int = 2000  # 最大分块大小
    min_chunk_size: int = 100  # 最小分块大小
    adaptive_sizing: bool = True  # 自适应分块大小

@dataclass
class MemoryStats:
    """内存统计"""
    current_usage_gb: float = 0.0
    peak_usage_gb: float = 0.0
    available_gb: float = 0.0
    chunks_processed: int = 0
    gc_collections: int = 0
    compression_ratio: float = 0.0
    pool_efficiency: float = 0.0

class MemoryMonitor:
    """内存监控器"""

    def __init__(self, target: MemoryTarget):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.target = target
        self.stats = MemoryStats()
        self.monitoring_active = False
        self.monitor_thread = None
        self.alerts = []
        self._lock = threading.Lock()

    def start_monitoring(self):
        """启动内存监控"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("内存监控已启动")

    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("内存监控已停止")

    def get_current_memory_gb(self) -> float:
        """获取当前内存使用量(GB)"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)

    def get_system_memory_info(self) -> Dict[str, float]:
        """获取系统内存信息"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024 ** 3),
            'available_gb': memory.available / (1024 ** 3),
            'used_gb': memory.used / (1024 ** 3),
            'percent': memory.percent
        }

    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 更新内存统计
                current_memory = self.get_current_memory_gb()
                with self._lock:
                    self.stats.current_usage_gb = current_memory
                    self.stats.peak_usage_gb = max(self.stats.peak_usage_gb, current_memory)

                    system_info = self.get_system_memory_info()
                    self.stats.available_gb = system_info['available_gb']

                # 检查阈值
                self._check_memory_thresholds(current_memory)

                threading.Event().wait(2)  # 每2秒检查一次

            except Exception as e:
                logger.error(f"内存监控出错: {e}")

    def _check_memory_thresholds(self, current_memory: float):
        """检查内存阈值"""
        timestamp = datetime.now()

        if current_memory >= self.target.critical_threshold:
            alert = {
                'level': 'CRITICAL',
                'message': f'内存使用严重: {current_memory:.2f}GB >= {self.target.critical_threshold}GB',
                'timestamp': timestamp,
                'memory_gb': current_memory
            }
            self.alerts.append(alert)
            logger.critical(alert['message'])

        elif current_memory >= self.target.warning_threshold:
            alert = {
                'level': 'WARNING',
                'message': f'内存使用警告: {current_memory:.2f}GB >= {self.target.warning_threshold}GB',
                'timestamp': timestamp,
                'memory_gb': current_memory
            }
            self.alerts.append(alert)
            logger.warning(alert['message'])

    def get_recent_alerts(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """获取最近的告警"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [alert for alert in self.alerts if alert['timestamp'] >= cutoff_time]

class DataChunker:
    """智能数据分块器"""

    def __init__(self, config: ChunkConfig, memory_monitor: MemoryMonitor):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.config = config
        self.memory_monitor = memory_monitor
        self.current_chunk_size = config.base_chunk_size

    def create_chunks(self, data: Union[List, pd.DataFrame],
                     item_size_estimator: Optional[callable] = None) -> Iterator[Union[List, pd.DataFrame]]:
        """创建数据块"""
        total_items = len(data)
        logger.info(f"开始分块处理，总数据量: {total_items}, 当前块大小: {self.current_chunk_size}")

        for i in range(0, total_items, self.current_chunk_size):
            # 动态调整块大小
            if self.config.adaptive_sizing:
                self._adjust_chunk_size()

            end_idx = min(i + self.current_chunk_size, total_items)

            if isinstance(data, pd.DataFrame):
                chunk = data.iloc[i:end_idx].copy()
            else:
                chunk = data[i:end_idx]

            # 估算内存使用
            if item_size_estimator:
                estimated_size_mb = item_size_estimator(chunk) / (1024 * 1024)
                logger.debug(f"块 {i//self.current_chunk_size + 1} 估计大小: {estimated_size_mb:.1f}MB")

            yield chunk

            # 清理内存
            del chunk
            if i % (self.current_chunk_size * 5) == 0:  # 每5个块清理一次
                gc.collect()

    def create_stock_batches(self, stock_codes: List[str],
                           memory_per_stock_mb: float = 10) -> Iterator[List[str]]:
        """根据内存限制创建股票批次"""
        available_memory_gb = self.memory_monitor.stats.available_gb
        safe_memory_gb = min(available_memory_gb * 0.8, self.memory_monitor.target.max_memory_gb * 0.7)

        max_stocks_per_batch = int((safe_memory_gb * 1024) / memory_per_stock_mb)
        batch_size = max(self.config.min_chunk_size,
                        min(max_stocks_per_batch, self.config.max_chunk_size))

        logger.info(f"计算得出最佳批次大小: {batch_size} (可用内存: {safe_memory_gb:.2f}GB)")

        for i in range(0, len(stock_codes), batch_size):
            yield stock_codes[i:i + batch_size]

    def _adjust_chunk_size(self):
        """动态调整块大小"""
        current_memory = self.memory_monitor.get_current_memory_gb()

        if current_memory >= self.memory_monitor.target.cleanup_threshold:
            # 内存使用过高，减小块大小
            new_size = max(self.config.min_chunk_size, int(self.current_chunk_size * 0.7))
            if new_size != self.current_chunk_size:
                logger.info(f"内存压力调整块大小: {self.current_chunk_size} -> {new_size}")
                self.current_chunk_size = new_size

        elif current_memory < self.memory_monitor.target.warning_threshold:
            # 内存使用较低，可以增大块大小
            new_size = min(self.config.max_chunk_size, int(self.current_chunk_size * 1.2))
            if new_size != self.current_chunk_size:
                logger.debug(f"内存充足调整块大小: {self.current_chunk_size} -> {new_size}")
                self.current_chunk_size = new_size

class MemoryPool:
    """内存池管理器"""

    def __init__(self, max_size_gb: float = 2.0):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.max_size_gb = max_size_gb
        self.pools = {
            'dataframes': OrderedDict(),
            'arrays': OrderedDict(),
            'objects': OrderedDict()
        }
        self.current_size_gb = 0.0
        self._lock = threading.Lock()

        # 弱引用追踪
        self.active_refs = weakref.WeakSet()

    def get_dataframe_pool(self, size_hint: Tuple[int, int]) -> Optional[pd.DataFrame]:
        """从池中获取DataFrame"""
        with self._lock:
            pool_key = self._size_to_key(size_hint)

            # 寻找合适大小的DataFrame
            for key in list(self.pools['dataframes'].keys()):
                if self._is_size_compatible(key, pool_key):
                    df = self.pools['dataframes'].pop(key)
                    logger.debug(f"从内存池获取DataFrame: {key}")
                    return df

            return None

    def return_dataframe_to_pool(self, df: pd.DataFrame):
        """将DataFrame返回到池中"""
        if df is None or df.empty:
            return

        size_key = self._size_to_key(df.shape)
        size_mb = self._estimate_dataframe_size(df)

        with self._lock:
            # 检查内存限制
            if self.current_size_gb + size_mb / 1024 > self.max_size_gb:
                self._cleanup_pool('dataframes')

            # 清理DataFrame并放入池中
            cleaned_df = df.copy()
            cleaned_df.reset_index(drop=True, inplace=True)

            self.pools['dataframes'][size_key] = cleaned_df
            self.current_size_gb += size_mb / 1024

            logger.debug(f"DataFrame返回到内存池: {size_key}, 大小: {size_mb:.1f}MB")

    def get_array_pool(self, shape: Tuple[int, ...], dtype: str) -> Optional[np.ndarray]:
        """从池中获取数组"""
        with self._lock:
            pool_key = f"{shape}_{dtype}"

            if pool_key in self.pools['arrays']:
                array = self.pools['arrays'].pop(pool_key)
                array.fill(0)  # 清零
                return array

            return None

    def return_array_to_pool(self, array: np.ndarray):
        """将数组返回到池中"""
        if array is None:
            return

        pool_key = f"{array.shape}_{array.dtype}"
        size_mb = array.nbytes / (1024 * 1024)

        with self._lock:
            if self.current_size_gb + size_mb / 1024 > self.max_size_gb:
                self._cleanup_pool('arrays')

            self.pools['arrays'][pool_key] = array.copy()
            self.current_size_gb += size_mb / 1024

    def _size_to_key(self, size: Tuple[int, int]) -> str:
        """将大小转换为池键"""
        return f"{size[0]}x{size[1]}"

    def _is_size_compatible(self, pool_key: str, requested_key: str) -> bool:
        """检查大小是否兼容"""
        try:
            pool_rows, pool_cols = map(int, pool_key.split('x'))
            req_rows, req_cols = map(int, requested_key.split('x'))

            # 允许10%的大小差异
            return (abs(pool_rows - req_rows) / req_rows < 0.1 and
                   abs(pool_cols - req_cols) / req_cols < 0.1)
        except:
            return False

    def _estimate_dataframe_size(self, df: pd.DataFrame) -> float:
        """估算DataFrame大小(MB)"""
        return df.memory_usage(deep=True).sum() / (1024 * 1024)

    def _cleanup_pool(self, pool_type: str):
        """清理内存池"""
        if pool_type in self.pools:
            # 移除最旧的项目
            removed_count = 0
            target_removal = max(1, len(self.pools[pool_type]) // 3)

            while self.pools[pool_type] and removed_count < target_removal:
                self.pools[pool_type].popitem(last=False)
                removed_count += 1

            logger.info(f"清理内存池 {pool_type}: 移除 {removed_count} 项")

    def get_pool_stats(self) -> Dict[str, Any]:
        """获取内存池统计信息"""
        with self._lock:
            return {
                'total_pools': sum(len(pool) for pool in self.pools.values()),
                'dataframes': len(self.pools['dataframes']),
                'arrays': len(self.pools['arrays']),
                'objects': len(self.pools['objects']),
                'estimated_size_gb': self.current_size_gb
            }

class DataCompressor:
    """数据压缩器"""

    @staticmethod
    def compress_dataframe(df: pd.DataFrame) -> bytes:
        """压缩DataFrame"""
        pickled_data = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
        compressed_data = zlib.compress(pickled_data, level=6)
        return compressed_data

    @staticmethod
    def decompress_dataframe(compressed_data: bytes) -> pd.DataFrame:
        """解压缩DataFrame"""
        decompressed_data = zlib.decompress(compressed_data)
        df = pickle.loads(decompressed_data)
        return df

    @staticmethod
    def compress_dict(data: Dict[str, Any]) -> bytes:
        """压缩字典数据"""
        pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        compressed_data = zlib.compress(pickled_data, level=6)
        return compressed_data

    @staticmethod
    def decompress_dict(compressed_data: bytes) -> Dict[str, Any]:
        """解压缩字典数据"""
        decompressed_data = zlib.decompress(compressed_data)
        data = pickle.loads(decompressed_data)
        return data

    @staticmethod
    def estimate_compression_ratio(original_size: int, compressed_size: int) -> float:
        """估算压缩比率"""
        if original_size == 0:
            return 0.0
        return (original_size - compressed_size) / original_size * 100

def memory_optimized(cleanup_vars: List[str] = None):
    """内存优化装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 执行前清理垃圾
            gc.collect()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # 执行后清理指定变量
                if cleanup_vars:
                    frame = sys._getframe()
                    for var_name in cleanup_vars:
                        if var_name in frame.f_locals:
                            del frame.f_locals[var_name]

                # 强制垃圾回收
                gc.collect()

        return wrapper
    return decorator

class MemoryOptimizationService:
    """
    内存管理优化器

    确保内存使用控制在4GB以下，实现高效的数据分块处理和内存回收
    """

    def __init__(self,
                 target: Optional[MemoryTarget] = None,
                 chunk_config: Optional[ChunkConfig] = None):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """初始化内存优化器"""
        self.target = target or MemoryTarget()
        self.chunk_config = chunk_config or ChunkConfig()
        self.logger = logger

        # 核心组件
        self.monitor = MemoryMonitor(self.target)
        self.chunker = DataChunker(self.chunk_config, self.monitor)
        self.memory_pool = MemoryPool(max_size_gb=self.target.max_memory_gb * 0.3)
        self.compressor = DataCompressor()

        # 启动监控
        self.monitor.start_monitoring()

        self.logger.info(f"内存优化器初始化完成，目标限制: {self.target.max_memory_gb}GB")

    def optimize_dataframe_processing(self,
                                    df: pd.DataFrame,
                                    processor_func: callable,
                                    **kwargs) -> List[Any]:
        """优化DataFrame处理"""
        results = []

        # 优化DataFrame内存使用
        optimized_df = self._optimize_dataframe_memory(df)

        # 创建数据块
        for chunk in self.chunker.create_chunks(optimized_df):
            try:
                # 检查内存状况
                self._check_memory_and_cleanup()

                # 处理数据块
                chunk_result = processor_func(chunk, **kwargs)
                results.append(chunk_result)

                # 清理块数据
                del chunk

            except MemoryError as e:
                self.logger.error(f"内存不足，跳过数据块: {e}")
                self._emergency_cleanup()
                continue

        return results

    def optimize_stock_batch_processing(self,
                                      stock_codes: List[str],
                                      processor_func: callable,
                                      memory_per_stock_mb: float = 10,
                                      **kwargs) -> List[Any]:
        """优化股票批次处理"""
        results = []
        batch_count = 0

        for batch in self.chunker.create_stock_batches(stock_codes, memory_per_stock_mb):
            batch_count += 1

            try:
                self.logger.info(f"处理第 {batch_count} 批，股票数量: {len(batch)}")

                # 检查内存
                self._check_memory_and_cleanup()

                # 处理批次
                batch_result = processor_func(batch, **kwargs)
                results.append(batch_result)

                # 更新统计
                self.monitor.stats.chunks_processed += 1

            except MemoryError as e:
                self.logger.error(f"批次处理内存不足: {e}")
                self._emergency_cleanup()
                continue

            except Exception as e:
                self.logger.error(f"批次处理失败: {e}")
                continue

        return results

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame内存使用"""
        optimized_df = df.copy()

        # 优化数值类型
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()

            if col_min >= 0:
                if col_max < 255:
                    optimized_df[col] = optimized_df[col].astype(np.uint8)
                elif col_max < 65535:
                    optimized_df[col] = optimized_df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    optimized_df[col] = optimized_df[col].astype(np.uint32)
            else:
                if col_min > -128 and col_max < 127:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32767:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483647:
                    optimized_df[col] = optimized_df[col].astype(np.int32)

        # 优化浮点类型
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')

        # 优化字符串类型
        for col in optimized_df.select_dtypes(include=['object']).columns:
            num_unique = optimized_df[col].nunique()
            total_count = len(optimized_df[col])

            if num_unique / total_count < 0.5:  # 低基数字符串转category
                optimized_df[col] = optimized_df[col].astype('category')

        return optimized_df

    def _check_memory_and_cleanup(self):
        """检查内存并清理"""
        current_memory = self.monitor.get_current_memory_gb()

        if current_memory >= self.target.cleanup_threshold:
            self.logger.warning(f"内存使用过高 ({current_memory:.2f}GB)，执行清理")
            self._perform_cleanup()

    def _perform_cleanup(self):
        """执行清理操作"""
        # 垃圾回收
        collected = gc.collect()
        self.monitor.stats.gc_collections += 1

        # 清理内存池
        self.memory_pool._cleanup_pool('dataframes')
        self.memory_pool._cleanup_pool('arrays')

        after_memory = self.monitor.get_current_memory_gb()
        self.logger.info(f"清理完成，回收 {collected} 个对象，当前内存: {after_memory:.2f}GB")

    def _emergency_cleanup(self):
        """紧急内存清理"""
        self.logger.critical("执行紧急内存清理")

        # 清空内存池
        with self.memory_pool._lock:
            for pool in self.memory_pool.pools.values():
                pool.clear()
            self.memory_pool.current_size_gb = 0.0

        # 强制垃圾回收
        for _ in range(3):
            gc.collect()

        self.logger.info(f"紧急清理后内存: {self.monitor.get_current_memory_gb():.2f}GB")

    def get_memory_report(self) -> Dict[str, Any]:
        """获取内存使用报告"""
        system_info = self.monitor.get_system_memory_info()
        pool_stats = self.memory_pool.get_pool_stats()
        recent_alerts = self.monitor.get_recent_alerts()

        return {
            'memory_stats': {
                'current_usage_gb': self.monitor.stats.current_usage_gb,
                'peak_usage_gb': self.monitor.stats.peak_usage_gb,
                'available_gb': self.monitor.stats.available_gb,
                'target_limit_gb': self.target.max_memory_gb,
                'usage_percentage': (self.monitor.stats.current_usage_gb / self.target.max_memory_gb) * 100
            },
            'system_info': system_info,
            'pool_stats': pool_stats,
            'processing_stats': {
                'chunks_processed': self.monitor.stats.chunks_processed,
                'gc_collections': self.monitor.stats.gc_collections
            },
            'recent_alerts': recent_alerts,
            'thresholds': {
                'warning_gb': self.target.warning_threshold,
                'critical_gb': self.target.critical_threshold,
                'cleanup_gb': self.target.cleanup_threshold
            }
        }

    def __del__(self):
        """析构函数"""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()