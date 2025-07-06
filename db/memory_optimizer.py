"""
内存优化器

实现智能内存管理和数据优化，解决大量数据同时加载的内存问题。
支持内存监控、数据分块处理和自动垃圾回收。

Author: System
Date: 2025-01-15
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Generator, Tuple
import psutil
import gc
import logging
import time
from dataclasses import dataclass
from contextlib import contextmanager

from utils.logger import getLogger

logger = getLogger(__name__)


@dataclass
class MemoryConfig:
    """内存管理配置"""
    max_memory_usage_percent: float = 80.0  # 最大内存使用百分比
    chunk_size_mb: int = 100  # 数据块大小(MB)
    gc_threshold: int = 1000  # 垃圾回收阈值
    warning_threshold_percent: float = 70.0  # 内存警告阈值
    auto_optimize: bool = True  # 自动优化
    
    
@dataclass
class MemoryStats:
    """内存统计信息"""
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    memory_usage_percent: float
    process_memory_mb: float
    
    def to_dict_Optimizer(self) -> Dict[str, Any]:
        return {
            'total_memory_gb': self.total_memory_gb,
            'available_memory_gb': self.available_memory_gb,
            'used_memory_gb': self.used_memory_gb,
            'memory_usage_percent': self.memory_usage_percent,
            'process_memory_mb': self.process_memory_mb
        }


class MemoryOptimizer:
    """
    内存优化器
    
    提供智能内存管理功能：
    - 实时内存监控
    - 数据分块处理
    - 自动内存优化
    - Data_frame内存优化
    - 垃圾回收管理
    """
    
    def __init___31(self, config: Optional[Memory_config] = None):
        self.config = config or Memory_config()
        self.process = psutil.Process()
        self.gc_counter = 0
        self.optimization_history = []
        
    def get_memory_stats(self) -> Memory_stats:
        """
        获取当前内存统计信息
        
        Returns:
            Memory_stats: 内存统计信息
        """
        # 系统内存信息
        memory = psutil.virtual_memory()
        
        # 进程内存信息
        process_memory = self.process.memory_info()
        
        return Memory_stats(
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            used_memory_gb=memory.used / (1024**3),
            memory_usage_percent=memory.percent,
            process_memory_mb=process_memory.rss / (1024**2)
        )
    
    def check_memory_usage(self) -> bool:
        """
        检查内存使用情况
        
        Returns:
            bool: 内存使用是否正常
        """
        stats = self.get_memory_stats()
        
        if stats.memory_usage_percent > self.config.max_memory_usage_percent:
            logger.warning(f"内存使用率过高: {stats.memory_usage_percent:.1f}%")
            return False
        elif stats.memory_usage_percent > self.config.warning_threshold_percent:
            logger.info(f"内存使用率警告: {stats.memory_usage_percent:.1f}%")
        
        return True
    
    @contextmanager
    def memory_monitor(self, operation_name: str = "操作"):
        """
        内存监控上下文管理器
        
        Args:
            operation_name: 操作名称
        """
        start_stats = self.get_memory_stats()
        start_time = time.time()
        
        logger.info(f"开始{operation_name} - 内存使用: {start_stats.process_memory_mb:.1f}MB")
        
        try:
            yield start_stats
        finally:
            end_stats = self.get_memory_stats()
            duration = time.time() - start_time
            
            memory_diff = end_stats.process_memory_mb - start_stats.process_memory_mb
            
            logger.info(f"完成{operation_name} - 耗时: {duration:.2f}秒, "
                       f"内存变化: {memory_diff:+.1f}MB, "
                       f"当前内存: {end_stats.process_memory_mb:.1f}MB")
            
            # 自动优化
            if self.config.auto_optimize and memory_diff > 0:
                self.auto_optimize_memory()
    
    def optimize_dataframe(self, df: pd.DataFrame, 
                          optimize_categories: bool = True) -> pd.DataFrame:
        """
        优化Data_frame内存使用
        
        Args:
            df: 原始Data_frame
            optimize_categories: 是否优化分类数据
            
        Returns:
            pd.DataFrame: 优化后的Data_frame
        """
        if df.empty:
            return df
        
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # 优化数值类型
        df = self._optimize_numeric_columns(df)
        
        # 优化字符串类型
        if optimize_categories:
            df = self._optimize_string_columns(df)
        
        # 优化日期时间类型
        df = self._optimize_datetime_columns(df)
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
        memory_saved = original_memory - optimized_memory
        
        if memory_saved > 0:
            logger.debug(f"DataFrame内存优化: {original_memory:.1f}MB -> {optimized_memory:.1f}MB "
                        f"(节省 {memory_saved:.1f}MB, {memory_saved/original_memory*100:.1f}%)")
        
        return df
    
    def _optimize_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数值列"""
        # 优化整数类型
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype('int32')
        
        # 优化浮点类型
        for col in df.select_dtypes(include=['float64']).columns:
            # 检查是否可以转换为float32而不丢失精度
            if self._can_downcast_float(df[col]):
                df[col] = df[col].astype('float32')
        
        return df
    
    def _optimize_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化字符串列"""
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                # 检查是否适合转换为分类数据
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # 如果唯一值比例小于50%，转换为分类
                    df[col] = df[col].astype('category')
        
        return df
    
    def _optimize_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化日期时间列"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # 尝试转换为日期时间类型
                try:
                    if df[col].str.match(r'\d{4}-\d{2}-\d{2}').all():
                        df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        return df
    
    def _can_downcast_float(self, series: pd.Series) -> bool:
        """检查浮点数是否可以安全转换为float32"""
        try:
            # 转换为float32并检查是否有精度损失
            float32_series = series.astype('float32')
            return np.allclose(series, float32_series, equal_nan=True)
        except:
            return False
    
    def chunk_dataframe(self, df: pd.DataFrame, 
                       chunk_size_mb: Optional[int] = None) -> Generator[pd.DataFrame, None, None]:
        """
        将Data_frame分块处理
        
        Args:
            df: 原始Data_frame
            chunk_size_mb: 块大小(MB)
            
        Yields:
            pd.DataFrame: 数据块
        """
        chunk_size_mb = chunk_size_mb or self.config.chunk_size_mb
        
        if df.empty:
            yield df
            return
        
        # 计算每个块的行数
        row_memory_mb = df.memory_usage(deep=True).sum() / len(df) / 1024**2
        rows_per_chunk = max(1, int(chunk_size_mb / row_memory_mb))
        
        logger.info(f"数据分块处理: {len(df)}行 -> 每块{rows_per_chunk}行 "
                   f"(目标大小: {chunk_size_mb}MB)")
        
        for i in range(0, len(df), rows_per_chunk):
            chunk = df.iloc[i:i + rows_per_chunk].copy()
            yield self.optimize_dataframe(chunk)
    
    def process_large_dataset(self, data_source: Any, 
                             processing_func: callable,
                             chunk_size_mb: Optional[int] = None) -> List[Any]:
        """
        处理大型数据集
        
        Args:
            data_source: 数据源
            processing_func: 处理函数
            chunk_size_mb: 块大小(MB)
            
        Returns:
            List[Any]: 处理结果列表
        """
        chunk_size_mb = chunk_size_mb or self.config.chunk_size_mb
        results = []
        
        with self.memory_monitor("大型数据集处理"):
            if isinstance(data_source, pd.DataFrame):
                # DataFrame分块处理
                for chunk in self.chunk_dataframe(data_source, chunk_size_mb):
                    with self.memory_monitor(f"处理数据块({len(chunk)}行)"):
                        result = processing_func(chunk)
                        results.append(result)
                        
                        # 检查内存并清理
                        self._check_and_cleanup()
            
            elif isinstance(data_source, dict):
                # 字典数据分块处理
                items = list(data_source.items())
                chunk_size = max(1, chunk_size_mb * 1024 // 10)  # 估算项目数量
                
                for i in range(0, len(items), chunk_size):
                    chunk_dict = dict(items[i:i + chunk_size])
                    with self.memory_monitor(f"处理数据块({len(chunk_dict)}项)"):
                        result = processing_func(chunk_dict)
                        results.append(result)
                        
                        # 检查内存并清理
                        self._check_and_cleanup()
            
            else:
                # 其他类型直接处理
                result = processing_func(data_source)
                results.append(result)
        
        return results
    
    def auto_optimize_memory(self):
        """自动内存优化"""
        self.gc_counter += 1
        
        # 执行垃圾回收
        if self.gc_counter >= self.config.gc_threshold:
            collected = gc.collect()
            self.gc_counter = 0
            logger.debug(f"执行垃圾回收: 回收了 {collected} 个对象")
        
        # 检查内存使用
        stats = self.get_memory_stats()
        
        if stats.memory_usage_percent > self.config.max_memory_usage_percent:
            logger.warning(f"内存使用率过高({stats.memory_usage_percent:.1f}%)，强制垃圾回收")
            
            # 强制垃圾回收
            for i in range(3):
                collected = gc.collect()
                if collected == 0:
                    break
            
            # 重新检查内存
            new_stats = self.get_memory_stats()
            memory_freed = stats.process_memory_mb - new_stats.process_memory_mb
            
            if memory_freed > 0:
                logger.info(f"内存优化完成: 释放了 {memory_freed:.1f}MB")
    
    def _check_and_cleanup(self):
        """检查内存并清理"""
        if not self.check_memory_usage():
            self.auto_optimize_memory()
    
    def optimize_stock_data_dict(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        优化股票数据字典
        
        Args:
            stock_data: 股票数据字典
            
        Returns:
            Dict[str, pd.DataFrame]: 优化后的股票数据字典
        """
        optimized_data = {}
        total_memory_saved = 0
        
        with self.memory_monitor("股票数据优化"):
            for code, df in stock_data.items():
                original_memory = df.memory_usage(deep=True).sum() / 1024**2
                
                optimized_df = self.optimize_dataframe(df)
                optimized_data[code] = optimized_df
                
                optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2
                memory_saved = original_memory - optimized_memory
                total_memory_saved += memory_saved
                
                # 定期清理
                if len(optimized_data) % 100 == 0:
                    self._check_and_cleanup()
        
        logger.info(f"股票数据优化完成: {len(stock_data)}只股票, "
                   f"总计节省内存 {total_memory_saved:.1f}MB")
        
        return optimized_data
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        获取优化报告
        
        Returns:
            Dict[str, Any]: 优化报告
        """
        stats = self.get_memory_stats()
        
        return {
            "current_memory_stats": stats.to_dict_Optimizer(),
            "optimization_config": {
                "max_memory_usage_percent": self.config.max_memory_usage_percent,
                "chunk_size_mb": self.config.chunk_size_mb,
                "gc_threshold": self.config.gc_threshold,
                "auto_optimize": self.config.auto_optimize
            },
            "gc_counter": self.gc_counter,
            "optimization_count": len(self.optimization_history)
        }
    
    def suggest_optimal_batch_size(self, single_item_memory_mb: float) -> int:
        """
        建议最优批次大小
        
        Args:
            single_item_memory_mb: 单个项目的内存使用(MB)
            
        Returns:
            int: 建议的批次大小
        """
        stats = self.get_memory_stats()
        available_memory_mb = stats.available_memory_gb * 1024
        
        # 保留30%的内存作为缓冲
        usable_memory_mb = available_memory_mb * 0.7
        
        # 计算建议的批次大小
        suggested_batch_size = max(1, int(usable_memory_mb / single_item_memory_mb))
        
        # 限制最大批次大小
        max_batch_size = 1000
        suggested_batch_size = min(suggested_batch_size, max_batch_size)
        
        logger.info(f"建议批次大小: {suggested_batch_size} "
                   f"(单项内存: {single_item_memory_mb:.1f}MB, "
                   f"可用内存: {available_memory_mb:.1f}MB)")
        
        return suggested_batch_size 