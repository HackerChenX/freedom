"""
内存优化器

实现智能内存管理和数据优化，解决大量数据同时加载的内存问题。
支持内存监控、数据分块处理和自动垃圾回收。

Author: System
Date: 2025-01-15
"""

from typing import List, Dict, Any, Optional, Generator, Tuple
import psutil
import gc
import time
from dataclasses import dataclass
from contextlib import contextmanager

from utils.logger import get_logger

logger = get_logger(__name__)


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
    
    def to_dict_memory_optimizer(self) -> Dict[str, Any]:
        return {
            'total_memory_gb': self.total_memory_gb,
            'available_memory_gb': self.available_memory_gb,
            'used_memory_gb': self.used_memory_gb,
            'memory_usage_percent': self.memory_usage_percent,
            'process_memory_mb': self.process_memory_mb
        }


class MemoryOptimizationService:
    """
    内存优化器
    
    提供智能内存管理功能：
    - 实时内存监控
    - 数据分块处理
    - 自动内存优化
    - DataFrame内存优化
    - 垃圾回收管理
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.process = psutil.Process()
        self.gc_counter = 0
        self.optimization_history = []
        
    def get_memory_stats(self) -> MemoryStats:
        """
        获取当前内存统计信息
        
        Returns:
            MemoryStats: 内存统计信息
        """
        # 系统内存信息
        memory = psutil.virtual_memory()
        
        # 进程内存信息
        process_memory = self.process.memory_info()
        
        return MemoryStats(
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
    def memory_monitor(self, operation_name: str = "operation"):
        """
        内存监控上下文管理器
        
        Args:
            operation_name: 操作名称
        """
        start_time = time.time()
        start_memory = self.get_memory_stats()
        
        try:
            logger.debug(f"开始监控操作: {operation_name}")
            yield
        finally:
            end_memory = self.get_memory_stats()
            duration = time.time() - start_time
            
            memory_change = end_memory.memory_usage_percent - start_memory.memory_usage_percent
            
            logger.debug(
                f"操作完成: {operation_name}, "
                f"耗时: {duration:.2f}s, "
                f"内存变化: {memory_change:+.1f}%"
            )
    
    def optimize_dataframe(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        优化DataFrame内存使用
        
        Args:
            df: 原始DataFrame
            inplace: 是否就地修改
            
        Returns:
            pd.DataFrame: 优化后的DataFrame
        """
        if df.empty:
            return df
        
        result_df = df if inplace else df.copy()
        original_memory = df.memory_usage(deep=True).sum()
        
        # 优化数值列的数据类型
        for col in result_df.select_dtypes(include=[np.number]).columns:
            col_data = result_df[col]
            
            # 检查是否可以使用更小的数据类型
            if col_data.dtype == 'float64':
                if col_data.min() >= np.finfo(np.float32).min and col_data.max() <= np.finfo(np.float32).max:
                    result_df[col] = col_data.astype(np.float32)
            
            elif col_data.dtype == 'int64':
                if col_data.min() >= np.iinfo(np.int32).min and col_data.max() <= np.iinfo(np.int32).max:
                    result_df[col] = col_data.astype(np.int32)
        
        # 优化字符串列
        for col in result_df.select_dtypes(include=['object']).columns:
            if result_df[col].dtype == 'object':
                try:
                    result_df[col] = result_df[col].astype('category')
                except:
                    pass  # 如果转换失败，保持原始类型
        
        optimized_memory = result_df.memory_usage(deep=True).sum()
        memory_saved = original_memory - optimized_memory
        
        if memory_saved > 0:
            logger.debug(f"DataFrame内存优化: 节省 {memory_saved / 1024**2:.1f}MB")
        
        return result_df
    
    def create_chunked_iterator(
        self, 
        data: List[Any], 
        chunk_size: Optional[int] = None
    ) -> Generator[List[Any], None, None]:
        """
        创建分块迭代器
        
        Args:
            data: 数据列表
            chunk_size: 块大小
            
        Yields:
            List[Any]: 数据块
        """
        if chunk_size is None:
            # 根据内存使用情况动态计算块大小
            memory_stats = self.get_memory_stats()
            if memory_stats.memory_usage_percent > 70:
                chunk_size = max(10, len(data) // 100)
            else:
                chunk_size = max(50, len(data) // 20)
        
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    
    def process_dataframe_chunks(
        self,
        df: pd.DataFrame,
        processing_func: callable,
        chunk_size: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """
        分块处理DataFrame
        
        Args:
            df: 输入DataFrame
            processing_func: 处理函数
            chunk_size: 块大小
            **kwargs: 额外参数
            
        Returns:
            List[Any]: 处理结果列表
        """
        if chunk_size is None:
            # 根据内存和数据大小动态计算
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            if memory_mb > self.config.chunk_size_mb:
                chunk_size = max(100, len(df) // int(memory_mb / self.config.chunk_size_mb))
            else:
                chunk_size = len(df)
        
        results = []
        
        with self.memory_monitor(f"分块处理DataFrame({len(df)}行)"):
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                
                try:
                    result = processing_func(chunk, **kwargs)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"处理数据块失败 [{i}:{i+chunk_size}]: {e}")
                
                # 定期检查内存
                if i % (chunk_size * 10) == 0:
                    self.auto_optimize_memory()
        
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
            "current_memory_stats": stats.to_dict(),
            "optimization_config": {
                "max_memory_usage_percent": self.config.max_memory_usage_percent,
                "chunk_size_mb": self.config.chunk_size_mb,
                "gc_threshold": self.config.gc_threshold,
                "auto_optimize": self.config.auto_optimize
            },
            "gc_counter": self.gc_counter,
            "optimization_count": len(self.optimization_history)
        } 