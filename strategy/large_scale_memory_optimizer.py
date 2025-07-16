"""
大规模股票选股内存优化器

专门解决4000+股票选股时的内存使用和GC压力问题。
提供流式处理、智能批次调整、内存监控和自动优化功能。

Author: System
Date: 2025-01-15
"""

import pandas as pd
import numpy as np
import gc
import psutil
import time
import logging
from typing import List, Dict, Any, Optional, Generator, Tuple, Iterator
from dataclasses import dataclass
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import weakref
import sys
from functools import wraps

from utils.logger import get_logger
from db.memory_optimizer import MemoryOptimizer, MemoryConfig

logger = get_logger(__name__)


@dataclass
class LargeScaleMemoryConfig:
    """大规模内存管理配置"""
    # 基础内存配置
    max_memory_usage_percent: float = 75.0  # 最大内存使用率
    critical_memory_threshold: float = 85.0  # 危险内存阈值
    warning_memory_threshold: float = 70.0   # 警告内存阈值
    
    # 批次处理配置
    initial_batch_size: int = 100           # 初始批次大小
    min_batch_size: int = 20                # 最小批次大小
    max_batch_size: int = 500               # 最大批次大小
    
    # GC配置
    gc_frequency: int = 50                  # GC频率（每处理N个股票）
    force_gc_threshold: float = 80.0        # 强制GC内存阈值
    
    # 数据管理配置
    max_cached_stocks: int = 200            # 最大缓存股票数
    enable_streaming: bool = True           # 启用流式处理
    enable_chunked_processing: bool = True  # 启用分块处理
    
    # 优化配置
    auto_batch_adjustment: bool = True      # 自动批次调整
    memory_pressure_relief: bool = True     # 内存压力释放
    dataframe_optimization: bool = True     # DataFrame优化


class LargeScaleMemoryOptimizer:
    """
    大规模股票选股内存优化器
    
    核心功能：
    1. 智能批次管理 - 根据内存使用动态调整批次大小
    2. 流式数据处理 - 避免一次性加载所有数据
    3. 分块计算优化 - 将大任务分解为小块处理
    4. 内存压力监控 - 实时监控并自动释放内存
    5. GC策略优化 - 智能垃圾回收管理
    """
    
    def __init__(self, config: Optional[LargeScaleMemoryConfig] = None):
        self.config = config or LargeScaleMemoryConfig()
        self.base_optimizer = MemoryOptimizer()
        
        # 状态管理
        self.current_batch_size = self.config.initial_batch_size
        self.processed_count = 0
        self.gc_count = 0
        self.memory_pressure_events = 0
        
        # 性能统计
        self.performance_stats = {
            'total_processed': 0,
            'memory_peak': 0.0,
            'gc_triggered': 0,
            'batch_adjustments': 0,
            'memory_pressure_relief_count': 0,
            'processing_time': 0.0,
            'memory_saved_mb': 0.0
        }
        
        # 内存监控
        self.memory_history = []
        self.batch_history = []
        
    @contextmanager
    def memory_managed_processing(self, operation_name: str = "processing"):
        """
        内存管理上下文管理器
        
        Args:
            operation_name: 操作名称
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            logger.info(f"🚀 开始内存管理处理: {operation_name}")
            logger.info(f"   初始内存使用: {start_memory:.1f}%")
            
            yield self
            
        finally:
            end_memory = self._get_memory_usage()
            processing_time = time.time() - start_time
            
            # 更新性能统计
            self.performance_stats['processing_time'] += processing_time
            self.performance_stats['memory_peak'] = max(
                self.performance_stats['memory_peak'], 
                end_memory
            )
            
            logger.info(f"✅ 完成内存管理处理: {operation_name}")
            logger.info(f"   结束内存使用: {end_memory:.1f}%")
            logger.info(f"   处理耗时: {processing_time:.2f}秒")
            
            # 最终清理
            self._force_memory_cleanup()
    
    def process_large_stock_selection(
        self, 
        stock_codes: List[str], 
        processing_func: callable,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        大规模股票选股处理
        
        Args:
            stock_codes: 股票代码列表
            processing_func: 处理函数
            **kwargs: 额外参数
            
        Yields:
            Dict[str, Any]: 处理结果
        """
        total_stocks = len(stock_codes)
        logger.info(f"🎯 开始大规模股票选股: {total_stocks}只股票")
        
        with self.memory_managed_processing("大规模股票选股"):
            # 分批处理
            for batch_idx, batch_codes in enumerate(self._create_dynamic_batches(stock_codes)):
                batch_start_time = time.time()
                
                logger.info(f"📦 处理批次 {batch_idx + 1}, 股票数: {len(batch_codes)}, "
                           f"批次大小: {self.current_batch_size}")
                
                try:
                    # 检查内存状态
                    if self._check_memory_pressure():
                        self._handle_memory_pressure()
                    
                    # 处理当前批次
                    batch_results = self._process_batch_with_memory_management(
                        batch_codes, processing_func, **kwargs
                    )
                    
                    # 返回结果
                    for result in batch_results:
                        yield result
                    
                    # 批次后清理
                    self._post_batch_cleanup(batch_idx)
                    
                    batch_time = time.time() - batch_start_time
                    logger.info(f"   批次处理完成，耗时: {batch_time:.2f}秒")
                    
                except Exception as e:
                    logger.error(f"批次 {batch_idx + 1} 处理失败: {e}")
                    # 内存紧急释放
                    self._emergency_memory_cleanup()
                    continue
        
        # 输出最终统计
        self._log_final_statistics(total_stocks)
    
    def _create_dynamic_batches(self, stock_codes: List[str]) -> Generator[List[str], None, None]:
        """
        创建动态调整的批次
        
        Args:
            stock_codes: 股票代码列表
            
        Yields:
            List[str]: 批次股票代码
        """
        total_stocks = len(stock_codes)
        processed = 0
        
        while processed < total_stocks:
            # 动态调整批次大小
            current_memory = self._get_memory_usage()
            self.current_batch_size = self._calculate_optimal_batch_size(current_memory)
            
            # 获取当前批次
            end_idx = min(processed + self.current_batch_size, total_stocks)
            batch = stock_codes[processed:end_idx]
            
            # 记录批次历史
            self.batch_history.append({
                'batch_size': len(batch),
                'memory_usage': current_memory,
                'timestamp': time.time()
            })
            
            yield batch
            processed = end_idx
    
    def _calculate_optimal_batch_size(self, memory_usage: float) -> int:
        """
        根据内存使用情况计算最优批次大小
        
        Args:
            memory_usage: 当前内存使用率
            
        Returns:
            int: 最优批次大小
        """
        if not self.config.auto_batch_adjustment:
            return self.current_batch_size
        
        # 基于内存使用率调整
        if memory_usage > self.config.critical_memory_threshold:
            # 危险区域：大幅减少批次
            new_size = max(self.config.min_batch_size, self.current_batch_size // 4)
            logger.warning(f"⚠️  内存危险({memory_usage:.1f}%)，批次大小: {self.current_batch_size} → {new_size}")
        elif memory_usage > self.config.warning_memory_threshold:
            # 警告区域：适度减少批次
            new_size = max(self.config.min_batch_size, int(self.current_batch_size * 0.7))
            logger.info(f"⚡ 内存警告({memory_usage:.1f}%)，批次大小: {self.current_batch_size} → {new_size}")
        elif memory_usage < 50:
            # 内存充足：可以增加批次
            new_size = min(self.config.max_batch_size, int(self.current_batch_size * 1.3))
            if new_size != self.current_batch_size:
                logger.info(f"📈 内存充足({memory_usage:.1f}%)，批次大小: {self.current_batch_size} → {new_size}")
        else:
            # 正常区域：保持当前大小
            new_size = self.current_batch_size
        
        if new_size != self.current_batch_size:
            self.performance_stats['batch_adjustments'] += 1
        
        return new_size
    
    def _process_batch_with_memory_management(
        self, 
        batch_codes: List[str], 
        processing_func: callable, 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        使用内存管理处理批次
        
        Args:
            batch_codes: 批次股票代码
            processing_func: 处理函数
            **kwargs: 额外参数
            
        Returns:
            List[Dict[str, Any]]: 批次处理结果
        """
        results = []
        
        for i, stock_code in enumerate(batch_codes):
            try:
                # 单股票处理
                result = processing_func(stock_code, **kwargs)
                if result:
                    results.append(result)
                
                # 定期检查内存
                if i % 10 == 0:
                    self._check_and_optimize_memory()
                
                self.processed_count += 1
                
            except Exception as e:
                logger.error(f"处理股票 {stock_code} 失败: {e}")
                continue
        
        return results
    
    def _check_memory_pressure(self) -> bool:
        """
        检查内存压力
        
        Returns:
            bool: 是否存在内存压力
        """
        memory_usage = self._get_memory_usage()
        return memory_usage > self.config.warning_memory_threshold
    
    def _handle_memory_pressure(self):
        """处理内存压力"""
        self.memory_pressure_events += 1
        self.performance_stats['memory_pressure_relief_count'] += 1
        
        logger.warning("🚨 检测到内存压力，执行内存优化...")
        
        # 1. 强制垃圾回收
        self._force_garbage_collection()
        
        # 2. 清理缓存
        self._clear_caches()
        
        # 3. 调整批次大小
        self.current_batch_size = max(
            self.config.min_batch_size, 
            self.current_batch_size // 2
        )
        
        logger.info(f"   内存优化完成，新批次大小: {self.current_batch_size}")
    
    def _post_batch_cleanup(self, batch_idx: int):
        """
        批次后清理
        
        Args:
            batch_idx: 批次索引
        """
        # 定期垃圾回收
        if batch_idx % self.config.gc_frequency == 0:
            self._force_garbage_collection()
        
        # 记录内存使用
        current_memory = self._get_memory_usage()
        self.memory_history.append({
            'batch_idx': batch_idx,
            'memory_usage': current_memory,
            'timestamp': time.time()
        })
        
        # 检查内存趋势
        if len(self.memory_history) > 5:
            self._analyze_memory_trend()
    
    def _check_and_optimize_memory(self):
        """检查并优化内存"""
        memory_usage = self._get_memory_usage()
        
        if memory_usage > self.config.force_gc_threshold:
            self._force_garbage_collection()
            
            # 再次检查
            new_memory_usage = self._get_memory_usage()
            memory_freed = memory_usage - new_memory_usage
            
            if memory_freed > 0:
                self.performance_stats['memory_saved_mb'] += memory_freed
                logger.debug(f"内存优化释放: {memory_freed:.1f}%")
    
    def _force_garbage_collection(self):
        """强制垃圾回收"""
        before_memory = self._get_memory_usage()
        
        # 多轮垃圾回收
        for i in range(3):
            collected = gc.collect()
            if collected == 0:
                break
        
        after_memory = self._get_memory_usage()
        memory_freed = before_memory - after_memory
        
        self.gc_count += 1
        self.performance_stats['gc_triggered'] += 1
        
        if memory_freed > 1:  # 超过1%的内存释放才记录
            logger.info(f"🗑️  垃圾回收释放内存: {memory_freed:.1f}%")
    
    def _clear_caches(self):
        """清理缓存"""
        # 清理pandas缓存
        if hasattr(pd, '_cache'):
            pd._cache.clear()
        
        # 清理numpy缓存
        if hasattr(np, '_cache'):
            np._cache.clear()
        
        logger.debug("缓存清理完成")
    
    def _emergency_memory_cleanup(self):
        """紧急内存清理"""
        logger.error("🚨 执行紧急内存清理...")
        
        # 1. 强制垃圾回收
        for _ in range(5):
            gc.collect()
        
        # 2. 清理所有缓存
        self._clear_caches()
        
        # 3. 重置批次大小到最小值
        self.current_batch_size = self.config.min_batch_size
        
        logger.info("紧急内存清理完成")
    
    def _force_memory_cleanup(self):
        """强制内存清理"""
        self._force_garbage_collection()
        self._clear_caches()
    
    def _analyze_memory_trend(self):
        """分析内存趋势"""
        if len(self.memory_history) < 3:
            return
        
        recent_usage = [entry['memory_usage'] for entry in self.memory_history[-3:]]
        
        # 检查内存是否持续上升
        if all(recent_usage[i] < recent_usage[i+1] for i in range(len(recent_usage)-1)):
            trend_increase = recent_usage[-1] - recent_usage[0]
            if trend_increase > 10:  # 内存上升超过10%
                logger.warning(f"检测到内存持续上升趋势: +{trend_increase:.1f}%")
                self._handle_memory_pressure()
    
    def _get_memory_usage(self) -> float:
        """
        获取当前内存使用率
        
        Returns:
            float: 内存使用率百分比
        """
        return psutil.virtual_memory().percent
    
    def _log_final_statistics(self, total_stocks: int):
        """
        记录最终统计信息
        
        Args:
            total_stocks: 总股票数
        """
        stats = self.performance_stats
        
        logger.info("📊 大规模股票选股完成统计:")
        logger.info(f"   总处理股票数: {total_stocks}")
        logger.info(f"   处理耗时: {stats['processing_time']:.2f}秒")
        logger.info(f"   平均速度: {total_stocks/stats['processing_time']:.1f}股/秒")
        logger.info(f"   内存峰值: {stats['memory_peak']:.1f}%")
        logger.info(f"   GC触发次数: {stats['gc_triggered']}")
        logger.info(f"   批次调整次数: {stats['batch_adjustments']}")
        logger.info(f"   内存压力处理次数: {stats['memory_pressure_relief_count']}")
        logger.info(f"   累计节省内存: {stats['memory_saved_mb']:.1f}%")
    
    def get_memory_optimization_report(self) -> Dict[str, Any]:
        """
        获取内存优化报告
        
        Returns:
            Dict[str, Any]: 优化报告
        """
        return {
            'config': {
                'max_memory_usage_percent': self.config.max_memory_usage_percent,
                'initial_batch_size': self.config.initial_batch_size,
                'auto_batch_adjustment': self.config.auto_batch_adjustment,
                'enable_streaming': self.config.enable_streaming
            },
            'performance_stats': self.performance_stats.copy(),
            'memory_history': self.memory_history[-10:],  # 最近10条记录
            'batch_history': self.batch_history[-10:],    # 最近10条记录
            'current_state': {
                'current_batch_size': self.current_batch_size,
                'processed_count': self.processed_count,
                'gc_count': self.gc_count,
                'memory_pressure_events': self.memory_pressure_events
            }
        }


def memory_optimized_stock_selection():
    """内存优化的股票选股示例"""
    def example_processing_func(stock_code: str) -> Dict[str, Any]:
        """示例处理函数"""
        # 模拟股票分析处理
        time.sleep(0.001)  # 模拟计算时间
        return {
            'stock_code': stock_code,
            'score': np.random.random(),
            'processed_at': time.time()
        }
    
    # 创建4000只股票的示例列表
    stock_codes = [f"{i:06d}.SH" for i in range(4000)]
    
    # 创建内存优化器
    config = LargeScaleMemoryConfig(
        max_memory_usage_percent=75.0,
        initial_batch_size=100,
        auto_batch_adjustment=True,
        enable_streaming=True
    )
    
    optimizer = LargeScaleMemoryOptimizer(config)
    
    # 执行优化的股票选股
    results = []
    for result in optimizer.process_large_stock_selection(
        stock_codes, 
        example_processing_func
    ):
        results.append(result)
    
    # 获取优化报告
    report = optimizer.get_memory_optimization_report()
    
    return results, report


if __name__ == "__main__":
    # 运行内存优化示例
    results, report = memory_optimized_stock_selection()
    
    print("内存优化股票选股完成!")
    print(f"处理结果数: {len(results)}")
    print(f"内存峰值: {report['performance_stats']['memory_peak']:.1f}%")
    print(f"GC触发次数: {report['performance_stats']['gc_triggered']}") 