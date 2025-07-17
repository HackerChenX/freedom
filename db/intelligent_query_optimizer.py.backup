#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
智能查询优化器

提供数据库查询性能优化、智能缓存和批量处理功能
"""

import time
import threading
import hashlib
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from functools import wraps
import gc

from utils.logger import getLogger
from utils.decorators import performance_monitor, exception_handler

logger = getLogger(__name__)


class IntelligentQueryOptimizer:
    """
    智能查询优化器
    
    功能特性：
    1. 智能查询缓存（多层缓存）
    2. 批量查询优化
    3. 查询模式识别
    4. 性能监控和自适应优化
    5. 内存使用优化
    """
    
    def __init__(self, 
                 max_cache_size: int = 1000,
                 default_cache_ttl: int = 1800,
                 batch_size: int = 100,
                 max_workers: int = 8):
        """
        初始化智能查询优化器
        
        Args:
            max_cache_size: 最大缓存条目数
            default_cache_ttl: 默认缓存TTL（秒）
            batch_size: 默认批次大小
            max_workers: 最大并发工作线程数
        """
        self.max_cache_size = max_cache_size
        self.default_cache_ttl = default_cache_ttl
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # 查询缓存（LRU）
        self.query_cache = OrderedDict()
        self.cache_timestamps = {}
        self.cache_lock = threading.RLock()
        
        # 预聚合缓存
        self.agg_cache = {}
        self.agg_timestamps = {}
        self.agg_lock = threading.RLock()
        
        # 查询模式统计
        self.query_patterns = defaultdict(int)
        self.popular_queries = OrderedDict()
        self.pattern_lock = threading.RLock()
        
        # 性能统计
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_queries': 0,
            'avg_query_time': 0.0,
            'batch_optimizations': 0,
            'memory_optimizations': 0,
            'pattern_matches': 0
        }
        self.stats_lock = threading.Lock()
        
        logger.info(f"智能查询优化器初始化完成 - 缓存大小: {max_cache_size}, TTL: {default_cache_ttl}s")
    
    def cache_query_result(self, func):
        """
        查询结果缓存装饰器
        
        Args:
            func: 要缓存的查询函数
            
        Returns:
            装饰后的函数
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)
            
            # 尝试从缓存获取
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                with self.stats_lock:
                    self.stats['cache_hits'] += 1
                logger.debug(f"缓存命中: {cache_key[:32]}...")
                return cached_result
            
            # 缓存未命中，执行查询
            with self.stats_lock:
                self.stats['cache_misses'] += 1
                self.stats['total_queries'] += 1
            
            start_time = time.time()
            result = func(*args, **kwargs)
            query_time = time.time() - start_time
            
            # 更新性能统计
            self._update_performance_stats(query_time)
            
            # 缓存结果
            self._cache_result(cache_key, result)
            
            # 更新查询模式
            self._update_query_pattern(cache_key)
            
            return result
        
        return wrapper
    
    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def optimize_batch_query(self, stock_codes: List[str], query_builder_func, **query_params) -> Dict[str, Any]:
        """
        批量查询优化
        
        Args:
            stock_codes: 股票代码列表
            query_builder_func: 查询构建函数
            **query_params: 查询参数
            
        Returns:
            Dict[str, Any]: 批量查询结果
        """
        total_codes = len(stock_codes)
        logger.info(f"开始批量查询优化 - 总计 {total_codes} 只股票")
        
        # 动态调整批次大小
        optimized_batch_size = self._calculate_optimal_batch_size(total_codes)
        
        # 检查缓存
        cached_results, missing_codes = self._check_batch_cache(stock_codes, query_params)
        
        results = cached_results.copy()
        
        if missing_codes:
            logger.info(f"缓存命中 {len(cached_results)} 只股票，需要查询 {len(missing_codes)} 只股票")
            
            # 批量查询未缓存的股票
            batched_results = self._execute_batch_queries(
                missing_codes, 
                query_builder_func, 
                optimized_batch_size, 
                **query_params
            )
            
            # 合并结果
            results.update(batched_results)
            
            # 缓存新结果
            self._cache_batch_results(batched_results, query_params)
        
        with self.stats_lock:
            self.stats['batch_optimizations'] += 1
        
        logger.info(f"批量查询完成 - 总计处理 {len(results)} 只股票")
        return results
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': sorted(kwargs.items())
        }
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """从缓存获取结果"""
        with self.cache_lock:
            if cache_key not in self.query_cache:
                return None
            
            # 检查是否过期
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp > self.default_cache_ttl:
                self._remove_from_cache(cache_key)
                return None
            
            # 更新LRU顺序
            result = self.query_cache[cache_key]
            del self.query_cache[cache_key]
            self.query_cache[cache_key] = result
            
            return result
    
    def _cache_result(self, cache_key: str, result: Any):
        """缓存查询结果"""
        with self.cache_lock:
            # LRU缓存管理
            if len(self.query_cache) >= self.max_cache_size:
                # 移除最久未使用的条目
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
                del self.cache_timestamps[oldest_key]
            
            self.query_cache[cache_key] = result
            self.cache_timestamps[cache_key] = time.time()
    
    def _remove_from_cache(self, cache_key: str):
        """从缓存中移除条目"""
        if cache_key in self.query_cache:
            del self.query_cache[cache_key]
        if cache_key in self.cache_timestamps:
            del self.cache_timestamps[cache_key]
    
    def _update_performance_stats(self, query_time: float):
        """更新性能统计"""
        with self.stats_lock:
            # 更新平均查询时间
            total_queries = self.stats['total_queries']
            current_avg = self.stats['avg_query_time']
            new_avg = (current_avg * (total_queries - 1) + query_time) / total_queries
            self.stats['avg_query_time'] = new_avg
    
    def _update_query_pattern(self, cache_key: str):
        """更新查询模式统计"""
        with self.pattern_lock:
            # 提取查询模式（前8位哈希）
            pattern = cache_key[:8]
            self.query_patterns[pattern] += 1
            
            # 更新热门查询
            if len(self.popular_queries) >= 50:
                # 移除最旧的查询
                self.popular_queries.popitem(last=False)
            
            self.popular_queries[cache_key] = time.time()
            
            with self.stats_lock:
                self.stats['pattern_matches'] += 1
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """计算最优批次大小"""
        if total_items <= 100:
            return min(20, total_items)
        elif total_items <= 1000:
            return min(50, total_items // 10)
        elif total_items <= 5000:
            return min(100, total_items // 20)
        else:
            return min(200, total_items // 50)
    
    def _check_batch_cache(self, stock_codes: List[str], query_params: dict) -> Tuple[Dict[str, Any], List[str]]:
        """检查批量查询的缓存状态"""
        cached_results = {}
        missing_codes = []
        
        for code in stock_codes:
            # 生成单个股票的缓存键
            cache_key = self._generate_cache_key(
                'stock_query', 
                (code,), 
                query_params
            )
            
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                cached_results[code] = cached_result
            else:
                missing_codes.append(code)
        
        return cached_results, missing_codes
    
    def _execute_batch_queries(self, stock_codes: List[str], query_builder_func, 
                              batch_size: int, **query_params) -> Dict[str, Any]:
        """执行批量查询"""
        results = {}
        
        # 分批处理
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i + batch_size]
            
            try:
                # 构建批量查询
                batch_query = query_builder_func(batch_codes, **query_params)
                
                # 执行查询（这里需要实际的数据库执行逻辑）
                batch_results = self._execute_single_batch(batch_query, batch_codes)
                
                # 合并结果
                results.update(batch_results)
                
                logger.debug(f"批次 {i//batch_size + 1} 完成，处理 {len(batch_codes)} 只股票")
                
            except Exception as e:
                logger.error(f"批量查询失败 - 批次 {i//batch_size + 1}: {e}")
                # 降级为单个查询
                for code in batch_codes:
                    try:
                        single_query = query_builder_func([code], **query_params)
                        single_result = self._execute_single_batch(single_query, [code])
                        results.update(single_result)
                    except Exception as single_e:
                        logger.error(f"单个股票查询也失败 {code}: {single_e}")
        
        return results
    
    def _execute_single_batch(self, query: str, stock_codes: List[str]) -> Dict[str, Any]:
        """执行单个批量查询（需要子类实现）"""
        # 这是一个抽象方法，需要在具体实现中连接实际的数据库
        # 这里返回模拟结果
        results = {}
        for code in stock_codes:
            # 模拟查询结果
            results[code] = pd.DataFrame({
                'code': [code],
                'date': [datetime.now().strftime('%Y-%m-%d')],
                'close': [100.0]
            })
        return results
    
    def _cache_batch_results(self, results: Dict[str, Any], query_params: dict):
        """缓存批量查询结果"""
        for code, result in results.items():
            cache_key = self._generate_cache_key(
                'stock_query', 
                (code,), 
                query_params
            )
            self._cache_result(cache_key, result)
    
    @performance_monitor(threshold=1.0)
    def optimize_memory_usage(self):
        """优化内存使用"""
        logger.info("开始内存优化...")
        
        # 清理过期缓存
        self._cleanup_expired_cache()
        
        # 强制垃圾回收
        collected = gc.collect()
        
        with self.stats_lock:
            self.stats['memory_optimizations'] += 1
        
        logger.info(f"内存优化完成 - 垃圾回收清理 {collected} 个对象")
    
    def _cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        
        with self.cache_lock:
            for key, timestamp in self.cache_timestamps.items():
                if current_time - timestamp > self.default_cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_from_cache(key)
        
        if expired_keys:
            logger.debug(f"清理 {len(expired_keys)} 个过期缓存条目")
    
    def get_cache_stats_intelligent_query_optimizer(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.cache_lock:
            cache_size = len(self.query_cache)
        
        with self.stats_lock:
            stats = self.stats.copy()
        
        # 计算缓存命中率
        total_requests = stats['cache_hits'] + stats['cache_misses']
        hit_rate = (stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': cache_size,
            'max_cache_size': self.max_cache_size,
            'cache_hit_rate': f"{hit_rate:.1f}%",
            'total_queries': stats['total_queries'],
            'avg_query_time': f"{stats['avg_query_time']:.3f}s",
            'batch_optimizations': stats['batch_optimizations'],
            'memory_optimizations': stats['memory_optimizations'],
            'pattern_matches': stats['pattern_matches']
        }
    
    def clear_cache_intelligent_query_optimizer(self):
        """清空所有缓存"""
        with self.cache_lock:
            self.query_cache.clear()
            self.cache_timestamps.clear()
        
        with self.agg_lock:
            self.agg_cache.clear()
            self.agg_timestamps.clear()
        
        logger.info("所有缓存已清空")


# 全局查询优化器实例
_global_optimizer = None

def get_query_optimizer() -> IntelligentQueryOptimizer:
    """获取全局查询优化器实例"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = IntelligentQueryOptimizer()
    return _global_optimizer 