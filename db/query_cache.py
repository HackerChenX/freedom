#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
智能查询缓存系统

提供多层缓存、预聚合数据和查询优化功能
"""

import time
import threading
import hashlib
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict, Ordered_dict

from utils.logger import getLogger

logger = getLogger(__name__)


class QueryCache:
    """
    智能查询缓存系统
    
    特性：
    - LRU缓存策略
    - 分层缓存（内存 + 磁盘）
    - 查询模式识别
    - 预聚合数据缓存
    """
    
    def __init___22(self, 
                 max_memory_size: int = 1000,
                 max_disk_size: int = 5000,
                 default_ttl: int = 1800,
                 enable_disk_cache: bool = True,
                 cache_dir: str = "cache"):
        """
        初始化查询缓存
        
        Args:
            max_memory_size: 内存缓存最大条目数
            max_disk_size: 磁盘缓存最大条目数
            default_ttl: 默认缓存有效期（秒）
            enable_disk_cache: 是否启用磁盘缓存
            cache_dir: 缓存目录
        """
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size
        self.default_ttl = default_ttl
        self.enable_disk_cache = enable_disk_cache
        self.cache_dir = cache_dir
        
        # 内存缓存（LRU）
        self.memory_cache = Ordered_dict()
        self.memory_timestamps = {}
        self.memory_access_count = defaultdict(int)
        
        # 磁盘缓存索引
        self.disk_cache_index = {}
        self.disk_timestamps = {}
        
        # 预聚合缓存
        self.aggregated_cache = {}
        self.aggregated_timestamps = {}
        
        # 查询模式统计
        self.query_patterns = defaultdict(int)
        self.popular_queries = Ordered_dict()
        
        # 线程安全锁
        self.memory_lock = threading.RLock()
        self.disk_lock = threading.RLock()
        self.pattern_lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'disk_hits': 0,
            'disk_misses': 0,
            'aggregated_hits': 0,
            'total_queries': 0,
            'cache_evictions': 0,
            'pattern_matches': 0
        }
        
        # 创建缓存目录
        if self.enable_disk_cache:
            import os
            os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"查询缓存系统初始化完成，内存缓存: {max_memory_size}, "
                   f"磁盘缓存: {'启用' if enable_disk_cache else '禁用'}")
    
    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
            ttl: 缓存有效期，None使用默认值
            
        Returns:
            缓存的数据，不存在或过期返回None
        """
        ttl = ttl or self.default_ttl
        self.stats['total_queries'] += 1
        
        # 1. 检查内存缓存
        result = self._get_from_memory(key, ttl)
        if result is not None:
            self.stats['memory_hits'] += 1
            self._update_query_pattern(key)
            return result
        
        self.stats['memory_misses'] += 1
        
        # 2. 检查预聚合缓存
        aggregated_result = self._get_from_aggregated_cache(key, ttl)
        if aggregated_result is not None:
            self.stats['aggregated_hits'] += 1
            # 将预聚合结果放入内存缓存
            self._set_memory_cache(key, aggregated_result, ttl)
            return aggregated_result
        
        # 3. 检查磁盘缓存
        if self.enable_disk_cache:
            disk_result = self._get_from_disk(key, ttl)
            if disk_result is not None:
                self.stats['disk_hits'] += 1
                # 将磁盘结果放入内存缓存
                self._set_memory_cache(key, disk_result, ttl)
                return disk_result
            
            self.stats['disk_misses'] += 1
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 缓存有效期
        """
        ttl = ttl or self.default_ttl
        
        # 设置内存缓存
        self._set_memory_cache(key, value, ttl)
        
        # 设置磁盘缓存（异步）
        if self.enable_disk_cache:
            threading.Thread(
                target=self._set_disk_cache,
                args=(key, value, ttl),
                daemon=True
            ).start()
        
        # 检查是否需要预聚合
        self._check_for_aggregation(key, value)
    
    def _get_from_memory(self, key: str, ttl: int) -> Optional[Any]:
        """从内存缓存获取数据"""
        with self.memory_lock:
            if key not in self.memory_cache:
                return None
            
            # 检查是否过期
            if time.time() - self.memory_timestamps.get(key, 0) > ttl:
                self._remove_from_memory(key)
                return None
            
            # 更新LRU顺序
            value = self.memory_cache[key]
            del self.memory_cache[key]
            self.memory_cache[key] = value
            
            # 更新访问统计
            self.memory_access_count[key] += 1
            
            return value
    
    def _set_memory_cache(self, key: str, value: Any, ttl: int):
        """设置内存缓存"""
        with self.memory_lock:
            # 检查缓存大小
            if len(self.memory_cache) >= self.max_memory_size and key not in self.memory_cache:
                self._evict_memory_cache()
            
            # 设置缓存
            self.memory_cache[key] = value
            self.memory_timestamps[key] = time.time()
            self.memory_access_count[key] = 1
    
    def _evict_memory_cache(self):
        """驱逐内存缓存项（LRU策略）"""
        if not self.memory_cache:
            return
        
        # 移除最久未使用的项
        oldest_key = next(iter(self.memory_cache))
        self._remove_from_memory(oldest_key)
        self.stats['cache_evictions'] += 1
    
    def _remove_from_memory(self, key: str):
        """从内存缓存移除项目"""
        self.memory_cache.pop(key, None)
        self.memory_timestamps.pop(key, None)
        self.memory_access_count.pop(key, None)
    
    def _get_from_disk(self, key: str, ttl: int) -> Optional[Any]:
        """从磁盘缓存获取数据"""
        with self.disk_lock:
            if key not in self.disk_cache_index:
                return None
            
            # 检查是否过期
            if time.time() - self.disk_timestamps.get(key, 0) > ttl:
                self._remove_from_disk(key)
                return None
            
            try:
                file_path = self.disk_cache_index[key]
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"读取磁盘缓存失败: {key}, 错误: {e}")
                self._remove_from_disk(key)
                return None
    
    def _set_disk_cache(self, key: str, value: Any, ttl: int):
        """设置磁盘缓存"""
        try:
            with self.disk_lock:
                # 检查缓存大小
                if len(self.disk_cache_index) >= self.max_disk_size and key not in self.disk_cache_index:
                    self._evict_disk_cache()
                
                # 生成文件路径
                import os
                file_name = f"{hashlib.md5(key.encode()).hexdigest()}.cache"
                file_path = os.path.join(self.cache_dir, file_name)
                
                # 保存到磁盘
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # 更新索引
                self.disk_cache_index[key] = file_path
                self.disk_timestamps[key] = time.time()
                
        except Exception as e:
            logger.warning(f"设置磁盘缓存失败: {key}, 错误: {e}")
    
    def _evict_disk_cache(self):
        """驱逐磁盘缓存项"""
        if not self.disk_cache_index:
            return
        
        # 找到最旧的项
        oldest_key = min(self.disk_timestamps.items(), key=lambda x: x[1])[0]
        self._remove_from_disk(oldest_key)
    
    def _remove_from_disk(self, key: str):
        """从磁盘缓存移除项目"""
        if key in self.disk_cache_index:
            try:
                import os
                file_path = self.disk_cache_index[key]
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"删除磁盘缓存文件失败: {e}")
            
            self.disk_cache_index.pop(key, None)
            self.disk_timestamps.pop(key, None)
    
    def _get_from_aggregated_cache(self, key: str, ttl: int) -> Optional[Any]:
        """从预聚合缓存获取数据"""
        # 检查是否有匹配的预聚合数据
        for agg_key, agg_value in self.aggregated_cache.items():
            if self._is_aggregated_match(key, agg_key):
                # 检查是否过期
                if time.time() - self.aggregated_timestamps.get(agg_key, 0) <= ttl:
                    return self._extract_from_aggregated(key, agg_value)
        
        return None
    
    def _check_for_aggregation(self, key: str, value: Any):
        """检查是否需要创建预聚合数据"""
        # 简单的预聚合策略：对于大数据集创建聚合
        if isinstance(value, pd.DataFrame) and len(value) > 1000:
            # 创建按股票代码的聚合
            if 'code' in value.columns:
                try:
                    aggregated = value.groupby('code').agg({
                        'close': ['mean', 'max', 'min', 'last'],
                        'volume': ['mean', 'sum'],
                        'date': ['min', 'max']
                    }).reset_index()
                    
                    agg_key = f"agg_by_code_{hashlib.md5(key.encode()).hexdigest()[:8]}"
                    self.aggregated_cache[agg_key] = aggregated
                    self.aggregated_timestamps[agg_key] = time.time()
                    
                except Exception as e:
                    logger.debug(f"创建预聚合数据失败: {e}")
    
    def _is_aggregated_match(self, query_key: str, agg_key: str) -> bool:
        """检查查询是否匹配预聚合数据"""
        # 简单的匹配策略
        return 'agg_by_code' in agg_key and 'code' in query_key
    
    def _extract_from_aggregated(self, query_key: str, agg_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """从预聚合数据中提取查询结果"""
        # 简单的提取策略
        try:
            # 如果查询包含特定股票代码，返回该股票的聚合数据
            if 'stock_code' in query_key:
                # 这里需要更复杂的逻辑来解析查询键并提取相关数据
                return agg_data.head(10)  # 简化实现
        except Exception as e:
            logger.debug(f"从预聚合数据提取失败: {e}")
        
        return None
    
    def _update_query_pattern(self, key: str):
        """更新查询模式统计"""
        with self.pattern_lock:
            # 提取查询模式
            pattern = self._extract_query_pattern(key)
            self.query_patterns[pattern] += 1
            
            # 更新热门查询
            if key in self.popular_queries:
                del self.popular_queries[key]
            self.popular_queries[key] = time.time()
            
            # 保持热门查询列表大小
            if len(self.popular_queries) > 100:
                oldest_key = next(iter(self.popular_queries))
                del self.popular_queries[oldest_key]
    
    def _extract_query_pattern(self, key: str) -> str:
        """提取查询模式"""
        # 简化的模式提取
        if 'stock_code' in key:
            return 'single_stock_query'
        elif 'level' in key:
            return 'level_based_query'
        elif 'date' in key:
            return 'date_range_query'
        else:
            return 'general_query'
    
    def clear_Cache_Query_Cache(self, pattern: Optional[str] = None):
        """清除缓存"""
        with self.memory_lock:
            if pattern is None:
                self.memory_cache.clear_Cache_Query_Cache()
                self.memory_timestamps.clear_Cache_Query_Cache()
                self.memory_access_count.clear_Cache_Query_Cache()
            else:
                keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    self._remove_from_memory(key)
        
        if self.enable_disk_cache:
            with self.disk_lock:
                if pattern is None:
                    for key in list(self.disk_cache_index.keys()):
                        self._remove_from_disk(key)
                else:
                    keys_to_remove = [k for k in self.disk_cache_index.keys() if pattern in k]
                    for key in keys_to_remove:
                        self._remove_from_disk(key)
        
        logger.info(f"缓存已清除，模式: {pattern or '全部'}")
    
    def get_stats_Cache_Query_Cache(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.memory_lock, self.disk_lock, self.pattern_lock:
            total_hits = self.stats['memory_hits'] + self.stats['disk_hits'] + self.stats['aggregated_hits']
            total_misses = self.stats['memory_misses'] + self.stats['disk_misses']
            
            return {
                'memory_cache_size': len(self.memory_cache),
                'disk_cache_size': len(self.disk_cache_index),
                'aggregated_cache_size': len(self.aggregated_cache),
                'total_queries': self.stats['total_queries'],
                'cache_hit_rate': total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0,
                'memory_hit_rate': self.stats['memory_hits'] / self.stats['total_queries'] if self.stats['total_queries'] > 0 else 0,
                'disk_hit_rate': self.stats['disk_hits'] / self.stats['total_queries'] if self.stats['total_queries'] > 0 else 0,
                'cache_evictions': self.stats['cache_evictions'],
                'top_query_patterns': dict(list(self.query_patterns.items())[:5])
            }


# 全局缓存实例
_query_cache = None
_cache_lock = threading.Lock()


def get_query_cache() -> Query_cache:
    """获取全局查询缓存实例"""
    global _query_cache
    
    if _query_cache is None:
        with _cache_lock:
            if _query_cache is None:
                _query_cache = Query_cache()
    
    return _query_cache
