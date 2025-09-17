#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股测试缓存管理器

提供多级缓存系统，优化数据访问和计算性能
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
import os
import pickle

from utils.logger import getLogger

logger = getLogger(__name__)

# 泛型类型定义
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


@dataclass
class CacheEntry(Generic[V]):
    """缓存条目"""
    value: V
    expiry: float  # 过期时间戳
    access_count: int = 0  # 访问次数
    last_access: float = field(default_factory=time.time)  # 最后访问时间


class LRUCache(Generic[K, V]):
    """LRU缓存实现"""
    
    def __init__(self, capacity: int = 1000, ttl: int = 3600):
        """
        初始化LRU缓存
        
        Args:
            capacity: 缓存容量
            ttl: 生存时间（秒）
        """
        self.capacity = capacity
        self.ttl = ttl
        self.cache: Dict[K, CacheEntry[V]] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: K) -> Optional[V]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[V]: 缓存值，不存在则返回None
        """
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            current_time = time.time()
            
            # 检查是否过期
            if current_time > entry.expiry:
                del self.cache[key]
                self.misses += 1
                return None
            
            # 更新访问信息
            entry.access_count += 1
            entry.last_access = current_time
            self.hits += 1
            
            return entry.value
    
    def put(self, key: K, value: V) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        with self.lock:
            # 如果已存在，更新值和过期时间
            if key in self.cache:
                entry = self.cache[key]
                entry.value = value
                entry.expiry = time.time() + self.ttl
                entry.access_count += 1
                entry.last_access = time.time()
                return
            
            # 如果缓存已满，移除最久未使用的条目
            if len(self.cache) >= self.capacity:
                self._evict()
            
            # 添加新条目
            self.cache[key] = CacheEntry(
                value=value,
                expiry=time.time() + self.ttl
            )
    
    def _evict(self) -> None:
        """移除最久未使用的缓存条目"""
        if not self.cache:
            return
        
        # 找到最久未使用的条目
        oldest_key = min(self.cache.items(), key=lambda x: x[1].last_access)[0]
        del self.cache[oldest_key]
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'capacity': self.capacity,
                'size': len(self.cache),
                'ttl': self.ttl,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'memory_usage_estimate': len(self.cache) * 200  # 粗略估计，每个条目约200字节
            }

class DiskCache(Generic[K, V]):
    """磁盘缓存实现"""
    
    def __init__(self, cache_dir: str, ttl: int = 86400):
        """
        初始化磁盘缓存
        
        Args:
            cache_dir: 缓存目录
            ttl: 生存时间（秒）
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: K) -> str:
        """
        获取缓存文件路径
        
        Args:
            key: 缓存键
            
        Returns:
            str: 缓存文件路径
        """
        # 将键转换为字符串并计算哈希值
        key_str = str(key)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        # 使用哈希值作为文件名
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def get(self, key: K) -> Optional[V]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[V]: 缓存值，不存在则返回None
        """
        with self.lock:
            cache_path = self._get_cache_path(key)
            
            if not os.path.exists(cache_path):
                self.misses += 1
                return None
            
            # 检查文件是否过期
            file_mtime = os.path.getmtime(cache_path)
            if time.time() - file_mtime > self.ttl:
                os.remove(cache_path)
                self.misses += 1
                return None
            
            try:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
                
                self.hits += 1
                return value
                
            except Exception as e:
                logger.error(f"读取缓存文件失败: {e}")
                self.misses += 1
                return None
    
    def put(self, key: K, value: V) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        with self.lock:
            cache_path = self._get_cache_path(key)
            
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                    
            except Exception as e:
                logger.error(f"写入缓存文件失败: {e}")
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    os.remove(os.path.join(self.cache_dir, filename))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            # 计算缓存大小
            cache_size = 0
            file_count = 0
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    file_path = os.path.join(self.cache_dir, filename)
                    cache_size += os.path.getsize(file_path)
                    file_count += 1
            
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'cache_dir': self.cache_dir,
                'file_count': file_count,
                'cache_size_mb': cache_size / (1024 * 1024),
                'ttl': self.ttl,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }

class CacheService:
    """多级缓存管理器"""
    
    def __init__(self, memory_capacity: int = 1000, memory_ttl: int = 3600, 
               disk_cache_dir: str = "cache", disk_ttl: int = 86400):
        """
        初始化缓存管理器
        
        Args:
            memory_capacity: 内存缓存容量
            memory_ttl: 内存缓存生存时间（秒）
            disk_cache_dir: 磁盘缓存目录
            disk_ttl: 磁盘缓存生存时间（秒）
        """
        self.memory_cache = LRUCache(capacity=memory_capacity, ttl=memory_ttl)
        self.disk_cache = DiskCache(cache_dir=disk_cache_dir, ttl=disk_ttl)
        
        # 缓存命名空间
        self.namespaces = {
            'stock_data': {},
            'indicator_values': {},
            'pattern_results': {},
            'verification_results': {}
        }
        
        logger.info(f"缓存管理器初始化完成，内存容量: {memory_capacity}，磁盘缓存: {disk_cache_dir}")
    
    def get(self, namespace: str, key: Any) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            namespace: 缓存命名空间
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存值，不存在则返回None
        """
        # 生成完整键
        full_key = f"{namespace}:{self._serialize_key(key)}"
        
        # 首先尝试从内存缓存获取
        value = self.memory_cache.get(full_key)
        if value is not None:
            return value
        
        # 然后尝试从磁盘缓存获取
        value = self.disk_cache.get(full_key)
        if value is not None:
            # 将值放入内存缓存
            self.memory_cache.put(full_key, value)
            return value
        
        return None
    
    def put(self, namespace: str, key: Any, value: Any, persist: bool = False) -> None:
        """
        设置缓存值
        
        Args:
            namespace: 缓存命名空间
            key: 缓存键
            value: 缓存值
            persist: 是否持久化到磁盘
        """
        # 生成完整键
        full_key = f"{namespace}:{self._serialize_key(key)}"
        
        # 放入内存缓存
        self.memory_cache.put(full_key, value)
        
        # 如果需要持久化，放入磁盘缓存
        if persist:
            self.disk_cache.put(full_key, value)
    
    def _serialize_key(self, key: Any) -> str:
        """
        序列化缓存键
        
        Args:
            key: 缓存键
            
        Returns:
            str: 序列化后的键
        """
        if isinstance(key, str):
            return key
        
        if isinstance(key, (int, float, bool)):
            return str(key)
        
        if isinstance(key, (list, tuple, dict)):
            # 使用JSON序列化复杂对象
            try:
                return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
            except:
                pass
        
        # 默认使用对象的字符串表示
        return hashlib.md5(str(key).encode()).hexdigest()
    
    def clear(self, namespace: Optional[str] = None) -> None:
        """
        清空缓存
        
        Args:
            namespace: 要清空的命名空间，None表示清空所有
        """
        if namespace is None:
            # 清空所有缓存
            self.memory_cache.clear()
            self.disk_cache.clear()
            return
        
        # 清空指定命名空间的缓存
        prefix = f"{namespace}:"
        
        # 清空内存缓存
        with self.memory_cache.lock:
            keys_to_remove = [k for k in self.memory_cache.cache.keys() if k.startswith(prefix)]
            for key in keys_to_remove:
                del self.memory_cache.cache[key]
        
        # 清空磁盘缓存（这里简化处理，直接清空所有）
        # 实际应用中可以优化为只清除特定命名空间的文件
        self.disk_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        memory_stats = self.memory_cache.get_stats()
        disk_stats = self.disk_cache.get_stats()
        
        # 计算总体命中率
        total_hits = memory_stats['hits'] + disk_stats['hits']
        total_misses = memory_stats['misses'] + disk_stats['misses']
        total_requests = total_hits + total_misses
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            'memory_cache': memory_stats,
            'disk_cache': disk_stats,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'overall_hit_rate': overall_hit_rate
        }


# 全局缓存管理器实例
_cache_manager = None


def get_cache_manager() -> CacheService:
    """
    获取全局缓存管理器实例
    
    Returns:
        CacheService: 缓存管理器实例
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheService()
    return _cache_manager


def main():
    """测试缓存管理器"""
    print("测试缓存管理器...")
    
    # 创建缓存管理器
    cache_manager = CacheService(memory_capacity=100, disk_cache_dir="test_cache")
    
    # 测试缓存操作
    print("\n测试内存缓存...")
    cache_manager.put("stock_data", "000001", {"name": "平安银行", "price": 10.5})
    result = cache_manager.get("stock_data", "000001")
    print(f"缓存结果: {result}")
    
    print("\n测试磁盘缓存...")
    cache_manager.put("indicator_values", "MACD_000001", {"dif": 0.5, "dea": 0.3}, persist=True)
    result = cache_manager.get("indicator_values", "MACD_000001")
    print(f"缓存结果: {result}")
    
    # 清空内存缓存，测试从磁盘加载
    cache_manager.memory_cache.clear()
    result = cache_manager.get("indicator_values", "MACD_000001")
    print(f"从磁盘加载: {result}")
    
    # 获取统计信息
    stats = cache_manager.get_stats()
    print("\n缓存统计:")
    print(f"内存缓存: {stats['memory_cache']['size']}/{stats['memory_cache']['capacity']}")
    print(f"磁盘缓存: {stats['disk_cache']['file_count']} 文件, {stats['disk_cache']['cache_size_mb']:.2f}MB")
    print(f"总命中率: {stats['overall_hit_rate']:.2%}")
    
    # 清理测试缓存
    cache_manager.clear()
    print("\n缓存已清空")


if __name__ == "__main__":
    main()