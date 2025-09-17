"""
多层缓存架构
提供内存缓存、持久化缓存和分布式缓存的统一接口
"""

import time
import pickle
import threading
import hashlib
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from pathlib import Path
import logging

from utils.enhanced_performance_monitor import performance_monitor
from utils.enhanced_exception_handler import exception_handler, ErrorSeverity, ErrorCategory
from config.unified_config_manager import get_config

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    errors: int = 0
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def record_hit(self):
        """记录命中"""
        self.hits += 1
    
    def record_miss(self):
        """记录未命中"""
        self.misses += 1
    
    def record_set(self):
        """记录设置"""
        self.sets += 1
    
    def record_delete(self):
        """记录删除"""
        self.deletes += 1
    
    def record_eviction(self):
        """记录驱逐"""
        self.evictions += 1
    
    def record_error(self):
        """记录错误"""
        self.errors += 1


class CacheInterface(ABC):
    """缓存接口"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """清空缓存"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """获取统计信息"""
        pass


@dataclass
class MemoryCacheEntry:
    """内存缓存条目"""
    value: Any
    created_time: float
    ttl: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl <= 0:
            return False
        return time.time() - self.created_time > self.ttl
    
    def access(self):
        """记录访问"""
        self.access_count += 1
        self.last_accessed = time.time()


class MemoryCache(CacheInterface):
    """内存缓存实现"""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        """
        初始化内存缓存
        
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认TTL（秒）
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, MemoryCacheEntry] = {}
        self.lock = threading.RLock()
        self.stats = CacheStats()
        
        # 启动清理线程
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """启动清理线程"""
        def cleanup_task():
            while True:
                try:
                    time.sleep(60)  # 每分钟清理一次
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"内存缓存清理任务出错: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired(self):
        """清理过期条目"""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self.cache[key]
                self.stats.record_eviction()
            
            if expired_keys:
                logger.debug(f"清理了 {len(expired_keys)} 个过期缓存条目")
    
    def _evict_lru(self):
        """驱逐最少使用的条目"""
        if len(self.cache) < self.max_size:
            return
        
        # 按访问次数和最后访问时间排序
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: (x[1].access_count, x[1].last_accessed)
        )
        
        # 移除最少使用的25%
        evict_count = max(1, len(sorted_items) // 4)
        for key, _ in sorted_items[:evict_count]:
            del self.cache[key]
            self.stats.record_eviction()
    
    @performance_monitor(threshold_seconds=0.01)
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                if entry.is_expired():
                    del self.cache[key]
                    self.stats.record_miss()
                    return None
                
                entry.access()
                self.stats.record_hit()
                return entry.value
            
            self.stats.record_miss()
            return None
    
    @performance_monitor(threshold_seconds=0.01)
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        try:
            with self.lock:
                # 检查是否需要驱逐
                if len(self.cache) >= self.max_size:
                    self._evict_lru()
                
                entry = MemoryCacheEntry(
                    value=value,
                    created_time=time.time(),
                    ttl=ttl or self.default_ttl
                )
                
                self.cache[key] = entry
                self.stats.record_set()
                return True
                
        except Exception as e:
            logger.error(f"设置内存缓存失败: {e}")
            self.stats.record_error()
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.record_delete()
                return True
            return False
    
    def clear(self) -> bool:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            return True
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    del self.cache[key]
                    return False
                return True
            return False
    
    def get_stats(self) -> CacheStats:
        """获取统计信息"""
        return self.stats


class FileCache(CacheInterface):
    """文件缓存实现"""
    
    def __init__(self, cache_dir: str = "cache", default_ttl: float = 3600.0):
        """
        初始化文件缓存
        
        Args:
            cache_dir: 缓存目录
            default_ttl: 默认TTL（秒）
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stats = CacheStats()
        
        # 启动清理线程
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """启动清理线程"""
        def cleanup_task():
            while True:
                try:
                    time.sleep(300)  # 每5分钟清理一次
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"文件缓存清理任务出错: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired(self):
        """清理过期文件"""
        try:
            current_time = time.time()
            expired_count = 0
            
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    # 检查文件修改时间
                    file_time = cache_file.stat().st_mtime
                    if current_time - file_time > self.default_ttl:
                        cache_file.unlink()
                        expired_count += 1
                        self.stats.record_eviction()
                except Exception as e:
                    logger.debug(f"清理缓存文件失败 {cache_file}: {e}")
            
            if expired_count > 0:
                logger.debug(f"清理了 {expired_count} 个过期文件缓存")
                
        except Exception as e:
            logger.error(f"文件缓存清理失败: {e}")
    
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        # 使用哈希避免文件名问题
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    @performance_monitor(threshold_seconds=0.1)
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        cache_path = self._get_cache_path(key)
        
        try:
            if not cache_path.exists():
                self.stats.record_miss()
                return None
            
            # 检查是否过期
            file_time = cache_path.stat().st_mtime
            if time.time() - file_time > self.default_ttl:
                cache_path.unlink()
                self.stats.record_miss()
                return None
            
            # 读取缓存数据
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            self.stats.record_hit()
            return data
            
        except Exception as e:
            logger.debug(f"读取文件缓存失败 {key}: {e}")
            self.stats.record_error()
            return None
    
    @performance_monitor(threshold_seconds=0.1)
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        cache_path = self._get_cache_path(key)
        
        try:
            with self.lock:
                # 写入缓存数据
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                
                self.stats.record_set()
                return True
                
        except Exception as e:
            logger.error(f"写入文件缓存失败 {key}: {e}")
            self.stats.record_error()
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        cache_path = self._get_cache_path(key)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
                self.stats.record_delete()
                return True
            return False
        except Exception as e:
            logger.error(f"删除文件缓存失败 {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """清空缓存"""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            return True
        except Exception as e:
            logger.error(f"清空文件缓存失败: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return False
        
        # 检查是否过期
        file_time = cache_path.stat().st_mtime
        if time.time() - file_time > self.default_ttl:
            try:
                cache_path.unlink()
            except:
                pass
            return False
        
        return True
    
    def get_stats(self) -> CacheStats:
        """获取统计信息"""
        return self.stats


class MultiLayerCache:
    """
    多层缓存管理器
    
    缓存层级：
    1. L1: 内存缓存（最快）
    2. L2: 文件缓存（持久化）
    3. L3: 分布式缓存（可选，如Redis）
    """
    
    def __init__(self,
                 enable_memory_cache: bool = True,
                 enable_file_cache: bool = True,
                 enable_distributed_cache: bool = False,
                 memory_cache_size: int = 1000,
                 memory_cache_ttl: float = 300.0,
                 file_cache_ttl: float = 3600.0,
                 cache_dir: str = "cache"):
        """
        初始化多层缓存
        
        Args:
            enable_memory_cache: 是否启用内存缓存
            enable_file_cache: 是否启用文件缓存
            enable_distributed_cache: 是否启用分布式缓存
            memory_cache_size: 内存缓存大小
            memory_cache_ttl: 内存缓存TTL
            file_cache_ttl: 文件缓存TTL
            cache_dir: 缓存目录
        """
        self.layers: List[CacheInterface] = []
        
        # L1: 内存缓存
        if enable_memory_cache:
            self.memory_cache = MemoryCache(
                max_size=memory_cache_size,
                default_ttl=memory_cache_ttl
            )
            self.layers.append(self.memory_cache)
        else:
            self.memory_cache = None
        
        # L2: 文件缓存
        if enable_file_cache:
            self.file_cache = FileCache(
                cache_dir=cache_dir,
                default_ttl=file_cache_ttl
            )
            self.layers.append(self.file_cache)
        else:
            self.file_cache = None
        
        # L3: 分布式缓存（预留接口）
        if enable_distributed_cache:
            # TODO: 实现Redis或其他分布式缓存
            logger.warning("分布式缓存暂未实现")
        
        self.total_stats = CacheStats()
        
        logger.info(f"多层缓存初始化完成 - 层数: {len(self.layers)}")
    
    @performance_monitor(threshold_seconds=0.05)
    def get(self, key: str) -> Optional[Any]:
        """
        从多层缓存获取数据
        
        按层级顺序查找，找到后回填到上层缓存
        """
        for i, cache_layer in enumerate(self.layers):
            try:
                value = cache_layer.get(key)
                if value is not None:
                    # 回填到上层缓存
                    for j in range(i):
                        try:
                            self.layers[j].set(key, value)
                        except Exception as e:
                            logger.debug(f"回填缓存失败 L{j+1}: {e}")
                    
                    self.total_stats.record_hit()
                    return value
            except Exception as e:
                logger.debug(f"缓存层 L{i+1} 获取失败: {e}")
        
        self.total_stats.record_miss()
        return None
    
    @performance_monitor(threshold_seconds=0.05)
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置多层缓存数据"""
        success = False
        
        for i, cache_layer in enumerate(self.layers):
            try:
                if cache_layer.set(key, value, ttl):
                    success = True
            except Exception as e:
                logger.debug(f"缓存层 L{i+1} 设置失败: {e}")
        
        if success:
            self.total_stats.record_set()
        
        return success
    
    def delete(self, key: str) -> bool:
        """删除多层缓存数据"""
        success = False
        
        for cache_layer in self.layers:
            try:
                if cache_layer.delete(key):
                    success = True
            except Exception as e:
                logger.debug(f"缓存层删除失败: {e}")
        
        if success:
            self.total_stats.record_delete()
        
        return success
    
    def clear(self) -> bool:
        """清空所有缓存层"""
        success = True
        
        for cache_layer in self.layers:
            try:
                cache_layer.clear()
            except Exception as e:
                logger.error(f"清空缓存层失败: {e}")
                success = False
        
        return success
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        layer_stats = []
        
        for i, cache_layer in enumerate(self.layers):
            stats = cache_layer.get_stats()
            layer_stats.append({
                'layer': f'L{i+1}',
                'type': cache_layer.__class__.__name__,
                'hits': stats.hits,
                'misses': stats.misses,
                'hit_rate': stats.hit_rate,
                'sets': stats.sets,
                'deletes': stats.deletes,
                'evictions': stats.evictions,
                'errors': stats.errors
            })
        
        return {
            'total_stats': {
                'hits': self.total_stats.hits,
                'misses': self.total_stats.misses,
                'hit_rate': self.total_stats.hit_rate,
                'sets': self.total_stats.sets,
                'deletes': self.total_stats.deletes
            },
            'layer_stats': layer_stats,
            'layer_count': len(self.layers)
        }


# 全局多层缓存实例
_multi_cache = None
_cache_lock = threading.Lock()


def get_multi_cache() -> MultiLayerCache:
    """获取全局多层缓存实例"""
    global _multi_cache
    
    if _multi_cache is None:
        with _cache_lock:
            if _multi_cache is None:
                _multi_cache = MultiLayerCache()
    
    return _multi_cache


# 导出主要类和函数
__all__ = [
    'CacheInterface',
    'MemoryCache',
    'FileCache',
    'MultiLayerCache',
    'CacheStats',
    'get_multi_cache'
]
