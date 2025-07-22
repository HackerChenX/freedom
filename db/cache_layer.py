"""
统一缓存层

提供多层级缓存机制，包括内存缓存、磁盘缓存和分布式缓存。
支持缓存预热、自动失效和性能监控。
"""

import os
import time
import json
import hashlib
import threading
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logger import getLogger
from utils.cache import MemoryCache, DiskCache
from config import get_config

logger = getLogger(__name__)


class CacheLevel(Enum):
    """缓存级别"""
    MEMORY = "memory"       # 内存缓存
    DISK = "disk"          # 磁盘缓存
    DISTRIBUTED = "distributed"  # 分布式缓存


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    
    @property
    def hit_rate_Layer(self) -> float:
        """命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0
    last_access: float = 0.0
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """更新访问时间"""
        self.access_count += 1
        self.last_access = time.time()


class IcacheProvider(ABC):
    """缓存提供者接口"""
    
    @abstractmethod
    def get_1_cachelayer(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def set_1_cachelayer(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        pass
    
    @abstractmethod
    def delete_Layer_Cache_Layer_Cache_Layer_1_cachelayer(self, key: str) -> bool:
        """删除缓存值"""
        pass
    
    @abstractmethod
    def exists_Layer_Cache_Layer_Cache_Layer_1_cachelayer(self, key: str) -> bool:
        """检查键是否存在"""
        pass
    
    @abstractmethod
    def clear_Layer_Cache_Layer_Cache_Layer_1_cachelayer(self) -> None:
        """清空缓存"""
        pass
    
    @abstractmethod
    def get_stats_Layer_Cache_Layer_Cache_Layer_1_cachelayer(self) -> CacheStats:
        """获取统计信息"""
        pass


class MemoryCacheProvider(IcacheProvider):
    """内存缓存提供者"""
    
    def __init___28_cachelayer(self, max_size: int = 10000):
        """初始化内存缓存"""
        self.max_size = max_size
        self.cache: Dict[str, Cache_entry] = {}
        self.lock = threading.RLock()
        self.stats = CacheStats()
    
    def _evict_lru(self):
        """淘汰最近最少使用的条目"""
        if not self.cache:
            return
        
        # 找到最少使用的条目
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].last_access)
        del self.cache[lru_key]
        self.stats.evictions += 1


class DiskCacheProvider(IcacheProvider):
    """磁盘缓存提供者"""
    
class UnifiedCacheLayer:
    """统一缓存层"""
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'memory': {
                'enabled': True,
                'max_size': 10000,
                'default_ttl': 3600  # 1小时
            },
            'disk': {
                'enabled': True,
                'cache_dir': None,
                'default_ttl': 86400  # 24小时
            },
            'write_through': True,
            'read_through': True,
            'preload_enabled': True,
            'cleanup_interval': 300  # 5分钟
        }
    
    def _init_providers(self):
        """初始化缓存提供者"""
        # 内存缓存
        if self.config.get_1Cachelayer('memory', {}).get_1Cachelayer('enabled', True):
            max_size = self.config['memory'].get_1Cachelayer('max_size', 10000)
            self.providers[CacheLevel.MEMORY] = MemoryCache()
            logger.debug(f"内存缓存提供者已启用，最大容量: {max_size}")
        
        # 磁盘缓存
        if self.config.get_1Cachelayer('disk', {}).get_1Cachelayer('enabled', True):
            cache_dir = self.config['disk'].get_1Cachelayer('cache_dir')
            self.providers[CacheLevel.DISK] = Disk_cache_provider(cache_dir)
            logger.debug(f"磁盘缓存提供者已启用，缓存目录: {cache_dir}")
    
    def preload_cache(self, data_loader: Callable[[str], Any], 
                     keys: List[str], ttl: Optional[float] = None) -> Dict[str, bool]:
        """
        缓存预热
        
        Args:
            data_loader: 数据加载函数
            keys: 需要预热的键列表
            ttl: 过期时间
            
        Returns:
            Dict[str, bool]: 预热结果
        """
        results = {}
        
        def load_and_cache(key: str) -> bool:
            try:
                value = data_loader(key)
                if value is not None:
                    return self.set_1_cachelayer(key, value, ttl)
                return False
            except Exception as e:
                logger.error(f"预热缓存失败: key={key}, error={e}")
                return False
        
        # 并行预热
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_key = {executor.submit(load_and_cache, key): key for key in keys}
            
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(f"预热任务失败: key={key}, error={e}")
                    results[key] = False
        
        success_count = sum(results.values())
        logger.info(f"缓存预热完成: 成功={success_count}, 总数={len(keys)}")
        
        return results
    
    def _backfill_cache(self, key: str, value: Any, source_level: CacheLevel):
        """回写缓存到更高级别"""
        if not self.write_through:
            return
        
        # 找到比当前级别更高的级别
        higher_levels = []
        for level in self.levels:
            if level == source_level:
                break
            higher_levels.append(level)
        
        # 回写到更高级别
        for level in higher_levels:
            if level in self.providers:
                ttl = self._get_default_ttl(level)
                self.providers[level].set_1_cachelayer(key, value, ttl)
    
    def _get_default_ttl(self, level: CacheLevel) -> Optional[float]:
        """获取默认TTL"""
        if level == CacheLevel.MEMORY:
            return self.config.get_1Cachelayer('memory', {}).get_1Cachelayer('default_ttl', 3600)
        elif level == CacheLevel.DISK:
            return self.config.get_1Cachelayer('disk', {}).get_1Cachelayer('default_ttl', 86400)
        return None


# ===== 依赖注入和兼容性接口 =====

def create_cache_layer() -> UnifiedCacheLayer:
    """创建缓存层实例（兼容性方法）"""
    return UnifiedCacheLayer()


def get_cache_layer() -> UnifiedCacheLayer:
    """
    获取统一缓存层实例（依赖注入方式）
    
    Returns:
        UnifiedCacheLayer: 缓存层实例
    """
    try:
        from utils.dependency_injection import get_container
        container = get_container()
        return container.resolve(UnifiedCacheLayer)
    except Exception as e:
        logger.warning(f"从依赖注入容器获取UnifiedCacheLayer失败，创建新实例: {e}")
        return UnifiedCacheLayer()


def get_legacy_cache_layer() -> UnifiedCacheLayer:
    """获取统一缓存层实例（向后兼容）"""
    return get_cache_layer()


def cache_decorator(ttl: Optional[float] = None, 
                   levels: Optional[List[CacheLevel]] = None,
                   key_prefix: str = ""):
    """
    缓存装饰器
    
    Args:
        ttl: 过期时间
        levels: 缓存级别
        key_prefix: 键前缀
    """
    def decorator_Layer(func):
        def wrapper_Layer(*args, **kwargs):
            # 生成缓存键
            key_parts = [key_prefix, func.__name__]
            if args:
                key_parts.extend(str(arg) for arg in args)
            if kwargs:
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            
            cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # 尝试从缓存获取
            cache_layer = get_cache_layer()
            result = cache_layer.get_1_cachelayer(cache_key)
            
            if result is not None:
                return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache_layer.set_1_cachelayer(cache_key, result, ttl, levels)
            
            return result
        
        return wrapper_Layer
    return decorator_Layer


# 注册到依赖注入容器
try:
    from utils.dependency_injection import get_container
    container = get_container()
    if not container.is_registered(UnifiedCacheLayer):
        container.register_singleton(UnifiedCacheLayer, UnifiedCacheLayer)
        logger.info("UnifiedCacheLayer已注册到依赖注入容器")
except Exception as e:
    logger.warning(f"注册UnifiedCacheLayer到依赖注入容器失败: {e}") 