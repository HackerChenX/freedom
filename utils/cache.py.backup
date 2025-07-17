"""
缓存管理模块

提供内存缓存、磁盘缓存和LRU缓存的实现
支持依赖注入和向后兼容的单例模式
"""

import os
import time
import threading
import pickle
import logging
from typing import Dict, Any, Tuple, Optional
from functools import wraps, lru_cache

from utils.dependency_injection import get_service


class MemoryCache:
    """
    内存缓存类，提供线程安全的内存缓存功能
    重构为普通类，支持依赖注入
    """
    
    def __init__(self):
        """初始化缓存"""
        self._cache: Dict[str, Tuple[Any, float, Optional[float]]] = {}  # (value, timestamp, ttl)
        self._lock = threading.Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            缓存值或默认值
        """
        with self._lock:
            if key in self._cache:
                value, timestamp, ttl = self._cache[key]
                # 检查是否过期
                if ttl is not None and time.time() - timestamp > ttl:
                    del self._cache[key]
                    return default
                return value
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None表示永不过期
        """
        with self._lock:
            self._cache[key] = (value, time.time(), ttl)
    
    def delete(self, key: str) -> bool:
        """
        删除缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否删除成功
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在且未过期
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否存在且未过期
        """
        with self._lock:
            if key in self._cache:
                _, timestamp, ttl = self._cache[key]
                # 检查是否过期
                if ttl is not None and time.time() - timestamp > ttl:
                    del self._cache[key]
                    return False
                return True
            return False
    
    def clear_cache(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def get_stats_cache(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict: 包含缓存统计信息的字典
        """
        with self._lock:
            total_items = len(self._cache)
            expired_items = 0
            for key in list(self._cache.keys()):
                _, timestamp, ttl = self._cache[key]
                if ttl is not None and time.time() - timestamp > ttl:
                    expired_items += 1
            
            return {
                'total_items': total_items,
                'active_items': total_items - expired_items,
                'expired_items': expired_items
            }


class DiskCache:
    """
    磁盘缓存类，提供持久化缓存功能
    """
    
    def __init__(self, base_dir: str = "cache"):
        """
        初始化磁盘缓存
        
        Args:
            base_dir: 缓存基础目录
        """
        self._base_dir = base_dir
        self._index_file = os.path.join(base_dir, "index.pkl")
        self._index: Dict[str, Tuple[str, float, Optional[float]]] = {}  # (file_path, timestamp, ttl)
        self._lock = threading.Lock()
        
        # 创建缓存目录
        os.makedirs(base_dir, exist_ok=True)
        
        # 加载索引
        self._load_index()
    
    def _load_index(self) -> None:
        """加载索引文件"""
        try:
            if os.path.exists(self._index_file):
                with open(self._index_file, 'rb') as f:
                    self._index = pickle.load(f)
        except Exception as e:
            logging.error(f"加载缓存索引失败: {e}")
            self._index = {}
    
    def _save_index(self) -> None:
        """保存索引文件"""
        try:
            with open(self._index_file, 'wb') as f:
                pickle.dump(self._index, f)
        except Exception as e:
            logging.error(f"保存缓存索引失败: {e}")
    
    def _get_file_path(self, key: str) -> str:
        """
        获取缓存文件路径
        
        Args:
            key: 缓存键
            
        Returns:
            str: 缓存文件路径
        """
        # 使用键的哈希值作为文件名，防止文件名过长或包含非法字符
        filename = f"{hash(key)}.cache"
        return os.path.join(self._base_dir, filename)
    
    def cleanup(self) -> int:
        """
        清理过期的缓存
        
        Returns:
            int: 清理的缓存项数量
        """
        with self._lock:
            count = 0
            for key in list(self._index.keys()):
                _, timestamp, ttl = self._index[key]
                if ttl is not None and time.time() - timestamp > ttl:
                    self.delete(key)
                    count += 1
            return count


class LRUCacheCache:
    """
    LRU (Least Recently Used) 缓存实现
    """
    
    def __init__(self, max_size: int = 128):
        """
        初始化LRU缓存
        
        Args:
            max_size: 最大缓存大小
        """
        self.max_size = max_size
        self._cache = {}
        self._order = []  # 访问顺序，最新访问的在末尾
        self._lock = threading.Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值并更新访问顺序"""
        with self._lock:
            if key in self._cache:
                # 更新访问顺序
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key]
            return default
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            if key in self._cache:
                # 更新现有值
                self._order.remove(key)
                self._order.append(key)
                self._cache[key] = value
            else:
                # 添加新值
                if len(self._cache) >= self.max_size:
                    # 移除最少使用的项
                    oldest_key = self._order.pop(0)
                    del self._cache[oldest_key]
                
                self._cache[key] = value
                self._order.append(key)


def cache_result(ttl: Optional[float] = None, 
                disk_cache: bool = False,
                key_prefix: str = ""):
    """
    缓存函数结果的装饰器
    
    Args:
        ttl: 过期时间（秒），None表示永不过期
        disk_cache: 是否使用磁盘缓存
        key_prefix: 缓存键前缀
        
    Returns:
        装饰器函数
    """
    def decorator_cache(func):
        @wraps(func)
        def wrapper_cache(*args, **kwargs):
            # 生成缓存键
            key = f"{key_prefix}{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 选择缓存类型
            if disk_cache:
                cache = get_disk_cache()
            else:
                cache = get_memory_cache()
            
            # 尝试从缓存获取
            result = cache.get(key)
            if result is not None:
                return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            
            return result
        return wrapper
    return decorator


# 向后兼容的单例接口
_legacy_memory_cache = None
_legacy_disk_cache = None
_legacy_lock = threading.Lock()


def get_memory_cache_cache() -> MemoryCache:
    """获取内存缓存实例（向后兼容）"""
    global _legacy_memory_cache
    if _legacy_memory_cache is None:
        with _legacy_lock:
            if _legacy_memory_cache is None:
                try:
                    # 尝试从容器获取实例
                    from utils.dependency_injection import get_container
                    container = get_container()
                    if container.is_registered(MemoryCache):
                        _legacy_memory_cache = container.resolve(MemoryCache)
                    else:
                        # 如果未注册，创建默认实例
                        _legacy_memory_cache = MemoryCache()
                except Exception:
                    # 如果容器不可用，创建默认实例
                    _legacy_memory_cache = MemoryCache()
    return _legacy_memory_cache


def get_disk_cache(base_dir: str = "cache") -> DiskCache:
    """获取磁盘缓存实例（向后兼容）"""
    global _legacy_disk_cache
    if _legacy_disk_cache is None:
        with _legacy_lock:
            if _legacy_disk_cache is None:
                _legacy_disk_cache = DiskCache(base_dir)
    return _legacy_disk_cache


# 现代化的依赖注入接口
def get_cache_service() -> MemoryCache:
    """通过依赖注入获取缓存服务"""
    return get_service(MemoryCache)


# 兼容性别名
def get_instance():
    """向后兼容的获取实例方法"""
    return get_memory_cache()


# 缓存实例别名（向后兼容）
cache = get_memory_cache
