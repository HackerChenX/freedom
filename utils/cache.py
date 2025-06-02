"""
缓存工具模块，提供内存缓存和持久化缓存功能
"""

import os
import pickle
import time
import threading
import logging
from typing import Dict, Any, Optional, Callable, Tuple, List, Union
from functools import wraps

from config import get_config
from utils.file_utils import ensure_dir


class MemoryCache:
    """
    内存缓存类，提供线程安全的内存缓存功能
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = MemoryCache()
        return cls._instance
    
    def __init__(self):
        """初始化缓存"""
        if MemoryCache._instance is not None:
            raise Exception("MemoryCache是单例类，请使用get_instance()方法获取实例")
        
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
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
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
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = DiskCache()
        return cls._instance
    
    def __init__(self):
        """初始化缓存"""
        if DiskCache._instance is not None:
            raise Exception("DiskCache是单例类，请使用get_instance()方法获取实例")
        
        self._base_dir = os.path.join(get_config('paths.output'), '.cache')
        ensure_dir(self._base_dir)
        self._lock = threading.Lock()
        self._index_file = os.path.join(self._base_dir, 'index.pkl')
        self._index: Dict[str, Tuple[str, float, Optional[float]]] = {}  # (file_path, timestamp, ttl)
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
            if key in self._index:
                file_path, timestamp, ttl = self._index[key]
                # 检查是否过期
                if ttl is not None and time.time() - timestamp > ttl:
                    self.delete(key)
                    return default
                
                # 读取缓存文件
                try:
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logging.error(f"读取缓存文件失败: {e}")
                    self.delete(key)
                    return default
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None表示永不过期
            
        Returns:
            bool: 是否设置成功
        """
        with self._lock:
            file_path = self._get_file_path(key)
            
            # 保存值到文件
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                logging.error(f"保存缓存文件失败: {e}")
                return False
            
            # 更新索引
            self._index[key] = (file_path, time.time(), ttl)
            self._save_index()
            return True
    
    def delete(self, key: str) -> bool:
        """
        删除缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否删除成功
        """
        with self._lock:
            if key in self._index:
                file_path, _, _ = self._index[key]
                
                # 删除缓存文件
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logging.error(f"删除缓存文件失败: {e}")
                
                # 更新索引
                del self._index[key]
                self._save_index()
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
            if key in self._index:
                file_path, timestamp, ttl = self._index[key]
                # 检查是否过期
                if ttl is not None and time.time() - timestamp > ttl:
                    self.delete(key)
                    return False
                return os.path.exists(file_path)
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            # 删除所有缓存文件
            for key in list(self._index.keys()):
                self.delete(key)
            
            # 清空索引
            self._index.clear()
            self._save_index()
    
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
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key_parts = [key_prefix if key_prefix else func.__name__]
            
            # 添加位置参数
            for arg in args:
                key_parts.append(str(arg))
            
            # 添加关键字参数（按键排序）
            for k in sorted(kwargs.keys()):
                key_parts.append(f"{k}={kwargs[k]}")
            
            cache_key = ":".join(key_parts)
            
            # 选择缓存实现
            cache = DiskCache.get_instance() if disk_cache else MemoryCache.get_instance()
            
            # 尝试从缓存获取
            if cache.exists(cache_key):
                return cache.get(cache_key)
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator 


class LRUCache:
    """
    LRU（最近最少使用）缓存实现
    
    使用OrderedDict保持项目的使用顺序，实现高效的LRU淘汰策略
    """
    
    def __init__(self, capacity: int):
        """
        初始化LRU缓存
        
        Args:
            capacity: 缓存容量
        """
        self._capacity = max(1, capacity)
        self._cache = {}
        self._usage_order = []
        self._lock = threading.Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值，并更新使用顺序
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            缓存值或默认值
        """
        with self._lock:
            if key not in self._cache:
                return default
            
            # 更新使用顺序
            self._usage_order.remove(key)
            self._usage_order.append(key)
            
            return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            # 如果键已存在，更新使用顺序
            if key in self._cache:
                self._usage_order.remove(key)
            # 如果缓存已满，删除最久未使用的项
            elif len(self._cache) >= self._capacity:
                oldest_key = self._usage_order.pop(0)
                del self._cache[oldest_key]
            
            # 添加新项
            self._cache[key] = value
            self._usage_order.append(key)
    
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
                self._usage_order.remove(key)
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否存在
        """
        with self._lock:
            return key in self._cache
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._usage_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict: 包含缓存统计信息的字典
        """
        with self._lock:
            return {
                'capacity': self._capacity,
                'size': len(self._cache),
                'usage': len(self._cache) / self._capacity if self._capacity > 0 else 0
            } 