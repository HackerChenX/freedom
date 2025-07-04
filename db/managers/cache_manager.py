"""
缓存管理器实现

实现ICacheManager接口，提供多级缓存功能
"""

import time
import threading
import hashlib
import pickle
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import OrderedDict

from db.interfaces.cache_interface import ICacheManager, IMultiLevelCache, ICacheStrategy, ICacheEventListener
from utils.logger import get_logger
from utils.decorators import performance_monitor

logger = get_logger(__name__)


class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化LRU缓存
        
        Args:
            max_size: 最大缓存条目数
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.access_times = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                # 移动到末尾（最近访问）
                value = self.cache.pop(key)
                self.cache[key] = value
                self.access_times[key] = time.time()
                return value
            return None
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存项"""
        with self.lock:
            if key in self.cache:
                # 更新现有项
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # 移除最久未使用的项
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
                self.access_times.pop(oldest_key, None)
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                self.access_times.pop(key, None)
                return True
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)


class TTLCache:
    """带TTL的缓存实现"""
    
    def __init__(self, default_ttl: int = 3600):
        """
        初始化TTL缓存
        
        Args:
            default_ttl: 默认TTL（秒）
        """
        self.default_ttl = default_ttl
        self.cache = {}
        self.expiry_times = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                # 检查是否过期
                if time.time() < self.expiry_times.get(key, 0):
                    return self.cache[key]
                else:
                    # 过期，删除
                    self.cache.pop(key, None)
                    self.expiry_times.pop(key, None)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存项"""
        with self.lock:
            self.cache[key] = value
            expiry_time = time.time() + (ttl or self.default_ttl)
            self.expiry_times[key] = expiry_time
    
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                self.expiry_times.pop(key, None)
                return True
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.expiry_times.clear()
    
    def cleanup_expired(self) -> int:
        """清理过期项"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, expiry_time in self.expiry_times.items()
                if current_time >= expiry_time
            ]
            
            for key in expired_keys:
                self.cache.pop(key, None)
                self.expiry_times.pop(key, None)
            
            return len(expired_keys)


class DefaultCacheStrategy(ICacheStrategy):
    """默认缓存策略"""
    
    def should_cache(self, key: str, value: Any) -> bool:
        """判断是否应该缓存"""
        # 基本策略：非空值才缓存
        return value is not None
    
    def get_ttl(self, key: str, value: Any) -> Optional[int]:
        """获取TTL"""
        # 根据键前缀设置不同TTL
        if key.startswith('stock_list'):
            return 600  # 10分钟
        elif key.startswith('industry'):
            return 3600  # 1小时
        elif key.startswith('stock_info'):
            return 300  # 5分钟
        else:
            return 1800  # 默认30分钟
    
    def should_evict(self, key: str, last_access: datetime) -> bool:
        """判断是否应该驱逐"""
        # 超过1小时未访问则驱逐
        return datetime.now() - last_access > timedelta(hours=1)


class CacheEventLogger(ICacheEventListener):
    """缓存事件日志记录器"""
    
    def on_cache_hit(self, key: str) -> None:
        """缓存命中事件"""
        logger.debug(f"缓存命中: {key}")
    
    def on_cache_miss(self, key: str) -> None:
        """缓存未命中事件"""
        logger.debug(f"缓存未命中: {key}")
    
    def on_cache_set(self, key: str, value: Any) -> None:
        """缓存设置事件"""
        logger.debug(f"缓存设置: {key}")
    
    def on_cache_evict(self, key: str, reason: str) -> None:
        """缓存驱逐事件"""
        logger.debug(f"缓存驱逐: {key}, 原因: {reason}")


class CacheManager(IMultiLevelCache):
    """
    缓存管理器实现
    
    提供多级缓存功能，支持LRU和TTL策略
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: int = 3600,
                 enable_multilevel: bool = True):
        """
        初始化缓存管理器
        
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认TTL（秒）
            enable_multilevel: 是否启用多级缓存
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_multilevel = enable_multilevel
        
        # L1缓存：LRU缓存，用于频繁访问的数据
        self.l1_cache = LRUCache(max_size=max_size // 2)
        
        # L2缓存：TTL缓存，用于带过期时间的数据
        self.l2_cache = TTLCache(default_ttl=default_ttl)
        
        # 缓存策略
        self.strategy = DefaultCacheStrategy()
        
        # 事件监听器
        self.event_listeners: List[ICacheEventListener] = [CacheEventLogger()]
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0,
            'l1_hits': 0,
            'l2_hits': 0
        }
        
        self.lock = threading.RLock()
        
        # 启动清理线程
        self._start_cleanup_thread()
        
        logger.info(f"缓存管理器初始化完成，多级缓存: {'启用' if enable_multilevel else '禁用'}")
    
    @performance_monitor(threshold=0.1)
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存数据，不存在返回None
        """
        with self.lock:
            # 先尝试L1缓存
            value = self.l1_cache.get(key)
            if value is not None:
                self.stats['hits'] += 1
                self.stats['l1_hits'] += 1
                self._notify_cache_hit(key)
                return value
            
            # 再尝试L2缓存
            if self.enable_multilevel:
                value = self.l2_cache.get(key)
                if value is not None:
                    # 提升到L1缓存
                    self.l1_cache.set(key, value)
                    self.stats['hits'] += 1
                    self.stats['l2_hits'] += 1
                    self._notify_cache_hit(key)
                    return value
            
            # 缓存未命中
            self.stats['misses'] += 1
            self._notify_cache_miss(key)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒）
            
        Returns:
            bool: 设置成功返回True
        """
        try:
            with self.lock:
                # 检查是否应该缓存
                if not self.strategy.should_cache(key, value):
                    return False
                
                # 获取TTL
                if ttl is None:
                    ttl = self.strategy.get_ttl(key, value)
                
                # 设置到L1缓存
                self.l1_cache.set(key, value)
                
                # 如果有TTL，也设置到L2缓存
                if ttl and self.enable_multilevel:
                    self.l2_cache.set(key, value, ttl)
                
                self.stats['sets'] += 1
                self._notify_cache_set(key, value)
                
                return True
                
        except Exception as e:
            logger.error(f"设置缓存失败: {key}, 错误: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        删除缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 删除成功返回True
        """
        with self.lock:
            deleted = False
            
            # 从L1缓存删除
            if self.l1_cache.delete(key):
                deleted = True
            
            # 从L2缓存删除
            if self.enable_multilevel and self.l2_cache.delete(key):
                deleted = True
            
            if deleted:
                self._notify_cache_evict(key, "manual_delete")
            
            return deleted
    
    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 存在返回True
        """
        return self.get(key) is not None
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self.lock:
            self.l1_cache.clear()
            if self.enable_multilevel:
                self.l2_cache.clear()
            
            # 重置统计
            self.stats = {
                'hits': 0,
                'misses': 0,
                'sets': 0,
                'evictions': 0,
                'l1_hits': 0,
                'l2_hits': 0
            }
            
            logger.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            hit_rate = 0.0
            total_requests = self.stats['hits'] + self.stats['misses']
            if total_requests > 0:
                hit_rate = self.stats['hits'] / total_requests
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'l1_size': self.l1_cache.size(),
                'l2_size': len(self.l2_cache.cache) if self.enable_multilevel else 0,
                'total_size': self.l1_cache.size() + (len(self.l2_cache.cache) if self.enable_multilevel else 0)
            }
    
    # IMultiLevelCache接口实现
    def get_from_level(self, key: str, level: int) -> Optional[Any]:
        """
        从指定级别获取缓存
        
        Args:
            key: 缓存键
            level: 缓存级别（1或2）
            
        Returns:
            Optional[Any]: 缓存数据
        """
        if level == 1:
            return self.l1_cache.get(key)
        elif level == 2 and self.enable_multilevel:
            return self.l2_cache.get(key)
        else:
            return None
    
    def set_to_level(self, key: str, value: Any, level: int, ttl: Optional[int] = None) -> bool:
        """
        设置到指定级别缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            level: 缓存级别
            ttl: 生存时间
            
        Returns:
            bool: 设置成功返回True
        """
        try:
            if level == 1:
                self.l1_cache.set(key, value)
                return True
            elif level == 2 and self.enable_multilevel:
                self.l2_cache.set(key, value, ttl)
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"设置缓存到级别{level}失败: {key}, 错误: {e}")
            return False
    
    def promote_to_higher_level(self, key: str, from_level: int, to_level: int) -> bool:
        """
        提升缓存到更高级别
        
        Args:
            key: 缓存键
            from_level: 源级别
            to_level: 目标级别
            
        Returns:
            bool: 提升成功返回True
        """
        value = self.get_from_level(key, from_level)
        if value is not None:
            return self.set_to_level(key, value, to_level)
        return False
    
    # 私有方法
    def _start_cleanup_thread(self) -> None:
        """启动清理线程"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(300)  # 每5分钟清理一次
                    if self.enable_multilevel:
                        expired_count = self.l2_cache.cleanup_expired()
                        if expired_count > 0:
                            logger.debug(f"清理过期缓存项: {expired_count}")
                            self.stats['evictions'] += expired_count
                except Exception as e:
                    logger.error(f"缓存清理线程错误: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _notify_cache_hit(self, key: str) -> None:
        """通知缓存命中事件"""
        for listener in self.event_listeners:
            try:
                listener.on_cache_hit(key)
            except Exception as e:
                logger.error(f"缓存事件监听器错误: {e}")
    
    def _notify_cache_miss(self, key: str) -> None:
        """通知缓存未命中事件"""
        for listener in self.event_listeners:
            try:
                listener.on_cache_miss(key)
            except Exception as e:
                logger.error(f"缓存事件监听器错误: {e}")
    
    def _notify_cache_set(self, key: str, value: Any) -> None:
        """通知缓存设置事件"""
        for listener in self.event_listeners:
            try:
                listener.on_cache_set(key, value)
            except Exception as e:
                logger.error(f"缓存事件监听器错误: {e}")
    
    def _notify_cache_evict(self, key: str, reason: str) -> None:
        """通知缓存驱逐事件"""
        for listener in self.event_listeners:
            try:
                listener.on_cache_evict(key, reason)
            except Exception as e:
                logger.error(f"缓存事件监听器错误: {e}")
    
    def add_event_listener(self, listener: ICacheEventListener) -> None:
        """添加事件监听器"""
        self.event_listeners.append(listener)
    
    def remove_event_listener(self, listener: ICacheEventListener) -> None:
        """移除事件监听器"""
        if listener in self.event_listeners:
            self.event_listeners.remove(listener) 