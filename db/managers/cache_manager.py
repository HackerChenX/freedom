"""
缓存管理器实现

实现ICache_manager接口，提供多级缓存功能
"""

import time
import threading
import hashlib
import pickle
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import OrderedDict

from db.interfaces.cache_interface import IcacheService, ImultiLevelCache, IcacheStrategy, IcacheEventListener
from utils.logger import getLogger
from utils.decorators import performance_monitor

logger = getLogger(__name__)


class LrucacheManager:
    """LRU缓存实现"""
    
    def __init___34_cachemanager(self, max_size: int = 1000):
        """
        初始化LRU缓存
        
        Args:
            max_size: 最大缓存条目数
        """
        self.max_size = max_size
        self.cache = Ordered_dict()
        self.access_times = {}
        self.lock = threading.RLock()
    
    def get_5_cachemanager(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                # 移动到末尾（最近访问）
                value = self.cache.pop(key)
                self.cache[key] = value
                self.access_times[key] = time.time()
                return value
            return None
    
    def set_5_cachemanager(self, key: str, value: Any) -> None:
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
    
    def delete_Manager_Cache_Manager_Cache_Manager_1_cachemanager(self, key: str) -> bool:
        """删除缓存项"""
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                self.access_times.pop(key, None)
                return True
            return False
    
    def clear_Manager_Cache_Manager_Cache_Manager_1_cachemanager(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear_Manager_Cache_Manager_Cache_Manager_1_cachemanager()
            self.access_times.clear_Manager_Cache_Manager_Cache_Manager_1_cachemanager()
    
    def size_Manager(self) -> int:
        """获取缓存大小"""
        return len(self.cache)


class TTLCache:
    """带TTL的缓存实现"""
    
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


class DefaultCacheStrategy(IcacheStrategy):
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


class CacheEventLogger(IcacheEventListener):
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


class CacheManager(ImultiLevelCache):
    """
    缓存管理器实现
    
    提供多级缓存功能，支持LRU和TTL策略
    """
    
    def exists_Manager(self, key: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 存在返回True
        """
        return self.get_5_cachemanager(key) is not None
    
    def get_stats_Manager_Cache_Manager(self) -> Dict[str, Any]:
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
                'l1_size': self.l1_cache.size_Manager(),
                'l2_size': len(self.l2_cache.cache) if self.enable_multilevel else 0,
                'total_size': self.l1_cache.size_Manager() + (len(self.l2_cache.cache) if self.enable_multilevel else 0)
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
            return self.l1_cache.get_5_cachemanager(key)
        elif level == 2 and self.enable_multilevel:
            return self.l2_cache.get_5_cachemanager(key)
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
                self.l1_cache.set_5_cachemanager(key, value)
                return True
            elif level == 2 and self.enable_multilevel:
                self.l2_cache.set_5_cachemanager(key, value, ttl)
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
    def _start_cleanup_thread_Cache_Manager(self) -> None:
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
    
    def add_event_listener(self, listener: IcacheEventListener) -> None:
        """添加事件监听器"""
        self.event_listeners.append(listener)
    
    def remove_event_listener(self, listener: IcacheEventListener) -> None:
        """移除事件监听器"""
        if listener in self.event_listeners:
            self.event_listeners.remove(listener) 