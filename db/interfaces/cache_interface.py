"""
缓存管理接口定义

定义缓存管理的标准接口
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from datetime import datetime


class ICacheManager(ABC):
    """
    缓存管理器接口
    
    定义缓存操作的标准接口
    """
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存数据，不存在返回None
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        删除缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 删除成功返回True
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 存在返回True
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        清空所有缓存
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        pass


class IMultiLevelCache(ICacheManager):
    """
    多级缓存接口
    """
    
    @abstractmethod
    def get_from_level(self, key: str, level: int) -> Optional[Any]:
        """
        从指定级别获取缓存
        
        Args:
            key: 缓存键
            level: 缓存级别
            
        Returns:
            Optional[Any]: 缓存数据
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass


class ICacheStrategy(ABC):
    """
    缓存策略接口
    """
    
    @abstractmethod
    def should_cache(self, key: str, value: Any) -> bool:
        """
        判断是否应该缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            
        Returns:
            bool: 应该缓存返回True
        """
        pass
    
    @abstractmethod
    def get_ttl(self, key: str, value: Any) -> Optional[int]:
        """
        获取缓存生存时间
        
        Args:
            key: 缓存键
            value: 缓存值
            
        Returns:
            Optional[int]: 生存时间（秒），None表示永不过期
        """
        pass
    
    @abstractmethod
    def should_evict(self, key: str, last_access: datetime) -> bool:
        """
        判断是否应该驱逐缓存
        
        Args:
            key: 缓存键
            last_access: 最后访问时间
            
        Returns:
            bool: 应该驱逐返回True
        """
        pass


class ICacheEventListener(ABC):
    """
    缓存事件监听器接口
    """
    
    @abstractmethod
    def on_cache_hit(self, key: str) -> None:
        """
        缓存命中事件
        
        Args:
            key: 缓存键
        """
        pass
    
    @abstractmethod
    def on_cache_miss(self, key: str) -> None:
        """
        缓存未命中事件
        
        Args:
            key: 缓存键
        """
        pass
    
    @abstractmethod
    def on_cache_set(self, key: str, value: Any) -> None:
        """
        缓存设置事件
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        pass
    
    @abstractmethod
    def on_cache_evict(self, key: str, reason: str) -> None:
        """
        缓存驱逐事件
        
        Args:
            key: 缓存键
            reason: 驱逐原因
        """
        pass 