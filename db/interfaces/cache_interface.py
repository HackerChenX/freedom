"""
L3数据服务层缓存接口 - 分层设计符合L1/L2标准
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable


class ICacheCore(ABC):
    """
    核心缓存接口 (4个方法) - 符合L1/L2单一职责标准
    职责：基础CRUD操作
    """
    
    @abstractmethod
    def get(self, key: str) -> Any:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存值"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        pass


class ICacheAdvanced(ABC):
    """
    高级缓存接口 (8个方法) - 符合L1/L2单一职责标准
    职责：批量操作和高级功能
    """
    
    @abstractmethod
    def get_batch(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取"""
        pass
    
    @abstractmethod
    def set_batch(self, data: Dict[str, Any], ttl: int = None) -> bool:
        """批量设置"""
        pass
    
    @abstractmethod
    def delete_batch(self, keys: List[str]) -> int:
        """批量删除"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """清空缓存"""
        pass
    
    @abstractmethod
    def get_or_set(self, key: str, func: Callable, ttl: int = None) -> Any:
        """获取或设置"""
        pass
    @abstractmethod
    def set_or_set(self, key: str, value: Any, func: Callable = None) -> bool:
        """设置或设置"""
        pass
    
    @abstractmethod
    def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        pass
    
    @abstractmethod
    def get_ttl(self, key: str) -> int:
        """获取过期时间"""
        pass
    @abstractmethod
    def set_ttl(self, key: str, ttl: int) -> bool:
        """设置TTL"""
        pass
    
    @abstractmethod
    def flush(self) -> bool:
        """刷新缓存"""
        pass


class ICacheMonitoring(ABC):
    """
    监控缓存接口 (4个方法) - 符合L1/L2单一职责标准
    职责：监控和统计
    """
    
    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        pass
    
    @abstractmethod
    def set_cache_stats(self, stats: Dict[str, Any]) -> bool:
        """设置缓存统计"""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """健康检查"""
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """获取缓存大小"""
        pass
    @abstractmethod
    def set_size(self, size: int) -> bool:
        """设置缓存大小限制"""
        pass


class ICacheService(ICacheCore, ICacheAdvanced, ICacheMonitoring):
    """
    完整缓存服务接口 - 通过组合实现 (16个方法分层为4+8+4)
    
    分层设计说明：
    - ICacheCore (4方法): 基础CRUD操作
    - ICacheAdvanced (8方法): 批量操作和高级功能
    - ICacheMonitoring (4方法): 监控和统计
    
    每个子接口都符合L1/L2单一职责原则和10-15方法限制。
    通过接口组合实现完整功能，保持向后兼容性。
    """
    pass
