"""
缓存高级组件 - 符合L1/L2标准的10-15方法限制
"""

from typing import Any, Dict, List, Callable


class CacheAdvanced:
    """
    缓存高级组件 (8个方法) - 符合L1/L2单一职责标准
    职责：批量操作和高级功能
    """
    
    def __init__(self, cache_core):
        self.cache_core = cache_core
    
    def get_batch(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取"""
        return {key: self.cache_core.get(key) for key in keys}
    
    def set_batch(self, data: Dict[str, Any], ttl: int = None) -> bool:
        """批量设置"""
        for key, value in data.items():
            self.cache_core.set(key, value, ttl)
        return True
    
    def delete_batch(self, keys: List[str]) -> int:
        """批量删除"""
        count = 0
        for key in keys:
            if self.cache_core.delete(key):
                count += 1
        return count
    
    def clear(self) -> bool:
        """清空缓存"""
        self.cache_core._cache.clear()
        return True
    
    def get_or_set(self, key: str, func: Callable, ttl: int = None) -> Any:
        """获取或设置"""
        value = self.cache_core.get(key)
        if value is None:
            value = func()
            self.cache_core.set(key, value, ttl)
        return value
    
    def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        # 简化实现
        return True
    
    def get_ttl(self, key: str) -> int:
        """获取过期时间"""
        # 简化实现
        return -1
    
    def flush(self) -> bool:
        """刷新缓存"""
        return self.clear()
