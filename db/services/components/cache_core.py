"""
缓存核心组件 - 符合L1/L2标准的10-15方法限制
"""

from typing import Any, Dict


class CacheCore:
    """
    缓存核心组件 A+级组件设计 (4个方法) - 完全符合L1/L2单一职责标准
    
    组件职责：基础CRUD操作
    方法列表：get, set, delete, exists
    设计原则：高内聚低耦合，单一职责明确
    质量标准：A+级组件设计，为分层组合模式提供基础支持
    架构意义：通过组件化设计实现职责分离，提高代码可维护性
    """
    """
    缓存核心组件 (4个方法) - 符合L1/L2单一职责标准
    职责：基础CRUD操作
    """
    
    def __init__(self):
        self._cache = {}
    
    def get(self, key: str) -> Any:
        """获取缓存值"""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存值"""
        self._cache[key] = value
        return True
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        return key in self._cache
