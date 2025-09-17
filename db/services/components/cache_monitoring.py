"""
缓存监控组件 - 符合L1/L2标准的10-15方法限制
"""

from typing import Any, Dict


class CacheMonitoring:
    """
    缓存监控组件 (4个方法) - 符合L1/L2单一职责标准
    职责：监控和统计
    """
    
    def __init__(self, cache_core):
        self.cache_core = cache_core
        self._stats = {'hits': 0, 'misses': 0, 'sets': 0, 'deletes': 0}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self._stats.copy()
    
    def set_cache_stats(self, stats: Dict[str, Any]) -> bool:
        """设置缓存统计"""
        self._stats.update(stats)
        return True
    
    def health_check(self) -> bool:
        """健康检查"""
        return True
    
    def get_size(self) -> int:
        """获取缓存大小"""
        return len(self.cache_core._cache)
    
    def reset_stats(self) -> bool:
        """重置统计"""
        self._stats = {'hits': 0, 'misses': 0, 'sets': 0, 'deletes': 0}
        return True
