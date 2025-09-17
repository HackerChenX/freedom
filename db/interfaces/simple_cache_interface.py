"""
简化缓存服务接口

替换复杂的ICacheService接口，提供生产级的缓存服务抽象
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List


class IBasicCacheService(ABC):
    """基础缓存服务接口"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """设置缓存值"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        pass
    
    @abstractmethod
    def clear_all(self) -> bool:
        """清空所有缓存"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        pass


class IStockCacheService(ABC):
    """股票数据缓存服务接口"""
    
    @abstractmethod
    def get_stock_data(self, code: str, start_date: str, end_date: str) -> Optional[Any]:
        """获取股票数据缓存"""
        pass
    
    @abstractmethod
    def set_stock_data(self, code: str, start_date: str, end_date: str, data: Any) -> bool:
        """设置股票数据缓存"""
        pass
    
    @abstractmethod
    def get_stock_basic(self, code: str) -> Optional[Dict[str, Any]]:
        """获取股票基础信息"""
        pass
    
    @abstractmethod
    def set_stock_basic(self, code: str, data: Dict[str, Any]) -> bool:
        """设置股票基础信息"""
        pass
    
    @abstractmethod
    def invalidate_stock(self, code: str) -> bool:
        """使股票相关缓存失效"""
        pass


class IIndicatorCacheService(ABC):
    """指标数据缓存服务接口"""
    
    @abstractmethod
    def get_indicator_data(self, code: str, indicator: str, period: int, 
                          start_date: str, end_date: str) -> Optional[Any]:
        """获取指标数据缓存"""
        pass
    
    @abstractmethod
    def set_indicator_data(self, code: str, indicator: str, period: int,
                          start_date: str, end_date: str, data: Any) -> bool:
        """设置指标数据缓存"""
        pass
    
    @abstractmethod
    def invalidate_indicator(self, code: str, indicator: str) -> bool:
        """使指标缓存失效"""
        pass


class IMarketCacheService(ABC):
    """市场数据缓存服务接口"""
    
    @abstractmethod
    def get_market_overview(self, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取市场概览"""
        pass
    
    @abstractmethod
    def set_market_overview(self, data: Dict[str, Any], date: Optional[str] = None) -> bool:
        """设置市场概览"""
        pass
    
    @abstractmethod
    def invalidate_market_data(self) -> bool:
        """使市场数据缓存失效"""
        pass


class ICompleteCacheService(IBasicCacheService, IStockCacheService, 
                           IIndicatorCacheService, IMarketCacheService):
    """完整的缓存服务接口
    
    组合所有缓存服务接口，提供完整的缓存功能
    """
    pass


# 向后兼容的别名
ICacheService = ICompleteCacheService
