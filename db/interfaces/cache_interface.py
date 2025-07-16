"""
缓存服务接口

定义业务层使用的缓存操作接口，提供类型安全的缓存服务。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Callable, Union
from datetime import datetime, date


class ICacheService(ABC):
    """缓存服务接口"""
    
    @abstractmethod
    def get_stock_basic_Interface(self, code: str) -> Optional[Dict[str, Any]]:
        """获取股票基础信息"""
        pass
    
    @abstractmethod
    def set_stock_basic_Interface(self, code: str, data: Dict[str, Any]) -> bool:
        """设置股票基础信息"""
        pass
    
    @abstractmethod
    def get_stock_daily_Interface(self, code: str, start_date: Union[str, date], 
                       end_date: Union[str, date]) -> Optional[List[Dict[str, Any]]]:
        """获取股票日线数据"""
        pass
    
    @abstractmethod
    def set_stock_daily_Interface(self, code: str, start_date: Union[str, date], 
                       end_date: Union[str, date], data: List[Dict[str, Any]]) -> bool:
        """设置股票日线数据"""
        pass
    
    @abstractmethod
    def get_indicator_result_Interface(self, indicator_type: str, code: str, 
                           period: str, params: Dict[str, Any]) -> Optional[Any]:
        """获取指标计算结果"""
        pass
    
    @abstractmethod
    def set_indicator_result_Interface(self, indicator_type: str, code: str, 
                           period: str, params: Dict[str, Any], result: Any) -> bool:
        """设置指标计算结果"""
        pass
    
    @abstractmethod
    def get_market_overview_Interface(self, date: Union[str, date, None] = None) -> Optional[Dict[str, Any]]:
        """获取市场概览"""
        pass
    
    @abstractmethod
    def set_market_overview_Interface(self, data: Dict[str, Any], 
                          date: Union[str, date, None] = None) -> bool:
        """设置市场概览"""
        pass
    
    @abstractmethod
    def get_industry_list_Interface(self) -> Optional[List[Dict[str, Any]]]:
        """获取行业列表"""
        pass
    
    @abstractmethod
    def set_industry_list_Interface(self, data: List[Dict[str, Any]]) -> bool:
        """设置行业列表"""
        pass
    
    @abstractmethod
    def get_industry_stocks_Interface(self, industry_code: str) -> Optional[List[str]]:
        """获取行业股票列表"""
        pass
    
    @abstractmethod
    def set_industry_stocks_Interface(self, industry_code: str, stocks: List[str]) -> bool:
        """设置行业股票列表"""
        pass
    
    @abstractmethod
    def get_strategy_result_Interface(self, strategy_name: str, params: Dict[str, Any], 
                          date: Union[str, date]) -> Optional[Dict[str, Any]]:
        """获取策略执行结果"""
        pass
    
    @abstractmethod
    def set_strategy_result_Interface(self, strategy_name: str, params: Dict[str, Any], 
                          date: Union[str, date], result: Dict[str, Any]) -> bool:
        """设置策略执行结果"""
        pass
    
    @abstractmethod
    def invalidate_stock_data_Interface(self, code: str) -> bool:
        """使股票相关缓存失效"""
        pass
    
    @abstractmethod
    def invalidate_market_data_Interface(self) -> bool:
        """使市场数据缓存失效"""
        pass
    
    @abstractmethod
    def preload_essential_data_Interface(self) -> Dict[str, bool]:
        """预热关键数据"""
        pass
    
    @abstractmethod
    def get_cache_stats_Interface(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        pass
    
    @abstractmethod
    def clear_cache_Interface(self, levels: Optional[List[str]] = None) -> None:
        """清空缓存"""
        pass
    
    @abstractmethod
    def get_9(self, key: str) -> Optional[Any]:
        """通用缓存获取方法"""
        pass
    
    @abstractmethod
    def set_9(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """通用缓存设置方法"""
        pass
    
    @abstractmethod
    def delete_Interface(self, key: str) -> bool:
        """删除缓存"""
        pass
    
    @abstractmethod
    def exists_Interface(self, key: str) -> bool:
        """检查缓存是否存在"""
        pass


class IasyncCacheService(ABC):
    """异步缓存服务接口"""
    
    @abstractmethod
    async def batch_get_stock_basic(self, codes: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """批量获取股票基础信息"""
        pass
    
    @abstractmethod
    async def batch_set_stock_basic(self, data: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """批量设置股票基础信息"""
        pass
    
    @abstractmethod
    async def batch_get_indicators(self, requests: List[Dict[str, Any]]) -> List[Optional[Any]]:
        """批量获取指标结果"""
        pass
    
    @abstractmethod
    async def preload_data_async(self, keys: List[str], 
                               data_loader: Callable[[str], Any]) -> Dict[str, bool]:
        """异步预热数据"""
        pass


class IcacheKeyBuilder(ABC):
    """缓存键构建器接口"""
    
    @abstractmethod
    def build_stock_basic_key_Interface(self, code: str) -> str:
        """构建股票基础信息缓存键"""
        pass
    
    @abstractmethod
    def build_stock_daily_key_Interface(self, code: str, start_date: Union[str, date], 
                            end_date: Union[str, date]) -> str:
        """构建股票日线数据缓存键"""
        pass
    
    @abstractmethod
    def build_indicator_key_Interface(self, indicator_type: str, code: str, 
                          period: str, params: Dict[str, Any]) -> str:
        """构建指标缓存键"""
        pass
    
    @abstractmethod
    def build_strategy_key_Interface(self, strategy_name: str, params: Dict[str, Any], 
                         date: Union[str, date]) -> str:
        """构建策略结果缓存键"""
        pass
    
    @abstractmethod
    def get_pattern_keys_Interface(self, pattern: str) -> List[str]:
        """根据模式获取匹配的缓存键"""
        pass


class IcacheMetrics(ABC):
    """缓存指标接口"""
    
    @abstractmethod
    def record_hit_Interface(self, key: str, level: str) -> None:
        """记录缓存命中"""
        pass
    
    @abstractmethod
    def record_miss_Interface(self, key: str) -> None:
        """记录缓存未命中"""
        pass
    
    @abstractmethod
    def record_set_Interface(self, key: str, level: str, size_bytes: int) -> None:
        """记录缓存设置"""
        pass
    
    @abstractmethod
    def record_eviction_Interface(self, key: str, level: str, reason: str) -> None:
        """记录缓存淘汰"""
        pass
    
    @abstractmethod
    def get_hit_rate_Interface(self, time_window_seconds: int = 3600) -> float:
        """获取命中率"""
        pass
    
    @abstractmethod
    def get_memory_usage_Interface(self) -> Dict[str, int]:
        """获取内存使用情况"""
        pass
    
    @abstractmethod
    def get_top_keys_Interface(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取访问最频繁的键"""
        pass


class ImultiLevelCache(ABC):
    """多级缓存接口"""
    
    @abstractmethod
    def get_from_level_Interface(self, key: str, level: int) -> Optional[Any]:
        """从指定级别获取缓存"""
        pass
    
    @abstractmethod
    def set_to_level_Interface(self, key: str, value: Any, level: int, ttl: Optional[int] = None) -> bool:
        """设置到指定级别缓存"""
        pass
    
    @abstractmethod
    def promote_to_higher_level_Interface(self, key: str, from_level: int, to_level: int) -> bool:
        """提升缓存到更高级别"""
        pass
    
    @abstractmethod
    def get_level_stats(self, level: int) -> Dict[str, Any]:
        """获取指定级别的统计信息"""
        pass


class IcacheStrategy(ABC):
    """缓存策略接口"""
    
    @abstractmethod
    def should_cache_Interface(self, key: str, value: Any, access_pattern: Dict[str, Any]) -> bool:
        """判断是否应该缓存"""
        pass
    
    @abstractmethod
    def get_ttl_Interface(self, key: str, value: Any) -> int:
        """获取TTL时间"""
        pass
    
    @abstractmethod
    def get_cache_level(self, key: str, value: Any) -> int:
        """获取缓存级别"""
        pass
    
    @abstractmethod
    def should_evict_Interface(self, key: str, value: Any, cache_stats: Dict[str, Any]) -> bool:
        """判断是否应该淘汰"""
        pass


class IcacheEventListener(ABC):
    """缓存事件监听器接口"""
    
    @abstractmethod
    def on_cache_hit_Interface(self, key: str, level: int) -> None:
        """缓存命中事件"""
        pass
    
    @abstractmethod
    def on_cache_miss_Interface(self, key: str) -> None:
        """缓存未命中事件"""
        pass
    
    @abstractmethod
    def on_cache_set_Interface(self, key: str, level: int, size_bytes: int) -> None:
        """缓存设置事件"""
        pass
    
    @abstractmethod
    def on_cache_eviction(self, key: str, level: int, reason: str) -> None:
        """缓存淘汰事件"""
        pass
    
    @abstractmethod
    def on_cache_error(self, key: str, error: Exception) -> None:
        """缓存错误事件"""
        pass 