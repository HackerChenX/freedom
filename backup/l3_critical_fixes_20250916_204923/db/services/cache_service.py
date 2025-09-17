"""
缓存服务 - L3数据服务层
统一的缓存管理服务，提供高效的数据缓存功能
"""

import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging

# 暂时不继承接口，避免复杂的抽象方法实现
# from db.interfaces.cache_interface import ICacheService
from utils.enhanced_exception_handler import exception_handler
from utils.enhanced_performance_monitor import performance_monitor
from utils.logger import get_logger

logger = get_logger(__name__)


class CacheService:
    """
    统一缓存服务
    L3数据服务层的标准缓存实现
    """
    
    def __init__(self):
        """初始化缓存服务"""
        self.logger = logger
        self.cache_layer = None  # 将在需要时初始化
        self.stock_config = {
            'stock_data': {
                'ttl': 300,  # 5分钟
                'levels': ['memory', 'disk']
            },
            'indicator_data': {
                'ttl': 600,  # 10分钟
                'levels': ['memory', 'disk']
            },
            'market_data': {
                'ttl': 60,   # 1分钟
                'levels': ['memory']
            },
            'industry_data': {
                'ttl': 3600, # 1小时
                'levels': ['memory', 'disk']
            }
        }
        
        self.logger.info("缓存服务初始化完成")
    
    # 实现ICacheService接口的抽象方法
    def get_stock_basic_Interface(self, code: str) -> Optional[Dict[str, Any]]:
        """获取股票基础信息"""
        key = f"stock:basic:{code}"
        if self.cache_layer:
            return self.cache_layer.get_8(key)
        return None
    
    def set_stock_basic_Interface(self, code: str, data: Dict[str, Any]) -> bool:
        """设置股票基础信息"""
        key = f"stock:basic:{code}"
        if self.cache_layer:
            return self.cache_layer.set_8(key, data, 3600)  # 1小时TTL
        return False
    
    def get_stock_daily_Interface(self, code: str, start_date: Union[str, datetime], 
                       end_date: Union[str, datetime]) -> Optional[List[Dict[str, Any]]]:
        """获取股票日线数据"""
        key = f"stock:daily:{code}:{start_date}:{end_date}"
        if self.cache_layer:
            return self.cache_layer.get_8(key)
        return None
    
    def set_stock_daily_Interface(self, code: str, start_date: Union[str, datetime], 
                       end_date: Union[str, datetime], data: List[Dict[str, Any]]) -> bool:
        """设置股票日线数据"""
        key = f"stock:daily:{code}:{start_date}:{end_date}"
        if self.cache_layer:
            return self.cache_layer.set_8(key, data, 300)  # 5分钟TTL
        return False
    
    def get_stock_minute_Interface(self, code: str, start_time: Union[str, datetime], 
                        end_time: Union[str, datetime]) -> Optional[List[Dict[str, Any]]]:
        """获取股票分钟数据"""
        key = f"stock:minute:{code}:{start_time}:{end_time}"
        if self.cache_layer:
            return self.cache_layer.get_8(key)
        return None
    
    def set_stock_minute_Interface(self, code: str, start_time: Union[str, datetime], 
                        end_time: Union[str, datetime], data: List[Dict[str, Any]]) -> bool:
        """设置股票分钟数据"""
        key = f"stock:minute:{code}:{start_time}:{end_time}"
        if self.cache_layer:
            return self.cache_layer.set_8(key, data, 60)  # 1分钟TTL
        return False
    
    def get_indicator_data_Interface(self, code: str, indicator: str, period: int, 
                          start_date: Union[str, datetime], end_date: Union[str, datetime]) -> Optional[Dict[str, Any]]:
        """获取指标数据"""
        key = f"indicator:{indicator}:{code}:{period}:{start_date}:{end_date}"
        if self.cache_layer:
            return self.cache_layer.get_8(key)
        return None
    
    def set_indicator_data_Interface(self, code: str, indicator: str, period: int, 
                          start_date: Union[str, datetime], end_date: Union[str, datetime], 
                          data: Dict[str, Any]) -> bool:
        """设置指标数据"""
        key = f"indicator:{indicator}:{code}:{period}:{start_date}:{end_date}"
        if self.cache_layer:
            return self.cache_layer.set_8(key, data, 600)  # 10分钟TTL
        return False
    
    def get_market_overview_Interface(self, date: Union[str, datetime]) -> Optional[Dict[str, Any]]:
        """获取市场概览"""
        key = f"market:overview:{date}"
        if self.cache_layer:
            return self.cache_layer.get_8(key)
        return None
    
    def set_market_overview_Interface(self, date: Union[str, datetime], data: Dict[str, Any]) -> bool:
        """设置市场概览"""
        key = f"market:overview:{date}"
        if self.cache_layer:
            return self.cache_layer.set_8(key, data, 60)  # 1分钟TTL
        return False
    
    def get_industry_stocks_Interface(self, industry: str) -> Optional[List[str]]:
        """获取行业股票列表"""
        key = f"industry:stocks:{industry}"
        if self.cache_layer:
            return self.cache_layer.get_8(key)
        return None
    
    def set_industry_stocks_Interface(self, industry: str, stocks: List[str]) -> bool:
        """设置行业股票列表"""
        key = f"industry:stocks:{industry}"
        if self.cache_layer:
            return self.cache_layer.set_8(key, stocks, 3600)  # 1小时TTL
        return False
    
    def invalidate_stock_Interface(self, code: str) -> bool:
        """使股票相关缓存失效"""
        if not self.cache_layer:
            return False
        
        patterns = [
            f"stock:*:{code}*",
            f"indicator:*:{code}*"
        ]
        
        success = True
        for pattern in patterns:
            try:
                if hasattr(self.cache_layer, 'delete_pattern'):
                    self.cache_layer.delete_pattern(pattern)
            except Exception as e:
                self.logger.warning(f"删除缓存模式失败 {pattern}: {e}")
                success = False
        
        return success
    
    def invalidate_market_Interface(self, date: Union[str, datetime]) -> bool:
        """使市场数据缓存失效"""
        if not self.cache_layer:
            return False
        
        key = f"market:overview:{date}"
        try:
            if hasattr(self.cache_layer, 'delete_8'):
                self.cache_layer.delete_8(key)
                return True
        except Exception as e:
            self.logger.warning(f"删除市场缓存失败 {key}: {e}")
        
        return False
    
    def get_cache_stats_Interface(self) -> Dict[str, Any]:
        """获取缓存统计"""
        stats = {
            'total_keys': 0,
            'memory_usage': 0,
            'hit_rate': 0.0,
            'miss_rate': 0.0
        }
        
        if self.cache_layer and hasattr(self.cache_layer, 'get_stats'):
            try:
                stats.update(self.cache_layer.get_stats())
            except Exception as e:
                self.logger.warning(f"获取缓存统计失败: {e}")
        
        return stats
    
    def clear_all_cache_Interface(self) -> bool:
        """清空所有缓存"""
        if not self.cache_layer:
            return False
        
        try:
            if hasattr(self.cache_layer, 'clear_all'):
                self.cache_layer.clear_all()
                self.logger.info("所有缓存已清空")
                return True
        except Exception as e:
            self.logger.error(f"清空缓存失败: {e}")
        
        return False
    
    # 额外的便利方法
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def get_stock_data(self, code: str, start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """获取股票数据缓存"""
        return self.get_stock_daily_Interface(code, start_date, end_date)
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def set_stock_data(self, code: str, start_date: str, end_date: str, data: Dict[str, Any]) -> bool:
        """设置股票数据缓存"""
        return self.set_stock_daily_Interface(code, start_date, end_date, [data])

    def get(self, key: str) -> Optional[Any]:
        """通用缓存获取方法"""
        if self.cache_layer and hasattr(self.cache_layer, 'get_8'):
            return self.cache_layer.get_8(key)
        return None

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """通用缓存设置方法"""
        if self.cache_layer and hasattr(self.cache_layer, 'set_8'):
            return self.cache_layer.set_8(key, value, ttl)
        return False

    def delete(self, key: str) -> bool:
        """删除缓存"""
        if self.cache_layer and hasattr(self.cache_layer, 'delete_8'):
            return self.cache_layer.delete_8(key)
        return False

    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        if self.cache_layer and hasattr(self.cache_layer, 'exists'):
            return self.cache_layer.exists(key)
        return False
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"CacheService(initialized={self.cache_layer is not None})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


# 全局缓存服务实例
_cache_service = None


def get_cache_service() -> CacheService:
    """获取全局缓存服务实例"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
