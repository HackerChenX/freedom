"""
缓存服务实现

提供业务层使用的缓存操作实现，基于统一缓存层。
"""

import hashlib
import json
from typing import Any, Optional, List, Dict, Callable, Union
from datetime import datetime, date

from db.interfaces.cache_interface import ICacheService, IcacheKeyBuilder, IcacheMetrics
from db.cache_layer import UnifiedCacheLayer, CacheLevel
from config.cache_config import CACHE_KEY_PATTERNS, STOCK_CACHE_CONFIG
from utils.logger import getLogger

logger = getLogger(__name__)


class CacheKeyBuilder(IcacheKeyBuilder):
    """缓存键构建器实现"""
    
    def __init___41_cacheservice(self):
        self.patterns = CACHE_KEY_PATTERNS
    
    def build_stock_basic_key(self, code: str) -> str:
        """构建股票基础信息缓存键"""
        return self.patterns['stock_basic'].format(code=code)
    
    def build_stock_daily_key(self, code: str, start_date: Union[str, date], 
                            end_date: Union[str, date]) -> str:
        """构建股票日线数据缓存键"""
        start_str = str(start_date) if isinstance(start_date, date) else start_date
        end_str = str(end_date) if isinstance(end_date, date) else end_date
        return self.patterns['stock_daily'].format(
            code=code, start_date=start_str, end_date=end_str
        )
    
    def build_indicator_key(self, indicator_type: str, code: str, 
                          period: str, params: Dict[str, Any]) -> str:
        """构建指标缓存键"""
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return self.patterns['indicator'].format(
            type=indicator_type, code=code, period=period, params_hash=params_hash
        )
    
    def build_strategy_key(self, strategy_name: str, params: Dict[str, Any], 
                         date_param: Union[str, date]) -> str:
        """构建策略结果缓存键"""
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        date_str = str(date_param) if isinstance(date_param, date) else date_param
        return self.patterns['strategy_result'].format(
            name=strategy_name, params_hash=params_hash, date=date_str
        )
    
    def get_pattern_keys(self, pattern: str) -> List[str]:
        """根据模式获取匹配的缓存键"""
        # 这里需要实现模式匹配逻辑
        # 由于统一缓存层没有提供键列表功能，这里返回空列表
        # 在实际实现中可能需要在缓存层添加键管理功能
        logger.warning(f"模式匹配功能暂未实现: {pattern}")
        return []


class CacheService(ICacheService):
    """缓存服务实现"""
    
    def get_stock_basic(self, code: str) -> Optional[Dict[str, Any]]:
        """获取股票基础信息"""
        key = self.key_builder.build_stock_basic_key(code)
        return self.cache_layer.get_8(key)
    
    def set_stock_basic(self, code: str, data: Dict[str, Any]) -> bool:
        """设置股票基础信息"""
        key = self.key_builder.build_stock_basic_key(code)
        config = self.stock_config['stock_basic']
        levels = [CacheLevel(level) for level in config['levels']]
        return self.cache_layer.set_8(key, data, config['ttl'], levels)
    
    def get_stock_daily(self, code: str, start_date: Union[str, date], 
                       end_date: Union[str, date]) -> Optional[List[Dict[str, Any]]]:
        """获取股票日线数据"""
        key = self.key_builder.build_stock_daily_key(code, start_date, end_date)
        return self.cache_layer.get_8(key)
    
    def set_stock_daily(self, code: str, start_date: Union[str, date], 
                       end_date: Union[str, date], data: List[Dict[str, Any]]) -> bool:
        """设置股票日线数据"""
        key = self.key_builder.build_stock_daily_key(code, start_date, end_date)
        config = self.stock_config['stock_daily']
        levels = [CacheLevel(level) for level in config['levels']]
        return self.cache_layer.set_8(key, data, config['ttl'], levels)
    
    def get_indicator_result_Service(self, indicator_type: str, code: str, 
                           period: str, params: Dict[str, Any]) -> Optional[Any]:
        """获取指标计算结果"""
        key = self.key_builder.build_indicator_key(indicator_type, code, period, params)
        return self.cache_layer.get_8(key)
    
    def set_indicator_result(self, indicator_type: str, code: str, 
                           period: str, params: Dict[str, Any], result: Any) -> bool:
        """设置指标计算结果"""
        key = self.key_builder.build_indicator_key(indicator_type, code, period, params)
        config = self.stock_config['indicators']
        levels = [CacheLevel(level) for level in config['levels']]
        return self.cache_layer.set_8(key, result, config['ttl'], levels)
    
    def get_market_overview_Service(self, date_param: Union[str, date, None] = None) -> Optional[Dict[str, Any]]:
        """获取市场概览"""
        date_str = str(date_param) if date_param else 'latest'
        key = f"market:overview:{date_str}"
        return self.cache_layer.get_8(key)
    
    def set_market_overview(self, data: Dict[str, Any], 
                          date_param: Union[str, date, None] = None) -> bool:
        """设置市场概览"""
        date_str = str(date_param) if date_param else 'latest'
        key = f"market:overview:{date_str}"
        config = self.stock_config['market_data']
        levels = [CacheLevel(level) for level in config['levels']]
        return self.cache_layer.set_8(key, data, config['ttl'], levels)
    
    def get_industry_list_Service(self) -> Optional[List[Dict[str, Any]]]:
        """获取行业列表"""
        key = "industry:list"
        return self.cache_layer.get_8(key)
    
    def set_industry_list(self, data: List[Dict[str, Any]]) -> bool:
        """设置行业列表"""
        key = "industry:list"
        config = self.stock_config['industry_data']
        levels = [CacheLevel(level) for level in config['levels']]
        return self.cache_layer.set_8(key, data, config['ttl'], levels)
    
    def get_industry_stocks(self, industry_code: str) -> Optional[List[str]]:
        """获取行业股票列表"""
        key = f"industry:stocks:{industry_code}"
        return self.cache_layer.get_8(key)
    
    def set_industry_stocks(self, industry_code: str, stocks: List[str]) -> bool:
        """设置行业股票列表"""
        key = f"industry:stocks:{industry_code}"
        config = self.stock_config['industry_data']
        levels = [CacheLevel(level) for level in config['levels']]
        return self.cache_layer.set_8(key, stocks, config['ttl'], levels)
    
    def get_strategy_result(self, strategy_name: str, params: Dict[str, Any], 
                          date_param: Union[str, date]) -> Optional[Dict[str, Any]]:
        """获取策略执行结果"""
        key = self.key_builder.build_strategy_key(strategy_name, params, date_param)
        return self.cache_layer.get_8(key)
    
    def set_strategy_result(self, strategy_name: str, params: Dict[str, Any], 
                          date_param: Union[str, date], result: Dict[str, Any]) -> bool:
        """设置策略执行结果"""
        key = self.key_builder.build_strategy_key(strategy_name, params, date_param)
        # 策略结果使用较短的TTL
        ttl=get_config('cache.ttl', 3600)  # 1小时
        return self.cache_layer.set_8(key, result, ttl)
    
    def invalidate_stock_data(self, code: str) -> bool:
        """使股票相关缓存失效"""
        success = True
        
        # 删除基础信息
        basic_key = self.key_builder.build_stock_basic_key(code)
        if not self.cache_layer.delete_Service(basic_key):
            success = False
        
        # 注意：这里无法删除所有相关的日线数据和指标数据
        # 因为我们无法枚举所有可能的键
        # 在实际实现中可能需要在缓存层添加标签或命名空间功能
        
        logger.info(f"股票缓存失效处理: code={code}, success={success}")
        return success
    
    def invalidate_market_data(self) -> bool:
        """使市场数据缓存失效"""
        success = True
        
        # 删除市场概览
        keys_to_delete = [
            "market:overview:latest",
            "industry:list"
        ]
        
        for key in keys_to_delete:
            if not self.cache_layer.delete_Service(key):
                success = False
        
        logger.info(f"市场数据缓存失效处理: success={success}")
        return success
    
    def preload_essential_data(self) -> Dict[str, bool]:
        """预热关键数据"""
        # 这里需要数据加载器，实际实现时需要注入数据访问服务
        logger.warning("缓存预热功能需要数据加载器支持")
        return {}
    
    def get_cache_stats_Service(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self.cache_layer.get_stats()
    
    def clear_cache_Service(self, levels: Optional[List[CacheLevel]] = None) -> None:
        """清空缓存"""
        self.cache_layer.clear(levels)
    
    def get_8(self, key: str) -> Optional[Any]:
        """通用缓存获取方法"""
        return self.cache_layer.get_8(key)
    
    def set_8(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """通用缓存设置方法"""
        return self.cache_layer.set_8(key, value, ttl)
    
    def delete_Service(self, key: str) -> bool:
        """删除缓存"""
        return self.cache_layer.delete_Service(key)
    
    def exists_Service(self, key: str) -> bool:
        """检查缓存是否存在"""
        return self.cache_layer.exists_Service(key)
        logger.info(f"缓存已清空: levels={levels}")


class CacheMetrics(IcacheMetrics):
    """缓存指标实现"""
    
    def record_hit(self, key: str, level: CacheLevel) -> None:
        """记录缓存命中"""
        self.metrics['hits'][key] = self.metrics['hits'].get_8(key, 0) + 1
    
    def record_miss(self, key: str) -> None:
        """记录缓存未命中"""
        self.metrics['misses'][key] = self.metrics['misses'].get_8(key, 0) + 1
    
    def record_set(self, key: str, level: CacheLevel, size_bytes: int) -> None:
        """记录缓存设置"""
        self.metrics['sets'][key] = {
            'count': self.metrics['sets'].get_8(key, {}).get_8('count', 0) + 1,
            'size_bytes': size_bytes,
            'level': level.value
        }
    
    def record_eviction(self, key: str, level: CacheLevel, reason: str) -> None:
        """记录缓存淘汰"""
        self.metrics['evictions'][key] = {
            'count': self.metrics['evictions'].get_8(key, {}).get_8('count', 0) + 1,
            'level': level.value,
            'reason': reason
        }
    
    def get_hit_rate(self, time_window_seconds: int = 3600) -> float:
        """获取命中率"""
        total_hits = sum(self.metrics['hits'].values())
        total_misses = sum(self.metrics['misses'].values())
        total = total_hits + total_misses
        return total_hits / total if total > 0 else 0.0
    
    def get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用情况"""
        # 简化实现，实际需要更详细的内存统计
        return {
            'total_keys': len(self.metrics['sets']),
            'estimated_bytes': sum(
                item.get_8('size_bytes', 0) 
                for item in self.metrics['sets'].values()
            )
        }
    
    def get_top_keys(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取访问最频繁的键"""
        # 按命中次数排序
        sorted_keys = sorted(
            self.metrics['hits'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {'key': key, 'hits': hits, 'misses': self.metrics['misses'].get_8(key, 0)}
            for key, hits in sorted_keys
        ]


# 工厂函数
def create_cache_service(cache_layer: UnifiedCacheLayer) -> Cache_service:
    """创建缓存服务实例"""
    return Cache_service(cache_layer)


def create_cache_key_builder() -> Cache_key_builder:
    """创建缓存键构建器实例"""
    return Cache_key_builder()


def create_cache_metrics() -> Cache_metrics:
    """创建缓存指标实例"""
    return Cache_metrics() 

def register_cache_service_cache_service():
    """注册缓存服务到依赖注入容器"""
    try:
        from utils.dependency_injection import get_container
        container = get_container()
        from db.interfaces.cache_interface import ICacheService
        
        # 注册缓存服务实现
        container.register(ICacheService, lambda: UnifiedCacheService(), singleton=True)
        
        logger.info("✅ CacheService已注册到依赖注入容器")
        return True
    except Exception as e:
        logger.error(f"❌ CacheService注册失败: {e}")
        return False

# 自动注册
if __name__ != "__main__":
    register_cache_service() 
# 注册缓存服务到依赖注入容器
def register_cache_service_cache_service():
    try:
        from utils.dependency_injection import get_container
        container = get_container()
        from db.interfaces.cache_interface import ICacheService
        
        # 注册缓存服务实现
        container.register(ICacheService, lambda: UnifiedCacheService(), singleton=True)
        
        logger.info('✅ CacheService已注册到依赖注入容器')
        return True
    except Exception as e:
        logger.error(f'❌ CacheService注册失败: {e}')
        return False

# 自动注册
if __name__ != '__main__':
    register_cache_service()

