"""
依赖注入容器 - 兼容性模块
提供与unified_container的兼容接口
"""

from typing import Dict, Any, TypeVar, Type, Optional, Callable
from utils.unified_container import get_container, UnifiedServiceContainer
from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class Container:
    """
    依赖注入容器 - 兼容性包装器
    提供简化的接口，兼容现有代码
    """
    
    def __init__(self):
        self._unified_container = get_container()
        self._services: Dict[str, Any] = {}
        self._setup_default_services()
    
    def _setup_default_services(self):
        """设置默认服务"""
        try:
            # 注册默认的Mock服务
            self._services["DataAccessInterface"] = MockDataAccessInterface()
            self._services["ICacheService"] = MockCacheService()
            self._services["Logger"] = logger
            
            logger.debug("默认服务注册完成")
        except Exception as e:
            logger.warning(f"设置默认服务失败: {e}")
    
    def register(self, service_name: str, service_instance: Any) -> None:
        """
        注册服务
        
        Args:
            service_name: 服务名称
            service_instance: 服务实例
        """
        self._services[service_name] = service_instance
        logger.debug(f"服务 {service_name} 注册成功")
    
    def resolve(self, service_name: str) -> Any:
        """
        解析服务
        
        Args:
            service_name: 服务名称
            
        Returns:
            Any: 服务实例
        """
        if service_name in self._services:
            return self._services[service_name]
        
        # 如果没有找到，返回None而不是抛出异常
        logger.warning(f"服务 {service_name} 未找到，返回None")
        return None
    
    def is_registered(self, service_name: str) -> bool:
        """
        检查服务是否已注册
        
        Args:
            service_name: 服务名称
            
        Returns:
            bool: 已注册返回True
        """
        return service_name in self._services
    
    def get_all_services(self) -> Dict[str, Any]:
        """获取所有已注册的服务"""
        return self._services.copy()


class MockDataAccessInterface:
    """Mock数据访问接口"""
    
    def __init__(self):
        self.name = "MockDataAccessInterface"
    
    def get_stock_data(self, *args, **kwargs):
        """Mock获取股票数据"""
        import pandas as pd
        return pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })
    
    def query(self, *args, **kwargs):
        """Mock查询方法"""
        return self.get_stock_data()


class MockCacheService:
    """Mock缓存服务"""
    
    def __init__(self):
        self.name = "MockCacheService"
        self._cache = {}
    
    def get(self, key: str) -> Any:
        """获取缓存"""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """设置缓存"""
        self._cache[key] = value
    
    def delete(self, key: str) -> None:
        """删除缓存"""
        self._cache.pop(key, None)
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()


# 创建全局容器实例
_global_container = Container()


def get_global_container() -> Container:
    """获取全局容器实例"""
    return _global_container


# 兼容性接口
container = _global_container

# 导出主要类和函数
__all__ = [
    'Container',
    'MockDataAccessInterface', 
    'MockCacheService',
    'container',
    'get_global_container'
]
