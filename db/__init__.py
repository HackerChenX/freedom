"""
数据库模块，提供数据库访问接口

包含接口定义、具体实现和依赖注入容器
"""

# 保持向后兼容
from db.db_manager import DBManager

# 新的接口定义
from db.interfaces.data_access_interface import IDataAccess, IStockDataProvider, IMarketDataProvider
from db.interfaces.cache_interface import ICacheManager
from db.interfaces.connection_interface import IConnectionManager
from db.interfaces.indicator_calculator_interface import IIndicatorCalculator, IIndicatorFactory

# 具体实现
from db.managers.data_access_manager import DataAccessManager
from db.managers.cache_manager import CacheManager
from db.managers.connection_manager import ConnectionManager

# 依赖注入容器
from db.container import get_container, configure_container, ServiceContainer

__all__ = [
    # 向后兼容
    'DBManager',
    
    # 接口定义
    'IDataAccess',
    'IStockDataProvider', 
    'IMarketDataProvider',
    'ICacheManager',
    'IConnectionManager',
    'IIndicatorCalculator',
    'IIndicatorFactory',
    
    # 具体实现
    'DataAccessManager',
    'CacheManager',
    'ConnectionManager',
    
    # 依赖注入
    'get_container',
    'configure_container',
    'ServiceContainer'
] 