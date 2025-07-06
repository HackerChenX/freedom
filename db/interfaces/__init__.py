"""
数据服务接口层

定义系统核心数据访问接口，实现分层架构的接口规范
"""

from db.interfaces.data_access_interface import DataAccessInterface
from db.interfaces.indicator_calculator_interface import IindicatorCalculator as IIndicatorCalculator
from db.interfaces.cache_interface import IcacheService as ICacheService
from db.interfaces.connection_interface import IConnectionManager

__all__ = [
    'DataAccessInterface',
    'IIndicatorCalculator', 
    'ICacheService',
    'IConnectionManager'
] 