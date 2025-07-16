"""
数据库接口模块初始化文件

集中导出所有数据库接口
"""

from db.interfaces.data_access_interface import DataAccessInterface
from db.interfaces.cache_interface import ICacheService
from db.interfaces.connection_interface import IconnectionManager

__all__ = [
    'DataAccessInterface',
    'ICacheService', 
    'IconnectionManager'
] 