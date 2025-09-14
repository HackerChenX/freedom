"""
数据库模块，提供数据库访问接口

包含接口定义、具体实现和依赖注入容器
"""

# 保持向后兼容
from db.db_manager import DBManager

# 新的接口定义
from db.interfaces.data_access_interface import DataAccessInterface
from db.interfaces.cache_interface import ICacheService
from db.interfaces.connection_interface import IconnectionManager

# 具体实现
from db.managers.data_access_manager import DataAccessManager
from db.managers.connection_manager import ConnectionManager

# 依赖注入容器
from utils.unified_container import get_container

import logging
logger = logging.getLogger(__name__)

# 数据服务层服务注册
def register_data_services(container):
    """
    注册数据服务层服务

    Args:
        container: 依赖注入容器
    """
    try:
        # 首先注册连接管理器
        from db.enhanced_connection_pool import get_connection_pool
        connection_pool = get_connection_pool()

        # 注册数据访问接口，传入连接管理器
        container.register_singleton(
            DataAccessInterface,
            factory=lambda: DataAccessManager(connection_manager=connection_pool)
        )

        logger.info("数据服务层服务注册完成")

    except Exception as e:
        logger.error(f"数据服务层服务注册失败: {e}")


# 自动注册核心服务到依赖注入容器
def _auto_register_services():
    """自动注册数据库层服务"""
    try:
        container = get_container()

        # 使用新的注册方法
        register_data_services(container)
        logger.info("✅ 数据库层服务已注册到依赖注入容器")

    except Exception as e:
        logger.warning(f"自动注册数据库服务失败: {e}")

# 执行自动注册
_auto_register_services()

__all__ = [
    # 向后兼容
    'DBManager',
    
    # 接口定义
    'DataAccessInterface',
    'ICacheService',
    'IconnectionManager',
    
    # 具体实现
    'DataAccessManager',
    'ConnectionManager',
    
    # 依赖注入
    'get_service',
    'configure_container',
    'get_container'
] 