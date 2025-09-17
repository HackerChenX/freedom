"""
数据服务层服务注册配置

负责注册L2（存储访问层）和L3（数据服务层）的服务
"""

import logging
from typing import TYPE_CHECKING

from utils.dependency_injection import get_container

if TYPE_CHECKING:
    from db.interfaces.data_access_interface import IDataAccess
    from db.interfaces.connection_manager_interface import IConnectionManager
    from db.interfaces.cache_service_interface import ICacheService

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """简单的服务注册器"""

    def __init__(self):
        """初始化服务注册器"""
        self.container = get_container()
        self.registered_services = {}
        logger.info("服务注册器初始化完成")

    def register_service(self, name: str, service_class):
        """注册服务"""
        try:
            self.registered_services[name] = service_class
            logger.info(f"服务 {name} 注册成功")
            return True
        except Exception as e:
            logger.error(f"服务 {name} 注册失败: {e}")
            return False

    def get_service(self, name: str):
        """获取服务"""
        if name in self.registered_services:
            return self.registered_services[name]()
        return None

    def list_services(self) -> list:
        """列出所有注册的服务"""
        return list(self.registered_services.keys())


def register_storage_services(container = None):
    """
    注册存储访问层服务（L2）
    
    Args:
        container: 服务容器
        
    Returns:
        配置后的容器
    """
    if container is None:
        container = get_container()

    try:
        logger.debug("注册存储访问层服务...")
        
        # 注册ClickHouse数据库连接（通过动态导入避免直接依赖）
        def create_clickhouse_db():
            import importlib
            clickhouse_module = importlib.import_module('db.clickhouse_db')
            return clickhouse_module.ClickhouseDB()
        
        container.register_singleton_by_name(
            'clickhouse_db',
            factory=create_clickhouse_db
        )
        
        # 注册连接管理器
        from db.interfaces.connection_manager_interface import IConnectionManager
        from db.connection_manager import ConnectionManager
        
        container.register_singleton(
            IConnectionManager,
            ConnectionManager,
            factory=lambda: get_connection_pool()
        )
        
        logger.debug("存储访问层服务注册完成")
        
    except ImportError as e:
        logger.warning(f"部分存储访问服务不可用: {e}")
    except Exception as e:
        logger.error(f"注册存储访问层服务失败: {e}")
    
    return container


def register_data_services(container = None):
    """
    注册数据服务层服务（L3）
    
    Args:
        container: 服务容器
        
    Returns:
        配置后的容器
    """
    if container is None:
        container = get_container()

    try:
        logger.debug("注册数据服务层服务...")
        
        # 注册数据访问接口
        from db.interfaces.data_access_interface import IDataAccess
        from db.managers.data_access_manager import DataAccessManager
        
        container.register_singleton(
            IDataAccess,
            DataAccessManager,
            factory=lambda: DataAccessManager()
        )
        
        # 注册缓存服务接口
        from db.interfaces.cache_service_interface import ICacheService
        from db.cache_service import CacheService
        
        container.register_singleton(
            ICacheService,
            CacheService,
            factory=lambda: CacheService()
        )
        
        # 注册缓存层
        from db.services.cache_service import CacheService
        container.register_singleton(
            CacheService,
            factory=lambda: CacheService()
        )
        
        logger.debug("数据服务层服务注册完成")
        
    except ImportError as e:
        logger.warning(f"部分数据服务不可用: {e}")
    except Exception as e:
        logger.error(f"注册数据服务层服务失败: {e}")
    
    return container


def configure_data_layer(container = None):
    """
    配置完整的数据层服务（L2 + L3）
    
    Args:
        container: 服务容器
        
    Returns:
        配置后的容器
    """
    if container is None:
        container = get_container()

    try:
        # 注册存储访问层服务
        register_storage_services(container)
        
        # 注册数据服务层服务
        register_data_services(container)
        
        logger.info("数据层服务配置完成")
        
    except Exception as e:
        logger.error(f"数据层服务配置失败: {e}")
        raise
    
    return container


def get_data_access_interface(container = None):
    """
    获取数据访问接口
    
    Args:
        container: 服务容器
        
    Returns:
        IDataAccess: 数据访问接口实例
    """
    if container is None:
        container = get_container()
    
    try:
        from db.interfaces.data_access_interface import IDataAccess
        if container.is_registered(IDataAccess):
            return container.resolve(IDataAccess)
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"获取数据访问接口失败: {e}")
    
    return None 