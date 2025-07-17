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
from utils.dependency_injection import get_service, configure_container, ServiceContainer

# 自动注册核心服务到依赖注入容器
def _auto_register_services():
    """自动注册数据库层服务"""
    try:
        from utils.dependency_injection import get_container
        container = get_container()
        
        # 注册DataAccessInterface
        if not container.is_registered(DataAccessInterface):
            container.register_singleton(
                DataAccessInterface,
                DataAccessManager,
                factory=lambda: DataAccessManager()
            )
            import logging
            logger = logging.getLogger(__name__)
            logger.info("✅ DataAccessInterface已注册到依赖注入容器")
            
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
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
    'ServiceContainer'
] 