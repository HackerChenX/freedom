"""
from utils.dependency_injection import get_config
核心服务层服务注册配置

负责注册L4(核心服务层)的服务
"""

from utils.logger import get_logger

import logging
from typing import TYPE_CHECKING

from utils.dependency_injection import ServiceContainer, get_container

if TYPE_CHECKING:
    from utils.cache import MemoryCache
    from utils.period_manager import PeriodManager
    from indicators.pattern_registry import PatternRegistry

logger = get_logger(__name__)


def register_core_services(container: ServiceContainer = None) -> ServiceContainer:
    """
    注册核心服务层服务(L4)

    Args:
        container: 服务容器

    Returns:
        ServiceContainer: 配置后的容器
    """
    if container is None:
        container = get_container()

    try:
        logger.debug("注册核心服务层服务...")

        # 注册缓存服务
        from utils.cache import MemoryCache

        container.register_singleton(MemoryCache, factory=lambda: MemoryCache())

        # 注册周期管理器
        from utils.period_manager import PeriodManager

        container.register_singleton(
            PeriodManager,
            factory=lambda: PeriodManager(
                get_config("cache.size"), data_access=_get_data_access_from_container(container)
            ),
        )

        # 注册模式注册表
        from indicators.pattern_registry import PatternRegistry

        container.register_singleton(PatternRegistry, factory=lambda: PatternRegistry())

        # 注册数据库管理器(向后兼容)
        from db.db_manager import DBManager

        container.register_singleton(
            DBManager, factory=lambda: DBManager(data_access=_get_data_access_from_container(container))
        )

        logger.debug("核心服务层服务注册完成")

    except ImportError as e:
        logger.warning(f"部分核心服务不可用: {e}")
    except Exception as e:
        logger.error(f"注册核心服务层服务失败: {e}")

    return container


def _get_data_access_from_container(container: ServiceContainer):
    """从容器获取数据访问接口"""
    try:
        from db.interfaces.data_access_interface import IDataAccess

        if container.is_registered(IDataAccess):
            return container.resolve(IDataAccess)
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"获取数据访问接口失败: {e}")

    return None


def configure_core_layer(container: ServiceContainer = None) -> ServiceContainer:
    """
    配置核心服务层

    Args:
        container: 服务容器

    Returns:
        ServiceContainer: 配置后的容器
    """
    if container is None:
        container = get_container()

    try:
        # 注册核心服务
        register_core_services(container)

        logger.info("核心服务层配置完成")

    except Exception as e:
        logger.error(f"核心服务层配置失败: {e}")
        raise

    return container


def get_memory_cache(container: ServiceContainer = None):
    """
    获取内存缓存服务

    Args:
        container: 服务容器

    Returns:
        MemoryCache: 内存缓存实例
    """
    if container is None:
        container = get_container()

    try:
        from utils.cache import MemoryCache

        if container.is_registered(MemoryCache):
            return container.resolve(MemoryCache)
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"获取内存缓存失败: {e}")

    return None


def get_period_manager(container: ServiceContainer = None):
    """
    获取周期管理器

    Args:
        container: 服务容器

    Returns:
        PeriodManager: 周期管理器实例
    """
    if container is None:
        container = get_container()

    try:
        from utils.period_manager import PeriodManager

        if container.is_registered(PeriodManager):
            return container.resolve(PeriodManager)
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"获取周期管理器失败: {e}")

    return None
