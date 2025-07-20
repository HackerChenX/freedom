"""
业务应用层服务注册配置

负责注册L5（业务应用层）的服务
"""

import logging
from typing import TYPE_CHECKING

from utils.dependency_injection import ServiceContainer, get_container

if TYPE_CHECKING:
    from analysis.integration.unified_data_adapter import UnifiedDataAdapter
    from analysis.integration.unified_analysis_engine import UnifiedAnalysisEngine

logger = logging.getLogger(__name__)


def register_business_services(container: ServiceContainer = None) -> ServiceContainer:
    """
    注册业务应用层服务（L5）
    
    Args:
        container: 服务容器
        
    Returns:
        ServiceContainer: 配置后的容器
    """
    if container is None:
        container = get_container()
    
    try:
        logger.debug("注册业务应用层服务...")
        
        # 注册统一数据适配器
        from analysis.integration.unified_data_adapter import UnifiedDataAdapter
        container.register_singleton(
            UnifiedDataAdapter,
            factory=lambda: UnifiedDataAdapter(
                data_access=_get_data_access_from_container(container),
                cache=_get_memory_cache_from_container(container)
            )
        )
        
        # 注册统一分析引擎
        from analysis.integration.unified_analysis_engine import UnifiedAnalysisEngine
        container.register_singleton(
            UnifiedAnalysisEngine,
            factory=lambda: UnifiedAnalysisEngine(
                data_adapter=container.resolve(UnifiedDataAdapter),
                period_manager=_get_period_manager_from_container(container)
            )
        )
        
        logger.debug("业务应用层服务注册完成")
        
    except ImportError as e:
        logger.warning(f"部分业务应用服务不可用: {e}")
    except Exception as e:
        logger.error(f"注册业务应用层服务失败: {e}")
    
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


def _get_memory_cache_from_container(container: ServiceContainer):
    """从容器获取内存缓存"""
    try:
        from utils.cache import MemoryCache
        if container.is_registered(MemoryCache):
            return container.resolve(MemoryCache)
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"获取内存缓存失败: {e}")
    
    return None


def _get_period_manager_from_container(container: ServiceContainer):
    """从容器获取周期管理器"""
    try:
        from utils.period_manager import PeriodManager
        if container.is_registered(PeriodManager):
            return container.resolve(PeriodManager)
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"获取周期管理器失败: {e}")
    
    return None


def configure_business_layer(container: ServiceContainer = None) -> ServiceContainer:
    """
    配置业务应用层
    
    Args:
        container: 服务容器
        
    Returns:
        ServiceContainer: 配置后的容器
    """
    if container is None:
        container = get_container()
    
    try:
        # 注册业务应用服务
        register_business_services(container)
        
        logger.info("业务应用层配置完成")
        
    except Exception as e:
        logger.error(f"业务应用层配置失败: {e}")
        raise
    
    return container


def get_unified_data_adapter(container: ServiceContainer = None):
    """
    获取统一数据适配器
    
    Args:
        container: 服务容器
        
    Returns:
        UnifiedDataAdapter: 统一数据适配器实例
    """
    if container is None:
        container = get_container()
    
    try:
        from analysis.integration.unified_data_adapter import UnifiedDataAdapter
        if container.is_registered(UnifiedDataAdapter):
            return container.resolve(UnifiedDataAdapter)
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"获取统一数据适配器失败: {e}")
    
    return None


def get_unified_analysis_engine(container: ServiceContainer = None):
    """
    获取统一分析引擎
    
    Args:
        container: 服务容器
        
    Returns:
        UnifiedAnalysisEngine: 统一分析引擎实例
    """
    if container is None:
        container = get_container()
    
    try:
        from analysis.integration.unified_analysis_engine import UnifiedAnalysisEngine
        if container.is_registered(UnifiedAnalysisEngine):
            return container.resolve(UnifiedAnalysisEngine)
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"获取统一分析引擎失败: {e}")
    
    return None 