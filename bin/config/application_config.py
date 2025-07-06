"""
应用级配置器

负责按照分层架构原则配置所有服务
"""

import logging
from typing import Dict, Any

from utils.dependency_injection import ServiceContainer, get_container

logger = logging.getLogger(__name__)


def configure_application(container: ServiceContainer = None) -> ServiceContainer:
    """
    配置完整的应用服务
    
    按照分层架构原则，从底层到高层依次配置：
    L1 基础设施层 -> L2 存储访问层 -> L3 数据服务层 -> L4 核心服务层 -> L5 业务应用层
    
    Args:
        container: 服务容器
        
    Returns:
        ServiceContainer: 配置后的容器
    """
    if container is None:
        container = get_container()
    
    try:
        logger.info("开始配置应用服务...")
        
        # L1 基础设施层
        from config.container_config import configure_base_infrastructure
        configure_base_infrastructure()
        
        # L2/L3 数据层
        from db.service_registry import configure_data_layer
        configure_data_layer(container)
        
        # L4 核心服务层
        from indicators.service_registry import configure_core_layer
        configure_core_layer(container)
        
        # L5 业务应用层
        from analysis.service_registry import configure_business_layer
        configure_business_layer(container)
        
        logger.info("应用服务配置完成")
        
    except Exception as e:
        logger.error(f"应用服务配置失败: {e}")
        raise
    
    return container


def configure_test_application(container: ServiceContainer = None) -> ServiceContainer:
    """
    配置测试应用服务
    
    Args:
        container: 服务容器
        
    Returns:
        ServiceContainer: 配置后的容器
    """
    if container is None:
        container = get_container()
    
    try:
        logger.info("开始配置测试应用服务...")
        
        # 清空现有配置
        container.clear()
        
        # 配置测试用的基础服务
        from utils.cache import MemoryCache
        container.register_singleton(
            MemoryCache,
            factory=lambda: MemoryCache()
        )
        
        from utils.period_manager import PeriodManager
        container.register_singleton(
            PeriodManager,
            factory=lambda: PeriodManager(cache_size=10)
        )
        
        logger.info("测试应用服务配置完成")
        
    except Exception as e:
        logger.error(f"测试应用服务配置失败: {e}")
        raise
    
    return container


def get_application_info(container: ServiceContainer = None) -> Dict[str, Any]:
    """
    获取应用配置信息
    
    Args:
        container: 服务容器
        
    Returns:
        Dict[str, Any]: 应用配置信息
    """
    if container is None:
        container = get_container()
    
    try:
        from config.container_config import get_container_info
        base_info = get_container_info(container)
        
        # 检查各层服务是否已配置
        layer_status = {
            'data_layer': _check_data_layer_configured(container),
            'core_layer': _check_core_layer_configured(container),
            'business_layer': _check_business_layer_configured(container)
        }
        
        return {
            **base_info,
            'layer_status': layer_status,
            'is_fully_configured': all(layer_status.values())
        }
        
    except Exception as e:
        logger.error(f"获取应用配置信息失败: {e}")
        return {'error': str(e)}


def _check_data_layer_configured(container: ServiceContainer) -> bool:
    """检查数据层是否已配置"""
    try:
        from db.interfaces.data_access_interface import IDataAccess
        return container.is_registered(IDataAccess)
    except ImportError:
        return False
    except Exception:
        return False


def _check_core_layer_configured(container: ServiceContainer) -> bool:
    """检查核心服务层是否已配置"""
    try:
        from utils.cache import MemoryCache
        from utils.period_manager import PeriodManager
        return (container.is_registered(MemoryCache) and 
                container.is_registered(PeriodManager))
    except ImportError:
        return False
    except Exception:
        return False


def _check_business_layer_configured(container: ServiceContainer) -> bool:
    """检查业务应用层是否已配置"""
    try:
        from analysis.integration.unified_data_adapter import UnifiedDataAdapter
        from analysis.integration.unified_analysis_engine import UnifiedAnalysisEngine
        return (container.is_registered(UnifiedDataAdapter) and 
                container.is_registered(UnifiedAnalysisEngine))
    except ImportError:
        return False
    except Exception:
        return False


def reset_application():
    """重置应用配置"""
    try:
        container = get_container()
        container.clear()
        logger.info("应用配置已重置")
    except Exception as e:
        logger.error(f"重置应用配置失败: {e}")


# 提供便捷的初始化函数
def initialize_application():
    """初始化应用"""
    try:
        container = configure_application()
        logger.info("应用初始化完成")
        return container
    except Exception as e:
        logger.error(f"应用初始化失败: {e}")
        raise


def initialize_test_application():
    """初始化测试应用"""
    try:
        container = configure_test_application()
        logger.info("测试应用初始化完成")
        return container
    except Exception as e:
        logger.error(f"测试应用初始化失败: {e}")
        raise 