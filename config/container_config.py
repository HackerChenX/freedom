"""
依赖注入容器基础配置

仅包含基础设施层的容器配置，不直接导入业务层模块
"""

import logging
from typing import TYPE_CHECKING, Any, Dict

from utils.dependency_injection import ServiceContainer, get_container

logger = logging.getLogger(__name__)


def create_base_container() -> ServiceContainer:
    """
    创建基础容器（仅包含基础设施层服务）
    
    Returns:
        ServiceContainer: 基础容器实例
    """
    container = get_container()
    
    try:
        # 清空现有配置
        container.clear()
        
        logger.info("基础容器创建完成")
        
    except Exception as e:
        logger.error(f"基础容器创建失败: {e}")
        raise
    
    return container


def register_service_factory(container: ServiceContainer, 
                           service_name: str, 
                           factory_func: callable,
                           singleton: bool = True) -> bool:
    """
    注册服务工厂函数
    
    Args:
        container: 服务容器
        service_name: 服务名称
        factory_func: 工厂函数
        singleton: 是否单例
        
    Returns:
        bool: 注册成功
    """
    try:
        if singleton:
            container.register_singleton_by_name(service_name, factory_func)
        else:
            container.register_transient_by_name(service_name, factory_func)
        
        logger.debug(f"服务 {service_name} 注册成功")
        return True
        
    except Exception as e:
        logger.error(f"服务 {service_name} 注册失败: {e}")
        return False


def get_service_by_name(container: ServiceContainer, service_name: str) -> Any:
    """
    按名称获取服务
    
    Args:
        container: 服务容器
        service_name: 服务名称
        
    Returns:
        Any: 服务实例
    """
    try:
        return container.resolve_by_name(service_name)
    except Exception as e:
        logger.error(f"获取服务 {service_name} 失败: {e}")
        return None


def reset_container():
    """重置容器配置"""
    try:
        container = get_container()
        container.clear()
        logger.info("容器已重置")
    except Exception as e:
        logger.error(f"重置容器失败: {e}")


def get_container_info(container: ServiceContainer = None) -> Dict[str, Any]:
    """
    获取容器信息
    
    Args:
        container: 服务容器
        
    Returns:
        Dict[str, Any]: 容器信息
    """
    if container is None:
        container = get_container()
    
    try:
        return {
            'registered_services': len(container._services) if hasattr(container, '_services') else 0,
            'singleton_services': len(container._singletons) if hasattr(container, '_singletons') else 0,
            'is_configured': True
        }
    except Exception as e:
        logger.error(f"获取容器信息失败: {e}")
        return {'error': str(e)}


# 基础配置函数
def configure_base_infrastructure():
    """配置基础设施层服务"""
    container = create_base_container()
    
    # 这里只配置真正的基础设施服务
    # 其他层的服务应该在各自的层中配置
    
    return container
