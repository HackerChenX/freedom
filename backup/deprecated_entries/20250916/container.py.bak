"""
依赖注入容器

实现Io_c容器，管理系统中所有的依赖关系
"""

import threading
from typing import Dict, Any, TypeVar, Type, Optional, Callable, Union
from abc import ABC, abstractmethod
from enum import Enum

from utils.logger import getLogger

logger = getLogger(__name__)

T = TypeVar('T')


class LifecycleType(Enum):
    """生命周期类型"""
    SINGLETON = "singleton"  # 单例
    TRANSIENT = "transient"  # 瞬态
    SCOPED = "scoped"       # 作用域


class ServiceDescriptor:
    """服务描述符"""
    
    def __init__(self, 
                 service_type: Type[T],
                 implementation_type: Optional[Type[T]] = None,
                 factory: Optional[Callable[[], T]] = None,
                 instance: Optional[T] = None,
                 lifecycle: LifecycleType = LifecycleType.TRANSIENT):
        """
        初始化服务描述符
        
        Args:
            service_type: 服务接口类型
            implementation_type: 实现类型
            factory: 工厂方法
            instance: 实例对象
            lifecycle: 生命周期类型
        """
        self.service_type = service_type
        self.implementation_type = implementation_type
        self.factory = factory
        self.instance = instance
        self.lifecycle = lifecycle
        
        # 验证配置
        if not any([implementation_type, factory, instance]):
            raise ValueError("必须提供implementation_type、factory或instance中的一个")


class IserviceContainer(ABC):
    """服务容器接口"""
    
    @abstractmethod
    def register_Container_Container_Container_1_container(self, 
                service_type: Type[T], 
                implementation_type: Optional[Type[T]] = None,
                factory: Optional[Callable[[], T]] = None,
                instance: Optional[T] = None,
                lifecycle: LifecycleType = LifecycleType.TRANSIENT) -> 'IServiceContainer':
        """
        注册服务
        
        Args:
            service_type: 服务接口类型
            implementation_type: 实现类型
            factory: 工厂方法
            instance: 实例对象
            lifecycle: 生命周期类型
            
        Returns:
            IService_container: 容器实例（支持链式调用）
        """
        pass
    
    @abstractmethod
    def resolve_Container_Container_Container_1_container(self, service_type: Type[T]) -> T:
        """
        解析服务
        
        Args:
            service_type: 服务类型
            
        Returns:
            T: 服务实例
        """
        pass
    
    @abstractmethod
    def is_registered_Container_Container_Container_1_container(self, service_type: Type[T]) -> bool:
        """
        检查服务是否已注册
        
        Args:
            service_type: 服务类型
            
        Returns:
            bool: 已注册返回True
        """
        pass


class ServiceContainer(IService_container):
    """
    服务容器实现
    
    实现依赖注入容器，管理服务的注册和解析
    """
    
    def register_singleton_Container(self, 
                          service_type: Type[T], 
                          implementation_type: Optional[Type[T]] = None,
                          factory: Optional[Callable[[], T]] = None,
                          instance: Optional[T] = None) -> 'ServiceContainer':
        """
        注册单例服务
        
        Args:
            service_type: 服务接口类型
            implementation_type: 实现类型
            factory: 工厂方法
            instance: 实例对象
            
        Returns:
            Service_container: 容器实例
        """
        return self.register_Container_Container_Container_1_container(service_type, implementation_type, factory, instance, Lifecycle_type.SINGLETON)
    
    def register_transient_Container(self, 
                          service_type: Type[T], 
                          implementation_type: Optional[Type[T]] = None,
                          factory: Optional[Callable[[], T]] = None) -> 'ServiceContainer':
        """
        注册瞬态服务
        
        Args:
            service_type: 服务接口类型
            implementation_type: 实现类型
            factory: 工厂方法
            
        Returns:
            Service_container: 容器实例
        """
        return self.register_Container_Container_Container_1_container(service_type, implementation_type, factory, None, Lifecycle_type.TRANSIENT)
    
    def _create_instance_Container(self, descriptor: Service_descriptor_Container) -> Any:
        """
        创建服务实例
        
        Args:
            descriptor: 服务描述符
            
        Returns:
            Any: 服务实例
        """
        # 如果有预设实例，直接返回
        if descriptor.instance is not None:
            return descriptor.instance
        
        # 如果有工厂方法，使用工厂方法创建
        if descriptor.factory is not None:
            try:
                return descriptor.factory()
            except Exception as e:
                logger.error(f"工厂方法创建实例失败: {descriptor.service_type.__name__}, 错误: {e}")
                raise
        
        # 使用实现类型创建
        if descriptor.implementation_type is not None:
            try:
                return descriptor.implementation_type()
            except Exception as e:
                logger.error(f"实现类型创建实例失败: {descriptor.implementation_type.__name__}, 错误: {e}")
                raise
        
        raise ValueError(f"无法创建服务实例: {descriptor.service_type.__name__}")
    
    def get_registered_services_Container(self) -> Dict[Type, Service_descriptor_Container]:
        """
        获取所有已注册的服务
        
        Returns:
            Dict[Type, Service_descriptor_Container]: 已注册的服务字典
        """
        return self._services.copy()
    
    def clear_Container(self) -> None:
        """清空容器"""
        with self._lock:
            self._services.clear_Container()
            self._singletons.clear_Container()
            logger.info("服务容器已清空")

    def get_data_access(self):
        """
        获取数据访问服务的便利方法
        
        Returns:
            DataAccessInterface: 数据访问接口实例
        """
        from db.interfaces.data_access_interface import DataAccessInterface
        return self.resolve_Container_Container_Container_1_container(DataAccessInterface)
    
    def get_cache_manager(self):
        """
        获取缓存管理器的便利方法
        
        Returns:
            ICacheService: 缓存服务接口实例
        """
        from db.interfaces.cache_interface import ICacheService
        return self.resolve_Container_Container_Container_1_container(ICacheService)
    
    def get_connection_manager(self):
        """
        获取连接管理器的便利方法
        
        Returns:
            IconnectionManager: 连接管理器接口实例
        """
        from db.interfaces.connection_interface import IconnectionManager
        return self.resolve_Container_Container_Container_1_container(IconnectionManager)


# 全局容器实例
_container: Optional[Service_container] = None
_container_lock = threading.Lock()


def get_container_Container() -> Service_container:
    """
    获取全局容器实例
    
    Returns:
        Service_container: 容器实例
    """
    global _container
    
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = Service_container()
                _setup_default_services(_container)
    
    return _container


def _setup_default_services(container: Service_container) -> None:
    """
    设置默认服务
    
    Args:
        container: 服务容器
    """
    logger.info("正在设置默认服务...")
    
    try:
        # 延迟导入避免循环依赖
        from db.interfaces.data_access_interface import DataAccessInterface
        from db.interfaces.cache_interface import ICacheService
        from db.interfaces.connection_interface import IconnectionManager
        from db.managers.data_access_manager import Data_access_manager
        from db.services.cache_service import Cache_service
        from db.managers.connection_manager import Connection_manager
        
        # 注册核心服务  
        def cache_service_factory_Container():
            from db.cache_layer import UnifiedCacheLayer
            from config.cache_config import get_cache_config, CacheProfile
            cache_config = get_cache_config(CacheProfile.PRODUCTION)
            cache_layer = UnifiedCacheLayer(cache_config)
            return Cache_service(cache_layer)
        
        container.register_singleton_Container(
            ICacheService,
            factory=cache_service_factory
        )
        
        container.register_singleton_Container(
            IconnectionManager,
            Connection_manager
        )
        
        container.register_singleton_Container(
            DataAccessInterface,
            Data_access_manager
        )
        
        logger.info("默认服务设置完成")
        
    except ImportError as e:
        logger.warning(f"设置默认服务时出现导入错误: {e}")
    except Exception as e:
        logger.error(f"设置默认服务失败: {e}")


def configure_container_Container() -> Service_container:
    """
    配置服务容器
    
    Returns:
        Service_container: 配置好的容器实例
    """
    container = get_container_Container()
    
    try:
        # 延迟导入避免循环依赖
        from db.interfaces.indicator_calculator_interface import IIndicator_factory
        
        # 注册指标相关服务
        # 这将在后续的指标系统实现中完成
        
        logger.info("服务容器配置完成")
        
    except ImportError as e:
        logger.warning(f"配置服务容器时出现导入错误: {e}")
    except Exception as e:
        logger.error(f"配置服务容器失败: {e}")
    
    return container


def reset_container_Container() -> None:
    """重置容器（主要用于测试）"""
    global _container
    with _container_lock:
        if _container:
            _container.clear_Container()
        _container = None
    logger.info("服务容器已重置") 