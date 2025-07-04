"""
依赖注入容器

实现IoC容器，管理系统中所有的依赖关系
"""

import threading
from typing import Dict, Any, TypeVar, Type, Optional, Callable, Union
from abc import ABC, abstractmethod
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)

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


class IServiceContainer(ABC):
    """服务容器接口"""
    
    @abstractmethod
    def register(self, 
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
            IServiceContainer: 容器实例（支持链式调用）
        """
        pass
    
    @abstractmethod
    def resolve(self, service_type: Type[T]) -> T:
        """
        解析服务
        
        Args:
            service_type: 服务类型
            
        Returns:
            T: 服务实例
        """
        pass
    
    @abstractmethod
    def is_registered(self, service_type: Type[T]) -> bool:
        """
        检查服务是否已注册
        
        Args:
            service_type: 服务类型
            
        Returns:
            bool: 已注册返回True
        """
        pass


class ServiceContainer(IServiceContainer):
    """
    服务容器实现
    
    实现依赖注入容器，管理服务的注册和解析
    """
    
    def __init__(self):
        """初始化服务容器"""
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = threading.RLock()
        
        logger.info("服务容器初始化完成")
    
    def register(self, 
                service_type: Type[T], 
                implementation_type: Optional[Type[T]] = None,
                factory: Optional[Callable[[], T]] = None,
                instance: Optional[T] = None,
                lifecycle: LifecycleType = LifecycleType.TRANSIENT) -> 'ServiceContainer':
        """
        注册服务
        
        Args:
            service_type: 服务接口类型
            implementation_type: 实现类型
            factory: 工厂方法
            instance: 实例对象
            lifecycle: 生命周期类型
            
        Returns:
            ServiceContainer: 容器实例（支持链式调用）
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type,
                factory=factory,
                instance=instance,
                lifecycle=lifecycle
            )
            
            self._services[service_type] = descriptor
            
            # 如果是单例且提供了实例，直接存储
            if lifecycle == LifecycleType.SINGLETON and instance is not None:
                self._singletons[service_type] = instance
            
            logger.debug(f"服务已注册: {service_type.__name__} -> "
                        f"{implementation_type.__name__ if implementation_type else 'factory/instance'} "
                        f"({lifecycle.value})")
            
            return self
    
    def register_singleton(self, 
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
            ServiceContainer: 容器实例
        """
        return self.register(service_type, implementation_type, factory, instance, LifecycleType.SINGLETON)
    
    def register_transient(self, 
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
            ServiceContainer: 容器实例
        """
        return self.register(service_type, implementation_type, factory, None, LifecycleType.TRANSIENT)
    
    def resolve(self, service_type: Type[T]) -> T:
        """
        解析服务
        
        Args:
            service_type: 服务类型
            
        Returns:
            T: 服务实例
            
        Raises:
            ValueError: 服务未注册时抛出
        """
        with self._lock:
            if not self.is_registered(service_type):
                raise ValueError(f"服务未注册: {service_type.__name__}")
            
            descriptor = self._services[service_type]
            
            # 单例模式
            if descriptor.lifecycle == LifecycleType.SINGLETON:
                if service_type in self._singletons:
                    return self._singletons[service_type]
                
                # 创建单例实例
                instance = self._create_instance(descriptor)
                self._singletons[service_type] = instance
                return instance
            
            # 瞬态模式
            elif descriptor.lifecycle == LifecycleType.TRANSIENT:
                return self._create_instance(descriptor)
            
            # 作用域模式（暂时按瞬态处理）
            else:
                return self._create_instance(descriptor)
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
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
    
    def is_registered(self, service_type: Type[T]) -> bool:
        """
        检查服务是否已注册
        
        Args:
            service_type: 服务类型
            
        Returns:
            bool: 已注册返回True
        """
        return service_type in self._services
    
    def get_registered_services(self) -> Dict[Type, ServiceDescriptor]:
        """
        获取所有已注册的服务
        
        Returns:
            Dict[Type, ServiceDescriptor]: 已注册的服务字典
        """
        return self._services.copy()
    
    def clear(self) -> None:
        """清空容器"""
        with self._lock:
            self._services.clear()
            self._singletons.clear()
            logger.info("服务容器已清空")


# 全局容器实例
_container: Optional[ServiceContainer] = None
_container_lock = threading.Lock()


def get_container() -> ServiceContainer:
    """
    获取全局容器实例
    
    Returns:
        ServiceContainer: 容器实例
    """
    global _container
    
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = ServiceContainer()
                _setup_default_services(_container)
    
    return _container


def _setup_default_services(container: ServiceContainer) -> None:
    """
    设置默认服务
    
    Args:
        container: 服务容器
    """
    logger.info("正在设置默认服务...")
    
    try:
        # 延迟导入避免循环依赖
        from db.interfaces.data_access_interface import IDataAccess
        from db.interfaces.cache_interface import ICacheManager
        from db.interfaces.connection_interface import IConnectionManager
        from db.managers.data_access_manager import DataAccessManager
        from db.managers.cache_manager import CacheManager
        from db.managers.connection_manager import ConnectionManager
        
        # 注册核心服务
        container.register_singleton(
            ICacheManager,
            CacheManager
        )
        
        container.register_singleton(
            IConnectionManager,
            ConnectionManager
        )
        
        container.register_singleton(
            IDataAccess,
            DataAccessManager
        )
        
        logger.info("默认服务设置完成")
        
    except ImportError as e:
        logger.warning(f"设置默认服务时出现导入错误: {e}")
    except Exception as e:
        logger.error(f"设置默认服务失败: {e}")


def configure_container() -> ServiceContainer:
    """
    配置服务容器
    
    Returns:
        ServiceContainer: 配置好的容器实例
    """
    container = get_container()
    
    try:
        # 延迟导入避免循环依赖
        from db.interfaces.indicator_calculator_interface import IIndicatorFactory
        
        # 注册指标相关服务
        # 这将在后续的指标系统实现中完成
        
        logger.info("服务容器配置完成")
        
    except ImportError as e:
        logger.warning(f"配置服务容器时出现导入错误: {e}")
    except Exception as e:
        logger.error(f"配置服务容器失败: {e}")
    
    return container


def reset_container() -> None:
    """重置容器（主要用于测试）"""
    global _container
    with _container_lock:
        if _container:
            _container.clear()
        _container = None
    logger.info("服务容器已重置") 