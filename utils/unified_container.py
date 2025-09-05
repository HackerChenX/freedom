"""
统一依赖注入容器
严格按照六层架构规范，提供标准化的服务注册和解析
"""

import threading
from typing import Dict, Any, TypeVar, Type, Optional, Callable, Union
from abc import ABC, abstractmethod
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifecycle(Enum):
    """服务生命周期类型"""
    SINGLETON = "singleton"  # 单例
    TRANSIENT = "transient"  # 瞬态
    SCOPED = "scoped"       # 作用域


class ServiceRegistration:
    """服务注册信息"""
    
    def __init__(self, 
                 service_type: Type[T],
                 implementation_type: Optional[Type[T]] = None,
                 factory: Optional[Callable[[], T]] = None,
                 instance: Optional[T] = None,
                 lifecycle: ServiceLifecycle = ServiceLifecycle.TRANSIENT):
        self.service_type = service_type
        self.implementation_type = implementation_type or service_type
        self.factory = factory
        self.instance = instance
        self.lifecycle = lifecycle
        self.created_instance = None


class IServiceContainer(ABC):
    """服务容器接口"""
    
    @abstractmethod
    def register(self, 
                service_type: Type[T], 
                implementation_type: Optional[Type[T]] = None,
                factory: Optional[Callable[[], T]] = None,
                instance: Optional[T] = None,
                lifecycle: ServiceLifecycle = ServiceLifecycle.TRANSIENT) -> 'IServiceContainer':
        """注册服务"""
        pass
    
    @abstractmethod
    def register_singleton(self, 
                          service_type: Type[T], 
                          implementation_type: Optional[Type[T]] = None,
                          factory: Optional[Callable[[], T]] = None,
                          instance: Optional[T] = None) -> 'IServiceContainer':
        """注册单例服务"""
        pass
    
    @abstractmethod
    def resolve(self, service_type: Type[T]) -> T:
        """解析服务"""
        pass
    
    @abstractmethod
    def is_registered(self, service_type: Type[T]) -> bool:
        """检查服务是否已注册"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空容器"""
        pass


class UnifiedServiceContainer(IServiceContainer):
    """
    统一服务容器实现
    
    严格按照六层架构规范，提供标准化的依赖注入功能
    """
    
    def __init__(self):
        self._services: Dict[Type, ServiceRegistration] = {}
        self._lock = threading.RLock()
        self._building_stack = set()  # 防止循环依赖
    
    def register(self, 
                service_type: Type[T], 
                implementation_type: Optional[Type[T]] = None,
                factory: Optional[Callable[[], T]] = None,
                instance: Optional[T] = None,
                lifecycle: ServiceLifecycle = ServiceLifecycle.TRANSIENT) -> 'UnifiedServiceContainer':
        """
        注册服务
        
        Args:
            service_type: 服务接口类型
            implementation_type: 实现类型
            factory: 工厂方法
            instance: 实例对象
            lifecycle: 生命周期类型
            
        Returns:
            UnifiedServiceContainer: 容器实例（支持链式调用）
        """
        with self._lock:
            registration = ServiceRegistration(
                service_type=service_type,
                implementation_type=implementation_type,
                factory=factory,
                instance=instance,
                lifecycle=lifecycle
            )
            
            self._services[service_type] = registration
            
            logger.debug(f"服务 {service_type.__name__} 注册成功 (lifecycle: {lifecycle.value})")
            
        return self
    
    def register_singleton(self, 
                          service_type: Type[T], 
                          implementation_type: Optional[Type[T]] = None,
                          factory: Optional[Callable[[], T]] = None,
                          instance: Optional[T] = None) -> 'UnifiedServiceContainer':
        """
        注册单例服务
        
        Args:
            service_type: 服务接口类型
            implementation_type: 实现类型
            factory: 工厂方法
            instance: 实例对象
            
        Returns:
            UnifiedServiceContainer: 容器实例
        """
        return self.register(service_type, implementation_type, factory, instance, ServiceLifecycle.SINGLETON)
    
    def register_transient(self, 
                          service_type: Type[T], 
                          implementation_type: Optional[Type[T]] = None,
                          factory: Optional[Callable[[], T]] = None) -> 'UnifiedServiceContainer':
        """
        注册瞬态服务
        
        Args:
            service_type: 服务接口类型
            implementation_type: 实现类型
            factory: 工厂方法
            
        Returns:
            UnifiedServiceContainer: 容器实例
        """
        return self.register(service_type, implementation_type, factory, None, ServiceLifecycle.TRANSIENT)
    
    def resolve(self, service_type: Type[T]) -> T:
        """
        解析服务
        
        Args:
            service_type: 服务类型
            
        Returns:
            T: 服务实例
            
        Raises:
            ValueError: 服务未注册
            RuntimeError: 循环依赖
        """
        with self._lock:
            if service_type not in self._services:
                raise ValueError(f"服务 {service_type.__name__} 未注册")
            
            # 检查循环依赖
            if service_type in self._building_stack:
                raise RuntimeError(f"检测到循环依赖: {service_type.__name__}")
            
            registration = self._services[service_type]
            
            # 如果是单例且已创建，直接返回
            if (registration.lifecycle == ServiceLifecycle.SINGLETON and 
                registration.created_instance is not None):
                return registration.created_instance
            
            # 如果提供了实例，直接返回
            if registration.instance is not None:
                if registration.lifecycle == ServiceLifecycle.SINGLETON:
                    registration.created_instance = registration.instance
                return registration.instance
            
            # 创建实例
            self._building_stack.add(service_type)
            try:
                if registration.factory:
                    instance = registration.factory()
                else:
                    instance = registration.implementation_type()
                
                # 如果是单例，缓存实例
                if registration.lifecycle == ServiceLifecycle.SINGLETON:
                    registration.created_instance = instance
                
                logger.debug(f"服务 {service_type.__name__} 实例创建成功")
                return instance
                
            finally:
                self._building_stack.discard(service_type)
    
    def is_registered(self, service_type: Type[T]) -> bool:
        """
        检查服务是否已注册
        
        Args:
            service_type: 服务类型
            
        Returns:
            bool: 已注册返回True
        """
        with self._lock:
            return service_type in self._services
    
    def clear(self) -> None:
        """清空容器"""
        with self._lock:
            self._services.clear()
            self._building_stack.clear()
            logger.debug("服务容器已清空")
    
    def get_registered_services(self) -> Dict[str, str]:
        """
        获取已注册的服务列表
        
        Returns:
            Dict[str, str]: 服务名称到生命周期的映射
        """
        with self._lock:
            return {
                service_type.__name__: registration.lifecycle.value
                for service_type, registration in self._services.items()
            }


# 全局容器实例
_container: Optional[UnifiedServiceContainer] = None
_container_lock = threading.Lock()


def get_container() -> UnifiedServiceContainer:
    """
    获取全局容器实例
    
    Returns:
        UnifiedServiceContainer: 容器实例
    """
    global _container
    
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = UnifiedServiceContainer()
                logger.info("统一服务容器已创建")
    
    return _container


def reset_container() -> None:
    """重置容器（主要用于测试）"""
    global _container
    with _container_lock:
        if _container:
            _container.clear()
        _container = None
    logger.info("服务容器已重置")


# 便捷函数
def register_service(service_type: Type[T], 
                    implementation_type: Optional[Type[T]] = None,
                    factory: Optional[Callable[[], T]] = None,
                    instance: Optional[T] = None,
                    singleton: bool = False) -> None:
    """
    便捷的服务注册函数
    
    Args:
        service_type: 服务接口类型
        implementation_type: 实现类型
        factory: 工厂方法
        instance: 实例对象
        singleton: 是否单例
    """
    container = get_container()
    if singleton:
        container.register_singleton(service_type, implementation_type, factory, instance)
    else:
        container.register_transient(service_type, implementation_type, factory)


def resolve_service(service_type: Type[T]) -> T:
    """
    便捷的服务解析函数
    
    Args:
        service_type: 服务类型
        
    Returns:
        T: 服务实例
    """
    container = get_container()
    return container.resolve(service_type)


def is_service_registered(service_type: Type[T]) -> bool:
    """
    检查服务是否已注册
    
    Args:
        service_type: 服务类型
        
    Returns:
        bool: 已注册返回True
    """
    container = get_container()
    return container.is_registered(service_type)


# 导出主要类和函数
__all__ = [
    'IServiceContainer',
    'UnifiedServiceContainer', 
    'ServiceLifecycle',
    'get_container',
    'reset_container',
    'register_service',
    'resolve_service',
    'is_service_registered'
]
