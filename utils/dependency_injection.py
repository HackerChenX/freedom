"""
依赖注入容器

提供统一的依赖注入管理，解决全局单例过度使用问题，
支持单例、瞬时和工厂模式的服务注册与解析。
"""

from typing import Dict, Any, Callable, TypeVar, Type, Optional, Set
from abc import ABC
import threading
from functools import wraps
import logging
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifetime:
    """服务生命周期枚举"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    FACTORY = "factory"


class ServiceDescriptor:
    """服务描述符"""
    
    def __init__(self, service_type: Type[T], implementation: Type[T] = None, 
                 factory: Callable = None, lifetime: str = ServiceLifetime.TRANSIENT):
        self.service_type = service_type
        self.implementation = implementation
        self.factory = factory
        self.lifetime = lifetime
        self.instance = None
        self.lock = threading.Lock()


class CircularDependencyError(Exception):
    """循环依赖异常"""
    pass


class ServiceNotRegisteredError(Exception):
    """服务未注册异常"""
    pass


class ServiceCreationError(Exception):
    """服务创建异常"""
    pass


class ServiceContainer:
    """优化后的依赖注入容器"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._building: Set[Type] = set()  # 循环依赖检测
        self._lock = threading.RLock()
    
    def register_singleton(self, interface: Type[T], 
                          implementation: Type[T] = None,
                          factory: Callable[[], T] = None) -> 'ServiceContainer':
        """
        注册单例服务
        
        Args:
            interface: 服务接口类型
            implementation: 实现类型
            factory: 工厂方法
            
        Returns:
            ServiceContainer: 支持链式调用
        """
        with self._lock:
            if factory:
                descriptor = ServiceDescriptor(
                    service_type=interface,
                    factory=factory,
                    lifetime=ServiceLifetime.SINGLETON
                )
            else:
                implementation = implementation or interface
                descriptor = ServiceDescriptor(
                    service_type=interface,
                    implementation=implementation,
                    lifetime=ServiceLifetime.SINGLETON
                )
            
            self._services[interface] = descriptor
            logger.debug(f"Registered singleton service: {interface.__name__}")
            
        return self
    
    def register_transient(self, interface: Type[T], 
                          implementation: Type[T] = None,
                          factory: Callable[[], T] = None) -> 'ServiceContainer':
        """
        注册瞬时服务
        
        Args:
            interface: 服务接口类型
            implementation: 实现类型
            factory: 工厂方法
            
        Returns:
            ServiceContainer: 支持链式调用
        """
        with self._lock:
            if factory:
                descriptor = ServiceDescriptor(
                    service_type=interface,
                    factory=factory,
                    lifetime=ServiceLifetime.TRANSIENT
                )
            else:
                implementation = implementation or interface
                descriptor = ServiceDescriptor(
                    service_type=interface,
                    implementation=implementation,
                    lifetime=ServiceLifetime.TRANSIENT
                )
            
            self._services[interface] = descriptor
            logger.debug(f"Registered transient service: {interface.__name__}")
            
        return self
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> 'ServiceContainer':
        """
        注册工厂方法
        
        Args:
            interface: 服务接口类型
            factory: 工厂方法
            
        Returns:
            ServiceContainer: 支持链式调用
        """
        return self.register_transient(interface, factory=factory)
    
    def register_instance(self, interface: Type[T], instance: T) -> 'ServiceContainer':
        """
        注册实例
        
        Args:
            interface: 服务接口类型
            instance: 服务实例
            
        Returns:
            ServiceContainer: 支持链式调用
        """
        with self._lock:
            self._instances[interface] = instance
            descriptor = ServiceDescriptor(
                service_type=interface,
                lifetime=ServiceLifetime.SINGLETON
            )
            descriptor.instance = instance
            self._services[interface] = descriptor
            logger.debug(f"Registered instance: {interface.__name__}")
            
        return self
    
    def resolve(self, interface: Type[T]) -> T:
        """
        解析服务实例
        
        Args:
            interface: 服务接口类型
            
        Returns:
            T: 服务实例
            
        Raises:
            ServiceNotRegisteredError: 服务未注册
            CircularDependencyError: 循环依赖
            ServiceCreationError: 服务创建失败
        """
        with self._lock:
            # 循环依赖检测
            if interface in self._building:
                raise CircularDependencyError(f"Circular dependency detected: {interface}")
            
            # 检查是否已有实例
            if interface in self._instances:
                return self._instances[interface]
            
            # 检查是否已注册
            if interface not in self._services:
                raise ServiceNotRegisteredError(f"Service not registered: {interface}")
            
            descriptor = self._services[interface]
            
            # 开始构建
            self._building.add(interface)
            try:
                if descriptor.factory:
                    instance = descriptor.factory()
                else:
                    instance = self._create_with_dependencies(descriptor.implementation)
                
                # 如果是单例，缓存实例
                if descriptor.lifetime == ServiceLifetime.SINGLETON:
                    self._instances[interface] = instance
                
                return instance
            except Exception as e:
                raise ServiceCreationError(f"Failed to create service {interface}: {e}")
            finally:
                self._building.discard(interface)
    
    def _create_with_dependencies(self, implementation: Type) -> Any:
        """
        创建带依赖注入的实例
        
        Args:
            implementation: 实现类型
            
        Returns:
            Any: 创建的实例
        """
        try:
            # 获取构造函数签名
            signature = inspect.signature(implementation.__init__)
            parameters = signature.parameters
            
            # 跳过self参数
            param_names = list(parameters.keys())[1:]
            
            # 解析依赖
            dependencies = {}
            for param_name in param_names:
                param = parameters[param_name]
                if param.annotation != inspect.Parameter.empty:
                    # 如果参数有类型注解，尝试解析
                    if param.annotation in self._services:
                        dependencies[param_name] = self.resolve(param.annotation)
                    elif param.default != inspect.Parameter.empty:
                        # 如果有默认值，使用默认值
                        dependencies[param_name] = param.default
                    else:
                        # 如果没有默认值且无法解析，抛出异常
                        raise ServiceCreationError(f"Cannot resolve dependency {param_name} for {implementation}")
            
            # 创建实例
            return implementation(**dependencies)
        
        except Exception as e:
            logger.error(f"Failed to create instance of {implementation}: {e}")
            # 如果依赖注入失败，尝试无参数构造
            try:
                return implementation()
            except Exception as fallback_error:
                raise ServiceCreationError(f"Failed to create {implementation}: {fallback_error}")
    
    def is_registered(self, interface: Type[T]) -> bool:
        """
        检查服务是否已注册
        
        Args:
            interface: 服务接口类型
            
        Returns:
            bool: 是否已注册
        """
        with self._lock:
            return interface in self._services
    
    def remove(self, interface: Type[T]) -> bool:
        """
        移除服务注册
        
        Args:
            interface: 服务接口类型
            
        Returns:
            bool: 是否移除成功
        """
        with self._lock:
            if interface in self._services:
                del self._services[interface]
                if interface in self._instances:
                    del self._instances[interface]
                logger.debug(f"Removed service: {interface.__name__}")
                return True
            return False
    
    def clear(self):
        """清空所有服务注册"""
        with self._lock:
            self._services.clear()
            self._instances.clear()
            self._building.clear()
            logger.debug("Cleared all services")
    
    def get_registered_services(self) -> Dict[Type, str]:
        """
        获取已注册的服务列表
        
        Returns:
            Dict[Type, str]: 服务类型到生命周期的映射
        """
        with self._lock:
            return {service_type: descriptor.lifetime 
                   for service_type, descriptor in self._services.items()}


# 全局容器实例
_container = None
_container_lock = threading.Lock()


def get_container() -> ServiceContainer:
    """获取全局容器实例"""
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = ServiceContainer()
    return _container


def configure_container() -> ServiceContainer:
    """配置容器"""
    container = get_container()
    
    # 这里可以添加默认的服务注册
    # 具体的服务注册将在 config/container_config.py 中进行
    
    return container


def reset_container():
    """重置容器（主要用于测试）"""
    global _container
    with _container_lock:
        if _container:
            _container.clear()
        _container = None


def get_service(interface: Type[T]) -> T:
    """
    获取服务实例的便捷方法
    
    Args:
        interface: 服务接口类型
        
    Returns:
        T: 服务实例
    """
    return get_container().resolve(interface)


def injectable(container: ServiceContainer = None):
    """
    依赖注入装饰器
    
    Args:
        container: 容器实例，默认使用全局容器
        
    Returns:
        装饰器函数
    """
    if container is None:
        container = get_container()
    
    def decorator(cls):
        # 为类添加容器引用
        cls._container = container
        
        original_init = cls.__init__
        
        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # 自动解析依赖
            signature = inspect.signature(original_init)
            parameters = signature.parameters
            
            # 跳过self参数
            param_names = list(parameters.keys())[1:]
            
            # 解析依赖
            for param_name in param_names:
                param = parameters[param_name]
                if param_name not in kwargs and param.annotation != inspect.Parameter.empty:
                    if container.is_registered(param.annotation):
                        kwargs[param_name] = container.resolve(param.annotation)
            
            # 调用原始构造函数
            original_init(self, *args, **kwargs)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


# 兼容性别名
DIContainer = ServiceContainer
ServiceNotRegisteredException = ServiceNotRegisteredError
ServiceCreationException = ServiceCreationError 