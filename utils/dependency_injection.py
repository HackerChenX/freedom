#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
依赖注入容器模块

提供轻量级的依赖注入容器实现，支持单例和临时服务注册
"""

import threading
from typing import Dict, Type, Any, Optional, TypeVar, Callable, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifetime(Enum):
    """服务生命周期枚举"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"


class ServiceDescriptor:
    """服务描述符"""
    
    def __init__(self, service_type: Type, implementation: Type = None, 
                 factory: Callable = None, lifetime: ServiceLifetime = ServiceLifetime.SINGLETON):
        self.service_type = service_type
        self.implementation = implementation or service_type
        self.factory = factory
        self.lifetime = lifetime


class ServiceContainer:
    """依赖注入容器"""
    
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
        注册临时服务（每次获取都创建新实例）
        
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
    
    def is_registered(self, interface: Type) -> bool:
        """检查服务是否已注册"""
        return interface in self._services
    
    def resolve(self, interface: Type[T]) -> T:
        """
        解析服务实例
        
        Args:
            interface: 服务接口类型
            
        Returns:
            T: 服务实例
            
        Raises:
            ServiceNotRegisteredException: 服务未注册
            CircularDependencyException: 循环依赖
        """
        if interface not in self._services:
            raise ServiceNotRegisteredException(f"Service not registered: {interface}")
            
        descriptor = self._services[interface]
        
        # 检查循环依赖
        if interface in self._building:
            raise CircularDependencyException(f"Circular dependency detected for: {interface}")
        
        # 对于单例服务，检查是否已有实例
        if descriptor.lifetime == ServiceLifetime.SINGLETON and interface in self._instances:
            return self._instances[interface]
        
        try:
            self._building.add(interface)
            
            # 创建实例
            if descriptor.factory:
                instance = descriptor.factory()
            else:
                instance = descriptor.implementation()
            
            # 对于单例服务，缓存实例
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                self._instances[interface] = instance
                
            return instance
            
        finally:
            self._building.discard(interface)
    
    def get(self, key: str, default=None):
        """
        获取服务实例（兼容旧的get方法调用）
        
        Args:
            key: 服务键名
            default: 默认值
            
        Returns:
            服务实例或默认值
        """
        try:
            # 如果key是字符串，尝试转换为类型
            if isinstance(key, str):
                # 对于常见的服务名称，直接返回相应实例
                if key == 'data_access':
                    from db.interfaces.data_access_interface import DataAccessInterface
                    return self.resolve(DataAccessInterface)
                else:
                    return default
            else:
                # 如果是类型，直接resolve
                return self.resolve(key)
        except Exception:
            return default
    
    def clear_dependency_injection(self):
        """清空容器"""
        with self._lock:
            self._services.clear()
            self._instances.clear()
            self._building.clear()
    
    def get_registered_services(self) -> Dict[Type, ServiceDescriptor]:
        """获取已注册的服务列表"""
        return self._services.copy()


class ServiceNotRegisteredException(Exception):
    """服务未注册异常"""
    pass


class CircularDependencyException(Exception):
    """循环依赖异常"""
    pass


# 全局容器实例
_default_container: Optional[ServiceContainer] = None
_container_lock = threading.Lock()


def get_container() -> ServiceContainer:
    """
    获取默认的依赖注入容器
    
    Returns:
        ServiceContainer: 默认容器实例
    """
    global _default_container
    
    if _default_container is None:
        with _container_lock:
            if _default_container is None:
                _default_container = ServiceContainer()
                _setup_default_services(_default_container)
    
    return _default_container


def _setup_default_services(container: ServiceContainer) -> None:
    """
    设置默认服务
    
    Args:
        container: 服务容器
    """
    try:
        # 自动注册DataAccessInterface服务
        from db.interfaces.data_access_interface import DataAccessInterface
        from db.managers.data_access_manager import DataAccessManager
        
        if not container.is_registered(DataAccessInterface):
            def create_data_access_manager():
                """创建DataAccessManager实例的工厂方法"""
                try:
                    # 首先尝试获取连接管理器
                    from db.connection_manager import get_connection_manager
                    connection_manager = get_connection_manager()
                    return DataAccessManager(connection_manager=connection_manager)
                except Exception as e:
                    logger.warning(f"无法获取连接管理器，使用默认配置: {e}")
                    return DataAccessManager()
            
            container.register_singleton(
                DataAccessInterface, 
                DataAccessManager,
                factory=create_data_access_manager
            )
            logger.info("✅ DataAccessInterface已自动注册到依赖注入容器")
        
    except ImportError as e:
        logger.warning(f"自动注册DataAccessInterface失败 - 导入错误: {e}")
    except Exception as e:
        logger.error(f"自动注册DataAccessInterface失败: {e}")


def get_service(interface: Type[T]) -> T:
    """
    获取服务实例（便捷方法）
    
    Args:
        interface: 服务接口类型
        
    Returns:
        T: 服务实例
    """
    return get_container().resolve(interface)


def configure_container(config_func: Callable[[ServiceContainer], None]) -> ServiceContainer:
    """
    配置依赖注入容器
    
    Args:
        config_func: 配置函数
        
    Returns:
        ServiceContainer: 配置后的容器
    """
    container = get_container()
    config_func(container)
    return container 