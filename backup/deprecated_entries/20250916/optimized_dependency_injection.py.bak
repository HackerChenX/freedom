"""
依赖注入容器优化设计
提供更加清晰的服务注册和解析机制
"""

import inspect
import threading
from typing import Dict, Type, TypeVar, Callable, Any, Optional, List
from abc import ABC, abstractmethod
from contextlib import contextmanager
from utils.logger import get_logger

T = TypeVar('T')
logger = get_logger(__name__)


class ServiceScope:
    """服务生命周期枚举"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceDescriptor:
    """服务描述符"""

    def __init__(self, interface: Type, implementation: Type,
                 scope: str = ServiceScope.TRANSIENT,
                 factory: Optional[Callable] = None,
                 dependencies: Optional[List[str]] = None):
        self.interface = interface
        self.implementation = implementation
        self.scope = scope
        self.factory = factory
        self.dependencies = dependencies or []
        self.instance = None


class IDependencyContainer(ABC):
    """依赖注入容器接口"""

    @abstractmethod
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """注册单例服务"""
        pass

    @abstractmethod
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """注册瞬态服务"""
        pass

    @abstractmethod
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """注册工厂方法"""
        pass

    @abstractmethod
    def resolve(self, interface: Type[T]) -> T:
        """解析服务实例"""
        pass

    @abstractmethod
    def is_registered(self, interface: Type) -> bool:
        """检查服务是否已注册"""
        pass


class OptimizedDependencyContainer(IDependencyContainer):
    """
    优化的依赖注入容器

    特性：
    1. 支持单例、瞬态、作用域生命周期
    2. 自动依赖解析
    3. 循环依赖检测
    4. 线程安全
    5. 性能优化
    """

    def __init__(self):
        self._services: Dict[str, ServiceDescriptor] = {}
        self._singletons: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._resolution_stack: List[str] = []

    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """注册单例服务"""
        with self._lock:
            key = self._get_service_key(interface)
            descriptor = ServiceDescriptor(
                interface=interface,
                implementation=implementation,
                scope=ServiceScope.SINGLETON
            )
            self._services[key] = descriptor
            logger.debug(f"注册单例服务: {key}")

    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """注册瞬态服务"""
        with self._lock:
            key = self._get_service_key(interface)
            descriptor = ServiceDescriptor(
                interface=interface,
                implementation=implementation,
                scope=ServiceScope.TRANSIENT
            )
            self._services[key] = descriptor
            logger.debug(f"注册瞬态服务: {key}")

    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """注册工厂方法"""
        with self._lock:
            key = self._get_service_key(interface)
            descriptor = ServiceDescriptor(
                interface=interface,
                implementation=None,
                scope=ServiceScope.TRANSIENT,
                factory=factory
            )
            self._services[key] = descriptor
            logger.debug(f"注册工厂服务: {key}")

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """注册服务实例"""
        with self._lock:
            key = self._get_service_key(interface)
            self._singletons[key] = instance
            descriptor = ServiceDescriptor(
                interface=interface,
                implementation=type(instance),
                scope=ServiceScope.SINGLETON
            )
            descriptor.instance = instance
            self._services[key] = descriptor
            logger.debug(f"注册服务实例: {key}")

    def resolve(self, interface: Type[T]) -> T:
        """解析服务实例"""
        key = self._get_service_key(interface)

        # 检查循环依赖
        if key in self._resolution_stack:
            cycle = " -> ".join(self._resolution_stack + [key])
            raise DependencyResolutionError(f"检测到循环依赖: {cycle}")

        try:
            self._resolution_stack.append(key)
            return self._resolve_service(key)
        finally:
            self._resolution_stack.remove(key)

    def _resolve_service(self, key: str) -> Any:
        """内部服务解析方法"""
        if key not in self._services:
            raise DependencyResolutionError(f"服务未注册: {key}")

        descriptor = self._services[key]

        # 单例模式检查
        if descriptor.scope == ServiceScope.SINGLETON:
            if key in self._singletons:
                return self._singletons[key]

            # 创建单例实例
            with self._lock:
                # 双重检查锁定
                if key in self._singletons:
                    return self._singletons[key]

                instance = self._create_instance(descriptor)
                self._singletons[key] = instance
                return instance

        # 瞬态模式
        return self._create_instance(descriptor)

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        if descriptor.factory:
            return descriptor.factory()

        implementation = descriptor.implementation
        if not implementation:
            raise DependencyResolutionError(f"服务实现未定义: {descriptor.interface}")

        # 获取构造函数参数
        constructor_params = self._get_constructor_parameters(implementation)

        # 解析依赖
        dependencies = {}
        for param_name, param_type in constructor_params.items():
            if param_type == type(None):  # Optional参数跳过
                continue

            try:
                dependencies[param_name] = self.resolve(param_type)
            except DependencyResolutionError:
                # 尝试使用参数名作为服务键
                try:
                    dependencies[param_name] = self._resolve_service(param_name)
                except DependencyResolutionError:
                    logger.warning(f"无法解析依赖参数: {param_name}")

        # 创建实例
        try:
            return implementation(**dependencies)
        except Exception as e:
            raise DependencyResolutionError(f"创建服务实例失败: {implementation.__name__}: {e}")

    def _get_constructor_parameters(self, cls: Type) -> Dict[str, Type]:
        """获取构造函数参数类型"""
        try:
            signature = inspect.signature(cls.__init__)
            params = {}

            for name, param in signature.parameters.items():
                if name == 'self':
                    continue

                param_type = param.annotation
                if param_type == inspect.Parameter.empty:
                    param_type = type(None)

                params[name] = param_type

            return params
        except Exception:
            return {}

    def is_registered(self, interface: Type) -> bool:
        """检查服务是否已注册"""
        key = self._get_service_key(interface)
        return key in self._services

    def _get_service_key(self, interface: Type) -> str:
        """获取服务键"""
        if hasattr(interface, '__name__'):
            return interface.__name__
        else:
            return str(interface)

    def get_registered_services(self) -> List[str]:
        """获取已注册的服务列表"""
        return list(self._services.keys())

    def clear(self) -> None:
        """清除所有注册的服务"""
        with self._lock:
            self._services.clear()
            self._singletons.clear()
            logger.debug("依赖容器已清除")

    @contextmanager
    def scope(self):
        """作用域上下文管理器"""
        scoped_instances = {}
        original_singletons = self._singletons.copy()

        try:
            # 在作用域内创建的实例
            yield scoped_instances
        finally:
            # 清理作用域实例
            self._singletons = original_singletons


class DependencyResolutionError(Exception):
    """依赖解析异常"""
    pass


class ServiceRegistry:
    """
    服务注册表
    提供便捷的服务注册方法
    """

    def __init__(self, container: OptimizedDependencyContainer):
        self.container = container

    def register_data_services(self):
        """注册数据服务"""
        from db.interfaces.optimized_data_access_interface import DataAccessInterface
        from db.managers.data_access_manager import DataAccessManager

        self.container.register_singleton(DataAccessInterface, DataAccessManager)

    def register_indicator_services(self):
        """注册指标服务"""
        from indicators.complete_indicator_registry import CompleteIndicatorRegistry

        self.container.register_singleton('indicator_registry', CompleteIndicatorRegistry)

    def register_strategy_services(self):
        """注册策略服务"""
        from strategy.strategy_manager import StrategyManager

        self.container.register_singleton('strategy_manager', StrategyManager)

    def register_monitoring_services(self):
        """注册监控服务"""
        from monitoring.market_monitor import MarketMonitor

        self.container.register_singleton('monitoring_service', MarketMonitor)

    def register_api_services(self):
        """注册API服务"""
        try:
            from api.main import APIService
            self.container.register_singleton('api_service', APIService)
        except ImportError:
            logger.warning("API服务不可用")

    def register_all_services(self):
        """注册所有核心服务"""
        self.register_data_services()
        self.register_indicator_services()
        self.register_strategy_services()
        self.register_monitoring_services()
        self.register_api_services()


# 全局容器实例
_global_container = None
_container_lock = threading.Lock()


def get_container() -> OptimizedDependencyContainer:
    """获取全局依赖容器实例"""
    global _global_container

    if _global_container is None:
        with _container_lock:
            if _global_container is None:
                _global_container = OptimizedDependencyContainer()

                # 注册核心服务
                registry = ServiceRegistry(_global_container)
                registry.register_all_services()

                logger.info("全局依赖容器初始化完成")

    return _global_container


def reset_container():
    """重置全局容器（主要用于测试）"""
    global _global_container
    with _container_lock:
        if _global_container:
            _global_container.clear()
        _global_container = None


# 装饰器支持
def inject(service_type: Type[T]) -> Callable:
    """依赖注入装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            container = get_container()
            service = container.resolve(service_type)
            return func(service, *args, **kwargs)
        return wrapper
    return decorator


def service(interface: Type[T], scope: str = ServiceScope.TRANSIENT):
    """服务注册装饰器"""
    def decorator(implementation: Type[T]):
        container = get_container()

        if scope == ServiceScope.SINGLETON:
            container.register_singleton(interface, implementation)
        else:
            container.register_transient(interface, implementation)

        return implementation
    return decorator