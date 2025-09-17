"""
简化的依赖注入模块

提供统一的服务访问接口，避免跨层直接依赖
"""

import logging
from typing import Any, Optional

# 导入标准日志函数
from utils.logger import get_logger

# 全局服务实例缓存
_services = {}


def get_config():
    """获取配置"""
    if 'config' not in _services:
        try:
            from config.unified_config_manager import get_config_manager
            _services['config'] = get_config_manager()
        except ImportError:
            # 如果配置模块不存在，返回空配置
            class EmptyConfig:
                def __getattr__(self, name):
                    return None
                def get(self, key, default=None):
                    return default
            _services['config'] = EmptyConfig()
    return _services['config']


def get_data_access():
    """获取数据访问接口"""
    if 'data_access' not in _services:
        try:
            from db.clickhouse_db import ClickHouseDB
            _services['data_access'] = ClickHouseDB()
        except ImportError:
            # 如果数据库模块不存在，返回空实现
            class EmptyDataAccess:
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
            _services['data_access'] = EmptyDataAccess()
    return _services['data_access']


def get_data_manager():
    """获取数据管理器"""
    if 'data_manager' not in _services:
        try:
            from db.managers.data_access_manager import DataAccessManager
            _services['data_manager'] = DataAccessManager()
        except ImportError:
            # 如果数据管理器不存在，返回空实现
            class EmptyDataAccessManager:
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
            _services['data_manager'] = EmptyDataAccessManager()
    return _services['data_manager']


def get_service(service_type: type) -> Any:
    """
    获取服务实例（通用接口）

    Args:
        service_type: 服务类型

    Returns:
        Any: 服务实例
    """
    service_name = service_type.__name__

    if service_name not in _services:
        # 特殊处理DataAccessInterface
        if service_name == 'DataAccessInterface':
            try:
                from db.managers.data_access_manager import DataAccessManager
                _services[service_name] = DataAccessManager()
                return _services[service_name]
            except Exception as e:
                print(f"创建DataAccessManager失败: {e}")

        # 尝试创建服务实例
        try:
            if hasattr(service_type, '__module__'):
                # 动态导入和创建
                module_name = service_type.__module__
                class_name = service_type.__name__

                import importlib
                module = importlib.import_module(module_name)
                service_class = getattr(module, class_name)
                _services[service_name] = service_class()
            else:
                # 直接创建
                _services[service_name] = service_type()
        except Exception as e:
            # 如果创建失败，返回空实现
            class EmptyService:
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
            _services[service_name] = EmptyService()

    return _services[service_name]


def get_container():
    """
    获取依赖注入容器（兼容性接口）

    Returns:
        简化的容器对象
    """
    class SimpleContainer:
        def resolve(self, service_type):
            return get_service(service_type)

        def is_registered(self, service_type):
            """检查服务是否已注册"""
            service_name = service_type.__name__ if hasattr(service_type, '__name__') else str(service_type)
            return service_name in _services

        def register(self, service_type, instance=None):
            """注册服务"""
            service_name = service_type.__name__ if hasattr(service_type, '__name__') else str(service_type)
            if instance is None:
                instance = service_type()
            _services[service_name] = instance
            return instance

        def register_singleton(self, service_type, instance=None):
            """注册单例服务（与register相同，因为我们的实现本身就是单例）"""
            return self.register(service_type, instance)

        def register_transient(self, service_type):
            """注册瞬态服务（每次调用都创建新实例）"""
            # 对于瞬态服务，我们不缓存实例
            service_name = f"transient_{service_type.__name__}"
            _services[service_name] = service_type
            return service_type

        def get(self, service_name):
            """根据名称获取服务"""
            if service_name in _services:
                return _services[service_name]
            raise ValueError(f"Service {service_name} not registered")

    return SimpleContainer()


def configure_container():
    """
    配置依赖注入容器

    注册常用的服务到容器中
    """
    container = get_container()

    # 注册日志服务
    try:
        logger = get_logger(__name__)
        container.register_singleton(type(logger), logger)
    except Exception:
        pass

    # 注册配置服务
    try:
        config = get_config()
        container.register_singleton(type(config), config)
    except Exception:
        pass

    # 注册数据访问服务
    try:
        data_access = get_data_access()
        container.register_singleton(type(data_access), data_access)
    except Exception:
        pass

    # 注册数据管理器
    try:
        data_manager = get_data_manager()
        container.register_singleton(type(data_manager), data_manager)
    except Exception:
        pass

    return container


def clear_services():
    """清除服务缓存（主要用于测试）"""
    global _services
    _services.clear()
