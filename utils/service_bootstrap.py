"""
服务启动器
在基础设施层提供服务启动功能，避免跨层依赖
"""

import logging
from typing import Optional
from utils.unified_container import get_container, UnifiedServiceContainer

logger = logging.getLogger(__name__)


class ServiceBootstrap:
    """
    服务启动器
    
    负责启动和配置依赖注入容器，但不直接导入业务层模块
    """
    
    def __init__(self, container: Optional[UnifiedServiceContainer] = None):
        self.container = container or get_container()
        self._initialized = False
    
    def bootstrap_infrastructure_services(self) -> bool:
        """
        启动基础设施层服务
        
        Returns:
            bool: 启动成功
        """
        if self._initialized:
            return True
        
        try:
            logger.info("启动基础设施层服务...")
            
            # 注册日志服务
            self._register_logger_service()

            # 注册统一配置管理器
            self._register_unified_config()

            # 注册配置服务（向后兼容）
            self._register_config_service()

            # 注册增强性能监控
            self._register_performance_monitor()

            # 注册增强异常处理器
            self._register_exception_handler()
            
            self._initialized = True
            logger.info("基础设施层服务启动完成")
            return True
            
        except Exception as e:
            logger.error(f"基础设施层服务启动失败: {e}")
            return False
    
    def _register_logger_service(self) -> None:
        """注册日志服务"""
        try:
            from utils.logger import get_logger
            self.container.register_singleton(
                logging.Logger,
                factory=lambda: get_logger(__name__)
            )
            logger.debug("日志服务注册成功")
        except Exception as e:
            logger.warning(f"日志服务注册失败: {e}")
    
    def _register_config_service(self) -> None:
        """注册配置服务"""
        try:
            from config.database_config_manager import DatabaseConfigManager
            self.container.register_singleton(
                DatabaseConfigManager,
                factory=lambda: DatabaseConfigManager()
            )
            logger.debug("配置服务注册成功")
        except Exception as e:
            logger.warning(f"配置服务注册失败: {e}")
    
    def _register_performance_monitor(self) -> None:
        """注册增强性能监控服务"""
        try:
            from utils.enhanced_performance_monitor import EnhancedPerformanceMonitor

            self.container.register_singleton(
                EnhancedPerformanceMonitor,
                factory=lambda: EnhancedPerformanceMonitor()
            )
            logger.debug("增强性能监控服务注册成功")
        except ImportError:
            logger.debug("增强性能监控模块不可用，跳过注册")
        except Exception as e:
            logger.warning(f"增强性能监控服务注册失败: {e}")

    def _register_exception_handler(self) -> None:
        """注册增强异常处理器服务"""
        try:
            from utils.enhanced_exception_handler import EnhancedExceptionHandler

            self.container.register_singleton(
                EnhancedExceptionHandler,
                factory=lambda: EnhancedExceptionHandler()
            )
            logger.debug("增强异常处理器服务注册成功")
        except ImportError:
            logger.debug("增强异常处理器模块不可用，跳过注册")
        except Exception as e:
            logger.warning(f"增强异常处理器服务注册失败: {e}")

    def _register_unified_config(self) -> None:
        """注册统一配置管理器"""
        try:
            from config.unified_config_manager import UnifiedConfigManager

            self.container.register_singleton(
                UnifiedConfigManager,
                factory=lambda: UnifiedConfigManager()
            )
            logger.debug("统一配置管理器注册成功")
        except ImportError:
            logger.debug("统一配置管理器模块不可用，跳过注册")
        except Exception as e:
            logger.warning(f"统一配置管理器注册失败: {e}")
    
    def register_layer_services(self, layer_name: str, registration_func: callable) -> bool:
        """
        注册指定层的服务
        
        Args:
            layer_name: 层名称
            registration_func: 注册函数
            
        Returns:
            bool: 注册成功
        """
        try:
            logger.debug(f"注册{layer_name}服务...")
            registration_func(self.container)
            logger.debug(f"{layer_name}服务注册完成")
            return True
        except Exception as e:
            logger.warning(f"{layer_name}服务注册失败: {e}")
            return False
    
    def get_container_status(self) -> dict:
        """
        获取容器状态
        
        Returns:
            dict: 容器状态信息
        """
        return {
            'initialized': self._initialized,
            'registered_services': self.container.get_registered_services(),
            'total_services': len(self.container.get_registered_services())
        }


# 全局服务启动器
_bootstrap: Optional[ServiceBootstrap] = None


def get_bootstrap() -> ServiceBootstrap:
    """获取全局服务启动器"""
    global _bootstrap
    if _bootstrap is None:
        _bootstrap = ServiceBootstrap()
    return _bootstrap


def initialize_infrastructure() -> bool:
    """
    初始化基础设施
    
    Returns:
        bool: 初始化成功
    """
    bootstrap = get_bootstrap()
    return bootstrap.bootstrap_infrastructure_services()


def get_container_status() -> dict:
    """
    获取容器状态
    
    Returns:
        dict: 容器状态信息
    """
    bootstrap = get_bootstrap()
    return bootstrap.get_container_status()


# 导出主要类和函数
__all__ = [
    'ServiceBootstrap',
    'get_bootstrap',
    'initialize_infrastructure',
    'get_container_status'
]
