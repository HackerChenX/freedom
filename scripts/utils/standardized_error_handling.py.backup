#!/usr/bin/env python3
"""
标准化错误处理系统

建立统一的错误处理机制，包括错误分类、严重程度、错误上下文等。
符合P1中优先级架构合规性修复要求。

Author: System Architecture Team
Date: 2025-07-16
Version: 1.0
"""

import os
import sys
import logging
import traceback
import functools
from enum import Enum
from typing import Dict, Any, Optional, Callable, Union, Type
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    CRITICAL = "critical"    # 系统崩溃级别
    HIGH = "high"           # 业务功能无法使用
    MEDIUM = "medium"       # 功能降级
    LOW = "low"             # 性能影响或提示信息
    INFO = "info"           # 仅记录信息


class ErrorCategory(Enum):
    """错误分类"""
    DATABASE = "database"           # 数据库相关错误
    NETWORK = "network"             # 网络连接错误
    COMPUTATION = "computation"     # 计算相关错误
    VALIDATION = "validation"       # 数据验证错误
    CONFIGURATION = "configuration" # 配置相关错误
    BUSINESS = "business"           # 业务逻辑错误
    SYSTEM = "system"               # 系统级错误
    EXTERNAL = "external"           # 外部服务错误


@dataclass
class ErrorContext:
    """错误上下文"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    function: str
    message: str
    details: Dict[str, Any]
    stack_trace: Optional[str] = None
    user_message: Optional[str] = None
    suggested_action: Optional[str] = None


class StandardizedError(Exception):
    """标准化错误基类"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        component: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggested_action: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.error_context = ErrorContext(
            error_id=self._generate_error_id(),
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            component=component,
            function=self._get_calling_function(),
            message=message,
            details=details or {},
            stack_trace=traceback.format_exc() if original_error else None,
            user_message=user_message,
            suggested_action=suggested_action
        )
        self.original_error = original_error
    
    def _generate_error_id(self) -> str:
        """生成错误ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _get_calling_function(self) -> str:
        """获取调用函数名"""
        import inspect
        frame = inspect.currentframe()
        try:
            # 回溯到调用StandardizedError的函数
            caller_frame = frame.f_back.f_back
            if caller_frame:
                return f"{caller_frame.f_code.co_filename}:{caller_frame.f_code.co_name}:{caller_frame.f_lineno}"
            return "unknown"
        finally:
            del frame


class DatabaseError(StandardizedError):
    """数据库错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATABASE)
        kwargs.setdefault('component', 'database')
        super().__init__(message, **kwargs)


class NetworkError(StandardizedError):
    """网络错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.NETWORK)
        kwargs.setdefault('component', 'network')
        super().__init__(message, **kwargs)


class ComputationError(StandardizedError):
    """计算错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.COMPUTATION)
        kwargs.setdefault('component', 'computation')
        super().__init__(message, **kwargs)


class ValidationError(StandardizedError):
    """验证错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        kwargs.setdefault('component', 'validation')
        super().__init__(message, **kwargs)


class ConfigurationError(StandardizedError):
    """配置错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        kwargs.setdefault('component', 'configuration')
        super().__init__(message, **kwargs)


class BusinessError(StandardizedError):
    """业务错误"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.BUSINESS)
        kwargs.setdefault('component', 'business')
        super().__init__(message, **kwargs)


class ErrorHandler:
    """标准化错误处理器"""
    
    def __init__(self):
        self.error_registry: Dict[str, ErrorContext] = {}
        self.handlers: Dict[ErrorCategory, Callable] = {}
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """设置默认错误处理器"""
        self.handlers[ErrorCategory.DATABASE] = self._handle_database_error
        self.handlers[ErrorCategory.NETWORK] = self._handle_network_error
        self.handlers[ErrorCategory.COMPUTATION] = self._handle_computation_error
        self.handlers[ErrorCategory.VALIDATION] = self._handle_validation_error
        self.handlers[ErrorCategory.CONFIGURATION] = self._handle_configuration_error
        self.handlers[ErrorCategory.BUSINESS] = self._handle_business_error
        self.handlers[ErrorCategory.SYSTEM] = self._handle_system_error
        self.handlers[ErrorCategory.EXTERNAL] = self._handle_external_error
    
    def handle_error_standardized_error_handling(self, error: StandardizedError) -> bool:
        """处理标准化错误"""
        try:
            # 记录错误
            self.error_registry[error.error_context.error_id] = error.error_context
            
            # 记录日志
            self._log_error(error)
            
            # 调用专门的处理器
            handler = self.handlers.get(error.error_context.category, self._handle_system_error)
            return handler(error)
            
        except Exception as e:
            logger.critical(f"错误处理器本身发生错误: {e}")
            return False
    
    def _log_error(self, error: StandardizedError):
        """记录错误日志"""
        context = error.error_context
        
        log_message = (
            f"[{context.error_id}] {context.category.value.upper()} ERROR in {context.component}: "
            f"{context.message}"
        )
        
        log_data = {
            'error_id': context.error_id,
            'severity': context.severity.value,
            'category': context.category.value,
            'component': context.component,
            'function': context.function,
            'details': context.details,
            'timestamp': context.timestamp.isoformat()
        }
        
        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra=log_data)
        elif context.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra=log_data)
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra=log_data)
        elif context.severity == ErrorSeverity.LOW:
            logger.info(log_message, extra=log_data)
        else:
            logger.debug(log_message, extra=log_data)
        
        # 打印堆栈跟踪（仅限高严重性错误）
        if context.stack_trace and context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            logger.error(f"[{context.error_id}] Stack trace:\n{context.stack_trace}")
    
    def _handle_database_error(self, error: StandardizedError) -> bool:
        """处理数据库错误"""
        context = error.error_context
        
        # 数据库错误的处理逻辑
        if "connection" in context.message.lower():
            # 连接错误，尝试重连
            logger.info(f"[{context.error_id}] 尝试数据库重连...")
            return True
        elif "timeout" in context.message.lower():
            # 超时错误，建议优化查询
            logger.warning(f"[{context.error_id}] 数据库查询超时，建议优化查询语句")
            return True
        else:
            # 其他数据库错误
            logger.error(f"[{context.error_id}] 数据库操作失败，请检查数据和权限")
            return False
    
    def _handle_network_error(self, error: StandardizedError) -> bool:
        """处理网络错误"""
        context = error.error_context
        logger.warning(f"[{context.error_id}] 网络错误，建议检查网络连接或稍后重试")
        return True
    
    def _handle_computation_error(self, error: StandardizedError) -> bool:
        """处理计算错误"""
        context = error.error_context
        logger.error(f"[{context.error_id}] 计算错误，请检查输入数据的有效性")
        return False
    
    def _handle_validation_error(self, error: StandardizedError) -> bool:
        """处理验证错误"""
        context = error.error_context
        logger.warning(f"[{context.error_id}] 数据验证失败，请检查输入参数")
        return True
    
    def _handle_configuration_error(self, error: StandardizedError) -> bool:
        """处理配置错误"""
        context = error.error_context
        logger.error(f"[{context.error_id}] 配置错误，请检查配置文件或环境变量")
        return False
    
    def _handle_business_error(self, error: StandardizedError) -> bool:
        """处理业务错误"""
        context = error.error_context
        logger.warning(f"[{context.error_id}] 业务逻辑错误，请检查业务流程")
        return True
    
    def _handle_system_error(self, error: StandardizedError) -> bool:
        """处理系统错误"""
        context = error.error_context
        logger.error(f"[{context.error_id}] 系统错误，需要技术支持")
        return False
    
    def _handle_external_error(self, error: StandardizedError) -> bool:
        """处理外部服务错误"""
        context = error.error_context
        logger.warning(f"[{context.error_id}] 外部服务错误，建议稍后重试或使用备用服务")
        return True
    
    def get_error_stats_standardized_error_handling(self) -> Dict[str, int]:
        """获取错误统计"""
        stats = {}
        for context in self.error_registry.values():
            key = f"{context.category.value}_{context.severity.value}"
            stats[key] = stats.get(key, 0) + 1
        return stats


# 全局错误处理器实例
_error_handler = ErrorHandler()


def exception_handler(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    component: str = None,
    reraise: bool = True,
    fallback_value: Any = None,
    user_message: str = None
):
    """异常处理装饰器
    
    Args:
        severity: 错误严重程度
        category: 错误分类
        component: 组件名称
        reraise: 是否重新抛出异常
        fallback_value: 异常时的回退值
        user_message: 用户友好的错误消息
    """
    def decorator_standardized_error_handling(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper_standardized_error_handling(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except StandardizedError:
                # 如果已经是标准化错误，直接处理
                raise
            except Exception as e:
                # 包装成标准化错误
                comp = component or func.__module__ + "." + func.__name__
                standardized_error = StandardizedError(
                    message=str(e),
                    severity=severity,
                    category=category,
                    component=comp,
                    user_message=user_message,
                    original_error=e
                )
                
                # 处理错误
                _error_handler.handle_error(standardized_error)
                
                if reraise:
                    raise standardized_error
                else:
                    return fallback_value
        
        return wrapper
    return decorator


def handle_error_standardized_error_handling(error: StandardizedError) -> bool:
    """处理标准化错误的便捷函数"""
    return _error_handler.handle_error(error)


def get_error_stats_standardized_error_handling() -> Dict[str, int]:
    """获取错误统计的便捷函数"""
    return _error_handler.get_error_stats()


def create_error_context_from_exception(
    e: Exception,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    component: str = "unknown"
) -> ErrorContext:
    """从普通异常创建错误上下文"""
    import uuid
    import inspect
    
    frame = inspect.currentframe()
    calling_function = "unknown"
    try:
        if frame and frame.f_back:
            caller_frame = frame.f_back
            calling_function = f"{caller_frame.f_code.co_filename}:{caller_frame.f_code.co_name}:{caller_frame.f_lineno}"
    finally:
        del frame
    
    return ErrorContext(
        error_id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(),
        severity=severity,
        category=category,
        component=component,
        function=calling_function,
        message=str(e),
        details={'original_type': type(e).__name__},
        stack_trace=traceback.format_exc()
    )


# 便捷的错误创建函数
def create_database_error(message: str, **kwargs) -> DatabaseError:
    """创建数据库错误"""
    return DatabaseError(message, **kwargs)


def create_network_error(message: str, **kwargs) -> NetworkError:
    """创建网络错误"""
    return NetworkError(message, **kwargs)


def create_computation_error(message: str, **kwargs) -> ComputationError:
    """创建计算错误"""
    return ComputationError(message, **kwargs)


def create_validation_error(message: str, **kwargs) -> ValidationError:
    """创建验证错误"""
    return ValidationError(message, **kwargs)


def create_configuration_error(message: str, **kwargs) -> ConfigurationError:
    """创建配置错误"""
    return ConfigurationError(message, **kwargs)


def create_business_error(message: str, **kwargs) -> BusinessError:
    """创建业务错误"""
    return BusinessError(message, **kwargs)


# 示例用法和测试函数
def demo_error_handling():
    """演示错误处理的使用"""
    logger.info("开始演示标准化错误处理...")
    
    # 1. 直接创建和处理错误
    try:
        raise create_database_error(
            "数据库连接失败",
            severity=ErrorSeverity.HIGH,
            details={'host': 'localhost', 'port': 9000},
            user_message="数据服务暂时不可用，请稍后重试",
            suggested_action="检查数据库服务状态"
        )
    except StandardizedError as e:
        handle_error(e)
    
    # 2. 使用装饰器处理错误
    @exception_handler(
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.COMPUTATION,
        component="demo",
        reraise=False,
        fallback_value=0
    )
    def risky_calculation(x, y):
        return x / y
    
    result = risky_calculation(10, 0)  # 会触发除零错误
    logger.info(f"计算结果（带错误处理）: {result}")
    
    # 3. 查看错误统计
    stats = get_error_stats()
    logger.info(f"错误统计: {stats}")


if __name__ == "__main__":
    demo_error_handling() 