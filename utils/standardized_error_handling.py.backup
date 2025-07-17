#!/usr/bin/env python3
"""
标准化错误处理系统

提供统一的异常处理机制，符合P1架构合规性要求
"""

import sys
import traceback
import functools
from typing import Type, Callable, Any, Optional, Dict, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


class ErrorSeverityStandardized_Error_Handling(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategoryStandardized_Error_Handling(Enum):
    """错误类别"""
    DATABASE = "database"
    NETWORK = "network"
    VALIDATION = "validation"
    CALCULATION = "calculation"
    SYSTEM = "system"
    BUSINESS = "business"


@dataclass
class ErrorContextStandardized_Error_Handling:
    """错误上下文"""
    module: str
    function: str
    line_number: int
    timestamp: datetime
    user_context: Optional[Dict[str, Any]] = None
    system_context: Optional[Dict[str, Any]] = None


class StandardizedErrorStandardized_Error_Handling(Exception):
    """标准化错误基类"""
    
    def __init__(
        self, 
        message: str,
        error_code: str = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: ErrorContext = None,
        original_exception: Exception = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.severity = severity
        self.category = category
        self.context = context
        self.original_exception = original_exception
        self.timestamp = datetime.now()
    
    def _generate_error_code(self) -> str:
        """生成错误代码"""
        return f"STD_{self.__class__.__name__.upper()}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    def to_dict_standardized_error_handling(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'timestamp': self.timestamp.isoformat(),
            'context': {
                'module': self.context.module if self.context else None,
                'function': self.context.function if self.context else None,
                'line_number': self.context.line_number if self.context else None
            },
            'original_exception': str(self.original_exception) if self.original_exception else None
        }


class DatabaseErrorStandardized_Error_Handling(StandardizedError):
    """数据库错误"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATABASE)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class NetworkErrorStandardized_Error_Handling(StandardizedError):
    """网络错误"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.NETWORK)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class ValidationErrorStandardized_Error_Handling(StandardizedError):
    """验证错误"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        super().__init__(message, **kwargs)


class CalculationError(StandardizedError):
    """计算错误"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CALCULATION)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class BusinessErrorStandardized_Error_Handling(StandardizedError):
    """业务错误"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.BUSINESS)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class ErrorHandlerStandardized_Error_Handling:
    """错误处理器"""
    
    def __init__(self):
        self.error_history: List[StandardizedError] = []
        self.error_handlers: Dict[ErrorCategory, Callable] = {}
        self.setup_default_handlers()
    
    def setup_default_handlers(self):
        """设置默认错误处理器"""
        self.error_handlers[ErrorCategory.DATABASE] = self._handle_database_error
        self.error_handlers[ErrorCategory.NETWORK] = self._handle_network_error
        self.error_handlers[ErrorCategory.VALIDATION] = self._handle_validation_error
        self.error_handlers[ErrorCategory.CALCULATION] = self._handle_calculation_error
        self.error_handlers[ErrorCategory.BUSINESS] = self._handle_business_error
        self.error_handlers[ErrorCategory.SYSTEM] = self._handle_system_error
    
    def handle_error_standardized_error_handling(self, error: StandardizedError) -> Optional[Any]:
        """处理错误"""
        # 记录错误
        self.error_history.append(error)
        
        # 记录日志
        self._log_error(error)
        
        # 执行特定处理器
        handler = self.error_handlers.get(error.category)
        if handler:
            return handler(error)
        
        return None
    
    def _log_error(self, error: StandardizedError):
        """记录错误日志"""
        log_message = f"[{error.error_code}] {error.message}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra=error.to_dict())
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra=error.to_dict())
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra=error.to_dict())
        else:
            logger.info(log_message, extra=error.to_dict())
    
    def _handle_database_error(self, error: DatabaseError) -> Optional[Any]:
        """处理数据库错误"""
        logger.error(f"数据库错误处理: {error.message}")
        # 可以添加数据库重连、降级服务等逻辑
        return None
    
    def _handle_network_error(self, error: NetworkError) -> Optional[Any]:
        """处理网络错误"""
        logger.warning(f"网络错误处理: {error.message}")
        # 可以添加重试、缓存等逻辑
        return None
    
    def _handle_validation_error(self, error: ValidationError) -> Optional[Any]:
        """处理验证错误"""
        logger.info(f"验证错误处理: {error.message}")
        # 可以添加数据清理、默认值等逻辑
        return None
    
    def _handle_calculation_error(self, error: CalculationError) -> Optional[Any]:
        """处理计算错误"""
        logger.warning(f"计算错误处理: {error.message}")
        # 可以添加计算重试、使用备用算法等逻辑
        return None
    
    def _handle_business_error(self, error: BusinessError) -> Optional[Any]:
        """处理业务错误"""
        logger.warning(f"业务错误处理: {error.message}")
        # 可以添加业务流程补偿等逻辑
        return None
    
    def _handle_system_error(self, error: StandardizedError) -> Optional[Any]:
        """处理系统错误"""
        logger.error(f"系统错误处理: {error.message}")
        # 可以添加系统监控、告警等逻辑
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        if not self.error_history:
            return {'total_errors': 0}
        
        stats = {
            'total_errors': len(self.error_history),
            'by_category': {},
            'by_severity': {},
            'recent_errors': []
        }
        
        for error in self.error_history:
            # 按类别统计
            category = error.category.value
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
            
            # 按严重程度统计
            severity = error.severity.value
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
        
        # 最近的错误
        stats['recent_errors'] = [
            error.to_dict() for error in self.error_history[-10:]
        ]
        
        return stats


# 全局错误处理器实例
_global_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    return _global_error_handler


def exception_handler_standardized_error_handling(
    error_category: ErrorCategory = ErrorCategory.SYSTEM,
    error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    reraise: bool = True,
    return_value: Any = None
):
    """异常处理装饰器"""
    
    def decorator_standardized_error_handling(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper_standardized_error_handling(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except StandardizedError:
                # 如果已经是标准化错误，直接重新抛出
                raise
            except Exception as e:
                # 获取错误上下文
                frame = sys._getframe()
                context = ErrorContext(
                    module=func.__module__,
                    function=func.__name__,
                    line_number=frame.f_lineno,
                    timestamp=datetime.now()
                )
                
                # 创建标准化错误
                standardized_error = StandardizedError(
                    message=f"Error in {func.__name__}: {str(e)}",
                    severity=error_severity,
                    category=error_category,
                    context=context,
                    original_exception=e
                )
                
                # 处理错误
                get_error_handler().handle_error(standardized_error)
                
                if reraise:
                    raise standardized_error
                else:
                    return return_value
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    error_category: ErrorCategory = ErrorCategory.SYSTEM,
    error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    default_return: Any = None,
    *args, **kwargs
) -> Any:
    """安全执行函数"""
    try:
        return func(*args, **kwargs)
    except StandardizedError as e:
        get_error_handler().handle_error(e)
        return default_return
    except Exception as e:
        # 获取错误上下文
        frame = sys._getframe()
        context = ErrorContext(
            module=func.__module__ if hasattr(func, '__module__') else 'unknown',
            function=func.__name__ if hasattr(func, '__name__') else 'unknown',
            line_number=frame.f_lineno,
            timestamp=datetime.now()
        )
        
        # 创建标准化错误
        standardized_error = StandardizedError(
            message=f"Error executing function: {str(e)}",
            severity=error_severity,
            category=error_category,
            context=context,
            original_exception=e
        )
        
        # 处理错误
        get_error_handler().handle_error(standardized_error)
        return default_return


def create_error_context(
    module: str = None,
    function: str = None,
    user_context: Dict[str, Any] = None,
    system_context: Dict[str, Any] = None
) -> ErrorContext:
    """创建错误上下文"""
    frame = sys._getframe()
    
    return ErrorContext(
        module=module or frame.f_globals.get('__name__', 'unknown'),
        function=function or frame.f_code.co_name,
        line_number=frame.f_lineno,
        timestamp=datetime.now(),
        user_context=user_context,
        system_context=system_context
    )


# 便捷的错误创建函数
def create_database_error_standardized_error_handling(message: str, **kwargs) -> DatabaseError:
    """创建数据库错误"""
    kwargs.setdefault('context', create_error_context())
    return DatabaseError(message, **kwargs)


def create_validation_error_standardized_error_handling(message: str, **kwargs) -> ValidationError:
    """创建验证错误"""
    kwargs.setdefault('context', create_error_context())
    return ValidationError(message, **kwargs)


def create_calculation_error(message: str, **kwargs) -> CalculationError:
    """创建计算错误"""
    kwargs.setdefault('context', create_error_context())
    return CalculationError(message, **kwargs)


def create_business_error_standardized_error_handling(message: str, **kwargs) -> BusinessError:
    """创建业务错误"""
    kwargs.setdefault('context', create_error_context())
    return BusinessError(message, **kwargs) 