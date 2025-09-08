"""
统一错误处理系统

提供标准化的错误处理、异常管理和错误恢复机制
"""

import traceback
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger
from utils.performance_monitor import performance_monitor

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    """错误分类"""
    SYSTEM = "SYSTEM"
    DATABASE = "DATABASE"
    NETWORK = "NETWORK"
    VALIDATION = "VALIDATION"
    BUSINESS = "BUSINESS"
    EXTERNAL = "EXTERNAL"


@dataclass
class ErrorInfo:
    """错误信息结构"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    stack_trace: str
    context: Dict[str, Any]


class UnifiedErrorHandler:
    """
    统一错误处理器
    
    提供标准化的错误处理、记录、通知和恢复机制
    """
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.error_callbacks: Dict[ErrorCategory, List[Callable]] = {}
        self.retry_strategies: Dict[ErrorCategory, Dict[str, Any]] = {}
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """设置默认重试策略"""
        self.retry_strategies = {
            ErrorCategory.DATABASE: {
                "max_retries": 3,
                "backoff_factor": 2.0,
                "initial_delay": 1.0
            },
            ErrorCategory.NETWORK: {
                "max_retries": 5,
                "backoff_factor": 1.5,
                "initial_delay": 0.5
            },
            ErrorCategory.EXTERNAL: {
                "max_retries": 2,
                "backoff_factor": 3.0,
                "initial_delay": 2.0
            }
        }
    
    def register_error_callback(self, category: ErrorCategory, callback: Callable):
        """注册错误回调函数"""
        if category not in self.error_callbacks:
            self.error_callbacks[category] = []
        self.error_callbacks[category].append(callback)
    
    def handle_error(self, error: Exception, category: ErrorCategory, 
                    severity: ErrorSeverity, context: Dict[str, Any] = None) -> ErrorInfo:
        """
        处理错误
        
        Args:
            error: 异常对象
            category: 错误分类
            severity: 错误严重程度
            context: 错误上下文
            
        Returns:
            ErrorInfo: 错误信息对象
        """
        error_info = ErrorInfo(
            error_id=self._generate_error_id(),
            category=category,
            severity=severity,
            message=str(error),
            details=self._extract_error_details(error),
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        # 记录错误
        self._log_error(error_info)
        
        # 保存错误历史
        self.error_history.append(error_info)
        
        # 执行回调
        self._execute_callbacks(error_info)
        
        # 根据严重程度决定是否需要立即处理
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._handle_critical_error(error_info)
        
        return error_info
    
    def _generate_error_id(self) -> str:
        """生成错误ID"""
        return f"ERR_{int(time.time() * 1000)}"
    
    def _extract_error_details(self, error: Exception) -> Dict[str, Any]:
        """提取错误详细信息"""
        return {
            "type": type(error).__name__,
            "args": error.args,
            "module": getattr(error, "__module__", "unknown")
        }
    
    def _log_error(self, error_info: ErrorInfo):
        """记录错误日志"""
        log_message = (
            f"[{error_info.severity.value}] {error_info.category.value} Error: "
            f"{error_info.message} (ID: {error_info.error_id})"
        )
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # 记录详细信息
        logger.debug(f"Error details: {error_info.details}")
        logger.debug(f"Error context: {error_info.context}")
    
    def _execute_callbacks(self, error_info: ErrorInfo):
        """执行错误回调"""
        callbacks = self.error_callbacks.get(error_info.category, [])
        for callback in callbacks:
            try:
                callback(error_info)
            except Exception as e:
                logger.error(f"错误回调执行失败: {e}")
    
    def _handle_critical_error(self, error_info: ErrorInfo):
        """处理关键错误"""
        logger.critical(f"关键错误发生: {error_info.error_id}")
        
        # 可以在这里添加紧急处理逻辑
        # 例如：发送告警、保存状态、优雅关闭等
    
    def retry_with_backoff(self, func: Callable, category: ErrorCategory, 
                          *args, **kwargs) -> Any:
        """
        带退避策略的重试机制
        
        Args:
            func: 要重试的函数
            category: 错误分类
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
        """
        strategy = self.retry_strategies.get(category, {
            "max_retries": 1,
            "backoff_factor": 1.0,
            "initial_delay": 1.0
        })
        
        max_retries = strategy["max_retries"]
        backoff_factor = strategy["backoff_factor"]
        delay = strategy["initial_delay"]
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    logger.warning(f"重试 {attempt + 1}/{max_retries}: {str(e)}")
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    # 最后一次尝试失败，记录错误
                    self.handle_error(e, category, ErrorSeverity.HIGH, {
                        "function": func.__name__,
                        "attempts": max_retries + 1,
                        "args": str(args)[:100],
                        "kwargs": str(kwargs)[:100]
                    })
        
        raise last_exception
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        if not self.error_history:
            return {"total_errors": 0}
        
        stats = {
            "total_errors": len(self.error_history),
            "by_category": {},
            "by_severity": {},
            "recent_errors": len([e for e in self.error_history 
                                if (datetime.now() - e.timestamp).seconds < 3600])
        }
        
        for error in self.error_history:
            # 按分类统计
            category = error.category.value
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            # 按严重程度统计
            severity = error.severity.value
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
        
        return stats


# 全局错误处理器实例
_error_handler = UnifiedErrorHandler()


def get_error_handler() -> UnifiedErrorHandler:
    """获取全局错误处理器实例"""
    return _error_handler


def error_handler(category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 reraise: bool = True,
                 default_return: Any = None,
                 retry_category: Optional[ErrorCategory] = None):
    """
    错误处理装饰器
    
    Args:
        category: 错误分类
        severity: 错误严重程度
        reraise: 是否重新抛出异常
        default_return: 默认返回值
        retry_category: 重试分类（如果设置则启用重试）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if retry_category:
                    # 使用重试机制
                    return _error_handler.retry_with_backoff(
                        func, retry_category, *args, **kwargs
                    )
                else:
                    # 直接执行
                    return func(*args, **kwargs)
            except Exception as e:
                # 处理错误
                error_info = _error_handler.handle_error(
                    e, category, severity, {
                        "function": func.__name__,
                        "module": func.__module__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                )
                
                if reraise:
                    raise
                else:
                    logger.warning(f"函数 {func.__name__} 执行失败，返回默认值: {default_return}")
                    return default_return
        
        return wrapper
    return decorator


def critical_section(category: ErrorCategory = ErrorCategory.SYSTEM):
    """
    关键代码段装饰器
    
    用于标记关键代码段，任何异常都会被标记为高严重程度
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _error_handler.handle_error(e, category, ErrorSeverity.CRITICAL, {
                    "critical_function": func.__name__,
                    "module": func.__module__
                })
                raise
        return wrapper
    return decorator
