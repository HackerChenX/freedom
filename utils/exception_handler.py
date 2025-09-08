"""
异常处理装饰器模块

提供统一的异常处理装饰器
"""

import functools
import traceback
from typing import Any, Callable, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def exception_handler(reraise: bool = True, default_return: Any = None, log_level: str = "error"):
    """
    异常处理装饰器
    
    Args:
        reraise: 是否重新抛出异常
        default_return: 异常时的默认返回值
        log_level: 日志级别 (error, warning, info)
    
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 记录异常信息
                error_msg = f"方法 {func.__name__} 执行失败: {str(e)}"
                
                if log_level == "error":
                    logger.error(error_msg)
                elif log_level == "warning":
                    logger.warning(error_msg)
                elif log_level == "info":
                    logger.info(error_msg)
                
                # 记录详细的堆栈跟踪（仅在debug模式下）
                logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
                
                if reraise:
                    raise
                
                return default_return
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default_return: Any = None, **kwargs) -> Any:
    """
    安全执行函数
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        default_return: 异常时的默认返回值
        **kwargs: 函数关键字参数
    
    Returns:
        函数执行结果或默认返回值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"安全执行失败 {func.__name__}: {str(e)}")
        return default_return


class ExceptionContext:
    """异常上下文管理器"""
    
    def __init__(self, operation_name: str, reraise: bool = True, default_return: Any = None):
        self.operation_name = operation_name
        self.reraise = reraise
        self.default_return = default_return
        self.exception = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.exception = exc_val
            logger.error(f"操作 {self.operation_name} 失败: {str(exc_val)}")
            
            if not self.reraise:
                return True  # 抑制异常
        
        return False  # 不抑制异常


# 常用的异常处理装饰器预设
def database_exception_handler(func: Callable) -> Callable:
    """数据库操作异常处理装饰器"""
    return exception_handler(reraise=True, log_level="error")(func)


def api_exception_handler(func: Callable) -> Callable:
    """API调用异常处理装饰器"""
    return exception_handler(reraise=False, default_return=None, log_level="warning")(func)


def calculation_exception_handler(func: Callable) -> Callable:
    """计算操作异常处理装饰器"""
    return exception_handler(reraise=True, log_level="error")(func)
