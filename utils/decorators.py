"""
装饰器工具模块

提供各种实用的装饰器函数，如异常处理、性能监控、缓存结果等
"""

import time
import functools
import inspect
import logging
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from utils.logger import get_logger
from utils.exceptions import BaseError

logger = get_logger(__name__)


def exception_handler(
    default_return: Any = None,
    log_level: str = "ERROR",
    reraise: bool = False,
):
    """
    异常处理装饰器，捕获并记录异常，提供默认返回值
    
    Args:
        default_return: 发生异常时的默认返回值
        log_level: 日志级别，可选 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        reraise: 是否重新抛出异常
        
    Returns:
        包装后的函数
    """
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    numeric_log_level = log_levels.get(log_level.upper(), logging.ERROR)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 获取调用者信息，便于排错
                frame = inspect.currentframe().f_back
                caller_info = f"{frame.f_code.co_filename}:{frame.f_lineno}"
                
                # 格式化错误信息
                err_msg = f"执行 {func.__name__} 时发生错误: {str(e)} [调用位置: {caller_info}]"
                logger.log(numeric_log_level, err_msg)
                
                # 记录详细的堆栈信息（仅在DEBUG级别）
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"详细堆栈: {traceback.format_exc()}")
                
                # 重新抛出异常
                if reraise:
                    raise
                
                # 返回默认值
                return default_return
        return wrapper
    return decorator


def safe_run(
    default_return: Any = None,
    error_logger: Optional[logging.Logger] = None,
    log_level: str = "ERROR",
    reraise: Union[Type[Exception], Tuple[Type[Exception], ...], None] = None,
    error_callback: Optional[Callable] = None,
):
    """
    安全执行装饰器，捕获并记录异常，提供默认返回值
    
    Args:
        default_return: 发生异常时的默认返回值
        error_logger: 用于记录错误的日志器，默认使用全局日志器
        log_level: 日志级别，可选 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        reraise: 需要重新抛出的异常类型
        error_callback: 发生异常时的回调函数，接收原始异常作为参数
        
    Returns:
        包装后的函数
    """
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    numeric_log_level = log_levels.get(log_level.upper(), logging.ERROR)
    logger_instance = error_logger or logger
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 获取调用者信息，便于排错
                frame = inspect.currentframe().f_back
                caller_info = f"{frame.f_code.co_filename}:{frame.f_lineno}"
                
                # 格式化错误信息
                err_msg = f"执行 {func.__name__} 时发生错误: {str(e)} [调用位置: {caller_info}]"
                logger_instance.log(numeric_log_level, err_msg)
                
                # 记录详细的堆栈信息（仅在DEBUG级别）
                if logger_instance.isEnabledFor(logging.DEBUG):
                    logger_instance.debug(f"详细堆栈: {traceback.format_exc()}")
                
                # 调用错误回调函数
                if error_callback is not None:
                    error_callback(e)
                
                # 重新抛出特定类型的异常
                if reraise is not None and isinstance(e, reraise):
                    raise
                
                # 返回默认值
                return default_return
        return wrapper
    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_on: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    logger: Optional[logging.Logger] = None,
):
    """
    重试装饰器，在失败时自动重试函数执行
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff_factor: 延迟时间的增长因子
        retry_on: 触发重试的异常类型
        logger: 用于记录重试信息的日志器，默认使用全局日志器
        
    Returns:
        包装后的函数
    """
    logger_instance = logger or get_logger(__name__)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    if attempt == max_attempts:
                        logger_instance.error(
                            f"已达到最大重试次数 {max_attempts}，放弃重试: {func.__name__}，错误: {str(e)}"
                        )
                        raise
                    
                    logger_instance.warning(
                        f"第 {attempt} 次尝试失败: {func.__name__}，错误: {str(e)}，"
                        f"{current_delay:.2f} 秒后重试..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                    attempt += 1
        return wrapper
    return decorator


def singleton(cls):
    """
    单例模式装饰器，确保类只有一个实例
    
    Args:
        cls: 要装饰的类
        
    Returns:
        装饰后的类
    """
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def performance_monitor(threshold: float = 0.5, logger: Optional[logging.Logger] = None):
    """
    性能监控装饰器，记录函数执行时间
    
    Args:
        threshold: 记录警告的时间阈值（秒）
        logger: 用于记录性能信息的日志器，默认使用全局日志器
        
    Returns:
        包装后的函数
    """
    logger_instance = logger or get_logger(__name__)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            # 记录执行时间
            if elapsed_time >= threshold:
                logger_instance.warning(
                    f"性能警告: {func.__name__} 执行时间为 {elapsed_time:.4f} 秒，"
                    f"超过阈值 {threshold} 秒"
                )
            else:
                logger_instance.debug(
                    f"性能信息: {func.__name__} 执行时间为 {elapsed_time:.4f} 秒"
                )
            
            return result
        return wrapper
    return decorator


def cache_result(ttl: float = 3600, max_size: int = 100):
    """
    结果缓存装饰器，缓存函数返回值以提高性能
    
    Args:
        ttl: 缓存有效期（秒）
        max_size: 最大缓存条目数
        
    Returns:
        包装后的函数
    """
    def decorator(func):
        cache = {}
        timestamps = {}
        call_counts = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 创建缓存键
            key = str((args, frozenset(kwargs.items())))
            
            # 检查缓存是否有效
            current_time = time.time()
            if key in cache and current_time - timestamps[key] < ttl:
                # 更新调用计数
                call_counts[key] = call_counts.get(key, 0) + 1
                return cache[key]
            
            # 缓存不存在或已过期，执行函数
            result = func(*args, **kwargs)
            
            # 如果缓存已满，移除最不常用的条目
            if len(cache) >= max_size:
                least_used = min(call_counts.items(), key=lambda x: x[1])[0]
                cache.pop(least_used, None)
                timestamps.pop(least_used, None)
                call_counts.pop(least_used, None)
            
            # 更新缓存
            cache[key] = result
            timestamps[key] = current_time
            call_counts[key] = 1
            
            return result
        
        # 添加清除缓存的方法
        def clear_cache():
            cache.clear()
            timestamps.clear()
            call_counts.clear()
        
        wrapper.clear_cache = clear_cache
        return wrapper
    return decorator


def validate_params(**param_validators):
    """
    参数验证装饰器，验证函数参数是否符合要求
    
    Args:
        **param_validators: 参数名和验证函数的映射
        
    Returns:
        包装后的函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数签名
            signature = inspect.signature(func)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 验证每个参数
            for param_name, validator in param_validators.items():
                if param_name in bound_args.arguments:
                    param_value = bound_args.arguments[param_name]
                    if not validator(param_value):
                        raise ValueError(f"参数 '{param_name}' 验证失败: {param_value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_calls(logger: Optional[logging.Logger] = None, level: str = "DEBUG"):
    """
    日志记录装饰器，记录函数调用和返回值
    
    Args:
        logger: 用于记录日志的日志器，默认使用全局日志器
        level: 日志级别，可选 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        
    Returns:
        包装后的函数
    """
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    numeric_log_level = log_levels.get(level.upper(), logging.DEBUG)
    logger_instance = logger or get_logger(__name__)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 记录函数调用
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            
            logger_instance.log(
                numeric_log_level,
                f"调用: {func.__name__}({signature})"
            )
            
            # 执行函数
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            # 记录函数返回
            logger_instance.log(
                numeric_log_level,
                f"返回: {func.__name__} -> {repr(result)[:100]}{'...' if len(repr(result)) > 100 else ''} "
                f"(耗时: {elapsed_time:.4f}秒)"
            )
            
            return result
        return wrapper
    return decorator


def deprecated(reason: str, logger: Optional[logging.Logger] = None):
    """
    废弃警告装饰器，标记函数为已废弃
    
    Args:
        reason: 废弃原因
        logger: 用于记录警告的日志器，默认使用全局日志器
        
    Returns:
        包装后的函数
    """
    logger_instance = logger or get_logger(__name__)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger_instance.warning(
                f"警告: 函数 {func.__name__} 已废弃 ({reason})，"
                "请使用替代方法。"
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def transaction(db_manager_attr: str):
    """
    事务装饰器，确保函数在事务中执行
    
    Args:
        db_manager_attr: 包含数据库管理器的实例属性名
        
    Returns:
        包装后的函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 获取数据库管理器
            db_manager = getattr(self, db_manager_attr)
            
            # 在事务中执行函数
            with db_manager.transaction():
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


def validate_dataframe(required_columns=None, min_rows=1, allow_empty=False):
    """
    DataFrame验证装饰器，检查传入的DataFrame是否符合要求
    
    Args:
        required_columns: 必需的列列表
        min_rows: 最小行数
        allow_empty: 是否允许空DataFrame
        
    Returns:
        包装后的函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 查找DataFrame参数
            import pandas as pd
            df_arg = None
            
            # 检查位置参数
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df_arg = arg
                    break
            
            # 检查关键字参数
            if df_arg is None:
                for arg_name, arg_value in kwargs.items():
                    if isinstance(arg_value, pd.DataFrame):
                        df_arg = arg_value
                        break
            
            # 如果找到DataFrame，验证它
            if df_arg is not None:
                # 检查是否为空
                if not allow_empty and df_arg.empty:
                    raise ValueError("DataFrame不能为空")
                
                # 检查行数
                if not df_arg.empty and len(df_arg) < min_rows:
                    raise ValueError(f"DataFrame必须至少包含{min_rows}行，当前为{len(df_arg)}行")
                
                # 检查必需列
                if required_columns:
                    missing_columns = [col for col in required_columns if col not in df_arg.columns]
                    if missing_columns:
                        raise ValueError(f"DataFrame缺少必需的列: {', '.join(missing_columns)}")
            
            # 执行原始函数
            return func(*args, **kwargs)
        return wrapper
    return decorator 