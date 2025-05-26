"""
装饰器工具模块

提供异常处理、日志记录、缓存等功能的装饰器
"""

import functools
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast
import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# 定义类型变量
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


def exception_handler(reraise: bool = False, default_return: Any = None) -> Callable[[F], F]:
    """
    异常处理装饰器
    
    Args:
        reraise: 是否重新抛出异常
        default_return: 发生异常时的返回值
        
    Returns:
        Callable: 装饰器函数
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 获取调用堆栈
                stack_trace = traceback.format_exc()
                
                # 记录异常信息
                logger.error(f"函数 {func.__name__} 执行出错: {e}\n{stack_trace}")
                
                # 重新抛出异常或返回默认值
                if reraise:
                    raise
                return default_return
        
        return cast(F, wrapper)
    
    return decorator


def timer(logger_func: Optional[Callable] = None) -> Callable[[F], F]:
    """
    函数执行时间计时器装饰器
    
    Args:
        logger_func: 日志记录函数，默认为标准日志记录器的info方法
        
    Returns:
        Callable: 装饰器函数
    """
    if logger_func is None:
        logger_func = logger.info
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            logger_func(f"函数 {func.__name__} 执行时间: {execution_time:.4f} 秒")
            
            return result
        
        return cast(F, wrapper)
    
    return decorator


def memoize(maxsize: int = 128) -> Callable[[F], F]:
    """
    函数结果缓存装饰器
    
    Args:
        maxsize: 缓存的最大条目数
        
    Returns:
        Callable: 装饰器函数
    """
    def decorator(func: F) -> F:
        cache: Dict[str, Any] = {}
        call_order: List[str] = []
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 创建缓存键
            key_parts = [str(arg) for arg in args]
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = '|'.join(key_parts)
            
            # 检查缓存
            if key in cache:
                # 更新调用顺序
                call_order.remove(key)
                call_order.append(key)
                return cache[key]
            
            # 计算结果
            result = func(*args, **kwargs)
            
            # 更新缓存
            cache[key] = result
            call_order.append(key)
            
            # 如果缓存超过最大容量，删除最旧的条目
            if len(cache) > maxsize:
                oldest_key = call_order.pop(0)
                del cache[oldest_key]
            
            return result
        
        # 添加清除缓存的方法
        def clear_cache() -> None:
            cache.clear()
            call_order.clear()
        
        wrapper.clear_cache = clear_cache  # type: ignore
        
        return cast(F, wrapper)
    
    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0, 
          backoff_factor: float = 2.0, exceptions: tuple = (Exception,)) -> Callable[[F], F]:
    """
    函数重试装饰器
    
    Args:
        max_attempts: 最大尝试次数
        delay: 初始延迟时间（秒）
        backoff_factor: 退避因子，每次尝试后延迟时间的倍数
        exceptions: 要捕获的异常类型
        
    Returns:
        Callable: 装饰器函数
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    
                    if attempts == max_attempts:
                        logger.error(f"函数 {func.__name__} 在 {max_attempts} 次尝试后失败: {e}")
                        raise
                    
                    logger.warning(f"函数 {func.__name__} 第 {attempts} 次尝试失败: {e}, 将在 {current_delay:.2f} 秒后重试")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
        
        return cast(F, wrapper)
    
    return decorator


def validate_dataframe(required_columns: List[str] = None, 
                     min_rows: int = 0) -> Callable[[F], F]:
    """
    DataFrame验证装饰器
    
    Args:
        required_columns: 必需的列名列表
        min_rows: 最小行数
        
    Returns:
        Callable: 装饰器函数
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 寻找DataFrame参数
            df = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df = arg
                    break
            
            if df is None:
                for _, value in kwargs.items():
                    if isinstance(value, pd.DataFrame):
                        df = value
                        break
            
            if df is not None:
                # 验证最小行数
                if min_rows > 0 and len(df) < min_rows:
                    raise ValueError(f"DataFrame至少需要 {min_rows} 行数据，但只有 {len(df)} 行")
                
                # 验证必需的列
                if required_columns:
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        raise ValueError(f"DataFrame缺少必需的列: {', '.join(missing_columns)}")
            
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


def log_calls(level: str = 'debug', 
             log_args: bool = True, 
             log_result: bool = False) -> Callable[[F], F]:
    """
    函数调用日志记录装饰器
    
    Args:
        level: 日志级别
        log_args: 是否记录参数
        log_result: 是否记录返回值
        
    Returns:
        Callable: 装饰器函数
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 获取日志方法
            log_method = getattr(logger, level.lower())
            
            # 记录函数调用
            if log_args:
                args_str = ', '.join([str(arg) for arg in args])
                kwargs_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
                params_str = ', '.join(filter(None, [args_str, kwargs_str]))
                log_method(f"调用函数 {func.__name__}({params_str})")
            else:
                log_method(f"调用函数 {func.__name__}")
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 记录返回值
            if log_result:
                if isinstance(result, pd.DataFrame):
                    log_method(f"函数 {func.__name__} 返回DataFrame，形状: {result.shape}")
                elif isinstance(result, np.ndarray):
                    log_method(f"函数 {func.__name__} 返回NumPy数组，形状: {result.shape}")
                else:
                    log_method(f"函数 {func.__name__} 返回: {result}")
            
            return result
        
        return cast(F, wrapper)
    
    return decorator


def safe_run(default_return: Any = None) -> Callable[[F], F]:
    """
    安全执行装饰器，等同于带默认返回值的异常处理装饰器
    
    Args:
        default_return: 发生异常时的返回值
        
    Returns:
        Callable: 装饰器函数
    """
    return exception_handler(reraise=False, default_return=default_return) 