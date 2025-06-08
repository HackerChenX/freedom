"""
装饰器工具类

提供各种通用的装饰器功能
"""

import time
import functools
import inspect
import logging
from typing import Dict, Any, Callable, Optional, Type
import threading
import traceback
from collections import OrderedDict
import sys
import os
import requests

# 获取日志记录器
logger = logging.getLogger(__name__)

def singleton(cls):
    """
    单例模式装饰器
    
    确保被装饰的类只有一个实例存在
    
    Args:
        cls: 要装饰的类
        
    Returns:
        装饰后的类，使用getInstance()方法获取实例
    """
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

def time_it(func):
    """
    计时装饰器，打印函数执行时间
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.4f}秒")
        return result
    return wrapper

def performance_monitor(threshold: float = 0.1):
    """
    性能监控装饰器，记录函数执行时间
    
    Args:
        threshold: 记录警告的时间阈值（秒）
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                
                # 获取函数信息
                func_name = func.__name__
                module_name = func.__module__
                
                # 记录执行时间
                if execution_time >= threshold:
                    logger.warning(f"性能警告: {module_name}.{func_name} 执行时间: {execution_time:.4f}秒")
                else:
                    logger.debug(f"{module_name}.{func_name} 执行时间: {execution_time:.4f}秒")
                
        return wrapper
    return decorator

def cache_result(max_size: int = 128, ttl: Optional[float] = None, cache_size: Optional[int] = None):
    """
    缓存装饰器，缓存函数返回结果
    
    Args:
        max_size: 缓存的最大项数
        ttl: 缓存项的生存时间（秒）
        cache_size: 旧参数，已弃用，请使用max_size
        
    Returns:
        装饰器函数
    """
    if cache_size is not None:
        max_size = cache_size  # 兼容旧参数
        
    def decorator(func):
        # 使用有序字典作为缓存，保证LRU特性
        cache = OrderedDict()
        cache_info = {
            "hits": 0,
            "misses": 0,
            "size": 0,
            "ttl": ttl
        }
        
        # 缓存锁，确保线程安全
        cache_lock = threading.RLock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key_parts = []
            
            # 添加位置参数
            for arg in args:
                try:
                    # 尝试使用哈希值
                    key_parts.append(hash(arg))
                except:
                    # 如果不可哈希，使用类型和id
                    key_parts.append(f"{type(arg)}_{id(arg)}")
            
            # 添加关键字参数
            for k, v in sorted(kwargs.items()):
                try:
                    key_parts.append(f"{k}_{hash(v)}")
                except:
                    key_parts.append(f"{k}_{type(v)}_{id(v)}")
            
            # 生成最终缓存键
            cache_key = hash(func.__module__ + func.__name__ + str(key_parts))
            
            with cache_lock:
                # 检查缓存
                if cache_key in cache:
                    # 获取缓存项
                    timestamp, result = cache[cache_key]
                    
                    # 检查TTL
                    if ttl is None or time.time() - timestamp < ttl:
                        # 移到末尾（最近使用）
                        cache.move_to_end(cache_key)
                        cache_info["hits"] += 1
                        return result
                    else:
                        # TTL过期，删除缓存项
                        del cache[cache_key]
                        cache_info["size"] -= 1
                
                # 缓存未命中，执行函数
                cache_info["misses"] += 1
                
                try:
                    result = func(*args, **kwargs)
                    
                    # 添加到缓存
                    cache[cache_key] = (time.time(), result)
                    cache_info["size"] += 1
                    
                    # 如果超过最大大小，删除最老的项
                    if len(cache) > max_size:
                        cache.popitem(last=False)  # FIFO
                        cache_info["size"] -= 1
                    
                    return result
                except Exception as e:
                    logger.error(f"缓存函数 {func.__name__} 执行出错: {e}")
                    raise
        
        # 添加缓存信息和清除方法
        def clear_cache():
            with cache_lock:
                cache.clear()
                cache_info["size"] = 0
                logger.info(f"已清除 {func.__name__} 的缓存")
        
        def get_cache_info():
            with cache_lock:
                return {
                    "hits": cache_info["hits"],
                    "misses": cache_info["misses"],
                    "size": cache_info["size"],
                    "max_size": max_size,
                    "ttl": ttl
                }
        
        wrapper.clear_cache = clear_cache
        wrapper.cache_info = get_cache_info
        
        return wrapper
    return decorator

def error_handling(default_return=None, logger=None, error_message="执行失败", retries=0, retry_delay=1):
    """
    错误处理装饰器
    
    Args:
        default_return: 发生错误时的默认返回值
        logger: 日志记录器，如果为None则使用全局logger
        error_message: 错误消息前缀
        retries: 重试次数
        retry_delay: 重试延迟（秒）
        
    Returns:
        装饰后的函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 使用提供的logger或全局logger
            log = logger or logging.getLogger(__name__)
            
            # 获取函数位置信息，便于定位错误
            func_name = func.__name__
            func_module = func.__module__
            
            # 记录函数调用
            arg_str = ", ".join([repr(a) for a in args] + [f"{k}={repr(v)}" for k, v in kwargs.items()])
            log.debug(f"调用 {func_module}.{func_name}({arg_str})")
            
            # 定义重试计数和最大重试次数
            retry_count = 0
            max_retries = retries
            
            while True:
                try:
                    # 执行被装饰的函数
                    result = func(*args, **kwargs)
                    
                    # 如果是重试成功的，记录日志
                    if retry_count > 0:
                        log.info(f"函数 {func_module}.{func_name} 在第 {retry_count} 次重试后成功执行")
                    
                    return result
                    
                except Exception as e:
                    # 判断是否可重试的错误
                    retriable = isinstance(e, (
                        ConnectionError, 
                        TimeoutError, 
                        requests.exceptions.RequestException
                    )) if 'requests' in sys.modules else isinstance(e, (ConnectionError, TimeoutError))
                    
                    # 获取异常信息和堆栈跟踪
                    exc_type = type(e).__name__
                    exc_msg = str(e)
                    exc_traceback = traceback.format_exc()
                    
                    # 判断是否应该重试
                    if retriable and retry_count < max_retries:
                        retry_count += 1
                        wait_time = retry_delay * (2 ** (retry_count - 1))  # 指数退避策略
                        
                        log.warning(
                            f"{error_message}: {exc_type} - {exc_msg} "
                            f"在 {func_module}.{func_name} 中. "
                            f"第 {retry_count}/{max_retries} 次重试，等待 {wait_time} 秒..."
                        )
                        
                        # 等待后重试
                        time.sleep(wait_time)
                        continue
                    
                    # 无法重试或重试次数已用完，记录详细错误信息
                    log.error(
                        f"{error_message}: {exc_type} - {exc_msg} "
                        f"在 {func_module}.{func_name} 中. "
                        f"参数: {arg_str}"
                    )
                    
                    # 在DEBUG级别记录完整的堆栈跟踪
                    log.debug(f"详细错误信息:\n{exc_traceback}")
                    
                    # 返回默认值
                    return default_return
                    
        return wrapper
    return decorator

def safe_run(default_return=None, error_logger=None, max_retry=3, retry_delay=1.0, silence_errors=False):
    """
    安全执行装饰器，捕获所有异常并返回默认值
    
    Args:
        default_return: 发生错误时的默认返回值
        error_logger: 错误日志记录器
        max_retry: 最大重试次数
        retry_delay: 重试间隔（秒）
        silence_errors: 是否静默处理错误（不记录日志）
        
    Returns:
        装饰后的函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数信息
            func_name = func.__name__
            func_module = func.__module__
            
            # 日志记录器
            logger = error_logger or logging.getLogger(func_module)
            
            # 获取调用点信息，用于更好的错误追踪
            caller_frame = inspect.currentframe().f_back
            caller_info = ""
            if caller_frame:
                caller_filename = caller_frame.f_code.co_filename
                caller_lineno = caller_frame.f_lineno
                caller_info = f" (从 {os.path.basename(caller_filename)}:{caller_lineno} 调用)"
            
            # 记录开始执行信息
            if not silence_errors:
                logger.debug(f"开始执行 {func_module}.{func_name}{caller_info}")
            
            retry_count = 0
            last_error = None
            
            while retry_count <= max_retry:
                try:
                    result = func(*args, **kwargs)
                    
                    # 如果是重试成功，记录信息
                    if retry_count > 0 and not silence_errors:
                        logger.info(f"函数 {func_name} 在第 {retry_count} 次重试后成功执行")
                        
                    return result
                    
                except Exception as e:
                    last_error = e
                    error_type = type(e).__name__
                    
                    # 判断是否是网络或IO相关的临时错误
                    retriable_error = isinstance(e, (
                        ConnectionError, TimeoutError, IOError, 
                        requests.exceptions.RequestException if 'requests' in sys.modules else Exception
                    ))
                    
                    # 只有对可重试的错误进行重试
                    if retriable_error and retry_count < max_retry:
                        retry_count += 1
                        wait_time = retry_delay * (1.5 ** (retry_count - 1))  # 指数退避
                        
                        if not silence_errors:
                            logger.warning(
                                f"执行 {func_name} 时出错 ({error_type}: {str(e)}), "
                                f"第 {retry_count}/{max_retry} 次重试, 等待 {wait_time:.1f}秒..."
                            )
                            
                        time.sleep(wait_time)
                        continue
                    
                    # 无法重试或重试次数已用完
                    if not silence_errors:
                        # 获取参数信息，但限制长度避免日志过大
                        arg_info = []
                        for i, arg in enumerate(args):
                            arg_str = repr(arg)
                            if len(arg_str) > 100:
                                arg_str = arg_str[:100] + "..."
                            arg_info.append(f"arg{i}={arg_str}")
                            
                        for k, v in kwargs.items():
                            v_str = repr(v)
                            if len(v_str) > 100:
                                v_str = v_str[:100] + "..."
                            arg_info.append(f"{k}={v_str}")
                            
                        arg_str = ", ".join(arg_info)
                        
                        # 记录详细错误信息
                        logger.error(
                            f"执行 {func_module}.{func_name} 失败: {error_type}: {str(e)}{caller_info}\n"
                            f"参数: {arg_str}"
                        )
                        
                        # Debug级别记录完整堆栈跟踪
                        logger.debug(f"详细堆栈:\n{traceback.format_exc()}")
                    
                    break
            
            # 返回默认值
            return default_return
            
        return wrapper
    return decorator

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
         exceptions: tuple = (Exception,)):
    """
    重试装饰器，在失败时自动重试
    
    Args:
        max_attempts: 最大尝试次数
        delay: 初始延迟时间（秒）
        backoff: 退避倍数
        exceptions: 要捕获的异常类型
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_attempts, delay
            
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    msg = f"{func.__name__} 失败，{mtries-1}次重试剩余，{mdelay}秒后重试: {e}"
                    logger.warning(msg)
                    
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            
            # 最后一次尝试
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def validate_args(*arg_validators, **kwarg_validators):
    """
    参数验证装饰器
    
    Args:
        arg_validators: 位置参数验证函数
        kwarg_validators: 关键字参数验证函数
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 验证位置参数
            for i, (arg, validator) in enumerate(zip(args, arg_validators)):
                if not validator(arg):
                    param_name = list(sig.parameters.keys())[i]
                    raise ValueError(f"参数 {param_name} 验证失败: {arg}")
            
            # 验证关键字参数
            for kwarg, value in kwargs.items():
                if kwarg in kwarg_validators and not kwarg_validators[kwarg](value):
                    raise ValueError(f"参数 {kwarg} 验证失败: {value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def log_calls(level: int = logging.DEBUG, args: bool = True, result: bool = False):
    """
    记录函数调用装饰器
    
    Args:
        level: 日志级别
        args: 是否记录参数
        result: 是否记录返回值
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            module_name = func.__module__
            
            # 记录调用信息
            if args:
                args_str = ", ".join([str(arg) for arg in args])
                kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                params_str = f"{args_str}, {kwargs_str}" if kwargs_str else args_str
                logger.log(level, f"调用 {module_name}.{func_name}({params_str})")
            else:
                logger.log(level, f"调用 {module_name}.{func_name}()")
            
            # 执行函数
            func_result = func(*args, **kwargs)
            
            # 记录返回值
            if result:
                logger.log(level, f"{module_name}.{func_name} 返回: {func_result}")
            
            return func_result
        
        return wrapper
    return decorator

def universal_method(func):
    """
    通用方法装饰器
    
    用于指标计算方法的装饰器，提供标准化的输入/输出处理和错误捕获
    
    Args:
        func: 要装饰的函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # 提取self参数
            self = args[0]
            
            # 记录方法调用
            logger.debug(f"调用指标方法 {self.__class__.__name__}.{func.__name__}")
            
            # 执行原始方法
            result = func(*args, **kwargs)
            
            # 如果结果是DataFrame，确保索引是正确的
            if hasattr(result, 'index') and hasattr(result, 'columns'):
                # 保留原始索引
                if hasattr(args[1], 'index'):
                    result.index = args[1].index
            
            return result
            
        except Exception as e:
            # 记录错误信息
            logger.error(f"指标方法 {func.__name__} 执行出错: {str(e)}")
            logger.debug(f"错误详情: {traceback.format_exc()}")
            
            # 抛出异常以便上层处理
            raise
    
    return wrapper 