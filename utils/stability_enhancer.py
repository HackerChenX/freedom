#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
系统稳定性增强模块

提供错误处理、重试机制、优雅降级和故障恢复功能
"""

import time
import threading
import functools
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import logging

from utils.logger import getLogger

logger = getLogger(__name__)


class RetryStrategy(Enum):
    """重试策略"""
    FIXED = "fixed"          # 固定间隔
    EXPONENTIAL = "exponential"  # 指数退避
    LINEAR = "linear"        # 线性增长


class CircuitBreakerState(Enum):
    """熔断器状态"""
    CLOSED = "closed"        # 关闭状态（正常）
    OPEN = "open"           # 开启状态（熔断）
    HALF_OPEN = "half_open"  # 半开状态（试探）


class StabilityError(Exception):
    """稳定性相关异常"""
    pass


class CircuitBreaker:
    """熔断器"""
    
    def __init___54_stabilityenhancer(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Union[Exception, tuple] = Exception):
        """
        初始化熔断器
        
        Args:
            failure_threshold: 失败阈值
            recovery_timeout: 恢复超时时间（秒）
            expected_exception: 预期的异常类型
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        
        self.lock = threading.Lock()
        
        logger.debug(f"熔断器初始化，失败阈值: {failure_threshold}, 恢复超时: {recovery_timeout}秒")
    
    def __call__(self, func):
        """装饰器调用"""
        @functools.wraps(func)
        def wrapper_Enhancer_Stability_Enhancer_Stability_Enhancer_1_stabilityenhancer(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func, *args, **kwargs):
        """执行函数调用"""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("熔断器进入半开状态，尝试恢复")
                else:
                    raise StabilityError("熔断器开启，拒绝执行")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """是否应该尝试重置"""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        """成功时的处理"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info("熔断器恢复到关闭状态")
        
        self.failure_count = 0
    
    def _on_failure(self):
        """失败时的处理"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"熔断器开启，失败次数: {self.failure_count}")
    
    def get_state(self) -> Dict[str, Any]:
        """获取熔断器状态"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


def retry(max_attempts: int = 3,
          delay: float = 1.0,
          strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
          backoff_factor: float = 2.0,
          exceptions: Union[Exception, tuple] = Exception,
          on_retry: Optional[Callable] = None):
    """
    重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间
        strategy: 重试策略
        backoff_factor: 退避因子
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
    """
    def decorator_stability_enhancer(func):
        @functools.wraps(func)
        def wrapper_stability_enhancer(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # 最后一次尝试，直接抛出异常
                        break
                    
                    # 调用重试回调
                    if on_retry:
                        on_retry(e, attempt + 1, max_attempts)
                    
                    logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次执行失败，{current_delay}秒后重试: {e}")
                    
                    # 等待重试
                    time.sleep(current_delay)
                    
                    # 计算下次延迟
                    if strategy == RetryStrategy.EXPONENTIAL:
                        current_delay *= backoff_factor
                    elif strategy == RetryStrategy.LINEAR:
                        current_delay += delay
                    # FIXED 策略保持不变
            
            # 所有重试都失败了
            logger.error(f"函数 {func.__name__} 在 {max_attempts} 次尝试后仍然失败")
            raise last_exception
        
        return wrapper
    return decorator


class GracefulDegradation:
    """优雅降级管理器"""
    
    def __init__(self):
        """初始化优雅降级管理器"""
        self.fallback_functions = {}
        self.degradation_rules = {}
        self.active_degradations = set()
        self.lock = threading.Lock()
        
        logger.debug("优雅降级管理器初始化完成")
    
    def register_fallback(self, service_name: str, fallback_func: Callable):
        """注册降级函数"""
        with self.lock:
            self.fallback_functions[service_name] = fallback_func
        
        logger.info(f"注册降级函数: {service_name}")
    
    def register_degradation_rule(self, service_name: str, 
                                 condition: Callable[[], bool],
                                 description: str = ""):
        """注册降级规则"""
        with self.lock:
            self.degradation_rules[service_name] = {
                'condition': condition,
                'description': description
            }
        
        logger.info(f"注册降级规则: {service_name} - {description}")
    
    def check_degradation(self, service_name: str) -> bool:
        """检查是否需要降级"""
        with self.lock:
            if service_name in self.degradation_rules:
                rule = self.degradation_rules[service_name]
                try:
                    should_degrade = rule['condition']()
                    
                    if should_degrade and service_name not in self.active_degradations:
                        self.active_degradations.add(service_name)
                        logger.warning(f"服务 {service_name} 开始降级: {rule['description']}")
                    elif not should_degrade and service_name in self.active_degradations:
                        self.active_degradations.remove(service_name)
                        logger.info(f"服务 {service_name} 恢复正常")
                    
                    return should_degrade
                    
                except Exception as e:
                    logger.error(f"检查降级规则失败 {service_name}: {e}")
                    return False
            
            return False
    
    def execute_with_fallback(self, service_name: str, primary_func: Callable, *args, **kwargs):
        """执行函数，必要时使用降级"""
        # 检查是否需要降级
        if self.check_degradation(service_name):
            if service_name in self.fallback_functions:
                logger.info(f"使用降级函数执行 {service_name}")
                return self.fallback_functions[service_name](*args, **kwargs)
            else:
                raise StabilityError(f"服务 {service_name} 需要降级但未找到降级函数")
        
        # 正常执行
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            # 主函数失败，尝试降级
            logger.warning(f"主函数 {service_name} 执行失败，尝试降级: {e}")
            
            if service_name in self.fallback_functions:
                try:
                    return self.fallback_functions[service_name](*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"降级函数也失败: {fallback_error}")
                    raise e
            else:
                raise e
    
    def get_status(self) -> Dict[str, Any]:
        """获取降级状态"""
        with self.lock:
            return {
                'active_degradations': list(self.active_degradations),
                'registered_services': list(self.fallback_functions.keys()),
                'degradation_rules_count': len(self.degradation_rules)
            }


class ErrorhandlerEnhancer:
    """错误处理器"""
    
    def __init__(self):
        """初始化错误处理器"""
        self.error_handlers = {}
        self.error_stats = {}
        self.lock = threading.Lock()
        
        logger.debug("错误处理器初始化完成")
    
    def register_handler(self, exception_type: type, handler: Callable):
        """注册错误处理器"""
        with self.lock:
            self.error_handlers[exception_type] = handler
        
        logger.info(f"注册错误处理器: {exception_type.__name__}")
    
    def handle_error_stability_enhancer(self, exception: Exception, context: Optional[Dict[str, Any]] = None):
        """处理错误"""
        exception_type = type(exception)
        
        # 更新错误统计
        with self.lock:
            if exception_type not in self.error_stats:
                self.error_stats[exception_type] = {
                    'count': 0,
                    'first_occurrence': datetime.now(),
                    'last_occurrence': None
                }
            
            self.error_stats[exception_type]['count'] += 1
            self.error_stats[exception_type]['last_occurrence'] = datetime.now()
        
        # 查找合适的处理器
        handler = None
        for exc_type, exc_handler in self.error_handlers.items():
            if isinstance(exception, exc_type):
                handler = exc_handler
                break
        
        if handler:
            try:
                logger.info(f"使用自定义处理器处理错误: {exception_type.__name__}")
                handler(exception, context)
            except Exception as handler_error:
                logger.error(f"错误处理器执行失败: {handler_error}")
        else:
            # 默认处理
            logger.error(f"未找到错误处理器，使用默认处理: {exception}")
            self._default_error_handling(exception, context)
    
    def _default_error_handling(self, exception: Exception, context: Optional[Dict[str, Any]]):
        """默认错误处理"""
        error_info = {
            'exception_type': type(exception).__name__,
            'message': str(exception),
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }
        
        # 记录详细错误信息
        logger.error(f"默认错误处理: {error_info}")
        
        # 可以在这里添加更多默认处理逻辑，如发送告警、记录到数据库等
    
    def get_error_stats_stability_enhancer(self) -> Dict[str, Any]:
        """获取错误统计"""
        with self.lock:
            stats = {}
            for exc_type, stat in self.error_stats.items():
                stats[exc_type.__name__] = {
                    'count': stat['count'],
                    'first_occurrence': stat['first_occurrence'].isoformat(),
                    'last_occurrence': stat['last_occurrence'].isoformat() if stat['last_occurrence'] else None
                }
            return stats


def timeout(seconds: float):
    """超时装饰器"""
    def decorator_stability_enhancer(func):
        @functools.wraps(func)
        def wrapper_stability_enhancer(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                logger.warning(f"函数 {func.__name__} 执行超时 ({seconds}秒)")
                raise TimeoutError(f"函数执行超时: {seconds}秒")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator


class StabilityManager:
    """稳定性管理器"""
    
    def __init__(self):
        """初始化稳定性管理器"""
        self.circuit_breakers = {}
        self.degradation_manager = GracefulDegradation()
        self.error_handler = ErrorhandlerEnhancer()
        self.stats = {
            'errors_handled': 0,
            'circuit_breaker_trips': 0,
            'degradations_activated': 0
        }
        
        logger.debug("稳定性管理器初始化完成")
    
    def create_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """创建熔断器"""
        breaker = CircuitBreaker(**kwargs)
        self.circuit_breakers[name] = breaker
        
        logger.info(f"创建熔断器: {name}")
        return breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """获取熔断器"""
        return self.circuit_breakers.get(name)
    
    def register_degradation(self, service_name: str, fallback_func: Callable, 
                           condition: Callable[[], bool], description: str = ""):
        """注册服务降级"""
        self.degradation_manager.register_fallback(service_name, fallback_func)
        self.degradation_manager.register_degradation_rule(service_name, condition, description)
    
    def execute_with_stability(self, service_name: str, func: Callable, *args, **kwargs):
        """使用稳定性保护执行函数"""
        try:
            # 检查熔断器
            if service_name in self.circuit_breakers:
                breaker = self.circuit_breakers[service_name]
                return breaker.call(func, *args, **kwargs)
            
            # 检查降级
            return self.degradation_manager.execute_with_fallback(service_name, func, *args, **kwargs)
            
        except Exception as e:
            # 错误处理
            self.error_handler.handle_error(e, {'service_name': service_name})
            self.stats['errors_handled'] += 1
            raise
    
    def get_stability_status(self) -> Dict[str, Any]:
        """获取稳定性状态"""
        circuit_breaker_status = {}
        for name, breaker in self.circuit_breakers.items():
            circuit_breaker_status[name] = breaker.get_state()
        
        return {
            'circuit_breakers': circuit_breaker_status,
            'degradation_status': self.degradation_manager.get_status(),
            'error_stats': self.error_handler.get_error_stats(),
            'overall_stats': self.stats
        }


# 全局稳定性管理器
_stability_manager = None
_manager_lock = threading.Lock()


def get_stability_manager() -> StabilityManager:
    """获取全局稳定性管理器"""
    global _stability_manager
    
    if _stability_manager is None:
        with _manager_lock:
            if _stability_manager is None:
                _stability_manager = StabilityManager()
    
    return _stability_manager
