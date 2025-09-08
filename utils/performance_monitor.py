"""
性能监控装饰器模块

提供统一的性能监控装饰器
"""

import time
import functools
from typing import Callable, Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceStats:
    """性能统计收集器"""
    
    def __init__(self):
        self.stats: Dict[str, Dict[str, Any]] = {}
    
    def record_execution(self, func_name: str, execution_time: float, success: bool = True):
        """记录执行统计"""
        if func_name not in self.stats:
            self.stats[func_name] = {
                'total_calls': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'max_time': 0.0,
                'min_time': float('inf'),
                'success_count': 0,
                'failure_count': 0
            }
        
        stats = self.stats[func_name]
        stats['total_calls'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['total_calls']
        stats['max_time'] = max(stats['max_time'], execution_time)
        stats['min_time'] = min(stats['min_time'], execution_time)
        
        if success:
            stats['success_count'] += 1
        else:
            stats['failure_count'] += 1
    
    def get_stats(self, func_name: Optional[str] = None) -> Dict[str, Any]:
        """获取统计信息"""
        if func_name:
            return self.stats.get(func_name, {})
        return self.stats.copy()
    
    def reset_stats(self, func_name: Optional[str] = None):
        """重置统计信息"""
        if func_name:
            self.stats.pop(func_name, None)
        else:
            self.stats.clear()


# 全局性能统计实例
_global_stats = PerformanceStats()


def performance_monitor(threshold_seconds: float = 1.0, log_slow: bool = True, collect_stats: bool = True):
    """
    性能监控装饰器
    
    Args:
        threshold_seconds: 慢查询阈值（秒）
        log_slow: 是否记录慢查询日志
        collect_stats: 是否收集统计信息
    
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                execution_time = time.time() - start_time
                
                # 记录统计信息
                if collect_stats:
                    _global_stats.record_execution(func.__name__, execution_time, success)
                
                # 记录慢查询日志
                if log_slow and execution_time > threshold_seconds:
                    logger.warning(
                        f"慢操作检测: {func.__name__} 执行时间 {execution_time:.2f}s "
                        f"(阈值: {threshold_seconds}s)"
                    )
                
                # 记录正常执行日志（debug级别）
                logger.debug(f"性能监控: {func.__name__} 执行时间 {execution_time:.2f}s")
        
        return wrapper
    return decorator


def get_performance_stats(func_name: Optional[str] = None) -> Dict[str, Any]:
    """获取性能统计信息"""
    return _global_stats.get_stats(func_name)


def reset_performance_stats(func_name: Optional[str] = None):
    """重置性能统计信息"""
    _global_stats.reset_stats(func_name)


class PerformanceContext:
    """性能监控上下文管理器"""
    
    def __init__(self, operation_name: str, threshold_seconds: float = 1.0):
        self.operation_name = operation_name
        self.threshold_seconds = threshold_seconds
        self.start_time = None
        self.execution_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.execution_time = time.time() - self.start_time
        
        # 记录统计信息
        success = exc_type is None
        _global_stats.record_execution(self.operation_name, self.execution_time, success)
        
        # 记录慢操作日志
        if self.execution_time > self.threshold_seconds:
            logger.warning(
                f"慢操作检测: {self.operation_name} 执行时间 {self.execution_time:.2f}s "
                f"(阈值: {self.threshold_seconds}s)"
            )


def measure_time(operation_name: str, threshold_seconds: float = 1.0):
    """创建性能监控上下文管理器"""
    return PerformanceContext(operation_name, threshold_seconds)


# 常用的性能监控装饰器预设
def database_performance_monitor(func: Callable) -> Callable:
    """数据库操作性能监控装饰器"""
    return performance_monitor(threshold_seconds=1.0, log_slow=True)(func)


def api_performance_monitor(func: Callable) -> Callable:
    """API调用性能监控装饰器"""
    return performance_monitor(threshold_seconds=2.0, log_slow=True)(func)


def calculation_performance_monitor(func: Callable) -> Callable:
    """计算操作性能监控装饰器"""
    return performance_monitor(threshold_seconds=0.5, log_slow=True)(func)
