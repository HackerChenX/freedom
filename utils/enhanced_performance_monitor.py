"""
增强的性能监控系统
提供方法级性能监控、系统资源监控和性能分析
"""

import functools
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

# 可选依赖
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标"""
    function_name: str
    module_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime
    args_count: int
    kwargs_count: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: float
    disk_usage: float
    network_io: Dict[str, int] = field(default_factory=dict)
    process_count: int = 0


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self.metrics: deque = deque(maxlen=max_records)
        self.system_metrics: deque = deque(maxlen=1000)
        self.function_stats: Dict[str, Dict] = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'error_count': 0,
            'last_execution': None
        })
        
        # 性能阈值配置
        self.thresholds = {
            'execution_time': 2.0,      # 执行时间阈值（秒）
            'memory_usage': 100.0,      # 内存使用阈值（MB）
            'cpu_usage': 80.0,          # CPU使用阈值（%）
            'error_rate': 5.0           # 错误率阈值（%）
        }
        
        # 启动系统监控
        self._start_system_monitoring()
    
    def record_metric(self, metric: PerformanceMetric):
        """记录性能指标"""
        self.metrics.append(metric)
        
        # 更新函数统计
        func_key = f"{metric.module_name}.{metric.function_name}"
        stats = self.function_stats[func_key]
        
        stats['count'] += 1
        stats['total_time'] += metric.execution_time
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], metric.execution_time)
        stats['max_time'] = max(stats['max_time'], metric.execution_time)
        stats['last_execution'] = metric.timestamp
        
        if not metric.success:
            stats['error_count'] += 1
        
        # 检查性能阈值
        self._check_thresholds(metric, stats)
    
    def _check_thresholds(self, metric: PerformanceMetric, stats: Dict):
        """检查性能阈值"""
        warnings = []
        
        # 检查执行时间
        if metric.execution_time > self.thresholds['execution_time']:
            warnings.append(f"执行时间过长: {metric.execution_time:.2f}s")
        
        # 检查内存使用
        if metric.memory_usage > self.thresholds['memory_usage']:
            warnings.append(f"内存使用过高: {metric.memory_usage:.2f}MB")
        
        # 检查错误率
        if stats['count'] >= 10:  # 至少10次执行后才检查错误率
            error_rate = (stats['error_count'] / stats['count']) * 100
            if error_rate > self.thresholds['error_rate']:
                warnings.append(f"错误率过高: {error_rate:.1f}%")
        
        # 记录警告
        if warnings:
            func_name = f"{metric.module_name}.{metric.function_name}"
            logger.warning(f"性能警告 [{func_name}]: {'; '.join(warnings)}")
    
    def _start_system_monitoring(self):
        """启动系统监控"""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil不可用，系统监控功能已禁用")
            return

        def monitor_system():
            while True:
                try:
                    # 获取系统指标
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')

                    # 网络IO（如果可用）
                    network_io = {}
                    try:
                        net_io = psutil.net_io_counters()
                        network_io = {
                            'bytes_sent': net_io.bytes_sent,
                            'bytes_recv': net_io.bytes_recv
                        }
                    except:
                        pass

                    system_metric = SystemMetrics(
                        timestamp=datetime.now(),
                        cpu_percent=cpu_percent,
                        memory_percent=memory.percent,
                        memory_available=memory.available / (1024 * 1024),  # MB
                        disk_usage=disk.percent,
                        network_io=network_io,
                        process_count=len(psutil.pids())
                    )

                    self.system_metrics.append(system_metric)

                    # 检查系统阈值
                    if cpu_percent > self.thresholds['cpu_usage']:
                        logger.warning(f"系统CPU使用率过高: {cpu_percent:.1f}%")

                    time.sleep(60)  # 每分钟监控一次

                except Exception as e:
                    logger.error(f"系统监控错误: {e}")
                    time.sleep(60)

        # 启动后台监控线程
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def get_performance_report(self, hours: int = 1) -> Dict[str, Any]:
        """获取性能报告"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # 过滤最近的指标
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        recent_system = [s for s in self.system_metrics if s.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {'message': '没有最近的性能数据'}
        
        # 计算总体统计
        total_executions = len(recent_metrics)
        successful_executions = len([m for m in recent_metrics if m.success])
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / total_executions
        
        # 最慢的函数
        slowest_functions = sorted(
            [(f, stats['avg_time']) for f, stats in self.function_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # 错误最多的函数
        error_functions = sorted(
            [(f, stats['error_count']) for f, stats in self.function_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # 系统资源统计
        system_stats = {}
        if recent_system:
            system_stats = {
                'avg_cpu': sum(s.cpu_percent for s in recent_system) / len(recent_system),
                'avg_memory': sum(s.memory_percent for s in recent_system) / len(recent_system),
                'min_memory_available': min(s.memory_available for s in recent_system),
                'max_cpu': max(s.cpu_percent for s in recent_system),
                'max_memory': max(s.memory_percent for s in recent_system)
            }
        
        return {
            'period_hours': hours,
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': (successful_executions / total_executions) * 100,
            'avg_execution_time': avg_execution_time,
            'slowest_functions': slowest_functions,
            'error_functions': error_functions,
            'system_stats': system_stats,
            'thresholds': self.thresholds
        }
    
    def get_function_stats(self, function_name: str = None) -> Dict[str, Any]:
        """获取函数统计信息"""
        if function_name:
            return self.function_stats.get(function_name, {})
        else:
            return dict(self.function_stats)
    
    def reset_stats(self):
        """重置统计信息"""
        self.metrics.clear()
        self.function_stats.clear()
        logger.info("性能统计已重置")


class EnhancedPerformanceMonitor:
    """
    增强的性能监控器
    
    提供方法级性能监控和系统资源监控
    """
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.enabled = True
    
    def monitor_function(self, 
                        threshold_seconds: float = 1.0,
                        log_slow: bool = True,
                        track_memory: bool = True) -> Callable:
        """
        函数性能监控装饰器
        
        Args:
            threshold_seconds: 性能警告阈值
            log_slow: 是否记录慢函数
            track_memory: 是否跟踪内存使用
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # 记录开始时间和内存
                start_time = time.time()
                start_memory = 0
                if track_memory and PSUTIL_AVAILABLE:
                    try:
                        process = psutil.Process()
                        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
                    except:
                        pass
                
                # 执行函数
                success = True
                error_message = None
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error_message = str(e)
                    raise
                finally:
                    # 计算执行时间
                    execution_time = time.time() - start_time
                    
                    # 计算内存使用
                    memory_usage = 0
                    if track_memory and PSUTIL_AVAILABLE:
                        try:
                            process = psutil.Process()
                            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
                            memory_usage = end_memory - start_memory
                        except:
                            pass

                    # 获取CPU使用率
                    cpu_usage = 0
                    if PSUTIL_AVAILABLE:
                        try:
                            cpu_usage = psutil.cpu_percent()
                        except:
                            pass
                    
                    # 创建性能指标
                    metric = PerformanceMetric(
                        function_name=func.__name__,
                        module_name=func.__module__,
                        execution_time=execution_time,
                        memory_usage=memory_usage,
                        cpu_usage=cpu_usage,
                        timestamp=datetime.now(),
                        args_count=len(args),
                        kwargs_count=len(kwargs),
                        success=success,
                        error_message=error_message
                    )
                    
                    # 记录指标
                    self.analyzer.record_metric(metric)
                    
                    # 记录慢函数
                    if log_slow and execution_time > threshold_seconds:
                        logger.warning(
                            f"慢函数检测: {func.__module__}.{func.__name__} "
                            f"执行时间 {execution_time:.2f}s (阈值: {threshold_seconds}s)"
                        )
            
            return wrapper
        return decorator
    
    def enable(self):
        """启用性能监控"""
        self.enabled = True
        logger.info("性能监控已启用")
    
    def disable(self):
        """禁用性能监控"""
        self.enabled = False
        logger.info("性能监控已禁用")
    
    def get_report(self, hours: int = 1) -> Dict[str, Any]:
        """获取性能报告"""
        return self.analyzer.get_performance_report(hours)
    
    def get_function_stats(self, function_name: str = None) -> Dict[str, Any]:
        """获取函数统计"""
        return self.analyzer.get_function_stats(function_name)
    
    def reset_stats(self):
        """重置统计"""
        self.analyzer.reset_stats()


# 全局性能监控器实例
_performance_monitor = EnhancedPerformanceMonitor()


def get_performance_monitor() -> EnhancedPerformanceMonitor:
    """获取全局性能监控器"""
    return _performance_monitor


def performance_monitor(threshold_seconds: float = 1.0,
                       log_slow: bool = True,
                       track_memory: bool = True):
    """
    性能监控装饰器
    
    Args:
        threshold_seconds: 性能警告阈值
        log_slow: 是否记录慢函数
        track_memory: 是否跟踪内存使用
    """
    return _performance_monitor.monitor_function(
        threshold_seconds=threshold_seconds,
        log_slow=log_slow,
        track_memory=track_memory
    )


# 导出主要类和函数
__all__ = [
    'PerformanceMetric',
    'SystemMetrics',
    'PerformanceAnalyzer',
    'EnhancedPerformanceMonitor',
    'get_performance_monitor',
    'performance_monitor'
]
