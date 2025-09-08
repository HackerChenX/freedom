"""
统一性能监控系统

提供全面的性能监控、分析和优化建议功能
"""

import time
import functools
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)

# 可选导入 psutil
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None


class PerformanceLevel(Enum):
    """性能等级"""
    EXCELLENT = "EXCELLENT"  # < 0.1s
    GOOD = "GOOD"           # 0.1s - 0.5s
    ACCEPTABLE = "ACCEPTABLE"  # 0.5s - 2.0s
    SLOW = "SLOW"           # 2.0s - 5.0s
    CRITICAL = "CRITICAL"   # > 5.0s


@dataclass
class PerformanceMetric:
    """性能指标"""
    function_name: str
    execution_time: float
    memory_before: float
    memory_after: float
    timestamp: datetime
    args_count: int
    kwargs_count: int
    thread_id: int


@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_percent: float
    memory_percent: float
    memory_available: float
    disk_usage: float
    timestamp: datetime


class UnifiedPerformanceMonitor:
    """统一性能监控器"""
    
    def __init__(self, max_history: int = 10000):
        self.metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.system_metrics: deque = deque(maxlen=1000)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        self.max_history = max_history
        
        # 性能阈值配置
        self.thresholds = {
            PerformanceLevel.EXCELLENT: 0.1,
            PerformanceLevel.GOOD: 0.5,
            PerformanceLevel.ACCEPTABLE: 2.0,
            PerformanceLevel.SLOW: 5.0
        }
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except Exception:
                return 0.0
        return 0.0
    
    def record_performance(self, function_name: str, execution_time: float,
                          memory_before: float, memory_after: float,
                          args_count: int = 0, kwargs_count: int = 0):
        """记录性能指标"""
        with self.lock:
            metric = PerformanceMetric(
                function_name=function_name,
                execution_time=execution_time,
                memory_before=memory_before,
                memory_after=memory_after,
                timestamp=datetime.now(),
                args_count=args_count,
                kwargs_count=kwargs_count,
                thread_id=threading.get_ident()
            )
            
            self.metrics[function_name].append(metric)
            self.call_counts[function_name] += 1
            
            # 限制历史记录数量
            if len(self.metrics[function_name]) > self.max_history:
                self.metrics[function_name] = self.metrics[function_name][-self.max_history:]
    
    def record_error(self, function_name: str):
        """记录错误"""
        with self.lock:
            self.error_counts[function_name] += 1
    
    def get_performance_level(self, execution_time: float) -> PerformanceLevel:
        """获取性能等级"""
        if execution_time < self.thresholds[PerformanceLevel.EXCELLENT]:
            return PerformanceLevel.EXCELLENT
        elif execution_time < self.thresholds[PerformanceLevel.GOOD]:
            return PerformanceLevel.GOOD
        elif execution_time < self.thresholds[PerformanceLevel.ACCEPTABLE]:
            return PerformanceLevel.ACCEPTABLE
        elif execution_time < self.thresholds[PerformanceLevel.SLOW]:
            return PerformanceLevel.SLOW
        else:
            return PerformanceLevel.CRITICAL
    
    def get_function_statistics(self, function_name: str) -> Dict[str, Any]:
        """获取函数性能统计"""
        with self.lock:
            if function_name not in self.metrics:
                return {}
            
            metrics = self.metrics[function_name]
            if not metrics:
                return {}
            
            execution_times = [m.execution_time for m in metrics]
            memory_usage = [m.memory_after - m.memory_before for m in metrics]
            
            # 计算统计信息
            avg_time = sum(execution_times) / len(execution_times)
            stats = {
                'function_name': function_name,
                'call_count': self.call_counts[function_name],
                'error_count': self.error_counts.get(function_name, 0),
                'success_rate': (self.call_counts[function_name] - self.error_counts.get(function_name, 0)) / self.call_counts[function_name] * 100,
                'execution_time': {
                    'total': sum(execution_times),
                    'average': avg_time,
                    'min': min(execution_times),
                    'max': max(execution_times),
                    'p95': self._percentile(execution_times, 95),
                    'p99': self._percentile(execution_times, 99)
                },
                'memory_usage': {
                    'average': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                    'max': max(memory_usage) if memory_usage else 0,
                    'min': min(memory_usage) if memory_usage else 0
                },
                'performance_level': self.get_performance_level(avg_time),
                'last_called': metrics[-1].timestamp.isoformat(),
                'thread_usage': len(set(m.thread_id for m in metrics))
            }
            
            return stats
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        if not HAS_PSUTIL:
            return {"status": "psutil_not_available"}
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 计算健康评分
            health_score = 100
            if cpu_percent > 80:
                health_score -= 20
            elif cpu_percent > 60:
                health_score -= 10
            
            if memory.percent > 90:
                health_score -= 30
            elif memory.percent > 70:
                health_score -= 15
            
            if disk.percent > 90:
                health_score -= 20
            elif disk.percent > 80:
                health_score -= 10
            
            return {
                'health_score': max(0, health_score),
                'status': 'healthy' if health_score > 70 else 'warning' if health_score > 40 else 'critical',
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / 1024 / 1024 / 1024,
                'disk_usage_percent': disk.percent,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取系统健康状态失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """获取性能报告"""
        with self.lock:
            # 获取所有函数统计
            all_stats = {}
            for func_name in self.metrics:
                stats = self.get_function_statistics(func_name)
                if stats:
                    all_stats[func_name] = stats
            
            # 找出性能问题
            slow_functions = [
                stats for stats in all_stats.values()
                if stats['performance_level'] in [PerformanceLevel.SLOW, PerformanceLevel.CRITICAL]
            ]
            
            # 找出高错误率函数
            error_prone_functions = [
                stats for stats in all_stats.values()
                if stats['success_rate'] < 95 and stats['call_count'] > 10
            ]
            
            # 找出高频调用函数
            frequent_functions = sorted(
                all_stats.values(),
                key=lambda x: x['call_count'],
                reverse=True
            )[:10]
        
        return {
            'report_period_hours': hours,
            'timestamp': datetime.now().isoformat(),
            'system_health': self.get_system_health(),
            'total_functions_monitored': len(all_stats),
            'total_calls': sum(stats['call_count'] for stats in all_stats.values()),
            'slow_functions': slow_functions,
            'error_prone_functions': error_prone_functions,
            'frequent_functions': frequent_functions,
            'performance_summary': {
                'excellent': len([s for s in all_stats.values() if s['performance_level'] == PerformanceLevel.EXCELLENT]),
                'good': len([s for s in all_stats.values() if s['performance_level'] == PerformanceLevel.GOOD]),
                'acceptable': len([s for s in all_stats.values() if s['performance_level'] == PerformanceLevel.ACCEPTABLE]),
                'slow': len([s for s in all_stats.values() if s['performance_level'] == PerformanceLevel.SLOW]),
                'critical': len([s for s in all_stats.values() if s['performance_level'] == PerformanceLevel.CRITICAL])
            }
        }


# 全局性能监控器实例
_performance_monitor = UnifiedPerformanceMonitor()


def get_performance_monitor() -> UnifiedPerformanceMonitor:
    """获取全局性能监控器实例"""
    return _performance_monitor


def performance_monitor(threshold_seconds: float = 1.0, 
                       log_slow: bool = True,
                       include_args: bool = False):
    """
    性能监控装饰器
    
    Args:
        threshold_seconds: 执行时间阈值（秒），超过此值会记录警告
        log_slow: 是否记录慢执行的日志
        include_args: 是否在日志中包含函数参数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            memory_before = monitor._get_memory_usage()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # 记录错误
                function_name = f"{func.__module__}.{func.__name__}"
                monitor.record_error(function_name)
                raise
            finally:
                execution_time = time.time() - start_time
                memory_after = monitor._get_memory_usage()
                
                # 记录性能指标
                function_name = f"{func.__module__}.{func.__name__}"
                monitor.record_performance(
                    function_name, execution_time, memory_before, memory_after,
                    len(args), len(kwargs)
                )
                
                # 如果执行时间超过阈值，记录警告
                if log_slow and execution_time > threshold_seconds:
                    if include_args:
                        args_str = f"args={args[:3]}..." if len(args) > 3 else f"args={args}"
                        kwargs_str = f"kwargs={list(kwargs.keys())}"
                        logger.warning(
                            f"慢执行检测: {function_name} 耗时 {execution_time:.3f}s "
                            f"(阈值: {threshold_seconds}s) - {args_str}, {kwargs_str}"
                        )
                    else:
                        logger.warning(
                            f"慢执行检测: {function_name} 耗时 {execution_time:.3f}s "
                            f"(阈值: {threshold_seconds}s)"
                        )
        
        return wrapper
    return decorator
