"""
高级性能监控系统
提供分级性能监控、瓶颈检测、优化建议和资源跟踪
"""

import functools
import time
import threading
import inspect
import gc
import traceback
import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import logging

# 可选依赖
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from utils.enhanced_exception_handler import exception_handler, ErrorSeverity, ErrorCategory
from config.unified_config_manager import get_config

logger = logging.getLogger(__name__)


class MonitoringLevel(Enum):
    """监控级别"""
    METHOD = "method"       # 方法级监控
    CLASS = "class"         # 类级监控
    MODULE = "module"       # 模块级监控
    SYSTEM = "system"       # 系统级监控


class PerformanceIssueType(Enum):
    """性能问题类型"""
    SLOW_EXECUTION = "slow_execution"
    HIGH_MEMORY = "high_memory"
    HIGH_CPU = "high_cpu"
    FREQUENT_CALLS = "frequent_calls"
    MEMORY_LEAK = "memory_leak"
    BLOCKING_IO = "blocking_io"


@dataclass
class AdvancedPerformanceMetric:
    """高级性能指标"""
    identifier: str
    level: MonitoringLevel
    execution_time: float
    memory_usage: float
    cpu_usage: float
    call_count: int
    timestamp: datetime
    thread_id: str
    process_id: int
    stack_depth: int
    gc_collections: int
    success: bool
    error_message: Optional[str] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    io_operations: int = 0
    network_calls: int = 0


@dataclass
class PerformanceBottleneck:
    """性能瓶颈"""
    identifier: str
    issue_type: PerformanceIssueType
    severity: str
    description: str
    impact_score: float
    frequency: int
    avg_time: float
    max_time: float
    suggested_actions: List[str]
    detected_at: datetime
    examples: List[str] = field(default_factory=list)


class AdvancedPerformanceAnalyzer:
    """高级性能分析器"""
    
    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self.metrics: deque = deque(maxlen=max_records)
        self.bottlenecks: List[PerformanceBottleneck] = []
        
        # 性能统计
        self.performance_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0,
            'memory_usage': [],
            'cpu_usage': [],
            'error_count': 0,
            'last_execution': None
        })
        
        # 阈值配置
        self.thresholds = {
            'slow_execution': get_config('performance.slow_execution_threshold', 2.0),
            'high_memory': get_config('performance.high_memory_threshold', 100.0),  # MB
            'high_cpu': get_config('performance.high_cpu_threshold', 80.0),  # %
            'frequent_calls': get_config('performance.frequent_calls_threshold', 100),
            'memory_growth': get_config('performance.memory_growth_threshold', 50.0)  # MB
        }
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 启动分析任务
        self._start_analysis_tasks()
    
    def _start_analysis_tasks(self):
        """启动分析任务"""
        # 瓶颈检测任务
        analysis_thread = threading.Thread(target=self._bottleneck_detection_task, daemon=True)
        analysis_thread.start()
        
        # 内存泄漏检测任务
        if PSUTIL_AVAILABLE:
            memory_thread = threading.Thread(target=self._memory_leak_detection_task, daemon=True)
            memory_thread.start()
    
    def record_metric(self, metric: AdvancedPerformanceMetric):
        """记录性能指标"""
        with self.lock:
            self.metrics.append(metric)
            
            # 更新统计信息
            stats = self.performance_stats[metric.identifier]
            stats['count'] += 1
            stats['total_time'] += metric.execution_time
            stats['min_time'] = min(stats['min_time'], metric.execution_time)
            stats['max_time'] = max(stats['max_time'], metric.execution_time)
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['last_execution'] = metric.timestamp
            
            if metric.memory_usage > 0:
                stats['memory_usage'].append(metric.memory_usage)
                # 保留最近100次记录
                if len(stats['memory_usage']) > 100:
                    stats['memory_usage'] = stats['memory_usage'][-100:]
            
            if metric.cpu_usage > 0:
                stats['cpu_usage'].append(metric.cpu_usage)
                if len(stats['cpu_usage']) > 100:
                    stats['cpu_usage'] = stats['cpu_usage'][-100:]
            
            if not metric.success:
                stats['error_count'] += 1
    
    def _bottleneck_detection_task(self):
        """瓶颈检测任务"""
        while True:
            try:
                time.sleep(60)  # 每分钟检测一次
                self._detect_bottlenecks()
            except Exception as e:
                logger.error(f"瓶颈检测任务出错: {e}")
    
    def _memory_leak_detection_task(self):
        """内存泄漏检测任务"""
        previous_memory = 0
        
        while True:
            try:
                time.sleep(300)  # 每5分钟检测一次
                
                if PSUTIL_AVAILABLE:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    if previous_memory > 0:
                        memory_growth = current_memory - previous_memory
                        
                        if memory_growth > self.thresholds['memory_growth']:
                            self._create_bottleneck(
                                identifier="system_memory",
                                issue_type=PerformanceIssueType.MEMORY_LEAK,
                                severity="warning",
                                description=f"检测到内存增长: {memory_growth:.2f}MB",
                                impact_score=memory_growth / 100,
                                frequency=1,
                                avg_time=0,
                                max_time=0,
                                suggested_actions=[
                                    "检查是否有内存泄漏",
                                    "分析大对象的生命周期",
                                    "考虑增加垃圾回收频率"
                                ]
                            )
                    
                    previous_memory = current_memory
                
            except Exception as e:
                logger.error(f"内存泄漏检测任务出错: {e}")
    
    def _detect_bottlenecks(self):
        """检测性能瓶颈"""
        with self.lock:
            current_time = datetime.now()
            
            for identifier, stats in self.performance_stats.items():
                if stats['count'] == 0:
                    continue
                
                # 检测慢执行
                if stats['avg_time'] > self.thresholds['slow_execution']:
                    self._create_bottleneck(
                        identifier=identifier,
                        issue_type=PerformanceIssueType.SLOW_EXECUTION,
                        severity="warning" if stats['avg_time'] < self.thresholds['slow_execution'] * 2 else "critical",
                        description=f"平均执行时间过长: {stats['avg_time']:.3f}s",
                        impact_score=stats['avg_time'] / self.thresholds['slow_execution'],
                        frequency=stats['count'],
                        avg_time=stats['avg_time'],
                        max_time=stats['max_time'],
                        suggested_actions=self._get_slow_execution_suggestions(identifier, stats)
                    )
                
                # 检测频繁调用
                if stats['count'] > self.thresholds['frequent_calls']:
                    self._create_bottleneck(
                        identifier=identifier,
                        issue_type=PerformanceIssueType.FREQUENT_CALLS,
                        severity="info",
                        description=f"调用频率过高: {stats['count']} 次",
                        impact_score=stats['count'] / self.thresholds['frequent_calls'],
                        frequency=stats['count'],
                        avg_time=stats['avg_time'],
                        max_time=stats['max_time'],
                        suggested_actions=[
                            "考虑添加缓存机制",
                            "批量处理多个请求",
                            "优化调用逻辑"
                        ]
                    )
                
                # 检测高内存使用
                if stats['memory_usage'] and NUMPY_AVAILABLE:
                    avg_memory = np.mean(stats['memory_usage'])
                    if avg_memory > self.thresholds['high_memory']:
                        self._create_bottleneck(
                            identifier=identifier,
                            issue_type=PerformanceIssueType.HIGH_MEMORY,
                            severity="warning",
                            description=f"内存使用过高: {avg_memory:.2f}MB",
                            impact_score=avg_memory / self.thresholds['high_memory'],
                            frequency=len(stats['memory_usage']),
                            avg_time=stats['avg_time'],
                            max_time=stats['max_time'],
                            suggested_actions=[
                                "优化数据结构",
                                "减少内存分配",
                                "使用生成器代替列表"
                            ]
                        )
    
    def _create_bottleneck(self, identifier: str, issue_type: PerformanceIssueType,
                          severity: str, description: str, impact_score: float,
                          frequency: int, avg_time: float, max_time: float,
                          suggested_actions: List[str]):
        """创建瓶颈记录"""
        # 检查是否已存在相同的瓶颈
        existing = next((b for b in self.bottlenecks 
                        if b.identifier == identifier and b.issue_type == issue_type), None)
        
        if existing:
            # 更新现有瓶颈
            existing.frequency += frequency
            existing.detected_at = datetime.now()
        else:
            # 创建新瓶颈
            bottleneck = PerformanceBottleneck(
                identifier=identifier,
                issue_type=issue_type,
                severity=severity,
                description=description,
                impact_score=impact_score,
                frequency=frequency,
                avg_time=avg_time,
                max_time=max_time,
                suggested_actions=suggested_actions,
                detected_at=datetime.now()
            )
            
            self.bottlenecks.append(bottleneck)
            
            # 保留最近100个瓶颈
            if len(self.bottlenecks) > 100:
                self.bottlenecks = self.bottlenecks[-100:]
            
            logger.warning(f"检测到性能瓶颈 [{severity.upper()}]: {identifier} - {description}")
    
    def _get_slow_execution_suggestions(self, identifier: str, stats: Dict) -> List[str]:
        """获取慢执行优化建议"""
        suggestions = []
        
        if 'indicator' in identifier.lower():
            suggestions.extend([
                "考虑使用向量化计算",
                "添加结果缓存",
                "优化数据预处理"
            ])
        
        if 'database' in identifier.lower() or 'query' in identifier.lower():
            suggestions.extend([
                "优化SQL查询",
                "添加数据库索引",
                "使用连接池"
            ])
        
        if stats['avg_time'] > 5.0:
            suggestions.append("考虑异步处理")
        
        if not suggestions:
            suggestions = [
                "分析算法复杂度",
                "优化数据结构",
                "减少不必要的计算"
            ]
        
        return suggestions
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """获取性能报告"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
            recent_bottlenecks = [b for b in self.bottlenecks if b.detected_at > cutoff_time]
        
        if not recent_metrics:
            return {'message': f'最近{hours}小时内无性能数据'}
        
        # 计算总体统计
        total_calls = len(recent_metrics)
        successful_calls = len([m for m in recent_metrics if m.success])
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / total_calls
        
        # 最慢的操作
        slowest_operations = sorted(recent_metrics, key=lambda x: x.execution_time, reverse=True)[:10]
        
        # 最频繁的操作
        call_counts = defaultdict(int)
        for metric in recent_metrics:
            call_counts[metric.identifier] += 1
        
        most_frequent = sorted(call_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'time_range': f'{hours}小时',
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'success_rate': (successful_calls / total_calls) * 100,
            'avg_execution_time': avg_execution_time,
            'bottlenecks_detected': len(recent_bottlenecks),
            'slowest_operations': [
                {
                    'identifier': m.identifier,
                    'execution_time': m.execution_time,
                    'memory_usage': m.memory_usage,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in slowest_operations
            ],
            'most_frequent_operations': [
                {'identifier': identifier, 'count': count}
                for identifier, count in most_frequent
            ],
            'recent_bottlenecks': [
                {
                    'identifier': b.identifier,
                    'issue_type': b.issue_type.value,
                    'severity': b.severity,
                    'description': b.description,
                    'impact_score': b.impact_score,
                    'suggested_actions': b.suggested_actions
                }
                for b in recent_bottlenecks
            ]
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """获取优化建议"""
        recommendations = []
        
        with self.lock:
            # 基于瓶颈生成建议
            for bottleneck in self.bottlenecks[-20:]:  # 最近20个瓶颈
                recommendations.append({
                    'priority': self._calculate_priority(bottleneck),
                    'target': bottleneck.identifier,
                    'issue': bottleneck.description,
                    'actions': bottleneck.suggested_actions,
                    'impact_score': bottleneck.impact_score
                })
            
            # 基于统计数据生成建议
            for identifier, stats in self.performance_stats.items():
                if stats['count'] > 0:
                    # 高错误率建议
                    error_rate = stats['error_count'] / stats['count']
                    if error_rate > 0.1:  # 错误率超过10%
                        recommendations.append({
                            'priority': 'high',
                            'target': identifier,
                            'issue': f'错误率过高: {error_rate:.1%}',
                            'actions': [
                                '增强错误处理',
                                '添加输入验证',
                                '改进异常恢复机制'
                            ],
                            'impact_score': error_rate * 10
                        })
        
        # 按优先级和影响分数排序
        recommendations.sort(key=lambda x: (
            {'critical': 3, 'high': 2, 'medium': 1, 'low': 0}.get(x.get('priority', 'low'), 0),
            x.get('impact_score', 0)
        ), reverse=True)
        
        return recommendations[:20]  # 返回前20个建议
    
    def _calculate_priority(self, bottleneck: PerformanceBottleneck) -> str:
        """计算优先级"""
        if bottleneck.impact_score > 5.0:
            return 'critical'
        elif bottleneck.impact_score > 2.0:
            return 'high'
        elif bottleneck.impact_score > 1.0:
            return 'medium'
        else:
            return 'low'


# 全局性能分析器实例
_performance_analyzer = None
_analyzer_lock = threading.Lock()


def get_performance_analyzer() -> AdvancedPerformanceAnalyzer:
    """获取全局性能分析器实例"""
    global _performance_analyzer
    
    if _performance_analyzer is None:
        with _analyzer_lock:
            if _performance_analyzer is None:
                _performance_analyzer = AdvancedPerformanceAnalyzer()
    
    return _performance_analyzer


def advanced_performance_monitor(level: MonitoringLevel = MonitoringLevel.METHOD,
                               threshold_seconds: float = 1.0,
                               track_memory: bool = True,
                               track_cpu: bool = True,
                               track_io: bool = False,
                               enable_bottleneck_detection: bool = True):
    """
    高级性能监控装饰器

    Args:
        level: 监控级别
        threshold_seconds: 性能警告阈值
        track_memory: 是否跟踪内存使用
        track_cpu: 是否跟踪CPU使用
        track_io: 是否跟踪IO操作
        enable_bottleneck_detection: 是否启用瓶颈检测
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取性能分析器
            analyzer = get_performance_analyzer()

            # 准备监控数据
            start_time = time.time()
            start_memory = 0
            start_cpu = 0
            gc_before = gc.get_count()

            # 获取调用信息
            frame = inspect.currentframe()
            stack_depth = len(inspect.stack())
            thread_id = threading.current_thread().ident
            process_id = os.getpid() if hasattr(os, 'getpid') else 0

            # 内存监控
            if track_memory and PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024  # MB
                except:
                    start_memory = 0

            # CPU监控
            if track_cpu and PSUTIL_AVAILABLE:
                try:
                    start_cpu = psutil.cpu_percent()
                except:
                    start_cpu = 0

            # 执行函数
            success = True
            error_message = None
            result = None
            input_size = 0
            output_size = 0

            try:
                # 估算输入大小
                if track_io:
                    input_size = _estimate_object_size(args) + _estimate_object_size(kwargs)

                result = func(*args, **kwargs)

                # 估算输出大小
                if track_io:
                    output_size = _estimate_object_size(result)

            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                # 计算性能指标
                end_time = time.time()
                execution_time = end_time - start_time

                # 内存使用
                memory_usage = 0
                if track_memory and PSUTIL_AVAILABLE:
                    try:
                        process = psutil.Process()
                        end_memory = process.memory_info().rss / 1024 / 1024  # MB
                        memory_usage = max(0, end_memory - start_memory)
                    except:
                        memory_usage = 0

                # CPU使用
                cpu_usage = 0
                if track_cpu and PSUTIL_AVAILABLE:
                    try:
                        cpu_usage = psutil.cpu_percent()
                    except:
                        cpu_usage = 0

                # GC统计
                gc_after = gc.get_count()
                gc_collections = sum(gc_after) - sum(gc_before)

                # 生成标识符
                if level == MonitoringLevel.METHOD:
                    identifier = f"{func.__module__}.{func.__qualname__}"
                elif level == MonitoringLevel.CLASS:
                    if args and hasattr(args[0], '__class__'):
                        identifier = f"{args[0].__class__.__module__}.{args[0].__class__.__name__}"
                    else:
                        identifier = f"{func.__module__}.{func.__qualname__}"
                elif level == MonitoringLevel.MODULE:
                    identifier = func.__module__
                else:
                    identifier = "system"

                # 创建性能指标
                metric = AdvancedPerformanceMetric(
                    identifier=identifier,
                    level=level,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    cpu_usage=cpu_usage,
                    call_count=1,
                    timestamp=datetime.now(),
                    thread_id=str(thread_id),
                    process_id=process_id,
                    stack_depth=stack_depth,
                    gc_collections=gc_collections,
                    success=success,
                    error_message=error_message,
                    input_size=input_size,
                    output_size=output_size
                )

                # 记录指标
                if enable_bottleneck_detection:
                    analyzer.record_metric(metric)

                # 性能警告
                if execution_time > threshold_seconds:
                    logger.warning(
                        f"性能警告 [{level.value.upper()}]: {identifier} "
                        f"执行时间 {execution_time:.3f}s (阈值: {threshold_seconds}s)"
                    )

                # 详细日志
                logger.debug(
                    f"性能监控 [{level.value}]: {identifier} - "
                    f"时间: {execution_time:.3f}s, "
                    f"内存: {memory_usage:.2f}MB, "
                    f"CPU: {cpu_usage:.1f}%, "
                    f"成功: {success}"
                )

            return result

        return wrapper
    return decorator


def _estimate_object_size(obj) -> int:
    """估算对象大小（字节）"""
    try:
        import sys
        if hasattr(obj, '__len__'):
            return sys.getsizeof(obj) + sum(sys.getsizeof(item) for item in obj)
        else:
            return sys.getsizeof(obj)
    except:
        return 0


# 便捷装饰器
def method_monitor(threshold_seconds: float = 1.0, track_memory: bool = True):
    """方法级性能监控装饰器"""
    return advanced_performance_monitor(
        level=MonitoringLevel.METHOD,
        threshold_seconds=threshold_seconds,
        track_memory=track_memory
    )


def class_monitor(threshold_seconds: float = 2.0, track_memory: bool = True):
    """类级性能监控装饰器"""
    return advanced_performance_monitor(
        level=MonitoringLevel.CLASS,
        threshold_seconds=threshold_seconds,
        track_memory=track_memory
    )


def module_monitor(threshold_seconds: float = 5.0, track_memory: bool = True):
    """模块级性能监控装饰器"""
    return advanced_performance_monitor(
        level=MonitoringLevel.MODULE,
        threshold_seconds=threshold_seconds,
        track_memory=track_memory
    )


def indicator_performance_monitor(threshold_seconds: float = 2.0):
    """指标计算专用性能监控装饰器"""
    return advanced_performance_monitor(
        level=MonitoringLevel.METHOD,
        threshold_seconds=threshold_seconds,
        track_memory=True,
        track_cpu=True,
        track_io=True,
        enable_bottleneck_detection=True
    )


# 导出主要类和装饰器
__all__ = [
    'AdvancedPerformanceAnalyzer',
    'AdvancedPerformanceMetric',
    'PerformanceBottleneck',
    'MonitoringLevel',
    'PerformanceIssueType',
    'get_performance_analyzer',
    'advanced_performance_monitor',
    'method_monitor',
    'class_monitor',
    'module_monitor',
    'indicator_performance_monitor'
]
