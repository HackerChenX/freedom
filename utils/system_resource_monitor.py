"""
系统资源监控系统
提供CPU、内存、磁盘、网络等系统资源的实时监控和分析
"""

import time
import threading
import platform
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
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

from utils.advanced_performance_monitor import get_performance_analyzer
from utils.enhanced_exception_handler import exception_handler, ErrorSeverity, ErrorCategory
from config.unified_config_manager import get_config
from db.sql_manager import SQLManager, QueryType

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESS = "process"


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SystemResourceMetric:
    """系统资源指标"""
    timestamp: datetime
    cpu_percent: float
    cpu_count: int
    memory_total: float  # GB
    memory_used: float   # GB
    memory_percent: float
    disk_total: float    # GB
    disk_used: float     # GB
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    swap_total: float = 0.0
    swap_used: float = 0.0
    swap_percent: float = 0.0


@dataclass
class ResourceAlert:
    """资源告警"""
    alert_id: str
    resource_type: ResourceType
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class ResourceTrend:
    """资源趋势"""
    resource_type: ResourceType
    direction: str  # increasing, decreasing, stable
    rate: float     # 变化率
    confidence: float  # 置信度
    prediction: float  # 预测值
    time_horizon: int  # 预测时间范围（分钟）


class SystemResourceMonitor:
    """
    系统资源监控器
    
    功能：
    - 实时资源监控
    - 资源告警
    - 趋势分析
    - 性能基准测试
    """
    
    def __init__(self, 
                 monitoring_interval: int = 30,
                 history_size: int = 2880,  # 24小时数据（30秒间隔）
                 enable_alerts: bool = True):
        """
        初始化系统资源监控器
        
        Args:
            monitoring_interval: 监控间隔（秒）
            history_size: 历史数据保留数量
            enable_alerts: 是否启用告警
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_alerts = enable_alerts
        
        # 检查依赖
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil 未安装，系统资源监控功能受限")
            return
        
        # 监控数据
        self.metrics_history: deque = deque(maxlen=history_size)
        self.alerts: List[ResourceAlert] = []
        
        # 告警阈值配置
        self.alert_thresholds = {
            'cpu_warning': get_config('resource_monitor.cpu_warning', 80.0),
            'cpu_critical': get_config('resource_monitor.cpu_critical', 95.0),
            'memory_warning': get_config('resource_monitor.memory_warning', 80.0),
            'memory_critical': get_config('resource_monitor.memory_critical', 95.0),
            'disk_warning': get_config('resource_monitor.disk_warning', 85.0),
            'disk_critical': get_config('resource_monitor.disk_critical', 95.0),
            'process_warning': get_config('resource_monitor.process_warning', 500),
            'process_critical': get_config('resource_monitor.process_critical', 1000)
        }
        
        # 监控状态
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.RLock()
        
        # 系统信息
        self.system_info = self._get_system_info()
        
        logger.info(f"系统资源监控器初始化完成 - 间隔: {monitoring_interval}s, 历史: {history_size}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': platform.python_version()
        }
        
        if PSUTIL_AVAILABLE:
            try:
                info.update({
                    'cpu_count_logical': psutil.cpu_count(logical=True),
                    'cpu_count_physical': psutil.cpu_count(logical=False),
                    'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
                    'boot_time': datetime.fromtimestamp(psutil.boot_time())
                })
            except Exception as e:
                logger.warning(f"获取系统信息失败: {e}")
        
        return info
    
    def start_monitoring(self):
        """启动资源监控"""
        if not PSUTIL_AVAILABLE:
            logger.error("无法启动资源监控：psutil 未安装")
            return
        
        if self.monitoring_active:
            logger.warning("资源监控已在运行中")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("系统资源监控已启动")
    
    def stop_monitoring(self):
        """停止资源监控"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("系统资源监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                metric = self._collect_metrics()
                
                with self.lock:
                    self.metrics_history.append(metric)
                
                # 检查告警
                if self.enable_alerts:
                    self._check_alerts(metric)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"资源监控循环出错: {e}")
                time.sleep(60)  # 出错后等待1分钟再继续
    
    def _collect_metrics(self) -> SystemResourceMetric:
        """收集系统指标"""
        try:
            # CPU信息
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 内存信息
            memory = psutil.virtual_memory()
            memory_total = memory.total / (1024**3)  # GB
            memory_used = memory.used / (1024**3)   # GB
            memory_percent = memory.percent
            
            # 交换空间
            swap = psutil.swap_memory()
            swap_total = swap.total / (1024**3)
            swap_used = swap.used / (1024**3)
            swap_percent = swap.percent
            
            # 磁盘信息
            disk = psutil.disk_usage('/')
            disk_total = disk.total / (1024**3)  # GB
            disk_used = disk.used / (1024**3)   # GB
            disk_percent = (disk.used / disk.total) * 100
            
            # 网络信息
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # 进程信息
            process_count = len(psutil.pids())
            
            # 负载平均值（仅Unix系统）
            load_average = (0.0, 0.0, 0.0)
            try:
                if hasattr(psutil, 'getloadavg'):
                    load_average = psutil.getloadavg()
            except:
                pass
            
            return SystemResourceMetric(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                memory_total=memory_total,
                memory_used=memory_used,
                memory_percent=memory_percent,
                disk_total=disk_total,
                disk_used=disk_used,
                disk_percent=disk_percent,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count,
                load_average=load_average,
                swap_total=swap_total,
                swap_used=swap_used,
                swap_percent=swap_percent
            )
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
            # 返回空指标
            return SystemResourceMetric(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                cpu_count=1,
                memory_total=0.0,
                memory_used=0.0,
                memory_percent=0.0,
                disk_total=0.0,
                disk_used=0.0,
                disk_percent=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                process_count=0
            )
    
    def _check_alerts(self, metric: SystemResourceMetric):
        """检查告警条件"""
        current_time = datetime.now()
        
        # CPU告警
        if metric.cpu_percent >= self.alert_thresholds['cpu_critical']:
            self._create_alert(
                ResourceType.CPU, AlertLevel.CRITICAL,
                f"CPU使用率严重过高: {metric.cpu_percent:.1f}%",
                metric.cpu_percent, self.alert_thresholds['cpu_critical']
            )
        elif metric.cpu_percent >= self.alert_thresholds['cpu_warning']:
            self._create_alert(
                ResourceType.CPU, AlertLevel.WARNING,
                f"CPU使用率过高: {metric.cpu_percent:.1f}%",
                metric.cpu_percent, self.alert_thresholds['cpu_warning']
            )
        
        # 内存告警
        if metric.memory_percent >= self.alert_thresholds['memory_critical']:
            self._create_alert(
                ResourceType.MEMORY, AlertLevel.CRITICAL,
                f"内存使用率严重过高: {metric.memory_percent:.1f}%",
                metric.memory_percent, self.alert_thresholds['memory_critical']
            )
        elif metric.memory_percent >= self.alert_thresholds['memory_warning']:
            self._create_alert(
                ResourceType.MEMORY, AlertLevel.WARNING,
                f"内存使用率过高: {metric.memory_percent:.1f}%",
                metric.memory_percent, self.alert_thresholds['memory_warning']
            )
        
        # 磁盘告警
        if metric.disk_percent >= self.alert_thresholds['disk_critical']:
            self._create_alert(
                ResourceType.DISK, AlertLevel.CRITICAL,
                f"磁盘使用率严重过高: {metric.disk_percent:.1f}%",
                metric.disk_percent, self.alert_thresholds['disk_critical']
            )
        elif metric.disk_percent >= self.alert_thresholds['disk_warning']:
            self._create_alert(
                ResourceType.DISK, AlertLevel.WARNING,
                f"磁盘使用率过高: {metric.disk_percent:.1f}%",
                metric.disk_percent, self.alert_thresholds['disk_warning']
            )
        
        # 进程数告警
        if metric.process_count >= self.alert_thresholds['process_critical']:
            self._create_alert(
                ResourceType.PROCESS, AlertLevel.CRITICAL,
                f"进程数过多: {metric.process_count}",
                metric.process_count, self.alert_thresholds['process_critical']
            )
        elif metric.process_count >= self.alert_thresholds['process_warning']:
            self._create_alert(
                ResourceType.PROCESS, AlertLevel.WARNING,
                f"进程数较多: {metric.process_count}",
                metric.process_count, self.alert_thresholds['process_warning']
            )
    
    def _create_alert(self, resource_type: ResourceType, level: AlertLevel,
                     message: str, value: float, threshold: float):
        """创建告警"""
        alert_id = f"{resource_type.value}_{level.value}_{int(time.time())}"
        
        alert = ResourceAlert(
            alert_id=alert_id,
            resource_type=resource_type,
            level=level,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # 保留最近1000个告警
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        logger.warning(f"资源告警 [{level.value.upper()}]: {message}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前系统状态"""
        if not PSUTIL_AVAILABLE:
            return {'error': 'psutil 未安装'}
        
        try:
            current_metric = self._collect_metrics()
            
            with self.lock:
                recent_alerts = [a for a in self.alerts 
                               if a.timestamp > datetime.now() - timedelta(hours=1)]
            
            return {
                'timestamp': current_metric.timestamp.isoformat(),
                'cpu': {
                    'percent': current_metric.cpu_percent,
                    'count': current_metric.cpu_count,
                    'load_average': current_metric.load_average
                },
                'memory': {
                    'total_gb': current_metric.memory_total,
                    'used_gb': current_metric.memory_used,
                    'percent': current_metric.memory_percent
                },
                'disk': {
                    'total_gb': current_metric.disk_total,
                    'used_gb': current_metric.disk_used,
                    'percent': current_metric.disk_percent
                },
                'network': {
                    'bytes_sent': current_metric.network_bytes_sent,
                    'bytes_recv': current_metric.network_bytes_recv
                },
                'processes': current_metric.process_count,
                'recent_alerts': len(recent_alerts),
                'system_info': self.system_info
            }
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {'error': str(e)}
    
    def get_resource_trends(self, hours: int = 24) -> Dict[str, ResourceTrend]:
        """获取资源趋势分析"""
        if not NUMPY_AVAILABLE:
            return {'error': 'numpy 未安装，无法进行趋势分析'}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if len(recent_metrics) < 10:
            return {'error': f'数据不足，需要至少10个数据点'}
        
        trends = {}
        
        # 分析各种资源的趋势
        resource_data = {
            ResourceType.CPU: [m.cpu_percent for m in recent_metrics],
            ResourceType.MEMORY: [m.memory_percent for m in recent_metrics],
            ResourceType.DISK: [m.disk_percent for m in recent_metrics],
            ResourceType.PROCESS: [m.process_count for m in recent_metrics]
        }
        
        for resource_type, values in resource_data.items():
            trend = self._analyze_trend(values)
            trends[resource_type.value] = ResourceTrend(
                resource_type=resource_type,
                direction=trend['direction'],
                rate=trend['rate'],
                confidence=trend['confidence'],
                prediction=trend['prediction'],
                time_horizon=60  # 1小时预测
            )
        
        return trends
    
    def _analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """分析数据趋势"""
        if len(values) < 2:
            return {
                'direction': 'stable',
                'rate': 0.0,
                'confidence': 0.0,
                'prediction': values[-1] if values else 0.0
            }
        
        # 计算线性趋势
        x = np.arange(len(values))
        y = np.array(values)
        
        # 线性回归
        slope, intercept = np.polyfit(x, y, 1)
        
        # 计算相关系数
        correlation = np.corrcoef(x, y)[0, 1]
        confidence = abs(correlation)
        
        # 确定趋势方向
        if abs(slope) < 0.1:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # 预测未来值（基于当前趋势）
        future_x = len(values) + 12  # 预测12个时间点后
        prediction = slope * future_x + intercept
        prediction = max(0, min(100, prediction))  # 限制在合理范围内
        
        return {
            'direction': direction,
            'rate': abs(slope),
            'confidence': confidence,
            'prediction': prediction
        }
    
    def get_performance_baseline(self) -> Dict[str, Any]:
        """获取性能基准"""
        if not PSUTIL_AVAILABLE:
            return {'error': 'psutil 未安装'}
        
        with self.lock:
            if len(self.metrics_history) < 100:
                return {'error': '数据不足，需要至少100个数据点'}
            
            # 计算基准统计
            cpu_values = [m.cpu_percent for m in self.metrics_history]
            memory_values = [m.memory_percent for m in self.metrics_history]
            disk_values = [m.disk_percent for m in self.metrics_history]
            
            if NUMPY_AVAILABLE:
                baseline = {
                    'cpu': {
                        'mean': np.mean(cpu_values),
                        'median': np.median(cpu_values),
                        'std': np.std(cpu_values),
                        'p95': np.percentile(cpu_values, 95),
                        'p99': np.percentile(cpu_values, 99)
                    },
                    'memory': {
                        'mean': np.mean(memory_values),
                        'median': np.median(memory_values),
                        'std': np.std(memory_values),
                        'p95': np.percentile(memory_values, 95),
                        'p99': np.percentile(memory_values, 99)
                    },
                    'disk': {
                        'mean': np.mean(disk_values),
                        'median': np.median(disk_values),
                        'std': np.std(disk_values),
                        'p95': np.percentile(disk_values, 95),
                        'p99': np.percentile(disk_values, 99)
                    },
                    'data_points': len(self.metrics_history),
                    'time_range_hours': (
                        self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp
                    ).total_seconds() / 3600
                }
            else:
                # 简单统计（无numpy）
                baseline = {
                    'cpu': {
                        'mean': sum(cpu_values) / len(cpu_values),
                        'min': min(cpu_values),
                        'max': max(cpu_values)
                    },
                    'memory': {
                        'mean': sum(memory_values) / len(memory_values),
                        'min': min(memory_values),
                        'max': max(memory_values)
                    },
                    'disk': {
                        'mean': sum(disk_values) / len(disk_values),
                        'min': min(disk_values),
                        'max': max(disk_values)
                    },
                    'data_points': len(self.metrics_history)
                }
        
        return baseline


# 全局资源监控器实例
_resource_monitor = None
_monitor_lock = threading.Lock()


def get_resource_monitor() -> SystemResourceMonitor:
    """获取全局系统资源监控器实例"""
    global _resource_monitor
    
    if _resource_monitor is None:
        with _monitor_lock:
            if _resource_monitor is None:
                _resource_monitor = SystemResourceMonitor()
    
    return _resource_monitor


# 导出主要类
__all__ = [
    'SystemResourceMonitor',
    'SystemResourceMetric',
    'ResourceAlert',
    'ResourceTrend',
    'ResourceType',
    'AlertLevel',
    'get_resource_monitor'
]
