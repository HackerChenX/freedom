#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
综合选股测试系统监控基础设施

提供实时测试监控、性能指标收集、告警管理和资源监控功能。
支持自定义告警阈值、多级告警和告警恢复检测。
遵循L2基础设施层规范。
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from .logging_config import get_test_logger

logger = get_test_logger('monitoring')


@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """告警信息"""
    name: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    timestamp: datetime
    resolved: bool = False
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, collection_interval: int = 5):
        """
        初始化指标收集器
        
        Args:
            collection_interval: 收集间隔（秒）
        """
        self.collection_interval = collection_interval
        self.metrics = defaultdict(lambda: deque(maxlen=1000))  # 保留最近1000个数据点
        self.collectors = {}
        self.running = False
        self.thread = None
        
        # 注册默认收集器
        self._register_default_collectors()
    
    def _register_default_collectors(self) -> None:
        """注册默认指标收集器"""
        self.register_collector('cpu_usage', self._collect_cpu_usage)
        self.register_collector('memory_usage', self._collect_memory_usage)
        self.register_collector('disk_io', self._collect_disk_io)
        self.register_collector('network_io', self._collect_network_io)
    
    def register_collector(self, name: str, collector_func: Callable[[], float]) -> None:
        """
        注册指标收集器
        
        Args:
            name: 指标名称
            collector_func: 收集函数
        """
        self.collectors[name] = collector_func
        logger.info(f"注册指标收集器: {name}")
    
    def _collect_cpu_usage(self) -> float:
        """收集CPU使用率"""
        return psutil.cpu_percent(interval=1)
    
    def _collect_memory_usage(self) -> float:
        """收集内存使用量（GB）"""
        memory = psutil.virtual_memory()
        return memory.used / (1024 ** 3)
    
    def _collect_disk_io(self) -> float:
        """收集磁盘I/O（MB/s）"""
        disk_io = psutil.disk_io_counters()
        if hasattr(self, '_last_disk_io'):
            read_bytes = disk_io.read_bytes - self._last_disk_io.read_bytes
            write_bytes = disk_io.write_bytes - self._last_disk_io.write_bytes
            total_bytes = read_bytes + write_bytes
            io_rate = total_bytes / (1024 ** 2) / self.collection_interval
        else:
            io_rate = 0.0
        
        self._last_disk_io = disk_io
        return io_rate
    
    def _collect_network_io(self) -> float:
        """收集网络I/O（MB/s）"""
        net_io = psutil.net_io_counters()
        if hasattr(self, '_last_net_io'):
            sent_bytes = net_io.bytes_sent - self._last_net_io.bytes_sent
            recv_bytes = net_io.bytes_recv - self._last_net_io.bytes_recv
            total_bytes = sent_bytes + recv_bytes
            io_rate = total_bytes / (1024 ** 2) / self.collection_interval
        else:
            io_rate = 0.0
        
        self._last_net_io = net_io
        return io_rate
    
    def start(self) -> None:
        """开始收集指标"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.thread.start()
        logger.info("指标收集器已启动")
    
    def stop(self) -> None:
        """停止收集指标"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("指标收集器已停止")
    
    def _collection_loop(self) -> None:
        """指标收集循环"""
        while self.running:
            try:
                timestamp = datetime.now()
                
                for name, collector_func in self.collectors.items():
                    try:
                        value = collector_func()
                        metric_point = MetricPoint(timestamp=timestamp, value=value)
                        self.metrics[name].append(metric_point)
                    except Exception as e:
                        logger.error(f"收集指标 {name} 失败: {e}")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"指标收集循环错误: {e}")
                time.sleep(1)
    
    def get_metric(self, name: str, duration_minutes: int = 10) -> List[MetricPoint]:
        """
        获取指定时间范围内的指标数据
        
        Args:
            name: 指标名称
            duration_minutes: 时间范围（分钟）
            
        Returns:
            List[MetricPoint]: 指标数据点列表
        """
        if name not in self.metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [point for point in self.metrics[name] if point.timestamp >= cutoff_time]
    
    def get_latest_metric(self, name: str) -> Optional[MetricPoint]:
        """
        获取最新的指标值
        
        Args:
            name: 指标名称
            
        Returns:
            Optional[MetricPoint]: 最新指标数据点
        """
        if name not in self.metrics or not self.metrics[name]:
            return None
        
        return self.metrics[name][-1]
    
    def get_metric_summary(self, name: str, duration_minutes: int = 10) -> Dict[str, float]:
        """
        获取指标统计摘要
        
        Args:
            name: 指标名称
            duration_minutes: 时间范围（分钟）
            
        Returns:
            Dict[str, float]: 统计摘要
        """
        data_points = self.get_metric(name, duration_minutes)
        if not data_points:
            return {}
        
        values = [point.value for point in data_points]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1]
        }
    
    def record_custom_metric(self, name: str, value: float, 
                           tags: Optional[Dict[str, str]] = None) -> None:
        """
        记录自定义指标
        
        Args:
            name: 指标名称
            value: 指标值
            tags: 标签
        """
        timestamp = datetime.now()
        metric_point = MetricPoint(
            timestamp=timestamp, 
            value=value, 
            tags=tags or {}
        )
        self.metrics[name].append(metric_point)


class AlertManager:
    """告警管理器"""
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        """
        初始化告警管理器
        
        Args:
            alert_thresholds: 告警阈值配置
        """
        self.alert_thresholds = alert_thresholds or {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_handlers = []
    
    def set_threshold(self, metric_name: str, threshold: float, 
                     comparison: str = 'greater') -> None:
        """
        设置告警阈值
        
        Args:
            metric_name: 指标名称
            threshold: 阈值
            comparison: 比较方式 ('greater', 'less', 'equal')
        """
        self.alert_thresholds[metric_name] = {
            'threshold': threshold,
            'comparison': comparison
        }
        logger.info(f"设置告警阈值: {metric_name} {comparison} {threshold}")
    
    def check_alerts(self, metrics_collector: MetricsCollector) -> List[Alert]:
        """
        检查告警条件
        
        Args:
            metrics_collector: 指标收集器
            
        Returns:
            List[Alert]: 新产生的告警列表
        """
        new_alerts = []
        
        for metric_name, config in self.alert_thresholds.items():
            latest_metric = metrics_collector.get_latest_metric(metric_name)
            if not latest_metric:
                continue
            
            threshold = config['threshold']
            comparison = config['comparison']
            value = latest_metric.value
            
            # 检查是否触发告警
            triggered = False
            if comparison == 'greater' and value > threshold:
                triggered = True
            elif comparison == 'less' and value < threshold:
                triggered = True
            elif comparison == 'equal' and abs(value - threshold) < 0.001:
                triggered = True
            
            alert_key = f"{metric_name}_{comparison}_{threshold}"
            
            if triggered:
                if alert_key not in self.active_alerts:
                    # 新告警
                    alert = Alert(
                        name=metric_name,
                        level='WARNING',
                        message=f"{metric_name} 值 {value:.2f} {comparison} 阈值 {threshold}",
                        timestamp=datetime.now(),
                        tags={'metric': metric_name, 'threshold': str(threshold)}
                    )
                    
                    self.active_alerts[alert_key] = alert
                    self.alert_history.append(alert)
                    new_alerts.append(alert)
                    
                    logger.warning(f"触发告警: {alert.message}")
            else:
                if alert_key in self.active_alerts:
                    # 告警恢复
                    alert = self.active_alerts[alert_key]
                    alert.resolved = True
                    del self.active_alerts[alert_key]
                    
                    logger.info(f"告警恢复: {alert.message}")
        
        return new_alerts
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        添加告警处理器
        
        Args:
            handler: 告警处理函数
        """
        self.alert_handlers.append(handler)
    
    def handle_alert(self, alert: Alert) -> None:
        """
        处理告警
        
        Args:
            alert: 告警对象
        """
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"告警处理器执行失败: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警列表"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """
        获取告警历史
        
        Args:
            hours: 时间范围（小时）
            
        Returns:
            List[Alert]: 告警历史列表
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]


class TestMonitoringSystem:
    """测试监控系统"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化测试监控系统
        
        Args:
            config: 监控配置
        """
        self.config = config or {}
        self.metrics_collector = MetricsCollector(
            collection_interval=self.config.get('metrics_collection_interval', 5)
        )
        self.alert_manager = AlertManager(
            alert_thresholds=self.config.get('alert_thresholds', {})
        )
        
        self.test_sessions = {}
        self.running = False
        self.monitor_thread = None
        
        # 设置默认告警阈值
        self._setup_default_thresholds()
        
        # 设置告警处理器
        self.alert_manager.add_alert_handler(self._default_alert_handler)
    
    def _setup_default_thresholds(self) -> None:
        """设置默认告警阈值"""
        default_thresholds = {
            'cpu_usage': {'threshold': 80.0, 'comparison': 'greater'},
            'memory_usage': {'threshold': 6.0, 'comparison': 'greater'},
        }
        
        for metric_name, config in default_thresholds.items():
            self.alert_manager.set_threshold(
                metric_name, 
                config['threshold'], 
                config['comparison']
            )
    
    def _default_alert_handler(self, alert: Alert) -> None:
        """默认告警处理器"""
        logger.warning(f"告警: {alert.name} - {alert.message}")
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self.running:
            return
        
        self.running = True
        self.metrics_collector.start()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("测试监控系统已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.running = False
        self.metrics_collector.stop()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("测试监控系统已停止")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.running:
            try:
                # 检查告警
                new_alerts = self.alert_manager.check_alerts(self.metrics_collector)
                
                # 处理新告警
                for alert in new_alerts:
                    self.alert_manager.handle_alert(alert)
                
                time.sleep(10)  # 每10秒检查一次告警
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(5)
    
    def start_test_session(self, session_id: str, test_name: str) -> None:
        """
        开始测试会话监控
        
        Args:
            session_id: 会话ID
            test_name: 测试名称
        """
        session_info = {
            'session_id': session_id,
            'test_name': test_name,
            'start_time': datetime.now(),
            'metrics': defaultdict(list)
        }
        
        self.test_sessions[session_id] = session_info
        logger.info(f"开始监控测试会话: {session_id} - {test_name}")
    
    def end_test_session(self, session_id: str) -> Dict[str, Any]:
        """
        结束测试会话监控
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict[str, Any]: 会话监控报告
        """
        if session_id not in self.test_sessions:
            return {}
        
        session_info = self.test_sessions[session_id]
        session_info['end_time'] = datetime.now()
        session_info['duration'] = (session_info['end_time'] - session_info['start_time']).total_seconds()
        
        # 生成会话报告
        report = self._generate_session_report(session_info)
        
        # 清理会话信息
        del self.test_sessions[session_id]
        
        logger.info(f"结束监控测试会话: {session_id}")
        return report
    
    def _generate_session_report(self, session_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成会话监控报告
        
        Args:
            session_info: 会话信息
            
        Returns:
            Dict[str, Any]: 监控报告
        """
        duration_minutes = session_info['duration'] / 60
        
        # 获取会话期间的系统指标
        system_metrics = {}
        for metric_name in ['cpu_usage', 'memory_usage', 'disk_io', 'network_io']:
            summary = self.metrics_collector.get_metric_summary(metric_name, int(duration_minutes) + 1)
            if summary:
                system_metrics[metric_name] = summary
        
        # 获取会话期间的告警
        session_alerts = [
            alert for alert in self.alert_manager.get_alert_history(int(duration_minutes / 60) + 1)
            if session_info['start_time'] <= alert.timestamp <= session_info['end_time']
        ]
        
        return {
            'session_id': session_info['session_id'],
            'test_name': session_info['test_name'],
            'start_time': session_info['start_time'].isoformat(),
            'end_time': session_info['end_time'].isoformat(),
            'duration_seconds': session_info['duration'],
            'system_metrics': system_metrics,
            'alerts_count': len(session_alerts),
            'alerts': [
                {
                    'name': alert.name,
                    'level': alert.level,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in session_alerts
            ]
        }
    
    def record_test_metric(self, session_id: str, metric_name: str, 
                          value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        记录测试指标
        
        Args:
            session_id: 会话ID
            metric_name: 指标名称
            value: 指标值
            tags: 标签
        """
        # 记录到全局指标
        self.metrics_collector.record_custom_metric(metric_name, value, tags)
        
        # 记录到会话指标
        if session_id in self.test_sessions:
            self.test_sessions[session_id]['metrics'][metric_name].append({
                'timestamp': datetime.now().isoformat(),
                'value': value,
                'tags': tags or {}
            })
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            Dict[str, Any]: 系统状态信息
        """
        status = {
            'monitoring_active': self.running,
            'active_sessions': len(self.test_sessions),
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'system_metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 获取最新系统指标
        for metric_name in ['cpu_usage', 'memory_usage', 'disk_io', 'network_io']:
            latest_metric = self.metrics_collector.get_latest_metric(metric_name)
            if latest_metric:
                status['system_metrics'][metric_name] = {
                    'value': latest_metric.value,
                    'timestamp': latest_metric.timestamp.isoformat()
                }
        
        return status
    
    def export_metrics(self, file_path: str, duration_hours: int = 1) -> None:
        """
        导出指标数据
        
        Args:
            file_path: 导出文件路径
            duration_hours: 时间范围（小时）
        """
        export_data = {
            'export_time': datetime.now().isoformat(),
            'duration_hours': duration_hours,
            'metrics': {}
        }
        
        # 导出所有指标数据
        for metric_name in self.metrics_collector.metrics.keys():
            data_points = self.metrics_collector.get_metric(metric_name, duration_hours * 60)
            export_data['metrics'][metric_name] = [
                {
                    'timestamp': point.timestamp.isoformat(),
                    'value': point.value,
                    'tags': point.tags
                }
                for point in data_points
            ]
        
        # 导出告警历史
        export_data['alerts'] = [
            {
                'name': alert.name,
                'level': alert.level,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'resolved': alert.resolved,
                'tags': alert.tags
            }
            for alert in self.alert_manager.get_alert_history(duration_hours)
        ]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"指标数据已导出到: {file_path}")


# 全局监控系统实例
_test_monitoring_system = None


def get_test_monitoring_system() -> TestMonitoringSystem:
    """
    获取全局测试监控系统实例
    
    Returns:
        TestMonitoringSystem: 监控系统实例
    """
    global _test_monitoring_system
    if _test_monitoring_system is None:
        _test_monitoring_system = TestMonitoringSystem()
    return _test_monitoring_system