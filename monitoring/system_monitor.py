#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
系统性能监控和告警系统

提供实时系统监控、性能告警和健康检查功能
"""

import time
import psutil
import threading
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from utils.logger import getLogger
from utils.decorators import exception_handler, performance_monitor

logger = getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """告警类型"""
    SYSTEM_PERFORMANCE = "系统性能"
    DATABASE_PERFORMANCE = "数据库性能"
    MEMORY_USAGE = "内存使用"
    CPU_USAGE = "CPU使用"
    DISK_USAGE = "磁盘使用"
    QUERY_PERFORMANCE = "查询性能"
    ERROR_RATE = "错误率"
    INDICATOR_PERFORMANCE = "指标性能"


@dataclass
class SystemMetrics:
    """系统指标数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    query_count: int
    avg_query_time: float
    error_count: int
    error_rate: float
    indicator_calc_time: float
    cache_hit_rate: float
    concurrent_users: int


@dataclass
class Alert:
    """告警数据类"""
    id: str
    timestamp: datetime
    level: AlertLevel
    type: AlertType
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    resolved_time: Optional[datetime] = None


class SystemMonitor:
    """
    系统性能监控器
    
    功能特性：
    1. 实时系统资源监控
    2. 性能指标收集
    3. 智能告警
    4. 健康检查
    5. 报告生成
    """
    
    def __init__(self, 
                 monitor_interval: int = 30,
                 alert_cooldown: int = 300,
                 max_alerts: int = 1000):
        """
        初始化系统监控器
        
        Args:
            monitor_interval: 监控间隔（秒）
            alert_cooldown: 告警冷却时间（秒）
            max_alerts: 最大告警数量
        """
        self.monitor_interval = monitor_interval
        self.alert_cooldown = alert_cooldown
        self.max_alerts = max_alerts
        
        # 监控数据存储
        self.system_metrics_history: List[SystemMetrics] = []
        self.performance_metrics_history: List[PerformanceMetrics] = []
        self.alerts: List[Alert] = []
        
        # 告警规则配置
        self.alert_rules = self._get_default_alert_rules()
        
        # 告警状态跟踪
        self.last_alerts = {}
        self.alert_counts = {}
        
        # 监控状态
        self.monitoring = False
        self.monitor_thread = None
        
        # 线程安全锁
        self.metrics_lock = threading.Lock()
        self.alerts_lock = threading.Lock()
        
        # 告警回调函数
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        logger.info(f"系统监控器初始化完成 - 监控间隔: {monitor_interval}s")
    
    def _get_default_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """获取默认告警规则"""
        return {
            'high_cpu_usage': {
                'type': AlertType.CPU_USAGE,
                'level': AlertLevel.WARNING,
                'condition': lambda metrics: metrics.cpu_percent > 80,
                'message_template': "CPU使用率过高: {cpu_percent:.1f}%",
                'cooldown': 300
            },
            'critical_cpu_usage': {
                'type': AlertType.CPU_USAGE,
                'level': AlertLevel.CRITICAL,
                'condition': lambda metrics: metrics.cpu_percent > 95,
                'message_template': "CPU使用率极高: {cpu_percent:.1f}%",
                'cooldown': 300
            },
            'high_memory_usage': {
                'type': AlertType.MEMORY_USAGE,
                'level': AlertLevel.WARNING,
                'condition': lambda metrics: metrics.memory_percent > 85,
                'message_template': "内存使用率过高: {memory_percent:.1f}%",
                'cooldown': 300
            },
            'critical_memory_usage': {
                'type': AlertType.MEMORY_USAGE,
                'level': AlertLevel.CRITICAL,
                'condition': lambda metrics: metrics.memory_percent > 95,
                'message_template': "内存使用率极高: {memory_percent:.1f}%",
                'cooldown': 300
            },
            'high_disk_usage': {
                'type': AlertType.DISK_USAGE,
                'level': AlertLevel.WARNING,
                'condition': lambda metrics: metrics.disk_percent > 85,
                'message_template': "磁盘使用率过高: {disk_percent:.1f}%",
                'cooldown': 600
            },
            'slow_query_performance': {
                'type': AlertType.QUERY_PERFORMANCE,
                'level': AlertLevel.WARNING,
                'condition': lambda metrics: hasattr(metrics, 'avg_query_time') and metrics.avg_query_time > 5.0,
                'message_template': "查询性能下降: 平均响应时间 {avg_query_time:.2f}s",
                'cooldown': 300
            },
            'high_error_rate': {
                'type': AlertType.ERROR_RATE,
                'level': AlertLevel.ERROR,
                'condition': lambda metrics: hasattr(metrics, 'error_rate') and metrics.error_rate > 0.05,
                'message_template': "错误率过高: {error_rate:.2%}",
                'cooldown': 300
            }
        }
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            logger.warning("监控已在运行中")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控主循环"""
        while self.monitoring:
            try:
                # 收集系统指标
                system_metrics = self._collect_system_metrics()
                
                # 收集性能指标（如果有的话）
                performance_metrics = self._collect_performance_metrics()
                
                # 存储指标
                self._store_metrics(system_metrics, performance_metrics)
                
                # 检查告警规则
                self._check_alert_rules(system_metrics, performance_metrics)
                
                # 清理旧数据
                self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
            
            time.sleep(self.monitor_interval)
    
    @exception_handler(reraise=False, default_return=None)
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free_gb = disk.free / (1024**3)
        
        # 网络使用情况
        network = psutil.net_io_counters()
        network_sent_mb = network.bytes_sent / (1024**2)
        network_recv_mb = network.bytes_recv / (1024**2)
        
        # 进程数量
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            disk_percent=disk_percent,
            disk_free_gb=disk_free_gb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            process_count=process_count
        )
    
    def _collect_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """收集性能指标（需要与应用系统集成）"""
        # 这里需要与实际的性能监控系统集成
        # 暂时返回模拟数据
        try:
            return PerformanceMetrics(
                timestamp=datetime.now(),
                query_count=0,
                avg_query_time=0.0,
                error_count=0,
                error_rate=0.0,
                indicator_calc_time=0.0,
                cache_hit_rate=0.0,
                concurrent_users=0
            )
        except Exception:
            return None
    
    def _store_metrics(self, system_metrics: SystemMetrics, performance_metrics: Optional[PerformanceMetrics]):
        """存储指标数据"""
        with self.metrics_lock:
            self.system_metrics_history.append(system_metrics)
            
            if performance_metrics:
                self.performance_metrics_history.append(performance_metrics)
            
            # 限制历史数据数量（保留最近24小时的数据）
            max_system_records = int(24 * 3600 / self.monitor_interval)
            if len(self.system_metrics_history) > max_system_records:
                self.system_metrics_history = self.system_metrics_history[-max_system_records:]
            
            if len(self.performance_metrics_history) > max_system_records:
                self.performance_metrics_history = self.performance_metrics_history[-max_system_records:]
    
    def _check_alert_rules(self, system_metrics: SystemMetrics, performance_metrics: Optional[PerformanceMetrics]):
        """检查告警规则"""
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # 检查系统指标
                if rule['condition'](system_metrics):
                    self._trigger_alert(rule_name, rule, system_metrics)
                
                # 检查性能指标
                if performance_metrics and hasattr(rule['condition'], '__code__'):
                    # 尝试用性能指标检查条件
                    try:
                        if rule['condition'](performance_metrics):
                            self._trigger_alert(rule_name, rule, performance_metrics)
                    except Exception:
                        pass  # 忽略不兼容的条件检查
                        
            except Exception as e:
                logger.error(f"检查告警规则 {rule_name} 时出错: {e}")
    
    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], metrics: Any):
        """触发告警"""
        current_time = time.time()
        
        # 检查冷却时间
        last_alert_time = self.last_alerts.get(rule_name, 0)
        if current_time - last_alert_time < rule.get('cooldown', self.alert_cooldown):
            return
        
        # 生成告警
        alert_id = f"{rule_name}_{int(current_time)}"
        message = rule['message_template'].format(**asdict(metrics))
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            level=rule['level'],
            type=rule['type'],
            message=message,
            details=asdict(metrics)
        )
        
        # 存储告警
        with self.alerts_lock:
            self.alerts.append(alert)
            
            # 限制告警数量
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
        
        # 更新告警状态
        self.last_alerts[rule_name] = current_time
        self.alert_counts[rule_name] = self.alert_counts.get(rule_name, 0) + 1
        
        # 执行告警回调
        self._execute_alert_callbacks(alert)
        
        logger.warning(f"触发告警: {alert.level.value} - {alert.message}")
    
    def _execute_alert_callbacks(self, alert: Alert):
        """执行告警回调"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"执行告警回调失败: {e}")
    
    def _cleanup_old_data(self):
        """清理旧数据"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self.metrics_lock:
            # 清理旧的系统指标
            self.system_metrics_history = [
                m for m in self.system_metrics_history 
                if m.timestamp > cutoff_time
            ]
            
            # 清理旧的性能指标
            self.performance_metrics_history = [
                m for m in self.performance_metrics_history 
                if m.timestamp > cutoff_time
            ]
        
        with self.alerts_lock:
            # 清理旧的告警（保留最近7天）
            alert_cutoff_time = datetime.now() - timedelta(days=7)
            self.alerts = [
                a for a in self.alerts 
                if a.timestamp > alert_cutoff_time
            ]
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """获取当前系统指标"""
        with self.metrics_lock:
            if self.system_metrics_history:
                return self.system_metrics_history[-1]
        return None
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """获取最近的告警"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.alerts_lock:
            return [
                alert for alert in self.alerts 
                if alert.timestamp > cutoff_time
            ]
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """获取系统健康报告"""
        current_metrics = self.get_current_metrics()
        recent_alerts = self.get_recent_alerts(1)  # 最近1小时的告警
        
        # 计算健康分数
        health_score = self._calculate_health_score(current_metrics, recent_alerts)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health_score': health_score,
            'status': self._get_health_status(health_score),
            'current_metrics': asdict(current_metrics) if current_metrics else None,
            'recent_alerts_count': len(recent_alerts),
            'critical_alerts_count': len([a for a in recent_alerts if a.level == AlertLevel.CRITICAL]),
            'monitoring_active': self.monitoring,
            'uptime_hours': self._get_uptime_hours()
        }
    
    def _calculate_health_score(self, metrics: Optional[SystemMetrics], alerts: List[Alert]) -> float:
        """计算系统健康分数（0-100）"""
        if not metrics:
            return 50.0  # 默认分数
        
        score = 100.0
        
        # CPU使用率影响
        if metrics.cpu_percent > 90:
            score -= 30
        elif metrics.cpu_percent > 80:
            score -= 15
        elif metrics.cpu_percent > 70:
            score -= 5
        
        # 内存使用率影响
        if metrics.memory_percent > 95:
            score -= 25
        elif metrics.memory_percent > 85:
            score -= 10
        elif metrics.memory_percent > 75:
            score -= 3
        
        # 磁盘使用率影响
        if metrics.disk_percent > 95:
            score -= 20
        elif metrics.disk_percent > 85:
            score -= 8
        elif metrics.disk_percent > 75:
            score -= 2
        
        # 告警影响
        for alert in alerts:
            if alert.level == AlertLevel.CRITICAL:
                score -= 15
            elif alert.level == AlertLevel.ERROR:
                score -= 8
            elif alert.level == AlertLevel.WARNING:
                score -= 3
        
        return max(0.0, min(100.0, score))
    
    def _get_health_status(self, health_score: float) -> str:
        """根据健康分数获取状态描述"""
        if health_score >= 90:
            return "优秀"
        elif health_score >= 80:
            return "良好"
        elif health_score >= 70:
            return "一般"
        elif health_score >= 60:
            return "警告"
        else:
            return "危险"
    
    def _get_uptime_hours(self) -> float:
        """获取监控运行时间（小时）"""
        if self.system_metrics_history:
            start_time = self.system_metrics_history[0].timestamp
            return (datetime.now() - start_time).total_seconds() / 3600
        return 0.0


# 全局监控器实例
_global_monitor = None

def get_system_monitor() -> SystemMonitor:
    """获取全局系统监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemMonitor()
    return _global_monitor 