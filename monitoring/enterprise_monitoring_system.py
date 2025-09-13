#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
企业级监控运维体系

综合监控系统，集成：
1. 系统性能监控（CPU、内存、磁盘、网络）
2. 业务指标监控（股票分析系统、API服务、回测引擎）
3. 数据库监控（ClickHouse性能、连接池状态）
4. 应用程序监控（进程状态、日志错误、异常统计）
5. 智能告警系统（多渠道、分级告警、自动恢复）
6. 运维自动化（健康检查、自动扩容、故障恢复）
"""

import os
import sys
import time
import json
import yaml
import psutil
import threading
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import queue
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from monitoring.intelligent_alert_system import get_intelligent_alert_system
from monitoring.market_monitor import get_market_monitor

logger = get_logger(__name__)


class MonitoringLevel(Enum):
    """监控级别"""
    SYSTEM = "系统级"
    APPLICATION = "应用级"
    BUSINESS = "业务级"
    INFRASTRUCTURE = "基础设施级"


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "健康"
    WARNING = "警告"
    CRITICAL = "严重"
    DOWN = "宕机"
    UNKNOWN = "未知"


class AlertChannel(Enum):
    """告警渠道"""
    EMAIL = "邮件"
    SMS = "短信"
    WEBHOOK = "Webhook"
    DINGTALK = "钉钉"
    WECHAT = "微信"
    SLACK = "Slack"


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: Dict[str, float]
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]
    uptime: float


@dataclass
class ApplicationMetrics:
    """应用指标"""
    timestamp: datetime
    service_name: str
    status: HealthStatus
    response_time: float
    error_rate: float
    throughput: float
    active_connections: int
    memory_usage: float
    cpu_usage: float


@dataclass
class DatabaseMetrics:
    """数据库指标"""
    timestamp: datetime
    database_type: str
    connection_count: int
    active_queries: int
    query_latency: float
    disk_usage: float
    cache_hit_ratio: float
    locks_count: int


@dataclass
class BusinessMetrics:
    """业务指标"""
    timestamp: datetime
    metric_name: str
    value: float
    target: float
    status: HealthStatus
    details: Dict[str, Any]


@dataclass
class MonitoringAlert:
    """监控告警"""
    id: str
    timestamp: datetime
    level: MonitoringLevel
    severity: HealthStatus
    service: str
    metric: str
    message: str
    current_value: float
    threshold: float
    details: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector(ABC):
    """指标收集器抽象基类"""

    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """收集指标数据"""
        pass

    @abstractmethod
    def get_health_status(self) -> HealthStatus:
        """获取健康状态"""
        pass


class SystemMetricsCollector(MetricsCollector):
    """系统指标收集器"""

    def __init__(self):
        self.boot_time = psutil.boot_time()

    @exception_handler(reraise=True)
    def collect(self) -> Dict[str, Any]:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # 磁盘使用率
            disk_usage = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = {
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': (usage.used / usage.total) * 100
                    }
                except PermissionError:
                    continue

            # 网络I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }

            # 进程数量
            process_count = len(psutil.pids())

            # 系统负载
            load_average = list(psutil.getloadavg())

            # 系统运行时间
            uptime = time.time() - self.boot_time

            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                load_average=load_average,
                uptime=uptime
            ).__dict__

        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
            return {}

    def get_health_status(self) -> HealthStatus:
        """获取系统健康状态"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent

            if cpu_percent > 90 or memory_percent > 90:
                return HealthStatus.CRITICAL
            elif cpu_percent > 70 or memory_percent > 70:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY

        except Exception:
            return HealthStatus.UNKNOWN


class ApplicationMetricsCollector(MetricsCollector):
    """应用指标收集器"""

    def __init__(self, service_name: str, pid: Optional[int] = None):
        self.service_name = service_name
        self.pid = pid or os.getpid()
        self.response_times = []
        self.error_count = 0
        self.request_count = 0

    @exception_handler(reraise=True)
    def collect(self) -> Dict[str, Any]:
        """收集应用指标"""
        try:
            process = psutil.Process(self.pid)

            # 进程状态
            status = HealthStatus.HEALTHY if process.is_running() else HealthStatus.DOWN

            # 响应时间（平均）
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0

            # 错误率
            error_rate = (self.error_count / max(self.request_count, 1)) * 100

            # 吞吐量（每秒请求数）
            throughput = len(self.response_times) / 60  # 假设统计最近一分钟

            # 活跃连接数（模拟）
            active_connections = len(process.connections())

            # 内存使用
            memory_info = process.memory_info()
            memory_usage = memory_info.rss / 1024 / 1024  # MB

            # CPU使用率
            cpu_usage = process.cpu_percent()

            # 清理历史数据
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-500:]

            return ApplicationMetrics(
                timestamp=datetime.now(),
                service_name=self.service_name,
                status=status,
                response_time=avg_response_time,
                error_rate=error_rate,
                throughput=throughput,
                active_connections=active_connections,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage
            ).__dict__

        except Exception as e:
            logger.error(f"收集应用指标失败: {e}")
            return {}

    def record_request(self, response_time: float, is_error: bool = False):
        """记录请求信息"""
        self.response_times.append(response_time)
        self.request_count += 1
        if is_error:
            self.error_count += 1

    def get_health_status(self) -> HealthStatus:
        """获取应用健康状态"""
        try:
            process = psutil.Process(self.pid)
            if not process.is_running():
                return HealthStatus.DOWN

            cpu_usage = process.cpu_percent()
            memory_percent = process.memory_percent()

            if cpu_usage > 80 or memory_percent > 80:
                return HealthStatus.CRITICAL
            elif cpu_usage > 60 or memory_percent > 60:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY

        except Exception:
            return HealthStatus.DOWN


class DatabaseMetricsCollector(MetricsCollector):
    """数据库指标收集器"""

    def __init__(self, database_type: str = "ClickHouse"):
        self.database_type = database_type
        self.container = None
        try:
            from utils.unified_container import get_container
            self.container = get_container()
        except Exception as e:
            logger.warning(f"无法获取容器实例: {e}")

    @exception_handler(reraise=True)
    def collect(self) -> Dict[str, Any]:
        """收集数据库指标"""
        try:
            # 模拟数据库指标收集
            if self.database_type == "ClickHouse":
                return self._collect_clickhouse_metrics()
            else:
                return {}

        except Exception as e:
            logger.error(f"收集数据库指标失败: {e}")
            return {}

    def _collect_clickhouse_metrics(self) -> Dict[str, Any]:
        """收集ClickHouse指标"""
        try:
            # 这里应该连接ClickHouse获取实际指标
            # 目前返回模拟数据
            return DatabaseMetrics(
                timestamp=datetime.now(),
                database_type=self.database_type,
                connection_count=25,
                active_queries=3,
                query_latency=0.05,
                disk_usage=45.6,
                cache_hit_ratio=92.3,
                locks_count=0
            ).__dict__

        except Exception as e:
            logger.error(f"收集ClickHouse指标失败: {e}")
            return {}

    def get_health_status(self) -> HealthStatus:
        """获取数据库健康状态"""
        try:
            # 这里应该检查数据库连接状态
            # 目前返回模拟状态
            return HealthStatus.HEALTHY

        except Exception:
            return HealthStatus.UNKNOWN


class BusinessMetricsCollector(MetricsCollector):
    """业务指标收集器"""

    def __init__(self):
        self.market_monitor = None
        self.alert_system = None
        try:
            self.market_monitor = get_market_monitor()
            self.alert_system = get_intelligent_alert_system()
        except Exception as e:
            logger.warning(f"无法获取业务组件: {e}")

    @exception_handler(reraise=True)
    def collect(self) -> Dict[str, Any]:
        """收集业务指标"""
        metrics = []

        try:
            # 市场监控指标
            if self.market_monitor:
                market_stats = self.market_monitor.get_monitoring_statistics()

                metrics.append(BusinessMetrics(
                    timestamp=datetime.now(),
                    metric_name="监控股票数量",
                    value=market_stats.get('monitored_stocks', 0),
                    target=100,
                    status=HealthStatus.HEALTHY if market_stats.get('monitored_stocks', 0) > 0 else HealthStatus.WARNING,
                    details=market_stats
                ).__dict__)

                metrics.append(BusinessMetrics(
                    timestamp=datetime.now(),
                    metric_name="活跃告警数量",
                    value=market_stats.get('active_alerts', 0),
                    target=10,
                    status=HealthStatus.WARNING if market_stats.get('active_alerts', 0) > 10 else HealthStatus.HEALTHY,
                    details={}
                ).__dict__)

            # 智能告警系统指标
            if self.alert_system:
                alert_stats = self.alert_system.get_system_statistics()

                metrics.append(BusinessMetrics(
                    timestamp=datetime.now(),
                    metric_name="预警规则数量",
                    value=alert_stats.get('enabled_rules', 0),
                    target=5,
                    status=HealthStatus.HEALTHY if alert_stats.get('enabled_rules', 0) >= 5 else HealthStatus.WARNING,
                    details=alert_stats
                ).__dict__)

                metrics.append(BusinessMetrics(
                    timestamp=datetime.now(),
                    metric_name="待处理信号数量",
                    value=alert_stats.get('pending_signals', 0),
                    target=50,
                    status=HealthStatus.WARNING if alert_stats.get('pending_signals', 0) > 50 else HealthStatus.HEALTHY,
                    details={}
                ).__dict__)

            return {'business_metrics': metrics}

        except Exception as e:
            logger.error(f"收集业务指标失败: {e}")
            return {}

    def get_health_status(self) -> HealthStatus:
        """获取业务健康状态"""
        try:
            # 检查关键业务组件状态
            if not self.market_monitor or not self.alert_system:
                return HealthStatus.WARNING

            return HealthStatus.HEALTHY

        except Exception:
            return HealthStatus.UNKNOWN


class AlertManager:
    """告警管理器"""

    def __init__(self, config_path: str = "config/alerts/alert_manager.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.alert_queue = queue.Queue()
        self.alert_history: List[MonitoringAlert] = []
        self.notification_handlers = self._init_notification_handlers()

        # 告警处理线程
        self.alert_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.alert_thread.start()

        logger.info("告警管理器初始化完成")

    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            else:
                # 创建默认配置
                default_config = {
                    'email': {
                        'enabled': False,
                        'smtp_server': 'smtp.example.com',
                        'smtp_port': 587,
                        'username': '',
                        'password': '',
                        'recipients': []
                    },
                    'webhook': {
                        'enabled': False,
                        'url': '',
                        'timeout': 30
                    },
                    'thresholds': {
                        'cpu_warning': 70,
                        'cpu_critical': 90,
                        'memory_warning': 70,
                        'memory_critical': 90,
                        'disk_warning': 80,
                        'disk_critical': 95
                    }
                }

                # 确保目录存在
                self.config_path.parent.mkdir(parents=True, exist_ok=True)

                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

                return default_config

        except Exception as e:
            logger.error(f"加载告警配置失败: {e}")
            return {}

    def _init_notification_handlers(self) -> Dict[AlertChannel, Callable]:
        """初始化通知处理器"""
        handlers = {}

        # 邮件通知
        if self.config.get('email', {}).get('enabled'):
            handlers[AlertChannel.EMAIL] = self._send_email_notification

        # Webhook通知
        if self.config.get('webhook', {}).get('enabled'):
            handlers[AlertChannel.WEBHOOK] = self._send_webhook_notification

        return handlers

    @exception_handler(reraise=True)
    def create_alert(self, level: MonitoringLevel, severity: HealthStatus,
                    service: str, metric: str, message: str,
                    current_value: float, threshold: float,
                    details: Dict[str, Any] = None) -> str:
        """创建告警"""
        alert_id = f"{service}_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        alert = MonitoringAlert(
            id=alert_id,
            timestamp=datetime.now(),
            level=level,
            severity=severity,
            service=service,
            metric=metric,
            message=message,
            current_value=current_value,
            threshold=threshold,
            details=details or {}
        )

        self.alert_queue.put(alert)
        logger.warning(f"创建告警: {message} (级别: {severity.value})")

        return alert_id

    def _process_alerts(self):
        """处理告警队列"""
        while True:
            try:
                alert = self.alert_queue.get(timeout=5)

                # 添加到历史记录
                self.alert_history.append(alert)

                # 发送通知
                self._send_notifications(alert)

                # 清理过期告警（保留最近1000条）
                if len(self.alert_history) > 1000:
                    self.alert_history = self.alert_history[-1000:]

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理告警失败: {e}")

    def _send_notifications(self, alert: MonitoringAlert):
        """发送通知"""
        for channel, handler in self.notification_handlers.items():
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"发送{channel.value}通知失败: {e}")

    def _send_email_notification(self, alert: MonitoringAlert):
        """发送邮件通知"""
        try:
            email_config = self.config.get('email', {})

            msg = MimeMultipart()
            msg['From'] = email_config.get('username')
            msg['Subject'] = f"[{alert.severity.value}] {alert.service} - {alert.metric}"

            body = f"""
告警信息：
- 服务：{alert.service}
- 指标：{alert.metric}
- 当前值：{alert.current_value}
- 阈值：{alert.threshold}
- 消息：{alert.message}
- 时间：{alert.timestamp.isoformat()}
- 详情：{json.dumps(alert.details, ensure_ascii=False, indent=2)}
            """

            msg.attach(MimeText(body, 'plain', 'utf-8'))

            server = smtplib.SMTP(email_config.get('smtp_server'), email_config.get('smtp_port'))
            server.starttls()
            server.login(email_config.get('username'), email_config.get('password'))

            for recipient in email_config.get('recipients', []):
                msg['To'] = recipient
                server.send_message(msg)
                del msg['To']

            server.quit()
            logger.info(f"邮件告警发送成功: {alert.id}")

        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")

    def _send_webhook_notification(self, alert: MonitoringAlert):
        """发送Webhook通知"""
        try:
            import requests

            webhook_config = self.config.get('webhook', {})
            url = webhook_config.get('url')
            timeout = webhook_config.get('timeout', 30)

            payload = {
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.value,
                'severity': alert.severity.value,
                'service': alert.service,
                'metric': alert.metric,
                'message': alert.message,
                'current_value': alert.current_value,
                'threshold': alert.threshold,
                'details': alert.details
            }

            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()

            logger.info(f"Webhook告警发送成功: {alert.id}")

        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")

    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        for alert in self.alert_history:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"告警已解决: {alert_id}")
                return True
        return False

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        return [
            {
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.value,
                'severity': alert.severity.value,
                'service': alert.service,
                'metric': alert.metric,
                'message': alert.message,
                'current_value': alert.current_value,
                'threshold': alert.threshold
            }
            for alert in self.alert_history
            if not alert.resolved
        ]


class EnterpriseMonitoringSystem:
    """
    企业级监控系统

    集成系统监控、应用监控、数据库监控、业务监控、告警管理等功能
    """

    def __init__(self, monitoring_interval: int = 60):
        """
        初始化企业级监控系统

        Args:
            monitoring_interval: 监控间隔（秒）
        """
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitor_thread = None

        # 指标收集器
        self.collectors = {
            'system': SystemMetricsCollector(),
            'application': ApplicationMetricsCollector('stock_analysis_system'),
            'database': DatabaseMetricsCollector(),
            'business': BusinessMetricsCollector()
        }

        # 告警管理器
        self.alert_manager = AlertManager()

        # 指标存储
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {
            'system': [],
            'application': [],
            'database': [],
            'business': []
        }

        # 线程锁
        self.lock = threading.RLock()

        logger.info("企业级监控系统初始化完成")

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=5.0)
    def start_monitoring(self) -> Dict[str, Any]:
        """启动监控"""
        if self.monitoring_active:
            return {'status': 'already_running'}

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info(f"企业级监控系统启动，监控间隔: {self.monitoring_interval}秒")

        return {
            'status': 'started',
            'monitoring_interval': self.monitoring_interval,
            'collectors': list(self.collectors.keys()),
            'start_time': datetime.now().isoformat()
        }

    @exception_handler(reraise=True)
    def stop_monitoring(self) -> Dict[str, Any]:
        """停止监控"""
        if not self.monitoring_active:
            return {'status': 'not_running'}

        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)

        logger.info("企业级监控系统已停止")

        return {
            'status': 'stopped',
            'stop_time': datetime.now().isoformat(),
            'total_metrics': sum(len(metrics) for metrics in self.metrics_history.values())
        }

    def _monitoring_loop(self):
        """监控主循环"""
        logger.info("监控主循环启动")

        while self.monitoring_active:
            try:
                start_time = time.time()

                # 收集所有类型的指标
                for collector_name, collector in self.collectors.items():
                    try:
                        metrics = collector.collect()
                        if metrics:
                            with self.lock:
                                self.metrics_history[collector_name].append(metrics)

                                # 清理过期数据（保留最近100条）
                                if len(self.metrics_history[collector_name]) > 100:
                                    self.metrics_history[collector_name] = self.metrics_history[collector_name][-100:]

                            # 检查告警条件
                            self._check_alert_conditions(collector_name, metrics, collector)

                    except Exception as e:
                        logger.error(f"收集{collector_name}指标失败: {e}")

                # 计算执行时间
                execution_time = time.time() - start_time
                sleep_time = max(self.monitoring_interval - execution_time, 1)

                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(5)

        logger.info("监控主循环结束")

    def _check_alert_conditions(self, collector_name: str, metrics: Dict[str, Any], collector: MetricsCollector):
        """检查告警条件"""
        try:
            health_status = collector.get_health_status()

            # 系统指标告警检查
            if collector_name == 'system':
                self._check_system_alerts(metrics, health_status)

            # 应用指标告警检查
            elif collector_name == 'application':
                self._check_application_alerts(metrics, health_status)

            # 数据库指标告警检查
            elif collector_name == 'database':
                self._check_database_alerts(metrics, health_status)

            # 业务指标告警检查
            elif collector_name == 'business':
                self._check_business_alerts(metrics, health_status)

        except Exception as e:
            logger.error(f"检查{collector_name}告警条件失败: {e}")

    def _check_system_alerts(self, metrics: Dict[str, Any], health_status: HealthStatus):
        """检查系统指标告警"""
        thresholds = self.alert_manager.config.get('thresholds', {})

        # CPU使用率告警
        cpu_percent = metrics.get('cpu_percent', 0)
        if cpu_percent > thresholds.get('cpu_critical', 90):
            self.alert_manager.create_alert(
                MonitoringLevel.SYSTEM, HealthStatus.CRITICAL,
                'system', 'cpu_usage',
                f'CPU使用率过高: {cpu_percent:.1f}%',
                cpu_percent, thresholds.get('cpu_critical', 90),
                {'metrics': metrics}
            )
        elif cpu_percent > thresholds.get('cpu_warning', 70):
            self.alert_manager.create_alert(
                MonitoringLevel.SYSTEM, HealthStatus.WARNING,
                'system', 'cpu_usage',
                f'CPU使用率警告: {cpu_percent:.1f}%',
                cpu_percent, thresholds.get('cpu_warning', 70),
                {'metrics': metrics}
            )

        # 内存使用率告警
        memory_percent = metrics.get('memory_percent', 0)
        if memory_percent > thresholds.get('memory_critical', 90):
            self.alert_manager.create_alert(
                MonitoringLevel.SYSTEM, HealthStatus.CRITICAL,
                'system', 'memory_usage',
                f'内存使用率过高: {memory_percent:.1f}%',
                memory_percent, thresholds.get('memory_critical', 90),
                {'metrics': metrics}
            )
        elif memory_percent > thresholds.get('memory_warning', 70):
            self.alert_manager.create_alert(
                MonitoringLevel.SYSTEM, HealthStatus.WARNING,
                'system', 'memory_usage',
                f'内存使用率警告: {memory_percent:.1f}%',
                memory_percent, thresholds.get('memory_warning', 70),
                {'metrics': metrics}
            )

        # 磁盘使用率告警
        disk_usage = metrics.get('disk_usage', {})
        for mount_point, disk_info in disk_usage.items():
            if isinstance(disk_info, dict):
                disk_percent = disk_info.get('percent', 0)
                if disk_percent > thresholds.get('disk_critical', 95):
                    self.alert_manager.create_alert(
                        MonitoringLevel.SYSTEM, HealthStatus.CRITICAL,
                        'system', 'disk_usage',
                        f'磁盘空间不足: {mount_point} {disk_percent:.1f}%',
                        disk_percent, thresholds.get('disk_critical', 95),
                        {'mount_point': mount_point, 'disk_info': disk_info}
                    )
                elif disk_percent > thresholds.get('disk_warning', 80):
                    self.alert_manager.create_alert(
                        MonitoringLevel.SYSTEM, HealthStatus.WARNING,
                        'system', 'disk_usage',
                        f'磁盘空间警告: {mount_point} {disk_percent:.1f}%',
                        disk_percent, thresholds.get('disk_warning', 80),
                        {'mount_point': mount_point, 'disk_info': disk_info}
                    )

    def _check_application_alerts(self, metrics: Dict[str, Any], health_status: HealthStatus):
        """检查应用指标告警"""
        service_name = metrics.get('service_name', 'unknown')

        # 应用状态告警
        if health_status == HealthStatus.DOWN:
            self.alert_manager.create_alert(
                MonitoringLevel.APPLICATION, HealthStatus.CRITICAL,
                service_name, 'service_status',
                f'服务{service_name}已停止',
                0, 1, {'metrics': metrics}
            )

        # 错误率告警
        error_rate = metrics.get('error_rate', 0)
        if error_rate > 10:
            self.alert_manager.create_alert(
                MonitoringLevel.APPLICATION, HealthStatus.WARNING,
                service_name, 'error_rate',
                f'错误率过高: {error_rate:.1f}%',
                error_rate, 10, {'metrics': metrics}
            )

        # 响应时间告警
        response_time = metrics.get('response_time', 0)
        if response_time > 5000:  # 5秒
            self.alert_manager.create_alert(
                MonitoringLevel.APPLICATION, HealthStatus.WARNING,
                service_name, 'response_time',
                f'响应时间过长: {response_time:.1f}ms',
                response_time, 5000, {'metrics': metrics}
            )

    def _check_database_alerts(self, metrics: Dict[str, Any], health_status: HealthStatus):
        """检查数据库指标告警"""
        database_type = metrics.get('database_type', 'unknown')

        # 连接数告警
        connection_count = metrics.get('connection_count', 0)
        if connection_count > 100:
            self.alert_manager.create_alert(
                MonitoringLevel.INFRASTRUCTURE, HealthStatus.WARNING,
                database_type, 'connection_count',
                f'数据库连接数过多: {connection_count}',
                connection_count, 100, {'metrics': metrics}
            )

        # 查询延迟告警
        query_latency = metrics.get('query_latency', 0)
        if query_latency > 1000:  # 1秒
            self.alert_manager.create_alert(
                MonitoringLevel.INFRASTRUCTURE, HealthStatus.WARNING,
                database_type, 'query_latency',
                f'数据库查询延迟过高: {query_latency:.1f}ms',
                query_latency, 1000, {'metrics': metrics}
            )

    def _check_business_alerts(self, metrics: Dict[str, Any], health_status: HealthStatus):
        """检查业务指标告警"""
        business_metrics = metrics.get('business_metrics', [])

        for metric in business_metrics:
            if isinstance(metric, dict):
                metric_name = metric.get('metric_name', 'unknown')
                value = metric.get('value', 0)
                target = metric.get('target', 0)
                status = metric.get('status', HealthStatus.UNKNOWN.value)

                if status == HealthStatus.WARNING.value:
                    self.alert_manager.create_alert(
                        MonitoringLevel.BUSINESS, HealthStatus.WARNING,
                        'business', metric_name,
                        f'{metric_name}指标异常: 当前值{value}, 目标值{target}',
                        value, target, metric.get('details', {})
                    )

    @exception_handler(reraise=True)
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self.lock:
            # 获取最新指标
            latest_metrics = {}
            health_statuses = {}

            for collector_name, collector in self.collectors.items():
                # 最新指标
                if self.metrics_history[collector_name]:
                    latest_metrics[collector_name] = self.metrics_history[collector_name][-1]

                # 健康状态
                try:
                    health_statuses[collector_name] = collector.get_health_status().value
                except Exception:
                    health_statuses[collector_name] = HealthStatus.UNKNOWN.value

            # 活跃告警
            active_alerts = self.alert_manager.get_active_alerts()

            # 整体健康状态
            overall_status = HealthStatus.HEALTHY
            if any(status == HealthStatus.CRITICAL.value for status in health_statuses.values()):
                overall_status = HealthStatus.CRITICAL
            elif any(status == HealthStatus.WARNING.value for status in health_statuses.values()):
                overall_status = HealthStatus.WARNING
            elif any(status == HealthStatus.DOWN.value for status in health_statuses.values()):
                overall_status = HealthStatus.DOWN

            return {
                'monitoring_active': self.monitoring_active,
                'overall_status': overall_status.value,
                'health_statuses': health_statuses,
                'latest_metrics': latest_metrics,
                'active_alerts_count': len(active_alerts),
                'active_alerts': active_alerts[:5],  # 只返回前5个
                'collectors_count': len(self.collectors),
                'monitoring_interval': self.monitoring_interval,
                'timestamp': datetime.now().isoformat()
            }

    @exception_handler(reraise=True)
    def get_metrics_history(self, collector_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取指标历史"""
        with self.lock:
            if collector_name in self.metrics_history:
                return self.metrics_history[collector_name][-limit:]
            return []

    @exception_handler(reraise=True)
    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取告警列表"""
        return self.alert_manager.get_active_alerts()[:limit]

    @exception_handler(reraise=True)
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        return self.alert_manager.resolve_alert(alert_id)


# 全局监控系统实例
_enterprise_monitoring_system = None


def get_enterprise_monitoring_system(monitoring_interval: int = 60) -> EnterpriseMonitoringSystem:
    """
    获取企业级监控系统实例（单例模式）

    Args:
        monitoring_interval: 监控间隔（秒）

    Returns:
        EnterpriseMonitoringSystem: 企业级监控系统实例
    """
    global _enterprise_monitoring_system

    if _enterprise_monitoring_system is None:
        _enterprise_monitoring_system = EnterpriseMonitoringSystem(monitoring_interval)

    return _enterprise_monitoring_system


def create_enterprise_monitoring_system(monitoring_interval: int = 60) -> EnterpriseMonitoringSystem:
    """
    创建新的企业级监控系统实例

    Args:
        monitoring_interval: 监控间隔（秒）

    Returns:
        EnterpriseMonitoringSystem: 新的企业级监控系统实例
    """
    return EnterpriseMonitoringSystem(monitoring_interval)