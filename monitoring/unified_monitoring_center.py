#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
统一监控运维管理中心

企业级7×24小时监控运维管理中心，整合所有监控功能：
1. 企业级监控系统管理
2. Prometheus+Grafana监控栈
3. ELK日志管理系统
4. 智能多渠道告警系统
5. 自动化运维工具集
6. ClickHouse数据库监控
7. 业务指标监控仪表盘
8. 故障自愈和自动恢复
"""

import os
import json
import yaml
import time
import threading
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor

# 导入各个监控组件
from monitoring.enterprise_monitoring_system import get_enterprise_monitoring_system
from monitoring.prometheus_grafana_integration import create_prometheus_grafana_integrator
from monitoring.elk_log_management import get_elk_log_manager
from monitoring.intelligent_multi_channel_alert_system import get_intelligent_alert_manager
from monitoring.automated_ops_toolkit import get_automated_ops_toolkit
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class SystemStatus(Enum):
    """系统状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class MonitoringComponent(Enum):
    """监控组件"""
    ENTERPRISE_MONITORING = "enterprise_monitoring"
    PROMETHEUS_GRAFANA = "prometheus_grafana"
    ELK_LOGGING = "elk_logging"
    INTELLIGENT_ALERTS = "intelligent_alerts"
    AUTOMATED_OPS = "automated_ops"


@dataclass
class ComponentStatus:
    """组件状态"""
    component: MonitoringComponent
    status: SystemStatus
    uptime: float
    last_check: datetime
    metrics: Dict[str, Any]
    errors: List[str]


@dataclass
class MonitoringConfiguration:
    """监控配置"""
    enterprise_monitoring: Dict[str, Any]
    prometheus_grafana: Dict[str, Any]
    elk_logging: Dict[str, Any]
    intelligent_alerts: Dict[str, Any]
    automated_ops: Dict[str, Any]
    global_settings: Dict[str, Any]


class UnifiedMonitoringCenter:
    """
    统一监控运维管理中心

    整合所有监控组件的中央管理系统
    """

    def __init__(self, config_path: str = "config/monitoring/unified_monitoring_center.yaml"):
        """
        初始化统一监控运维管理中心

        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # 组件状态跟踪
        self.component_statuses: Dict[MonitoringComponent, ComponentStatus] = {}
        self.system_start_time = datetime.now()
        self.last_health_check = datetime.now()

        # 初始化各个监控组件
        self.components = {}
        self.monitoring_active = False

        # 统计数据
        self.performance_metrics = {
            'total_alerts_processed': 0,
            'total_auto_heals': 0,
            'total_deployments': 0,
            'total_backups': 0,
            'avg_response_time': 0.0,
            'system_uptime': 0.0
        }

        # 线程锁
        self.lock = threading.RLock()

        logger.info("统一监控运维管理中心初始化完成")

    def _load_config(self) -> MonitoringConfiguration:
        """加载监控配置"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = self._create_default_config()

            return MonitoringConfiguration(
                enterprise_monitoring=config_data.get('enterprise_monitoring', {}),
                prometheus_grafana=config_data.get('prometheus_grafana', {}),
                elk_logging=config_data.get('elk_logging', {}),
                intelligent_alerts=config_data.get('intelligent_alerts', {}),
                automated_ops=config_data.get('automated_ops', {}),
                global_settings=config_data.get('global_settings', {})
            )

        except Exception as e:
            logger.error(f"加载监控配置失败: {e}")
            return self._create_default_config()

    def _create_default_config(self) -> MonitoringConfiguration:
        """创建默认配置"""
        default_config = {
            'global_settings': {
                'monitoring_interval': 60,
                'health_check_interval': 30,
                'data_retention_days': 30,
                'max_alert_rate': 100,
                'enable_auto_healing': True,
                'enable_notifications': True
            },
            'enterprise_monitoring': {
                'enabled': True,
                'monitoring_interval': 60,
                'alert_thresholds': {
                    'cpu_warning': 70,
                    'cpu_critical': 90,
                    'memory_warning': 70,
                    'memory_critical': 90,
                    'disk_warning': 80,
                    'disk_critical': 95
                }
            },
            'prometheus_grafana': {
                'enabled': True,
                'prometheus_port': 8090,
                'grafana_url': 'http://localhost:3000',
                'auto_create_dashboards': True,
                'metrics_retention': '30d'
            },
            'elk_logging': {
                'enabled': True,
                'elasticsearch_hosts': ['localhost:9200'],
                'kibana_url': 'http://localhost:5601',
                'log_retention_days': 30,
                'auto_create_indices': True
            },
            'intelligent_alerts': {
                'enabled': True,
                'deduplication_window': 300,
                'escalation_enabled': True,
                'channels': {
                    'email': {'enabled': False},
                    'dingtalk': {'enabled': False},
                    'slack': {'enabled': False},
                    'webhook': {'enabled': False}
                }
            },
            'automated_ops': {
                'enabled': True,
                'auto_healing_enabled': True,
                'backup_enabled': True,
                'deployment_automation': True,
                'health_monitoring_interval': 60
            }
        }

        # 确保配置目录存在
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存默认配置
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

        return MonitoringConfiguration(**default_config)

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=30.0)
    def initialize_monitoring_stack(self) -> Dict[str, Any]:
        """初始化完整监控栈"""
        initialization_results = {
            'status': 'initializing',
            'components': {},
            'start_time': datetime.now(),
            'errors': []
        }

        try:
            with self.lock:
                # 1. 初始化企业级监控系统
                if self.config.enterprise_monitoring.get('enabled', True):
                    try:
                        enterprise_monitoring = get_enterprise_monitoring_system(
                            monitoring_interval=self.config.enterprise_monitoring.get('monitoring_interval', 60)
                        )
                        start_result = enterprise_monitoring.start_monitoring()
                        self.components[MonitoringComponent.ENTERPRISE_MONITORING] = enterprise_monitoring

                        initialization_results['components']['enterprise_monitoring'] = {
                            'status': 'initialized',
                            'details': start_result
                        }

                        # 更新组件状态
                        self._update_component_status(
                            MonitoringComponent.ENTERPRISE_MONITORING,
                            SystemStatus.HEALTHY,
                            {}
                        )

                        logger.info("企业级监控系统初始化成功")

                    except Exception as e:
                        error_msg = f"企业级监控系统初始化失败: {e}"
                        logger.error(error_msg)
                        initialization_results['errors'].append(error_msg)
                        initialization_results['components']['enterprise_monitoring'] = {
                            'status': 'failed',
                            'error': str(e)
                        }

                # 2. 初始化Prometheus+Grafana监控栈
                if self.config.prometheus_grafana.get('enabled', True):
                    try:
                        prometheus_grafana = create_prometheus_grafana_integrator(
                            prometheus_port=self.config.prometheus_grafana.get('prometheus_port', 8090),
                            grafana_url=self.config.prometheus_grafana.get('grafana_url', 'http://localhost:3000')
                        )

                        # 启动Prometheus+Grafana集成（在后台线程中）
                        integration_thread = threading.Thread(
                            target=self._start_prometheus_grafana,
                            args=(prometheus_grafana,),
                            daemon=True
                        )
                        integration_thread.start()

                        self.components[MonitoringComponent.PROMETHEUS_GRAFANA] = prometheus_grafana

                        initialization_results['components']['prometheus_grafana'] = {
                            'status': 'initialized',
                            'prometheus_port': self.config.prometheus_grafana.get('prometheus_port', 8090),
                            'grafana_url': self.config.prometheus_grafana.get('grafana_url')
                        }

                        # 更新组件状态
                        self._update_component_status(
                            MonitoringComponent.PROMETHEUS_GRAFANA,
                            SystemStatus.HEALTHY,
                            {'prometheus_port': self.config.prometheus_grafana.get('prometheus_port', 8090)}
                        )

                        logger.info("Prometheus+Grafana监控栈初始化成功")

                    except Exception as e:
                        error_msg = f"Prometheus+Grafana监控栈初始化失败: {e}"
                        logger.error(error_msg)
                        initialization_results['errors'].append(error_msg)
                        initialization_results['components']['prometheus_grafana'] = {
                            'status': 'failed',
                            'error': str(e)
                        }

                # 3. 初始化ELK日志管理系统
                if self.config.elk_logging.get('enabled', True):
                    try:
                        elk_manager = get_elk_log_manager(
                            elasticsearch_hosts=self.config.elk_logging.get('elasticsearch_hosts', ['localhost:9200']),
                            kibana_url=self.config.elk_logging.get('kibana_url', 'http://localhost:5601')
                        )

                        # 设置日志基础设施
                        setup_result = elk_manager.setup_logging_infrastructure()
                        self.components[MonitoringComponent.ELK_LOGGING] = elk_manager

                        initialization_results['components']['elk_logging'] = {
                            'status': 'initialized',
                            'details': setup_result
                        }

                        # 更新组件状态
                        self._update_component_status(
                            MonitoringComponent.ELK_LOGGING,
                            SystemStatus.HEALTHY,
                            {'elasticsearch_hosts': self.config.elk_logging.get('elasticsearch_hosts')}
                        )

                        logger.info("ELK日志管理系统初始化成功")

                    except Exception as e:
                        error_msg = f"ELK日志管理系统初始化失败: {e}"
                        logger.error(error_msg)
                        initialization_results['errors'].append(error_msg)
                        initialization_results['components']['elk_logging'] = {
                            'status': 'failed',
                            'error': str(e)
                        }

                # 4. 初始化智能告警系统
                if self.config.intelligent_alerts.get('enabled', True):
                    try:
                        alert_manager = get_intelligent_alert_manager()
                        self.components[MonitoringComponent.INTELLIGENT_ALERTS] = alert_manager

                        # 获取告警统计
                        alert_stats = alert_manager.get_alert_statistics()

                        initialization_results['components']['intelligent_alerts'] = {
                            'status': 'initialized',
                            'statistics': alert_stats
                        }

                        # 更新组件状态
                        self._update_component_status(
                            MonitoringComponent.INTELLIGENT_ALERTS,
                            SystemStatus.HEALTHY,
                            alert_stats
                        )

                        logger.info("智能告警系统初始化成功")

                    except Exception as e:
                        error_msg = f"智能告警系统初始化失败: {e}"
                        logger.error(error_msg)
                        initialization_results['errors'].append(error_msg)
                        initialization_results['components']['intelligent_alerts'] = {
                            'status': 'failed',
                            'error': str(e)
                        }

                # 5. 初始化自动化运维工具集
                if self.config.automated_ops.get('enabled', True):
                    try:
                        ops_toolkit = get_automated_ops_toolkit()

                        # 初始化自动化运维
                        init_result = ops_toolkit.initialize_automated_ops()
                        self.components[MonitoringComponent.AUTOMATED_OPS] = ops_toolkit

                        initialization_results['components']['automated_ops'] = {
                            'status': 'initialized',
                            'details': init_result
                        }

                        # 更新组件状态
                        self._update_component_status(
                            MonitoringComponent.AUTOMATED_OPS,
                            SystemStatus.HEALTHY,
                            init_result
                        )

                        logger.info("自动化运维工具集初始化成功")

                    except Exception as e:
                        error_msg = f"自动化运维工具集初始化失败: {e}"
                        logger.error(error_msg)
                        initialization_results['errors'].append(error_msg)
                        initialization_results['components']['automated_ops'] = {
                            'status': 'failed',
                            'error': str(e)
                        }

                # 启动统一监控
                self.monitoring_active = True
                self._start_unified_monitoring()

                # 设置最终状态
                if initialization_results['errors']:
                    initialization_results['status'] = 'partial_success'
                else:
                    initialization_results['status'] = 'success'

                initialization_results['end_time'] = datetime.now()
                initialization_results['duration'] = (
                    initialization_results['end_time'] - initialization_results['start_time']
                ).total_seconds()

                logger.info(f"监控栈初始化完成: {initialization_results['status']}")
                return initialization_results

        except Exception as e:
            logger.error(f"监控栈初始化失败: {e}")
            initialization_results['status'] = 'failed'
            initialization_results['error'] = str(e)
            return initialization_results

    def _start_prometheus_grafana(self, prometheus_grafana):
        """启动Prometheus+Grafana（后台线程）"""
        try:
            # 启动集成
            start_result = prometheus_grafana.start_integration()
            if start_result['status'] == 'started':
                # 设置监控栈（创建仪表盘）
                prometheus_grafana.setup_monitoring_stack()
                logger.info("Prometheus+Grafana集成启动成功")
            else:
                logger.error(f"Prometheus+Grafana集成启动失败: {start_result}")
        except Exception as e:
            logger.error(f"Prometheus+Grafana后台启动失败: {e}")

    def _start_unified_monitoring(self):
        """启动统一监控循环"""
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # 执行健康检查
                    self._perform_health_checks()

                    # 更新性能指标
                    self._update_performance_metrics()

                    # 检查告警条件
                    self._check_system_alerts()

                    # 休眠
                    time.sleep(self.config.global_settings.get('health_check_interval', 30))

                except Exception as e:
                    logger.error(f"统一监控循环出错: {e}")
                    time.sleep(5)

        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

        logger.info("统一监控循环已启动")

    def _update_component_status(self, component: MonitoringComponent, status: SystemStatus, metrics: Dict[str, Any]):
        """更新组件状态"""
        with self.lock:
            self.component_statuses[component] = ComponentStatus(
                component=component,
                status=status,
                uptime=(datetime.now() - self.system_start_time).total_seconds(),
                last_check=datetime.now(),
                metrics=metrics,
                errors=[]
            )

    def _perform_health_checks(self):
        """执行健康检查"""
        try:
            with self.lock:
                for component_type, component in self.components.items():
                    try:
                        # 根据组件类型执行相应的健康检查
                        if component_type == MonitoringComponent.ENTERPRISE_MONITORING:
                            status = component.get_system_status()
                            health_status = SystemStatus.HEALTHY if status['overall_status'] == 'healthy' else SystemStatus.WARNING
                            self._update_component_status(component_type, health_status, status)

                        elif component_type == MonitoringComponent.AUTOMATED_OPS:
                            overview = component.get_system_overview()
                            # 判断系统健康状态
                            health_status = SystemStatus.HEALTHY
                            if overview.get('health_monitoring', {}).get('current_status'):
                                for health_result in overview['health_monitoring']['current_status']:
                                    if health_result.get('status') in ['critical', 'down']:
                                        health_status = SystemStatus.CRITICAL
                                        break
                                    elif health_result.get('status') == 'warning':
                                        health_status = SystemStatus.WARNING

                            self._update_component_status(component_type, health_status, overview)

                        elif component_type == MonitoringComponent.INTELLIGENT_ALERTS:
                            stats = component.get_alert_statistics()
                            # 根据活跃告警数量判断状态
                            active_alerts = stats.get('total_active_alerts', 0)
                            if active_alerts > 10:
                                health_status = SystemStatus.WARNING
                            elif active_alerts > 50:
                                health_status = SystemStatus.CRITICAL
                            else:
                                health_status = SystemStatus.HEALTHY

                            self._update_component_status(component_type, health_status, stats)

                        else:
                            # 其他组件默认为健康
                            self._update_component_status(component_type, SystemStatus.HEALTHY, {})

                    except Exception as e:
                        logger.error(f"健康检查失败 {component_type.value}: {e}")
                        self._update_component_status(component_type, SystemStatus.CRITICAL, {'error': str(e)})

                self.last_health_check = datetime.now()

        except Exception as e:
            logger.error(f"执行健康检查时出错: {e}")

    def _update_performance_metrics(self):
        """更新性能指标"""
        try:
            with self.lock:
                # 系统运行时间
                self.performance_metrics['system_uptime'] = (datetime.now() - self.system_start_time).total_seconds()

                # 从各组件收集指标
                if MonitoringComponent.INTELLIGENT_ALERTS in self.components:
                    alert_stats = self.components[MonitoringComponent.INTELLIGENT_ALERTS].get_alert_statistics()
                    self.performance_metrics['total_alerts_processed'] = alert_stats.get('total_active_alerts', 0)

                if MonitoringComponent.AUTOMATED_OPS in self.components:
                    ops_overview = self.components[MonitoringComponent.AUTOMATED_OPS].get_system_overview()
                    heal_stats = ops_overview.get('auto_healing', {}).get('statistics', {})
                    backup_stats = ops_overview.get('backup_management', {}).get('statistics', {})

                    self.performance_metrics['total_auto_heals'] = heal_stats.get('total_heals', 0)
                    self.performance_metrics['total_backups'] = backup_stats.get('total_backups', 0)

        except Exception as e:
            logger.error(f"更新性能指标失败: {e}")

    def _check_system_alerts(self):
        """检查系统告警条件"""
        try:
            # 检查组件状态，如果有关键组件异常则发送告警
            critical_components = [
                component for component, status in self.component_statuses.items()
                if status.status in [SystemStatus.CRITICAL, SystemStatus.DEGRADED]
            ]

            if critical_components and MonitoringComponent.INTELLIGENT_ALERTS in self.components:
                alert_manager = self.components[MonitoringComponent.INTELLIGENT_ALERTS]

                for component in critical_components:
                    component_status = self.component_statuses[component]

                    alert_manager.send_alert(
                        name=f"{component.value}_critical",
                        description=f"监控组件 {component.value} 状态异常",
                        severity=alert_manager.AlertSeverity.CRITICAL,
                        source="unified_monitoring_center",
                        labels={
                            'component': component.value,
                            'status': component_status.status.value
                        },
                        annotations={
                            'uptime': str(component_status.uptime),
                            'last_check': component_status.last_check.isoformat(),
                            'errors': ','.join(component_status.errors)
                        }
                    )

        except Exception as e:
            logger.error(f"检查系统告警时出错: {e}")

    @exception_handler(reraise=True)
    def get_unified_dashboard(self) -> Dict[str, Any]:
        """获取统一监控仪表盘数据"""
        with self.lock:
            dashboard_data = {
                'system_overview': {
                    'status': self._get_overall_system_status(),
                    'uptime': (datetime.now() - self.system_start_time).total_seconds(),
                    'last_health_check': self.last_health_check.isoformat(),
                    'components_count': len(self.components),
                    'active_components': len([
                        status for status in self.component_statuses.values()
                        if status.status == SystemStatus.HEALTHY
                    ])
                },
                'component_statuses': {
                    component.value: {
                        'status': status.status.value,
                        'uptime': status.uptime,
                        'last_check': status.last_check.isoformat(),
                        'metrics': status.metrics,
                        'errors': status.errors
                    }
                    for component, status in self.component_statuses.items()
                },
                'performance_metrics': self.performance_metrics,
                'recent_activities': self._get_recent_activities(),
                'resource_usage': self._get_resource_usage(),
                'timestamp': datetime.now().isoformat()
            }

            return dashboard_data

    def _get_overall_system_status(self) -> str:
        """获取系统整体状态"""
        if not self.component_statuses:
            return SystemStatus.DEGRADED.value

        # 统计各状态的组件数量
        status_counts = {}
        for status in self.component_statuses.values():
            status_counts[status.status] = status_counts.get(status.status, 0) + 1

        # 判断整体状态
        if status_counts.get(SystemStatus.CRITICAL, 0) > 0:
            return SystemStatus.CRITICAL.value
        elif status_counts.get(SystemStatus.DEGRADED, 0) > 0:
            return SystemStatus.DEGRADED.value
        elif status_counts.get(SystemStatus.WARNING, 0) > 0:
            return SystemStatus.WARNING.value
        else:
            return SystemStatus.HEALTHY.value

    def _get_recent_activities(self) -> List[Dict[str, Any]]:
        """获取最近活动"""
        activities = []

        try:
            # 从告警系统获取最近告警
            if MonitoringComponent.INTELLIGENT_ALERTS in self.components:
                alert_manager = self.components[MonitoringComponent.INTELLIGENT_ALERTS]
                recent_alerts = alert_manager.get_active_alerts()

                for alert in recent_alerts[:5]:  # 最近5个告警
                    activities.append({
                        'type': 'alert',
                        'timestamp': alert.get('timestamp'),
                        'message': f"告警: {alert.get('name')} ({alert.get('severity')})",
                        'component': 'intelligent_alerts'
                    })

            # 从自动化运维获取最近自愈动作
            if MonitoringComponent.AUTOMATED_OPS in self.components:
                ops_toolkit = self.components[MonitoringComponent.AUTOMATED_OPS]
                overview = ops_toolkit.get_system_overview()

                recent_heals = overview.get('auto_healing', {}).get('recent_actions', [])
                for heal in recent_heals[:3]:  # 最近3个自愈动作
                    activities.append({
                        'type': 'auto_heal',
                        'timestamp': heal.get('timestamp'),
                        'message': f"自愈: {heal.get('service_name')} - {heal.get('action')}",
                        'component': 'automated_ops'
                    })

            # 按时间排序
            activities.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        except Exception as e:
            logger.error(f"获取最近活动失败: {e}")

        return activities[:10]  # 返回最近10个活动

    def _get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        try:
            if MonitoringComponent.ENTERPRISE_MONITORING in self.components:
                enterprise_monitoring = self.components[MonitoringComponent.ENTERPRISE_MONITORING]
                system_status = enterprise_monitoring.get_system_status()

                latest_metrics = system_status.get('latest_metrics', {})
                system_metrics = latest_metrics.get('system', {})

                return {
                    'cpu_usage': system_metrics.get('cpu_percent', 0),
                    'memory_usage': system_metrics.get('memory_percent', 0),
                    'disk_usage': system_metrics.get('disk_usage', {}),
                    'network_io': system_metrics.get('network_io', {}),
                    'process_count': system_metrics.get('process_count', 0),
                    'load_average': system_metrics.get('load_average', [])
                }
            else:
                return {}

        except Exception as e:
            logger.error(f"获取资源使用情况失败: {e}")
            return {}

    @exception_handler(reraise=True)
    def export_monitoring_configuration(self, export_dir: str = "config/monitoring/export") -> Dict[str, Any]:
        """导出监控配置"""
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        exported_configs = {
            'timestamp': datetime.now().isoformat(),
            'files': {}
        }

        try:
            # 导出统一配置
            unified_config_file = export_path / 'unified_monitoring_center.yaml'
            with open(unified_config_file, 'w', encoding='utf-8') as f:
                yaml.dump(asdict(self.config), f, default_flow_style=False, allow_unicode=True)
            exported_configs['files']['unified_config'] = str(unified_config_file)

            # 导出Prometheus+Grafana配置
            if MonitoringComponent.PROMETHEUS_GRAFANA in self.components:
                prometheus_grafana = self.components[MonitoringComponent.PROMETHEUS_GRAFANA]
                pg_configs = prometheus_grafana.export_configuration(str(export_path / 'prometheus_grafana'))
                exported_configs['files'].update(pg_configs)

            # 导出ELK配置
            if MonitoringComponent.ELK_LOGGING in self.components:
                elk_manager = self.components[MonitoringComponent.ELK_LOGGING]

                # 导出Logstash配置
                logstash_config = elk_manager.export_logstash_config(str(export_path / 'elk'))
                exported_configs['files']['logstash_config'] = logstash_config

                # 导出Docker Compose配置
                compose_config = elk_manager.export_docker_compose(str(export_path / 'elk'))
                exported_configs['files']['elk_docker_compose'] = compose_config

            # 导出运维脚本
            scripts_dir = export_path / 'scripts'
            scripts_dir.mkdir(exist_ok=True)

            self._export_deployment_scripts(scripts_dir)
            exported_configs['files']['deployment_scripts'] = str(scripts_dir)

            logger.info(f"监控配置导出完成: {export_path}")
            return exported_configs

        except Exception as e:
            logger.error(f"导出监控配置失败: {e}")
            return {'error': str(e)}

    def _export_deployment_scripts(self, scripts_dir: Path):
        """导出部署脚本"""
        # 启动脚本
        start_script = scripts_dir / 'start_monitoring.sh'
        start_script_content = """#!/bin/bash

# 股票分析系统监控栈启动脚本

echo "启动监控栈..."

# 启动Elasticsearch
docker-compose -f elk/docker-compose.yml up -d elasticsearch
sleep 30

# 启动Logstash
docker-compose -f elk/docker-compose.yml up -d logstash
sleep 10

# 启动Kibana
docker-compose -f elk/docker-compose.yml up -d kibana
sleep 30

# 启动Prometheus和Grafana
docker-compose -f prometheus_grafana/docker-compose.yml up -d

echo "监控栈启动完成"
echo "Grafana: http://localhost:3000"
echo "Kibana: http://localhost:5601"
echo "Prometheus: http://localhost:9090"
"""

        with open(start_script, 'w') as f:
            f.write(start_script_content)
        start_script.chmod(0o755)

        # 停止脚本
        stop_script = scripts_dir / 'stop_monitoring.sh'
        stop_script_content = """#!/bin/bash

# 股票分析系统监控栈停止脚本

echo "停止监控栈..."

# 停止Prometheus+Grafana
docker-compose -f prometheus_grafana/docker-compose.yml down

# 停止ELK栈
docker-compose -f elk/docker-compose.yml down

echo "监控栈已停止"
"""

        with open(stop_script, 'w') as f:
            f.write(stop_script_content)
        stop_script.chmod(0o755)

        # 健康检查脚本
        health_script = scripts_dir / 'health_check.sh'
        health_script_content = """#!/bin/bash

# 监控系统健康检查脚本

echo "检查监控系统健康状态..."

# 检查Elasticsearch
echo "检查Elasticsearch..."
curl -s http://localhost:9200/_cluster/health | jq .

# 检查Kibana
echo "检查Kibana..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:5601/

# 检查Prometheus
echo "检查Prometheus..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:9090/

# 检查Grafana
echo "检查Grafana..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/

echo "健康检查完成"
"""

        with open(health_script, 'w') as f:
            f.write(health_script_content)
        health_script.chmod(0o755)

    @exception_handler(reraise=True)
    def shutdown_monitoring_center(self) -> Dict[str, Any]:
        """关闭监控中心"""
        shutdown_results = {
            'status': 'shutting_down',
            'components_shutdown': {},
            'start_time': datetime.now()
        }

        try:
            with self.lock:
                self.monitoring_active = False

                # 关闭各个组件
                for component_type, component in self.components.items():
                    try:
                        if component_type == MonitoringComponent.ENTERPRISE_MONITORING:
                            result = component.stop_monitoring()
                            shutdown_results['components_shutdown']['enterprise_monitoring'] = result

                        elif component_type == MonitoringComponent.PROMETHEUS_GRAFANA:
                            result = component.stop_integration()
                            shutdown_results['components_shutdown']['prometheus_grafana'] = result

                        elif component_type == MonitoringComponent.ELK_LOGGING:
                            component.close()
                            shutdown_results['components_shutdown']['elk_logging'] = {'status': 'stopped'}

                        elif component_type == MonitoringComponent.INTELLIGENT_ALERTS:
                            component.close()
                            shutdown_results['components_shutdown']['intelligent_alerts'] = {'status': 'stopped'}

                        elif component_type == MonitoringComponent.AUTOMATED_OPS:
                            component.stop_all_services()
                            shutdown_results['components_shutdown']['automated_ops'] = {'status': 'stopped'}

                        logger.info(f"组件 {component_type.value} 已关闭")

                    except Exception as e:
                        error_msg = f"关闭组件 {component_type.value} 失败: {e}"
                        logger.error(error_msg)
                        shutdown_results['components_shutdown'][component_type.value] = {'status': 'failed', 'error': str(e)}

                shutdown_results['status'] = 'shutdown_complete'
                shutdown_results['end_time'] = datetime.now()

                logger.info("统一监控运维管理中心已关闭")
                return shutdown_results

        except Exception as e:
            logger.error(f"关闭监控中心失败: {e}")
            shutdown_results['status'] = 'shutdown_failed'
            shutdown_results['error'] = str(e)
            return shutdown_results


# 全局管理中心实例
_unified_monitoring_center = None


def get_unified_monitoring_center(config_path: str = "config/monitoring/unified_monitoring_center.yaml") -> UnifiedMonitoringCenter:
    """
    获取统一监控运维管理中心实例（单例模式）

    Args:
        config_path: 配置文件路径

    Returns:
        UnifiedMonitoringCenter: 统一监控运维管理中心实例
    """
    global _unified_monitoring_center

    if _unified_monitoring_center is None:
        _unified_monitoring_center = UnifiedMonitoringCenter(config_path)

    return _unified_monitoring_center


def create_unified_monitoring_center(config_path: str = "config/monitoring/unified_monitoring_center.yaml") -> UnifiedMonitoringCenter:
    """
    创建新的统一监控运维管理中心实例

    Args:
        config_path: 配置文件路径

    Returns:
        UnifiedMonitoringCenter: 新的统一监控运维管理中心实例
    """
    return UnifiedMonitoringCenter(config_path)