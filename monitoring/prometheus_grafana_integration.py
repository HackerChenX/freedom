#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Prometheus + Grafana 监控集成模块

提供与Prometheus指标采集系统和Grafana可视化系统的集成：
1. Prometheus指标导出器
2. 自定义业务指标收集
3. Grafana仪表盘自动配置
4. 监控告警规则管理
5. 指标查询和分析接口
"""

import os
import json
import time
import yaml
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest,
    start_http_server, CONTENT_TYPE_LATEST
)
from flask import Flask, Response

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor

logger = get_logger(__name__)


@dataclass
class PrometheusConfig:
    """Prometheus配置"""
    metrics_port: int = 8090
    pushgateway_url: str = ""
    scrape_interval: str = "15s"
    job_name: str = "stock_analysis_system"
    instance: str = "localhost"


@dataclass
class GrafanaConfig:
    """Grafana配置"""
    api_url: str = "http://localhost:3000"
    api_key: str = ""
    username: str = "admin"
    password: str = "admin"
    organization_id: int = 1


class PrometheusMetricsExporter:
    """
    Prometheus指标导出器

    收集系统和业务指标，以Prometheus格式导出
    """

    def __init__(self, config: PrometheusConfig):
        """
        初始化Prometheus指标导出器

        Args:
            config: Prometheus配置
        """
        self.config = config
        self.registry = CollectorRegistry()

        # 系统指标
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU使用率百分比',
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            '内存使用率百分比',
            registry=self.registry
        )

        self.disk_usage = Gauge(
            'system_disk_usage_percent',
            '磁盘使用率百分比',
            ['mountpoint'],
            registry=self.registry
        )

        self.network_bytes = Counter(
            'system_network_bytes_total',
            '网络传输字节数',
            ['direction'],  # sent/recv
            registry=self.registry
        )

        # 应用指标
        self.http_requests = Counter(
            'http_requests_total',
            'HTTP请求总数',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP请求持续时间',
            ['method', 'endpoint'],
            registry=self.registry
        )

        self.active_connections = Gauge(
            'application_active_connections',
            '活跃连接数',
            registry=self.registry
        )

        self.error_rate = Gauge(
            'application_error_rate_percent',
            '应用错误率百分比',
            registry=self.registry
        )

        # 业务指标
        self.stock_analysis_count = Counter(
            'stock_analysis_total',
            '股票分析总数',
            ['analysis_type'],
            registry=self.registry
        )

        self.stock_analysis_duration = Histogram(
            'stock_analysis_duration_seconds',
            '股票分析耗时',
            ['analysis_type'],
            registry=self.registry
        )

        self.indicator_calculation_count = Counter(
            'indicator_calculation_total',
            '技术指标计算总数',
            ['indicator_name'],
            registry=self.registry
        )

        self.alert_count = Counter(
            'alerts_total',
            '告警总数',
            ['alert_type', 'severity'],
            registry=self.registry
        )

        self.monitored_stocks = Gauge(
            'monitored_stocks_count',
            '监控股票数量',
            registry=self.registry
        )

        self.pending_signals = Gauge(
            'pending_signals_count',
            '待处理信号数量',
            registry=self.registry
        )

        # 数据库指标
        self.database_connections = Gauge(
            'database_connections_count',
            '数据库连接数',
            ['database_type'],
            registry=self.registry
        )

        self.database_query_duration = Histogram(
            'database_query_duration_seconds',
            '数据库查询耗时',
            ['database_type', 'query_type'],
            registry=self.registry
        )

        self.database_cache_hit_ratio = Gauge(
            'database_cache_hit_ratio_percent',
            '数据库缓存命中率百分比',
            ['database_type'],
            registry=self.registry
        )

        # Flask应用用于提供指标接口
        self.flask_app = Flask(__name__)
        self._setup_flask_routes()

        # 指标收集线程
        self.metrics_thread = None
        self.collecting_active = False

        logger.info(f"Prometheus指标导出器初始化完成，端口: {config.metrics_port}")

    def _setup_flask_routes(self):
        """设置Flask路由"""

        @self.flask_app.route('/metrics')
        def metrics():
            """提供Prometheus格式的指标"""
            return Response(
                generate_latest(self.registry),
                mimetype=CONTENT_TYPE_LATEST
            )

        @self.flask_app.route('/health')
        def health():
            """健康检查接口"""
            return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}

    @exception_handler(reraise=True)
    def start_metrics_server(self):
        """启动指标服务器"""
        try:
            # 启动指标收集
            self.collecting_active = True
            self.metrics_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
            self.metrics_thread.start()

            # 启动Flask服务器
            self.flask_app.run(
                host='0.0.0.0',
                port=self.config.metrics_port,
                debug=False,
                threaded=True
            )

            logger.info(f"Prometheus指标服务器启动成功: http://0.0.0.0:{self.config.metrics_port}/metrics")

        except Exception as e:
            logger.error(f"启动指标服务器失败: {e}")
            raise

    def stop_metrics_server(self):
        """停止指标服务器"""
        self.collecting_active = False
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)
        logger.info("Prometheus指标服务器已停止")

    def _collect_metrics_loop(self):
        """指标收集循环"""
        while self.collecting_active:
            try:
                # 收集系统指标
                self._collect_system_metrics()

                # 收集业务指标
                self._collect_business_metrics()

                # 收集数据库指标
                self._collect_database_metrics()

                time.sleep(15)  # 每15秒收集一次

            except Exception as e:
                logger.error(f"收集指标失败: {e}")
                time.sleep(5)

    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            import psutil

            # CPU使用率
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage.set(cpu_percent)

            # 内存使用率
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)

            # 磁盘使用率
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_percent = (usage.used / usage.total) * 100
                    self.disk_usage.labels(mountpoint=partition.mountpoint).set(disk_percent)
                except PermissionError:
                    continue

            # 网络I/O
            network = psutil.net_io_counters()
            self.network_bytes.labels(direction='sent')._value._value = network.bytes_sent
            self.network_bytes.labels(direction='recv')._value._value = network.bytes_recv

        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")

    def _collect_business_metrics(self):
        """收集业务指标"""
        try:
            # 获取市场监控统计
            from monitoring.market_monitor import get_market_monitor
            market_monitor = get_market_monitor()
            stats = market_monitor.get_monitoring_statistics()

            self.monitored_stocks.set(stats.get('monitored_stocks', 0))

            # 告警统计
            for level, count in stats.get('level_statistics', {}).items():
                self.alert_count.labels(alert_type='market', severity=level.lower()).inc(count)

            # 获取智能告警系统统计
            from monitoring.intelligent_alert_system import get_intelligent_alert_system
from db.sql_manager import SQLManager, QueryType
            alert_system = get_intelligent_alert_system()
            alert_stats = alert_system.get_system_statistics()

            self.pending_signals.set(alert_stats.get('pending_signals', 0))

        except Exception as e:
            logger.debug(f"收集业务指标失败: {e}")

    def _collect_database_metrics(self):
        """收集数据库指标"""
        try:
            # ClickHouse指标（模拟）
            self.database_connections.labels(database_type='clickhouse').set(25)
            self.database_cache_hit_ratio.labels(database_type='clickhouse').set(92.3)

        except Exception as e:
            logger.debug(f"收集数据库指标失败: {e}")

    # 业务指标记录方法

    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """记录HTTP请求"""
        self.http_requests.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
        self.http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    def record_stock_analysis(self, analysis_type: str, duration: float):
        """记录股票分析"""
        self.stock_analysis_count.labels(analysis_type=analysis_type).inc()
        self.stock_analysis_duration.labels(analysis_type=analysis_type).observe(duration)

    def record_indicator_calculation(self, indicator_name: str):
        """记录指标计算"""
        self.indicator_calculation_count.labels(indicator_name=indicator_name).inc()

    def record_alert(self, alert_type: str, severity: str):
        """记录告警"""
        self.alert_count.labels(alert_type=alert_type, severity=severity).inc()

    def record_database_query(self, database_type: str, query_type: str, duration: float):
        """记录数据库查询"""
        self.database_query_duration.labels(database_type=database_type, query_type=query_type).observe(duration)

    def update_active_connections(self, count: int):
        """更新活跃连接数"""
        self.active_connections.set(count)

    def update_error_rate(self, rate: float):
        """更新错误率"""
        self.error_rate.set(rate)


class GrafanaDashboardManager:
    """
    Grafana仪表盘管理器

    自动创建和管理Grafana仪表盘
    """

    def __init__(self, config: GrafanaConfig):
        """
        初始化Grafana仪表盘管理器

        Args:
            config: Grafana配置
        """
        self.config = config
        self.session = requests.Session()

        # 设置认证
        if config.api_key:
            self.session.headers.update({'Authorization': f'Bearer {config.api_key}'})
        else:
            self.session.auth = (config.username, config.password)

        logger.info("Grafana仪表盘管理器初始化完成")

    @exception_handler(reraise=True)
    def create_system_dashboard(self) -> Dict[str, Any]:
        """创建系统监控仪表盘"""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "股票分析系统 - 系统监控",
                "tags": ["system", "monitoring"],
                "timezone": "browser",
                "refresh": "30s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "CPU使用率",
                        "type": "stat",
                        "targets": [{
                            "expr": "system_cpu_usage_percent",
                            "legendFormat": "CPU %"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percent",
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 70},
                                        {"color": "red", "value": 90}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "内存使用率",
                        "type": "stat",
                        "targets": [{
                            "expr": "system_memory_usage_percent",
                            "legendFormat": "Memory %"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percent",
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 70},
                                        {"color": "red", "value": 90}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "磁盘使用率",
                        "type": "stat",
                        "targets": [{
                            "expr": "system_disk_usage_percent",
                            "legendFormat": "{{mountpoint}}"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percent",
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 80},
                                        {"color": "red", "value": 95}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0}
                    },
                    {
                        "id": 4,
                        "title": "网络流量",
                        "type": "timeseries",
                        "targets": [
                            {
                                "expr": "rate(system_network_bytes_total[5m])",
                                "legendFormat": "{{direction}}"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "Bps"
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4}
                    },
                    {
                        "id": 5,
                        "title": "HTTP请求量",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "rate(http_requests_total[5m])",
                            "legendFormat": "{{method}} {{endpoint}}"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "reqps"
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4}
                    }
                ]
            },
            "overwrite": True
        }

        return self._create_dashboard(dashboard_config)

    @exception_handler(reraise=True)
    def create_business_dashboard(self) -> Dict[str, Any]:
        """创建业务监控仪表盘"""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "股票分析系统 - 业务监控",
                "tags": ["business", "stock", "analysis"],
                "timezone": "browser",
                "refresh": "30s",
                "time": {
                    "from": "now-6h",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "监控股票数量",
                        "type": "stat",
                        "targets": [{
                            "expr": "monitored_stocks_count",
                            "legendFormat": "监控股票"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "short",
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "yellow", "value": 10},
                                        {"color": "green", "value": 50}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "待处理信号",
                        "type": "stat",
                        "targets": [{
                            "expr": "pending_signals_count",
                            "legendFormat": "待处理信号"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "short",
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 20},
                                        {"color": "red", "value": 50}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "告警统计",
                        "type": "piechart",
                        "targets": [{
                            "expr": "alerts_total",
                            "legendFormat": "{{severity}}"
                        }],
                        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0}
                    },
                    {
                        "id": 4,
                        "title": "股票分析TPS",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "rate(stock_analysis_total[5m])",
                            "legendFormat": "{{analysis_type}}"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "ops"
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4}
                    },
                    {
                        "id": 5,
                        "title": "分析耗时分布",
                        "type": "histogram",
                        "targets": [{
                            "expr": "histogram_quantile(0.95, stock_analysis_duration_seconds_bucket)",
                            "legendFormat": "95th percentile"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "s"
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12}
                    },
                    {
                        "id": 6,
                        "title": "技术指标计算量",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "rate(indicator_calculation_total[5m])",
                            "legendFormat": "{{indicator_name}}"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "ops"
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12}
                    }
                ]
            },
            "overwrite": True
        }

        return self._create_dashboard(dashboard_config)

    @exception_handler(reraise=True)
    def create_database_dashboard(self) -> Dict[str, Any]:
        """创建数据库监控仪表盘"""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "股票分析系统 - 数据库监控",
                "tags": ["database", "clickhouse", "performance"],
                "timezone": "browser",
                "refresh": "30s",
                "time": {
                    "from": "now-2h",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "数据库连接数",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "database_connections_count",
                            "legendFormat": "{{database_type}}"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "short"
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "查询响应时间",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "histogram_quantile(0.95, database_query_duration_seconds_bucket)",
                            "legendFormat": "95th percentile - {{database_type}}"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "s"
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "缓存命中率",
                        "type": "stat",
                        "targets": [{
                            "expr": "database_cache_hit_ratio_percent",
                            "legendFormat": "{{database_type}}"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percent",
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "yellow", "value": 80},
                                        {"color": "green", "value": 90}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 4, "w": 12, "x": 0, "y": 8}
                    }
                ]
            },
            "overwrite": True
        }

        return self._create_dashboard(dashboard_config)

    def _create_dashboard(self, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """创建仪表盘"""
        try:
            url = f"{self.config.api_url}/api/dashboards/db"
            response = self.session.post(url, json=dashboard_config)
            response.raise_for_status()

            result = response.json()
            logger.info(f"仪表盘创建成功: {dashboard_config['dashboard']['title']}")

            return result

        except Exception as e:
            logger.error(f"创建仪表盘失败: {e}")
            return {}

    @exception_handler(reraise=True)
    def create_alert_rules(self) -> List[Dict[str, Any]]:
        """创建告警规则"""
        alert_rules = [
            {
                "alert": "HighCpuUsage",
                "expr": "system_cpu_usage_percent > 90",
                "for": "2m",
                "labels": {
                    "severity": "critical"
                },
                "annotations": {
                    "summary": "CPU使用率过高",
                    "description": "CPU使用率超过90%，当前值: {{ $value }}%"
                }
            },
            {
                "alert": "HighMemoryUsage",
                "expr": "system_memory_usage_percent > 90",
                "for": "2m",
                "labels": {
                    "severity": "critical"
                },
                "annotations": {
                    "summary": "内存使用率过高",
                    "description": "内存使用率超过90%，当前值: {{ $value }}%"
                }
            },
            {
                "alert": "HighDiskUsage",
                "expr": "system_disk_usage_percent > 95",
                "for": "1m",
                "labels": {
                    "severity": "critical"
                },
                "annotations": {
                    "summary": "磁盘空间不足",
                    "description": "磁盘 {{ $labels.mountpoint }} 使用率超过95%，当前值: {{ $value }}%"
                }
            },
            {
                "alert": "HighErrorRate",
                "expr": "application_error_rate_percent > 10",
                "for": "5m",
                "labels": {
                    "severity": "warning"
                },
                "annotations": {
                    "summary": "应用错误率过高",
                    "description": "应用错误率超过10%，当前值: {{ $value }}%"
                }
            },
            {
                "alert": "TooManyPendingSignals",
                "expr": "pending_signals_count > 100",
                "for": "10m",
                "labels": {
                    "severity": "warning"
                },
                "annotations": {
                    "summary": "待处理信号过多",
                    "description": "待处理信号数量超过100个，当前值: {{ $value }}"
                }
            }
        ]

        results = []
        for rule in alert_rules:
            try:
                # 这里应该调用Prometheus告警规则API
                # 目前只是记录日志
                logger.info(f"创建告警规则: {rule['alert']}")
                results.append(rule)

            except Exception as e:
                logger.error(f"创建告警规则失败 {rule['alert']}: {e}")

        return results


class PrometheusGrafanaIntegrator:
    """
    Prometheus + Grafana 集成器

    统一管理Prometheus指标采集和Grafana可视化
    """

    def __init__(self, prometheus_config: PrometheusConfig, grafana_config: GrafanaConfig):
        """
        初始化集成器

        Args:
            prometheus_config: Prometheus配置
            grafana_config: Grafana配置
        """
        self.prometheus_config = prometheus_config
        self.grafana_config = grafana_config

        self.metrics_exporter = PrometheusMetricsExporter(prometheus_config)
        self.dashboard_manager = GrafanaDashboardManager(grafana_config)

        self.integration_active = False

        logger.info("Prometheus + Grafana 集成器初始化完成")

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=10.0)
    def start_integration(self) -> Dict[str, Any]:
        """启动集成"""
        if self.integration_active:
            return {'status': 'already_running'}

        try:
            # 启动Prometheus指标导出器（在后台线程中）
            metrics_thread = threading.Thread(
                target=self.metrics_exporter.start_metrics_server,
                daemon=True
            )
            metrics_thread.start()

            # 等待服务器启动
            time.sleep(2)

            self.integration_active = True

            logger.info("Prometheus + Grafana 集成启动成功")

            return {
                'status': 'started',
                'prometheus_endpoint': f'http://localhost:{self.prometheus_config.metrics_port}/metrics',
                'grafana_url': self.grafana_config.api_url,
                'start_time': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"启动集成失败: {e}")
            return {'status': 'failed', 'error': str(e)}

    @exception_handler(reraise=True)
    def stop_integration(self):
        """停止集成"""
        if not self.integration_active:
            return {'status': 'not_running'}

        self.metrics_exporter.stop_metrics_server()
        self.integration_active = False

        logger.info("Prometheus + Grafana 集成已停止")
        return {'status': 'stopped', 'stop_time': datetime.now().isoformat()}

    @exception_handler(reraise=True)
    def setup_monitoring_stack(self) -> Dict[str, Any]:
        """设置完整的监控栈"""
        results = {}

        try:
            # 创建系统监控仪表盘
            system_dashboard = self.dashboard_manager.create_system_dashboard()
            results['system_dashboard'] = system_dashboard

            # 创建业务监控仪表盘
            business_dashboard = self.dashboard_manager.create_business_dashboard()
            results['business_dashboard'] = business_dashboard

            # 创建数据库监控仪表盘
            database_dashboard = self.dashboard_manager.create_database_dashboard()
            results['database_dashboard'] = database_dashboard

            # 创建告警规则
            alert_rules = self.dashboard_manager.create_alert_rules()
            results['alert_rules'] = alert_rules

            logger.info("监控栈设置完成")

        except Exception as e:
            logger.error(f"设置监控栈失败: {e}")
            results['error'] = str(e)

        return results

    def get_metrics_exporter(self) -> PrometheusMetricsExporter:
        """获取指标导出器"""
        return self.metrics_exporter

    def get_dashboard_manager(self) -> GrafanaDashboardManager:
        """获取仪表盘管理器"""
        return self.dashboard_manager

    @exception_handler(reraise=True)
    def export_configuration(self, config_dir: str = "config/monitoring") -> Dict[str, str]:
        """导出监控配置文件"""
        config_path = Path(config_dir)
        config_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        try:
            # Prometheus配置
            prometheus_config = {
                'global': {
                    'scrape_interval': self.prometheus_config.scrape_interval,
                    'evaluation_interval': '15s'
                },
                'scrape_configs': [
                    {
                        'job_name': self.prometheus_config.job_name,
                        'static_configs': [
                            {
                                'targets': [
                                    f'{self.prometheus_config.instance}:{self.prometheus_config.metrics_port}'
                                ]
                            }
                        ],
                        'metrics_path': '/metrics',
                        'scrape_interval': self.prometheus_config.scrape_interval
                    }
                ]
            }

            prometheus_file = config_path / 'prometheus.yml'
            with open(prometheus_file, 'w', encoding='utf-8') as f:
                yaml.dump(prometheus_config, f, default_flow_style=False)

            exported_files['prometheus'] = str(prometheus_file)

            # Docker Compose配置
            docker_compose = {
                'version': '3.8',
                'services': {
                    'prometheus': {
                        'image': 'prom/prometheus:latest',
                        'container_name': 'prometheus',
                        'ports': ['9090:9090'],
                        'volumes': [
                            f'{prometheus_file.absolute()}:/etc/prometheus/prometheus.yml'
                        ],
                        'command': [
                            '--config.file=/etc/prometheus/prometheus.yml',
                            '--storage.tsdb.path=/prometheus',
                            '--web.console.libraries=/etc/prometheus/console_libraries',
                            '--web.console.templates=/etc/prometheus/consoles',
                            '--storage.tsdb.retention.time=200h',
                            '--web.enable-lifecycle'
                        ]
                    },
                    'grafana': {
                        'image': 'grafana/grafana:latest',
                        'container_name': 'grafana',
                        'ports': ['3000:3000'],
                        'environment': [
                            'GF_SECURITY_ADMIN_USER=admin',
                            'GF_SECURITY_ADMIN_PASSWORD=admin'
                        ],
                        'volumes': [
                            'grafana-storage:/var/lib/grafana'
                        ]
                    }
                },
                'volumes': {
                    'grafana-storage': {}
                }
            }

            compose_file = config_path / 'docker-compose.yml'
            with open(compose_file, 'w', encoding='utf-8') as f:
                yaml.dump(docker_compose, f, default_flow_style=False)

            exported_files['docker_compose'] = str(compose_file)

            logger.info(f"监控配置已导出到: {config_path}")

        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            exported_files['error'] = str(e)

        return exported_files


# 工厂函数
def create_prometheus_grafana_integrator(
    prometheus_port: int = 8090,
    grafana_url: str = "http://localhost:3000",
    grafana_api_key: str = ""
) -> PrometheusGrafanaIntegrator:
    """
    创建Prometheus + Grafana集成器

    Args:
        prometheus_port: Prometheus指标端口
        grafana_url: Grafana API地址
        grafana_api_key: Grafana API密钥

    Returns:
        PrometheusGrafanaIntegrator: 集成器实例
    """
    prometheus_config = PrometheusConfig(metrics_port=prometheus_port)
    grafana_config = GrafanaConfig(api_url=grafana_url, api_key=grafana_api_key)

    return PrometheusGrafanaIntegrator(prometheus_config, grafana_config)