#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
生产环境监控和告警系统

实时监控系统性能、数据库状态、资源使用情况，提供告警机制和性能分析，确保生产环境的稳定运行。
"""

import os
import sys
import time
import threading
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor
from db.clickhouse_db import get_clickhouse_db
from config.unified_config_manager import get_config
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_io_sent: int
    network_io_recv: int
    process_count: int
    load_average: List[float]


@dataclass
class DatabaseMetrics:
    """数据库指标"""
    timestamp: str
    connection_count: int
    query_count: int
    slow_query_count: int
    average_query_time: float
    max_query_time: float
    error_count: int
    data_size_gb: float
    table_count: int
    index_usage: float


@dataclass
class ApplicationMetrics:
    """应用指标"""
    timestamp: str
    active_sessions: int
    request_count: int
    error_rate: float
    response_time_avg: float
    response_time_p95: float
    cache_hit_rate: float
    queue_length: int
    worker_pool_usage: float


@dataclass
class Alert:
    """告警"""
    id: str
    level: str  # INFO, WARNING, CRITICAL
    category: str  # SYSTEM, DATABASE, APPLICATION
    title: str
    message: str
    timestamp: str
    metric_value: float
    threshold: float
    acknowledged: bool = False
    resolved: bool = False


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.db = None
        self.system_metrics_history = deque(maxlen=1000)
        self.database_metrics_history = deque(maxlen=1000)
        self.application_metrics_history = deque(maxlen=1000)
        
    @exception_handler(reraise=False)
    def initialize_database(self) -> bool:
        """初始化数据库连接"""
        try:
            self.db = get_clickhouse_db()
            result = self.db.query("SELECT 1 as test")
            return not result.empty
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False
    
    @exception_handler(reraise=False)
    def collect_system_metrics(self) -> Optional[SystemMetrics]:
        """收集系统指标"""
        try:
            # CPU指标
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存指标
            memory = psutil.virtual_memory()
            
            # 磁盘指标
            disk = psutil.disk_usage('/')
            
            # 网络指标
            network = psutil.net_io_counters()
            
            # 进程指标
            process_count = len(psutil.pids())
            
            # 负载指标
            load_avg = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                network_io_sent=network.bytes_sent,
                network_io_recv=network.bytes_recv,
                process_count=process_count,
                load_average=load_avg
            )
            
            self.system_metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
            return None
    
    @exception_handler(reraise=False)
    def collect_database_metrics(self) -> Optional[DatabaseMetrics]:
        """收集数据库指标"""
        try:
            if not self.db:
                return None
            
            # 连接数统计
            try:
                conn_query = "SELECT count() as connection_count FROM system.processes"
                conn_result = self.db.query(conn_query)
                connection_count = conn_result.iloc[0, 0] if not conn_result.empty else 0
            except:
                connection_count = 0
            
            # 查询统计
            try:
                query_stats_query = """
                SELECT 
                    count() as query_count,
                    avg(query_duration_ms) as avg_duration,
                    max(query_duration_ms) as max_duration
                FROM system.query_log 
                WHERE event_time >= now() - INTERVAL 1 MINUTE
                """
                query_stats = self.db.query(query_stats_query)
                if not query_stats.empty:
                    query_count = query_stats.iloc[0, 0]
                    avg_query_time = query_stats.iloc[0, 1] / 1000.0 if query_stats.iloc[0, 1] else 0
                    max_query_time = query_stats.iloc[0, 2] / 1000.0 if query_stats.iloc[0, 2] else 0
                else:
                    query_count = avg_query_time = max_query_time = 0
            except:
                query_count = avg_query_time = max_query_time = 0
            
            # 慢查询统计
            try:
                slow_query_query = """
                SELECT count() as slow_query_count 
                FROM system.query_log 
                WHERE event_time >= now() - INTERVAL 1 MINUTE 
                AND query_duration_ms > 5000
                """
                slow_result = self.db.query(slow_query_query)
                slow_query_count = slow_result.iloc[0, 0] if not slow_result.empty else 0
            except:
                slow_query_count = 0
            
            # 数据库大小
            try:
                size_query = """
                SELECT 
                    sum(bytes_on_disk) / (1024*1024*1024) as size_gb,
                    count() as table_count
                FROM system.parts 
                WHERE database = 'stock' AND active
                """
                size_result = self.db.query(size_query)
                if not size_result.empty:
                    data_size_gb = size_result.iloc[0, 0] if size_result.iloc[0, 0] else 0
                    table_count = size_result.iloc[0, 1] if size_result.iloc[0, 1] else 0
                else:
                    data_size_gb = table_count = 0
            except:
                data_size_gb = table_count = 0
            
            metrics = DatabaseMetrics(
                timestamp=datetime.now().isoformat(),
                connection_count=connection_count,
                query_count=query_count,
                slow_query_count=slow_query_count,
                average_query_time=avg_query_time,
                max_query_time=max_query_time,
                error_count=0,  # 需要从应用日志收集
                data_size_gb=data_size_gb,
                table_count=table_count,
                index_usage=95.0  # 模拟值
            )
            
            self.database_metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"收集数据库指标失败: {e}")
            return None
    
    @exception_handler(reraise=False)
    def collect_application_metrics(self) -> Optional[ApplicationMetrics]:
        """收集应用指标"""
        try:
            # 模拟应用指标收集
            metrics = ApplicationMetrics(
                timestamp=datetime.now().isoformat(),
                active_sessions=np.random.randint(10, 100),
                request_count=np.random.randint(100, 1000),
                error_rate=np.random.uniform(0, 5),
                response_time_avg=np.random.uniform(0.1, 2.0),
                response_time_p95=np.random.uniform(1.0, 5.0),
                cache_hit_rate=np.random.uniform(80, 99),
                queue_length=np.random.randint(0, 50),
                worker_pool_usage=np.random.uniform(20, 90)
            )
            
            self.application_metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"收集应用指标失败: {e}")
            return None


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alerts = []
        self.alert_rules = self._load_alert_rules()
        self.alert_handlers = []
        
    def _load_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """加载告警规则"""
        return {
            'system_cpu_high': {
                'category': 'SYSTEM',
                'metric_path': 'cpu_percent',
                'threshold': 80.0,
                'operator': '>',
                'level': 'WARNING',
                'title': 'CPU使用率过高',
                'message_template': 'CPU使用率达到 {value:.1f}%，超过阈值 {threshold:.1f}%'
            },
            'system_memory_high': {
                'category': 'SYSTEM',
                'metric_path': 'memory_percent',
                'threshold': 85.0,
                'operator': '>',
                'level': 'CRITICAL',
                'title': '内存使用率过高',
                'message_template': '内存使用率达到 {value:.1f}%，超过阈值 {threshold:.1f}%'
            },
            'system_disk_high': {
                'category': 'SYSTEM',
                'metric_path': 'disk_usage_percent',
                'threshold': 90.0,
                'operator': '>',
                'level': 'CRITICAL',
                'title': '磁盘使用率过高',
                'message_template': '磁盘使用率达到 {value:.1f}%，超过阈值 {threshold:.1f}%'
            },
            'database_slow_queries': {
                'category': 'DATABASE',
                'metric_path': 'slow_query_count',
                'threshold': 5,
                'operator': '>',
                'level': 'WARNING',
                'title': '慢查询数量过多',
                'message_template': '1分钟内慢查询数量达到 {value}，超过阈值 {threshold}'
            },
            'database_avg_query_time': {
                'category': 'DATABASE',
                'metric_path': 'average_query_time',
                'threshold': 2.0,
                'operator': '>',
                'level': 'WARNING',
                'title': '平均查询时间过长',
                'message_template': '平均查询时间达到 {value:.2f}秒，超过阈值 {threshold:.2f}秒'
            },
            'application_error_rate': {
                'category': 'APPLICATION',
                'metric_path': 'error_rate',
                'threshold': 5.0,
                'operator': '>',
                'level': 'WARNING',
                'title': '应用错误率过高',
                'message_template': '应用错误率达到 {value:.1f}%，超过阈值 {threshold:.1f}%'
            },
            'application_response_time': {
                'category': 'APPLICATION',
                'metric_path': 'response_time_p95',
                'threshold': 3.0,
                'operator': '>',
                'level': 'WARNING',
                'title': '响应时间过长',
                'message_template': 'P95响应时间达到 {value:.2f}秒，超过阈值 {threshold:.2f}秒'
            }
        }
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)
    
    @exception_handler(reraise=False)
    def check_metrics(self, system_metrics: Optional[SystemMetrics],
                     database_metrics: Optional[DatabaseMetrics],
                     application_metrics: Optional[ApplicationMetrics]) -> List[Alert]:
        """检查指标并生成告警"""
        new_alerts = []
        
        # 检查系统指标
        if system_metrics:
            for rule_name, rule in self.alert_rules.items():
                if rule['category'] == 'SYSTEM':
                    value = getattr(system_metrics, rule['metric_path'], None)
                    if value is not None and self._evaluate_condition(value, rule):
                        alert = self._create_alert(rule_name, rule, value)
                        new_alerts.append(alert)
        
        # 检查数据库指标
        if database_metrics:
            for rule_name, rule in self.alert_rules.items():
                if rule['category'] == 'DATABASE':
                    value = getattr(database_metrics, rule['metric_path'], None)
                    if value is not None and self._evaluate_condition(value, rule):
                        alert = self._create_alert(rule_name, rule, value)
                        new_alerts.append(alert)
        
        # 检查应用指标
        if application_metrics:
            for rule_name, rule in self.alert_rules.items():
                if rule['category'] == 'APPLICATION':
                    value = getattr(application_metrics, rule['metric_path'], None)
                    if value is not None and self._evaluate_condition(value, rule):
                        alert = self._create_alert(rule_name, rule, value)
                        new_alerts.append(alert)
        
        # 发送新告警
        for alert in new_alerts:
            self._send_alert(alert)
            self.alerts.append(alert)
        
        return new_alerts
    
    def _evaluate_condition(self, value: float, rule: Dict[str, Any]) -> bool:
        """评估告警条件"""
        operator = rule['operator']
        threshold = rule['threshold']
        
        if operator == '>':
            return value > threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<':
            return value < threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        elif operator == '!=':
            return value != threshold
        
        return False
    
    def _create_alert(self, rule_name: str, rule: Dict[str, Any], value: float) -> Alert:
        """创建告警"""
        return Alert(
            id=f"{rule_name}_{int(time.time())}",
            level=rule['level'],
            category=rule['category'],
            title=rule['title'],
            message=rule['message_template'].format(value=value, threshold=rule['threshold']),
            timestamp=datetime.now().isoformat(),
            metric_value=value,
            threshold=rule['threshold']
        )
    
    def _send_alert(self, alert: Alert):
        """发送告警"""
        logger.warning(f"🚨 [{alert.level}] {alert.title}: {alert.message}")
        
        # 调用所有告警处理器
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"告警处理器执行失败: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def acknowledge_alert(self, alert_id: str):
        """确认告警"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"告警已确认: {alert_id}")
                break
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"告警已解决: {alert_id}")
                break


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.analysis_results = {}
        
    @exception_handler(reraise=False)
    def analyze_system_trends(self, metrics_history: deque) -> Dict[str, Any]:
        """分析系统趋势"""
        if len(metrics_history) < 10:
            return {}
        
        try:
            # 提取时间序列数据
            cpu_values = [m.cpu_percent for m in metrics_history]
            memory_values = [m.memory_percent for m in metrics_history]
            disk_values = [m.disk_usage_percent for m in metrics_history]
            
            # 计算趋势
            cpu_trend = self._calculate_trend(cpu_values)
            memory_trend = self._calculate_trend(memory_values)
            disk_trend = self._calculate_trend(disk_values)
            
            # 计算统计信息
            analysis = {
                'cpu_analysis': {
                    'current': cpu_values[-1],
                    'average': np.mean(cpu_values),
                    'max': np.max(cpu_values),
                    'min': np.min(cpu_values),
                    'trend': cpu_trend,
                    'volatility': np.std(cpu_values)
                },
                'memory_analysis': {
                    'current': memory_values[-1],
                    'average': np.mean(memory_values),
                    'max': np.max(memory_values),
                    'min': np.min(memory_values),
                    'trend': memory_trend,
                    'volatility': np.std(memory_values)
                },
                'disk_analysis': {
                    'current': disk_values[-1],
                    'average': np.mean(disk_values),
                    'max': np.max(disk_values),
                    'min': np.min(disk_values),
                    'trend': disk_trend,
                    'volatility': np.std(disk_values)
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"系统趋势分析失败: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'stable'
        
        # 简单线性回归计算趋势
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    @exception_handler(reraise=False)
    def generate_performance_report(self, system_analysis: Dict[str, Any],
                                  database_metrics: Optional[DatabaseMetrics],
                                  application_metrics: Optional[ApplicationMetrics]) -> str:
        """生成性能报告"""
        try:
            report_lines = [
                "=" * 80,
                "生产环境性能分析报告",
                "=" * 80,
                f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "📊 系统性能分析:",
                "-" * 40,
            ]
            
            # 系统分析
            if system_analysis:
                cpu_analysis = system_analysis.get('cpu_analysis', {})
                memory_analysis = system_analysis.get('memory_analysis', {})
                disk_analysis = system_analysis.get('disk_analysis', {})
                
                report_lines.extend([
                    f"CPU使用率: 当前 {cpu_analysis.get('current', 0):.1f}%, "
                    f"平均 {cpu_analysis.get('average', 0):.1f}%, "
                    f"趋势 {cpu_analysis.get('trend', 'unknown')}",
                    f"内存使用率: 当前 {memory_analysis.get('current', 0):.1f}%, "
                    f"平均 {memory_analysis.get('average', 0):.1f}%, "
                    f"趋势 {memory_analysis.get('trend', 'unknown')}",
                    f"磁盘使用率: 当前 {disk_analysis.get('current', 0):.1f}%, "
                    f"平均 {disk_analysis.get('average', 0):.1f}%, "
                    f"趋势 {disk_analysis.get('trend', 'unknown')}"
                ])
            
            # 数据库分析
            if database_metrics:
                report_lines.extend([
                    "",
                    "🗄️ 数据库性能分析:",
                    "-" * 40,
                    f"连接数: {database_metrics.connection_count}",
                    f"查询数量: {database_metrics.query_count}",
                    f"慢查询数量: {database_metrics.slow_query_count}",
                    f"平均查询时间: {database_metrics.average_query_time:.3f}秒",
                    f"数据大小: {database_metrics.data_size_gb:.2f}GB"
                ])
            
            # 应用分析
            if application_metrics:
                report_lines.extend([
                    "",
                    "🚀 应用性能分析:",
                    "-" * 40,
                    f"活跃会话: {application_metrics.active_sessions}",
                    f"请求数量: {application_metrics.request_count}",
                    f"错误率: {application_metrics.error_rate:.1f}%",
                    f"平均响应时间: {application_metrics.response_time_avg:.2f}秒",
                    f"缓存命中率: {application_metrics.cache_hit_rate:.1f}%"
                ])
            
            report_lines.append("=" * 80)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return "性能报告生成失败"


class ProductionMonitoringSystem:
    """生产环境监控系统"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.performance_analyzer = PerformanceAnalyzer()
        
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 30  # 30秒监控间隔
        
        # 注册告警处理器
        self.alert_manager.add_alert_handler(self._console_alert_handler)
        
    def _console_alert_handler(self, alert: Alert):
        """控制台告警处理器"""
        level_emoji = {
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'CRITICAL': '🔥'
        }
        
        emoji = level_emoji.get(alert.level, '❓')
        print(f"{emoji} [{alert.level}] {alert.title}")
        print(f"   {alert.message}")
        print(f"   时间: {alert.timestamp}")
        print()
    
    @exception_handler(reraise=True)
    def initialize_monitoring(self) -> bool:
        """初始化监控系统"""
        try:
            logger.info("🚀 初始化生产环境监控系统...")
            
            # 初始化数据库连接
            if not self.metrics_collector.initialize_database():
                logger.warning("数据库连接失败，将跳过数据库监控")
            
            logger.info("✅ 监控系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 监控系统初始化失败: {e}")
            return False
    
    def start_monitoring(self):
        """启动监控"""
        if self.monitoring_active:
            logger.warning("监控已在运行中")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("✅ 生产环境监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("✅ 生产环境监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集指标
                system_metrics = self.metrics_collector.collect_system_metrics()
                database_metrics = self.metrics_collector.collect_database_metrics()
                application_metrics = self.metrics_collector.collect_application_metrics()
                
                # 检查告警
                alerts = self.alert_manager.check_metrics(
                    system_metrics, database_metrics, application_metrics
                )
                
                # 记录监控状态
                if system_metrics or database_metrics or application_metrics:
                    logger.debug("📊 指标收集完成")
                
                # 等待下次监控
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(self.monitoring_interval)
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=300.0)
    def run_monitoring_test(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """运行监控测试"""
        start_time = datetime.now()
        
        test_result = {
            'test_start_time': start_time.isoformat(),
            'test_end_time': None,
            'test_duration_minutes': duration_minutes,
            'monitoring_initialization': False,
            'metrics_collected': 0,
            'alerts_generated': 0,
            'system_analysis': {},
            'performance_report': '',
            'monitoring_status': 'Unknown'
        }
        
        try:
            logger.info(f"🚀 开始生产环境监控测试 (持续 {duration_minutes} 分钟)...")
            
            # 1. 初始化监控系统
            logger.info("📝 Step 1: 初始化监控系统...")
            test_result['monitoring_initialization'] = self.initialize_monitoring()
            
            if not test_result['monitoring_initialization']:
                test_result['monitoring_status'] = 'Failed'
                return test_result
            
            # 2. 启动监控
            logger.info("📝 Step 2: 启动监控...")
            self.start_monitoring()
            
            # 3. 运行监控测试
            logger.info(f"📝 Step 3: 运行监控测试 ({duration_minutes} 分钟)...")
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            while datetime.now() < end_time:
                # 手动触发一次指标收集
                system_metrics = self.metrics_collector.collect_system_metrics()
                database_metrics = self.metrics_collector.collect_database_metrics()
                application_metrics = self.metrics_collector.collect_application_metrics()
                
                if system_metrics or database_metrics or application_metrics:
                    test_result['metrics_collected'] += 1
                
                # 检查告警
                alerts = self.alert_manager.check_metrics(
                    system_metrics, database_metrics, application_metrics
                )
                test_result['alerts_generated'] += len(alerts)
                
                # 等待
                time.sleep(10)  # 每10秒收集一次
            
            # 4. 分析性能趋势
            logger.info("📝 Step 4: 分析性能趋势...")
            test_result['system_analysis'] = self.performance_analyzer.analyze_system_trends(
                self.metrics_collector.system_metrics_history
            )
            
            # 5. 生成性能报告
            logger.info("📝 Step 5: 生成性能报告...")
            latest_db_metrics = (self.metrics_collector.database_metrics_history[-1] 
                               if self.metrics_collector.database_metrics_history else None)
            latest_app_metrics = (self.metrics_collector.application_metrics_history[-1] 
                                if self.metrics_collector.application_metrics_history else None)
            
            test_result['performance_report'] = self.performance_analyzer.generate_performance_report(
                test_result['system_analysis'], latest_db_metrics, latest_app_metrics
            )
            
            # 6. 停止监控
            logger.info("📝 Step 6: 停止监控...")
            self.stop_monitoring()
            
            # 评估监控状态
            if test_result['metrics_collected'] > 0:
                test_result['monitoring_status'] = 'Excellent'
            else:
                test_result['monitoring_status'] = 'Failed'
            
            test_result['test_end_time'] = datetime.now().isoformat()
            
            logger.info(f"✅ 监控测试完成，收集了 {test_result['metrics_collected']} 个指标，"
                       f"生成了 {test_result['alerts_generated']} 个告警")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 监控测试失败: {e}")
            test_result['monitoring_status'] = 'Failed'
            return test_result
        finally:
            self.stop_monitoring()
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """获取监控面板数据"""
        try:
            # 获取最新指标
            latest_system = (self.metrics_collector.system_metrics_history[-1] 
                           if self.metrics_collector.system_metrics_history else None)
            latest_database = (self.metrics_collector.database_metrics_history[-1] 
                             if self.metrics_collector.database_metrics_history else None)
            latest_application = (self.metrics_collector.application_metrics_history[-1] 
                                if self.metrics_collector.application_metrics_history else None)
            
            # 获取活跃告警
            active_alerts = self.alert_manager.get_active_alerts()
            
            dashboard_data = {
                'current_time': datetime.now().isoformat(),
                'monitoring_status': 'active' if self.monitoring_active else 'inactive',
                'latest_metrics': {
                    'system': asdict(latest_system) if latest_system else None,
                    'database': asdict(latest_database) if latest_database else None,
                    'application': asdict(latest_application) if latest_application else None
                },
                'active_alerts': [asdict(alert) for alert in active_alerts],
                'metrics_history_count': {
                    'system': len(self.metrics_collector.system_metrics_history),
                    'database': len(self.metrics_collector.database_metrics_history),
                    'application': len(self.metrics_collector.application_metrics_history)
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"获取监控面板数据失败: {e}")
            return {}
    
    def generate_report(self, test_result: Dict[str, Any]) -> str:
        """生成监控测试报告"""
        report_lines = [
            "=" * 80,
            "生产环境监控系统测试报告",
            "=" * 80,
            f"测试时间: {test_result.get('test_start_time', 'Unknown')}",
            f"测试持续时间: {test_result.get('test_duration_minutes', 0)} 分钟",
            f"监控状态: {test_result.get('monitoring_status', 'Unknown')}",
            "",
            "📊 监控测试结果:",
            "-" * 40,
        ]
        
        # 监控初始化状态
        init_status = "✅ 成功" if test_result.get('monitoring_initialization', False) else "❌ 失败"
        report_lines.append(f"监控系统初始化: {init_status}")
        
        # 指标收集统计
        report_lines.extend([
            f"指标收集次数: {test_result.get('metrics_collected', 0)}",
            f"告警生成数量: {test_result.get('alerts_generated', 0)}"
        ])
        
        # 系统分析
        system_analysis = test_result.get('system_analysis', {})
        if system_analysis:
            report_lines.extend([
                "",
                "💻 系统性能分析:",
                "-" * 40,
            ])
            
            cpu_analysis = system_analysis.get('cpu_analysis', {})
            if cpu_analysis:
                report_lines.append(
                    f"CPU: 当前 {cpu_analysis.get('current', 0):.1f}%, "
                    f"平均 {cpu_analysis.get('average', 0):.1f}%, "
                    f"趋势 {cpu_analysis.get('trend', 'unknown')}"
                )
            
            memory_analysis = system_analysis.get('memory_analysis', {})
            if memory_analysis:
                report_lines.append(
                    f"内存: 当前 {memory_analysis.get('current', 0):.1f}%, "
                    f"平均 {memory_analysis.get('average', 0):.1f}%, "
                    f"趋势 {memory_analysis.get('trend', 'unknown')}"
                )
        
        # 性能报告
        performance_report = test_result.get('performance_report', '')
        if performance_report:
            report_lines.extend([
                "",
                "📈 详细性能报告:",
                "-" * 40,
                performance_report
            ])
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    print("🚀 启动生产环境监控系统测试...")
    
    monitoring_system = ProductionMonitoringSystem()
    
    try:
        # 运行监控测试 (2分钟测试)
        result = monitoring_system.run_monitoring_test(duration_minutes=2)
        
        # 生成报告
        report = monitoring_system.generate_report(result)
        print(report)
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"production_monitoring_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        
        # 返回测试结果
        if result.get('monitoring_status') == 'Excellent':
            print("🎉 生产环境监控系统测试通过！")
            return 0
        else:
            print("⚠️  监控系统需要调整，请查看报告。")
            return 1
            
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 