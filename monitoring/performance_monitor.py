#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
性能监控和告警系统

实时监控系统性能指标，设置告警阈值
"""

import time
import threading
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import deque, defaultdict
import psutil

from utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMetric:
    """性能指标"""
    
    def __init__(self, name: str, value: float, timestamp: Optional[datetime] = None):
        self.name = name
        self.value = value
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat()
        }


class AlertRule:
    """告警规则"""
    
    def __init__(self, 
                 name: str,
                 metric_name: str,
                 condition: str,  # 'gt', 'lt', 'eq', 'gte', 'lte'
                 threshold: float,
                 severity: str = 'warning',  # 'info', 'warning', 'error', 'critical'
                 duration: int = 60,  # 持续时间（秒）
                 callback: Optional[Callable] = None):
        self.name = name
        self.metric_name = metric_name
        self.condition = condition
        self.threshold = threshold
        self.severity = severity
        self.duration = duration
        self.callback = callback
        
        # 状态跟踪
        self.triggered_time = None
        self.is_active = False
        self.trigger_count = 0
        self.last_trigger_time = None
    
    def check(self, metric: PerformanceMetric) -> bool:
        """检查是否触发告警"""
        if metric.name != self.metric_name:
            return False
        
        # 检查条件
        triggered = False
        if self.condition == 'gt':
            triggered = metric.value > self.threshold
        elif self.condition == 'lt':
            triggered = metric.value < self.threshold
        elif self.condition == 'gte':
            triggered = metric.value >= self.threshold
        elif self.condition == 'lte':
            triggered = metric.value <= self.threshold
        elif self.condition == 'eq':
            triggered = metric.value == self.threshold
        
        current_time = datetime.now()
        
        if triggered:
            if not self.is_active:
                if self.triggered_time is None:
                    self.triggered_time = current_time
                elif (current_time - self.triggered_time).total_seconds() >= self.duration:
                    # 持续时间达到，激活告警
                    self.is_active = True
                    self.trigger_count += 1
                    self.last_trigger_time = current_time
                    return True
            else:
                # 已经激活，更新最后触发时间
                self.last_trigger_time = current_time
        else:
            # 条件不满足，重置状态
            self.triggered_time = None
            self.is_active = False
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'metric_name': self.metric_name,
            'condition': self.condition,
            'threshold': self.threshold,
            'severity': self.severity,
            'duration': self.duration,
            'is_active': self.is_active,
            'trigger_count': self.trigger_count,
            'last_trigger_time': self.last_trigger_time.isoformat() if self.last_trigger_time else None
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, 
                 collection_interval: int = 10,
                 retention_hours: int = 24,
                 enable_alerts: bool = True):
        """
        初始化性能监控器
        
        Args:
            collection_interval: 数据收集间隔（秒）
            retention_hours: 数据保留时间（小时）
            enable_alerts: 是否启用告警
        """
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.enable_alerts = enable_alerts
        
        # 指标存储
        self.metrics_history = defaultdict(lambda: deque(maxlen=int(retention_hours * 3600 / collection_interval)))
        self.current_metrics = {}
        
        # 告警规则
        self.alert_rules = []
        self.active_alerts = []
        self.alert_history = deque(maxlen=1000)
        
        # 监控状态
        self.is_running = False
        self.monitor_thread = None
        
        # 线程安全锁
        self.metrics_lock = threading.RLock()
        self.alerts_lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'total_metrics_collected': 0,
            'total_alerts_triggered': 0,
            'monitor_start_time': None,
            'last_collection_time': None
        }
        
        # 初始化默认告警规则
        self._setup_default_alert_rules()
        
        logger.info(f"性能监控器初始化完成，收集间隔: {collection_interval}秒，数据保留: {retention_hours}小时")
    
    def _setup_default_alert_rules(self):
        """设置默认告警规则"""
        default_rules = [
            # CPU使用率告警
            AlertRule(
                name='CPU使用率过高',
                metric_name='cpu_usage_percent',
                condition='gt',
                threshold=75.0,  # 降低阈值，更早预警
                severity='warning',
                duration=60
            ),
            AlertRule(
                name='CPU使用率严重过高',
                metric_name='cpu_usage_percent',
                condition='gt',
                threshold=90.0,  # 降低阈值
                severity='critical',
                duration=30
            ),
            
            # 内存使用率告警
            AlertRule(
                name='内存使用率过高',
                metric_name='memory_usage_percent',
                condition='gt',
                threshold=80.0,  # 降低阈值
                severity='warning',
                duration=60
            ),
            AlertRule(
                name='内存使用率严重过高',
                metric_name='memory_usage_percent',
                condition='gt',
                threshold=95.0,
                severity='critical',
                duration=30
            ),
            
            # 数据库查询性能告警
            AlertRule(
                name='数据库查询响应时间过长',
                metric_name='avg_query_time',
                condition='gt',
                threshold=2.0,  # 降低阈值，基于优化后的性能
                severity='warning',
                duration=120
            ),
            AlertRule(
                name='数据库查询响应时间严重过长',
                metric_name='avg_query_time',
                condition='gt',
                threshold=10.0,
                severity='error',
                duration=60
            ),
            
            # 连接池告警
            AlertRule(
                name='连接池使用率过高',
                metric_name='connection_pool_usage',
                condition='gt',
                threshold=90.0,
                severity='warning',
                duration=60
            ),
            
            # 缓存命中率告警
            AlertRule(
                name='缓存命中率过低',
                metric_name='cache_hit_rate',
                condition='lt',
                threshold=0.6,  # 提高阈值，基于优化后的缓存性能
                severity='warning',
                duration=300
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_running:
            logger.warning("监控已在运行中")
            return
        
        self.is_running = True
        self.stats['monitor_start_time'] = datetime.now()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("性能监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集系统指标
                self._collect_system_metrics()
                
                # 收集应用指标
                self._collect_application_metrics()
                
                # 检查告警
                if self.enable_alerts:
                    self._check_alerts()
                
                # 清理过期数据
                self._cleanup_expired_data()
                
                self.stats['last_collection_time'] = datetime.now()
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
            
            time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.add_metric('cpu_usage_percent', cpu_percent)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            self.add_metric('memory_usage_percent', memory.percent)
            self.add_metric('memory_available_gb', memory.available / (1024**3))
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            self.add_metric('disk_usage_percent', (disk.used / disk.total) * 100)
            
            # 网络I/O
            network = psutil.net_io_counters()
            if network:
                self.add_metric('network_bytes_sent', network.bytes_sent)
                self.add_metric('network_bytes_recv', network.bytes_recv)
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
    
    def _collect_application_metrics(self):
        """收集应用指标"""
        try:
            # 尝试获取数据管理器统计
            try:
                from db.enhanced_data_manager import get_enhanced_data_manager
                data_manager = get_enhanced_data_manager()
                stats = data_manager.get_stats()
                
                self.add_metric('avg_query_time', stats.get('avg_query_time', 0))
                self.add_metric('cache_hit_rate', stats.get('cache_hit_rate', 0))
                self.add_metric('total_queries', stats.get('total_queries', 0))
                self.add_metric('concurrent_queries', stats.get('concurrent_queries', 0))
                
                # 连接池指标
                pool_stats = stats.get('connection_pool', {})
                if pool_stats:
                    total_connections = pool_stats.get('total_connections', 1)
                    active_connections = pool_stats.get('current_active', 0)
                    usage_rate = (active_connections / total_connections) * 100 if total_connections > 0 else 0
                    self.add_metric('connection_pool_usage', usage_rate)
                    self.add_metric('connection_pool_total', total_connections)
                    self.add_metric('connection_pool_active', active_connections)
                
            except Exception as e:
                logger.debug(f"获取数据管理器统计失败: {e}")
            
            # 尝试获取连接池统计
            try:
                from db.enhanced_connection_pool import get_connection_pool
                pool = get_connection_pool()
                pool_stats = pool.get_stats()
                
                self.add_metric('connection_requests', pool_stats.get('total_requests', 0))
                self.add_metric('connection_errors', pool_stats.get('total_errors', 0))
                self.add_metric('connection_avg_response_time', pool_stats.get('avg_response_time', 0))
                
            except Exception as e:
                logger.debug(f"获取连接池统计失败: {e}")
            
        except Exception as e:
            logger.error(f"收集应用指标失败: {e}")
    
    def add_metric(self, name: str, value: float, timestamp: Optional[datetime] = None):
        """添加指标"""
        metric = PerformanceMetric(name, value, timestamp)
        
        with self.metrics_lock:
            self.metrics_history[name].append(metric)
            self.current_metrics[name] = metric
            self.stats['total_metrics_collected'] += 1
    
    def get_current_metrics(self) -> Dict[str, PerformanceMetric]:
        """获取当前指标"""
        with self.metrics_lock:
            return self.current_metrics.copy()
    
    def get_metric_history(self, name: str, hours: int = 1) -> List[PerformanceMetric]:
        """获取指标历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.metrics_lock:
            history = self.metrics_history.get(name, deque())
            return [m for m in history if m.timestamp >= cutoff_time]
    
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        with self.alerts_lock:
            self.alert_rules.append(rule)
        logger.info(f"添加告警规则: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """移除告警规则"""
        with self.alerts_lock:
            self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
        logger.info(f"移除告警规则: {rule_name}")
    
    def _check_alerts(self):
        """检查告警"""
        current_metrics = self.get_current_metrics()
        
        with self.alerts_lock:
            for rule in self.alert_rules:
                if rule.metric_name in current_metrics:
                    metric = current_metrics[rule.metric_name]
                    
                    if rule.check(metric):
                        # 触发告警
                        alert = {
                            'rule_name': rule.name,
                            'metric_name': rule.metric_name,
                            'metric_value': metric.value,
                            'threshold': rule.threshold,
                            'severity': rule.severity,
                            'timestamp': datetime.now(),
                            'message': f"{rule.name}: {rule.metric_name}={metric.value}, 阈值={rule.threshold}"
                        }
                        
                        self.active_alerts.append(alert)
                        self.alert_history.append(alert)
                        self.stats['total_alerts_triggered'] += 1
                        
                        logger.warning(f"告警触发: {alert['message']}")
                        
                        # 执行回调
                        if rule.callback:
                            try:
                                rule.callback(alert)
                            except Exception as e:
                                logger.error(f"执行告警回调失败: {e}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        with self.alerts_lock:
            return [alert.copy() for alert in self.active_alerts]
    
    def clear_alert(self, rule_name: str):
        """清除告警"""
        with self.alerts_lock:
            self.active_alerts = [a for a in self.active_alerts if a['rule_name'] != rule_name]
            
            # 重置规则状态
            for rule in self.alert_rules:
                if rule.name == rule_name:
                    rule.is_active = False
                    rule.triggered_time = None
                    break
        
        logger.info(f"清除告警: {rule_name}")
    
    def _cleanup_expired_data(self):
        """清理过期数据"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self.metrics_lock:
            for name, history in self.metrics_history.items():
                # 移除过期指标
                while history and history[0].timestamp < cutoff_time:
                    history.popleft()
        
        with self.alerts_lock:
            # 清理过期的活跃告警（超过1小时未更新）
            alert_cutoff = datetime.now() - timedelta(hours=1)
            self.active_alerts = [
                a for a in self.active_alerts 
                if a['timestamp'] >= alert_cutoff
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        with self.metrics_lock, self.alerts_lock:
            return {
                'is_running': self.is_running,
                'collection_interval': self.collection_interval,
                'retention_hours': self.retention_hours,
                'total_metrics_collected': self.stats['total_metrics_collected'],
                'total_alerts_triggered': self.stats['total_alerts_triggered'],
                'monitor_start_time': self.stats['monitor_start_time'].isoformat() if self.stats['monitor_start_time'] else None,
                'last_collection_time': self.stats['last_collection_time'].isoformat() if self.stats['last_collection_time'] else None,
                'active_alerts_count': len(self.active_alerts),
                'alert_rules_count': len(self.alert_rules),
                'metrics_types_count': len(self.metrics_history)
            }
    
    def export_metrics(self, output_file: str, hours: int = 1):
        """导出指标数据"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'time_range_hours': hours,
            'metrics': {}
        }
        
        with self.metrics_lock:
            for name, history in self.metrics_history.items():
                recent_metrics = [
                    m.to_dict() for m in history 
                    if m.timestamp >= cutoff_time
                ]
                if recent_metrics:
                    export_data['metrics'][name] = recent_metrics
        
        # 保存到文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"指标数据已导出到: {output_file}")


# 全局监控实例
_performance_monitor = None
_monitor_lock = threading.Lock()


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控实例"""
    global _performance_monitor
    
    if _performance_monitor is None:
        with _monitor_lock:
            if _performance_monitor is None:
                _performance_monitor = PerformanceMonitor()
    
    return _performance_monitor


def start_monitoring():
    """启动全局监控"""
    monitor = get_performance_monitor()
    monitor.start_monitoring()


def stop_monitoring():
    """停止全局监控"""
    global _performance_monitor

    if _performance_monitor:
        _performance_monitor.stop_monitoring()


class HealthChecker:
    """系统健康检查器"""

    def __init__(self):
        self.checks = {}
        self.last_check_results = {}

    def register_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """注册健康检查"""
        self.checks[name] = check_func
        logger.info(f"注册健康检查: {name}")

    def run_check(self, name: str) -> Dict[str, Any]:
        """运行单个健康检查"""
        if name not in self.checks:
            return {'status': 'error', 'message': f'检查 {name} 不存在'}

        try:
            start_time = time.time()
            result = self.checks[name]()
            duration = time.time() - start_time

            result.update({
                'check_name': name,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            })

            self.last_check_results[name] = result
            return result

        except Exception as e:
            error_result = {
                'status': 'error',
                'message': str(e),
                'check_name': name,
                'timestamp': datetime.now().isoformat()
            }
            self.last_check_results[name] = error_result
            return error_result

    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有健康检查"""
        results = {}
        overall_status = 'healthy'

        for name in self.checks:
            result = self.run_check(name)
            results[name] = result

            if result.get('status') == 'error':
                overall_status = 'unhealthy'
            elif result.get('status') == 'warning' and overall_status == 'healthy':
                overall_status = 'warning'

        return {
            'overall_status': overall_status,
            'checks': results,
            'timestamp': datetime.now().isoformat()
        }


def database_health_check() -> Dict[str, Any]:
    """数据库健康检查"""
    try:
        from db.enhanced_connection_pool import get_connection_pool
        pool = get_connection_pool()

        with pool.get_connection() as conn:
            start_time = time.time()
            result = conn.execute("SELECT 1")
            query_time = time.time() - start_time

            pool_stats = pool.get_stats()

            status = 'healthy'
            if query_time > 5.0:
                status = 'warning'
            elif query_time > 10.0:
                status = 'error'

            return {
                'status': status,
                'query_time': query_time,
                'connection_pool_stats': pool_stats,
                'message': f'数据库响应时间: {query_time:.3f}秒'
            }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'数据库连接失败: {e}'
        }


def system_health_check() -> Dict[str, Any]:
    """系统健康检查"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        status = 'healthy'
        issues = []

        if cpu_percent > 90:
            status = 'error'
            issues.append(f'CPU使用率过高: {cpu_percent}%')
        elif cpu_percent > 80:
            status = 'warning'
            issues.append(f'CPU使用率较高: {cpu_percent}%')

        if memory.percent > 95:
            status = 'error'
            issues.append(f'内存使用率过高: {memory.percent}%')
        elif memory.percent > 85:
            if status == 'healthy':
                status = 'warning'
            issues.append(f'内存使用率较高: {memory.percent}%')

        disk_usage = (disk.used / disk.total) * 100
        if disk_usage > 95:
            status = 'error'
            issues.append(f'磁盘使用率过高: {disk_usage:.1f}%')
        elif disk_usage > 85:
            if status == 'healthy':
                status = 'warning'
            issues.append(f'磁盘使用率较高: {disk_usage:.1f}%')

        return {
            'status': status,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk_usage,
            'issues': issues,
            'message': '系统正常' if not issues else '; '.join(issues)
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'系统检查失败: {e}'
        }


# 全局健康检查器
_health_checker = None


def get_health_checker() -> HealthChecker:
    """获取全局健康检查器"""
    global _health_checker

    if _health_checker is None:
        _health_checker = HealthChecker()

        # 注册默认检查
        _health_checker.register_check('database', database_health_check)
        _health_checker.register_check('system', system_health_check)

    return _health_checker
