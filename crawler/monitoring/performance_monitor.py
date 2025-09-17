"""
性能监控模块

监控爬虫系统的性能指标，包括响应时间、成功率、资源使用等
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class Performance_metrics:
    """性能指标类"""

    def __init__(self, max_history=1000):
        self.max_history = max_history

        # 基础指标
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # 响应时间统计
        self.response_times = deque(maxlen=max_history)
        self.avg_response_time = 0.0
        self.min_response_time = float('inf')
        self.max_response_time = 0.0

        # 错误统计
        self.error_counts = defaultdict(int)
        self.error_history = deque(maxlen=max_history)

        # 时间窗口统计
        self.hourly_stats = defaultdict(lambda: {'requests': 0, 'success': 0, 'errors': 0})

        # 系统资源
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.disk_usage = 0.0

        self.start_time = datetime.now()

    def record_request_Monitor_Performance_Monitor_Performance_Monitor_performancemonitor(self, success: bool, response_time: float, error_type: str = None):
        """记录请求"""
        self.total_requests += 1

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_counts[error_type] += 1
                self.error_history.append({
                    'timestamp': datetime.now(),
                    'error_type': error_type
                })

        # 记录响应时间
        if response_time > 0:
            self.response_times.append(response_time)
            self.min_response_time = min(self.min_response_time, response_time)
            self.max_response_time = max(self.max_response_time, response_time)

            if self.response_times:
                self.avg_response_time = sum(self.response_times) / len(self.response_times)

        # 更新小时统计
        hour_key = datetime.now().strftime('%Y-%m-%d %H')
        self.hourly_stats[hour_key]['requests'] += 1
        if success:
            self.hourly_stats[hour_key]['success'] += 1
        else:
            self.hourly_stats[hour_key]['errors'] += 1

    def update_system_metrics(self):
        """更新系统资源指标"""
        try:
            # 简化版本，避免依赖psutil
            import os

            # CPU使用率（简化版）
            try:
                with open('/proc/loadavg', 'r') as f:
                    load_avg = float(f.read().split()[0])
                    self.cpu_usage = min(load_avg * 100, 100.0)
            except:
                self.cpu_usage = 0.0

            # 内存使用率（简化版）
            try:
                with open('/proc/meminfo', 'r') as f:
                    lines = f.readlines()
                    total = int([line for line in lines if 'MemTotal' in line][0].split()[1])
                    available = int([line for line in lines if 'MemAvailable' in line][0].split()[1])
                    self.memory_usage = ((total - available) / total) * 100
            except:
                self.memory_usage = 0.0

        except Exception as e:
            logger.warning(f"更新系统指标失败: {e}")

    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def get_error_rate(self) -> float:
        """获取错误率"""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100

    def get_requests_per_hour(self) -> float:
        """获取每小时请求数"""
        runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        if runtime_hours == 0:
            return 0.0
        return self.total_requests / runtime_hours

    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.get_success_rate(),
            'error_rate': self.get_error_rate(),
            'avg_response_time': self.avg_response_time,
            'min_response_time': self.min_response_time if self.min_response_time != float('inf') else 0,
            'max_response_time': self.max_response_time,
            'requests_per_hour': self.get_requests_per_hour(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'top_errors': dict(list(self.error_counts.items())[:5])
        }


class Performance_monitor:
    """性能监控器"""

    def __init__(self, update_interval=60):
        self.metrics = Performance_metrics()
        self.update_interval = update_interval
        self.running = False
        self.monitor_thread = None

        # 告警阈值
        self.thresholds = {
            'error_rate': 10.0,  # 错误率超过10%
            'avg_response_time': 30.0,  # 平均响应时间超过30秒
            'cpu_usage': 80.0,  # CPU使用率超过80%
            'memory_usage': 85.0,  # 内存使用率超过85%
            'success_rate': 90.0  # 成功率低于90%
        }

        # 告警回调
        self.alert_callbacks = []

    def start_Monitor(self):
        """启动监控"""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start_Monitor()
        logger.info("性能监控器启动")

    def stop_Monitor(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("性能监控器停止")

    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 更新系统指标
                self.metrics.update_system_metrics()

                # 检查告警条件
                self._check_alerts()

                # 等待下次更新
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(self.update_interval)

    def _check_alerts(self):
        """检查告警条件"""
        alerts = []

        # 检查错误率
        error_rate = self.metrics.get_error_rate()
        if error_rate > self.thresholds['error_rate']:
            alerts.append({
                'type': 'error_rate',
                'message': f'错误率过高: {error_rate:.2f}%',
                'value': error_rate,
                'threshold': self.thresholds['error_rate']
            })

        # 检查响应时间
        if self.metrics.avg_response_time > self.thresholds['avg_response_time']:
            alerts.append({
                'type': 'response_time',
                'message': f'平均响应时间过长: {self.metrics.avg_response_time:.2f}秒',
                'value': self.metrics.avg_response_time,
                'threshold': self.thresholds['avg_response_time']
            })

        # 检查成功率
        success_rate = self.metrics.get_success_rate()
        if success_rate < self.thresholds['success_rate'] and self.metrics.total_requests > 10:
            alerts.append({
                'type': 'success_rate',
                'message': f'成功率过低: {success_rate:.2f}%',
                'value': success_rate,
                'threshold': self.thresholds['success_rate']
            })

        # 检查系统资源
        if self.metrics.cpu_usage > self.thresholds['cpu_usage']:
            alerts.append({
                'type': 'cpu_usage',
                'message': f'CPU使用率过高: {self.metrics.cpu_usage:.2f}%',
                'value': self.metrics.cpu_usage,
                'threshold': self.thresholds['cpu_usage']
            })

        if self.metrics.memory_usage > self.thresholds['memory_usage']:
            alerts.append({
                'type': 'memory_usage',
                'message': f'内存使用率过高: {self.metrics.memory_usage:.2f}%',
                'value': self.metrics.memory_usage,
                'threshold': self.thresholds['memory_usage']
            })

        # 触发告警
        for alert in alerts:
            self._trigger_alert(alert)

    def _trigger_alert(self, alert: Dict[str, Any]):
        """触发告警"""
        logger.warning(f"性能告警: {alert['message']}")

        # 调用告警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")

    def add_alert_callback(self, callback):
        """添加告警回调"""
        self.alert_callbacks.append(callback)

    def record_request_Monitor_Performance_Monitor_Performance_Monitor_performancemonitor(self, success: bool, response_time: float, error_type: str = None):
        """记录请求"""
        self.metrics.record_request_Monitor_Performance_Monitor_Performance_Monitor_performancemonitor(success, response_time, error_type)

    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        return self.metrics.get_summary()

    def set_threshold(self, metric_name: str, value: float):
        """设置告警阈值"""
        if metric_name in self.thresholds:
            self.thresholds[metric_name] = value
            logger.info(f"更新告警阈值: {metric_name} = {value}")
        else:
            logger.warning(f"未知的指标名称: {metric_name}")

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        metrics = self.get_metrics()

        # 计算健康分数
        health_score = 100.0
        issues = []

        if metrics['error_rate'] > self.thresholds['error_rate']:
            health_score -= 20
            issues.append(f"错误率过高: {metrics['error_rate']:.2f}%")

        if metrics['avg_response_time'] > self.thresholds['avg_response_time']:
            health_score -= 15
            issues.append(f"响应时间过长: {metrics['avg_response_time']:.2f}秒")

        if metrics['success_rate'] < self.thresholds['success_rate'] and metrics['total_requests'] > 10:
            health_score -= 25
            issues.append(f"成功率过低: {metrics['success_rate']:.2f}%")

        if metrics['cpu_usage'] > self.thresholds['cpu_usage']:
            health_score -= 10
            issues.append(f"CPU使用率过高: {metrics['cpu_usage']:.2f}%")

        if metrics['memory_usage'] > self.thresholds['memory_usage']:
            health_score -= 10
            issues.append(f"内存使用率过高: {metrics['memory_usage']:.2f}%")

        # 确定健康状态
        if health_score >= 90:
            status = "健康"
        elif health_score >= 70:
            status = "警告"
        elif health_score >= 50:
            status = "异常"
        else:
            status = "严重"

        return {
            'status': status,
            'health_score': max(health_score, 0),
            'issues': issues,
            'metrics': metrics
        }