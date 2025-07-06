"""
告警管理模块

处理系统告警，支持多种通知方式
"""

import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
from utils.logger import get_logger

logger = get_logger(__name__)


class Alert_rule:
    """告警规则"""

    def __init__(self, name: str, condition: Callable, message: str,
                 severity: str = "warning", cooldown: int = 300):
        self.name = name
        self.condition = condition
        self.message = message
        self.severity = severity  # info, warning, error, critical
        self.cooldown = cooldown  # 冷却时间（秒）
        self.last_triggered = None

    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        """检查是否应该触发告警"""
        # 检查冷却时间
        if self.last_triggered:
            if (datetime.now() - self.last_triggered).total_seconds() < self.cooldown:
                return False

        # 检查条件
        try:
            return self.condition(metrics)
        except Exception as e:
            logger.error(f"告警规则 {self.name} 条件检查失败: {e}")
            return False

    def trigger(self):
        """触发告警"""
        self.last_triggered = datetime.now()


class Notification_channel:
    """通知渠道基类"""

    def __init__(self, name: str):
        self.name = name

    def send_Manager_Alert_Manager_Alert_Manager_alertmanager(self, alert: Dict[str, Any]) -> bool:
        """发送通知"""
        raise Not_implemented_error


class Log_notification(Notification_channel):
    """日志通知"""

    def __init__(self):
        super().__init__("log")

    def send_Manager_Alert_Manager_Alert_Manager_alertmanager(self, alert: Dict[str, Any]) -> bool:
        """发送日志通知"""
        try:
            severity = alert['severity'].upper()
            message = f"[{severity}] {alert['rule_name']}: {alert['message']}"

            if severity == 'CRITICAL':
                logger.critical(message)
            elif severity == 'ERROR':
                logger.error(message)
            elif severity == 'WARNING':
                logger.warning(message)
            else:
                logger.info(message)

            return True
        except Exception as e:
            logger.error(f"日志告警发送失败: {e}")
            return False


class Webhook_notification(Notification_channel):
    """Webhook通知"""

    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        super().__init__("webhook")
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}

    def send_Manager_Alert_Manager_Alert_Manager_alertmanager(self, alert: Dict[str, Any]) -> bool:
        """发送Webhook通知"""
        try:
            payload = {
                'alert_type': 'crawler_system',
                'timestamp': alert['timestamp'],
                'severity': alert['severity'],
                'rule_name': alert['rule_name'],
                'message': alert['message'],
                'metrics': alert.get('metrics', {})
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"Webhook告警发送成功: {alert['rule_name']}")
                return True
            else:
                logger.error(f"Webhook告警发送失败: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Webhook告警发送失败: {e}")
            return False


class Alert_manager:
    """告警管理器"""

    def __init__(self):
        self.rules = {}
        self.channels = {}
        self.alert_history = []
        self.max_history = 1000

        # 默认添加日志通知渠道
        self.add_channel(Log_notification())

        # 添加默认告警规则
        self._add_default_rules_Alert_Manager()

    def _add_default_rules_Alert_Manager(self):
        """添加默认告警规则"""
        # 高错误率告警
        self.add_rule_Manager(Alert_rule(
            name="high_error_rate",
            condition=lambda m: m.get('error_rate', 0) > 15.0,
            message="错误率过高，可能存在系统问题",
            severity="error",
            cooldown=600  # 10分钟冷却
        ))

        # 低成功率告警
        self.add_rule_Manager(Alert_rule(
            name="low_success_rate",
            condition=lambda m: m.get('success_rate', 100) < 85.0 and m.get('total_requests', 0) > 20,
            message="成功率过低，需要检查系统状态",
            severity="warning",
            cooldown=300  # 5分钟冷却
        ))

        # 响应时间过长告警
        self.add_rule_Manager(Alert_rule(
            name="slow_response",
            condition=lambda m: m.get('avg_response_time', 0) > 30.0,
            message="平均响应时间过长，可能存在性能问题",
            severity="warning",
            cooldown=300
        ))

        # 系统资源告警
        self.add_rule_Manager(Alert_rule(
            name="high_cpu_usage",
            condition=lambda m: m.get('cpu_usage', 0) > 85.0,
            message="CPU使用率过高",
            severity="warning",
            cooldown=600
        ))

        self.add_rule_Manager(Alert_rule(
            name="high_memory_usage",
            condition=lambda m: m.get('memory_usage', 0) > 90.0,
            message="内存使用率过高",
            severity="error",
            cooldown=600
        ))

        # 无请求告警
        self.add_rule_Manager(Alert_rule(
            name="no_requests",
            condition=lambda m: m.get('requests_per_hour', 0) < 1.0 and m.get('runtime_hours', 0) > 1.0,
            message="系统长时间无请求，可能已停止工作",
            severity="critical",
            cooldown=1800  # 30分钟冷却
        ))

    def add_rule_Manager(self, rule: Alert_rule):
        """添加告警规则"""
        self.rules[rule.name] = rule
        logger.info(f"添加告警规则: {rule.name}")

    def remove_rule_Manager(self, rule_name: str):
        """移除告警规则"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"移除告警规则: {rule_name}")

    def add_channel(self, channel: Notification_channel):
        """添加通知渠道"""
        self.channels[channel.name] = channel
        logger.info(f"添加通知渠道: {channel.name}")

    def remove_channel(self, channel_name: str):
        """移除通知渠道"""
        if channel_name in self.channels:
            del self.channels[channel_name]
            logger.info(f"移除通知渠道: {channel_name}")

    def check_alerts(self, metrics: Dict[str, Any]):
        """检查告警"""
        triggered_alerts = []

        for rule_name, rule in self.rules.items():
            if rule.should_trigger(metrics):
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'rule_name': rule_name,
                    'message': rule.message,
                    'severity': rule.severity,
                    'metrics': metrics
                }

                # 触发告警
                rule.trigger()
                triggered_alerts.append(alert)

                # 发送通知
                self._send_notifications(alert)

                # 记录历史
                self._record_alert(alert)

        return triggered_alerts

    def _send_notifications(self, alert: Dict[str, Any]):
        """发送通知"""
        for channel_name, channel in self.channels.items():
            try:
                success = channel.send_Manager_Alert_Manager_Alert_Manager_alertmanager(alert)
                if success:
                    logger.debug(f"通知发送成功: {channel_name}")
                else:
                    logger.warning(f"通知发送失败: {channel_name}")
            except Exception as e:
                logger.error(f"通知渠道 {channel_name} 发送失败: {e}")

    def _record_alert(self, alert: Dict[str, Any]):
        """记录告警历史"""
        self.alert_history.append(alert)

        # 限制历史记录数量
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取告警历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_alerts = []
        for alert in self.alert_history:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if alert_time >= cutoff_time:
                recent_alerts.append(alert)

        return recent_alerts

    def get_alert_stats(self, hours: int = 24) -> Dict[str, Any]:
        """获取告警统计"""
        recent_alerts = self.get_alert_history(hours)

        stats = {
            'total_alerts': len(recent_alerts),
            'by_severity': defaultdict(int),
            'by_rule': defaultdict(int),
            'by_hour': defaultdict(int)
        }

        for alert in recent_alerts:
            stats['by_severity'][alert['severity']] += 1
            stats['by_rule'][alert['rule_name']] += 1

            # 按小时统计
            alert_time = datetime.fromisoformat(alert['timestamp'])
            hour_key = alert_time.strftime('%Y-%m-%d %H')
            stats['by_hour'][hour_key] += 1

        return dict(stats)

    def test_notifications(self):
        """测试通知渠道"""
        test_alert = {
            'timestamp': datetime.now().isoformat(),
            'rule_name': 'test_alert',
            'message': '这是一个测试告警',
            'severity': 'info',
            'metrics': {'test': True}
        }

        results = {}
        for channel_name, channel in self.channels.items():
            try:
                success = channel.send_Manager_Alert_Manager_Alert_Manager_alertmanager(test_alert)
                results[channel_name] = 'success' if success else 'failed'
            except Exception as e:
                results[channel_name] = f'error: {str(e)}'

        return results