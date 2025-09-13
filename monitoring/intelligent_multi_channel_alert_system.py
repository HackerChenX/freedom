#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
多渠道智能告警系统

提供企业级多渠道告警解决方案：
1. 多种告警渠道（邮件、短信、钉钉、微信、Slack、Webhook）
2. 智能告警规则引擎
3. 告警去重和聚合
4. 告警升级和降级
5. 告警静默和抑制
6. 告警恢复通知
7. 告警统计和分析
"""

import os
import json
import time
import yaml
import smtplib
import requests
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import queue
import hashlib
import hmac
import base64
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """告警状态"""
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "email"
    SMS = "sms"
    DINGTALK = "dingtalk"
    WECHAT = "wechat"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PHONE_CALL = "phone_call"


@dataclass
class Alert:
    """告警数据类"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    source: str
    timestamp: datetime
    labels: Dict[str, str]
    annotations: Dict[str, str]
    fingerprint: str
    resolved_at: Optional[datetime] = None
    silenced_until: Optional[datetime] = None
    suppressed_by: Optional[str] = None
    escalation_level: int = 0
    notification_count: int = 0
    last_notification: Optional[datetime] = None


@dataclass
class NotificationTemplate:
    """通知模板"""
    channel: NotificationChannel
    title_template: str
    body_template: str
    format_type: str = "text"  # text, html, markdown


@dataclass
class EscalationRule:
    """升级规则"""
    name: str
    condition: Dict[str, Any]
    escalation_levels: List[Dict[str, Any]]
    enabled: bool = True


@dataclass
class SilenceRule:
    """静默规则"""
    id: str
    matcher: Dict[str, str]
    start_time: datetime
    end_time: datetime
    comment: str
    created_by: str


class NotificationHandler(ABC):
    """通知处理器抽象基类"""

    @abstractmethod
    def send_notification(self, alert: Alert, template: NotificationTemplate,
                         recipients: List[str]) -> bool:
        """发送通知"""
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置"""
        pass


class EmailNotificationHandler(NotificationHandler):
    """邮件通知处理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化邮件通知处理器

        Args:
            config: 邮件配置
        """
        self.config = config
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.use_tls = config.get('use_tls', True)

    def send_notification(self, alert: Alert, template: NotificationTemplate,
                         recipients: List[str]) -> bool:
        """发送邮件通知"""
        try:
            # 渲染模板
            title = self._render_template(template.title_template, alert)
            body = self._render_template(template.body_template, alert)

            # 创建邮件
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['Subject'] = title

            if template.format_type == 'html':
                msg.attach(MimeText(body, 'html', 'utf-8'))
            else:
                msg.attach(MimeText(body, 'plain', 'utf-8'))

            # 连接SMTP服务器
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)

            # 发送邮件
            success_count = 0
            for recipient in recipients:
                try:
                    msg['To'] = recipient
                    server.send_message(msg)
                    del msg['To']
                    success_count += 1
                except Exception as e:
                    logger.error(f"发送邮件到 {recipient} 失败: {e}")

            server.quit()

            logger.info(f"邮件通知发送完成: 成功 {success_count}/{len(recipients)}")
            return success_count > 0

        except Exception as e:
            logger.error(f"发送邮件通知失败: {e}")
            return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证邮件配置"""
        required_keys = ['smtp_server', 'username', 'password']
        return all(key in config for key in required_keys)

    def _render_template(self, template: str, alert: Alert) -> str:
        """渲染模板"""
        try:
            # 简单的模板渲染
            context = {
                'alert_name': alert.name,
                'alert_description': alert.description,
                'alert_severity': alert.severity.value,
                'alert_status': alert.status.value,
                'alert_source': alert.source,
                'alert_timestamp': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'alert_labels': json.dumps(alert.labels, ensure_ascii=False),
                'alert_annotations': json.dumps(alert.annotations, ensure_ascii=False)
            }

            rendered = template
            for key, value in context.items():
                rendered = rendered.replace(f'{{{key}}}', str(value))

            return rendered

        except Exception as e:
            logger.error(f"模板渲染失败: {e}")
            return template


class DingTalkNotificationHandler(NotificationHandler):
    """钉钉通知处理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化钉钉通知处理器

        Args:
            config: 钉钉配置
        """
        self.config = config
        self.webhook_url = config.get('webhook_url')
        self.secret = config.get('secret')

    def send_notification(self, alert: Alert, template: NotificationTemplate,
                         recipients: List[str]) -> bool:
        """发送钉钉通知"""
        try:
            # 生成时间戳和签名
            timestamp = str(round(time.time() * 1000))
            sign = self._generate_sign(timestamp)

            # 构造请求URL
            url = f"{self.webhook_url}&timestamp={timestamp}&sign={sign}"

            # 渲染消息内容
            title = self._render_template(template.title_template, alert)
            body = self._render_template(template.body_template, alert)

            # 构造消息体
            if template.format_type == 'markdown':
                payload = {
                    "msgtype": "markdown",
                    "markdown": {
                        "title": title,
                        "text": body
                    }
                }
            else:
                payload = {
                    "msgtype": "text",
                    "text": {
                        "content": f"{title}\n\n{body}"
                    }
                }

            # 如果有@用户
            if recipients:
                payload["at"] = {
                    "atMobiles": recipients,
                    "isAtAll": False
                }

            # 发送请求
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            result = response.json()
            if result.get('errcode') == 0:
                logger.info("钉钉通知发送成功")
                return True
            else:
                logger.error(f"钉钉通知发送失败: {result}")
                return False

        except Exception as e:
            logger.error(f"发送钉钉通知失败: {e}")
            return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证钉钉配置"""
        return 'webhook_url' in config

    def _generate_sign(self, timestamp: str) -> str:
        """生成钉钉签名"""
        if not self.secret:
            return ""

        string_to_sign = f'{timestamp}\n{self.secret}'
        hmac_code = hmac.new(
            self.secret.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        sign = base64.b64encode(hmac_code).decode('utf-8')
        return sign

    def _render_template(self, template: str, alert: Alert) -> str:
        """渲染模板"""
        try:
            context = {
                'alert_name': alert.name,
                'alert_description': alert.description,
                'alert_severity': alert.severity.value,
                'alert_status': alert.status.value,
                'alert_source': alert.source,
                'alert_timestamp': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'severity_emoji': self._get_severity_emoji(alert.severity)
            }

            rendered = template
            for key, value in context.items():
                rendered = rendered.replace(f'{{{key}}}', str(value))

            return rendered

        except Exception as e:
            logger.error(f"模板渲染失败: {e}")
            return template

    def _get_severity_emoji(self, severity: AlertSeverity) -> str:
        """获取严重程度对应的emoji"""
        emoji_map = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }
        return emoji_map.get(severity, "")


class WebhookNotificationHandler(NotificationHandler):
    """Webhook通知处理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化Webhook通知处理器

        Args:
            config: Webhook配置
        """
        self.config = config
        self.webhook_url = config.get('webhook_url')
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 10)

    def send_notification(self, alert: Alert, template: NotificationTemplate,
                         recipients: List[str]) -> bool:
        """发送Webhook通知"""
        try:
            # 构造载荷
            payload = {
                "alert": {
                    "id": alert.id,
                    "name": alert.name,
                    "description": alert.description,
                    "severity": alert.severity.value,
                    "status": alert.status.value,
                    "source": alert.source,
                    "timestamp": alert.timestamp.isoformat(),
                    "labels": alert.labels,
                    "annotations": alert.annotations,
                    "fingerprint": alert.fingerprint
                },
                "recipients": recipients,
                "template": {
                    "title": self._render_template(template.title_template, alert),
                    "body": self._render_template(template.body_template, alert)
                }
            }

            # 发送请求
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            logger.info("Webhook通知发送成功")
            return True

        except Exception as e:
            logger.error(f"发送Webhook通知失败: {e}")
            return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证Webhook配置"""
        return 'webhook_url' in config

    def _render_template(self, template: str, alert: Alert) -> str:
        """渲染模板"""
        try:
            context = {
                'alert_name': alert.name,
                'alert_description': alert.description,
                'alert_severity': alert.severity.value,
                'alert_status': alert.status.value,
                'alert_source': alert.source,
                'alert_timestamp': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }

            rendered = template
            for key, value in context.items():
                rendered = rendered.replace(f'{{{key}}}', str(value))

            return rendered

        except Exception as e:
            logger.error(f"模板渲染失败: {e}")
            return template


class SlackNotificationHandler(NotificationHandler):
    """Slack通知处理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化Slack通知处理器

        Args:
            config: Slack配置
        """
        self.config = config
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel')
        self.username = config.get('username', 'AlertBot')

    def send_notification(self, alert: Alert, template: NotificationTemplate,
                         recipients: List[str]) -> bool:
        """发送Slack通知"""
        try:
            # 渲染消息内容
            title = self._render_template(template.title_template, alert)
            body = self._render_template(template.body_template, alert)

            # 构造消息体
            payload = {
                "channel": self.channel,
                "username": self.username,
                "text": title,
                "attachments": [
                    {
                        "color": self._get_color_by_severity(alert.severity),
                        "fields": [
                            {
                                "title": "描述",
                                "value": body,
                                "short": False
                            },
                            {
                                "title": "严重程度",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "来源",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "时间",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ],
                        "footer": "股票分析系统告警",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }

            # 发送请求
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("Slack通知发送成功")
            return True

        except Exception as e:
            logger.error(f"发送Slack通知失败: {e}")
            return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证Slack配置"""
        return 'webhook_url' in config

    def _render_template(self, template: str, alert: Alert) -> str:
        """渲染模板"""
        try:
            context = {
                'alert_name': alert.name,
                'alert_description': alert.description,
                'alert_severity': alert.severity.value,
                'alert_status': alert.status.value,
                'alert_source': alert.source,
                'alert_timestamp': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }

            rendered = template
            for key, value in context.items():
                rendered = rendered.replace(f'{{{key}}}', str(value))

            return rendered

        except Exception as e:
            logger.error(f"模板渲染失败: {e}")
            return template

    def _get_color_by_severity(self, severity: AlertSeverity) -> str:
        """根据严重程度获取颜色"""
        color_map = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "danger",
            AlertSeverity.CRITICAL: "danger"
        }
        return color_map.get(severity, "good")


class AlertDeduplicator:
    """告警去重器"""

    def __init__(self, dedup_window: int = 300):  # 5分钟去重窗口
        """
        初始化告警去重器

        Args:
            dedup_window: 去重时间窗口（秒）
        """
        self.dedup_window = dedup_window
        self.recent_alerts: Dict[str, Alert] = {}
        self.lock = threading.RLock()

    def should_send_alert(self, alert: Alert) -> bool:
        """判断是否应该发送告警"""
        with self.lock:
            fingerprint = alert.fingerprint

            # 检查是否有相同指纹的告警
            if fingerprint in self.recent_alerts:
                recent_alert = self.recent_alerts[fingerprint]

                # 检查时间窗口
                time_diff = (alert.timestamp - recent_alert.timestamp).total_seconds()
                if time_diff < self.dedup_window:
                    # 在去重窗口内，不发送
                    return False

            # 更新最近告警
            self.recent_alerts[fingerprint] = alert

            # 清理过期的告警记录
            self._cleanup_expired_alerts(alert.timestamp)

            return True

    def _cleanup_expired_alerts(self, current_time: datetime):
        """清理过期的告警记录"""
        expired_fingerprints = []

        for fingerprint, alert in self.recent_alerts.items():
            time_diff = (current_time - alert.timestamp).total_seconds()
            if time_diff > self.dedup_window:
                expired_fingerprints.append(fingerprint)

        for fingerprint in expired_fingerprints:
            del self.recent_alerts[fingerprint]


class AlertEscalator:
    """告警升级器"""

    def __init__(self):
        self.escalation_rules: List[EscalationRule] = []
        self.escalation_timers: Dict[str, threading.Timer] = {}
        self.lock = threading.RLock()

    def add_escalation_rule(self, rule: EscalationRule):
        """添加升级规则"""
        self.escalation_rules.append(rule)

    def process_alert(self, alert: Alert, alert_manager) -> bool:
        """处理告警升级"""
        with self.lock:
            # 查找匹配的升级规则
            for rule in self.escalation_rules:
                if self._matches_rule(alert, rule):
                    self._schedule_escalation(alert, rule, alert_manager)
                    return True

            return False

    def _matches_rule(self, alert: Alert, rule: EscalationRule) -> bool:
        """检查告警是否匹配升级规则"""
        if not rule.enabled:
            return False

        condition = rule.condition

        # 检查严重程度
        if 'severity' in condition:
            if alert.severity.value not in condition['severity']:
                return False

        # 检查标签
        if 'labels' in condition:
            for key, values in condition['labels'].items():
                if key not in alert.labels:
                    return False
                if alert.labels[key] not in values:
                    return False

        return True

    def _schedule_escalation(self, alert: Alert, rule: EscalationRule, alert_manager):
        """安排告警升级"""
        for level_config in rule.escalation_levels:
            delay = level_config.get('delay', 300)  # 默认5分钟
            escalation_timer = threading.Timer(
                delay,
                self._escalate_alert,
                args=(alert, level_config, alert_manager)
            )
            escalation_timer.start()

            timer_key = f"{alert.fingerprint}_{level_config.get('level', 0)}"
            self.escalation_timers[timer_key] = escalation_timer

    def _escalate_alert(self, alert: Alert, level_config: Dict[str, Any], alert_manager):
        """执行告警升级"""
        try:
            level = level_config.get('level', 1)
            channels = level_config.get('channels', [])
            recipients = level_config.get('recipients', [])

            # 更新告警升级级别
            alert.escalation_level = level

            # 发送升级通知
            for channel in channels:
                alert_manager.send_notification(alert, NotificationChannel(channel), recipients)

            logger.info(f"告警升级成功: {alert.id} -> Level {level}")

        except Exception as e:
            logger.error(f"告警升级失败: {e}")

    def cancel_escalation(self, alert_fingerprint: str):
        """取消告警升级"""
        with self.lock:
            cancelled_timers = []
            for timer_key in list(self.escalation_timers.keys()):
                if timer_key.startswith(alert_fingerprint):
                    timer = self.escalation_timers[timer_key]
                    timer.cancel()
                    cancelled_timers.append(timer_key)

            for timer_key in cancelled_timers:
                del self.escalation_timers[timer_key]


class IntelligentAlertManager:
    """
    智能告警管理器

    核心告警管理功能，整合所有告警处理组件
    """

    def __init__(self, config_path: str = "config/alerts/intelligent_alert_manager.yaml"):
        """
        初始化智能告警管理器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # 初始化组件
        self.notification_handlers = self._init_notification_handlers()
        self.notification_templates = self._load_notification_templates()
        self.deduplicator = AlertDeduplicator(
            dedup_window=self.config.get('deduplication_window', 300)
        )
        self.escalator = AlertEscalator()

        # 告警存储
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: List[Alert] = []
        self.silence_rules: List[SilenceRule] = []

        # 通知队列
        self.notification_queue = queue.Queue()
        self.processing_active = True

        # 启动通知处理线程
        self.notification_thread = threading.Thread(target=self._process_notifications, daemon=True)
        self.notification_thread.start()

        # 线程锁
        self.lock = threading.RLock()

        # 加载升级规则
        self._load_escalation_rules()

        logger.info("智能告警管理器初始化完成")

    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            else:
                # 创建默认配置
                default_config = {
                    'deduplication_window': 300,
                    'notification_channels': {
                        'email': {
                            'enabled': False,
                            'smtp_server': '',
                            'username': '',
                            'password': '',
                            'recipients': []
                        },
                        'dingtalk': {
                            'enabled': False,
                            'webhook_url': '',
                            'secret': ''
                        },
                        'slack': {
                            'enabled': False,
                            'webhook_url': '',
                            'channel': '#alerts'
                        },
                        'webhook': {
                            'enabled': False,
                            'webhook_url': '',
                            'headers': {}
                        }
                    },
                    'escalation_rules': [],
                    'silence_rules': []
                }

                # 确保目录存在
                self.config_path.parent.mkdir(parents=True, exist_ok=True)

                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

                return default_config

        except Exception as e:
            logger.error(f"加载告警配置失败: {e}")
            return {}

    def _init_notification_handlers(self) -> Dict[NotificationChannel, NotificationHandler]:
        """初始化通知处理器"""
        handlers = {}
        channels_config = self.config.get('notification_channels', {})

        # 邮件通知
        if channels_config.get('email', {}).get('enabled'):
            handlers[NotificationChannel.EMAIL] = EmailNotificationHandler(
                channels_config['email']
            )

        # 钉钉通知
        if channels_config.get('dingtalk', {}).get('enabled'):
            handlers[NotificationChannel.DINGTALK] = DingTalkNotificationHandler(
                channels_config['dingtalk']
            )

        # Slack通知
        if channels_config.get('slack', {}).get('enabled'):
            handlers[NotificationChannel.SLACK] = SlackNotificationHandler(
                channels_config['slack']
            )

        # Webhook通知
        if channels_config.get('webhook', {}).get('enabled'):
            handlers[NotificationChannel.WEBHOOK] = WebhookNotificationHandler(
                channels_config['webhook']
            )

        return handlers

    def _load_notification_templates(self) -> Dict[str, NotificationTemplate]:
        """加载通知模板"""
        templates = {}

        # 默认邮件模板
        templates['email_default'] = NotificationTemplate(
            channel=NotificationChannel.EMAIL,
            title_template="[{alert_severity}] {alert_name}",
            body_template="""
告警详情：
- 告警名称：{alert_name}
- 描述：{alert_description}
- 严重程度：{alert_severity}
- 状态：{alert_status}
- 来源：{alert_source}
- 时间：{alert_timestamp}
- 标签：{alert_labels}
- 注释：{alert_annotations}
""",
            format_type="text"
        )

        # 默认钉钉模板
        templates['dingtalk_default'] = NotificationTemplate(
            channel=NotificationChannel.DINGTALK,
            title_template="{severity_emoji} {alert_name}",
            body_template="""
## {severity_emoji} {alert_name}

**描述：** {alert_description}

**严重程度：** {alert_severity}

**来源：** {alert_source}

**时间：** {alert_timestamp}
""",
            format_type="markdown"
        )

        # 默认Slack模板
        templates['slack_default'] = NotificationTemplate(
            channel=NotificationChannel.SLACK,
            title_template="{alert_name}",
            body_template="{alert_description}",
            format_type="text"
        )

        # 默认Webhook模板
        templates['webhook_default'] = NotificationTemplate(
            channel=NotificationChannel.WEBHOOK,
            title_template="{alert_name}",
            body_template="{alert_description}",
            format_type="json"
        )

        return templates

    def _load_escalation_rules(self):
        """加载升级规则"""
        escalation_configs = self.config.get('escalation_rules', [])

        for rule_config in escalation_configs:
            rule = EscalationRule(**rule_config)
            self.escalator.add_escalation_rule(rule)

        logger.info(f"加载 {len(escalation_configs)} 个升级规则")

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=2.0)
    def send_alert(self, name: str, description: str, severity: AlertSeverity,
                  source: str, labels: Dict[str, str] = None,
                  annotations: Dict[str, str] = None) -> str:
        """
        发送告警

        Args:
            name: 告警名称
            description: 告警描述
            severity: 严重程度
            source: 告警来源
            labels: 标签
            annotations: 注释

        Returns:
            str: 告警ID
        """
        # 创建告警
        alert_id = self._generate_alert_id(name, source, labels)
        fingerprint = self._generate_fingerprint(name, labels)

        alert = Alert(
            id=alert_id,
            name=name,
            description=description,
            severity=severity,
            status=AlertStatus.FIRING,
            source=source,
            timestamp=datetime.now(),
            labels=labels or {},
            annotations=annotations or {},
            fingerprint=fingerprint
        )

        # 检查静默规则
        if self._is_silenced(alert):
            logger.info(f"告警被静默: {alert.id}")
            return alert_id

        # 检查去重
        if not self.deduplicator.should_send_alert(alert):
            logger.info(f"告警已去重: {alert.id}")
            return alert_id

        # 存储告警
        with self.lock:
            self.active_alerts[alert.fingerprint] = alert

        # 添加到通知队列
        self.notification_queue.put(('send', alert))

        # 处理升级规则
        self.escalator.process_alert(alert, self)

        logger.info(f"创建告警: {alert.id} ({severity.value})")
        return alert_id

    @exception_handler(reraise=True)
    def resolve_alert(self, name: str, source: str, labels: Dict[str, str] = None) -> bool:
        """
        解决告警

        Args:
            name: 告警名称
            source: 告警来源
            labels: 标签

        Returns:
            bool: 是否成功解决
        """
        fingerprint = self._generate_fingerprint(name, labels)

        with self.lock:
            if fingerprint in self.active_alerts:
                alert = self.active_alerts[fingerprint]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()

                # 移到已解决列表
                self.resolved_alerts.append(alert)
                del self.active_alerts[fingerprint]

                # 取消升级
                self.escalator.cancel_escalation(fingerprint)

                # 发送解决通知
                self.notification_queue.put(('resolve', alert))

                logger.info(f"告警已解决: {alert.id}")
                return True

        return False

    @exception_handler(reraise=True)
    def add_silence_rule(self, matcher: Dict[str, str], duration_hours: int,
                        comment: str, created_by: str) -> str:
        """
        添加静默规则

        Args:
            matcher: 匹配器
            duration_hours: 静默时长（小时）
            comment: 注释
            created_by: 创建者

        Returns:
            str: 静默规则ID
        """
        silence_id = f"silence_{int(time.time())}_{len(self.silence_rules)}"
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)

        silence_rule = SilenceRule(
            id=silence_id,
            matcher=matcher,
            start_time=start_time,
            end_time=end_time,
            comment=comment,
            created_by=created_by
        )

        self.silence_rules.append(silence_rule)
        logger.info(f"添加静默规则: {silence_id} ({duration_hours}小时)")

        return silence_id

    def _is_silenced(self, alert: Alert) -> bool:
        """检查告警是否被静默"""
        current_time = datetime.now()

        for silence_rule in self.silence_rules:
            # 检查时间范围
            if not (silence_rule.start_time <= current_time <= silence_rule.end_time):
                continue

            # 检查匹配器
            if self._matches_silence_rule(alert, silence_rule):
                return True

        return False

    def _matches_silence_rule(self, alert: Alert, silence_rule: SilenceRule) -> bool:
        """检查告警是否匹配静默规则"""
        for key, value in silence_rule.matcher.items():
            if key == 'name' and alert.name != value:
                return False
            elif key == 'source' and alert.source != value:
                return False
            elif key == 'severity' and alert.severity.value != value:
                return False
            elif key in alert.labels and alert.labels[key] != value:
                return False

        return True

    def _process_notifications(self):
        """处理通知队列"""
        while self.processing_active:
            try:
                notification_type, alert = self.notification_queue.get(timeout=1)

                if notification_type == 'send':
                    self._send_alert_notifications(alert)
                elif notification_type == 'resolve':
                    self._send_resolve_notifications(alert)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理通知失败: {e}")

    def _send_alert_notifications(self, alert: Alert):
        """发送告警通知"""
        # 根据严重程度选择通知渠道
        channels = self._get_channels_by_severity(alert.severity)

        for channel in channels:
            try:
                self.send_notification(alert, channel)
            except Exception as e:
                logger.error(f"发送{channel.value}通知失败: {e}")

    def _send_resolve_notifications(self, alert: Alert):
        """发送解决通知"""
        # 只发送到原始通知渠道
        channels = self._get_channels_by_severity(alert.severity)

        for channel in channels:
            try:
                self.send_notification(alert, channel, is_resolved=True)
            except Exception as e:
                logger.error(f"发送{channel.value}解决通知失败: {e}")

    def _get_channels_by_severity(self, severity: AlertSeverity) -> List[NotificationChannel]:
        """根据严重程度获取通知渠道"""
        channel_mapping = {
            AlertSeverity.INFO: [NotificationChannel.EMAIL],
            AlertSeverity.WARNING: [NotificationChannel.EMAIL, NotificationChannel.DINGTALK],
            AlertSeverity.ERROR: [NotificationChannel.EMAIL, NotificationChannel.DINGTALK, NotificationChannel.SLACK],
            AlertSeverity.CRITICAL: [NotificationChannel.EMAIL, NotificationChannel.DINGTALK, NotificationChannel.SLACK, NotificationChannel.WEBHOOK]
        }

        channels = channel_mapping.get(severity, [NotificationChannel.EMAIL])
        # 只返回已配置的渠道
        return [channel for channel in channels if channel in self.notification_handlers]

    def send_notification(self, alert: Alert, channel: NotificationChannel,
                         recipients: List[str] = None, is_resolved: bool = False):
        """发送单个通知"""
        if channel not in self.notification_handlers:
            logger.warning(f"通知渠道 {channel.value} 未配置")
            return False

        # 选择模板
        template_key = f"{channel.value}_default"
        if template_key not in self.notification_templates:
            logger.error(f"找不到模板: {template_key}")
            return False

        template = self.notification_templates[template_key]

        # 如果是解决通知，修改模板
        if is_resolved:
            template.title_template = template.title_template.replace("{alert_name}", "{alert_name} [已解决]")
            template.body_template = template.body_template + "\n\n✅ 告警已解决"

        # 获取收件人
        if not recipients:
            recipients = self._get_default_recipients(channel)

        # 发送通知
        handler = self.notification_handlers[channel]
        success = handler.send_notification(alert, template, recipients)

        # 更新告警统计
        if success:
            alert.notification_count += 1
            alert.last_notification = datetime.now()

        return success

    def _get_default_recipients(self, channel: NotificationChannel) -> List[str]:
        """获取默认收件人"""
        channels_config = self.config.get('notification_channels', {})
        channel_config = channels_config.get(channel.value, {})

        if channel == NotificationChannel.EMAIL:
            return channel_config.get('recipients', [])
        elif channel == NotificationChannel.DINGTALK:
            return channel_config.get('at_mobiles', [])
        else:
            return []

    def _generate_alert_id(self, name: str, source: str, labels: Dict[str, str] = None) -> str:
        """生成告警ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fingerprint = self._generate_fingerprint(name, labels)[:8]
        return f"{source}_{name}_{fingerprint}_{timestamp}"

    def _generate_fingerprint(self, name: str, labels: Dict[str, str] = None) -> str:
        """生成告警指纹"""
        import hashlib

        # 构造指纹字符串
        fingerprint_data = name

        if labels:
            sorted_labels = sorted(labels.items())
            label_str = ",".join(f"{k}={v}" for k, v in sorted_labels)
            fingerprint_data += f"_{label_str}"

        # 生成MD5哈希
        return hashlib.md5(fingerprint_data.encode('utf-8')).hexdigest()

    @exception_handler(reraise=True)
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        with self.lock:
            return [
                {
                    'id': alert.id,
                    'name': alert.name,
                    'description': alert.description,
                    'severity': alert.severity.value,
                    'status': alert.status.value,
                    'source': alert.source,
                    'timestamp': alert.timestamp.isoformat(),
                    'labels': alert.labels,
                    'annotations': alert.annotations,
                    'escalation_level': alert.escalation_level,
                    'notification_count': alert.notification_count,
                    'last_notification': alert.last_notification.isoformat() if alert.last_notification else None
                }
                for alert in self.active_alerts.values()
            ]

    @exception_handler(reraise=True)
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        with self.lock:
            active_alerts = list(self.active_alerts.values())
            resolved_alerts = self.resolved_alerts[-100:]  # 最近100个已解决告警

            # 按严重程度统计
            severity_stats = {}
            for severity in AlertSeverity:
                severity_stats[severity.value] = len([
                    alert for alert in active_alerts if alert.severity == severity
                ])

            # 按来源统计
            source_stats = {}
            for alert in active_alerts:
                source_stats[alert.source] = source_stats.get(alert.source, 0) + 1

            # 按状态统计
            status_stats = {}
            for status in AlertStatus:
                status_stats[status.value] = len([
                    alert for alert in active_alerts if alert.status == status
                ])

            return {
                'total_active_alerts': len(active_alerts),
                'total_resolved_alerts': len(resolved_alerts),
                'severity_distribution': severity_stats,
                'source_distribution': source_stats,
                'status_distribution': status_stats,
                'silence_rules_count': len(self.silence_rules),
                'notification_channels': list(self.notification_handlers.keys()),
                'timestamp': datetime.now().isoformat()
            }

    def close(self):
        """关闭告警管理器"""
        self.processing_active = False

        if self.notification_thread.is_alive():
            self.notification_thread.join(timeout=5)

        # 取消所有升级定时器
        for timer in self.escalator.escalation_timers.values():
            timer.cancel()

        logger.info("智能告警管理器已关闭")


# 全局告警管理器实例
_intelligent_alert_manager = None


def get_intelligent_alert_manager(config_path: str = "config/alerts/intelligent_alert_manager.yaml") -> IntelligentAlertManager:
    """
    获取智能告警管理器实例（单例模式）

    Args:
        config_path: 配置文件路径

    Returns:
        IntelligentAlertManager: 智能告警管理器实例
    """
    global _intelligent_alert_manager

    if _intelligent_alert_manager is None:
        _intelligent_alert_manager = IntelligentAlertManager(config_path)

    return _intelligent_alert_manager


def create_intelligent_alert_manager(config_path: str = "config/alerts/intelligent_alert_manager.yaml") -> IntelligentAlertManager:
    """
    创建新的智能告警管理器实例

    Args:
        config_path: 配置文件路径

    Returns:
        IntelligentAlertManager: 新的智能告警管理器实例
    """
    return IntelligentAlertManager(config_path)