#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能风险预警通知系统

专业量化交易系统的智能预警模块，提供多层级、多渠道的风险预警通知。
基于机器学习算法和专家规则，实现精准的风险识别和及时的预警推送。

核心特性：
1. 多层级预警 - 信息、警告、严重、紧急四级预警体系
2. 智能预警规则 - 基于机器学习的动态阈值调整
3. 多渠道通知 - 邮件、短信、微信、钉钉等多种通知方式
4. 预警聚合 - 避免重复预警，智能合并相关预警
5. 预警抑制 - 防止预警风暴，设置预警频率限制
6. 历史追踪 - 预警历史记录和效果分析

技术架构：
- 实时风险监控
- 智能预警引擎
- 多渠道通知服务
- 预警管理后台
"""

import time
import json
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import hashlib
import uuid

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from utils.unified_container import get_container
from utils.numerical_stability_manager import get_stability_manager
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class AlertLevel(Enum):
    """预警级别"""
    INFO = "信息"
    WARNING = "警告"
    CRITICAL = "严重"
    EMERGENCY = "紧急"


class AlertType(Enum):
    """预警类型"""
    RISK_THRESHOLD = "风险阈值"
    PERFORMANCE_ANOMALY = "绩效异常"
    POSITION_RISK = "仓位风险"
    MARKET_RISK = "市场风险"
    LIQUIDITY_RISK = "流动性风险"
    SYSTEM_ERROR = "系统错误"
    COMPLIANCE_VIOLATION = "合规违规"
    DATA_QUALITY = "数据质量"


class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "邮件"
    SMS = "短信"
    WECHAT = "微信"
    DINGTALK = "钉钉"
    SLACK = "Slack"
    WEBHOOK = "Webhook"
    IN_APP = "应用内"
    SYSTEM_LOG = "系统日志"


class AlertStatus(Enum):
    """预警状态"""
    ACTIVE = "激活"
    ACKNOWLEDGED = "已确认"
    RESOLVED = "已解决"
    SUPPRESSED = "已抑制"
    EXPIRED = "已过期"


@dataclass
class AlertRule:
    """预警规则定义"""
    rule_id: str
    rule_name: str
    alert_type: AlertType
    alert_level: AlertLevel
    enabled: bool = True

    # 触发条件
    condition_expression: str = ""             # 条件表达式
    threshold_value: float = 0.0              # 阈值
    duration_seconds: int = 0                 # 持续时间（秒）

    # 通知设置
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    notification_template: str = ""           # 通知模板

    # 抑制设置
    suppression_duration: int = 3600          # 抑制时长（秒）
    max_alerts_per_hour: int = 10            # 每小时最大预警数

    # 收件人设置
    recipients: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'alert_type': self.alert_type.value,
            'alert_level': self.alert_level.value,
            'enabled': self.enabled,
            'condition_expression': self.condition_expression,
            'threshold_value': self.threshold_value,
            'duration_seconds': self.duration_seconds,
            'notification_channels': [ch.value for ch in self.notification_channels],
            'suppression_duration': self.suppression_duration,
            'max_alerts_per_hour': self.max_alerts_per_hour,
            'recipients': self.recipients
        }


@dataclass
class AlertEvent:
    """预警事件"""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    title: str
    message: str

    # 事件数据
    trigger_value: float = 0.0
    threshold_value: float = 0.0
    related_data: Dict[str, Any] = field(default_factory=dict)

    # 时间信息
    trigger_time: datetime = field(default_factory=datetime.now)
    first_trigger_time: Optional[datetime] = None
    last_update_time: datetime = field(default_factory=datetime.now)

    # 状态信息
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_time: Optional[datetime] = None
    resolved_time: Optional[datetime] = None

    # 通知信息
    notification_sent: List[NotificationChannel] = field(default_factory=list)
    notification_count: int = 0

    def get_fingerprint(self) -> str:
        """获取预警指纹（用于去重）"""
        fingerprint_data = f"{self.rule_id}_{self.alert_type.value}_{self.related_data.get('stock_code', '')}"
        return hashlib.md5(fingerprint_data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'rule_id': self.rule_id,
            'alert_type': self.alert_type.value,
            'alert_level': self.alert_level.value,
            'title': self.title,
            'message': self.message,
            'trigger_value': self.trigger_value,
            'threshold_value': self.threshold_value,
            'related_data': self.related_data,
            'trigger_time': self.trigger_time.isoformat(),
            'first_trigger_time': self.first_trigger_time.isoformat() if self.first_trigger_time else None,
            'last_update_time': self.last_update_time.isoformat(),
            'status': self.status.value,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_time': self.acknowledged_time.isoformat() if self.acknowledged_time else None,
            'resolved_time': self.resolved_time.isoformat() if self.resolved_time else None,
            'notification_sent': [ch.value for ch in self.notification_sent],
            'notification_count': self.notification_count
        }


@dataclass
class NotificationConfig:
    """通知配置"""
    # 邮件配置
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""

    # 短信配置
    sms_api_key: str = ""
    sms_api_secret: str = ""

    # 微信配置
    wechat_corpid: str = ""
    wechat_corpsecret: str = ""

    # 钉钉配置
    dingtalk_webhook: str = ""
    dingtalk_secret: str = ""

    # Webhook配置
    webhook_urls: List[str] = field(default_factory=list)


class IntelligentRiskAlertSystem:
    """
    智能风险预警通知系统

    提供全方位的风险预警和通知服务
    """

    def __init__(self,
                 notification_config: Optional[NotificationConfig] = None,
                 data_manager=None):
        """
        初始化智能预警系统

        Args:
            notification_config: 通知配置
            data_manager: 数据管理器
        """
        self.notification_config = notification_config or NotificationConfig()
        self.data_manager = data_manager or get_container().resolve("data_manager")
        self.stability_manager = get_stability_manager()

        # 预警规则管理
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: deque = deque(maxlen=10000)

        # 抑制管理
        self.suppression_tracker: Dict[str, List[datetime]] = defaultdict(list)
        self.suppressed_alerts: Set[str] = set()

        # 通知管理
        self.notification_queue = asyncio.Queue()
        self.notification_tasks: List[asyncio.Task] = []

        # 统计信息
        self.alert_statistics = {
            'total_alerts': 0,
            'alerts_by_level': defaultdict(int),
            'alerts_by_type': defaultdict(int),
            'notifications_sent': defaultdict(int),
            'suppressed_count': 0,
            'acknowledged_count': 0,
            'resolved_count': 0
        }

        # 系统状态
        self.is_running = False
        self._stop_event = threading.Event()
        self._monitoring_thread = None

        # 加载默认规则
        self._load_default_rules()

        logger.info("智能风险预警通知系统初始化完成")

    def _load_default_rules(self):
        """加载默认预警规则"""
        default_rules = [
            # 风险阈值预警
            AlertRule(
                rule_id="RISK_001",
                rule_name="组合风险评分过高",
                alert_type=AlertType.RISK_THRESHOLD,
                alert_level=AlertLevel.CRITICAL,
                condition_expression="risk_score > threshold_value",
                threshold_value=85.0,
                duration_seconds=300,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.IN_APP],
                notification_template="组合风险评分达到 {trigger_value:.1f}，超过阈值 {threshold_value:.1f}，请及时关注。",
                suppression_duration=1800,
                max_alerts_per_hour=5,
                recipients=["risk_manager@company.com"]
            ),

            # 回撤预警
            AlertRule(
                rule_id="RISK_002",
                rule_name="最大回撤预警",
                alert_type=AlertType.PERFORMANCE_ANOMALY,
                alert_level=AlertLevel.WARNING,
                condition_expression="max_drawdown > threshold_value",
                threshold_value=0.05,
                duration_seconds=0,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.DINGTALK],
                notification_template="组合最大回撤达到 {trigger_value:.2%}，超过预警阈值 {threshold_value:.2%}。",
                suppression_duration=3600,
                max_alerts_per_hour=3,
                recipients=["portfolio_manager@company.com"]
            ),

            # 仓位风险预警
            AlertRule(
                rule_id="POS_001",
                rule_name="单仓仓位过大",
                alert_type=AlertType.POSITION_RISK,
                alert_level=AlertLevel.WARNING,
                condition_expression="position_ratio > threshold_value",
                threshold_value=0.08,
                duration_seconds=0,
                notification_channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
                notification_template="股票 {stock_code} 仓位比例 {trigger_value:.2%} 超过限额 {threshold_value:.2%}。",
                suppression_duration=1800,
                max_alerts_per_hour=8,
                recipients=["trader@company.com"]
            ),

            # VaR预警
            AlertRule(
                rule_id="RISK_003",
                rule_name="VaR风险过高",
                alert_type=AlertType.MARKET_RISK,
                alert_level=AlertLevel.CRITICAL,
                condition_expression="var_1d > threshold_value",
                threshold_value=50000.0,
                duration_seconds=600,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SMS],
                notification_template="组合日VaR达到 {trigger_value:,.0f} 元，超过风险限额 {threshold_value:,.0f} 元。",
                suppression_duration=1200,
                max_alerts_per_hour=6,
                recipients=["risk_manager@company.com", "cro@company.com"]
            ),

            # 系统错误预警
            AlertRule(
                rule_id="SYS_001",
                rule_name="系统响应时间异常",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.EMERGENCY,
                condition_expression="response_time > threshold_value",
                threshold_value=10.0,
                duration_seconds=120,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.DINGTALK],
                notification_template="系统响应时间 {trigger_value:.1f}ms 超过阈值 {threshold_value:.1f}ms，可能影响交易执行。",
                suppression_duration=600,
                max_alerts_per_hour=12,
                recipients=["sysadmin@company.com", "dev_team@company.com"]
            )
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

        logger.info(f"已加载 {len(default_rules)} 条默认预警规则")

    async def trigger_alert(self,
                           rule_id: str,
                           trigger_value: float,
                           related_data: Optional[Dict[str, Any]] = None) -> Optional[AlertEvent]:
        """
        触发预警

        Args:
            rule_id: 规则ID
            trigger_value: 触发值
            related_data: 相关数据

        Returns:
            AlertEvent: 预警事件（如果成功创建）
        """
        try:
            if rule_id not in self.alert_rules:
                logger.warning(f"未找到预警规则: {rule_id}")
                return None

            rule = self.alert_rules[rule_id]

            # 检查规则是否启用
            if not rule.enabled:
                return None

            # 创建预警事件
            alert_event = AlertEvent(
                alert_id=str(uuid.uuid4()),
                rule_id=rule_id,
                alert_type=rule.alert_type,
                alert_level=rule.alert_level,
                title=rule.rule_name,
                message=rule.notification_template.format(
                    trigger_value=trigger_value,
                    threshold_value=rule.threshold_value,
                    **related_data or {}
                ),
                trigger_value=trigger_value,
                threshold_value=rule.threshold_value,
                related_data=related_data or {}
            )

            # 检查是否需要抑制
            if self._should_suppress_alert(alert_event):
                logger.debug(f"预警被抑制: {rule_id}")
                self.alert_statistics['suppressed_count'] += 1
                return None

            # 检查是否为重复预警
            fingerprint = alert_event.get_fingerprint()
            existing_alert = self._find_existing_alert(fingerprint)

            if existing_alert:
                # 更新现有预警
                existing_alert.last_update_time = datetime.now()
                existing_alert.trigger_value = trigger_value
                existing_alert.notification_count += 1
                alert_event = existing_alert
            else:
                # 新增预警
                alert_event.first_trigger_time = alert_event.trigger_time
                self.active_alerts[alert_event.alert_id] = alert_event

            # 更新统计信息
            self.alert_statistics['total_alerts'] += 1
            self.alert_statistics['alerts_by_level'][rule.alert_level.value] += 1
            self.alert_statistics['alerts_by_type'][rule.alert_type.value] += 1

            # 发送通知
            await self._send_notifications(alert_event, rule)

            # 记录历史
            self.alert_history.append(alert_event)

            logger.info(f"预警触发: {rule.rule_name} - {alert_event.message}")

            return alert_event

        except Exception as e:
            logger.error(f"触发预警异常: {e}")
            return None

    def _should_suppress_alert(self, alert_event: AlertEvent) -> bool:
        """检查是否应该抑制预警"""
        rule = self.alert_rules.get(alert_event.rule_id)
        if not rule:
            return False

        fingerprint = alert_event.get_fingerprint()

        # 检查抑制列表
        if fingerprint in self.suppressed_alerts:
            return True

        # 检查频率限制
        current_time = datetime.now()
        hour_ago = current_time - timedelta(hours=1)

        recent_alerts = [
            alert_time for alert_time in self.suppression_tracker[fingerprint]
            if alert_time > hour_ago
        ]

        if len(recent_alerts) >= rule.max_alerts_per_hour:
            # 添加到抑制列表
            self.suppressed_alerts.add(fingerprint)
            # 设置抑制过期时间
            asyncio.create_task(self._remove_suppression_after_delay(fingerprint, rule.suppression_duration))
            return True

        # 记录本次预警时间
        self.suppression_tracker[fingerprint].append(current_time)

        return False

    async def _remove_suppression_after_delay(self, fingerprint: str, delay_seconds: int):
        """延迟移除抑制"""
        await asyncio.sleep(delay_seconds)
        self.suppressed_alerts.discard(fingerprint)

    def _find_existing_alert(self, fingerprint: str) -> Optional[AlertEvent]:
        """查找现有预警"""
        for alert in self.active_alerts.values():
            if alert.get_fingerprint() == fingerprint and alert.status == AlertStatus.ACTIVE:
                return alert
        return None

    async def _send_notifications(self, alert_event: AlertEvent, rule: AlertRule):
        """发送通知"""
        for channel in rule.notification_channels:
            try:
                notification_task = {
                    'channel': channel,
                    'alert_event': alert_event,
                    'rule': rule
                }
                await self.notification_queue.put(notification_task)

            except Exception as e:
                logger.error(f"添加通知任务失败 {channel.value}: {e}")

    async def _process_notifications(self):
        """处理通知队列"""
        while not self._stop_event.is_set():
            try:
                # 等待通知任务
                notification_task = await asyncio.wait_for(
                    self.notification_queue.get(),
                    timeout=1.0
                )

                channel = notification_task['channel']
                alert_event = notification_task['alert_event']
                rule = notification_task['rule']

                # 根据渠道发送通知
                success = await self._send_notification_by_channel(channel, alert_event, rule)

                if success:
                    alert_event.notification_sent.append(channel)
                    self.alert_statistics['notifications_sent'][channel.value] += 1

                # 标记任务完成
                self.notification_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"处理通知异常: {e}")

    async def _send_notification_by_channel(self,
                                          channel: NotificationChannel,
                                          alert_event: AlertEvent,
                                          rule: AlertRule) -> bool:
        """通过指定渠道发送通知"""
        try:
            if channel == NotificationChannel.EMAIL:
                return await self._send_email_notification(alert_event, rule)
            elif channel == NotificationChannel.SMS:
                return await self._send_sms_notification(alert_event, rule)
            elif channel == NotificationChannel.WECHAT:
                return await self._send_wechat_notification(alert_event, rule)
            elif channel == NotificationChannel.DINGTALK:
                return await self._send_dingtalk_notification(alert_event, rule)
            elif channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook_notification(alert_event, rule)
            elif channel == NotificationChannel.IN_APP:
                return await self._send_in_app_notification(alert_event, rule)
            elif channel == NotificationChannel.SYSTEM_LOG:
                return await self._send_system_log_notification(alert_event, rule)
            else:
                logger.warning(f"不支持的通知渠道: {channel}")
                return False

        except Exception as e:
            logger.error(f"发送 {channel.value} 通知失败: {e}")
            return False

    async def _send_email_notification(self, alert_event: AlertEvent, rule: AlertRule) -> bool:
        """发送邮件通知（模拟实现）"""
        # 实际实现应该使用 smtplib 或 aiosmtplib
        logger.info(f"[EMAIL] {alert_event.title}: {alert_event.message}")

        # 模拟发送延迟
        await asyncio.sleep(0.1)

        return True

    async def _send_sms_notification(self, alert_event: AlertEvent, rule: AlertRule) -> bool:
        """发送短信通知（模拟实现）"""
        # 实际实现应该集成短信服务商API
        if alert_event.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            logger.info(f"[SMS] {alert_event.title}: {alert_event.message}")
            await asyncio.sleep(0.2)
            return True

        return False  # 只对严重和紧急预警发送短信

    async def _send_wechat_notification(self, alert_event: AlertEvent, rule: AlertRule) -> bool:
        """发送微信通知（模拟实现）"""
        # 实际实现应该使用企业微信API
        logger.info(f"[WECHAT] {alert_event.title}: {alert_event.message}")
        await asyncio.sleep(0.15)
        return True

    async def _send_dingtalk_notification(self, alert_event: AlertEvent, rule: AlertRule) -> bool:
        """发送钉钉通知（模拟实现）"""
        # 实际实现应该使用钉钉机器人API
        logger.info(f"[DINGTALK] {alert_event.title}: {alert_event.message}")
        await asyncio.sleep(0.12)
        return True

    async def _send_webhook_notification(self, alert_event: AlertEvent, rule: AlertRule) -> bool:
        """发送Webhook通知（模拟实现）"""
        # 实际实现应该发送HTTP请求到配置的Webhook URL
        webhook_data = {
            'alert_id': alert_event.alert_id,
            'level': alert_event.alert_level.value,
            'title': alert_event.title,
            'message': alert_event.message,
            'timestamp': alert_event.trigger_time.isoformat()
        }

        logger.info(f"[WEBHOOK] {json.dumps(webhook_data, ensure_ascii=False)}")
        await asyncio.sleep(0.05)
        return True

    async def _send_in_app_notification(self, alert_event: AlertEvent, rule: AlertRule) -> bool:
        """发送应用内通知"""
        # 实际实现应该推送到应用内通知队列
        logger.info(f"[IN-APP] {alert_event.title}: {alert_event.message}")
        return True

    async def _send_system_log_notification(self, alert_event: AlertEvent, rule: AlertRule) -> bool:
        """发送系统日志通知"""
        if alert_event.alert_level == AlertLevel.EMERGENCY:
            logger.critical(f"[ALERT] {alert_event.message}")
        elif alert_event.alert_level == AlertLevel.CRITICAL:
            logger.error(f"[ALERT] {alert_event.message}")
        elif alert_event.alert_level == AlertLevel.WARNING:
            logger.warning(f"[ALERT] {alert_event.message}")
        else:
            logger.info(f"[ALERT] {alert_event.message}")

        return True

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """确认预警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_time = datetime.now()

            self.alert_statistics['acknowledged_count'] += 1

            logger.info(f"预警已确认: {alert_id} by {acknowledged_by}")
            return True

        return False

    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """解决预警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_time = datetime.now()

            # 从活动预警中移除
            del self.active_alerts[alert_id]

            self.alert_statistics['resolved_count'] += 1

            logger.info(f"预警已解决: {alert_id} by {resolved_by}")
            return True

        return False

    def add_alert_rule(self, rule: AlertRule):
        """添加预警规则"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"添加预警规则: {rule.rule_name}")

    def remove_alert_rule(self, rule_id: str) -> bool:
        """移除预警规则"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"移除预警规则: {rule_id}")
            return True
        return False

    def enable_alert_rule(self, rule_id: str, enabled: bool = True):
        """启用/禁用预警规则"""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = enabled
            status = "启用" if enabled else "禁用"
            logger.info(f"预警规则 {rule_id} 已{status}")

    def get_active_alerts(self) -> List[AlertEvent]:
        """获取活动预警列表"""
        return list(self.active_alerts.values())

    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取预警统计信息"""
        return {
            **self.alert_statistics,
            'active_alerts_count': len(self.active_alerts),
            'alert_rules_count': len(self.alert_rules),
            'enabled_rules_count': sum(1 for rule in self.alert_rules.values() if rule.enabled),
            'suppressed_fingerprints': len(self.suppressed_alerts)
        }

    def get_alert_history(self, hours: int = 24) -> List[AlertEvent]:
        """获取历史预警"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.trigger_time > cutoff_time
        ]

    async def start_notification_processing(self):
        """启动通知处理"""
        if self.is_running:
            logger.warning("通知处理已在运行")
            return

        self.is_running = True
        self._stop_event.clear()

        # 启动通知处理任务
        for i in range(3):  # 启动3个并发处理任务
            task = asyncio.create_task(self._process_notifications())
            self.notification_tasks.append(task)

        logger.info("智能预警通知处理已启动")

    async def stop_notification_processing(self):
        """停止通知处理"""
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()

        # 等待处理完队列中的任务
        await self.notification_queue.join()

        # 取消处理任务
        for task in self.notification_tasks:
            task.cancel()

        # 等待所有任务完成
        await asyncio.gather(*self.notification_tasks, return_exceptions=True)
        self.notification_tasks.clear()

        logger.info("智能预警通知处理已停止")

    def export_alert_rules(self) -> str:
        """导出预警规则（JSON格式）"""
        rules_data = [rule.to_dict() for rule in self.alert_rules.values()]
        return json.dumps(rules_data, indent=2, ensure_ascii=False)

    def import_alert_rules(self, rules_json: str) -> int:
        """导入预警规则"""
        try:
            rules_data = json.loads(rules_json)
            imported_count = 0

            for rule_data in rules_data:
                rule = AlertRule(
                    rule_id=rule_data['rule_id'],
                    rule_name=rule_data['rule_name'],
                    alert_type=AlertType(rule_data['alert_type']),
                    alert_level=AlertLevel(rule_data['alert_level']),
                    enabled=rule_data.get('enabled', True),
                    condition_expression=rule_data.get('condition_expression', ''),
                    threshold_value=rule_data.get('threshold_value', 0.0),
                    duration_seconds=rule_data.get('duration_seconds', 0),
                    notification_channels=[
                        NotificationChannel(ch) for ch in rule_data.get('notification_channels', [])
                    ],
                    notification_template=rule_data.get('notification_template', ''),
                    suppression_duration=rule_data.get('suppression_duration', 3600),
                    max_alerts_per_hour=rule_data.get('max_alerts_per_hour', 10),
                    recipients=rule_data.get('recipients', [])
                )

                self.alert_rules[rule.rule_id] = rule
                imported_count += 1

            logger.info(f"成功导入 {imported_count} 条预警规则")
            return imported_count

        except Exception as e:
            logger.error(f"导入预警规则失败: {e}")
            return 0


# 全局实例管理
_intelligent_alert_system = None

def get_intelligent_alert_system(
    notification_config: Optional[NotificationConfig] = None,
    data_manager=None
) -> IntelligentRiskAlertSystem:
    """
    获取智能预警系统实例（单例模式）

    Args:
        notification_config: 通知配置
        data_manager: 数据管理器

    Returns:
        IntelligentRiskAlertSystem: 智能预警系统实例
    """
    global _intelligent_alert_system

    if _intelligent_alert_system is None:
        _intelligent_alert_system = IntelligentRiskAlertSystem(
            notification_config=notification_config,
            data_manager=data_manager
        )

    return _intelligent_alert_system


if __name__ == "__main__":
    # 演示使用
    import asyncio

    async def demo_alert_system():
        print("=== 智能风险预警通知系统演示 ===")

        # 创建预警系统
        alert_system = get_intelligent_alert_system()

        # 启动通知处理
        await alert_system.start_notification_processing()

        # 模拟触发各种预警
        test_alerts = [
            {
                'rule_id': 'RISK_001',
                'trigger_value': 87.5,
                'related_data': {'portfolio_id': 'P001'}
            },
            {
                'rule_id': 'RISK_002',
                'trigger_value': 0.08,
                'related_data': {'portfolio_id': 'P001', 'drawdown_duration': 5}
            },
            {
                'rule_id': 'POS_001',
                'trigger_value': 0.12,
                'related_data': {'stock_code': '000001', 'stock_name': '平安银行'}
            },
            {
                'rule_id': 'RISK_003',
                'trigger_value': 65000.0,
                'related_data': {'portfolio_id': 'P001', 'confidence_level': '95%'}
            },
            {
                'rule_id': 'SYS_001',
                'trigger_value': 15.2,
                'related_data': {'system_component': 'risk_engine'}
            }
        ]

        # 触发预警
        for alert_data in test_alerts:
            alert_event = await alert_system.trigger_alert(
                rule_id=alert_data['rule_id'],
                trigger_value=alert_data['trigger_value'],
                related_data=alert_data['related_data']
            )

            if alert_event:
                print(f"预警已触发: {alert_event.title}")

            # 间隔一点时间
            await asyncio.sleep(0.5)

        # 等待通知处理完成
        await asyncio.sleep(2)

        # 获取统计信息
        stats = alert_system.get_alert_statistics()
        print(f"\n--- 预警统计 ---")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")

        # 获取活动预警
        active_alerts = alert_system.get_active_alerts()
        print(f"\n--- 活动预警 ({len(active_alerts)}) ---")
        for alert in active_alerts[:3]:  # 只显示前3个
            print(f"- {alert.title}: {alert.message}")

        # 确认一个预警
        if active_alerts:
            first_alert = active_alerts[0]
            alert_system.acknowledge_alert(first_alert.alert_id, "demo_user")
            print(f"\n已确认预警: {first_alert.title}")

        # 解决一个预警
        if len(active_alerts) > 1:
            second_alert = active_alerts[1]
            alert_system.resolve_alert(second_alert.alert_id, "demo_user")
            print(f"已解决预警: {second_alert.title}")

        # 停止通知处理
        await alert_system.stop_notification_processing()

        print("\n演示完成")

    # 运行演示
    asyncio.run(demo_alert_system())