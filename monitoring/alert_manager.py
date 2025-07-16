#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
告警管理器

提供告警通知、告警历史管理和告警规则配置功能
"""

import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from utils.logger import getLogger
from utils.decorators import exception_handler

logger = getLogger(__name__)


class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "email"
    LOG = "log"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class NotificationConfig:
    """通知配置"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = None


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    condition: str
    level: str
    message_template: str
    cooldown_seconds: int = 300
    enabled: bool = True
    notification_channels: List[NotificationChannel] = None


class AlertManager:
    """
    告警管理器
    
    功能特性：
    1. 告警通知发送
    2. 告警历史管理
    3. 告警规则配置
    4. 多渠道通知支持
    5. 告警抑制和聚合
    """
    
    def __init__(self, config_file: str = None):
        """
        初始化告警管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        
        # 通知配置
        self.notification_configs = self._load_notification_configs()
        
        # 告警规则
        self.alert_rules = self._load_alert_rules()
        
        # 告警历史
        self.alert_history = []
        self.alert_stats = {}
        
        # 告警抑制（防止重复告警）
        self.suppressed_alerts = {}
        self.suppression_lock = threading.Lock()
        
        # 通知队列
        self.notification_queue = []
        self.notification_lock = threading.Lock()
        
        # 通知处理线程
        self.notification_thread = None
        self.notification_running = False
        
        logger.info("告警管理器初始化完成")
    
    def _load_notification_configs(self) -> Dict[str, NotificationConfig]:
        """加载通知配置"""
        default_configs = {
            'email': NotificationConfig(
                channel=NotificationChannel.EMAIL,
                enabled=False,
                config={
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'from_email': '',
                    'to_emails': []
                }
            ),
            'log': NotificationConfig(
                channel=NotificationChannel.LOG,
                enabled=True,
                config={}
            ),
            'console': NotificationConfig(
                channel=NotificationChannel.CONSOLE,
                enabled=True,
                config={}
            ),
            'webhook': NotificationConfig(
                channel=NotificationChannel.WEBHOOK,
                enabled=False,
                config={
                    'url': '',
                    'headers': {},
                    'timeout': 10
                }
            )
        }
        
        # 如果有配置文件，加载配置
        if self.config_file:
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    
                # 更新默认配置
                for channel, config in file_config.get('notifications', {}).items():
                    if channel in default_configs:
                        default_configs[channel].enabled = config.get('enabled', True)
                        default_configs[channel].config.update(config.get('config', {}))
                        
            except Exception as e:
                logger.warning(f"加载通知配置失败: {e}，使用默认配置")
        
        return default_configs
    
    def _load_alert_rules(self) -> Dict[str, AlertRule]:
        """加载告警规则"""
        default_rules = {
            'high_cpu': AlertRule(
                name='high_cpu',
                condition='cpu_percent > 80',
                level='WARNING',
                message_template='CPU使用率过高: {cpu_percent:.1f}%',
                cooldown_seconds=300,
                notification_channels=[NotificationChannel.LOG, NotificationChannel.CONSOLE]
            ),
            'high_memory': AlertRule(
                name='high_memory',
                condition='memory_percent > 85',
                level='WARNING',
                message_template='内存使用率过高: {memory_percent:.1f}%',
                cooldown_seconds=300,
                notification_channels=[NotificationChannel.LOG, NotificationChannel.CONSOLE]
            ),
            'slow_query': AlertRule(
                name='slow_query',
                condition='avg_query_time > 5.0',
                level='WARNING',
                message_template='查询响应时间过长: {avg_query_time:.2f}s',
                cooldown_seconds=300,
                notification_channels=[NotificationChannel.LOG, NotificationChannel.EMAIL]
            ),
            'high_error_rate': AlertRule(
                name='high_error_rate',
                condition='error_rate > 0.05',
                level='ERROR',
                message_template='错误率过高: {error_rate:.2%}',
                cooldown_seconds=180,
                notification_channels=[NotificationChannel.LOG, NotificationChannel.EMAIL, NotificationChannel.CONSOLE]
            )
        }
        
        # 如果有配置文件，加载规则
        if self.config_file:
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    
                # 加载自定义规则
                for rule_name, rule_config in file_config.get('alert_rules', {}).items():
                    default_rules[rule_name] = AlertRule(
                        name=rule_name,
                        condition=rule_config.get('condition', ''),
                        level=rule_config.get('level', 'INFO'),
                        message_template=rule_config.get('message_template', ''),
                        cooldown_seconds=rule_config.get('cooldown_seconds', 300),
                        enabled=rule_config.get('enabled', True),
                        notification_channels=[
                            NotificationChannel(ch) for ch in rule_config.get('notification_channels', ['log'])
                        ]
                    )
                    
            except Exception as e:
                logger.warning(f"加载告警规则失败: {e}，使用默认规则")
        
        return default_rules
    
    def start_notification_service(self):
        """启动通知服务"""
        if self.notification_running:
            return
        
        self.notification_running = True
        self.notification_thread = threading.Thread(target=self._notification_worker, daemon=True)
        self.notification_thread.start()
        
        logger.info("告警通知服务已启动")
    
    def stop_notification_service(self):
        """停止通知服务"""
        self.notification_running = False
        if self.notification_thread:
            self.notification_thread.join()
        
        logger.info("告警通知服务已停止")
    
    def _notification_worker(self):
        """通知处理工作线程"""
        while self.notification_running:
            try:
                # 处理通知队列
                notifications_to_send = []
                
                with self.notification_lock:
                    if self.notification_queue:
                        notifications_to_send = self.notification_queue.copy()
                        self.notification_queue.clear()
                
                # 发送通知
                for notification in notifications_to_send:
                    self._send_notification(notification)
                
                time.sleep(1)  # 避免过于频繁的轮询
                
            except Exception as e:
                logger.error(f"通知处理出错: {e}")
    
    @exception_handler(reraise=False)
    def handle_alert(self, alert_data: Dict[str, Any]):
        """
        处理告警
        
        Args:
            alert_data: 告警数据
        """
        alert_id = alert_data.get('id', '')
        alert_level = alert_data.get('level', 'INFO')
        alert_type = alert_data.get('type', '')
        alert_message = alert_data.get('message', '')
        alert_details = alert_data.get('details', {})
        
        # 检查告警抑制
        if self._is_alert_suppressed(alert_id, alert_type):
            logger.debug(f"告警被抑制: {alert_id}")
            return
        
        # 记录告警历史
        self._record_alert_history(alert_data)
        
        # 更新告警统计
        self._update_alert_stats(alert_level, alert_type)
        
        # 确定通知渠道
        notification_channels = self._get_notification_channels(alert_type, alert_level)
        
        # 生成通知
        notification = {
            'alert_id': alert_id,
            'timestamp': datetime.now(),
            'level': alert_level,
            'type': alert_type,
            'message': alert_message,
            'details': alert_details,
            'channels': notification_channels
        }
        
        # 添加到通知队列
        with self.notification_lock:
            self.notification_queue.append(notification)
        
        # 设置告警抑制
        self._set_alert_suppression(alert_id, alert_type)
        
        logger.info(f"处理告警: {alert_level} - {alert_message}")
    
    def _is_alert_suppressed(self, alert_id: str, alert_type: str) -> bool:
        """检查告警是否被抑制"""
        with self.suppression_lock:
            suppression_key = f"{alert_type}_{alert_id}"
            
            if suppression_key in self.suppressed_alerts:
                suppression_time = self.suppressed_alerts[suppression_key]
                
                # 检查抑制是否过期
                if time.time() - suppression_time < 300:  # 5分钟抑制期
                    return True
                else:
                    # 抑制过期，移除
                    del self.suppressed_alerts[suppression_key]
        
        return False
    
    def _set_alert_suppression(self, alert_id: str, alert_type: str):
        """设置告警抑制"""
        with self.suppression_lock:
            suppression_key = f"{alert_type}_{alert_id}"
            self.suppressed_alerts[suppression_key] = time.time()
    
    def _record_alert_history(self, alert_data: Dict[str, Any]):
        """记录告警历史"""
        alert_record = {
            'timestamp': datetime.now(),
            'data': alert_data
        }
        
        self.alert_history.append(alert_record)
        
        # 限制历史记录数量
        if len(self.alert_history) > 10000:
            self.alert_history = self.alert_history[-5000:]  # 保留最近5000条
    
    def _update_alert_stats(self, alert_level: str, alert_type: str):
        """更新告警统计"""
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        if current_hour not in self.alert_stats:
            self.alert_stats[current_hour] = {}
        
        key = f"{alert_level}_{alert_type}"
        self.alert_stats[current_hour][key] = self.alert_stats[current_hour].get(key, 0) + 1
        
        # 清理旧统计数据（保留24小时）
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alert_stats = {
            k: v for k, v in self.alert_stats.items() 
            if k > cutoff_time
        }
    
    def _get_notification_channels(self, alert_type: str, alert_level: str) -> List[NotificationChannel]:
        """获取通知渠道"""
        # 根据告警类型和级别确定通知渠道
        channels = [NotificationChannel.LOG, NotificationChannel.CONSOLE]
        
        # 严重告警增加邮件通知
        if alert_level in ['ERROR', 'CRITICAL']:
            channels.append(NotificationChannel.EMAIL)
        
        # 过滤启用的通知渠道
        enabled_channels = []
        for channel in channels:
            config = self.notification_configs.get(channel.value)
            if config and config.enabled:
                enabled_channels.append(channel)
        
        return enabled_channels
    
    @exception_handler(reraise=False)
    def _send_notification(self, notification: Dict[str, Any]):
        """发送通知"""
        for channel in notification['channels']:
            try:
                if channel == NotificationChannel.LOG:
                    self._send_log_notification(notification)
                elif channel == NotificationChannel.CONSOLE:
                    self._send_console_notification(notification)
                elif channel == NotificationChannel.EMAIL:
                    self._send_email_notification(notification)
                elif channel == NotificationChannel.WEBHOOK:
                    self._send_webhook_notification(notification)
                    
            except Exception as e:
                logger.error(f"发送 {channel.value} 通知失败: {e}")
    
    def _send_log_notification(self, notification: Dict[str, Any]):
        """发送日志通知"""
        level = notification['level']
        message = notification['message']
        
        if level == 'CRITICAL':
            logger.critical(f"[告警] {message}")
        elif level == 'ERROR':
            logger.error(f"[告警] {message}")
        elif level == 'WARNING':
            logger.warning(f"[告警] {message}")
        else:
            logger.info(f"[告警] {message}")
    
    def _send_console_notification(self, notification: Dict[str, Any]):
        """发送控制台通知"""
        timestamp = notification['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        level = notification['level']
        message = notification['message']
        
        print(f"[{timestamp}] {level}: {message}")
    
    def _send_email_notification(self, notification: Dict[str, Any]):
        """发送邮件通知"""
        email_config = self.notification_configs.get('email')
        if not email_config or not email_config.enabled:
            return
        
        config = email_config.config
        if not config.get('to_emails'):
            return
        
        # 构造邮件内容
        subject = f"[{notification['level']}] 系统告警 - {notification['type']}"
        body = f"""
        告警时间: {notification['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        告警级别: {notification['level']}
        告警类型: {notification['type']}
        告警消息: {notification['message']}
        
        详细信息:
        {json.dumps(notification['details'], indent=2, ensure_ascii=False)}
        """
        
        # 发送邮件
        try:
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            
            text = msg.as_string()
            server.sendmail(config['from_email'], config['to_emails'], text)
            server.quit()
            
            logger.info(f"邮件告警发送成功: {subject}")
            
        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")
    
    def _send_webhook_notification(self, notification: Dict[str, Any]):
        """发送Webhook通知"""
        webhook_config = self.notification_configs.get('webhook')
        if not webhook_config or not webhook_config.enabled:
            return
        
        config = webhook_config.config
        if not config.get('url'):
            return
        
        try:
            import requests
            
            payload = {
                'timestamp': notification['timestamp'].isoformat(),
                'level': notification['level'],
                'type': notification['type'],
                'message': notification['message'],
                'details': notification['details']
            }
            
            response = requests.post(
                config['url'],
                json=payload,
                headers=config.get('headers', {}),
                timeout=config.get('timeout', 10)
            )
            
            if response.status_code == 200:
                logger.info("Webhook告警发送成功")
            else:
                logger.warning(f"Webhook告警响应异常: {response.status_code}")
                
        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")
    
    def get_alert_stats(self, hours: int = 24) -> Dict[str, Any]:
        """获取告警统计信息"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        stats = {
            'total_alerts': 0,
            'by_level': {},
            'by_type': {},
            'by_hour': {},
            'recent_alerts': []
        }
        
        # 统计告警历史
        for record in self.alert_history:
            if record['timestamp'] > cutoff_time:
                alert_data = record['data']
                level = alert_data.get('level', 'INFO')
                alert_type = alert_data.get('type', 'UNKNOWN')
                hour = record['timestamp'].strftime('%H:00')
                
                stats['total_alerts'] += 1
                stats['by_level'][level] = stats['by_level'].get(level, 0) + 1
                stats['by_type'][alert_type] = stats['by_type'].get(alert_type, 0) + 1
                stats['by_hour'][hour] = stats['by_hour'].get(hour, 0) + 1
                
                # 最近10条告警
                if len(stats['recent_alerts']) < 10:
                    stats['recent_alerts'].append({
                        'timestamp': record['timestamp'].isoformat(),
                        'level': level,
                        'type': alert_type,
                        'message': alert_data.get('message', '')
                    })
        
        return stats
    
    def clear_alert_history(self):
        """清空告警历史"""
        self.alert_history.clear()
        self.alert_stats.clear()
        logger.info("告警历史已清空")


# 全局告警管理器实例
_global_alert_manager = None

def get_alert_manager() -> AlertManager:
    """获取全局告警管理器实例"""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
    return _global_alert_manager 