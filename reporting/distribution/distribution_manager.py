#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动化报告分发管理系统

核心功能：
- 邮件自动发送
- 文件系统归档
- FTP/SFTP上传
- API接口分发
- 消息通知（钉钉、企业微信）
- 分发状态跟踪和重试机制

支持的分发渠道：
- SMTP邮件服务
- 本地/网络文件系统
- 云存储服务
- REST API接口
- 即时通讯工具
"""

import os
import sys
import json
import smtplib
import ftplib
import shutil
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import requests
import schedule

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, root_dir)

from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, exception_handler

logger = get_logger(__name__)


@dataclass
class EmailConfig:
    """邮件配置"""
    smtp_server: str
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    sender_name: str = "量化系统"
    sender_email: str = ""


@dataclass
class FTPConfig:
    """FTP配置"""
    host: str
    port: int = 21
    username: str = ""
    password: str = ""
    remote_path: str = "/"
    use_sftp: bool = False


@dataclass
class APIConfig:
    """API配置"""
    endpoint: str
    method: str = "POST"
    headers: Dict[str, str] = None
    auth_token: str = ""
    timeout: int = 30


@dataclass
class DistributionTask:
    """分发任务"""
    task_id: str
    report_files: Dict[str, str]
    recipients: List[str]
    distribution_channels: List[str]
    priority: int = 1  # 1-高, 2-中, 3-低
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime
    updated_at: datetime
    error_message: str = ""


class DistributionManager:
    """
    自动化报告分发管理器

    功能：
    1. 多渠道报告分发
    2. 分发状态管理
    3. 失败重试机制
    4. 分发历史记录
    5. 定时分发任务
    """

    def __init__(self,
                 config: Optional[Any] = None,
                 work_dir: str = "./distribution"):
        """
        初始化分发管理器

        Args:
            config: 配置对象
            work_dir: 工作目录
        """
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # 分发配置
        self.email_config = None
        self.ftp_config = None
        self.api_configs = {}

        # 任务队列和历史
        self.task_queue = []
        self.task_history = []

        # 分发统计
        self.distribution_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'email_sent': 0,
            'files_archived': 0,
            'api_calls': 0
        }

        # 初始化配置
        self._load_configurations()

        logger.info(f"分发管理器初始化完成，工作目录: {self.work_dir}")

    def _load_configurations(self):
        """加载分发配置"""
        try:
            # 从环境变量或配置文件加载
            if hasattr(self.config, 'email_recipients') and self.config.email_recipients:
                self.email_config = EmailConfig(
                    smtp_server=os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                    smtp_port=int(os.getenv('SMTP_PORT', '587')),
                    username=os.getenv('SMTP_USERNAME', ''),
                    password=os.getenv('SMTP_PASSWORD', ''),
                    sender_email=os.getenv('SENDER_EMAIL', ''),
                    sender_name=os.getenv('SENDER_NAME', '量化报告系统')
                )
                logger.info("邮件配置加载完成")

            # FTP配置
            if os.getenv('FTP_HOST'):
                self.ftp_config = FTPConfig(
                    host=os.getenv('FTP_HOST'),
                    port=int(os.getenv('FTP_PORT', '21')),
                    username=os.getenv('FTP_USERNAME', ''),
                    password=os.getenv('FTP_PASSWORD', ''),
                    remote_path=os.getenv('FTP_REMOTE_PATH', '/reports')
                )
                logger.info("FTP配置加载完成")

            # API配置
            if os.getenv('WEBHOOK_URL'):
                self.api_configs['webhook'] = APIConfig(
                    endpoint=os.getenv('WEBHOOK_URL'),
                    auth_token=os.getenv('WEBHOOK_TOKEN', ''),
                    headers={'Content-Type': 'application/json'}
                )
                logger.info("Webhook配置加载完成")

        except Exception as e:
            logger.warning(f"加载分发配置时出现警告: {e}")

    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def distribute_reports(self,
                          report_files: Dict[str, str],
                          request: Any) -> Dict[str, Any]:
        """
        分发报告

        Args:
            report_files: 报告文件字典
            request: 报告生成请求

        Returns:
            Dict[str, Any]: 分发结果
        """
        if not (request.config.auto_email or request.config.auto_archive):
            return {'distribution_enabled': False}

        # 创建分发任务
        task = DistributionTask(
            task_id=f"dist_{request.request_id}",
            report_files=report_files,
            recipients=request.config.email_recipients or [],
            distribution_channels=self._get_enabled_channels(request.config),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        logger.info(f"开始分发报告: {task.task_id}")

        try:
            # 执行分发
            results = self._execute_distribution(task)

            # 更新任务状态
            task.status = "completed"
            task.updated_at = datetime.now()

            # 更新统计
            self.distribution_stats['total_tasks'] += 1
            self.distribution_stats['successful_tasks'] += 1

            # 记录历史
            self.task_history.append(task)

            logger.info(f"报告分发完成: {task.task_id}")

            return {
                'distribution_enabled': True,
                'task_id': task.task_id,
                'status': 'completed',
                'results': results,
                'distribution_time': (task.updated_at - task.created_at).total_seconds()
            }

        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            task.updated_at = datetime.now()

            self.distribution_stats['total_tasks'] += 1
            self.distribution_stats['failed_tasks'] += 1

            # 添加到重试队列
            if task.retry_count < task.max_retries:
                self.task_queue.append(task)

            logger.error(f"报告分发失败: {task.task_id} - {e}")

            return {
                'distribution_enabled': True,
                'task_id': task.task_id,
                'status': 'failed',
                'error': str(e)
            }

    def _get_enabled_channels(self, config: Any) -> List[str]:
        """获取启用的分发渠道"""
        channels = []

        if config.auto_email and self.email_config:
            channels.append('email')

        if config.auto_archive:
            channels.append('archive')

        if self.ftp_config:
            channels.append('ftp')

        if self.api_configs:
            channels.extend(self.api_configs.keys())

        return channels

    def _execute_distribution(self, task: DistributionTask) -> Dict[str, Any]:
        """执行分发任务"""
        results = {}

        for channel in task.distribution_channels:
            try:
                if channel == 'email':
                    result = self._send_email(task)
                    results['email'] = result
                elif channel == 'archive':
                    result = self._archive_files(task)
                    results['archive'] = result
                elif channel == 'ftp':
                    result = self._upload_to_ftp(task)
                    results['ftp'] = result
                elif channel in self.api_configs:
                    result = self._send_to_api(task, channel)
                    results[channel] = result

            except Exception as e:
                logger.error(f"分发渠道 {channel} 失败: {e}")
                results[channel] = {'status': 'failed', 'error': str(e)}

        return results

    def _send_email(self, task: DistributionTask) -> Dict[str, Any]:
        """发送邮件"""
        if not self.email_config or not task.recipients:
            return {'status': 'skipped', 'reason': 'no_config_or_recipients'}

        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = f"{self.email_config.sender_name} <{self.email_config.sender_email}>"
            msg['To'] = ', '.join(task.recipients)
            msg['Subject'] = f"量化策略回测报告 - {datetime.now().strftime('%Y-%m-%d')}"

            # 邮件正文
            body = self._generate_email_body(task)
            msg.attach(MIMEText(body, 'html', 'utf-8'))

            # 添加附件
            for format_type, file_path in task.report_files.items():
                if file_path and Path(file_path).exists():
                    with open(file_path, 'rb') as f:
                        attachment = MIMEApplication(f.read())
                        attachment.add_header('Content-Disposition', 'attachment',
                                            filename=Path(file_path).name)
                        msg.attach(attachment)

            # 发送邮件
            with smtplib.SMTP(self.email_config.smtp_server, self.email_config.smtp_port) as server:
                if self.email_config.use_tls:
                    server.starttls()

                if self.email_config.username and self.email_config.password:
                    server.login(self.email_config.username, self.email_config.password)

                server.send_message(msg)

            self.distribution_stats['email_sent'] += 1

            logger.info(f"邮件发送成功: {task.task_id}")

            return {
                'status': 'success',
                'recipients': len(task.recipients),
                'attachments': len([f for f in task.report_files.values() if f])
            }

        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
            raise

    def _generate_email_body(self, task: DistributionTask) -> str:
        """生成邮件正文"""
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; line-height: 1.6; }}
                .header {{ background-color: #2E86AB; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; color: #6c757d; }}
                .file-list {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>量化策略回测报告</h1>
                <p>系统自动生成于 {datetime.now().strftime('%Y年%m月%d日 %H:%M')}</p>
            </div>

            <div class="content">
                <h2>报告摘要</h2>
                <p>本邮件包含最新的量化策略回测分析报告，请查看附件获取详细信息。</p>

                <div class="file-list">
                    <h3>附件列表:</h3>
                    <ul>
                    {chr(10).join([f'<li>{Path(file_path).name} ({format_type.upper()})</li>'
                                  for format_type, file_path in task.report_files.items()
                                  if file_path and Path(file_path).exists()])}
                    </ul>
                </div>

                <p><strong>注意事项：</strong></p>
                <ul>
                    <li>本报告仅供内部使用，请勿转发给无关人员</li>
                    <li>报告数据基于历史回测，不构成投资建议</li>
                    <li>如有疑问，请联系量化团队</li>
                </ul>
            </div>

            <div class="footer">
                <p>此邮件由量化报告系统自动发送，请勿回复</p>
                <p>如需取消订阅，请联系系统管理员</p>
            </div>
        </body>
        </html>
        """

    def _archive_files(self, task: DistributionTask) -> Dict[str, Any]:
        """归档文件"""
        try:
            # 创建归档目录
            archive_date = datetime.now().strftime('%Y%m%d')
            archive_dir = self.work_dir / "archive" / archive_date
            archive_dir.mkdir(parents=True, exist_ok=True)

            archived_files = []

            # 复制文件到归档目录
            for format_type, file_path in task.report_files.items():
                if file_path and Path(file_path).exists():
                    source_file = Path(file_path)
                    target_file = archive_dir / f"{task.task_id}_{source_file.name}"

                    shutil.copy2(source_file, target_file)
                    archived_files.append(str(target_file))

            # 清理旧归档（保留指定天数）
            retention_days = getattr(self.config, 'archive_retention_days', 90)
            self._cleanup_old_archives(retention_days)

            self.distribution_stats['files_archived'] += len(archived_files)

            logger.info(f"文件归档完成: {task.task_id}")

            return {
                'status': 'success',
                'archive_path': str(archive_dir),
                'archived_files': len(archived_files)
            }

        except Exception as e:
            logger.error(f"文件归档失败: {e}")
            raise

    def _cleanup_old_archives(self, retention_days: int):
        """清理旧归档"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            archive_root = self.work_dir / "archive"

            if not archive_root.exists():
                return

            for date_dir in archive_root.iterdir():
                if date_dir.is_dir():
                    try:
                        # 解析日期目录名
                        dir_date = datetime.strptime(date_dir.name, '%Y%m%d')
                        if dir_date < cutoff_date:
                            shutil.rmtree(date_dir)
                            logger.info(f"清理旧归档: {date_dir}")
                    except ValueError:
                        # 跳过无法解析日期的目录
                        continue

        except Exception as e:
            logger.warning(f"清理旧归档失败: {e}")

    def _upload_to_ftp(self, task: DistributionTask) -> Dict[str, Any]:
        """上传到FTP服务器"""
        if not self.ftp_config:
            return {'status': 'skipped', 'reason': 'no_ftp_config'}

        try:
            uploaded_files = []

            with ftplib.FTP() as ftp:
                ftp.connect(self.ftp_config.host, self.ftp_config.port)

                if self.ftp_config.username:
                    ftp.login(self.ftp_config.username, self.ftp_config.password)
                else:
                    ftp.login()

                # 切换到远程目录
                try:
                    ftp.cwd(self.ftp_config.remote_path)
                except ftplib.error_perm:
                    # 创建远程目录
                    self._create_ftp_directory(ftp, self.ftp_config.remote_path)
                    ftp.cwd(self.ftp_config.remote_path)

                # 上传文件
                for format_type, file_path in task.report_files.items():
                    if file_path and Path(file_path).exists():
                        filename = Path(file_path).name
                        remote_filename = f"{task.task_id}_{filename}"

                        with open(file_path, 'rb') as f:
                            ftp.storbinary(f'STOR {remote_filename}', f)

                        uploaded_files.append(remote_filename)

            logger.info(f"FTP上传完成: {task.task_id}")

            return {
                'status': 'success',
                'uploaded_files': len(uploaded_files),
                'remote_path': self.ftp_config.remote_path
            }

        except Exception as e:
            logger.error(f"FTP上传失败: {e}")
            raise

    def _create_ftp_directory(self, ftp: ftplib.FTP, path: str):
        """创建FTP目录"""
        parts = path.strip('/').split('/')
        current_path = ''

        for part in parts:
            current_path += '/' + part
            try:
                ftp.mkd(current_path)
            except ftplib.error_perm:
                # 目录可能已存在
                pass

    def _send_to_api(self, task: DistributionTask, channel: str) -> Dict[str, Any]:
        """发送到API接口"""
        api_config = self.api_configs.get(channel)
        if not api_config:
            return {'status': 'skipped', 'reason': 'no_api_config'}

        try:
            # 准备API数据
            payload = {
                'task_id': task.task_id,
                'timestamp': datetime.now().isoformat(),
                'report_files': {
                    format_type: Path(file_path).name
                    for format_type, file_path in task.report_files.items()
                    if file_path and Path(file_path).exists()
                },
                'recipients': task.recipients
            }

            headers = api_config.headers or {}
            if api_config.auth_token:
                headers['Authorization'] = f"Bearer {api_config.auth_token}"

            # 发送API请求
            response = requests.request(
                method=api_config.method,
                url=api_config.endpoint,
                json=payload,
                headers=headers,
                timeout=api_config.timeout
            )

            response.raise_for_status()

            self.distribution_stats['api_calls'] += 1

            logger.info(f"API调用成功: {channel} - {task.task_id}")

            return {
                'status': 'success',
                'response_code': response.status_code,
                'endpoint': api_config.endpoint
            }

        except Exception as e:
            logger.error(f"API调用失败: {channel} - {e}")
            raise

    def retry_failed_tasks(self):
        """重试失败的任务"""
        retry_tasks = [task for task in self.task_queue if task.retry_count < task.max_retries]

        for task in retry_tasks:
            try:
                logger.info(f"重试分发任务: {task.task_id} (第{task.retry_count + 1}次)")

                task.status = "processing"
                task.retry_count += 1
                task.updated_at = datetime.now()

                results = self._execute_distribution(task)

                task.status = "completed"
                self.task_queue.remove(task)
                self.task_history.append(task)

                logger.info(f"任务重试成功: {task.task_id}")

            except Exception as e:
                task.status = "failed"
                task.error_message = str(e)

                if task.retry_count >= task.max_retries:
                    logger.error(f"任务重试次数已用完: {task.task_id}")
                    self.task_queue.remove(task)
                    self.task_history.append(task)

    def get_distribution_status(self) -> Dict[str, Any]:
        """获取分发状态"""
        return {
            'stats': self.distribution_stats,
            'pending_tasks': len(self.task_queue),
            'recent_history': [
                {
                    'task_id': task.task_id,
                    'status': task.status,
                    'created_at': task.created_at.isoformat(),
                    'channels': task.distribution_channels
                }
                for task in self.task_history[-10:]  # 最近10条记录
            ],
            'configuration': {
                'email_enabled': self.email_config is not None,
                'ftp_enabled': self.ftp_config is not None,
                'api_channels': list(self.api_configs.keys())
            }
        }

    def schedule_periodic_retry(self, interval_minutes: int = 30):
        """定期重试失败任务"""
        schedule.every(interval_minutes).minutes.do(self.retry_failed_tasks)
        logger.info(f"已设置定期重试，间隔: {interval_minutes} 分钟")

    def clear_history(self, days: int = 30):
        """清理历史记录"""
        cutoff_date = datetime.now() - timedelta(days=days)
        original_count = len(self.task_history)

        self.task_history = [
            task for task in self.task_history
            if task.created_at > cutoff_date
        ]

        cleaned_count = original_count - len(self.task_history)
        logger.info(f"清理了 {cleaned_count} 条历史记录")

        return cleaned_count