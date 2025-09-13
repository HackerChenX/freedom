#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
自动化运维工具集

提供企业级自动化运维解决方案：
1. 自动化部署和回滚
2. 健康检查和自愈
3. 性能调优建议
4. 自动扩容和缩容
5. 数据备份和恢复
6. 故障诊断工具
7. 运维任务调度
8. 配置管理自动化
"""

import os
import sys
import json
import yaml
import time
import shutil
import subprocess
import threading
import schedule
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import psutil
import paramiko
import docker

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor

logger = get_logger(__name__)


class DeploymentStatus(Enum):
    """部署状态"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"
    RECOVERING = "recovering"


class AutoHealAction(Enum):
    """自愈动作"""
    RESTART_SERVICE = "restart_service"
    KILL_PROCESS = "kill_process"
    CLEAR_CACHE = "clear_cache"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    RELOAD_CONFIG = "reload_config"
    CLEAN_LOGS = "clean_logs"


@dataclass
class DeploymentConfig:
    """部署配置"""
    service_name: str
    version: str
    deployment_strategy: str = "rolling"  # rolling, blue_green, canary
    health_check_url: str = ""
    health_check_timeout: int = 30
    rollback_on_failure: bool = True
    backup_before_deploy: bool = True
    restart_after_deploy: bool = True


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    service_name: str
    status: HealthStatus
    timestamp: datetime
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class AutoHealRule:
    """自愈规则"""
    name: str
    condition: Dict[str, Any]
    action: AutoHealAction
    parameters: Dict[str, Any]
    cooldown: int = 300  # 冷却时间（秒）
    max_attempts: int = 3
    enabled: bool = True


@dataclass
class BackupConfig:
    """备份配置"""
    name: str
    source_path: str
    backup_path: str
    retention_days: int = 30
    compress: bool = True
    backup_schedule: str = "0 2 * * *"  # cron格式


class AutomationTask(ABC):
    """自动化任务抽象基类"""

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """执行任务"""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """验证任务配置"""
        pass

    @abstractmethod
    def rollback(self) -> Dict[str, Any]:
        """回滚任务"""
        pass


class DeploymentAutomator(AutomationTask):
    """部署自动化器"""

    def __init__(self, config: DeploymentConfig, ssh_config: Optional[Dict[str, Any]] = None):
        """
        初始化部署自动化器

        Args:
            config: 部署配置
            ssh_config: SSH连接配置
        """
        self.config = config
        self.ssh_config = ssh_config
        self.deployment_history: List[Dict[str, Any]] = []
        self.current_deployment: Optional[Dict[str, Any]] = None

        logger.info(f"部署自动化器初始化完成: {config.service_name}")

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=300.0)
    def execute(self) -> Dict[str, Any]:
        """执行部署"""
        deployment_id = f"deploy_{self.config.service_name}_{int(time.time())}"

        deployment_record = {
            'id': deployment_id,
            'service_name': self.config.service_name,
            'version': self.config.version,
            'strategy': self.config.deployment_strategy,
            'status': DeploymentStatus.PENDING,
            'start_time': datetime.now(),
            'steps': []
        }

        self.current_deployment = deployment_record
        self.deployment_history.append(deployment_record)

        try:
            # 更新状态为部署中
            deployment_record['status'] = DeploymentStatus.DEPLOYING

            # 执行部署前备份
            if self.config.backup_before_deploy:
                backup_result = self._backup_current_version()
                deployment_record['steps'].append({
                    'name': 'backup',
                    'status': 'success' if backup_result else 'failed',
                    'timestamp': datetime.now(),
                    'details': backup_result
                })

            # 执行部署步骤
            if self.config.deployment_strategy == "rolling":
                deploy_result = self._rolling_deployment()
            elif self.config.deployment_strategy == "blue_green":
                deploy_result = self._blue_green_deployment()
            elif self.config.deployment_strategy == "canary":
                deploy_result = self._canary_deployment()
            else:
                raise ValueError(f"不支持的部署策略: {self.config.deployment_strategy}")

            deployment_record['steps'].append({
                'name': 'deployment',
                'status': 'success' if deploy_result['success'] else 'failed',
                'timestamp': datetime.now(),
                'details': deploy_result
            })

            if not deploy_result['success']:
                raise Exception(f"部署失败: {deploy_result.get('error')}")

            # 健康检查
            if self.config.health_check_url:
                health_result = self._health_check()
                deployment_record['steps'].append({
                    'name': 'health_check',
                    'status': 'success' if health_result['healthy'] else 'failed',
                    'timestamp': datetime.now(),
                    'details': health_result
                })

                if not health_result['healthy'] and self.config.rollback_on_failure:
                    logger.warning("健康检查失败，开始回滚")
                    rollback_result = self.rollback()
                    deployment_record['rollback'] = rollback_result
                    deployment_record['status'] = DeploymentStatus.ROLLED_BACK
                    return deployment_record

            # 重启服务
            if self.config.restart_after_deploy:
                restart_result = self._restart_service()
                deployment_record['steps'].append({
                    'name': 'restart',
                    'status': 'success' if restart_result else 'failed',
                    'timestamp': datetime.now(),
                    'details': {'restarted': restart_result}
                })

            # 部署成功
            deployment_record['status'] = DeploymentStatus.DEPLOYED
            deployment_record['end_time'] = datetime.now()

            logger.info(f"部署成功: {deployment_id}")
            return deployment_record

        except Exception as e:
            logger.error(f"部署失败: {e}")
            deployment_record['status'] = DeploymentStatus.FAILED
            deployment_record['error'] = str(e)
            deployment_record['end_time'] = datetime.now()

            # 自动回滚
            if self.config.rollback_on_failure:
                logger.info("开始自动回滚")
                rollback_result = self.rollback()
                deployment_record['rollback'] = rollback_result

            return deployment_record

    def validate(self) -> bool:
        """验证部署配置"""
        try:
            if not self.config.service_name or not self.config.version:
                return False

            if self.config.deployment_strategy not in ["rolling", "blue_green", "canary"]:
                return False

            return True

        except Exception as e:
            logger.error(f"验证部署配置失败: {e}")
            return False

    @exception_handler(reraise=True)
    def rollback(self) -> Dict[str, Any]:
        """回滚部署"""
        if not self.deployment_history or len(self.deployment_history) < 2:
            return {'success': False, 'error': '没有可回滚的版本'}

        try:
            # 获取前一个成功的部署
            previous_deployment = None
            for deployment in reversed(self.deployment_history[:-1]):
                if deployment['status'] == DeploymentStatus.DEPLOYED:
                    previous_deployment = deployment
                    break

            if not previous_deployment:
                return {'success': False, 'error': '找不到可回滚的版本'}

            logger.info(f"回滚到版本: {previous_deployment['version']}")

            # 执行回滚操作
            rollback_steps = []

            # 停止当前服务
            stop_result = self._stop_service()
            rollback_steps.append({
                'name': 'stop_service',
                'status': 'success' if stop_result else 'failed',
                'timestamp': datetime.now()
            })

            # 恢复前一个版本
            restore_result = self._restore_version(previous_deployment['version'])
            rollback_steps.append({
                'name': 'restore_version',
                'status': 'success' if restore_result else 'failed',
                'timestamp': datetime.now()
            })

            # 启动服务
            start_result = self._start_service()
            rollback_steps.append({
                'name': 'start_service',
                'status': 'success' if start_result else 'failed',
                'timestamp': datetime.now()
            })

            # 验证回滚
            if self.config.health_check_url:
                health_result = self._health_check()
                rollback_steps.append({
                    'name': 'health_check',
                    'status': 'success' if health_result['healthy'] else 'failed',
                    'timestamp': datetime.now()
                })

            # 更新当前部署状态
            if self.current_deployment:
                self.current_deployment['status'] = DeploymentStatus.ROLLED_BACK

            return {
                'success': True,
                'previous_version': previous_deployment['version'],
                'steps': rollback_steps,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"回滚失败: {e}")
            return {'success': False, 'error': str(e)}

    def _rolling_deployment(self) -> Dict[str, Any]:
        """滚动部署"""
        try:
            # 模拟滚动部署过程
            logger.info(f"开始滚动部署 {self.config.service_name} v{self.config.version}")

            # 逐步替换实例
            steps = [
                "下载新版本",
                "更新配置文件",
                "重启第一批实例",
                "验证第一批实例",
                "重启第二批实例",
                "验证第二批实例",
                "更新负载均衡配置"
            ]

            for step in steps:
                logger.info(f"执行步骤: {step}")
                time.sleep(1)  # 模拟执行时间

            return {
                'success': True,
                'strategy': 'rolling',
                'steps_completed': steps,
                'timestamp': datetime.now()
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'strategy': 'rolling'
            }

    def _blue_green_deployment(self) -> Dict[str, Any]:
        """蓝绿部署"""
        try:
            logger.info(f"开始蓝绿部署 {self.config.service_name} v{self.config.version}")

            steps = [
                "创建绿色环境",
                "部署新版本到绿色环境",
                "验证绿色环境",
                "切换流量到绿色环境",
                "验证流量切换",
                "清理蓝色环境"
            ]

            for step in steps:
                logger.info(f"执行步骤: {step}")
                time.sleep(1)  # 模拟执行时间

            return {
                'success': True,
                'strategy': 'blue_green',
                'steps_completed': steps,
                'timestamp': datetime.now()
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'strategy': 'blue_green'
            }

    def _canary_deployment(self) -> Dict[str, Any]:
        """金丝雀部署"""
        try:
            logger.info(f"开始金丝雀部署 {self.config.service_name} v{self.config.version}")

            steps = [
                "部署金丝雀实例",
                "配置流量分割(5%)",
                "监控金丝雀指标",
                "增加流量比例(20%)",
                "继续监控指标",
                "增加流量比例(50%)",
                "最终切换全部流量"
            ]

            for step in steps:
                logger.info(f"执行步骤: {step}")
                time.sleep(1)  # 模拟执行时间

            return {
                'success': True,
                'strategy': 'canary',
                'steps_completed': steps,
                'timestamp': datetime.now()
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'strategy': 'canary'
            }

    def _backup_current_version(self) -> Dict[str, Any]:
        """备份当前版本"""
        try:
            backup_dir = f"/backup/{self.config.service_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(backup_dir, exist_ok=True)

            # 模拟备份过程
            logger.info(f"备份当前版本到: {backup_dir}")
            return {
                'backup_path': backup_dir,
                'size': 1024 * 1024,  # 模拟备份大小
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"备份失败: {e}")
            return {}

    def _health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            import requests

            start_time = time.time()
            response = requests.get(
                self.config.health_check_url,
                timeout=self.config.health_check_timeout
            )
            response_time = time.time() - start_time

            healthy = response.status_code == 200

            return {
                'healthy': healthy,
                'status_code': response.status_code,
                'response_time': response_time,
                'timestamp': datetime.now()
            }

        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now()
            }

    def _restart_service(self) -> bool:
        """重启服务"""
        try:
            # 模拟重启服务
            logger.info(f"重启服务: {self.config.service_name}")
            return True
        except Exception as e:
            logger.error(f"重启服务失败: {e}")
            return False

    def _stop_service(self) -> bool:
        """停止服务"""
        try:
            logger.info(f"停止服务: {self.config.service_name}")
            return True
        except Exception:
            return False

    def _start_service(self) -> bool:
        """启动服务"""
        try:
            logger.info(f"启动服务: {self.config.service_name}")
            return True
        except Exception:
            return False

    def _restore_version(self, version: str) -> bool:
        """恢复版本"""
        try:
            logger.info(f"恢复到版本: {version}")
            return True
        except Exception:
            return False


class HealthMonitor:
    """健康监控器"""

    def __init__(self, check_interval: int = 60):
        """
        初始化健康监控器

        Args:
            check_interval: 检查间隔（秒）
        """
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.health_history: List[HealthCheckResult] = []

        logger.info("健康监控器初始化完成")

    def add_health_check(self, service_name: str, check_function: Callable[[], HealthCheckResult]):
        """添加健康检查"""
        self.health_checks[service_name] = check_function
        logger.info(f"添加健康检查: {service_name}")

    @exception_handler(reraise=True)
    def start_monitoring(self) -> Dict[str, Any]:
        """启动健康监控"""
        if self.monitoring_active:
            return {'status': 'already_running'}

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("健康监控已启动")
        return {
            'status': 'started',
            'check_interval': self.check_interval,
            'services': list(self.health_checks.keys()),
            'start_time': datetime.now()
        }

    def stop_monitoring(self):
        """停止健康监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("健康监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                for service_name, check_function in self.health_checks.items():
                    try:
                        result = check_function()
                        self.health_history.append(result)

                        # 保留最近1000条记录
                        if len(self.health_history) > 1000:
                            self.health_history = self.health_history[-1000:]

                        # 记录健康状态变化
                        if result.status != HealthStatus.HEALTHY:
                            logger.warning(f"服务健康状态异常: {service_name} - {result.status.value}")

                    except Exception as e:
                        logger.error(f"健康检查失败 {service_name}: {e}")

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(5)

    def get_health_status(self, service_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取健康状态"""
        if service_name:
            # 获取特定服务的最新状态
            for result in reversed(self.health_history):
                if result.service_name == service_name:
                    return [asdict(result)]
            return []
        else:
            # 获取所有服务的最新状态
            latest_results = {}
            for result in reversed(self.health_history):
                if result.service_name not in latest_results:
                    latest_results[result.service_name] = result

            return [asdict(result) for result in latest_results.values()]


class AutoHealer:
    """自动自愈器"""

    def __init__(self):
        self.heal_rules: List[AutoHealRule] = []
        self.action_handlers: Dict[AutoHealAction, Callable] = {}
        self.heal_history: List[Dict[str, Any]] = []
        self.cooldown_tracker: Dict[str, datetime] = {}

        # 注册默认动作处理器
        self._register_default_handlers()

        logger.info("自动自愈器初始化完成")

    def _register_default_handlers(self):
        """注册默认动作处理器"""
        self.action_handlers[AutoHealAction.RESTART_SERVICE] = self._restart_service
        self.action_handlers[AutoHealAction.KILL_PROCESS] = self._kill_process
        self.action_handlers[AutoHealAction.CLEAR_CACHE] = self._clear_cache
        self.action_handlers[AutoHealAction.SCALE_UP] = self._scale_up
        self.action_handlers[AutoHealAction.SCALE_DOWN] = self._scale_down
        self.action_handlers[AutoHealAction.RELOAD_CONFIG] = self._reload_config
        self.action_handlers[AutoHealAction.CLEAN_LOGS] = self._clean_logs

    def add_heal_rule(self, rule: AutoHealRule):
        """添加自愈规则"""
        self.heal_rules.append(rule)
        logger.info(f"添加自愈规则: {rule.name}")

    def register_action_handler(self, action: AutoHealAction, handler: Callable):
        """注册自定义动作处理器"""
        self.action_handlers[action] = handler

    @exception_handler(reraise=True)
    def process_health_result(self, health_result: HealthCheckResult) -> Optional[Dict[str, Any]]:
        """处理健康检查结果"""
        if health_result.status == HealthStatus.HEALTHY:
            return None

        # 查找匹配的自愈规则
        for rule in self.heal_rules:
            if not rule.enabled:
                continue

            if self._matches_condition(health_result, rule.condition):
                # 检查冷却时间
                cooldown_key = f"{rule.name}_{health_result.service_name}"
                if self._is_in_cooldown(cooldown_key, rule.cooldown):
                    continue

                # 执行自愈动作
                heal_result = self._execute_heal_action(health_result, rule)

                # 更新冷却时间
                self.cooldown_tracker[cooldown_key] = datetime.now()

                return heal_result

        return None

    def _matches_condition(self, health_result: HealthCheckResult, condition: Dict[str, Any]) -> bool:
        """检查是否匹配条件"""
        # 检查服务状态
        if 'status' in condition:
            if health_result.status.value not in condition['status']:
                return False

        # 检查服务名称
        if 'service_name' in condition:
            if health_result.service_name not in condition['service_name']:
                return False

        # 检查响应时间
        if 'response_time_threshold' in condition and health_result.response_time:
            if health_result.response_time < condition['response_time_threshold']:
                return False

        # 检查错误消息关键词
        if 'error_keywords' in condition and health_result.error_message:
            keywords = condition['error_keywords']
            if not any(keyword in health_result.error_message for keyword in keywords):
                return False

        return True

    def _is_in_cooldown(self, cooldown_key: str, cooldown_seconds: int) -> bool:
        """检查是否在冷却期内"""
        if cooldown_key not in self.cooldown_tracker:
            return False

        last_action_time = self.cooldown_tracker[cooldown_key]
        elapsed = (datetime.now() - last_action_time).total_seconds()
        return elapsed < cooldown_seconds

    def _execute_heal_action(self, health_result: HealthCheckResult, rule: AutoHealRule) -> Dict[str, Any]:
        """执行自愈动作"""
        heal_record = {
            'id': f"heal_{int(time.time())}",
            'service_name': health_result.service_name,
            'rule_name': rule.name,
            'action': rule.action.value,
            'timestamp': datetime.now(),
            'success': False,
            'details': {}
        }

        try:
            if rule.action in self.action_handlers:
                handler = self.action_handlers[rule.action]
                result = handler(health_result, rule.parameters)
                heal_record['success'] = result.get('success', False)
                heal_record['details'] = result
            else:
                heal_record['details'] = {'error': f'未找到动作处理器: {rule.action.value}'}

            self.heal_history.append(heal_record)

            if heal_record['success']:
                logger.info(f"自愈成功: {rule.name} - {rule.action.value}")
            else:
                logger.error(f"自愈失败: {rule.name} - {rule.action.value}")

            return heal_record

        except Exception as e:
            heal_record['details'] = {'error': str(e)}
            self.heal_history.append(heal_record)
            logger.error(f"执行自愈动作失败: {e}")
            return heal_record

    # 默认动作处理器实现
    def _restart_service(self, health_result: HealthCheckResult, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """重启服务"""
        try:
            service_name = parameters.get('service_name', health_result.service_name)

            # 使用systemctl重启服务
            result = subprocess.run(
                ['sudo', 'systemctl', 'restart', service_name],
                capture_output=True,
                text=True,
                timeout=60
            )

            success = result.returncode == 0
            return {
                'success': success,
                'action': 'restart_service',
                'service_name': service_name,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _kill_process(self, health_result: HealthCheckResult, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """终止进程"""
        try:
            process_name = parameters.get('process_name')
            pid = parameters.get('pid')

            if pid:
                # 终止指定PID的进程
                os.kill(int(pid), 9)
                return {
                    'success': True,
                    'action': 'kill_process',
                    'pid': pid
                }
            elif process_name:
                # 终止指定名称的进程
                killed_pids = []
                for proc in psutil.process_iter(['pid', 'name']):
                    if proc.info['name'] == process_name:
                        proc.kill()
                        killed_pids.append(proc.info['pid'])

                return {
                    'success': len(killed_pids) > 0,
                    'action': 'kill_process',
                    'process_name': process_name,
                    'killed_pids': killed_pids
                }

            return {'success': False, 'error': '未指定进程名称或PID'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _clear_cache(self, health_result: HealthCheckResult, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """清理缓存"""
        try:
            cache_paths = parameters.get('cache_paths', ['/tmp/cache', '/var/cache/app'])
            cleared_paths = []

            for cache_path in cache_paths:
                if os.path.exists(cache_path):
                    if os.path.isdir(cache_path):
                        shutil.rmtree(cache_path)
                        os.makedirs(cache_path)
                    else:
                        os.remove(cache_path)
                    cleared_paths.append(cache_path)

            return {
                'success': True,
                'action': 'clear_cache',
                'cleared_paths': cleared_paths
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _scale_up(self, health_result: HealthCheckResult, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """扩容"""
        try:
            # 这里应该集成容器编排平台API（如Kubernetes、Docker Swarm）
            service_name = parameters.get('service_name', health_result.service_name)
            replicas = parameters.get('replicas', 1)

            logger.info(f"模拟扩容: {service_name} +{replicas} 实例")

            return {
                'success': True,
                'action': 'scale_up',
                'service_name': service_name,
                'added_replicas': replicas
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _scale_down(self, health_result: HealthCheckResult, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """缩容"""
        try:
            service_name = parameters.get('service_name', health_result.service_name)
            replicas = parameters.get('replicas', 1)

            logger.info(f"模拟缩容: {service_name} -{replicas} 实例")

            return {
                'success': True,
                'action': 'scale_down',
                'service_name': service_name,
                'removed_replicas': replicas
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _reload_config(self, health_result: HealthCheckResult, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """重载配置"""
        try:
            service_name = parameters.get('service_name', health_result.service_name)

            # 发送SIGHUP信号重载配置
            result = subprocess.run(
                ['sudo', 'systemctl', 'reload', service_name],
                capture_output=True,
                text=True,
                timeout=30
            )

            success = result.returncode == 0
            return {
                'success': success,
                'action': 'reload_config',
                'service_name': service_name,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _clean_logs(self, health_result: HealthCheckResult, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """清理日志"""
        try:
            log_paths = parameters.get('log_paths', ['/var/log/app'])
            max_size_mb = parameters.get('max_size_mb', 100)
            cleaned_files = []

            for log_path in log_paths:
                if os.path.exists(log_path):
                    if os.path.isfile(log_path):
                        # 单个日志文件
                        size_mb = os.path.getsize(log_path) / (1024 * 1024)
                        if size_mb > max_size_mb:
                            open(log_path, 'w').close()  # 清空文件
                            cleaned_files.append(log_path)
                    elif os.path.isdir(log_path):
                        # 日志目录
                        for root, dirs, files in os.walk(log_path):
                            for file in files:
                                if file.endswith('.log'):
                                    file_path = os.path.join(root, file)
                                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                                    if size_mb > max_size_mb:
                                        open(file_path, 'w').close()
                                        cleaned_files.append(file_path)

            return {
                'success': True,
                'action': 'clean_logs',
                'cleaned_files': cleaned_files,
                'max_size_mb': max_size_mb
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_heal_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取自愈历史"""
        return self.heal_history[-limit:]

    def get_heal_statistics(self) -> Dict[str, Any]:
        """获取自愈统计"""
        if not self.heal_history:
            return {}

        success_count = sum(1 for record in self.heal_history if record['success'])
        failure_count = len(self.heal_history) - success_count

        # 按动作类型统计
        action_stats = {}
        for record in self.heal_history:
            action = record['action']
            if action not in action_stats:
                action_stats[action] = {'success': 0, 'failure': 0}

            if record['success']:
                action_stats[action]['success'] += 1
            else:
                action_stats[action]['failure'] += 1

        # 按服务统计
        service_stats = {}
        for record in self.heal_history:
            service = record['service_name']
            if service not in service_stats:
                service_stats[service] = {'success': 0, 'failure': 0}

            if record['success']:
                service_stats[service]['success'] += 1
            else:
                service_stats[service]['failure'] += 1

        return {
            'total_heals': len(self.heal_history),
            'success_count': success_count,
            'failure_count': failure_count,
            'success_rate': (success_count / len(self.heal_history)) * 100,
            'action_statistics': action_stats,
            'service_statistics': service_stats,
            'active_rules': len([rule for rule in self.heal_rules if rule.enabled]),
            'timestamp': datetime.now()
        }


class BackupManager:
    """备份管理器"""

    def __init__(self):
        self.backup_configs: List[BackupConfig] = []
        self.backup_history: List[Dict[str, Any]] = []
        self.scheduler_active = False
        self.scheduler_thread: Optional[threading.Thread] = None

        logger.info("备份管理器初始化完成")

    def add_backup_config(self, config: BackupConfig):
        """添加备份配置"""
        self.backup_configs.append(config)
        logger.info(f"添加备份配置: {config.name}")

    @exception_handler(reraise=True)
    def start_scheduler(self):
        """启动备份调度器"""
        if self.scheduler_active:
            return

        self.scheduler_active = True

        # 注册所有备份任务
        for config in self.backup_configs:
            schedule.every().day.at("02:00").do(self._scheduled_backup, config)

        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        logger.info("备份调度器已启动")

    def stop_scheduler(self):
        """停止备份调度器"""
        self.scheduler_active = False
        schedule.clear()

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)

        logger.info("备份调度器已停止")

    def _scheduler_loop(self):
        """调度循环"""
        while self.scheduler_active:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次

    def _scheduled_backup(self, config: BackupConfig):
        """执行调度备份"""
        try:
            result = self.create_backup(config)
            if result['success']:
                logger.info(f"调度备份成功: {config.name}")
            else:
                logger.error(f"调度备份失败: {config.name} - {result.get('error')}")
        except Exception as e:
            logger.error(f"调度备份异常: {config.name} - {e}")

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=300.0)
    def create_backup(self, config: BackupConfig) -> Dict[str, Any]:
        """创建备份"""
        backup_id = f"backup_{config.name}_{int(time.time())}"

        backup_record = {
            'id': backup_id,
            'name': config.name,
            'source_path': config.source_path,
            'backup_path': config.backup_path,
            'timestamp': datetime.now(),
            'success': False,
            'size_bytes': 0,
            'duration_seconds': 0
        }

        start_time = time.time()

        try:
            # 确保备份目录存在
            backup_dir = Path(config.backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)

            # 创建时间戳子目录
            timestamp_dir = backup_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
            timestamp_dir.mkdir(exist_ok=True)

            # 执行备份
            if config.compress:
                backup_file = timestamp_dir / f"{config.name}.tar.gz"
                self._create_compressed_backup(config.source_path, backup_file)
            else:
                self._create_direct_backup(config.source_path, timestamp_dir)

            # 计算备份大小
            backup_record['size_bytes'] = self._calculate_backup_size(timestamp_dir)
            backup_record['backup_full_path'] = str(timestamp_dir)
            backup_record['success'] = True

            # 清理过期备份
            self._cleanup_old_backups(config)

            logger.info(f"备份创建成功: {backup_id}")

        except Exception as e:
            backup_record['error'] = str(e)
            logger.error(f"创建备份失败: {e}")

        finally:
            backup_record['duration_seconds'] = time.time() - start_time
            self.backup_history.append(backup_record)

        return backup_record

    def _create_compressed_backup(self, source_path: str, backup_file: Path):
        """创建压缩备份"""
        import tarfile

        with tarfile.open(backup_file, 'w:gz') as tar:
            tar.add(source_path, arcname=os.path.basename(source_path))

    def _create_direct_backup(self, source_path: str, backup_dir: Path):
        """创建直接备份"""
        if os.path.isfile(source_path):
            shutil.copy2(source_path, backup_dir)
        elif os.path.isdir(source_path):
            dest_path = backup_dir / os.path.basename(source_path)
            shutil.copytree(source_path, dest_path)

    def _calculate_backup_size(self, backup_path: Path) -> int:
        """计算备份大小"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(backup_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        return total_size

    def _cleanup_old_backups(self, config: BackupConfig):
        """清理过期备份"""
        backup_dir = Path(config.backup_path)
        if not backup_dir.exists():
            return

        cutoff_date = datetime.now() - timedelta(days=config.retention_days)

        for item in backup_dir.iterdir():
            if item.is_dir():
                try:
                    # 从目录名解析时间戳
                    dir_timestamp = datetime.strptime(item.name, '%Y%m%d_%H%M%S')
                    if dir_timestamp < cutoff_date:
                        shutil.rmtree(item)
                        logger.info(f"清理过期备份: {item}")
                except ValueError:
                    # 目录名不符合时间戳格式，跳过
                    continue

    @exception_handler(reraise=True)
    def restore_backup(self, backup_id: str, restore_path: str) -> Dict[str, Any]:
        """恢复备份"""
        # 查找备份记录
        backup_record = None
        for record in self.backup_history:
            if record['id'] == backup_id:
                backup_record = record
                break

        if not backup_record or not backup_record['success']:
            return {'success': False, 'error': '找不到有效的备份记录'}

        try:
            source_path = backup_record['backup_full_path']

            # 确保恢复目录存在
            os.makedirs(os.path.dirname(restore_path), exist_ok=True)

            # 执行恢复
            if os.path.isdir(source_path):
                if os.path.exists(restore_path):
                    shutil.rmtree(restore_path)
                shutil.copytree(source_path, restore_path)
            else:
                shutil.copy2(source_path, restore_path)

            logger.info(f"备份恢复成功: {backup_id} -> {restore_path}")

            return {
                'success': True,
                'backup_id': backup_id,
                'restore_path': restore_path,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"备份恢复失败: {e}")
            return {'success': False, 'error': str(e)}

    def get_backup_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取备份历史"""
        return self.backup_history[-limit:]

    def get_backup_statistics(self) -> Dict[str, Any]:
        """获取备份统计"""
        if not self.backup_history:
            return {}

        success_count = sum(1 for record in self.backup_history if record['success'])
        total_size = sum(record['size_bytes'] for record in self.backup_history if record['success'])

        return {
            'total_backups': len(self.backup_history),
            'successful_backups': success_count,
            'failed_backups': len(self.backup_history) - success_count,
            'success_rate': (success_count / len(self.backup_history)) * 100 if self.backup_history else 0,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'active_configs': len(self.backup_configs),
            'timestamp': datetime.now()
        }


class AutomatedOpsToolkit:
    """
    自动化运维工具集

    整合所有自动化运维功能的核心管理器
    """

    def __init__(self):
        self.deployment_automator: Optional[DeploymentAutomator] = None
        self.health_monitor = HealthMonitor()
        self.auto_healer = AutoHealer()
        self.backup_manager = BackupManager()

        # 集成智能告警系统
        try:
            from monitoring.intelligent_multi_channel_alert_system import get_intelligent_alert_manager
            self.alert_manager = get_intelligent_alert_manager()
        except ImportError:
            logger.warning("无法导入智能告警系统")
            self.alert_manager = None

        logger.info("自动化运维工具集初始化完成")

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=10.0)
    def initialize_automated_ops(self, config_path: str = "config/automated_ops.yaml") -> Dict[str, Any]:
        """初始化自动化运维"""
        try:
            # 加载配置
            config = self._load_config(config_path)

            # 初始化健康监控
            self._setup_health_monitoring(config.get('health_monitoring', {}))

            # 初始化自动自愈
            self._setup_auto_healing(config.get('auto_healing', {}))

            # 初始化备份管理
            self._setup_backup_management(config.get('backup_management', {}))

            # 启动各个组件
            health_result = self.health_monitor.start_monitoring()
            self.auto_healer  # 自愈器不需要启动，它响应健康检查结果
            self.backup_manager.start_scheduler()

            logger.info("自动化运维系统初始化完成")

            return {
                'status': 'initialized',
                'health_monitoring': health_result,
                'auto_healing_rules': len(self.auto_healer.heal_rules),
                'backup_configs': len(self.backup_manager.backup_configs),
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"初始化自动化运维失败: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        else:
            # 创建默认配置
            default_config = {
                'health_monitoring': {
                    'check_interval': 60,
                    'services': [
                        {
                            'name': 'stock_analysis_api',
                            'url': 'http://localhost:8000/health',
                            'timeout': 10
                        }
                    ]
                },
                'auto_healing': {
                    'rules': [
                        {
                            'name': 'restart_on_down',
                            'condition': {
                                'status': ['down', 'critical']
                            },
                            'action': 'restart_service',
                            'parameters': {},
                            'cooldown': 300,
                            'max_attempts': 3
                        }
                    ]
                },
                'backup_management': {
                    'configs': [
                        {
                            'name': 'app_config',
                            'source_path': '/app/config',
                            'backup_path': '/backup/config',
                            'retention_days': 30,
                            'compress': True
                        }
                    ]
                }
            }

            # 确保目录存在
            config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

            return default_config

    def _setup_health_monitoring(self, config: Dict[str, Any]):
        """设置健康监控"""
        services = config.get('services', [])

        for service_config in services:
            service_name = service_config['name']

            def create_health_check(name, url, timeout):
                def health_check_function():
                    try:
                        import requests
                        start_time = time.time()

                        response = requests.get(url, timeout=timeout)
                        response_time = time.time() - start_time

                        if response.status_code == 200:
                            status = HealthStatus.HEALTHY
                        elif response.status_code >= 500:
                            status = HealthStatus.CRITICAL
                        else:
                            status = HealthStatus.WARNING

                        return HealthCheckResult(
                            service_name=name,
                            status=status,
                            timestamp=datetime.now(),
                            response_time=response_time
                        )

                    except requests.exceptions.Timeout:
                        return HealthCheckResult(
                            service_name=name,
                            status=HealthStatus.CRITICAL,
                            timestamp=datetime.now(),
                            error_message="请求超时"
                        )

                    except Exception as e:
                        return HealthCheckResult(
                            service_name=name,
                            status=HealthStatus.DOWN,
                            timestamp=datetime.now(),
                            error_message=str(e)
                        )

                return health_check_function

            check_function = create_health_check(
                service_name,
                service_config['url'],
                service_config.get('timeout', 10)
            )

            self.health_monitor.add_health_check(service_name, check_function)

    def _setup_auto_healing(self, config: Dict[str, Any]):
        """设置自动自愈"""
        rules = config.get('rules', [])

        for rule_config in rules:
            rule = AutoHealRule(
                name=rule_config['name'],
                condition=rule_config['condition'],
                action=AutoHealAction(rule_config['action']),
                parameters=rule_config.get('parameters', {}),
                cooldown=rule_config.get('cooldown', 300),
                max_attempts=rule_config.get('max_attempts', 3)
            )

            self.auto_healer.add_heal_rule(rule)

    def _setup_backup_management(self, config: Dict[str, Any]):
        """设置备份管理"""
        configs = config.get('configs', [])

        for backup_config_data in configs:
            backup_config = BackupConfig(
                name=backup_config_data['name'],
                source_path=backup_config_data['source_path'],
                backup_path=backup_config_data['backup_path'],
                retention_days=backup_config_data.get('retention_days', 30),
                compress=backup_config_data.get('compress', True)
            )

            self.backup_manager.add_backup_config(backup_config)

    @exception_handler(reraise=True)
    def create_deployment(self, service_name: str, version: str,
                         strategy: str = "rolling") -> Dict[str, Any]:
        """创建部署任务"""
        deployment_config = DeploymentConfig(
            service_name=service_name,
            version=version,
            deployment_strategy=strategy,
            health_check_url=f"http://localhost:8000/health",
            rollback_on_failure=True
        )

        self.deployment_automator = DeploymentAutomator(deployment_config)
        return self.deployment_automator.execute()

    @exception_handler(reraise=True)
    def get_system_overview(self) -> Dict[str, Any]:
        """获取系统概览"""
        # 获取健康状态
        health_status = self.health_monitor.get_health_status()

        # 获取自愈统计
        heal_stats = self.auto_healer.get_heal_statistics()

        # 获取备份统计
        backup_stats = self.backup_manager.get_backup_statistics()

        # 处理健康检查结果并触发自愈
        auto_heal_actions = []
        for health_result_data in health_status:
            health_result = HealthCheckResult(**health_result_data)
            if health_result.status != HealthStatus.HEALTHY:
                heal_result = self.auto_healer.process_health_result(health_result)
                if heal_result:
                    auto_heal_actions.append(heal_result)

        return {
            'health_monitoring': {
                'active': self.health_monitor.monitoring_active,
                'services_count': len(self.health_monitor.health_checks),
                'current_status': health_status
            },
            'auto_healing': {
                'rules_count': len(self.auto_healer.heal_rules),
                'statistics': heal_stats,
                'recent_actions': auto_heal_actions
            },
            'backup_management': {
                'configs_count': len(self.backup_manager.backup_configs),
                'scheduler_active': self.backup_manager.scheduler_active,
                'statistics': backup_stats
            },
            'deployment': {
                'active': self.deployment_automator is not None,
                'current_deployment': self.deployment_automator.current_deployment if self.deployment_automator else None
            },
            'timestamp': datetime.now()
        }

    def stop_all_services(self):
        """停止所有服务"""
        self.health_monitor.stop_monitoring()
        self.backup_manager.stop_scheduler()
        logger.info("所有自动化运维服务已停止")


# 全局工具集实例
_automated_ops_toolkit = None


def get_automated_ops_toolkit() -> AutomatedOpsToolkit:
    """
    获取自动化运维工具集实例（单例模式）

    Returns:
        AutomatedOpsToolkit: 自动化运维工具集实例
    """
    global _automated_ops_toolkit

    if _automated_ops_toolkit is None:
        _automated_ops_toolkit = AutomatedOpsToolkit()

    return _automated_ops_toolkit


def create_automated_ops_toolkit() -> AutomatedOpsToolkit:
    """
    创建新的自动化运维工具集实例

    Returns:
        AutomatedOpsToolkit: 新的自动化运维工具集实例
    """
    return AutomatedOpsToolkit()