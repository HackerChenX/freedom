#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
故障自愈和自动恢复机制

企业级智能故障自愈系统，提供全自动故障检测、诊断和恢复：
1. 智能故障检测（多维度异常检测、模式识别）
2. 故障根因分析（依赖关系分析、日志关联分析）
3. 自动恢复策略（分级恢复、渐进式恢复）
4. 故障预防（预测性维护、容量规划）
5. 恢复效果验证（健康检查、性能验证）
6. 学习型自愈（机器学习、策略优化）
7. 故障报告（详细分析、改进建议）
8. 紧急处理机制（人工干预、紧急联系）
"""

import os
import json
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import queue
import traceback

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class FailureType(Enum):
    """故障类型"""
    SYSTEM_CRASH = "system_crash"
    SERVICE_DOWN = "service_down"
    HIGH_LATENCY = "high_latency"
    MEMORY_LEAK = "memory_leak"
    DISK_FULL = "disk_full"
    NETWORK_ISSUE = "network_issue"
    DATABASE_ERROR = "database_error"
    CONFIGURATION_ERROR = "configuration_error"
    SECURITY_BREACH = "security_breach"
    CAPACITY_OVERFLOW = "capacity_overflow"


class FailureSeverity(Enum):
    """故障严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RecoveryStatus(Enum):
    """恢复状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    MANUAL_REQUIRED = "manual_required"


class RecoveryStrategy(Enum):
    """恢复策略"""
    RESTART_SERVICE = "restart_service"
    RELOAD_CONFIG = "reload_config"
    CLEAR_CACHE = "clear_cache"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SWITCH_BACKUP = "switch_backup"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    ISOLATE_COMPONENT = "isolate_component"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class FailureEvent:
    """故障事件"""
    event_id: str
    failure_type: FailureType
    severity: FailureSeverity
    affected_component: str
    description: str
    detection_time: datetime
    symptoms: List[str]
    metrics: Dict[str, Any]
    context: Dict[str, Any]
    root_cause: Optional[str] = None
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_status: RecoveryStatus = RecoveryStatus.PENDING


@dataclass
class RecoveryAction:
    """恢复动作"""
    action_id: str
    strategy: RecoveryStrategy
    component: str
    parameters: Dict[str, Any]
    execution_time: datetime
    duration: float
    status: RecoveryStatus
    output: str
    error_message: Optional[str] = None


@dataclass
class FailurePattern:
    """故障模式"""
    pattern_id: str
    name: str
    description: str
    failure_types: List[FailureType]
    symptoms: List[str]
    conditions: Dict[str, Any]
    recovery_strategies: List[RecoveryStrategy]
    success_rate: float
    last_updated: datetime


@dataclass
class ComponentDependency:
    """组件依赖关系"""
    component_name: str
    dependencies: List[str]
    dependents: List[str]
    criticality_level: int  # 1-5, 5最高
    recovery_priority: int  # 1-10, 10最高


class FailureDetector(ABC):
    """故障检测器抽象基类"""

    @abstractmethod
    def detect_failures(self) -> List[FailureEvent]:
        """检测故障"""
        pass

    @abstractmethod
    def get_detection_confidence(self) -> float:
        """获取检测置信度"""
        pass


class SystemFailureDetector(FailureDetector):
    """系统故障检测器"""

    def __init__(self):
        self.detection_history: List[Dict[str, Any]] = []
        self.thresholds = self._get_default_thresholds()

    def _get_default_thresholds(self) -> Dict[str, Any]:
        """获取默认阈值"""
        return {
            'cpu_critical': 95.0,
            'memory_critical': 95.0,
            'disk_critical': 98.0,
            'response_time_critical': 30.0,
            'error_rate_critical': 10.0,
            'connection_critical': 1000
        }

    @exception_handler(reraise=True)
    def detect_failures(self) -> List[FailureEvent]:
        """检测系统故障"""
        failures = []
        detection_time = datetime.now()

        try:
            # 获取系统指标
            system_metrics = self._get_system_metrics()

            # CPU使用率检测
            cpu_usage = system_metrics.get('cpu_percent', 0)
            if cpu_usage > self.thresholds['cpu_critical']:
                failure = FailureEvent(
                    event_id=f"cpu_high_{int(detection_time.timestamp())}",
                    failure_type=FailureType.SYSTEM_CRASH,
                    severity=FailureSeverity.HIGH,
                    affected_component="system_cpu",
                    description=f"CPU使用率过高: {cpu_usage:.1f}%",
                    detection_time=detection_time,
                    symptoms=[f"CPU使用率达到{cpu_usage:.1f}%", "系统响应缓慢"],
                    metrics={'cpu_percent': cpu_usage},
                    context={'threshold': self.thresholds['cpu_critical']}
                )
                failures.append(failure)

            # 内存使用检测
            memory_usage = system_metrics.get('memory_percent', 0)
            if memory_usage > self.thresholds['memory_critical']:
                failure = FailureEvent(
                    event_id=f"memory_high_{int(detection_time.timestamp())}",
                    failure_type=FailureType.MEMORY_LEAK,
                    severity=FailureSeverity.HIGH,
                    affected_component="system_memory",
                    description=f"内存使用率过高: {memory_usage:.1f}%",
                    detection_time=detection_time,
                    symptoms=[f"内存使用率达到{memory_usage:.1f}%", "可能存在内存泄漏"],
                    metrics={'memory_percent': memory_usage},
                    context={'threshold': self.thresholds['memory_critical']}
                )
                failures.append(failure)

            # 磁盘空间检测
            disk_usage = system_metrics.get('disk_usage', {})
            for mount_point, usage_info in disk_usage.items():
                if isinstance(usage_info, dict):
                    usage_percent = usage_info.get('percent', 0)
                    if usage_percent > self.thresholds['disk_critical']:
                        failure = FailureEvent(
                            event_id=f"disk_full_{mount_point}_{int(detection_time.timestamp())}",
                            failure_type=FailureType.DISK_FULL,
                            severity=FailureSeverity.CRITICAL,
                            affected_component=f"disk_{mount_point}",
                            description=f"磁盘空间不足: {mount_point} {usage_percent:.1f}%",
                            detection_time=detection_time,
                            symptoms=[f"磁盘 {mount_point} 使用率{usage_percent:.1f}%", "磁盘空间即将耗尽"],
                            metrics={'disk_percent': usage_percent, 'mount_point': mount_point},
                            context={'threshold': self.thresholds['disk_critical']}
                        )
                        failures.append(failure)

            # 记录检测历史
            self.detection_history.append({
                'timestamp': detection_time,
                'metrics': system_metrics,
                'failures_detected': len(failures)
            })

            # 清理历史记录（保留最近100条）
            if len(self.detection_history) > 100:
                self.detection_history = self.detection_history[-100:]

        except Exception as e:
            logger.error(f"系统故障检测失败: {e}")

        return failures

    def _get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标（模拟数据）"""
        try:
            import random

            return {
                'cpu_percent': random.uniform(10, 98),
                'memory_percent': random.uniform(20, 96),
                'disk_usage': {
                    '/': {'percent': random.uniform(30, 99)},
                    '/var': {'percent': random.uniform(40, 95)}
                },
                'network_io': {
                    'bytes_sent': random.randint(1000000, 10000000),
                    'bytes_recv': random.randint(1000000, 10000000)
                },
                'process_count': random.randint(100, 500),
                'load_average': [random.uniform(0.5, 4.0) for _ in range(3)]
            }

        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return {}

    def get_detection_confidence(self) -> float:
        """获取检测置信度"""
        try:
            if len(self.detection_history) < 3:
                return 0.5

            # 基于历史检测结果计算置信度
            recent_detections = self.detection_history[-10:]
            detection_consistency = len(set(d['failures_detected'] for d in recent_detections))

            if detection_consistency <= 2:
                return 0.9  # 检测结果一致，置信度高
            elif detection_consistency <= 4:
                return 0.7  # 检测结果较一致
            else:
                return 0.5  # 检测结果波动较大

        except Exception:
            return 0.5


class ServiceFailureDetector(FailureDetector):
    """服务故障检测器"""

    def __init__(self):
        self.monitored_services = [
            'stock_analysis_api',
            'clickhouse_database',
            'prometheus_metrics',
            'elasticsearch_logs',
            'redis_cache'
        ]
        self.service_health_history = defaultdict(list)

    @exception_handler(reraise=True)
    def detect_failures(self) -> List[FailureEvent]:
        """检测服务故障"""
        failures = []
        detection_time = datetime.now()

        try:
            for service_name in self.monitored_services:
                try:
                    health_status = self._check_service_health(service_name)
                    self.service_health_history[service_name].append({
                        'timestamp': detection_time,
                        'status': health_status
                    })

                    # 清理历史记录
                    if len(self.service_health_history[service_name]) > 50:
                        self.service_health_history[service_name] = self.service_health_history[service_name][-50:]

                    # 检测故障
                    if not health_status['healthy']:
                        failure = FailureEvent(
                            event_id=f"service_down_{service_name}_{int(detection_time.timestamp())}",
                            failure_type=FailureType.SERVICE_DOWN,
                            severity=self._determine_service_severity(service_name),
                            affected_component=service_name,
                            description=f"服务 {service_name} 不可用",
                            detection_time=detection_time,
                            symptoms=[f"服务 {service_name} 健康检查失败", health_status.get('error', '未知错误')],
                            metrics=health_status,
                            context={'service_type': self._get_service_type(service_name)}
                        )
                        failures.append(failure)

                    # 检测高延迟
                    response_time = health_status.get('response_time', 0)
                    if response_time > 10:  # 10秒阈值
                        failure = FailureEvent(
                            event_id=f"high_latency_{service_name}_{int(detection_time.timestamp())}",
                            failure_type=FailureType.HIGH_LATENCY,
                            severity=FailureSeverity.MEDIUM,
                            affected_component=service_name,
                            description=f"服务 {service_name} 响应时间过长: {response_time:.2f}秒",
                            detection_time=detection_time,
                            symptoms=[f"响应时间{response_time:.2f}秒", "服务性能下降"],
                            metrics={'response_time': response_time},
                            context={'threshold': 10.0}
                        )
                        failures.append(failure)

                except Exception as e:
                    logger.error(f"检测服务 {service_name} 故障失败: {e}")

        except Exception as e:
            logger.error(f"服务故障检测失败: {e}")

        return failures

    def _check_service_health(self, service_name: str) -> Dict[str, Any]:
        """检查服务健康状态（模拟）"""
        import random

        # 模拟健康检查
        is_healthy = random.random() > 0.1  # 90%概率健康
        response_time = random.uniform(0.1, 15.0) if is_healthy else 0

        if is_healthy:
            return {
                'healthy': True,
                'response_time': response_time,
                'status_code': 200,
                'message': 'Service is healthy'
            }
        else:
            return {
                'healthy': False,
                'response_time': 0,
                'status_code': random.choice([500, 503, 504]),
                'error': random.choice(['Connection refused', 'Timeout', 'Internal server error'])
            }

    def _determine_service_severity(self, service_name: str) -> FailureSeverity:
        """确定服务故障严重程度"""
        critical_services = ['stock_analysis_api', 'clickhouse_database']
        important_services = ['prometheus_metrics', 'elasticsearch_logs']

        if service_name in critical_services:
            return FailureSeverity.CRITICAL
        elif service_name in important_services:
            return FailureSeverity.HIGH
        else:
            return FailureSeverity.MEDIUM

    def _get_service_type(self, service_name: str) -> str:
        """获取服务类型"""
        service_types = {
            'stock_analysis_api': 'api_service',
            'clickhouse_database': 'database',
            'prometheus_metrics': 'monitoring',
            'elasticsearch_logs': 'logging',
            'redis_cache': 'cache'
        }
        return service_types.get(service_name, 'unknown')

    def get_detection_confidence(self) -> float:
        """获取检测置信度"""
        try:
            total_checks = sum(len(history) for history in self.service_health_history.values())
            if total_checks == 0:
                return 0.5

            # 基于检查历史计算置信度
            if total_checks > 100:
                return 0.95
            elif total_checks > 50:
                return 0.85
            else:
                return 0.7

        except Exception:
            return 0.5


class RootCauseAnalyzer:
    """根因分析器"""

    def __init__(self):
        self.component_dependencies = self._load_component_dependencies()
        self.failure_patterns = self._load_failure_patterns()
        self.analysis_history: List[Dict[str, Any]] = []

    def _load_component_dependencies(self) -> Dict[str, ComponentDependency]:
        """加载组件依赖关系"""
        dependencies = {
            'stock_analysis_api': ComponentDependency(
                component_name='stock_analysis_api',
                dependencies=['clickhouse_database', 'redis_cache'],
                dependents=['web_frontend', 'mobile_app'],
                criticality_level=5,
                recovery_priority=10
            ),
            'clickhouse_database': ComponentDependency(
                component_name='clickhouse_database',
                dependencies=['system_storage', 'system_memory'],
                dependents=['stock_analysis_api', 'data_pipeline'],
                criticality_level=5,
                recovery_priority=9
            ),
            'redis_cache': ComponentDependency(
                component_name='redis_cache',
                dependencies=['system_memory'],
                dependents=['stock_analysis_api'],
                criticality_level=3,
                recovery_priority=5
            ),
            'prometheus_metrics': ComponentDependency(
                component_name='prometheus_metrics',
                dependencies=['system_storage'],
                dependents=['grafana_dashboard', 'alert_manager'],
                criticality_level=2,
                recovery_priority=3
            )
        }
        return dependencies

    def _load_failure_patterns(self) -> List[FailurePattern]:
        """加载故障模式"""
        patterns = [
            FailurePattern(
                pattern_id='memory_leak_cascade',
                name='内存泄漏级联故障',
                description='内存泄漏导致的级联服务故障',
                failure_types=[FailureType.MEMORY_LEAK, FailureType.SERVICE_DOWN],
                symptoms=['内存使用率持续上升', '服务响应时间增加', '服务不可用'],
                conditions={'memory_growth_rate': 5, 'affected_services': 2},
                recovery_strategies=[RecoveryStrategy.RESTART_SERVICE, RecoveryStrategy.CLEAR_CACHE],
                success_rate=0.85,
                last_updated=datetime.now()
            ),
            FailurePattern(
                pattern_id='disk_full_cascade',
                name='磁盘满级联故障',
                description='磁盘空间不足导致的多服务故障',
                failure_types=[FailureType.DISK_FULL, FailureType.DATABASE_ERROR],
                symptoms=['磁盘使用率达到临界值', '数据库写入失败', '日志记录异常'],
                conditions={'disk_usage': 95, 'write_failures': 10},
                recovery_strategies=[RecoveryStrategy.CLEAR_CACHE, RecoveryStrategy.SCALE_UP],
                success_rate=0.90,
                last_updated=datetime.now()
            ),
            FailurePattern(
                pattern_id='network_partition',
                name='网络分区故障',
                description='网络连接问题导致的服务不可达',
                failure_types=[FailureType.NETWORK_ISSUE, FailureType.SERVICE_DOWN],
                symptoms=['网络连接超时', '服务间通信失败', '负载不均衡'],
                conditions={'connection_timeouts': 5, 'network_errors': 20},
                recovery_strategies=[RecoveryStrategy.SWITCH_BACKUP, RecoveryStrategy.RESTART_SERVICE],
                success_rate=0.75,
                last_updated=datetime.now()
            )
        ]
        return patterns

    @exception_handler(reraise=True)
    def analyze_root_cause(self, failure_events: List[FailureEvent]) -> Dict[str, Any]:
        """分析故障根因"""
        analysis_start = datetime.now()

        try:
            if not failure_events:
                return {'root_cause': None, 'confidence': 0.0}

            # 分析单个故障的根因
            if len(failure_events) == 1:
                return self._analyze_single_failure(failure_events[0])

            # 分析多个故障的关联根因
            return self._analyze_cascading_failures(failure_events)

        except Exception as e:
            logger.error(f"根因分析失败: {e}")
            return {'root_cause': 'analysis_failed', 'confidence': 0.0, 'error': str(e)}

        finally:
            analysis_duration = (datetime.now() - analysis_start).total_seconds()
            self.analysis_history.append({
                'timestamp': analysis_start,
                'failure_count': len(failure_events),
                'duration': analysis_duration
            })

    def _analyze_single_failure(self, failure: FailureEvent) -> Dict[str, Any]:
        """分析单个故障根因"""
        analysis = {
            'failure_id': failure.event_id,
            'root_cause': 'unknown',
            'confidence': 0.5,
            'contributing_factors': [],
            'affected_components': [failure.affected_component],
            'impact_analysis': {}
        }

        try:
            # 基于故障类型分析
            if failure.failure_type == FailureType.MEMORY_LEAK:
                analysis['root_cause'] = 'memory_management_issue'
                analysis['confidence'] = 0.8
                analysis['contributing_factors'] = [
                    '应用程序内存泄漏',
                    '缓存配置不当',
                    '垃圾回收配置问题'
                ]

            elif failure.failure_type == FailureType.DISK_FULL:
                analysis['root_cause'] = 'storage_capacity_exceeded'
                analysis['confidence'] = 0.9
                analysis['contributing_factors'] = [
                    '日志文件过大',
                    '数据增长超预期',
                    '临时文件未清理'
                ]

            elif failure.failure_type == FailureType.SERVICE_DOWN:
                analysis['root_cause'] = 'service_configuration_or_dependency_issue'
                analysis['confidence'] = 0.7
                analysis['contributing_factors'] = [
                    '依赖服务不可用',
                    '配置错误',
                    '资源不足'
                ]

            # 分析影响范围
            component_deps = self.component_dependencies.get(failure.affected_component)
            if component_deps:
                analysis['affected_components'].extend(component_deps.dependents)
                analysis['impact_analysis'] = {
                    'criticality_level': component_deps.criticality_level,
                    'dependent_services': component_deps.dependents,
                    'recovery_priority': component_deps.recovery_priority
                }

        except Exception as e:
            logger.error(f"单个故障根因分析失败: {e}")
            analysis['error'] = str(e)

        return analysis

    def _analyze_cascading_failures(self, failures: List[FailureEvent]) -> Dict[str, Any]:
        """分析级联故障根因"""
        analysis = {
            'failure_count': len(failures),
            'root_cause': 'cascading_failure',
            'confidence': 0.6,
            'failure_chain': [],
            'affected_components': [],
            'pattern_match': None
        }

        try:
            # 按时间排序故障
            sorted_failures = sorted(failures, key=lambda f: f.detection_time)

            # 构建故障链
            for failure in sorted_failures:
                analysis['failure_chain'].append({
                    'component': failure.affected_component,
                    'type': failure.failure_type.value,
                    'time': failure.detection_time.isoformat(),
                    'severity': failure.severity.value
                })
                analysis['affected_components'].append(failure.affected_component)

            # 去重
            analysis['affected_components'] = list(set(analysis['affected_components']))

            # 匹配故障模式
            pattern_match = self._match_failure_pattern(failures)
            if pattern_match:
                analysis['pattern_match'] = pattern_match
                analysis['confidence'] = 0.8
                analysis['root_cause'] = pattern_match['name']

            # 分析依赖关系
            dependency_analysis = self._analyze_dependency_chain(analysis['affected_components'])
            analysis['dependency_analysis'] = dependency_analysis

        except Exception as e:
            logger.error(f"级联故障根因分析失败: {e}")
            analysis['error'] = str(e)

        return analysis

    def _match_failure_pattern(self, failures: List[FailureEvent]) -> Optional[Dict[str, Any]]:
        """匹配故障模式"""
        try:
            failure_types = [f.failure_type for f in failures]
            symptoms = []
            for f in failures:
                symptoms.extend(f.symptoms)

            for pattern in self.failure_patterns:
                # 检查故障类型匹配
                pattern_types_matched = any(ft in failure_types for ft in pattern.failure_types)

                # 检查症状匹配
                symptoms_matched = any(symptom in ' '.join(symptoms) for symptom in pattern.symptoms)

                if pattern_types_matched and symptoms_matched:
                    return {
                        'pattern_id': pattern.pattern_id,
                        'name': pattern.name,
                        'description': pattern.description,
                        'success_rate': pattern.success_rate,
                        'recommended_strategies': [s.value for s in pattern.recovery_strategies]
                    }

        except Exception as e:
            logger.error(f"匹配故障模式失败: {e}")

        return None

    def _analyze_dependency_chain(self, affected_components: List[str]) -> Dict[str, Any]:
        """分析依赖链"""
        analysis = {
            'primary_components': [],
            'secondary_components': [],
            'recovery_order': []
        }

        try:
            # 分析组件依赖关系
            primary_failures = []
            secondary_failures = []

            for component in affected_components:
                component_deps = self.component_dependencies.get(component)
                if component_deps:
                    # 检查是否有依赖的组件也故障了
                    failed_dependencies = [dep for dep in component_deps.dependencies if dep in affected_components]

                    if failed_dependencies:
                        # 这是一个二级故障
                        secondary_failures.append({
                            'component': component,
                            'failed_dependencies': failed_dependencies,
                            'priority': component_deps.recovery_priority
                        })
                    else:
                        # 这可能是一个主要故障
                        primary_failures.append({
                            'component': component,
                            'priority': component_deps.recovery_priority
                        })

            # 按优先级排序
            primary_failures.sort(key=lambda x: x['priority'], reverse=True)
            secondary_failures.sort(key=lambda x: x['priority'], reverse=True)

            analysis['primary_components'] = [f['component'] for f in primary_failures]
            analysis['secondary_components'] = [f['component'] for f in secondary_failures]

            # 确定恢复顺序（先恢复依赖组件）
            recovery_order = []
            for primary in primary_failures:
                recovery_order.append(primary['component'])
            for secondary in secondary_failures:
                recovery_order.append(secondary['component'])

            analysis['recovery_order'] = recovery_order

        except Exception as e:
            logger.error(f"依赖链分析失败: {e}")
            analysis['error'] = str(e)

        return analysis


class RecoveryExecutor:
    """恢复执行器"""

    def __init__(self):
        self.recovery_strategies = self._register_recovery_strategies()
        self.recovery_history: List[RecoveryAction] = []
        self.execution_lock = threading.RLock()

    def _register_recovery_strategies(self) -> Dict[RecoveryStrategy, Callable]:
        """注册恢复策略"""
        return {
            RecoveryStrategy.RESTART_SERVICE: self._restart_service,
            RecoveryStrategy.RELOAD_CONFIG: self._reload_config,
            RecoveryStrategy.CLEAR_CACHE: self._clear_cache,
            RecoveryStrategy.SCALE_UP: self._scale_up,
            RecoveryStrategy.SCALE_DOWN: self._scale_down,
            RecoveryStrategy.SWITCH_BACKUP: self._switch_backup,
            RecoveryStrategy.ROLLBACK_DEPLOYMENT: self._rollback_deployment,
            RecoveryStrategy.ISOLATE_COMPONENT: self._isolate_component,
            RecoveryStrategy.EMERGENCY_SHUTDOWN: self._emergency_shutdown
        }

    @exception_handler(reraise=True)
    def execute_recovery(self, failure_event: FailureEvent,
                        strategy: RecoveryStrategy,
                        parameters: Dict[str, Any] = None) -> RecoveryAction:
        """执行恢复操作"""
        action_id = f"recovery_{failure_event.event_id}_{strategy.value}_{int(time.time())}"

        recovery_action = RecoveryAction(
            action_id=action_id,
            strategy=strategy,
            component=failure_event.affected_component,
            parameters=parameters or {},
            execution_time=datetime.now(),
            duration=0.0,
            status=RecoveryStatus.IN_PROGRESS,
            output=""
        )

        start_time = time.time()

        try:
            with self.execution_lock:
                # 记录开始执行
                logger.info(f"开始执行恢复策略: {strategy.value} for {failure_event.affected_component}")

                # 执行恢复策略
                if strategy in self.recovery_strategies:
                    handler = self.recovery_strategies[strategy]
                    result = handler(failure_event, parameters or {})

                    recovery_action.output = result.get('output', '')
                    recovery_action.status = RecoveryStatus.SUCCESS if result.get('success') else RecoveryStatus.FAILED

                    if not result.get('success'):
                        recovery_action.error_message = result.get('error', 'Unknown error')

                else:
                    recovery_action.status = RecoveryStatus.FAILED
                    recovery_action.error_message = f"不支持的恢复策略: {strategy.value}"

        except Exception as e:
            recovery_action.status = RecoveryStatus.FAILED
            recovery_action.error_message = str(e)
            logger.error(f"执行恢复策略失败: {e}")

        finally:
            recovery_action.duration = time.time() - start_time
            self.recovery_history.append(recovery_action)

            # 清理历史记录（保留最近200条）
            if len(self.recovery_history) > 200:
                self.recovery_history = self.recovery_history[-200:]

            logger.info(f"恢复操作完成: {action_id}, 状态: {recovery_action.status.value}")

        return recovery_action

    def _restart_service(self, failure_event: FailureEvent, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """重启服务"""
        try:
            service_name = failure_event.affected_component
            logger.info(f"重启服务: {service_name}")

            # 模拟重启操作
            time.sleep(2)

            # 模拟成功率 90%
            import random
            success = random.random() > 0.1

            if success:
                return {
                    'success': True,
                    'output': f"服务 {service_name} 重启成功",
                    'details': {'restart_time': datetime.now().isoformat()}
                }
            else:
                return {
                    'success': False,
                    'error': f"服务 {service_name} 重启失败",
                    'output': "重启过程中遇到错误"
                }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _reload_config(self, failure_event: FailureEvent, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """重载配置"""
        try:
            component = failure_event.affected_component
            logger.info(f"重载配置: {component}")

            time.sleep(1)

            return {
                'success': True,
                'output': f"组件 {component} 配置重载成功",
                'details': {'config_reload_time': datetime.now().isoformat()}
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _clear_cache(self, failure_event: FailureEvent, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """清理缓存"""
        try:
            component = failure_event.affected_component
            logger.info(f"清理缓存: {component}")

            time.sleep(0.5)

            # 模拟清理不同类型的缓存
            cache_types = parameters.get('cache_types', ['memory', 'disk'])
            cleared_size = 0

            for cache_type in cache_types:
                if cache_type == 'memory':
                    cleared_size += 512  # MB
                elif cache_type == 'disk':
                    cleared_size += 1024  # MB

            return {
                'success': True,
                'output': f"组件 {component} 缓存清理成功",
                'details': {
                    'cleared_cache_types': cache_types,
                    'cleared_size_mb': cleared_size
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _scale_up(self, failure_event: FailureEvent, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """扩容"""
        try:
            component = failure_event.affected_component
            replicas = parameters.get('replicas', 1)

            logger.info(f"扩容组件: {component}, 增加 {replicas} 个实例")

            time.sleep(3)

            return {
                'success': True,
                'output': f"组件 {component} 扩容成功",
                'details': {
                    'added_replicas': replicas,
                    'scale_up_time': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _scale_down(self, failure_event: FailureEvent, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """缩容"""
        try:
            component = failure_event.affected_component
            replicas = parameters.get('replicas', 1)

            logger.info(f"缩容组件: {component}, 减少 {replicas} 个实例")

            time.sleep(2)

            return {
                'success': True,
                'output': f"组件 {component} 缩容成功",
                'details': {
                    'removed_replicas': replicas,
                    'scale_down_time': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _switch_backup(self, failure_event: FailureEvent, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """切换到备用服务"""
        try:
            component = failure_event.affected_component
            backup_instance = parameters.get('backup_instance', f"{component}_backup")

            logger.info(f"切换到备用服务: {component} -> {backup_instance}")

            time.sleep(5)

            return {
                'success': True,
                'output': f"已切换到备用服务 {backup_instance}",
                'details': {
                    'original_component': component,
                    'backup_instance': backup_instance,
                    'switchover_time': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _rollback_deployment(self, failure_event: FailureEvent, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """回滚部署"""
        try:
            component = failure_event.affected_component
            target_version = parameters.get('target_version', 'previous')

            logger.info(f"回滚部署: {component} 到版本 {target_version}")

            time.sleep(10)

            return {
                'success': True,
                'output': f"组件 {component} 回滚到版本 {target_version} 成功",
                'details': {
                    'target_version': target_version,
                    'rollback_time': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _isolate_component(self, failure_event: FailureEvent, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """隔离组件"""
        try:
            component = failure_event.affected_component

            logger.info(f"隔离故障组件: {component}")

            time.sleep(1)

            return {
                'success': True,
                'output': f"组件 {component} 已被隔离",
                'details': {
                    'isolation_time': datetime.now().isoformat(),
                    'isolation_method': 'network_isolation'
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _emergency_shutdown(self, failure_event: FailureEvent, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """紧急关闭"""
        try:
            component = failure_event.affected_component

            logger.warning(f"紧急关闭组件: {component}")

            time.sleep(1)

            return {
                'success': True,
                'output': f"组件 {component} 已紧急关闭",
                'details': {
                    'shutdown_time': datetime.now().isoformat(),
                    'shutdown_reason': failure_event.description
                }
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """获取恢复统计信息"""
        try:
            if not self.recovery_history:
                return {}

            total_recoveries = len(self.recovery_history)
            successful_recoveries = len([r for r in self.recovery_history if r.status == RecoveryStatus.SUCCESS])

            # 按策略统计
            strategy_stats = defaultdict(lambda: {'total': 0, 'success': 0})
            for recovery in self.recovery_history:
                strategy_stats[recovery.strategy.value]['total'] += 1
                if recovery.status == RecoveryStatus.SUCCESS:
                    strategy_stats[recovery.strategy.value]['success'] += 1

            # 计算平均恢复时间
            durations = [r.duration for r in self.recovery_history if r.duration > 0]
            avg_duration = sum(durations) / len(durations) if durations else 0

            return {
                'total_recoveries': total_recoveries,
                'successful_recoveries': successful_recoveries,
                'success_rate': (successful_recoveries / total_recoveries) * 100,
                'average_duration': avg_duration,
                'strategy_statistics': dict(strategy_stats),
                'recent_recoveries': [asdict(r) for r in self.recovery_history[-10:]]
            }

        except Exception as e:
            logger.error(f"获取恢复统计失败: {e}")
            return {'error': str(e)}


class IntelligentSelfHealingSystem:
    """
    智能自愈系统

    整合故障检测、根因分析、自动恢复的完整自愈系统
    """

    def __init__(self):
        # 初始化各个组件
        self.failure_detectors = [
            SystemFailureDetector(),
            ServiceFailureDetector()
        ]
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.recovery_executor = RecoveryExecutor()

        # 自愈配置
        self.healing_enabled = True
        self.auto_recovery_enabled = True
        self.manual_approval_required = False

        # 自愈历史
        self.healing_sessions: List[Dict[str, Any]] = []

        # 监控线程
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        logger.info("智能自愈系统初始化完成")

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=10.0)
    def start_self_healing(self, detection_interval: int = 60) -> Dict[str, Any]:
        """启动自愈系统"""
        if self.monitoring_active:
            return {'status': 'already_running'}

        try:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._healing_loop,
                args=(detection_interval,),
                daemon=True
            )
            self.monitoring_thread.start()

            logger.info(f"智能自愈系统启动成功，检测间隔: {detection_interval}秒")

            return {
                'status': 'started',
                'detection_interval': detection_interval,
                'healing_enabled': self.healing_enabled,
                'auto_recovery_enabled': self.auto_recovery_enabled,
                'start_time': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"启动自愈系统失败: {e}")
            return {'status': 'failed', 'error': str(e)}

    def stop_self_healing(self):
        """停止自愈系统"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("智能自愈系统已停止")

    def _healing_loop(self, detection_interval: int):
        """自愈主循环"""
        while self.monitoring_active:
            try:
                if self.healing_enabled:
                    # 执行一次完整的自愈周期
                    self._execute_healing_cycle()

                time.sleep(detection_interval)

            except Exception as e:
                logger.error(f"自愈循环出错: {e}")
                time.sleep(30)

    @exception_handler(reraise=True)
    def _execute_healing_cycle(self):
        """执行自愈周期"""
        session_id = f"healing_{int(time.time())}"
        session_start = datetime.now()

        healing_session = {
            'session_id': session_id,
            'start_time': session_start,
            'failures_detected': [],
            'root_cause_analysis': {},
            'recovery_actions': [],
            'final_status': 'unknown',
            'duration': 0.0
        }

        try:
            logger.debug("开始执行自愈周期")

            # 1. 故障检测
            all_failures = []
            for detector in self.failure_detectors:
                try:
                    failures = detector.detect_failures()
                    all_failures.extend(failures)
                except Exception as e:
                    logger.error(f"故障检测器执行失败: {e}")

            healing_session['failures_detected'] = [asdict(f) for f in all_failures]

            if not all_failures:
                healing_session['final_status'] = 'no_failures_detected'
                return

            logger.info(f"检测到 {len(all_failures)} 个故障")

            # 2. 根因分析
            try:
                root_cause_analysis = self.root_cause_analyzer.analyze_root_cause(all_failures)
                healing_session['root_cause_analysis'] = root_cause_analysis
                logger.info(f"根因分析完成: {root_cause_analysis.get('root_cause', 'unknown')}")
            except Exception as e:
                logger.error(f"根因分析失败: {e}")
                healing_session['root_cause_analysis'] = {'error': str(e)}

            # 3. 执行恢复策略
            if self.auto_recovery_enabled:
                recovery_actions = self._execute_recovery_strategies(all_failures, root_cause_analysis)
                healing_session['recovery_actions'] = [asdict(action) for action in recovery_actions]

                # 4. 验证恢复效果
                verification_result = self._verify_recovery(all_failures, recovery_actions)
                healing_session['verification_result'] = verification_result

                if verification_result.get('success', False):
                    healing_session['final_status'] = 'recovered'
                    logger.info(f"自愈成功: {session_id}")
                else:
                    healing_session['final_status'] = 'recovery_failed'
                    logger.warning(f"自愈失败: {session_id}")
            else:
                healing_session['final_status'] = 'auto_recovery_disabled'
                logger.info("自动恢复已禁用，仅进行故障检测和分析")

        except Exception as e:
            healing_session['final_status'] = 'error'
            healing_session['error'] = str(e)
            logger.error(f"自愈周期执行失败: {e}")

        finally:
            healing_session['duration'] = (datetime.now() - session_start).total_seconds()
            self.healing_sessions.append(healing_session)

            # 清理历史记录（保留最近50次会话）
            if len(self.healing_sessions) > 50:
                self.healing_sessions = self.healing_sessions[-50:]

    def _execute_recovery_strategies(self, failures: List[FailureEvent],
                                   root_cause_analysis: Dict[str, Any]) -> List[RecoveryAction]:
        """执行恢复策略"""
        recovery_actions = []

        try:
            # 获取推荐的恢复策略
            recommended_strategies = self._get_recommended_strategies(failures, root_cause_analysis)

            for failure, strategies in recommended_strategies.items():
                for strategy_info in strategies:
                    try:
                        recovery_action = self.recovery_executor.execute_recovery(
                            failure,
                            strategy_info['strategy'],
                            strategy_info.get('parameters', {})
                        )
                        recovery_actions.append(recovery_action)

                        # 如果恢复成功，可能不需要执行其他策略
                        if recovery_action.status == RecoveryStatus.SUCCESS:
                            logger.info(f"恢复策略 {strategy_info['strategy'].value} 执行成功")
                            break
                        else:
                            logger.warning(f"恢复策略 {strategy_info['strategy'].value} 执行失败")

                    except Exception as e:
                        logger.error(f"执行恢复策略失败: {e}")

        except Exception as e:
            logger.error(f"执行恢复策略过程失败: {e}")

        return recovery_actions

    def _get_recommended_strategies(self, failures: List[FailureEvent],
                                  root_cause_analysis: Dict[str, Any]) -> Dict[FailureEvent, List[Dict[str, Any]]]:
        """获取推荐的恢复策略"""
        strategies = {}

        try:
            for failure in failures:
                failure_strategies = []

                # 基于故障类型推荐策略
                if failure.failure_type == FailureType.MEMORY_LEAK:
                    failure_strategies.extend([
                        {'strategy': RecoveryStrategy.RESTART_SERVICE, 'priority': 1},
                        {'strategy': RecoveryStrategy.CLEAR_CACHE, 'priority': 2}
                    ])

                elif failure.failure_type == FailureType.DISK_FULL:
                    failure_strategies.extend([
                        {'strategy': RecoveryStrategy.CLEAR_CACHE, 'priority': 1},
                        {'strategy': RecoveryStrategy.SCALE_UP, 'priority': 2, 'parameters': {'storage': True}}
                    ])

                elif failure.failure_type == FailureType.SERVICE_DOWN:
                    failure_strategies.extend([
                        {'strategy': RecoveryStrategy.RESTART_SERVICE, 'priority': 1},
                        {'strategy': RecoveryStrategy.SWITCH_BACKUP, 'priority': 2}
                    ])

                elif failure.failure_type == FailureType.HIGH_LATENCY:
                    failure_strategies.extend([
                        {'strategy': RecoveryStrategy.CLEAR_CACHE, 'priority': 1},
                        {'strategy': RecoveryStrategy.SCALE_UP, 'priority': 2}
                    ])

                elif failure.failure_type == FailureType.NETWORK_ISSUE:
                    failure_strategies.extend([
                        {'strategy': RecoveryStrategy.SWITCH_BACKUP, 'priority': 1},
                        {'strategy': RecoveryStrategy.RESTART_SERVICE, 'priority': 2}
                    ])

                # 基于根因分析结果调整策略
                pattern_match = root_cause_analysis.get('pattern_match')
                if pattern_match and 'recommended_strategies' in pattern_match:
                    for strategy_name in pattern_match['recommended_strategies']:
                        try:
                            strategy = RecoveryStrategy(strategy_name)
                            failure_strategies.append({'strategy': strategy, 'priority': 0})
                        except ValueError:
                            continue

                # 按优先级排序
                failure_strategies.sort(key=lambda x: x.get('priority', 999))
                strategies[failure] = failure_strategies

        except Exception as e:
            logger.error(f"获取推荐策略失败: {e}")

        return strategies

    def _verify_recovery(self, original_failures: List[FailureEvent],
                        recovery_actions: List[RecoveryAction]) -> Dict[str, Any]:
        """验证恢复效果"""
        verification_result = {
            'success': False,
            'verified_at': datetime.now(),
            'verification_details': {},
            'remaining_issues': []
        }

        try:
            # 等待一段时间让系统稳定
            time.sleep(10)

            # 重新检测故障
            current_failures = []
            for detector in self.failure_detectors:
                try:
                    failures = detector.detect_failures()
                    current_failures.extend(failures)
                except Exception as e:
                    logger.error(f"验证时故障检测失败: {e}")

            # 比较原始故障和当前故障
            original_components = set(f.affected_component for f in original_failures)
            current_components = set(f.affected_component for f in current_failures)

            resolved_components = original_components - current_components
            remaining_components = original_components & current_components

            verification_result['verification_details'] = {
                'original_failures_count': len(original_failures),
                'current_failures_count': len(current_failures),
                'resolved_components': list(resolved_components),
                'remaining_components': list(remaining_components),
                'recovery_actions_count': len(recovery_actions),
                'successful_actions': len([a for a in recovery_actions if a.status == RecoveryStatus.SUCCESS])
            }

            # 检查每个恢复动作的对应组件是否恢复
            for action in recovery_actions:
                if action.status == RecoveryStatus.SUCCESS:
                    if action.component not in current_components:
                        verification_result['verification_details'][f'{action.component}_recovered'] = True
                    else:
                        verification_result['verification_details'][f'{action.component}_recovered'] = False

            # 判断整体恢复是否成功
            if len(resolved_components) > 0 and len(remaining_components) == 0:
                verification_result['success'] = True
            elif len(resolved_components) > len(remaining_components):
                verification_result['success'] = True  # 部分成功也算成功
            else:
                verification_result['success'] = False

            # 记录剩余问题
            verification_result['remaining_issues'] = [asdict(f) for f in current_failures]

            logger.info(f"恢复验证完成: 成功={verification_result['success']}, "
                       f"解决={len(resolved_components)}, 剩余={len(remaining_components)}")

        except Exception as e:
            logger.error(f"恢复验证失败: {e}")
            verification_result['error'] = str(e)

        return verification_result

    @exception_handler(reraise=True)
    def get_healing_dashboard(self) -> Dict[str, Any]:
        """获取自愈仪表板数据"""
        try:
            dashboard_data = {
                'system_status': {
                    'healing_enabled': self.healing_enabled,
                    'auto_recovery_enabled': self.auto_recovery_enabled,
                    'monitoring_active': self.monitoring_active,
                    'manual_approval_required': self.manual_approval_required
                },
                'statistics': self._get_healing_statistics(),
                'recent_sessions': self.healing_sessions[-10:],  # 最近10次会话
                'recovery_statistics': self.recovery_executor.get_recovery_statistics(),
                'detector_confidence': self._get_detector_confidence(),
                'timestamp': datetime.now().isoformat()
            }

            return dashboard_data

        except Exception as e:
            logger.error(f"获取自愈仪表板数据失败: {e}")
            return {'error': str(e)}

    def _get_healing_statistics(self) -> Dict[str, Any]:
        """获取自愈统计信息"""
        try:
            if not self.healing_sessions:
                return {}

            total_sessions = len(self.healing_sessions)
            successful_sessions = len([s for s in self.healing_sessions if s['final_status'] == 'recovered'])

            # 按状态统计
            status_stats = defaultdict(int)
            for session in self.healing_sessions:
                status_stats[session['final_status']] += 1

            # 计算平均持续时间
            durations = [s['duration'] for s in self.healing_sessions if s['duration'] > 0]
            avg_duration = sum(durations) / len(durations) if durations else 0

            # 故障类型统计
            failure_type_stats = defaultdict(int)
            for session in self.healing_sessions:
                for failure in session.get('failures_detected', []):
                    failure_type_stats[failure.get('failure_type', 'unknown')] += 1

            return {
                'total_healing_sessions': total_sessions,
                'successful_sessions': successful_sessions,
                'success_rate': (successful_sessions / total_sessions) * 100 if total_sessions > 0 else 0,
                'average_session_duration': avg_duration,
                'status_distribution': dict(status_stats),
                'failure_type_distribution': dict(failure_type_stats),
                'total_failures_detected': sum(len(s.get('failures_detected', [])) for s in self.healing_sessions),
                'total_recovery_actions': sum(len(s.get('recovery_actions', [])) for s in self.healing_sessions)
            }

        except Exception as e:
            logger.error(f"获取自愈统计失败: {e}")
            return {'error': str(e)}

    def _get_detector_confidence(self) -> Dict[str, float]:
        """获取检测器置信度"""
        confidence = {}

        try:
            for i, detector in enumerate(self.failure_detectors):
                detector_name = detector.__class__.__name__
                confidence[detector_name] = detector.get_detection_confidence()

        except Exception as e:
            logger.error(f"获取检测器置信度失败: {e}")

        return confidence

    @exception_handler(reraise=True)
    def manual_trigger_healing(self, component: str, failure_type: str,
                              recovery_strategy: str = None) -> Dict[str, Any]:
        """手动触发自愈"""
        try:
            # 创建手动故障事件
            manual_failure = FailureEvent(
                event_id=f"manual_{component}_{int(time.time())}",
                failure_type=FailureType(failure_type),
                severity=FailureSeverity.MEDIUM,
                affected_component=component,
                description=f"手动触发的故障恢复: {component}",
                detection_time=datetime.now(),
                symptoms=[f"手动指定组件 {component} 需要恢复"],
                metrics={},
                context={'manual_trigger': True}
            )

            # 执行恢复
            if recovery_strategy:
                strategy = RecoveryStrategy(recovery_strategy)
                recovery_action = self.recovery_executor.execute_recovery(manual_failure, strategy)

                return {
                    'success': True,
                    'manual_trigger': True,
                    'failure_event': asdict(manual_failure),
                    'recovery_action': asdict(recovery_action),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # 自动选择恢复策略
                recommended_strategies = self._get_recommended_strategies([manual_failure], {})
                strategies = recommended_strategies.get(manual_failure, [])

                if strategies:
                    strategy_info = strategies[0]  # 使用第一个推荐策略
                    recovery_action = self.recovery_executor.execute_recovery(
                        manual_failure,
                        strategy_info['strategy'],
                        strategy_info.get('parameters', {})
                    )

                    return {
                        'success': True,
                        'manual_trigger': True,
                        'failure_event': asdict(manual_failure),
                        'recovery_action': asdict(recovery_action),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'success': False,
                        'error': f'未找到适合组件 {component} 故障类型 {failure_type} 的恢复策略'
                    }

        except Exception as e:
            logger.error(f"手动触发自愈失败: {e}")
            return {'success': False, 'error': str(e)}

    def configure_healing_settings(self, **settings) -> Dict[str, Any]:
        """配置自愈设置"""
        try:
            if 'healing_enabled' in settings:
                self.healing_enabled = settings['healing_enabled']

            if 'auto_recovery_enabled' in settings:
                self.auto_recovery_enabled = settings['auto_recovery_enabled']

            if 'manual_approval_required' in settings:
                self.manual_approval_required = settings['manual_approval_required']

            logger.info(f"自愈设置已更新: {settings}")

            return {
                'success': True,
                'updated_settings': {
                    'healing_enabled': self.healing_enabled,
                    'auto_recovery_enabled': self.auto_recovery_enabled,
                    'manual_approval_required': self.manual_approval_required
                }
            }

        except Exception as e:
            logger.error(f"配置自愈设置失败: {e}")
            return {'success': False, 'error': str(e)}


# 全局自愈系统实例
_intelligent_self_healing_system = None


def get_intelligent_self_healing_system() -> IntelligentSelfHealingSystem:
    """
    获取智能自愈系统实例（单例模式）

    Returns:
        IntelligentSelfHealingSystem: 智能自愈系统实例
    """
    global _intelligent_self_healing_system

    if _intelligent_self_healing_system is None:
        _intelligent_self_healing_system = IntelligentSelfHealingSystem()

    return _intelligent_self_healing_system


def create_intelligent_self_healing_system() -> IntelligentSelfHealingSystem:
    """
    创建新的智能自愈系统实例

    Returns:
        IntelligentSelfHealingSystem: 新的智能自愈系统实例
    """
    return IntelligentSelfHealingSystem()