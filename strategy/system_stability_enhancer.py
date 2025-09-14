"""
系统稳定性增强器
完善错误处理和监控机制，提升系统整体稳定性
"""

import time
import threading
import traceback
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import logging

from utils.enhanced_exception_handler import exception_handler, ErrorSeverity, ErrorCategory
from utils.enhanced_performance_monitor import performance_monitor
from utils.unified_container import get_container
from config.unified_config_manager import get_config

logger = logging.getLogger(__name__)


class StabilityLevel(Enum):
    """稳定性级别"""
    CRITICAL = "critical"      # 关键稳定性
    HIGH = "high"             # 高稳定性
    MEDIUM = "medium"         # 中等稳定性
    LOW = "low"               # 低稳定性


class RecoveryStrategy(Enum):
    """恢复策略"""
    RETRY = "retry"                    # 重试
    FALLBACK = "fallback"             # 降级
    CIRCUIT_BREAKER = "circuit_breaker" # 熔断
    GRACEFUL_DEGRADATION = "graceful_degradation"  # 优雅降级


@dataclass
class StabilityConfig:
    """稳定性配置"""
    # 重试配置
    max_retry_attempts: int = 3
    retry_delay_base: float = 1.0
    retry_delay_multiplier: float = 2.0
    retry_max_delay: float = 60.0
    
    # 熔断器配置
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    circuit_breaker_recovery_threshold: int = 3
    
    # 健康检查配置
    health_check_interval: int = 30
    health_check_timeout: int = 10
    
    # 监控配置
    error_rate_threshold: float = 5.0  # 5%
    response_time_threshold: float = 5.0  # 5秒
    memory_threshold_mb: float = 2048
    cpu_threshold_percent: float = 85.0


@dataclass
class StabilityMetrics:
    """稳定性指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    stability_score: float = 100.0
    active_circuit_breakers: int = 0
    recovery_attempts: int = 0


@dataclass
class CircuitBreakerState:
    """熔断器状态"""
    name: str
    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None


class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, name: str, config: StabilityConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState(name=name)
        self.lock = threading.Lock()
    
    @contextmanager
    def call(self):
        """熔断器调用上下文"""
        if not self._can_execute():
            raise RuntimeError(f"熔断器 {self.name} 处于开启状态")
        
        start_time = time.time()
        try:
            yield
            self._on_success()
        except Exception as e:
            self._on_failure()
            raise
    
    def _can_execute(self) -> bool:
        """检查是否可以执行"""
        with self.lock:
            if self.state.state == "closed":
                return True
            elif self.state.state == "open":
                if (self.state.next_attempt_time and 
                    datetime.now() >= self.state.next_attempt_time):
                    self.state.state = "half_open"
                    return True
                return False
            elif self.state.state == "half_open":
                return True
            return False
    
    def _on_success(self):
        """成功回调"""
        with self.lock:
            if self.state.state == "half_open":
                self.state.state = "closed"
                self.state.failure_count = 0
            self.state.last_success_time = datetime.now()
    
    def _on_failure(self):
        """失败回调"""
        with self.lock:
            self.state.failure_count += 1
            self.state.last_failure_time = datetime.now()
            
            if self.state.failure_count >= self.config.circuit_breaker_threshold:
                self.state.state = "open"
                self.state.next_attempt_time = (
                    datetime.now() + timedelta(seconds=self.config.circuit_breaker_timeout)
                )
    
    def get_state(self) -> Dict[str, Any]:
        """获取熔断器状态"""
        with self.lock:
            return {
                "name": self.state.name,
                "state": self.state.state,
                "failure_count": self.state.failure_count,
                "last_failure_time": self.state.last_failure_time,
                "last_success_time": self.state.last_success_time,
                "next_attempt_time": self.state.next_attempt_time
            }


class RetryManager:
    """重试管理器"""
    
    def __init__(self, config: StabilityConfig):
        self.config = config
        self.retry_stats = {}
        self.lock = threading.Lock()
    
    def retry_operation(self, operation_name: str, operation_func, max_attempts: Optional[int] = None):
        """重试操作"""
        max_attempts = max_attempts or self.config.max_retry_attempts
        last_exception = None

        for attempt in range(max_attempts):
            try:
                result = operation_func()
                self._record_success(operation_name, attempt + 1)
                return result
            except Exception as e:
                last_exception = e
                self._record_failure(operation_name, attempt + 1, e)

                if attempt < max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"操作 {operation_name} 第{attempt + 1}次尝试失败，{delay}秒后重试: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"操作 {operation_name} 在{max_attempts}次尝试后最终失败: {e}")

        # 所有重试都失败
        if last_exception:
            raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        delay = self.config.retry_delay_base * (self.config.retry_delay_multiplier ** attempt)
        return min(delay, self.config.retry_max_delay)
    
    def _record_success(self, operation_name: str, attempts: int):
        """记录成功"""
        with self.lock:
            if operation_name not in self.retry_stats:
                self.retry_stats[operation_name] = {
                    "total_operations": 0,
                    "total_attempts": 0,
                    "success_count": 0,
                    "failure_count": 0
                }
            
            stats = self.retry_stats[operation_name]
            stats["total_operations"] += 1
            stats["total_attempts"] += attempts
            stats["success_count"] += 1
    
    def _record_failure(self, operation_name: str, attempts: int, error: Exception):
        """记录失败"""
        with self.lock:
            if operation_name not in self.retry_stats:
                self.retry_stats[operation_name] = {
                    "total_operations": 0,
                    "total_attempts": 0,
                    "success_count": 0,
                    "failure_count": 0
                }
            
            stats = self.retry_stats[operation_name]
            stats["total_operations"] += 1
            stats["total_attempts"] += attempts
            stats["failure_count"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取重试统计"""
        with self.lock:
            return dict(self.retry_stats)


class HealthChecker:
    """健康检查器"""
    
    def __init__(self, config: StabilityConfig):
        self.config = config
        self.health_checks = {}
        self.health_status = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
    
    def register_health_check(self, name: str, check_func: Callable[[], bool], 
                            critical: bool = False):
        """注册健康检查"""
        with self.lock:
            self.health_checks[name] = {
                "func": check_func,
                "critical": critical,
                "last_check": None,
                "last_result": None,
                "failure_count": 0
            }
    
    def start_monitoring(self):
        """开始健康监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="health_checker",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("健康检查监控已启动")
    
    def stop_monitoring(self):
        """停止健康监控"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("健康检查监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                self._run_health_checks()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"健康检查循环出错: {e}")
                time.sleep(60)
    
    def _run_health_checks(self):
        """运行健康检查"""
        with self.lock:
            for name, check_info in self.health_checks.items():
                try:
                    start_time = time.time()
                    result = check_info["func"]()
                    check_time = time.time() - start_time
                    
                    check_info["last_check"] = datetime.now()
                    check_info["last_result"] = result
                    
                    if result:
                        check_info["failure_count"] = 0
                        self.health_status[name] = {
                            "status": "healthy",
                            "last_check": check_info["last_check"],
                            "check_time": check_time
                        }
                    else:
                        check_info["failure_count"] += 1
                        self.health_status[name] = {
                            "status": "unhealthy",
                            "last_check": check_info["last_check"],
                            "failure_count": check_info["failure_count"],
                            "check_time": check_time
                        }
                        
                        if check_info["critical"]:
                            logger.error(f"关键健康检查失败: {name}")
                        else:
                            logger.warning(f"健康检查失败: {name}")
                            
                except Exception as e:
                    logger.error(f"健康检查 {name} 执行失败: {e}")
                    self.health_status[name] = {
                        "status": "error",
                        "error": str(e),
                        "last_check": datetime.now()
                    }
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        with self.lock:
            overall_status = "healthy"
            critical_failures = 0
            total_checks = len(self.health_status)
            healthy_checks = 0
            
            for name, status in self.health_status.items():
                if status["status"] == "healthy":
                    healthy_checks += 1
                elif status["status"] == "unhealthy":
                    check_info = self.health_checks.get(name, {})
                    if check_info.get("critical", False):
                        critical_failures += 1
                        overall_status = "critical"
                    elif overall_status == "healthy":
                        overall_status = "warning"
            
            return {
                "overall_status": overall_status,
                "total_checks": total_checks,
                "healthy_checks": healthy_checks,
                "critical_failures": critical_failures,
                "health_score": (healthy_checks / total_checks * 100) if total_checks > 0 else 100,
                "details": dict(self.health_status)
            }


class SystemStabilityEnhancer:
    """系统稳定性增强器主控制器"""

    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()
        self.circuit_breakers = {}
        self.retry_manager = RetryManager(self.config)
        self.health_checker = HealthChecker(self.config)

        self.stability_metrics = []
        self.error_handlers = {}
        self.recovery_strategies = {}
        self.is_active = False
        self.lock = threading.Lock()

        # 注册到容器
        container = get_container()
        container.register(SystemStabilityEnhancer, instance=self)

        # 注册默认健康检查
        self._register_default_health_checks()

        # 注册默认恢复策略
        self._register_default_recovery_strategies()

    @performance_monitor(threshold_seconds=2.0)
    @exception_handler(reraise=True)
    def start_stability_system(self) -> Dict[str, Any]:
        """启动稳定性系统"""
        if self.is_active:
            return {"status": "already_active"}

        start_time = time.time()

        try:
            # 启动健康检查
            self.health_checker.start_monitoring()

            # 初始化熔断器
            self._initialize_circuit_breakers()

            # 运行初始健康检查
            initial_health = self.health_checker.get_health_status()

            self.is_active = True

            return {
                "status": "started",
                "startup_time": time.time() - start_time,
                "initial_health": initial_health,
                "circuit_breakers": len(self.circuit_breakers),
                "health_checks": len(self.health_checker.health_checks)
            }

        except Exception as e:
            logger.error(f"启动稳定性系统失败: {e}")
            raise

    def stop_stability_system(self) -> Dict[str, Any]:
        """停止稳定性系统"""
        if not self.is_active:
            return {"status": "not_active"}

        # 停止健康检查
        self.health_checker.stop_monitoring()

        self.is_active = False

        return {
            "status": "stopped",
            "final_health": self.health_checker.get_health_status(),
            "retry_stats": self.retry_manager.get_stats()
        }

    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """获取熔断器"""
        with self.lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(name, self.config)
            return self.circuit_breakers[name]

    @contextmanager
    def stable_execution(self, operation_name: str,
                        use_circuit_breaker: bool = True,
                        use_retry: bool = True,
                        max_retry_attempts: Optional[int] = None):
        """稳定执行上下文管理器"""
        circuit_breaker = None
        if use_circuit_breaker:
            circuit_breaker = self.get_circuit_breaker(operation_name)

        if use_retry:
            with self.retry_manager.retry_context(operation_name, max_retry_attempts):
                if circuit_breaker:
                    with circuit_breaker.call():
                        yield
                else:
                    yield
        else:
            if circuit_breaker:
                with circuit_breaker.call():
                    yield
            else:
                yield

    def register_error_handler(self, error_type: type, handler: Callable):
        """注册错误处理器"""
        with self.lock:
            self.error_handlers[error_type] = handler

    def register_recovery_strategy(self, operation_name: str, strategy: Callable):
        """注册恢复策略"""
        with self.lock:
            self.recovery_strategies[operation_name] = strategy

    @exception_handler(reraise=False)
    def handle_error(self, error: Exception, operation_name: str = "unknown") -> bool:
        """处理错误"""
        error_type = type(error)

        # 尝试使用注册的错误处理器
        with self.lock:
            if error_type in self.error_handlers:
                try:
                    return self.error_handlers[error_type](error, operation_name)
                except Exception as handler_error:
                    logger.error(f"错误处理器执行失败: {handler_error}")

        # 尝试使用恢复策略
        with self.lock:
            if operation_name in self.recovery_strategies:
                try:
                    return self.recovery_strategies[operation_name](error)
                except Exception as recovery_error:
                    logger.error(f"恢复策略执行失败: {recovery_error}")

        # 默认错误处理
        logger.error(f"未处理的错误 {operation_name}: {error}")
        return False

    def _register_default_health_checks(self):
        """注册默认健康检查"""
        # 内存健康检查
        def memory_health_check():
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.percent < self.config.memory_threshold_mb
            except Exception:
                return True  # 无法检查时假设健康

        self.health_checker.register_health_check(
            "memory_usage", memory_health_check, critical=True
        )

        # CPU健康检查
        def cpu_health_check():
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                return cpu_percent < self.config.cpu_threshold_percent
            except Exception:
                return True

        self.health_checker.register_health_check(
            "cpu_usage", cpu_health_check, critical=False
        )

        # 数据库连接健康检查
        def database_health_check():
            try:
                from db.optimized_connection_pool import get_optimized_pool
                pool = get_optimized_pool()
                return pool.available_connections.qsize() > 0
            except Exception:
                return False

        self.health_checker.register_health_check(
            "database_connection", database_health_check, critical=True
        )

        # 指标注册健康检查
        def indicator_registry_health_check():
            try:
                from indicators.complete_indicator_registry import get_indicator_registry
                registry = get_indicator_registry()
                return len(registry.get_registered_indicators()) > 100
            except Exception:
                return False

        self.health_checker.register_health_check(
            "indicator_registry", indicator_registry_health_check, critical=True
        )

    def _register_default_recovery_strategies(self):
        """注册默认恢复策略"""
        # 数据库连接恢复策略（任务5整合：使用增强连接池）
        def database_recovery_strategy(error: Exception) -> bool:
            try:
                from db.enhanced_connection_pool import initialize_connection_pool
                initialize_connection_pool()
                logger.info("数据库连接池已重新初始化")
                return True
            except Exception as e:
                logger.error(f"数据库连接恢复失败: {e}")
                return False

        self.register_recovery_strategy("database_connection", database_recovery_strategy)

        # 内存清理恢复策略
        def memory_cleanup_strategy(error: Exception) -> bool:
            try:
                import gc
                collected = gc.collect()
                logger.info(f"内存清理完成，回收对象: {collected}")
                return True
            except Exception as e:
                logger.error(f"内存清理失败: {e}")
                return False

        self.register_recovery_strategy("memory_cleanup", memory_cleanup_strategy)

        # 缓存清理恢复策略
        def cache_cleanup_strategy(error: Exception) -> bool:
            try:
                from db.multi_layer_cache import get_multi_cache
                cache = get_multi_cache()
                cache.clear()
                logger.info("缓存已清理")
                return True
            except Exception as e:
                logger.error(f"缓存清理失败: {e}")
                return False

        self.register_recovery_strategy("cache_cleanup", cache_cleanup_strategy)

    def _initialize_circuit_breakers(self):
        """初始化熔断器"""
        # 为关键操作创建熔断器
        critical_operations = [
            "database_query",
            "indicator_calculation",
            "strategy_execution",
            "cache_operation",
            "file_operation"
        ]

        with self.lock:
            for operation in critical_operations:
                if operation not in self.circuit_breakers:
                    self.circuit_breakers[operation] = CircuitBreaker(operation, self.config)

    def get_stability_report(self) -> Dict[str, Any]:
        """获取稳定性报告"""
        health_status = self.health_checker.get_health_status()
        retry_stats = self.retry_manager.get_stats()

        # 获取熔断器状态
        circuit_breaker_states = {}
        with self.lock:
            for name, cb in self.circuit_breakers.items():
                circuit_breaker_states[name] = cb.get_state()

        # 计算稳定性评分
        stability_score = self._calculate_stability_score(health_status, circuit_breaker_states)

        return {
            "stability_score": stability_score,
            "health_status": health_status,
            "circuit_breakers": circuit_breaker_states,
            "retry_statistics": retry_stats,
            "system_status": "stable" if stability_score > 80 else "unstable" if stability_score > 60 else "critical",
            "recommendations": self._generate_stability_recommendations(stability_score, health_status)
        }

    def _calculate_stability_score(self, health_status: Dict[str, Any],
                                 circuit_breaker_states: Dict[str, Any]) -> float:
        """计算稳定性评分"""
        score = 100.0

        # 健康检查评分
        health_score = health_status.get("health_score", 100)
        score = score * (health_score / 100)

        # 熔断器评分
        open_breakers = sum(1 for state in circuit_breaker_states.values()
                           if state["state"] == "open")
        if open_breakers > 0:
            score -= open_breakers * 10

        # 错误率评分
        total_operations = sum(
            stats.get("total_operations", 0)
            for stats in self.retry_manager.get_stats().values()
        )
        total_failures = sum(
            stats.get("failure_count", 0)
            for stats in self.retry_manager.get_stats().values()
        )

        if total_operations > 0:
            error_rate = (total_failures / total_operations) * 100
            if error_rate > self.config.error_rate_threshold:
                score -= (error_rate - self.config.error_rate_threshold) * 5

        return max(score, 0.0)

    def _generate_stability_recommendations(self, stability_score: float,
                                          health_status: Dict[str, Any]) -> List[str]:
        """生成稳定性建议"""
        recommendations = []

        if stability_score < 60:
            recommendations.append("系统稳定性严重不足，建议立即检查关键组件")
        elif stability_score < 80:
            recommendations.append("系统稳定性需要改善，建议优化错误处理机制")

        # 基于健康检查的建议
        for name, status in health_status.get("details", {}).items():
            if status.get("status") == "unhealthy":
                if name == "memory_usage":
                    recommendations.append("内存使用过高，建议进行内存优化")
                elif name == "cpu_usage":
                    recommendations.append("CPU使用率过高，建议优化计算密集型操作")
                elif name == "database_connection":
                    recommendations.append("数据库连接异常，建议检查连接池配置")
                elif name == "indicator_registry":
                    recommendations.append("指标注册异常，建议重新初始化指标系统")

        return recommendations


# 全局稳定性增强器实例
_stability_enhancer = None
_enhancer_lock = threading.Lock()


def get_stability_enhancer() -> SystemStabilityEnhancer:
    """获取全局稳定性增强器实例"""
    global _stability_enhancer

    if _stability_enhancer is None:
        with _enhancer_lock:
            if _stability_enhancer is None:
                _stability_enhancer = SystemStabilityEnhancer()

    return _stability_enhancer


def initialize_stability_enhancer(config: Optional[StabilityConfig] = None) -> SystemStabilityEnhancer:
    """初始化稳定性增强器"""
    global _stability_enhancer

    with _enhancer_lock:
        if _stability_enhancer is not None:
            _stability_enhancer.stop_stability_system()

        _stability_enhancer = SystemStabilityEnhancer(config)
        logger.info("系统稳定性增强器已初始化")

    return _stability_enhancer


# 导出主要类和函数
__all__ = [
    'StabilityLevel',
    'RecoveryStrategy',
    'StabilityConfig',
    'StabilityMetrics',
    'CircuitBreakerState',
    'CircuitBreaker',
    'RetryManager',
    'HealthChecker',
    'SystemStabilityEnhancer',
    'get_stability_enhancer',
    'initialize_stability_enhancer'
]
