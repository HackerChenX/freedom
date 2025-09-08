"""
系统健康检查工具

提供全面的系统健康检查、诊断和修复建议
"""

import os
import sys
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from utils.unified_error_handler import get_error_handler, ErrorCategory, ErrorSeverity
from utils.unified_performance_monitor import get_performance_monitor

logger = get_logger(__name__)

# 可选导入
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    component: str
    status: HealthStatus
    score: float  # 0-100
    message: str
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime


class SystemHealthChecker:
    """系统健康检查器"""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: List[HealthCheckResult] = []
        self.check_intervals: Dict[str, int] = {}  # 检查间隔（秒）
        self.last_check_times: Dict[str, datetime] = {}
        
        # 注册默认健康检查
        self._register_default_checks()
    
    def _register_default_checks(self):
        """注册默认健康检查"""
        self.register_health_check("system_resources", self._check_system_resources, 60)
        self.register_health_check("database_connection", self._check_database_connection, 300)
        self.register_health_check("indicator_registry", self._check_indicator_registry, 600)
        self.register_health_check("error_rates", self._check_error_rates, 180)
        self.register_health_check("performance_metrics", self._check_performance_metrics, 120)
        self.register_health_check("disk_space", self._check_disk_space, 300)
        self.register_health_check("memory_leaks", self._check_memory_leaks, 600)
    
    def register_health_check(self, name: str, check_func: Callable, interval: int = 300):
        """
        注册健康检查
        
        Args:
            name: 检查名称
            check_func: 检查函数
            interval: 检查间隔（秒）
        """
        self.health_checks[name] = check_func
        self.check_intervals[name] = interval
        logger.info(f"注册健康检查: {name} (间隔: {interval}秒)")
    
    def run_health_check(self, check_name: str) -> HealthCheckResult:
        """运行单个健康检查"""
        if check_name not in self.health_checks:
            return HealthCheckResult(
                component=check_name,
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message=f"未知的健康检查: {check_name}",
                details={},
                recommendations=[],
                timestamp=datetime.now()
            )
        
        try:
            result = self.health_checks[check_name]()
            self.last_check_times[check_name] = datetime.now()
            self.health_history.append(result)
            
            # 限制历史记录数量
            if len(self.health_history) > 1000:
                self.health_history = self.health_history[-1000:]
            
            return result
        except Exception as e:
            logger.error(f"健康检查 {check_name} 执行失败: {e}")
            return HealthCheckResult(
                component=check_name,
                status=HealthStatus.CRITICAL,
                score=0.0,
                message=f"健康检查执行失败: {str(e)}",
                details={"error": str(e)},
                recommendations=["检查健康检查函数实现"],
                timestamp=datetime.now()
            )
    
    def run_all_health_checks(self, force: bool = False) -> Dict[str, HealthCheckResult]:
        """运行所有健康检查"""
        results = {}
        current_time = datetime.now()
        
        for check_name in self.health_checks:
            # 检查是否需要运行
            if not force:
                last_check = self.last_check_times.get(check_name)
                interval = self.check_intervals.get(check_name, 300)
                
                if last_check and (current_time - last_check).seconds < interval:
                    continue
            
            results[check_name] = self.run_health_check(check_name)
        
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """获取整体健康状态"""
        # 运行所有健康检查
        check_results = self.run_all_health_checks()
        
        if not check_results:
            # 使用最近的检查结果
            recent_results = {}
            for result in reversed(self.health_history[-len(self.health_checks):]):
                if result.component not in recent_results:
                    recent_results[result.component] = result
            check_results = recent_results
        
        if not check_results:
            return {
                "overall_status": HealthStatus.UNKNOWN,
                "overall_score": 0.0,
                "message": "没有可用的健康检查结果",
                "components": {},
                "timestamp": datetime.now().isoformat()
            }
        
        # 计算整体评分
        total_score = sum(result.score for result in check_results.values())
        overall_score = total_score / len(check_results)
        
        # 确定整体状态
        critical_count = sum(1 for r in check_results.values() if r.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for r in check_results.values() if r.status == HealthStatus.WARNING)
        
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_count > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # 收集所有建议
        all_recommendations = []
        for result in check_results.values():
            all_recommendations.extend(result.recommendations)
        
        return {
            "overall_status": overall_status,
            "overall_score": overall_score,
            "message": f"系统整体健康评分: {overall_score:.1f}/100",
            "components": {name: {
                "status": result.status.value,
                "score": result.score,
                "message": result.message,
                "details": result.details
            } for name, result in check_results.items()},
            "recommendations": list(set(all_recommendations)),
            "critical_issues": critical_count,
            "warning_issues": warning_count,
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_system_resources(self) -> HealthCheckResult:
        """检查系统资源"""
        if not HAS_PSUTIL:
            return HealthCheckResult(
                component="system_resources",
                status=HealthStatus.WARNING,
                score=50.0,
                message="psutil未安装，无法检查系统资源",
                details={},
                recommendations=["安装psutil库以获得完整的系统监控"],
                timestamp=datetime.now()
            )
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # 计算评分
            score = 100.0
            recommendations = []
            
            if cpu_percent > 90:
                score -= 40
                recommendations.append("CPU使用率过高，考虑优化或增加资源")
            elif cpu_percent > 70:
                score -= 20
                recommendations.append("CPU使用率较高，建议监控")
            
            if memory.percent > 90:
                score -= 40
                recommendations.append("内存使用率过高，检查内存泄漏")
            elif memory.percent > 70:
                score -= 20
                recommendations.append("内存使用率较高，建议监控")
            
            # 确定状态
            if score >= 80:
                status = HealthStatus.HEALTHY
            elif score >= 60:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL
            
            return HealthCheckResult(
                component="system_resources",
                status=status,
                score=max(0, score),
                message=f"CPU: {cpu_percent:.1f}%, 内存: {memory.percent:.1f}%",
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / 1024 / 1024 / 1024
                },
                recommendations=recommendations,
                timestamp=datetime.now()
            )
        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                status=HealthStatus.CRITICAL,
                score=0.0,
                message=f"系统资源检查失败: {str(e)}",
                details={"error": str(e)},
                recommendations=["检查系统监控工具"],
                timestamp=datetime.now()
            )
    
    def _check_database_connection(self) -> HealthCheckResult:
        """检查数据库连接"""
        try:
            from db.enhanced_connection_pool import ClickHouseConnectionPool
            
            pool = ClickHouseConnectionPool()
            with pool.get_connection() as conn:
                # 执行简单查询测试连接
                result = conn.execute("SELECT 1")
                
            return HealthCheckResult(
                component="database_connection",
                status=HealthStatus.HEALTHY,
                score=100.0,
                message="数据库连接正常",
                details={"connection_test": "success"},
                recommendations=[],
                timestamp=datetime.now()
            )
        except Exception as e:
            return HealthCheckResult(
                component="database_connection",
                status=HealthStatus.CRITICAL,
                score=0.0,
                message=f"数据库连接失败: {str(e)}",
                details={"error": str(e)},
                recommendations=["检查数据库服务状态", "验证连接配置"],
                timestamp=datetime.now()
            )
    
    def _check_indicator_registry(self) -> HealthCheckResult:
        """检查指标注册表"""
        try:
            from indicators.complete_indicator_registry import get_indicator_registry
            
            registry = get_indicator_registry()
            total_indicators = len(registry.get_all_indicators())
            
            if total_indicators >= 100:
                score = 100.0
                status = HealthStatus.HEALTHY
                message = f"指标注册表正常，共 {total_indicators} 个指标"
                recommendations = []
            elif total_indicators >= 50:
                score = 80.0
                status = HealthStatus.WARNING
                message = f"指标数量较少，共 {total_indicators} 个指标"
                recommendations = ["考虑增加更多技术指标"]
            else:
                score = 40.0
                status = HealthStatus.CRITICAL
                message = f"指标数量过少，共 {total_indicators} 个指标"
                recommendations = ["检查指标注册表配置", "重新加载指标"]
            
            return HealthCheckResult(
                component="indicator_registry",
                status=status,
                score=score,
                message=message,
                details={"total_indicators": total_indicators},
                recommendations=recommendations,
                timestamp=datetime.now()
            )
        except Exception as e:
            return HealthCheckResult(
                component="indicator_registry",
                status=HealthStatus.CRITICAL,
                score=0.0,
                message=f"指标注册表检查失败: {str(e)}",
                details={"error": str(e)},
                recommendations=["检查指标注册表模块"],
                timestamp=datetime.now()
            )
    
    def _check_error_rates(self) -> HealthCheckResult:
        """检查错误率"""
        try:
            error_handler = get_error_handler()
            stats = error_handler.get_error_statistics()
            
            total_errors = stats.get("total_errors", 0)
            recent_errors = stats.get("recent_errors", 0)
            
            # 计算评分
            if recent_errors == 0:
                score = 100.0
                status = HealthStatus.HEALTHY
                message = "最近1小时无错误"
                recommendations = []
            elif recent_errors <= 5:
                score = 80.0
                status = HealthStatus.WARNING
                message = f"最近1小时有 {recent_errors} 个错误"
                recommendations = ["监控错误趋势"]
            else:
                score = 40.0
                status = HealthStatus.CRITICAL
                message = f"最近1小时有 {recent_errors} 个错误，总计 {total_errors} 个"
                recommendations = ["检查错误日志", "修复高频错误"]
            
            return HealthCheckResult(
                component="error_rates",
                status=status,
                score=score,
                message=message,
                details=stats,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
        except Exception as e:
            return HealthCheckResult(
                component="error_rates",
                status=HealthStatus.WARNING,
                score=50.0,
                message=f"错误率检查失败: {str(e)}",
                details={"error": str(e)},
                recommendations=["检查错误处理系统"],
                timestamp=datetime.now()
            )
    
    def _check_performance_metrics(self) -> HealthCheckResult:
        """检查性能指标"""
        try:
            monitor = get_performance_monitor()
            report = monitor.get_performance_report(hours=1)
            
            slow_functions = len(report.get("slow_functions", []))
            total_functions = report.get("total_functions_monitored", 0)
            
            if total_functions == 0:
                score = 50.0
                status = HealthStatus.WARNING
                message = "没有性能监控数据"
                recommendations = ["启动性能监控"]
            elif slow_functions == 0:
                score = 100.0
                status = HealthStatus.HEALTHY
                message = "所有函数性能正常"
                recommendations = []
            elif slow_functions <= total_functions * 0.1:
                score = 80.0
                status = HealthStatus.WARNING
                message = f"有 {slow_functions} 个慢函数"
                recommendations = ["优化慢函数性能"]
            else:
                score = 40.0
                status = HealthStatus.CRITICAL
                message = f"有 {slow_functions} 个慢函数，占比过高"
                recommendations = ["紧急优化性能", "检查系统负载"]
            
            return HealthCheckResult(
                component="performance_metrics",
                status=status,
                score=score,
                message=message,
                details={
                    "slow_functions": slow_functions,
                    "total_functions": total_functions,
                    "performance_summary": report.get("performance_summary", {})
                },
                recommendations=recommendations,
                timestamp=datetime.now()
            )
        except Exception as e:
            return HealthCheckResult(
                component="performance_metrics",
                status=HealthStatus.WARNING,
                score=50.0,
                message=f"性能指标检查失败: {str(e)}",
                details={"error": str(e)},
                recommendations=["检查性能监控系统"],
                timestamp=datetime.now()
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """检查磁盘空间"""
        if not HAS_PSUTIL:
            return HealthCheckResult(
                component="disk_space",
                status=HealthStatus.WARNING,
                score=50.0,
                message="无法检查磁盘空间",
                details={},
                recommendations=["安装psutil库"],
                timestamp=datetime.now()
            )
        
        try:
            disk = psutil.disk_usage('/')
            usage_percent = disk.percent
            
            if usage_percent < 70:
                score = 100.0
                status = HealthStatus.HEALTHY
                message = f"磁盘使用率正常: {usage_percent:.1f}%"
                recommendations = []
            elif usage_percent < 85:
                score = 70.0
                status = HealthStatus.WARNING
                message = f"磁盘使用率较高: {usage_percent:.1f}%"
                recommendations = ["清理临时文件", "监控磁盘使用"]
            else:
                score = 30.0
                status = HealthStatus.CRITICAL
                message = f"磁盘使用率过高: {usage_percent:.1f}%"
                recommendations = ["立即清理磁盘空间", "扩展存储容量"]
            
            return HealthCheckResult(
                component="disk_space",
                status=status,
                score=score,
                message=message,
                details={
                    "usage_percent": usage_percent,
                    "free_gb": disk.free / 1024 / 1024 / 1024,
                    "total_gb": disk.total / 1024 / 1024 / 1024
                },
                recommendations=recommendations,
                timestamp=datetime.now()
            )
        except Exception as e:
            return HealthCheckResult(
                component="disk_space",
                status=HealthStatus.WARNING,
                score=50.0,
                message=f"磁盘空间检查失败: {str(e)}",
                details={"error": str(e)},
                recommendations=["检查磁盘监控工具"],
                timestamp=datetime.now()
            )
    
    def _check_memory_leaks(self) -> HealthCheckResult:
        """检查内存泄漏"""
        if not HAS_PSUTIL:
            return HealthCheckResult(
                component="memory_leaks",
                status=HealthStatus.WARNING,
                score=50.0,
                message="无法检查内存泄漏",
                details={},
                recommendations=["安装psutil库"],
                timestamp=datetime.now()
            )
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # 简单的内存泄漏检测（基于内存使用量）
            if memory_mb < 500:
                score = 100.0
                status = HealthStatus.HEALTHY
                message = f"内存使用正常: {memory_mb:.1f}MB"
                recommendations = []
            elif memory_mb < 1000:
                score = 80.0
                status = HealthStatus.WARNING
                message = f"内存使用较高: {memory_mb:.1f}MB"
                recommendations = ["监控内存使用趋势"]
            else:
                score = 40.0
                status = HealthStatus.CRITICAL
                message = f"内存使用过高: {memory_mb:.1f}MB，可能存在内存泄漏"
                recommendations = ["检查内存泄漏", "重启应用程序"]
            
            return HealthCheckResult(
                component="memory_leaks",
                status=status,
                score=score,
                message=message,
                details={
                    "memory_mb": memory_mb,
                    "memory_percent": process.memory_percent()
                },
                recommendations=recommendations,
                timestamp=datetime.now()
            )
        except Exception as e:
            return HealthCheckResult(
                component="memory_leaks",
                status=HealthStatus.WARNING,
                score=50.0,
                message=f"内存泄漏检查失败: {str(e)}",
                details={"error": str(e)},
                recommendations=["检查内存监控工具"],
                timestamp=datetime.now()
            )


# 全局健康检查器实例
_health_checker = SystemHealthChecker()


def get_health_checker() -> SystemHealthChecker:
    """获取全局健康检查器实例"""
    return _health_checker


def health_check(component_name: str = None):
    """
    健康检查装饰器
    
    Args:
        component_name: 组件名称
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            checker = get_health_checker()
            
            # 如果指定了组件名称，运行对应的健康检查
            if component_name and component_name in checker.health_checks:
                result = checker.run_health_check(component_name)
                if result.status == HealthStatus.CRITICAL:
                    logger.error(f"组件 {component_name} 健康检查失败: {result.message}")
                    # 可以选择是否继续执行
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
