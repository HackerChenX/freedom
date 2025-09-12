"""
性能优化和稳定性提升主控制器
统一管理性能优化和稳定性增强功能
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from strategy.performance_stability_optimizer import (
    PerformanceStabilityOptimizer, 
    PerformanceOptimizationConfig,
    get_performance_optimizer,
    initialize_performance_optimizer
)
from strategy.system_stability_enhancer import (
    SystemStabilityEnhancer,
    StabilityConfig,
    get_stability_enhancer,
    initialize_stability_enhancer
)
from utils.enhanced_performance_monitor import performance_monitor
from utils.enhanced_exception_handler import exception_handler, ErrorSeverity, ErrorCategory
from utils.unified_container import get_container
from config.unified_config_manager import get_config

logger = logging.getLogger(__name__)


@dataclass
class PerformanceStabilityReport:
    """性能稳定性报告"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 性能指标
    performance_score: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    response_time_avg: float = 0.0
    
    # 稳定性指标
    stability_score: float = 0.0
    error_rate: float = 0.0
    health_score: float = 0.0
    circuit_breaker_status: Dict[str, str] = field(default_factory=dict)
    
    # 优化结果
    optimization_results: List[Dict[str, Any]] = field(default_factory=list)
    improvement_summary: Dict[str, float] = field(default_factory=dict)
    
    # 建议
    recommendations: List[str] = field(default_factory=list)
    
    # 总体评估
    overall_status: str = "unknown"
    overall_score: float = 0.0


class PerformanceStabilityController:
    """性能优化和稳定性提升主控制器"""
    
    def __init__(self, 
                 performance_config: Optional[PerformanceOptimizationConfig] = None,
                 stability_config: Optional[StabilityConfig] = None):
        self.performance_config = performance_config or PerformanceOptimizationConfig()
        self.stability_config = stability_config or StabilityConfig()
        
        # 初始化组件
        self.performance_optimizer = initialize_performance_optimizer(self.performance_config)
        self.stability_enhancer = initialize_stability_enhancer(self.stability_config)
        
        self.reports_history = []
        self.is_running = False
        self.monitoring_thread = None
        self.lock = threading.Lock()
        
        # 注册到容器
        container = get_container()
        container.register(PerformanceStabilityController, instance=self)
    
    @performance_monitor(threshold_seconds=10.0)
    @exception_handler(reraise=True)
    def start_system(self) -> Dict[str, Any]:
        """启动性能优化和稳定性系统"""
        if self.is_running:
            return {"status": "already_running"}
        
        start_time = time.time()
        results = {}
        
        try:
            # 启动性能优化系统
            perf_result = self.performance_optimizer.start_optimization_system()
            results["performance_system"] = perf_result
            
            # 启动稳定性增强系统
            stability_result = self.stability_enhancer.start_stability_system()
            results["stability_system"] = stability_result
            
            # 启动监控
            self._start_monitoring()
            
            self.is_running = True
            
            # 生成初始报告
            initial_report = self.generate_comprehensive_report()
            results["initial_report"] = initial_report
            
            return {
                "status": "started",
                "startup_time": time.time() - start_time,
                "components": results,
                "overall_score": initial_report.overall_score
            }
            
        except Exception as e:
            logger.error(f"启动性能稳定性系统失败: {e}")
            raise
    
    def stop_system(self) -> Dict[str, Any]:
        """停止性能优化和稳定性系统"""
        if not self.is_running:
            return {"status": "not_running"}
        
        # 停止监控
        self._stop_monitoring()
        
        # 停止组件
        perf_result = self.performance_optimizer.stop_optimization_system()
        stability_result = self.stability_enhancer.stop_stability_system()
        
        self.is_running = False
        
        # 生成最终报告
        final_report = self.generate_comprehensive_report()
        
        return {
            "status": "stopped",
            "performance_system": perf_result,
            "stability_system": stability_result,
            "final_report": final_report,
            "total_reports": len(self.reports_history)
        }
    
    @performance_monitor(threshold_seconds=30.0)
    @exception_handler(reraise=True)
    def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """运行全面优化"""
        if not self.is_running:
            raise RuntimeError("系统未启动")
        
        start_time = time.time()
        
        # 生成优化前报告
        before_report = self.generate_comprehensive_report()
        
        # 运行性能优化
        performance_results = self.performance_optimizer.run_comprehensive_optimization()
        
        # 等待一段时间让系统稳定
        time.sleep(5)
        
        # 生成优化后报告
        after_report = self.generate_comprehensive_report()
        
        # 计算改善效果
        improvement = self._calculate_improvement(before_report, after_report)
        
        return {
            "optimization_time": time.time() - start_time,
            "before_report": before_report,
            "after_report": after_report,
            "performance_results": performance_results,
            "improvement": improvement,
            "success": after_report.overall_score > before_report.overall_score
        }
    
    def generate_comprehensive_report(self) -> PerformanceStabilityReport:
        """生成综合报告"""
        try:
            # 获取性能优化报告
            performance_report = self.performance_optimizer.get_optimization_report()
            
            # 获取稳定性报告
            stability_report = self.stability_enhancer.get_stability_report()
            
            # 计算综合指标
            performance_score = self._calculate_performance_score(performance_report)
            stability_score = stability_report.get("stability_score", 0)
            overall_score = (performance_score + stability_score) / 2
            
            # 确定总体状态
            if overall_score >= 90:
                overall_status = "excellent"
            elif overall_score >= 80:
                overall_status = "good"
            elif overall_score >= 70:
                overall_status = "acceptable"
            elif overall_score >= 60:
                overall_status = "poor"
            else:
                overall_status = "critical"
            
            # 收集建议
            recommendations = []
            recommendations.extend(stability_report.get("recommendations", []))
            recommendations.extend(self._generate_performance_recommendations(performance_report))
            
            # 创建报告
            report = PerformanceStabilityReport(
                performance_score=performance_score,
                stability_score=stability_score,
                overall_score=overall_score,
                overall_status=overall_status,
                recommendations=recommendations,
                optimization_results=performance_report.get("optimization_types", []),
                circuit_breaker_status={
                    name: state.get("state", "unknown") 
                    for name, state in stability_report.get("circuit_breakers", {}).items()
                }
            )
            
            # 保存报告
            with self.lock:
                self.reports_history.append(report)
                # 保留最近100个报告
                if len(self.reports_history) > 100:
                    self.reports_history = self.reports_history[-100:]
            
            return report
            
        except Exception as e:
            logger.error(f"生成综合报告失败: {e}")
            return PerformanceStabilityReport(overall_status="error")
    
    def _calculate_performance_score(self, performance_report: Dict[str, Any]) -> float:
        """计算性能评分"""
        try:
            # 基础分数
            score = 100.0
            
            # 根据优化成功率调整
            total_optimizations = performance_report.get("total_optimizations", 0)
            successful_optimizations = performance_report.get("successful_optimizations", 0)
            
            if total_optimizations > 0:
                success_rate = successful_optimizations / total_optimizations
                score = score * success_rate
            
            # 根据健康状态调整
            health = performance_report.get("current_health", {})
            health_score = health.get("stability_score", 100)
            score = (score + health_score) / 2
            
            return max(score, 0.0)
            
        except Exception:
            return 50.0  # 默认中等分数
    
    def _generate_performance_recommendations(self, performance_report: Dict[str, Any]) -> List[str]:
        """生成性能建议"""
        recommendations = []
        
        try:
            # 检查优化成功率
            total_optimizations = performance_report.get("total_optimizations", 0)
            successful_optimizations = performance_report.get("successful_optimizations", 0)
            
            if total_optimizations > 0:
                success_rate = successful_optimizations / total_optimizations
                if success_rate < 0.8:
                    recommendations.append("优化成功率较低，建议检查系统配置和资源限制")
            
            # 检查健康状态
            health = performance_report.get("current_health", {})
            if health.get("stability_score", 100) < 80:
                recommendations.append("系统健康状态不佳，建议进行全面检查")
            
            # 检查内存使用
            memory_usage = health.get("memory_usage_mb", 0)
            if memory_usage > 1500:
                recommendations.append("内存使用过高，建议进行内存优化")
            
            # 检查CPU使用
            cpu_usage = health.get("cpu_usage", 0)
            if cpu_usage > 80:
                recommendations.append("CPU使用率过高，建议优化计算密集型操作")
                
        except Exception as e:
            logger.warning(f"生成性能建议失败: {e}")
        
        return recommendations
    
    def _calculate_improvement(self, before: PerformanceStabilityReport, 
                             after: PerformanceStabilityReport) -> Dict[str, float]:
        """计算改善效果"""
        return {
            "overall_score_improvement": after.overall_score - before.overall_score,
            "performance_score_improvement": after.performance_score - before.performance_score,
            "stability_score_improvement": after.stability_score - before.stability_score,
            "improvement_percentage": (
                (after.overall_score - before.overall_score) / before.overall_score * 100
                if before.overall_score > 0 else 0
            )
        }
    
    def _start_monitoring(self):
        """启动监控"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="performance_stability_monitor",
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("性能稳定性监控已启动")
    
    def _stop_monitoring(self):
        """停止监控"""
        # 监控线程是守护线程，会自动停止
        logger.info("性能稳定性监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 每5分钟生成一次报告
                time.sleep(300)
                
                if self.is_running:
                    report = self.generate_comprehensive_report()
                    
                    # 检查是否需要自动优化
                    if report.overall_score < 70:
                        logger.warning(f"系统性能稳定性评分较低: {report.overall_score:.1f}")
                        
                        # 如果评分过低，尝试自动优化
                        if report.overall_score < 50:
                            try:
                                logger.info("启动自动优化...")
                                self.performance_optimizer.run_comprehensive_optimization()
                            except Exception as e:
                                logger.error(f"自动优化失败: {e}")
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(60)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if not self.is_running:
            return {"status": "stopped"}
        
        latest_report = self.reports_history[-1] if self.reports_history else None
        
        return {
            "status": "running",
            "is_monitoring": self.monitoring_thread and self.monitoring_thread.is_alive(),
            "latest_report": latest_report,
            "total_reports": len(self.reports_history),
            "performance_optimizer_active": self.performance_optimizer.is_running,
            "stability_enhancer_active": self.stability_enhancer.is_active
        }
    
    def get_historical_trends(self, hours: int = 24) -> Dict[str, Any]:
        """获取历史趋势"""
        if not self.reports_history:
            return {"status": "no_data"}
        
        # 获取指定时间范围内的报告
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_reports = [
            report for report in self.reports_history
            if report.timestamp >= cutoff_time
        ]
        
        if not recent_reports:
            return {"status": "no_recent_data"}
        
        # 计算趋势
        overall_scores = [r.overall_score for r in recent_reports]
        performance_scores = [r.performance_score for r in recent_reports]
        stability_scores = [r.stability_score for r in recent_reports]
        
        return {
            "time_range_hours": hours,
            "report_count": len(recent_reports),
            "overall_score_trend": {
                "min": min(overall_scores),
                "max": max(overall_scores),
                "avg": sum(overall_scores) / len(overall_scores),
                "current": overall_scores[-1],
                "trend": "improving" if overall_scores[-1] > overall_scores[0] else "declining"
            },
            "performance_score_trend": {
                "min": min(performance_scores),
                "max": max(performance_scores),
                "avg": sum(performance_scores) / len(performance_scores),
                "current": performance_scores[-1]
            },
            "stability_score_trend": {
                "min": min(stability_scores),
                "max": max(stability_scores),
                "avg": sum(stability_scores) / len(stability_scores),
                "current": stability_scores[-1]
            }
        }


# 全局控制器实例
_performance_stability_controller = None
_controller_lock = threading.Lock()


def get_performance_stability_controller() -> PerformanceStabilityController:
    """获取全局性能稳定性控制器实例"""
    global _performance_stability_controller

    if _performance_stability_controller is None:
        with _controller_lock:
            if _performance_stability_controller is None:
                _performance_stability_controller = PerformanceStabilityController()

    return _performance_stability_controller


def initialize_performance_stability_controller(
    performance_config: Optional[PerformanceOptimizationConfig] = None,
    stability_config: Optional[StabilityConfig] = None
) -> PerformanceStabilityController:
    """初始化性能稳定性控制器"""
    global _performance_stability_controller

    with _controller_lock:
        if _performance_stability_controller is not None:
            _performance_stability_controller.stop_system()

        _performance_stability_controller = PerformanceStabilityController(
            performance_config, stability_config
        )
        logger.info("性能稳定性控制器已初始化")

    return _performance_stability_controller


# 导出主要类和函数
__all__ = [
    'PerformanceStabilityReport',
    'PerformanceStabilityController',
    'get_performance_stability_controller',
    'initialize_performance_stability_controller'
]
