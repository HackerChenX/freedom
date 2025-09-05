"""
统一性能监控系统
整合高级性能监控、指标优化、系统资源监控的统一管理平台
"""

import time
import threading
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging

from utils.advanced_performance_monitor import get_performance_analyzer, AdvancedPerformanceAnalyzer
from utils.system_resource_monitor import get_resource_monitor, SystemResourceMonitor
from indicators.performance_optimization_engine import get_batch_optimizer, BatchCalculationOptimizer
from utils.enhanced_exception_handler import exception_handler, ErrorSeverity, ErrorCategory
from config.unified_config_manager import get_config

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSystemStatus:
    """性能系统状态"""
    timestamp: datetime
    performance_analyzer_active: bool
    resource_monitor_active: bool
    batch_optimizer_active: bool
    total_performance_records: int
    total_resource_records: int
    active_optimization_tasks: int
    system_health_score: float
    recommendations_count: int


class UnifiedPerformanceSystem:
    """
    统一性能监控系统
    
    功能：
    - 整合所有性能监控组件
    - 提供统一的性能报告
    - 自动优化建议
    - 性能基准测试
    """
    
    def __init__(self, 
                 enable_auto_optimization: bool = True,
                 report_interval: int = 3600,  # 1小时
                 enable_benchmarking: bool = True):
        """
        初始化统一性能监控系统
        
        Args:
            enable_auto_optimization: 是否启用自动优化
            report_interval: 报告生成间隔（秒）
            enable_benchmarking: 是否启用基准测试
        """
        self.enable_auto_optimization = enable_auto_optimization
        self.report_interval = report_interval
        self.enable_benchmarking = enable_benchmarking
        
        # 获取各个组件
        self.performance_analyzer = get_performance_analyzer()
        self.resource_monitor = get_resource_monitor()
        self.batch_optimizer = get_batch_optimizer()
        
        # 系统状态
        self.system_active = False
        self.last_report_time = None
        self.performance_history = []
        
        # 报告配置
        self.report_config = {
            'output_dir': get_config('performance.report_dir', 'reports/performance'),
            'auto_generate': get_config('performance.auto_generate_reports', True),
            'include_charts': get_config('performance.include_charts', False)
        }
        
        # 确保报告目录存在
        Path(self.report_config['output_dir']).mkdir(parents=True, exist_ok=True)
        
        # 基准测试结果
        self.benchmark_results = {}
        
        # 线程控制
        self.monitor_thread = None
        self.lock = threading.RLock()
        
        logger.info("统一性能监控系统初始化完成")
    
    def start_system(self):
        """启动性能监控系统"""
        if self.system_active:
            logger.warning("性能监控系统已在运行中")
            return
        
        try:
            # 启动各个组件
            self.resource_monitor.start_monitoring()
            
            # 启动系统监控线程
            self.system_active = True
            self.monitor_thread = threading.Thread(target=self._system_monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            logger.info("统一性能监控系统已启动")
            
            # 运行初始基准测试
            if self.enable_benchmarking:
                self._run_initial_benchmark()
            
        except Exception as e:
            logger.error(f"启动性能监控系统失败: {e}")
            self.system_active = False
            raise
    
    def stop_system(self):
        """停止性能监控系统"""
        self.system_active = False
        
        # 停止各个组件
        self.resource_monitor.stop_monitoring()
        
        # 等待监控线程结束
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("统一性能监控系统已停止")
    
    def _system_monitoring_loop(self):
        """系统监控循环"""
        while self.system_active:
            try:
                # 收集系统状态
                status = self._collect_system_status()
                
                with self.lock:
                    self.performance_history.append(status)
                    
                    # 保留最近24小时的数据
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    self.performance_history = [
                        s for s in self.performance_history 
                        if s.timestamp > cutoff_time
                    ]
                
                # 检查是否需要生成报告
                if self._should_generate_report():
                    self._generate_performance_report()
                
                # 自动优化检查
                if self.enable_auto_optimization:
                    self._check_auto_optimization()
                
                time.sleep(300)  # 每5分钟检查一次
                
            except Exception as e:
                logger.error(f"系统监控循环出错: {e}")
                time.sleep(60)
    
    def _collect_system_status(self) -> PerformanceSystemStatus:
        """收集系统状态"""
        try:
            # 获取各组件状态
            resource_status = self.resource_monitor.get_current_status()
            performance_report = self.performance_analyzer.get_performance_report(1)  # 最近1小时
            optimization_stats = self.batch_optimizer.get_optimization_stats()
            
            # 计算系统健康分数
            health_score = self._calculate_health_score(resource_status, performance_report)
            
            # 获取优化建议数量
            recommendations = self.performance_analyzer.get_optimization_recommendations()
            
            return PerformanceSystemStatus(
                timestamp=datetime.now(),
                performance_analyzer_active=True,
                resource_monitor_active=self.resource_monitor.monitoring_active,
                batch_optimizer_active=True,
                total_performance_records=len(self.performance_analyzer.metrics),
                total_resource_records=len(self.resource_monitor.metrics_history),
                active_optimization_tasks=optimization_stats['task_stats']['active_tasks'],
                system_health_score=health_score,
                recommendations_count=len(recommendations)
            )
            
        except Exception as e:
            logger.error(f"收集系统状态失败: {e}")
            return PerformanceSystemStatus(
                timestamp=datetime.now(),
                performance_analyzer_active=False,
                resource_monitor_active=False,
                batch_optimizer_active=False,
                total_performance_records=0,
                total_resource_records=0,
                active_optimization_tasks=0,
                system_health_score=0.0,
                recommendations_count=0
            )
    
    def _calculate_health_score(self, resource_status: Dict, performance_report: Dict) -> float:
        """计算系统健康分数"""
        try:
            score = 100.0
            
            # 资源使用评分
            if 'cpu' in resource_status:
                cpu_percent = resource_status['cpu']['percent']
                if cpu_percent > 90:
                    score -= 20
                elif cpu_percent > 80:
                    score -= 10
                elif cpu_percent > 70:
                    score -= 5
            
            if 'memory' in resource_status:
                memory_percent = resource_status['memory']['percent']
                if memory_percent > 90:
                    score -= 20
                elif memory_percent > 80:
                    score -= 10
                elif memory_percent > 70:
                    score -= 5
            
            if 'disk' in resource_status:
                disk_percent = resource_status['disk']['percent']
                if disk_percent > 95:
                    score -= 15
                elif disk_percent > 85:
                    score -= 8
                elif disk_percent > 75:
                    score -= 3
            
            # 性能评分
            if 'success_rate' in performance_report:
                success_rate = performance_report['success_rate']
                if success_rate < 90:
                    score -= (100 - success_rate) * 0.5
            
            if 'avg_execution_time' in performance_report:
                avg_time = performance_report['avg_execution_time']
                if avg_time > 5.0:
                    score -= min(20, (avg_time - 5.0) * 2)
            
            if 'bottlenecks_detected' in performance_report:
                bottlenecks = performance_report['bottlenecks_detected']
                score -= min(15, bottlenecks * 2)
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"计算健康分数失败: {e}")
            return 50.0  # 默认中等分数
    
    def _should_generate_report(self) -> bool:
        """检查是否应该生成报告"""
        if not self.report_config['auto_generate']:
            return False
        
        if self.last_report_time is None:
            return True
        
        time_since_last = (datetime.now() - self.last_report_time).total_seconds()
        return time_since_last >= self.report_interval
    
    def _generate_performance_report(self):
        """生成性能报告"""
        try:
            # 收集所有数据
            resource_status = self.resource_monitor.get_current_status()
            performance_report = self.performance_analyzer.get_performance_report(24)  # 24小时
            optimization_stats = self.batch_optimizer.get_optimization_stats()
            resource_trends = self.resource_monitor.get_resource_trends(24)
            recommendations = self.performance_analyzer.get_optimization_recommendations()
            
            # 生成报告内容
            report_data = {
                'report_info': {
                    'generated_at': datetime.now().isoformat(),
                    'report_type': 'unified_performance',
                    'time_range_hours': 24
                },
                'system_status': resource_status,
                'performance_analysis': performance_report,
                'optimization_stats': optimization_stats,
                'resource_trends': {k: asdict(v) for k, v in resource_trends.items()} if isinstance(resource_trends, dict) else resource_trends,
                'recommendations': recommendations,
                'benchmark_results': self.benchmark_results,
                'system_health': {
                    'current_score': self._calculate_health_score(resource_status, performance_report),
                    'trend': self._calculate_health_trend()
                }
            }
            
            # 保存报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"unified_performance_report_{timestamp}.json"
            filepath = Path(self.report_config['output_dir']) / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.last_report_time = datetime.now()
            logger.info(f"性能报告已生成: {filepath}")
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
    
    def _calculate_health_trend(self) -> str:
        """计算健康趋势"""
        with self.lock:
            if len(self.performance_history) < 5:
                return 'insufficient_data'
            
            recent_scores = [s.system_health_score for s in self.performance_history[-5:]]
            
            if len(recent_scores) < 2:
                return 'stable'
            
            # 简单趋势分析
            first_half = recent_scores[:len(recent_scores)//2]
            second_half = recent_scores[len(recent_scores)//2:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            diff = second_avg - first_avg
            
            if diff > 5:
                return 'improving'
            elif diff < -5:
                return 'declining'
            else:
                return 'stable'
    
    def _check_auto_optimization(self):
        """检查自动优化机会"""
        try:
            recommendations = self.performance_analyzer.get_optimization_recommendations()
            
            # 自动应用低风险优化
            for rec in recommendations:
                if rec.get('priority') == 'high' and rec.get('impact_score', 0) > 3.0:
                    logger.info(f"发现高优先级优化机会: {rec.get('target')} - {rec.get('issue')}")
                    # 这里可以实现自动优化逻辑
                    
        except Exception as e:
            logger.error(f"自动优化检查失败: {e}")
    
    def _run_initial_benchmark(self):
        """运行初始基准测试"""
        try:
            logger.info("开始运行性能基准测试...")
            
            # CPU基准测试
            cpu_score = self._benchmark_cpu()
            
            # 内存基准测试
            memory_score = self._benchmark_memory()
            
            # 磁盘基准测试
            disk_score = self._benchmark_disk()
            
            self.benchmark_results = {
                'cpu_score': cpu_score,
                'memory_score': memory_score,
                'disk_score': disk_score,
                'overall_score': (cpu_score + memory_score + disk_score) / 3,
                'benchmark_time': datetime.now().isoformat()
            }
            
            logger.info(f"基准测试完成 - 总分: {self.benchmark_results['overall_score']:.2f}")
            
        except Exception as e:
            logger.error(f"基准测试失败: {e}")
    
    def _benchmark_cpu(self) -> float:
        """CPU基准测试"""
        try:
            import math
            
            start_time = time.time()
            
            # 简单的CPU密集型计算
            result = 0
            for i in range(1000000):
                result += math.sqrt(i)
            
            cpu_time = time.time() - start_time
            
            # 分数计算（越快分数越高）
            score = max(0, 100 - cpu_time * 10)
            return min(100, score)
            
        except Exception as e:
            logger.error(f"CPU基准测试失败: {e}")
            return 50.0
    
    def _benchmark_memory(self) -> float:
        """内存基准测试"""
        try:
            start_time = time.time()
            
            # 内存分配和访问测试
            data = []
            for i in range(100000):
                data.append([j for j in range(100)])
            
            # 访问数据
            total = sum(sum(row) for row in data)
            
            memory_time = time.time() - start_time
            
            # 分数计算
            score = max(0, 100 - memory_time * 5)
            return min(100, score)
            
        except Exception as e:
            logger.error(f"内存基准测试失败: {e}")
            return 50.0
    
    def _benchmark_disk(self) -> float:
        """磁盘基准测试"""
        try:
            import tempfile
            import os
            
            start_time = time.time()
            
            # 创建临时文件进行读写测试
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                # 写入测试
                test_data = b'0' * 1024 * 1024  # 1MB
                for _ in range(10):
                    tmp_file.write(test_data)
                tmp_file.flush()
                
                # 读取测试
                tmp_file.seek(0)
                while tmp_file.read(1024 * 1024):
                    pass
            
            # 清理
            os.unlink(tmp_file.name)
            
            disk_time = time.time() - start_time
            
            # 分数计算
            score = max(0, 100 - disk_time * 2)
            return min(100, score)
            
        except Exception as e:
            logger.error(f"磁盘基准测试失败: {e}")
            return 50.0
    
    def get_system_overview(self) -> Dict[str, Any]:
        """获取系统概览"""
        try:
            current_status = self._collect_system_status()
            resource_status = self.resource_monitor.get_current_status()
            
            return {
                'system_active': self.system_active,
                'current_status': asdict(current_status),
                'resource_summary': resource_status,
                'benchmark_results': self.benchmark_results,
                'configuration': {
                    'auto_optimization': self.enable_auto_optimization,
                    'report_interval': self.report_interval,
                    'benchmarking': self.enable_benchmarking
                }
            }
            
        except Exception as e:
            logger.error(f"获取系统概览失败: {e}")
            return {'error': str(e)}
    
    def run_performance_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """运行性能分析"""
        try:
            # 获取各组件的分析结果
            performance_report = self.performance_analyzer.get_performance_report(hours)
            resource_trends = self.resource_monitor.get_resource_trends(hours)
            optimization_stats = self.batch_optimizer.get_optimization_stats()
            recommendations = self.performance_analyzer.get_optimization_recommendations()
            
            # 综合分析
            analysis = {
                'analysis_time': datetime.now().isoformat(),
                'time_range_hours': hours,
                'performance_summary': performance_report,
                'resource_trends': {k: asdict(v) for k, v in resource_trends.items()} if isinstance(resource_trends, dict) else resource_trends,
                'optimization_summary': optimization_stats,
                'top_recommendations': recommendations[:10],  # 前10个建议
                'system_health': {
                    'current_score': self._calculate_health_score(
                        self.resource_monitor.get_current_status(),
                        performance_report
                    ),
                    'trend': self._calculate_health_trend()
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"性能分析失败: {e}")
            return {'error': str(e)}


# 全局统一性能系统实例
_unified_performance_system = None
_system_lock = threading.Lock()


def get_unified_performance_system() -> UnifiedPerformanceSystem:
    """获取全局统一性能监控系统实例"""
    global _unified_performance_system
    
    if _unified_performance_system is None:
        with _system_lock:
            if _unified_performance_system is None:
                _unified_performance_system = UnifiedPerformanceSystem()
    
    return _unified_performance_system


# 导出主要类
__all__ = [
    'UnifiedPerformanceSystem',
    'PerformanceSystemStatus',
    'get_unified_performance_system'
]
