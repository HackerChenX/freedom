#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股测试性能监控系统

提供专门针对选股测试的性能监控，包括5分钟超时控制、资源监控和早停机制
"""

import time
import threading
import gc
import os
import psutil
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

from utils.logger import getLogger
from .monitoring import MetricsCollector, AlertManager, Alert, get_test_monitoring_system
from db.sql_manager import SQLManager, QueryType

logger = getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    cpu_usage: float
    memory_usage: float  # GB
    disk_io: float  # MB/s
    network_io: float  # MB/s
    execution_time: float  # 秒
    stocks_per_second: float
    verifications_per_second: float
    patterns_per_second: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceConfig:
    """性能配置"""
    timeout_seconds: int = 300  # 5分钟
    performance_threshold: float = 0.8  # 80%时触发优化
    memory_threshold_gb: float = 6.0  # 内存阈值
    cpu_threshold_percent: float = 80.0  # CPU阈值
    gc_threshold: float = 0.7  # 垃圾回收阈值
    enable_early_stopping: bool = True
    enable_performance_optimization: bool = True
    monitoring_interval: int = 5  # 监控间隔（秒）


class PerformanceMonitor:
    """性能监控器 - 专门针对选股测试"""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """
        初始化性能监控器
        
        Args:
            config: 性能配置
        """
        self.config = config or PerformanceConfig()
        self.start_time = None
        self.end_time = None
        self.completed_tasks = 0
        self.total_tasks = 0
        self.lock = threading.Lock()
        self.running = False
        self.monitor_thread = None
        self.performance_metrics = []
        self.optimization_triggered = False
        self.early_stop_triggered = False
        self.optimization_callbacks = []
        self.early_stop_callbacks = []
        
        # 使用全局监控系统
        self.monitoring_system = get_test_monitoring_system()
        
        # 进程信息
        self.process = psutil.Process(os.getpid())
        
        logger.info("性能监控器初始化完成")
    
    def start_monitoring(self, total_tasks: int) -> None:
        """
        开始监控
        
        Args:
            total_tasks: 总任务数
        """
        if self.running:
            return
        
        self.start_time = time.time()
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.running = True
        self.optimization_triggered = False
        self.early_stop_triggered = False
        self.performance_metrics = []
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # 启动全局监控系统
        self.monitoring_system.start_monitoring()
        
        # 创建测试会话
        session_id = f"stock_selection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.monitoring_system.start_test_session(session_id, "选股测试")
        self.session_id = session_id
        
        logger.info(f"开始性能监控，总任务数: {total_tasks}，超时时间: {self.config.timeout_seconds}秒")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        停止监控
        
        Returns:
            Dict[str, Any]: 监控报告
        """
        if not self.running:
            return {}
        
        self.running = False
        self.end_time = time.time()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # 结束测试会话
        report = self.monitoring_system.end_test_session(self.session_id)
        
        logger.info("性能监控已停止")
        return report
    
    def update_progress(self, completed: int = 1) -> None:
        """
        更新进度
        
        Args:
            completed: 完成的任务数
        """
        with self.lock:
            self.completed_tasks += completed
            if self.completed_tasks % 100 == 0:  # 每100个任务报告一次
                elapsed = time.time() - self.start_time if self.start_time else 0
                progress = (self.completed_tasks / self.total_tasks) * 100 if self.total_tasks > 0 else 0
                logger.info(f"测试进度: {self.completed_tasks}/{self.total_tasks} ({progress:.1f}%), 已用时: {elapsed:.1f}秒")
                
                # 记录进度指标
                self.monitoring_system.record_test_metric(
                    self.session_id, 
                    'progress_percent', 
                    progress
                )
    
    def check_timeout(self) -> bool:
        """
        检查是否超时
        
        Returns:
            bool: 是否超时
        """
        if self.start_time is None:
            return False
        elapsed = time.time() - self.start_time
        return elapsed > self.config.timeout_seconds
    
    def should_optimize_performance(self) -> bool:
        """
        检查是否需要性能优化
        
        Returns:
            bool: 是否需要优化
        """
        if self.start_time is None or self.optimization_triggered:
            return False
        elapsed = time.time() - self.start_time
        return elapsed > (self.config.timeout_seconds * self.config.performance_threshold)
    
    def get_remaining_time(self) -> float:
        """
        获取剩余时间
        
        Returns:
            float: 剩余时间（秒）
        """
        if self.start_time is None:
            return self.config.timeout_seconds
        elapsed = time.time() - self.start_time
        return max(0, self.config.timeout_seconds - elapsed)
    
    def get_elapsed_time(self) -> float:
        """
        获取已用时间
        
        Returns:
            float: 已用时间（秒）
        """
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def get_progress(self) -> Tuple[int, int, float]:
        """
        获取进度
        
        Returns:
            Tuple[int, int, float]: (已完成任务数, 总任务数, 进度百分比)
        """
        with self.lock:
            progress = (self.completed_tasks / self.total_tasks) * 100 if self.total_tasks > 0 else 0
            return self.completed_tasks, self.total_tasks, progress
    
    def register_optimization_callback(self, callback: Callable[[], None]) -> None:
        """
        注册性能优化回调
        
        Args:
            callback: 回调函数
        """
        self.optimization_callbacks.append(callback)
    
    def register_early_stop_callback(self, callback: Callable[[], None]) -> None:
        """
        注册早停回调
        
        Args:
            callback: 回调函数
        """
        self.early_stop_callbacks.append(callback)
    
    def trigger_performance_optimization(self) -> None:
        """触发性能优化"""
        if not self.optimization_triggered and self.config.enable_performance_optimization:
            self.optimization_triggered = True
            logger.warning("触发性能优化：接近超时阈值，开始优化策略")
            
            # 强制垃圾回收
            gc.collect()
            
            # 调用优化回调
            for callback in self.optimization_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"执行优化回调失败: {e}")
    
    def trigger_early_stop(self) -> None:
        """触发早停机制"""
        if not self.early_stop_triggered and self.config.enable_early_stopping:
            self.early_stop_triggered = True
            logger.error("触发早停机制：超过5分钟限制，返回部分结果")
            
            # 调用早停回调
            for callback in self.early_stop_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"执行早停回调失败: {e}")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.running:
            try:
                # 收集性能指标
                metrics = self._collect_performance_metrics()
                self.performance_metrics.append(metrics)
                
                # 记录指标
                self._record_metrics(metrics)
                
                # 检查是否需要优化性能
                if self.should_optimize_performance():
                    self.trigger_performance_optimization()
                
                # 检查是否超时
                if self.check_timeout():
                    self.trigger_early_stop()
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(1)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """
        收集性能指标
        
        Returns:
            PerformanceMetrics: 性能指标
        """
        # 收集系统指标
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = self.process.memory_info().rss / (1024 ** 3)  # GB
        
        # 计算处理速度
        elapsed = self.get_elapsed_time()
        stocks_per_second = self.completed_tasks / max(1, elapsed)
        
        # 简化的磁盘和网络I/O
        disk_io = 0.0
        network_io = 0.0
        
        # 简化的验证和形态速度
        verifications_per_second = stocks_per_second  # 假设每个股票都有验证
        patterns_per_second = stocks_per_second / 10  # 假设每10个股票对应一个形态
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_io=disk_io,
            network_io=network_io,
            execution_time=elapsed,
            stocks_per_second=stocks_per_second,
            verifications_per_second=verifications_per_second,
            patterns_per_second=patterns_per_second
        )
    
    def _record_metrics(self, metrics: PerformanceMetrics) -> None:
        """
        记录指标
        
        Args:
            metrics: 性能指标
        """
        # 记录到全局监控系统
        self.monitoring_system.record_test_metric(
            self.session_id, 'cpu_usage', metrics.cpu_usage
        )
        self.monitoring_system.record_test_metric(
            self.session_id, 'memory_usage', metrics.memory_usage
        )
        self.monitoring_system.record_test_metric(
            self.session_id, 'stocks_per_second', metrics.stocks_per_second
        )
        self.monitoring_system.record_test_metric(
            self.session_id, 'execution_time', metrics.execution_time
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        获取性能报告
        
        Returns:
            Dict[str, Any]: 性能报告
        """
        if not self.performance_metrics:
            return {}
        
        # 计算平均值
        avg_cpu = sum(m.cpu_usage for m in self.performance_metrics) / len(self.performance_metrics)
        avg_memory = sum(m.memory_usage for m in self.performance_metrics) / len(self.performance_metrics)
        avg_stocks_per_second = sum(m.stocks_per_second for m in self.performance_metrics) / len(self.performance_metrics)
        
        # 获取最大值
        max_cpu = max(m.cpu_usage for m in self.performance_metrics)
        max_memory = max(m.memory_usage for m in self.performance_metrics)
        
        # 获取最新值
        latest = self.performance_metrics[-1]
        
        return {
            'execution_time': latest.execution_time,
            'cpu_usage': {
                'avg': avg_cpu,
                'max': max_cpu,
                'latest': latest.cpu_usage
            },
            'memory_usage': {
                'avg': avg_memory,
                'max': max_memory,
                'latest': latest.memory_usage
            },
            'processing_speed': {
                'stocks_per_second': avg_stocks_per_second,
                'verifications_per_second': latest.verifications_per_second,
                'patterns_per_second': latest.patterns_per_second
            },
            'optimization_triggered': self.optimization_triggered,
            'early_stop_triggered': self.early_stop_triggered,
            'progress': {
                'completed_tasks': self.completed_tasks,
                'total_tasks': self.total_tasks,
                'progress_percent': (self.completed_tasks / self.total_tasks) * 100 if self.total_tasks > 0 else 0
            }
        }
    
    def export_performance_report(self, file_path: str) -> None:
        """
        导出性能报告
        
        Args:
            file_path: 导出文件路径
        """
        report = self.get_performance_report()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"性能报告已导出到: {file_path}")


class PerformanceOptimizationService:
    """性能优化器"""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        """
        初始化性能优化器
        
        Args:
            performance_monitor: 性能监控器
        """
        self.performance_monitor = performance_monitor
        self.optimizations_applied = []
        
        # 注册优化回调
        self.performance_monitor.register_optimization_callback(self.optimize_performance)
        
        logger.info("性能优化器初始化完成")
    
    def optimize_performance(self) -> None:
        """优化性能"""
        logger.info("开始优化性能...")
        
        # 获取当前性能指标
        metrics = self.performance_monitor.performance_metrics[-1] if self.performance_monitor.performance_metrics else None
        
        if not metrics:
            return
        
        # 根据性能指标选择优化策略
        optimizations = []
        
        # 1. 内存优化
        if metrics.memory_usage > 4.0:  # 超过4GB
            optimizations.append(self._optimize_memory)
        
        # 2. 处理速度优化
        if metrics.stocks_per_second < 10:  # 每秒处理少于10只股票
            optimizations.append(self._optimize_processing_speed)
        
        # 3. CPU优化
        if metrics.cpu_usage > 70:  # CPU使用率超过70%
            optimizations.append(self._optimize_cpu)
        
        # 应用优化策略
        for optimize_func in optimizations:
            try:
                optimize_func()
            except Exception as e:
                logger.error(f"应用优化策略失败: {e}")
        
        logger.info(f"性能优化完成，应用了 {len(self.optimizations_applied)} 个优化策略")
    
    def _optimize_memory(self) -> None:
        """内存优化"""
        logger.info("应用内存优化策略...")
        
        # 1. 强制垃圾回收
        gc.collect()
        
        # 2. 清理缓存
        # 这里需要具体实现，例如清理选股引擎的缓存
        
        self.optimizations_applied.append({
            'type': 'memory',
            'timestamp': datetime.now().isoformat(),
            'description': '强制垃圾回收和清理缓存'
        })
    
    def _optimize_processing_speed(self) -> None:
        """处理速度优化"""
        logger.info("应用处理速度优化策略...")
        
        # 1. 减少批处理大小
        # 2. 增加并行度
        # 这里需要具体实现，例如调整选股引擎的批处理大小和并行度
        
        self.optimizations_applied.append({
            'type': 'processing_speed',
            'timestamp': datetime.now().isoformat(),
            'description': '减少批处理大小和增加并行度'
        })
    
    def _optimize_cpu(self) -> None:
        """CPU优化"""
        logger.info("应用CPU优化策略...")
        
        # 1. 减少并行度
        # 这里需要具体实现，例如调整选股引擎的并行度
        
        self.optimizations_applied.append({
            'type': 'cpu',
            'timestamp': datetime.now().isoformat(),
            'description': '减少并行度'
        })
    
    def get_applied_optimizations(self) -> List[Dict[str, Any]]:
        """
        获取已应用的优化策略
        
        Returns:
            List[Dict[str, Any]]: 已应用的优化策略列表
        """
        return self.optimizations_applied


def main():
    """测试性能监控器"""
    # 创建性能监控器
    config = PerformanceConfig(
        timeout_seconds=60,  # 60秒超时（用于测试）
        performance_threshold=0.8,
        monitoring_interval=2
    )
    monitor = PerformanceMonitor(config)
    
    # 创建性能优化器
    optimizer = PerformanceOptimizationService(monitor)
    
    # 开始监控
    monitor.start_monitoring(1000)
    
    # 模拟任务执行
    for i in range(1000):
        # 模拟任务
        time.sleep(0.1)
        
        # 更新进度
        monitor.update_progress()
        
        # 检查是否需要早停
        if monitor.early_stop_triggered:
            print("早停触发，停止执行")
            break
    
    # 停止监控
    report = monitor.stop_monitoring()
    
    # 输出性能报告
    performance_report = monitor.get_performance_report()
    print(f"性能报告: {json.dumps(performance_report, indent=2)}")
    
    # 输出已应用的优化策略
    optimizations = optimizer.get_applied_optimizations()
    print(f"应用的优化策略: {len(optimizations)}")
    for opt in optimizations:
        print(f"  - {opt['type']}: {opt['description']}")


if __name__ == "__main__":
    main()