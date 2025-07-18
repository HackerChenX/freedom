#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行测试执行框架

基于现有TestRunner扩展，提供高性能的并行测试执行能力
支持任务分发、资源管理和负载均衡
"""

import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import queue
import psutil
import gc

from utils.logger import getLogger
from .test_infrastructure import TestRunner, TestResult, TestStatus, TestPriority
from .performance_monitor import get_performance_monitor
from .error_handler import get_error_handler, ErrorCategory

logger = getLogger(__name__)


class ExecutionMode(Enum):
    """执行模式"""
    THREAD_POOL = "thread_pool"      # 线程池模式
    PROCESS_POOL = "process_pool"    # 进程池模式
    ASYNC = "async"                  # 异步模式
    HYBRID = "hybrid"                # 混合模式


@dataclass
class TaskDistribution:
    """任务分发配置"""
    mode: ExecutionMode = ExecutionMode.THREAD_POOL
    max_workers: int = 20
    batch_size: int = 100
    priority_based: bool = True
    load_balancing: bool = True


@dataclass
class ResourceLimits:
    """资源限制"""
    max_memory_mb: int = 8192        # 最大内存使用(MB)
    max_cpu_percent: float = 80.0    # 最大CPU使用率
    max_execution_time: int = 300    # 最大执行时间(秒)
    gc_threshold: float = 0.8        # 垃圾回收阈值


@dataclass
class ExecutionStats:
    """执行统计"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    execution_time: float = 0.0
    avg_task_time: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)


class ParallelTestExecutor:
    """并行测试执行器"""
    
    def __init__(self, 
                 task_distribution: Optional[TaskDistribution] = None,
                 resource_limits: Optional[ResourceLimits] = None):
        """
        初始化并行测试执行器
        
        Args:
            task_distribution: 任务分发配置
            resource_limits: 资源限制配置
        """
        self.task_distribution = task_distribution or TaskDistribution()
        self.resource_limits = resource_limits or ResourceLimits()
        
        # 复用现有组件
        self.test_runner = TestRunner(max_workers=self.task_distribution.max_workers)
        self.performance_monitor = get_performance_monitor()
        self.error_handler = get_error_handler()
        
        # 执行器
        self.thread_executor = None
        self.process_executor = None
        
        # 状态管理
        self.is_running = False
        self.start_time = None
        self.task_queue = queue.PriorityQueue()
        self.results = {}
        self.stats = ExecutionStats()
        
        # 资源监控
        self.resource_monitor_thread = None
        self.resource_monitor_stop = threading.Event()
        
        logger.info(f"并行测试执行器初始化完成 - 模式: {self.task_distribution.mode.value}, 工作线程: {self.task_distribution.max_workers}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self._initialize_executors()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self._cleanup_executors()
    
    def _initialize_executors(self):
        """初始化执行器"""
        if self.task_distribution.mode in [ExecutionMode.THREAD_POOL, ExecutionMode.HYBRID]:
            self.thread_executor = ThreadPoolExecutor(
                max_workers=self.task_distribution.max_workers,
                thread_name_prefix="TestExecutor"
            )
        
        if self.task_distribution.mode in [ExecutionMode.PROCESS_POOL, ExecutionMode.HYBRID]:
            # 进程池使用较少的工作进程
            process_workers = min(self.task_distribution.max_workers // 2, psutil.cpu_count())
            self.process_executor = ProcessPoolExecutor(max_workers=process_workers)
    
    def _cleanup_executors(self):
        """清理执行器"""
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
            self.thread_executor = None
        
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
            self.process_executor = None
        
        # 停止资源监控
        if self.resource_monitor_thread:
            self.resource_monitor_stop.set()
            self.resource_monitor_thread.join(timeout=5)
    
    async def execute_tasks(self, 
                          tasks: List[Dict[str, Any]],
                          task_type: str = "parallel_test") -> Dict[str, Any]:
        """
        执行并行任务
        
        Args:
            tasks: 任务列表，每个任务包含 {name, func, args, kwargs, priority}
            task_type: 任务类型
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        self.is_running = True
        self.start_time = time.time()
        self.stats = ExecutionStats(total_tasks=len(tasks))
        
        logger.info(f"开始执行 {len(tasks)} 个并行任务 - 类型: {task_type}")
        
        try:
            # 启动资源监控
            self._start_resource_monitoring()
            
            # 根据执行模式分发任务
            if self.task_distribution.mode == ExecutionMode.ASYNC:
                results = await self._execute_async_tasks(tasks)
            elif self.task_distribution.mode == ExecutionMode.PROCESS_POOL:
                results = await self._execute_process_tasks(tasks)
            elif self.task_distribution.mode == ExecutionMode.HYBRID:
                results = await self._execute_hybrid_tasks(tasks)
            else:  # THREAD_POOL
                results = await self._execute_thread_tasks(tasks)
            
            # 更新统计信息
            self.stats.execution_time = time.time() - self.start_time
            self.stats.completed_tasks = len([r for r in results.values() if r.get('success', False)])
            self.stats.failed_tasks = len([r for r in results.values() if not r.get('success', False)])
            
            if self.stats.completed_tasks > 0:
                self.stats.avg_task_time = self.stats.execution_time / self.stats.completed_tasks
            
            logger.info(f"并行任务执行完成: {self.stats.completed_tasks}/{self.stats.total_tasks} 成功")
            
            return {
                'results': results,
                'stats': self.stats,
                'execution_summary': self._generate_execution_summary()
            }
            
        except Exception as e:
            logger.error(f"并行任务执行失败: {e}")
            raise
        finally:
            self.is_running = False
            self._stop_resource_monitoring()
    
    async def _execute_thread_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用线程池执行任务"""
        results = {}
        
        # 按优先级排序任务
        if self.task_distribution.priority_based:
            tasks = sorted(tasks, key=lambda t: t.get('priority', TestPriority.MEDIUM).value, reverse=True)
        
        # 分批处理任务
        batches = [
            tasks[i:i + self.task_distribution.batch_size] 
            for i in range(0, len(tasks), self.task_distribution.batch_size)
        ]
        
        for batch_idx, batch in enumerate(batches):
            logger.debug(f"处理批次 {batch_idx + 1}/{len(batches)}: {len(batch)} 个任务")
            
            # 检查资源限制
            if not self._check_resource_limits():
                logger.warning("资源限制达到，触发优化")
                self._optimize_performance()
            
            # 提交批次任务
            future_to_task = {}
            for task in batch:
                future = self.thread_executor.submit(self._execute_single_task, task)
                future_to_task[future] = task
            
            # 收集批次结果
            for future in as_completed(future_to_task, timeout=self.resource_limits.max_execution_time):
                task = future_to_task[future]
                task_name = task.get('name', 'unknown')
                
                try:
                    result = future.result()
                    results[task_name] = result
                except Exception as e:
                    logger.error(f"任务 {task_name} 执行失败: {e}")
                    results[task_name] = {'success': False, 'error': str(e)}
        
        return results
    
    async def _execute_async_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用异步方式执行任务"""
        results = {}
        
        # 创建异步任务
        async_tasks = []
        for task in tasks:
            async_task = self._execute_async_task(task)
            async_tasks.append((task.get('name', 'unknown'), async_task))
        
        # 并发执行异步任务
        for task_name, async_task in async_tasks:
            try:
                result = await async_task
                results[task_name] = result
            except Exception as e:
                logger.error(f"异步任务 {task_name} 执行失败: {e}")
                results[task_name] = {'success': False, 'error': str(e)}
        
        return results
    
    async def _execute_process_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用进程池执行任务"""
        results = {}
        
        # 只有CPU密集型任务才使用进程池
        cpu_intensive_tasks = [t for t in tasks if t.get('cpu_intensive', False)]
        other_tasks = [t for t in tasks if not t.get('cpu_intensive', False)]
        
        # 进程池执行CPU密集型任务
        if cpu_intensive_tasks:
            future_to_task = {}
            for task in cpu_intensive_tasks:
                future = self.process_executor.submit(self._execute_single_task, task)
                future_to_task[future] = task
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                task_name = task.get('name', 'unknown')
                
                try:
                    result = future.result()
                    results[task_name] = result
                except Exception as e:
                    logger.error(f"进程任务 {task_name} 执行失败: {e}")
                    results[task_name] = {'success': False, 'error': str(e)}
        
        # 线程池执行其他任务
        if other_tasks:
            thread_results = await self._execute_thread_tasks(other_tasks)
            results.update(thread_results)
        
        return results
    
    async def _execute_hybrid_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用混合模式执行任务"""
        # 混合模式结合了线程池和进程池
        return await self._execute_process_tasks(tasks)
    
    def _execute_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个任务"""
        task_name = task.get('name', 'unknown')
        task_func = task.get('func')
        task_args = task.get('args', [])
        task_kwargs = task.get('kwargs', {})
        
        start_time = time.time()
        
        try:
            # 使用现有的TestRunner执行任务
            if callable(task_func):
                result = self.test_runner.run_test(
                    test_func=task_func,
                    test_name=task_name,
                    **task_kwargs
                )
                
                return {
                    'success': result.status == TestStatus.PASSED,
                    'result': result.details,
                    'execution_time': time.time() - start_time,
                    'metrics': result.metrics
                }
            else:
                return {'success': False, 'error': 'Invalid task function'}
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    async def _execute_async_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行异步任务"""
        task_name = task.get('name', 'unknown')
        task_func = task.get('func')
        task_kwargs = task.get('kwargs', {})
        
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(task_func):
                result = await task_func(**task_kwargs)
            else:
                # 在线程池中执行同步函数
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_executor, 
                    lambda: task_func(**task_kwargs)
                )
            
            return {
                'success': True,
                'result': result,
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _start_resource_monitoring(self):
        """启动资源监控"""
        self.resource_monitor_stop.clear()
        self.resource_monitor_thread = threading.Thread(
            target=self._resource_monitor_loop,
            daemon=True
        )
        self.resource_monitor_thread.start()
    
    def _stop_resource_monitoring(self):
        """停止资源监控"""
        if self.resource_monitor_thread:
            self.resource_monitor_stop.set()
            self.resource_monitor_thread.join(timeout=5)
    
    def _resource_monitor_loop(self):
        """资源监控循环"""
        while not self.resource_monitor_stop.is_set():
            try:
                # 监控CPU和内存使用
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / (1024 ** 2)
                
                self.stats.resource_usage = {
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'memory_percent': memory_info.percent
                }
                
                # 检查资源限制
                if not self._check_resource_limits():
                    logger.warning("资源使用超限，触发性能优化")
                    self._optimize_performance()
                
                time.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                logger.error(f"资源监控错误: {e}")
                time.sleep(5)
    
    def _check_resource_limits(self) -> bool:
        """检查资源限制"""
        if not self.stats.resource_usage:
            return True
        
        # 检查内存限制
        if self.stats.resource_usage.get('memory_mb', 0) > self.resource_limits.max_memory_mb:
            return False
        
        # 检查CPU限制
        if self.stats.resource_usage.get('cpu_percent', 0) > self.resource_limits.max_cpu_percent:
            return False
        
        # 检查执行时间限制
        if self.start_time and (time.time() - self.start_time) > self.resource_limits.max_execution_time:
            return False
        
        return True
    
    def _optimize_performance(self):
        """性能优化"""
        try:
            # 强制垃圾回收
            gc.collect()
            
            # 减少批次大小
            if self.task_distribution.batch_size > 10:
                self.task_distribution.batch_size = max(10, self.task_distribution.batch_size // 2)
                logger.info(f"优化: 批次大小调整为 {self.task_distribution.batch_size}")
            
            # 减少工作线程数
            if self.task_distribution.max_workers > 5:
                self.task_distribution.max_workers = max(5, self.task_distribution.max_workers // 2)
                logger.info(f"优化: 工作线程数调整为 {self.task_distribution.max_workers}")
            
        except Exception as e:
            logger.error(f"性能优化失败: {e}")
    
    def _generate_execution_summary(self) -> Dict[str, Any]:
        """生成执行摘要"""
        return {
            'execution_mode': self.task_distribution.mode.value,
            'total_tasks': self.stats.total_tasks,
            'completed_tasks': self.stats.completed_tasks,
            'failed_tasks': self.stats.failed_tasks,
            'success_rate': self.stats.completed_tasks / self.stats.total_tasks if self.stats.total_tasks > 0 else 0,
            'execution_time': self.stats.execution_time,
            'avg_task_time': self.stats.avg_task_time,
            'tasks_per_second': self.stats.completed_tasks / self.stats.execution_time if self.stats.execution_time > 0 else 0,
            'resource_usage': self.stats.resource_usage,
            'configuration': {
                'max_workers': self.task_distribution.max_workers,
                'batch_size': self.task_distribution.batch_size,
                'resource_limits': {
                    'max_memory_mb': self.resource_limits.max_memory_mb,
                    'max_cpu_percent': self.resource_limits.max_cpu_percent,
                    'max_execution_time': self.resource_limits.max_execution_time
                }
            }
        }


# 全局并行执行器实例
_parallel_executor = None


def get_parallel_executor(task_distribution: Optional[TaskDistribution] = None,
                         resource_limits: Optional[ResourceLimits] = None) -> ParallelTestExecutor:
    """
    获取全局并行执行器实例
    
    Args:
        task_distribution: 任务分发配置
        resource_limits: 资源限制配置
        
    Returns:
        ParallelTestExecutor: 并行执行器实例
    """
    global _parallel_executor
    if _parallel_executor is None:
        _parallel_executor = ParallelTestExecutor(task_distribution, resource_limits)
    return _parallel_executor


async def main():
    """测试并行执行器"""
    print("测试并行执行器...")
    
    # 创建测试任务
    def test_task(task_id: int, duration: float = 1.0):
        """测试任务函数"""
        time.sleep(duration)
        return f"Task {task_id} completed"
    
    tasks = [
        {
            'name': f'test_task_{i}',
            'func': test_task,
            'kwargs': {'task_id': i, 'duration': 0.5},
            'priority': TestPriority.HIGH if i % 3 == 0 else TestPriority.MEDIUM
        }
        for i in range(20)
    ]
    
    # 配置并行执行器
    task_distribution = TaskDistribution(
        mode=ExecutionMode.THREAD_POOL,
        max_workers=10,
        batch_size=5
    )
    
    resource_limits = ResourceLimits(
        max_memory_mb=4096,
        max_cpu_percent=70.0,
        max_execution_time=60
    )
    
    # 执行并行任务
    with ParallelTestExecutor(task_distribution, resource_limits) as executor:
        results = await executor.execute_tasks(tasks, "test_parallel_execution")
    
    print(f"执行完成:")
    print(f"总任务: {results['stats'].total_tasks}")
    print(f"成功: {results['stats'].completed_tasks}")
    print(f"失败: {results['stats'].failed_tasks}")
    print(f"执行时间: {results['stats'].execution_time:.2f}秒")
    print(f"平均任务时间: {results['stats'].avg_task_time:.2f}秒")


if __name__ == "__main__":
    asyncio.run(main())