from utils.container import container
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
并行处理优化器

实现高效的多进程股票数据并行回测处理，包括：
1. 智能任务调度和负载均衡
2. 进程间通信优化
3. 数据分片和内存共享
4. 故障恢复和超时处理
"""

import os
import time
import pickle
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass
from multiprocessing import Process, Queue, Manager, Value, Lock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TaskConfig:
    """任务配置"""
    task_id: str
    stock_codes: List[str]
    start_date: str
    end_date: str
    indicators: List[str]
    priority: int = 1  # 1-5, 5为最高优先级

@dataclass
class ProcessorConfig:
    """处理器配置"""
    max_workers: int = 8
    task_timeout: int = 300  # 任务超时时间(秒)
    memory_threshold_gb: float = 6.0  # 内存阈值
    batch_size: int = 100  # 批次大小
    retry_attempts: int = 3  # 重试次数
    enable_load_balancing: bool = True  # 启用负载均衡

@dataclass
class WorkerStatus:
    """工作进程状态"""
    worker_id: int
    status: str = "idle"  # idle, busy, error, timeout
    current_task: Optional[str] = None
    processed_count: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_heartbeat: float = 0.0

class TaskQueue:
    """智能任务队列"""

    def __init__(self, maxsize: int = 0):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.queue = Queue(maxsize)
        self.priority_queue = Queue(maxsize)
        self.completed_tasks = {}
        self.failed_tasks = {}
        self._lock = Lock()

    def put_task(self, task: TaskConfig):
        """添加任务"""
        if task.priority >= 4:
            self.priority_queue.put(task)
        else:
            self.queue.put(task)

    def get_task(self, timeout: Optional[float] = None) -> Optional[TaskConfig]:
        """获取任务(优先级优先)"""
        try:
            # 优先处理高优先级任务
            if not self.priority_queue.empty():
                return self.priority_queue.get(timeout=timeout)
            else:
                return self.queue.get(timeout=timeout)
        except:
            return None

    def mark_completed(self, task_id: str, result: Any):
        """标记任务完成"""
        with self._lock:
            self.completed_tasks[task_id] = result

    def mark_failed(self, task_id: str, error: str):
        """标记任务失败"""
        with self._lock:
            self.failed_tasks[task_id] = error

class WorkerMonitor:
    """工作进程监控器"""

    def __init__(self, config: ProcessorConfig):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.config = config
        self.worker_stats = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self._lock = Lock()

    def start_monitoring(self):
        """启动监控"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def register_worker(self, worker_id: int, process: Process):
        """注册工作进程"""
        with self._lock:
            self.worker_stats[worker_id] = {
                'status': WorkerStatus(worker_id),
                'process': process,
                'start_time': time.time()
            }

    def update_worker_status(self, worker_id: int, status: str, task_id: Optional[str] = None):
        """更新工作进程状态"""
        with self._lock:
            if worker_id in self.worker_stats:
                worker_status = self.worker_stats[worker_id]['status']
                worker_status.status = status
                worker_status.current_task = task_id
                worker_status.last_heartbeat = time.time()

    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                self._update_system_metrics()
                self._check_worker_health()
                time.sleep(5)  # 每5秒监控一次
            except Exception as e:
                logger.error(f"监控循环出错: {e}")

    def _update_system_metrics(self):
        """更新系统指标"""
        with self._lock:
            for worker_id, worker_info in self.worker_stats.items():
                try:
                    process = worker_info['process']
                    if process.is_alive():
                        psutil_process = psutil.Process(process.pid)
                        worker_status = worker_info['status']

                        worker_status.memory_usage_mb = psutil_process.memory_info().rss / 1024 / 1024
                        worker_status.cpu_usage_percent = psutil_process.cpu_percent()
                except Exception as e:
                    logger.warning(f"更新工作进程 {worker_id} 指标失败: {e}")

    def _check_worker_health(self):
        """检查工作进程健康状态"""
        current_time = time.time()

        for worker_id, worker_info in self.worker_stats.items():
            worker_status = worker_info['status']
            process = worker_info['process']

            # 检查进程是否存活
            if not process.is_alive():
                worker_status.status = "dead"
                logger.error(f"工作进程 {worker_id} 已死亡")

            # 检查心跳超时
            elif worker_status.last_heartbeat > 0:
                heartbeat_timeout = current_time - worker_status.last_heartbeat
                if heartbeat_timeout > self.config.task_timeout:
                    worker_status.status = "timeout"
                    logger.warning(f"工作进程 {worker_id} 心跳超时: {heartbeat_timeout:.1f}秒")

            # 检查内存使用
            memory_gb = worker_status.memory_usage_mb / 1024
            if memory_gb > self.config.memory_threshold_gb:
                logger.warning(f"工作进程 {worker_id} 内存使用过高: {memory_gb:.2f}GB")

    def get_worker_statistics(self) -> Dict[str, Any]:
        """获取工作进程统计信息"""
        with self._lock:
            stats = {}
            for worker_id, worker_info in self.worker_stats.items():
                worker_status = worker_info['status']
                stats[f"worker_{worker_id}"] = {
                    'status': worker_status.status,
                    'processed_count': worker_status.processed_count,
                    'memory_mb': worker_status.memory_usage_mb,
                    'cpu_percent': worker_status.cpu_usage_percent,
                    'uptime_seconds': time.time() - worker_info['start_time']
                }
            return stats

class LoadBalancer:
    """负载均衡器"""

    def __init__(self, config: ProcessorConfig):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.config = config
        self.worker_loads = {}
        self._lock = Lock()

    def update_worker_load(self, worker_id: int, load_score: float):
        """更新工作进程负载"""
        with self._lock:
            self.worker_loads[worker_id] = {
                'load_score': load_score,
                'update_time': time.time()
            }

    def get_optimal_worker(self, available_workers: List[int]) -> Optional[int]:
        """获取最佳工作进程"""
        if not available_workers:
            return None

        with self._lock:
            # 计算每个工作进程的负载分数
            worker_scores = {}
            for worker_id in available_workers:
                if worker_id in self.worker_loads:
                    load_info = self.worker_loads[worker_id]
                    # 负载分数越低越好
                    worker_scores[worker_id] = load_info['load_score']
                else:
                    worker_scores[worker_id] = 0.0  # 新工作进程，负载为0

            # 选择负载最低的工作进程
            optimal_worker = min(worker_scores.items(), key=lambda x: x[1])[0]
            return optimal_worker

    def calculate_load_score(self, worker_status: WorkerStatus) -> float:
        """计算工作进程负载分数"""
        # 综合考虑内存使用、CPU使用和任务数量
        memory_factor = worker_status.memory_usage_mb / 1024 / self.config.memory_threshold_gb
        cpu_factor = worker_status.cpu_usage_percent / 100
        task_factor = worker_status.processed_count / 1000  # 归一化处理数量

        load_score = memory_factor * 0.4 + cpu_factor * 0.4 + task_factor * 0.2
        return load_score

def worker_process_function(worker_id: int,
                          task_queue: TaskQueue,
                          result_queue: Queue,
                          status_queue: Queue,
                          config: ProcessorConfig):
    """工作进程函数"""
    logger.info(f"工作进程 {worker_id} 启动")

    processed_count = 0

    # 动态导入以避免序列化问题
    from analysis.buypoints.high_performance_backtest_engine import HighPerformanceBacktestEngine
from db.sql_manager import SQLManager, QueryType

    # 在工作进程中初始化引擎
    engine = HighPerformanceBacktestEngine()

    while True:
        try:
            # 发送心跳
            status_queue.put({
                'worker_id': worker_id,
                'status': 'idle',
                'processed_count': processed_count,
                'timestamp': time.time()
            })

            # 获取任务
            task = task_queue.get_task(timeout=10)
            if task is None:
                continue

            # 更新状态为忙碌
            status_queue.put({
                'worker_id': worker_id,
                'status': 'busy',
                'task_id': task.task_id,
                'timestamp': time.time()
            })

            logger.info(f"工作进程 {worker_id} 开始处理任务 {task.task_id}")

            # 处理任务
            start_time = time.time()
            result = engine.run_high_performance_backtest(
                stock_codes=task.stock_codes,
                start_date=task.start_date,
                end_date=task.end_date,
                indicators=task.indicators
            )

            execution_time = time.time() - start_time
            processed_count += 1

            # 发送结果
            result_queue.put({
                'task_id': task.task_id,
                'worker_id': worker_id,
                'result': result,
                'execution_time': execution_time,
                'timestamp': time.time()
            })

            # 标记任务完成
            task_queue.mark_completed(task.task_id, result)

            logger.info(f"工作进程 {worker_id} 完成任务 {task.task_id}, 耗时 {execution_time:.2f}秒")

        except Exception as e:
            logger.error(f"工作进程 {worker_id} 处理任务失败: {e}")
            if 'task' in locals():
                task_queue.mark_failed(task.task_id, str(e))
                result_queue.put({
                    'task_id': task.task_id,
                    'worker_id': worker_id,
                    'error': str(e),
                    'timestamp': time.time()
                })

class ParallelProcessingOptimizer:
    """
    并行处理优化器

    实现高效的多进程股票数据并行回测处理
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """初始化并行处理优化器"""
        self.config = config or ProcessorConfig()
        self.logger = logger

        # 核心组件
        self.task_queue = TaskQueue()
        self.result_queue = Queue()
        self.status_queue = Queue()
        self.worker_monitor = WorkerMonitor(self.config)
        self.load_balancer = LoadBalancer(self.config)

        # 工作进程管理
        self.workers = []
        self.worker_processes = {}

        # 性能统计
        self.performance_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_processing_time': 0.0,
            'average_task_time': 0.0,
            'throughput_per_second': 0.0
        }

        self.logger.info(f"并行处理优化器初始化完成，最大工作进程数: {self.config.max_workers}")

    def start_workers(self):
        """启动工作进程"""
        self.logger.info("启动工作进程...")

        for worker_id in range(self.config.max_workers):
            process = Process(
                target=worker_process_function,
                args=(worker_id, self.task_queue, self.result_queue,
                     self.status_queue, self.config)
            )
            process.start()

            self.worker_processes[worker_id] = process
            self.worker_monitor.register_worker(worker_id, process)

        # 启动监控
        self.worker_monitor.start_monitoring()

        # 启动状态处理线程
        self.status_thread = threading.Thread(target=self._process_status_updates)
        self.status_thread.daemon = True
        self.status_thread.start()

        self.logger.info(f"成功启动 {len(self.worker_processes)} 个工作进程")

    def submit_backtest_tasks(self,
                            stock_codes: List[str],
                            start_date: str,
                            end_date: str,
                            indicators: List[str] = None) -> List[str]:
        """
        提交回测任务

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            indicators: 技术指标列表

        Returns:
            List[str]: 任务ID列表
        """
        if indicators is None:
            indicators = ['MA', 'MACD', 'RSI', 'BOLL']

        # 将股票分批创建任务
        tasks = []
        task_ids = []

        batches = self._create_batches(stock_codes, self.config.batch_size)

        for i, batch in enumerate(batches):
            task_id = f"backtest_{int(time.time())}_{i}"
            task = TaskConfig(
                task_id=task_id,
                stock_codes=batch,
                start_date=start_date,
                end_date=end_date,
                indicators=indicators,
                priority=3  # 中等优先级
            )

            tasks.append(task)
            task_ids.append(task_id)

        # 提交任务
        for task in tasks:
            self.task_queue.put_task(task)
            self.performance_stats['total_tasks'] += 1

        self.logger.info(f"提交了 {len(tasks)} 个回测任务，覆盖 {len(stock_codes)} 只股票")
        return task_ids

    def collect_results(self, task_ids: List[str], timeout: int = 600) -> Dict[str, Any]:
        """
        收集任务结果

        Args:
            task_ids: 任务ID列表
            timeout: 超时时间(秒)

        Returns:
            Dict[str, Any]: 收集到的结果
        """
        results = {}
        collected_tasks = set()
        start_time = time.time()

        self.logger.info(f"开始收集 {len(task_ids)} 个任务的结果...")

        while len(collected_tasks) < len(task_ids):
            # 检查超时
            if time.time() - start_time > timeout:
                self.logger.warning(f"收集结果超时，已收集 {len(collected_tasks)}/{len(task_ids)} 个结果")
                break

            try:
                # 获取结果
                result_data = self.result_queue.get(timeout=5)
                task_id = result_data['task_id']

                if task_id in task_ids:
                    results[task_id] = result_data
                    collected_tasks.add(task_id)

                    # 更新统计信息
                    if 'error' not in result_data:
                        self.performance_stats['completed_tasks'] += 1
                        self.performance_stats['total_processing_time'] += result_data.get('execution_time', 0)
                    else:
                        self.performance_stats['failed_tasks'] += 1

                    self.logger.info(f"收集到任务 {task_id} 的结果 ({len(collected_tasks)}/{len(task_ids)})")

            except:
                continue

        # 计算性能指标
        if self.performance_stats['completed_tasks'] > 0:
            self.performance_stats['average_task_time'] = (
                self.performance_stats['total_processing_time'] /
                self.performance_stats['completed_tasks']
            )

        total_time = time.time() - start_time
        if total_time > 0:
            self.performance_stats['throughput_per_second'] = len(collected_tasks) / total_time

        self.logger.info(f"结果收集完成: {len(collected_tasks)} 个成功, "
                        f"平均耗时 {self.performance_stats['average_task_time']:.2f}秒, "
                        f"吞吐量 {self.performance_stats['throughput_per_second']:.1f} 任务/秒")

        return results

    def run_optimized_parallel_backtest(self,
                                      stock_codes: List[str],
                                      start_date: str,
                                      end_date: str,
                                      indicators: List[str] = None) -> Dict[str, Any]:
        """
        运行优化的并行回测

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            indicators: 技术指标列表

        Returns:
            Dict[str, Any]: 完整的回测结果和性能指标
        """
        start_time = time.time()
        self.logger.info(f"开始优化并行回测，股票数量: {len(stock_codes)}")

        # 启动工作进程
        if not self.worker_processes:
            self.start_workers()

        # 提交任务
        task_ids = self.submit_backtest_tasks(stock_codes, start_date, end_date, indicators)

        # 收集结果
        results = self.collect_results(task_ids)

        # 汇总结果
        total_time = time.time() - start_time
        summary = self._generate_parallel_summary(results, total_time)

        self.logger.info(f"并行回测完成: 总耗时 {total_time:.2f}秒, "
                        f"处理速度 {len(stock_codes)/total_time:.1f} 股票/秒")

        return {
            'results': results,
            'summary': summary,
            'performance_metrics': self.performance_stats,
            'worker_statistics': self.worker_monitor.get_worker_statistics()
        }

    def _process_status_updates(self):
        """处理状态更新"""
        while True:
            try:
                status_data = self.status_queue.get(timeout=10)
                worker_id = status_data['worker_id']
                status = status_data['status']
                task_id = status_data.get('task_id')

                self.worker_monitor.update_worker_status(worker_id, status, task_id)

            except:
                continue

    def _create_batches(self, items: List[str], batch_size: int) -> List[List[str]]:
        """创建批次"""
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        return batches

    def _generate_parallel_summary(self, results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """生成并行处理汇总"""
        successful_tasks = len([r for r in results.values() if 'error' not in r])
        failed_tasks = len([r for r in results.values() if 'error' in r])

        # 汇总所有成功任务的结果
        all_backtest_results = []
        for result_data in results.values():
            if 'error' not in result_data and 'result' in result_data:
                task_results = result_data['result'].get('results', [])
                all_backtest_results.extend(task_results)

        return {
            'total_tasks': len(results),
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': successful_tasks / len(results) * 100 if results else 0,
            'total_stocks_processed': len(all_backtest_results),
            'total_execution_time': total_time,
            'average_stocks_per_second': len(all_backtest_results) / total_time if total_time > 0 else 0,
            'worker_efficiency': self._calculate_worker_efficiency()
        }

    def _calculate_worker_efficiency(self) -> Dict[str, float]:
        """计算工作进程效率"""
        worker_stats = self.worker_monitor.get_worker_statistics()
        efficiency = {}

        for worker_name, stats in worker_stats.items():
            if stats['uptime_seconds'] > 0:
                efficiency[worker_name] = stats['processed_count'] / stats['uptime_seconds'] * 60  # 任务/分钟
            else:
                efficiency[worker_name] = 0.0

        return efficiency

    def shutdown_workers(self):
        """关闭工作进程"""
        self.logger.info("关闭工作进程...")

        # 停止监控
        self.worker_monitor.stop_monitoring()

        # 终止所有工作进程
        for worker_id, process in self.worker_processes.items():
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()

        self.worker_processes.clear()
        self.logger.info("所有工作进程已关闭")

    def __del__(self):
        """析构函数"""
        self.shutdown_workers()