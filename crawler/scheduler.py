"""
爬虫调度器

负责任务调度、爬虫管理、监控等功能
"""

import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logger import get_logger

logger = get_logger(__name__)


class CrawlerTask:
    """爬虫任务"""

    def __init__(self, task_id: str, spider_name: str, url: str,
                 priority: int = 1, params: Dict[str, Any] = None):
        self.task_id = task_id
        self.spider_name = spider_name
        self.url = url
        self.priority = priority
        self.params = params or {}
        self.created_time = datetime.now()
        self.status = 'pending'  # pending, running, completed, failed
        self.retry_count = 0
        self.max_retries = 3

    def __lt__(self, other):
        """用于优先级队列排序"""
        return self.priority > other.priority


class TaskQueue:
    """任务队列"""

    def __init__(self):
        import queue
        self.queue = queue.PriorityQueue()
        self.completed_tasks = []
        self.failed_tasks = []

    def add_task(self, task: CrawlerTask):
        """添加任务"""
        self.queue.put(task)
        logger.info(f"添加任务: {task.task_id} - {task.spider_name}")

    def get_task(self) -> Optional[CrawlerTask]:
        """获取任务"""
        try:
            return self.queue.get_nowait()
        except:
            return None

    def mark_completed(self, task: CrawlerTask):
        """标记任务完成"""
        task.status = 'completed'
        self.completed_tasks.append(task)
        logger.info(f"任务完成: {task.task_id}")

    def mark_failed(self, task: CrawlerTask):
        """标记任务失败"""
        task.status = 'failed'
        task.retry_count += 1

        if task.retry_count < task.max_retries:
            # 重新加入队列
            task.status = 'pending'
            self.queue.put(task)
            logger.warning(f"任务重试: {task.task_id} (第{task.retry_count}次)")
        else:
            self.failed_tasks.append(task)
            logger.error(f"任务失败: {task.task_id}")

    def get_stats(self) -> Dict[str, int]:
        """获取队列统计"""
        return {
            'pending': self.queue.qsize(),
            'completed': len(self.completed_tasks),
            'failed': len(self.failed_tasks)
        }


class CrawlerPool:
    """爬虫池"""

    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_tasks = {}

    def submit_task(self, task: CrawlerTask, spider_func):
        """提交任务"""
        future = self.executor.submit(spider_func, task)
        self.running_tasks[task.task_id] = future
        return future

    def get_running_count(self) -> int:
        """获取运行中任务数量"""
        return len(self.running_tasks)

    def cleanup_completed(self):
        """清理已完成任务"""
        completed_ids = []
        for task_id, future in self.running_tasks.items():
            if future.done():
                completed_ids.append(task_id)

        for task_id in completed_ids:
            del self.running_tasks[task_id]


class CrawlerMonitor:
    """爬虫监控器"""

    def __init__(self):
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'success_rate': 0.0,
            'avg_response_time': 0.0,
            'start_time': datetime.now()
        }
        self.response_times = []

    def record_task_start(self, task: CrawlerTask):
        """记录任务开始"""
        self.stats['total_tasks'] += 1
        task.start_time = datetime.now()

    def record_task_complete(self, task: CrawlerTask, response_time: float):
        """记录任务完成"""
        self.stats['completed_tasks'] += 1
        self.response_times.append(response_time)
        self._update_stats()

    def record_task_failed(self, task: CrawlerTask):
        """记录任务失败"""
        self.stats['failed_tasks'] += 1
        self._update_stats()

    def _update_stats(self):
        """更新统计信息"""
        total = self.stats['completed_tasks'] + self.stats['failed_tasks']
        if total > 0:
            self.stats['success_rate'] = self.stats['completed_tasks'] / total

        if self.response_times:
            self.stats['avg_response_time'] = sum(self.response_times) / len(self.response_times)

    def get_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        runtime = datetime.now() - self.stats['start_time']
        stats = self.stats.copy()
        stats['runtime_hours'] = runtime.total_seconds() / 3600
        return stats


class CrawlerScheduler:
    """爬虫任务调度器"""

    def __init__(self, max_workers: int = 5):
        self.task_queue = TaskQueue()
        self.crawler_pool = CrawlerPool(max_workers)
        self.monitor = CrawlerMonitor()
        self.running = False
        self.scheduler_thread = None

        # 注册的爬虫
        self.spiders = {}

    def register_spider(self, name: str, spider_class):
        """注册爬虫"""
        self.spiders[name] = spider_class
        logger.info(f"注册爬虫: {name}")

    def add_task(self, spider_name: str, url: str, priority: int = 1,
                 params: Dict[str, Any] = None) -> str:
        """添加爬虫任务"""
        import time
        task_id = f"{spider_name}_{int(time.time())}"
        task = CrawlerTask(task_id, spider_name, url, priority, params)
        self.task_queue.add_task(task)
        return task_id

    def start(self):
        """启动调度器"""
        self.running = True
        logger.info("爬虫调度器启动")

    def stop(self):
        """停止调度器"""
        self.running = False
        logger.info("爬虫调度器停止")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'queue_stats': self.task_queue.get_stats(),
            'pool_stats': {
                'running_tasks': self.crawler_pool.get_running_count(),
                'max_workers': self.crawler_pool.max_workers
            },
            'monitor_stats': self.monitor.get_stats()
        }