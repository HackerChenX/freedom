"""
调度管理器

管理爬虫任务的自动化调度和执行
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
from utils.logger import get_logger

logger = get_logger(__name__)


class Scheduled_task:
    """调度任务"""

    def __init__(self, name: str, func: Callable, interval: int,
                 enabled: bool = True, max_retries: int = 3):
        self.name = name
        self.func = func
        self.interval = interval  # 执行间隔（秒）
        self.enabled = enabled
        self.max_retries = max_retries

        self.last_run = None
        self.next_run = None
        self.run_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.last_error = None

        self._calculate_next_run()

    def _calculate_next_run(self):
        """计算下次执行时间"""
        if self.last_run:
            self.next_run = self.last_run + timedelta(seconds=self.interval)
        else:
            self.next_run = datetime.now() + timedelta(seconds=self.interval)

    def should_run(self) -> bool:
        """检查是否应该执行"""
        if not self.enabled:
            return False
        return datetime.now() >= self.next_run

    def execute_schedulermanager(self) -> bool:
        """执行任务"""
        if not self.enabled:
            return False

        self.run_count += 1
        self.last_run = datetime.now()

        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                logger.info(f"执行调度任务: {self.name}")
                result = self.func()

                self.success_count += 1
                self.last_error = None
                self._calculate_next_run()

                logger.info(f"调度任务执行成功: {self.name}")
                return True

            except Exception as e:
                retry_count += 1
                self.last_error = str(e)

                if retry_count <= self.max_retries:
                    logger.warning(f"调度任务执行失败，重试 {retry_count}/{self.max_retries}: {self.name} - {e}")
                    time.sleep(min(retry_count * 2, 30))  # 指数退避
                else:
                    logger.error(f"调度任务执行失败，已达最大重试次数: {self.name} - {e}")
                    self.failure_count += 1
                    self._calculate_next_run()
                    return False

        return False