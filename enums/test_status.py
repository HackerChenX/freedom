"""
测试状态枚举

定义了测试运行过程中可能的状态
"""

from enum import Enum


class TestStatus(Enum):
    """测试状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    WARNING = "warning"  # 添加警告状态，某些测试中会用到 