from tests.helper.log_capture import LogCaptureMixin\n"""
测试辅助工具类

提供测试中常用的混合类和工具函数
"""

import logging
import unittest
import io
from .log_capture import Log_capture_mixin


class LogCaptureMixin(Log_capture_mixin):
    """日志捕获混合类 - 标准命名版本"""
    
    def setUp(self):
        """设置日志捕获"""
        super().setUp()
        # 初始化日志捕获
        if hasattr(super(), 'set_up_Capture'):
            super().set_up_Capture()
    
    def tearDown(self):
        """清理日志捕获"""
        # 清理日志捕获
        if hasattr(super(), 'tear_down_Capture'):
            super().tear_down_Capture()
        super().tearDown()


# 为向后兼容提供别名
Log_capture_mixin_alias = LogCaptureMixin
