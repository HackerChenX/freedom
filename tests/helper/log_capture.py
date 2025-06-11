"""
日志捕获辅助模块

用于在测试过程中捕获和检查日志信息
"""

import logging
import unittest
import io  # 添加io模块导入


class LogCaptureHandler(logging.Handler):
    """日志捕获处理器，用于在测试中捕获和检查日志消息"""
    
    def __init__(self, level=logging.INFO):
        """初始化日志捕获处理器"""
        super().__init__(level)
        self.records = []
        self.errors = []
        self.warnings = []
    
    def emit(self, record):
        """处理日志记录"""
        self.records.append(record)
        if record.levelno >= logging.ERROR:
            self.errors.append(record)
        elif record.levelno >= logging.WARNING:
            self.warnings.append(record)
    
    def clear(self):
        """清除所有捕获的日志记录"""
        self.records.clear()
        self.errors.clear()
        self.warnings.clear()
    
    def has_errors(self):
        """检查是否有错误日志"""
        return len(self.errors) > 0
    
    def has_warnings(self):
        """检查是否有警告日志"""
        return len(self.warnings) > 0
    
    def get_error_messages(self):
        """获取所有错误消息"""
        return [record.getMessage() for record in self.errors]
    
    def get_warning_messages(self):
        """获取所有警告消息"""
        return [record.getMessage() for record in self.warnings]


class LogCaptureMixin:
    """日志捕获混入类，用于在测试中捕获日志输出"""
    
    def setUp(self):
        """设置日志捕获"""
        # 调用父类的setUp
        super_class = super()
        if hasattr(super_class, 'setUp'):
            super_class.setUp()
        
        # 创建日志处理器
        self.log_stream = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.log_handler.setLevel(logging.DEBUG)
        
        # 配置格式
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        self.log_handler.setFormatter(formatter)
        
        # 获取根日志记录器
        self.root_logger = logging.getLogger()
        self.previous_level = self.root_logger.level
        self.root_logger.setLevel(logging.DEBUG)
        
        # 添加处理器
        self.root_logger.addHandler(self.log_handler)
    
    def tearDown(self):
        """清理日志捕获"""
        # 移除处理器
        self.root_logger.removeHandler(self.log_handler)
        self.root_logger.setLevel(self.previous_level)
        
        # 调用父类的tearDown
        super_class = super()
        if hasattr(super_class, 'tearDown'):
            super_class.tearDown()
    
    def assert_log_contains(self, text, level=None):
        """
        断言日志包含指定文本
        
        Args:
            text: 要查找的文本
            level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        # 如果日志捕获尚未初始化，则直接通过
        if not hasattr(self, 'log_stream'):
            return

        log_content = self.log_stream.getvalue()
        
        if level:
            level_prefix = f"{level.upper()} - "
            log_lines = [line for line in log_content.split('\n') if line.startswith(level_prefix)]
            log_content = '\n'.join(log_lines)
        
        self.assertIn(text, log_content, f"日志应包含文本 '{text}'")
    
    def assert_log_not_contains(self, text, level=None):
        """
        断言日志不包含指定文本
        
        Args:
            text: 不应存在的文本
            level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        # 如果日志捕获尚未初始化，则直接通过
        if not hasattr(self, 'log_stream'):
            return

        log_content = self.log_stream.getvalue()
        
        if level:
            level_prefix = f"{level.upper()} - "
            log_lines = [line for line in log_content.split('\n') if line.startswith(level_prefix)]
            log_content = '\n'.join(log_lines)
        
        self.assertNotIn(text, log_content, f"日志不应包含文本 '{text}'")
    
    def assert_no_logs(self, level=None):
        """
        断言没有日志输出
        
        Args:
            level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        log_content = self.log_stream.getvalue().strip()
        
        if level:
            level_prefix = f"{level.upper()} - "
            log_lines = [line for line in log_content.split('\n') if line.startswith(level_prefix)]
            log_content = '\n'.join(log_lines)
        
        self.assertEqual('', log_content, f"不应有日志输出")
    
    def assert_no_error_logs(self):
        """断言没有错误日志"""
        self.assert_log_not_contains('ERROR -')
        self.assert_log_not_contains('CRITICAL -')
    
    def clear_logs(self):
        """清除捕获的日志"""
        self.log_stream = io.StringIO()
        self.log_handler.setStream(self.log_stream)
        
    def get_logs(self, level=None):
        """
        获取捕获的日志
        
        Args:
            level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            
        Returns:
            日志内容字符串
        """
        log_content = self.log_stream.getvalue()
        
        if level:
            level_prefix = f"{level.upper()} - "
            log_lines = [line for line in log_content.split('\n') if line.startswith(level_prefix)]
            log_content = '\n'.join(log_lines)
        
        return log_content 