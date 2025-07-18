#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
综合选股测试系统日志配置

提供详细的测试操作日志、错误日志、性能指标日志和审计日志功能。
支持多级日志、文件输出、实时监控和日志分析。
遵循L2基础设施层规范。
"""

import os
import logging
import logging.handlers
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from utils.logger import get_logger


class TestLogManager:
    """综合测试日志管理器"""
    
    def __init__(self, log_dir: str = "logs/comprehensive_test", 
                 log_level: str = "INFO",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_audit: bool = True,
                 max_file_size_mb: int = 10,
                 backup_count: int = 5):
        """
        初始化测试日志管理器
        
        Args:
            log_dir: 日志目录
            log_level: 日志级别
            enable_console: 是否启用控制台日志
            enable_file: 是否启用文件日志
            enable_audit: 是否启用审计日志
            max_file_size_mb: 日志文件最大大小（MB）
            backup_count: 日志文件备份数量
        """
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.loggers = {}
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_audit = enable_audit
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        
        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.error_log_dir = self.log_dir / "errors"
        self.performance_log_dir = self.log_dir / "performance"
        self.audit_log_dir = self.log_dir / "audit"
        self.session_log_dir = self.log_dir / "sessions"
        
        for dir_path in [self.error_log_dir, self.performance_log_dir, 
                        self.audit_log_dir, self.session_log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化主日志器
        self._setup_main_logger()
        
        # 初始化专用日志器
        self._setup_specialized_loggers()
        
        # 记录日志系统启动
        self.get_logger().info(f"日志系统初始化完成，日志级别: {logging.getLevelName(self.log_level)}")
        self.get_logger().info(f"日志目录: {self.log_dir}")
    
    def _setup_main_logger(self) -> None:
        """设置主日志器"""
        main_logger = logging.getLogger('comprehensive_test')
        main_logger.setLevel(self.log_level)
        
        # 清除现有处理器
        main_logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        main_logger.addHandler(console_handler)
        
        # 文件处理器
        log_file = self.log_dir / f"comprehensive_test_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        main_logger.addHandler(file_handler)
        
        self.loggers['main'] = main_logger
    
    def _setup_specialized_loggers(self) -> None:
        """设置专用日志器"""
        specialized_loggers = {
            'data_validation': '数据验证日志',
            'performance': '性能测试日志',
            'architecture': '架构合规性日志',
            'functional': '功能测试日志',
            'integration': '集成测试日志',
            'monitoring': '监控日志',
            'error': '错误日志'
        }
        
        for logger_name, description in specialized_loggers.items():
            logger = logging.getLogger(f'comprehensive_test.{logger_name}')
            logger.setLevel(self.log_level)
            
            # 文件处理器
            log_file = self.log_dir / f"{logger_name}_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # 防止日志重复
            logger.propagate = False
            
            self.loggers[logger_name] = logger
    
    def get_logger(self, name: str = 'main') -> logging.Logger:
        """
        获取指定的日志器
        
        Args:
            name: 日志器名称
            
        Returns:
            logging.Logger: 日志器实例
        """
        return self.loggers.get(name, self.loggers['main'])
    
    def log_test_start(self, test_name: str, test_type: str = 'main') -> None:
        """
        记录测试开始
        
        Args:
            test_name: 测试名称
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        logger.info(f"{'='*50}")
        logger.info(f"开始执行测试: {test_name}")
        logger.info(f"测试类型: {test_type}")
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*50}")
    
    def log_test_end(self, test_name: str, success: bool, 
                     duration: float, test_type: str = 'main') -> None:
        """
        记录测试结束
        
        Args:
            test_name: 测试名称
            success: 是否成功
            duration: 执行时间
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        status = "成功" if success else "失败"
        logger.info(f"{'='*50}")
        logger.info(f"测试结束: {test_name}")
        logger.info(f"执行状态: {status}")
        logger.info(f"执行时间: {duration:.2f}秒")
        logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*50}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any], 
                               test_type: str = 'performance') -> None:
        """
        记录性能指标
        
        Args:
            metrics: 性能指标字典
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        logger.info("性能指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    def log_error(self, error: Exception, context: str = "", 
                  test_type: str = 'error') -> None:
        """
        记录错误信息
        
        Args:
            error: 异常对象
            context: 错误上下文
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        logger.error(f"错误发生: {context}")
        logger.error(f"错误类型: {type(error).__name__}")
        logger.error(f"错误信息: {str(error)}")
        logger.exception("详细错误堆栈:")
    
    def log_data_quality_issue(self, issue: str, details: Dict[str, Any],
                              test_type: str = 'data_validation') -> None:
        """
        记录数据质量问题
        
        Args:
            issue: 问题描述
            details: 问题详情
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        logger.warning(f"数据质量问题: {issue}")
        for key, value in details.items():
            logger.warning(f"  {key}: {value}")
    
    def log_architecture_violation(self, violation: str, location: str,
                                  test_type: str = 'architecture') -> None:
        """
        记录架构违规
        
        Args:
            violation: 违规描述
            location: 违规位置
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        logger.error(f"架构违规: {violation}")
        logger.error(f"违规位置: {location}")
    
    def log_test_progress(self, current: int, total: int, 
                         description: str = "", test_type: str = 'main') -> None:
        """
        记录测试进度
        
        Args:
            current: 当前进度
            total: 总数
            description: 描述
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        percentage = (current / total) * 100 if total > 0 else 0
        logger.info(f"测试进度: {current}/{total} ({percentage:.1f}%) {description}")
    
    def create_test_session_log(self, session_id: str) -> logging.Logger:
        """
        创建测试会话专用日志器
        
        Args:
            session_id: 会话ID
            
        Returns:
            logging.Logger: 会话日志器
        """
        session_logger = logging.getLogger(f'comprehensive_test.session.{session_id}')
        session_logger.setLevel(self.log_level)
        
        # 会话日志文件
        log_file = self.log_dir / f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.log_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        session_logger.addHandler(file_handler)
        
        # 防止日志重复
        session_logger.propagate = False
        
        return session_logger
    
    def cleanup_old_logs(self, days_to_keep: int = 7) -> None:
        """
        清理旧日志文件
        
        Args:
            days_to_keep: 保留天数
        """
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    self.get_logger().info(f"删除旧日志文件: {log_file}")
                except Exception as e:
                    self.get_logger().error(f"删除日志文件失败 {log_file}: {e}")
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        获取日志摘要信息
        
        Returns:
            Dict[str, Any]: 日志摘要
        """
        summary = {
            'log_directory': str(self.log_dir),
            'log_level': logging.getLevelName(self.log_level),
            'active_loggers': list(self.loggers.keys()),
            'log_files': []
        }
        
        # 统计日志文件
        for log_file in self.log_dir.glob("*.log"):
            file_info = {
                'name': log_file.name,
                'size': log_file.stat().st_size,
                'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
            }
            summary['log_files'].append(file_info)
        
        return summary


# 全局日志管理器实例
_test_log_manager = None


def get_test_log_manager() -> TestLogManager:
    """
    获取全局测试日志管理器实例
    
    Returns:
        TestLogManager: 日志管理器实例
    """
    global _test_log_manager
    if _test_log_manager is None:
        _test_log_manager = TestLogManager()
    return _test_log_manager


def get_test_logger(name: str = 'main') -> logging.Logger:
    """
    获取测试日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        logging.Logger: 日志器实例
    """
    return get_test_log_manager().get_logger(name)

class AuditLogger:
    """审计日志管理器"""
    
    def __init__(self, log_dir: str = "logs/comprehensive_test", 
                 log_level: str = "INFO",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_audit: bool = True,
                 max_file_size_mb: int = 10,
                 backup_count: int = 5):
        """
        初始化审计日志管理器
        
        Args:
            log_dir: 日志目录
            log_level: 日志级别
            enable_console: 是否启用控制台日志
            enable_file: 是否启用文件日志
            enable_audit: 是否启用审计日志
            max_file_size_mb: 日志文件最大大小（MB）
            backup_count: 日志文件备份数量
        """
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.loggers = {}
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_audit = enable_audit
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        
        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.error_log_dir = self.log_dir / "errors"
        self.performance_log_dir = self.log_dir / "performance"
        self.audit_log_dir = self.log_dir / "audit"
        self.session_log_dir = self.log_dir / "sessions"
        
        for dir_path in [self.error_log_dir, self.performance_log_dir, 
                        self.audit_log_dir, self.session_log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化主日志器
        self._setup_main_logger()
        
        # 初始化专用日志器
        self._setup_specialized_loggers()
        
        # 初始化审计日志
        if self.enable_audit:
            self._setup_audit_logger()
        
        # 记录日志系统启动
        self.get_logger().info(f"日志系统初始化完成，日志级别: {logging.getLevelName(self.log_level)}")
        self.get_logger().info(f"日志目录: {self.log_dir}")
        
        # 记录系统信息
        self._log_system_info()
    
    def _setup_main_logger(self) -> None:
        """设置主日志器"""
        main_logger = logging.getLogger('comprehensive_test.audit')
        main_logger.setLevel(self.log_level)
        
        # 清除现有处理器
        main_logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        main_logger.addHandler(console_handler)
        
        # 文件处理器
        log_file = self.log_dir / f"comprehensive_test_audit_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        main_logger.addHandler(file_handler)
        
        self.loggers['main'] = main_logger
    
    def _setup_specialized_loggers(self) -> None:
        """设置专用日志器"""
        specialized_loggers = {
            'data_validation': '数据验证日志',
            'performance': '性能测试日志',
            'architecture': '架构合规性日志',
            'functional': '功能测试日志',
            'integration': '集成测试日志',
            'monitoring': '监控日志',
            'error': '错误日志'
        }
        
        for logger_name, description in specialized_loggers.items():
            logger = logging.getLogger(f'comprehensive_test.audit.{logger_name}')
            logger.setLevel(self.log_level)
            
            # 文件处理器
            log_file = self.log_dir / f"{logger_name}_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # 防止日志重复
            logger.propagate = False
            
            self.loggers[logger_name] = logger
    
    def get_logger(self, name: str = 'main') -> logging.Logger:
        """
        获取指定的日志器
        
        Args:
            name: 日志器名称
            
        Returns:
            logging.Logger: 日志器实例
        """
        return self.loggers.get(name, self.loggers['main'])
    
    def log_test_start(self, test_name: str, test_type: str = 'main') -> None:
        """
        记录测试开始
        
        Args:
            test_name: 测试名称
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        logger.info(f"{'='*50}")
        logger.info(f"开始执行测试: {test_name}")
        logger.info(f"测试类型: {test_type}")
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*50}")
    
    def log_test_end(self, test_name: str, success: bool, 
                     duration: float, test_type: str = 'main') -> None:
        """
        记录测试结束
        
        Args:
            test_name: 测试名称
            success: 是否成功
            duration: 执行时间
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        status = "成功" if success else "失败"
        logger.info(f"{'='*50}")
        logger.info(f"测试结束: {test_name}")
        logger.info(f"执行状态: {status}")
        logger.info(f"执行时间: {duration:.2f}秒")
        logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*50}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any], 
                               test_type: str = 'performance') -> None:
        """
        记录性能指标
        
        Args:
            metrics: 性能指标字典
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        logger.info("性能指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    def log_error(self, error: Exception, context: str = "", 
                  test_type: str = 'error') -> None:
        """
        记录错误信息
        
        Args:
            error: 异常对象
            context: 错误上下文
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        logger.error(f"错误发生: {context}")
        logger.error(f"错误类型: {type(error).__name__}")
        logger.error(f"错误信息: {str(error)}")
        logger.exception("详细错误堆栈:")
    
    def log_data_quality_issue(self, issue: str, details: Dict[str, Any],
                              test_type: str = 'data_validation') -> None:
        """
        记录数据质量问题
        
        Args:
            issue: 问题描述
            details: 问题详情
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        logger.warning(f"数据质量问题: {issue}")
        for key, value in details.items():
            logger.warning(f"  {key}: {value}")
    
    def log_architecture_violation(self, violation: str, location: str,
                                  test_type: str = 'architecture') -> None:
        """
        记录架构违规
        
        Args:
            violation: 违规描述
            location: 违规位置
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        logger.error(f"架构违规: {violation}")
        logger.error(f"违规位置: {location}")
    
    def log_test_progress(self, current: int, total: int, 
                         description: str = "", test_type: str = 'main') -> None:
        """
        记录测试进度
        
        Args:
            current: 当前进度
            total: 总数
            description: 描述
            test_type: 测试类型
        """
        logger = self.get_logger(test_type)
        percentage = (current / total) * 100 if total > 0 else 0
        logger.info(f"测试进度: {current}/{total} ({percentage:.1f}%) {description}")
    
    def create_test_session_log(self, session_id: str) -> logging.Logger:
        """
        创建测试会话专用日志器
        
        Args:
            session_id: 会话ID
            
        Returns:
            logging.Logger: 会话日志器
        """
        session_logger = logging.getLogger(f'comprehensive_test.audit.session.{session_id}')
        session_logger.setLevel(self.log_level)
        
        # 会话日志文件
        log_file = self.log_dir / f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.log_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        session_logger.addHandler(file_handler)
        
        # 防止日志重复
        session_logger.propagate = False
        
        return session_logger
    
    def cleanup_old_logs(self, days_to_keep: int = 7) -> None:
        """
        清理旧日志文件
        
        Args:
            days_to_keep: 保留天数
        """
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    self.get_logger().info(f"删除旧日志文件: {log_file}")
                except Exception as e:
                    self.get_logger().error(f"删除日志文件失败 {log_file}: {e}")
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        获取日志摘要信息
        
        Returns:
            Dict[str, Any]: 日志摘要
        """
        summary = {
            'log_directory': str(self.log_dir),
            'log_level': logging.getLevelName(self.log_level),
            'active_loggers': list(self.loggers.keys()),
            'log_files': []
        }
        
        # 统计日志文件
        for log_file in self.log_dir.glob("*.log"):
            file_info = {
                'name': log_file.name,
                'size': log_file.stat().st_size,
                'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
            }
            summary['log_files'].append(file_info)
        
        return summary


# 全局审计日志管理器实例
_audit_log_manager = None


def get_audit_log_manager() -> AuditLogger:
    """
    获取全局审计日志管理器实例
    
    Returns:
        AuditLogger: 审计日志管理器实例
    """
    global _audit_log_manager
    if _audit_log_manager is None:
        _audit_log_manager = AuditLogger()
    return _audit_log_manager


def get_audit_logger(name: str = 'main') -> logging.Logger:
    """
    获取审计日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        logging.Logger: 日志器实例
    """
    return get_audit_log_manager().get_logger(name)

    def _log_system_info(self):
        """记录系统信息"""
        import platform
        import sys
        import psutil
        
        logger = self.get_logger()
        
        logger.info("系统信息:")
        logger.info(f"  操作系统: {platform.system()} {platform.release()} ({platform.version()})")
        logger.info(f"  Python版本: {sys.version}")
        logger.info(f"  处理器: {platform.processor()}")
        
        # 内存信息
        mem = psutil.virtual_memory()
        logger.info(f"  总内存: {mem.total / (1024**3):.2f} GB")
        logger.info(f"  可用内存: {mem.available / (1024**3):.2f} GB")
        
        # 磁盘信息
        disk = psutil.disk_usage('/')
        logger.info(f"  磁盘总空间: {disk.total / (1024**3):.2f} GB")
        logger.info(f"  磁盘可用空间: {disk.free / (1024**3):.2f} GB")
    
    def _setup_audit_logger(self):
        """设置审计日志器"""
        audit_logger = logging.getLogger('comprehensive_test.audit')
        audit_logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        audit_logger.handlers.clear()
        
        # 审计日志文件
        audit_file = self.audit_log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(audit_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 使用JSON格式记录审计日志
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    'timestamp': datetime.now().isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'user': getattr(record, 'user', 'system'),
                    'action': getattr(record, 'action', 'unknown'),
                    'resource': getattr(record, 'resource', 'unknown'),
                    'result': getattr(record, 'result', 'unknown'),
                    'details': getattr(record, 'details', {})
                }
                return json.dumps(log_data, ensure_ascii=False)
        
        file_handler.setFormatter(JsonFormatter())
        audit_logger.addHandler(file_handler)
        
        # 防止日志重复
        audit_logger.propagate = False
        
        self.loggers['audit'] = audit_logger
    
    def log_audit(self, action: str, resource: str, result: str, 
                 user: str = "system", details: Dict[str, Any] = None):
        """
        记录审计日志
        
        Args:
            action: 操作类型
            resource: 资源
            result: 结果
            user: 用户
            details: 详细信息
        """
        if not self.enable_audit:
            return
        
        logger = self.get_logger('audit')
        
        # 创建日志记录
        record = logging.LogRecord(
            name=logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"{action} {resource}: {result}",
            args=(),
            exc_info=None
        )
        
        # 添加自定义属性
        record.user = user
        record.action = action
        record.resource = resource
        record.result = result
        record.details = details or {}
        
        # 记录日志
        logger.handle(record)
    
    def log_stack_trace(self, error: Exception, context: str = "", 
                       test_type: str = 'error'):
        """
        记录详细的堆栈跟踪
        
        Args:
            error: 异常对象
            context: 错误上下文
            test_type: 测试类型
        """
        import traceback
        
        logger = self.get_logger(test_type)
        
        # 获取堆栈跟踪
        stack_trace = traceback.format_exception(type(error), error, error.__traceback__)
        
        # 记录错误信息
        logger.error(f"错误发生: {context}")
        logger.error(f"错误类型: {type(error).__name__}")
        logger.error(f"错误信息: {str(error)}")
        logger.error("堆栈跟踪:")
        for line in stack_trace:
            logger.error(line.rstrip())
        
        # 保存到错误日志文件
        error_file = self.error_log_dir / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        try:
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"时间: {datetime.now().isoformat()}\n")
                f.write(f"上下文: {context}\n")
                f.write(f"错误类型: {type(error).__name__}\n")
                f.write(f"错误信息: {str(error)}\n")
                f.write("堆栈跟踪:\n")
                f.write(''.join(stack_trace))
            
            logger.info(f"详细错误日志已保存到: {error_file}")
            
        except Exception as e:
            logger.error(f"保存错误日志失败: {e}")
    
    def log_performance_event(self, event_name: str, duration_ms: float, 
                            context: Dict[str, Any] = None):
        """
        记录性能事件
        
        Args:
            event_name: 事件名称
            duration_ms: 持续时间（毫秒）
            context: 上下文信息
        """
        logger = self.get_logger('performance')
        
        # 基本信息
        log_data = {
            'event': event_name,
            'duration_ms': duration_ms,
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加上下文
        if context:
            log_data.update(context)
        
        # 记录日志
        if duration_ms > 1000:  # 超过1秒的操作记录为警告
            logger.warning(f"性能事件: {event_name} - {duration_ms:.2f}ms", extra=log_data)
        else:
            logger.info(f"性能事件: {event_name} - {duration_ms:.2f}ms", extra=log_data)
        
        # 保存到性能日志文件
        perf_file = self.performance_log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(perf_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"保存性能日志失败: {e}")
    
    def create_session_logger(self, session_id: str, test_name: str) -> logging.Logger:
        """
        创建会话专用日志器
        
        Args:
            session_id: 会话ID
            test_name: 测试名称
            
        Returns:
            logging.Logger: 会话日志器
        """
        # 创建会话目录
        session_dir = self.session_log_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # 创建日志器
        logger_name = f'comprehensive_test.session.{session_id}'
        session_logger = logging.getLogger(logger_name)
        session_logger.setLevel(self.log_level)
        
        # 清除现有处理器
        session_logger.handlers.clear()
        
        # 会话日志文件
        log_file = session_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.log_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        session_logger.addHandler(file_handler)
        
        # 防止日志重复
        session_logger.propagate = False
        
        # 记录会话开始
        session_logger.info(f"{'='*50}")
        session_logger.info(f"会话开始: {session_id}")
        session_logger.info(f"测试名称: {test_name}")
        session_logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        session_logger.info(f"{'='*50}")
        
        # 记录审计日志
        self.log_audit(
            action="SESSION_START",
            resource=f"session:{session_id}",
            result="success",
            details={"test_name": test_name}
        )
        
        return session_logger
    
    def close_session_logger(self, session_id: str, success: bool = True, 
                           duration: float = None, results: Dict[str, Any] = None):
        """
        关闭会话日志器
        
        Args:
            session_id: 会话ID
            success: 是否成功
            duration: 执行时间
            results: 测试结果
        """
        logger_name = f'comprehensive_test.session.{session_id}'
        session_logger = logging.getLogger(logger_name)
        
        # 记录会话结束
        status = "成功" if success else "失败"
        session_logger.info(f"{'='*50}")
        session_logger.info(f"会话结束: {session_id}")
        session_logger.info(f"执行状态: {status}")
        
        if duration is not None:
            session_logger.info(f"执行时间: {duration:.2f}秒")
        
        if results:
            session_logger.info("测试结果摘要:")
            for key, value in results.items():
                session_logger.info(f"  {key}: {value}")
        
        session_logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        session_logger.info(f"{'='*50}")
        
        # 关闭处理器
        for handler in session_logger.handlers:
            handler.close()
            session_logger.removeHandler(handler)
        
        # 记录审计日志
        self.log_audit(
            action="SESSION_END",
            resource=f"session:{session_id}",
            result="success" if success else "failure",
            details={
                "duration": duration,
                "results_summary": results
            }
        )