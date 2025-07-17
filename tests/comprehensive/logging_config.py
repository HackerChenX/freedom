#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试日志系统配置

提供统一的测试日志管理，支持多级日志、文件输出、实时监控等功能。
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
    """测试日志管理器"""
    
    def __init__(self, log_dir: str = "logs/comprehensive_test", 
                 log_level: str = "INFO"):
        """
        初始化测试日志管理器
        
        Args:
            log_dir: 日志目录
            log_level: 日志级别
        """
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.loggers = {}
        
        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化主日志器
        self._setup_main_logger()
        
        # 初始化专用日志器
        self._setup_specialized_loggers()
    
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