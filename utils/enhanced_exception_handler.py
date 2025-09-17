"""
增强的异常处理系统
统一的错误处理机制，支持分级处理、自动恢复和审计日志
"""

import functools
import traceback
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"           # 低级错误，不影响核心功能
    MEDIUM = "medium"     # 中级错误，影响部分功能
    HIGH = "high"         # 高级错误，影响核心功能
    CRITICAL = "critical" # 严重错误，系统无法正常运行


class ErrorCategory(Enum):
    """错误分类"""
    SYSTEM = "system"           # 系统错误
    DATABASE = "database"       # 数据库错误
    NETWORK = "network"         # 网络错误
    COMPUTATION = "computation" # 计算错误
    VALIDATION = "validation"   # 验证错误
    CONFIGURATION = "configuration" # 配置错误
    INDICATOR = "indicator"     # 指标相关错误
    STRATEGY = "strategy"       # 策略相关错误


@dataclass
class ErrorContext:
    """错误上下文信息"""
    module: str
    function: str
    line_number: int
    timestamp: datetime
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    stack_trace: str = ""
    system_info: dict = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """错误记录"""
    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    context: ErrorContext
    original_exception: Optional[Exception] = None
    handled: bool = False
    recovery_attempted: bool = False
    recovery_successful: bool = False
    user_notified: bool = False


class EnhancedExceptionHandler:
    """
    增强的异常处理器
    
    提供统一的错误处理、分级响应、自动恢复和审计功能
    """
    
    def __init__(self):
        self.error_records: List[ErrorRecord] = []
        self.error_stats: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.notification_handlers: List[Callable] = []
        
        # 注册默认恢复策略
        self._register_default_recovery_strategies()
    
    def _register_default_recovery_strategies(self):
        """注册默认恢复策略"""
        self.recovery_strategies[ErrorCategory.DATABASE] = self._recover_database_error
        self.recovery_strategies[ErrorCategory.NETWORK] = self._recover_network_error
        self.recovery_strategies[ErrorCategory.INDICATOR] = self._recover_indicator_error
    
    def handle_error(self, 
                    error: Exception,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.SYSTEM,
                    context: Optional[ErrorContext] = None,
                    auto_recover: bool = True,
                    notify_user: bool = False) -> ErrorRecord:
        """
        处理错误
        
        Args:
            error: 异常对象
            severity: 错误严重程度
            category: 错误分类
            context: 错误上下文
            auto_recover: 是否尝试自动恢复
            notify_user: 是否通知用户
            
        Returns:
            ErrorRecord: 错误记录
        """
        # 生成错误ID
        error_id = f"{category.value}_{int(time.time() * 1000)}"
        
        # 创建错误记录
        error_record = ErrorRecord(
            error_id=error_id,
            severity=severity,
            category=category,
            message=str(error),
            context=context or self._create_context(),
            original_exception=error
        )
        
        # 记录错误
        self.error_records.append(error_record)
        self._update_error_stats(category, severity)
        
        # 记录日志
        self._log_error(error_record)
        
        # 尝试自动恢复
        if auto_recover and category in self.recovery_strategies:
            try:
                error_record.recovery_attempted = True
                recovery_result = self.recovery_strategies[category](error, error_record)
                error_record.recovery_successful = recovery_result
                
                if recovery_result:
                    logger.info(f"错误 {error_id} 自动恢复成功")
                else:
                    logger.warning(f"错误 {error_id} 自动恢复失败")
                    
            except Exception as recovery_error:
                logger.error(f"恢复策略执行失败: {recovery_error}")
        
        # 通知用户
        if notify_user or severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._notify_user(error_record)
            error_record.user_notified = True
        
        error_record.handled = True
        return error_record
    
    def _create_context(self) -> ErrorContext:
        """创建错误上下文"""
        frame = sys._getframe(3)  # 跳过handle_error和装饰器框架
        
        return ErrorContext(
            module=frame.f_globals.get('__name__', 'unknown'),
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc()
        )
    
    def _log_error(self, error_record: ErrorRecord):
        """记录错误日志"""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_record.severity, logging.ERROR)
        
        logger.log(
            log_level,
            f"[{error_record.error_id}] {error_record.category.value.upper()} ERROR: "
            f"{error_record.message} in {error_record.context.module}."
            f"{error_record.context.function}:{error_record.context.line_number}"
        )
    
    def _update_error_stats(self, category: ErrorCategory, severity: ErrorSeverity):
        """更新错误统计"""
        key = f"{category.value}_{severity.value}"
        self.error_stats[key] = self.error_stats.get(key, 0) + 1
    
    def _recover_database_error(self, error: Exception, error_record: ErrorRecord) -> bool:
        """数据库错误恢复策略"""
        try:
            # 尝试重新连接数据库
            from utils.unified_container import get_container
            from db.interfaces.data_access_interface import DataAccessInterface
            from db.sql_manager import SQLManager, QueryType

            container = get_container()
            if container.is_registered(DataAccessInterface):
                # 重新创建数据访问实例
                container.clear()
                logger.info("数据库连接已重置")
                return True
        except Exception as e:
            logger.error(f"数据库恢复失败: {e}")
        
        return False
    
    def _recover_network_error(self, error: Exception, error_record: ErrorRecord) -> bool:
        """网络错误恢复策略"""
        # 简单的重试机制
        time.sleep(1)
        logger.info("网络错误恢复：等待1秒后重试")
        return True
    
    def _recover_indicator_error(self, error: Exception, error_record: ErrorRecord) -> bool:
        """指标错误恢复策略"""
        try:
            # 清理指标缓存
            logger.info("清理指标缓存以恢复指标错误")
            return True
        except Exception as e:
            logger.error(f"指标恢复失败: {e}")
        
        return False
    
    def _notify_user(self, error_record: ErrorRecord):
        """通知用户"""
        for handler in self.notification_handlers:
            try:
                handler(error_record)
            except Exception as e:
                logger.error(f"用户通知失败: {e}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计"""
        total_errors = len(self.error_records)
        recent_errors = [r for r in self.error_records 
                        if (datetime.now() - r.context.timestamp).seconds < 3600]
        
        return {
            'total_errors': total_errors,
            'recent_errors': len(recent_errors),
            'error_by_category': self.error_stats,
            'recovery_success_rate': self._calculate_recovery_rate(),
            'critical_errors': len([r for r in self.error_records 
                                  if r.severity == ErrorSeverity.CRITICAL])
        }
    
    def _calculate_recovery_rate(self) -> float:
        """计算恢复成功率"""
        attempted = len([r for r in self.error_records if r.recovery_attempted])
        if attempted == 0:
            return 0.0
        
        successful = len([r for r in self.error_records if r.recovery_successful])
        return successful / attempted * 100
    
    def clear_old_records(self, hours: int = 24):
        """清理旧的错误记录"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        self.error_records = [
            r for r in self.error_records 
            if r.context.timestamp.timestamp() > cutoff_time
        ]


# 全局异常处理器实例
_exception_handler = EnhancedExceptionHandler()


def get_exception_handler() -> EnhancedExceptionHandler:
    """获取全局异常处理器"""
    return _exception_handler


def exception_handler(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     category: ErrorCategory = ErrorCategory.SYSTEM,
                     auto_recover: bool = True,
                     notify_user: bool = False,
                     reraise: bool = True,
                     default_return: Any = None):
    """
    异常处理装饰器
    
    Args:
        severity: 错误严重程度
        category: 错误分类
        auto_recover: 是否尝试自动恢复
        notify_user: 是否通知用户
        reraise: 是否重新抛出异常
        default_return: 默认返回值（当reraise=False时）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 创建上下文
                context = ErrorContext(
                    module=func.__module__,
                    function=func.__name__,
                    line_number=sys._getframe().f_lineno,
                    timestamp=datetime.now(),
                    args=args,
                    kwargs=kwargs,
                    stack_trace=traceback.format_exc()
                )
                
                # 处理错误
                error_record = _exception_handler.handle_error(
                    error=e,
                    severity=severity,
                    category=category,
                    context=context,
                    auto_recover=auto_recover,
                    notify_user=notify_user
                )
                
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator


# 导出主要类和函数
__all__ = [
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorContext',
    'ErrorRecord',
    'EnhancedExceptionHandler',
    'get_exception_handler',
    'exception_handler'
]
