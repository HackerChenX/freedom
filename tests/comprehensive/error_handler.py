#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误处理系统

提供集中式错误处理、重试机制和错误诊断
"""

import time
import traceback
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import functools
import sys

from utils.logger import getLogger

logger = getLogger(__name__)

# 泛型类型定义
T = TypeVar('T')
R = TypeVar('R')


class ErrorCategory(Enum):
    """错误类别"""
    CONFIGURATION = "配置错误"
    DATA_ACCESS = "数据访问错误"
    INDICATOR = "指标错误"
    PATTERN = "形态错误"
    VERIFICATION = "验证错误"
    SYSTEM = "系统错误"
    TIMEOUT = "超时错误"
    MEMORY = "内存错误"
    UNKNOWN = "未知错误"


@dataclass
class ErrorInfo:
    """错误信息"""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    message: str
    exception: Optional[Exception] = None
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    is_fatal: bool = False
    is_handled: bool = False
    retry_count: int = 0
    max_retries: int = 0


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        """初始化错误处理器"""
        self.errors = []
        self.error_counts = {category: 0 for category in ErrorCategory}
        self.fatal_error = None
        self.error_handlers = {}
        self.retry_policies = {}
        
        # 注册默认错误处理器
        self._register_default_handlers()
        
        logger.info("错误处理器初始化完成")
    
    def _register_default_handlers(self):
        """注册默认错误处理器"""
        # 配置错误处理器
        self.register_error_handler(
            ErrorCategory.CONFIGURATION,
            lambda error: logger.error(f"配置错误: {error.message}")
        )
        
        # 数据访问错误处理器
        self.register_error_handler(
            ErrorCategory.DATA_ACCESS,
            lambda error: logger.error(f"数据访问错误: {error.message}")
        )
        
        # 指标错误处理器
        self.register_error_handler(
            ErrorCategory.INDICATOR,
            lambda error: logger.error(f"指标错误: {error.message}")
        )
        
        # 形态错误处理器
        self.register_error_handler(
            ErrorCategory.PATTERN,
            lambda error: logger.error(f"形态错误: {error.message}")
        )
        
        # 验证错误处理器
        self.register_error_handler(
            ErrorCategory.VERIFICATION,
            lambda error: logger.error(f"验证错误: {error.message}")
        )
        
        # 系统错误处理器
        self.register_error_handler(
            ErrorCategory.SYSTEM,
            lambda error: logger.error(f"系统错误: {error.message}")
        )
        
        # 超时错误处理器
        self.register_error_handler(
            ErrorCategory.TIMEOUT,
            lambda error: logger.error(f"超时错误: {error.message}")
        )
        
        # 内存错误处理器
        self.register_error_handler(
            ErrorCategory.MEMORY,
            lambda error: logger.error(f"内存错误: {error.message}")
        )
        
        # 未知错误处理器
        self.register_error_handler(
            ErrorCategory.UNKNOWN,
            lambda error: logger.error(f"未知错误: {error.message}")
        )
    
    def register_error_handler(self, 
                             category: ErrorCategory, 
                             handler: Callable[[ErrorInfo], None]) -> None:
        """
        注册错误处理器
        
        Args:
            category: 错误类别
            handler: 处理函数
        """
        self.error_handlers[category] = handler
        logger.debug(f"注册错误处理器: {category.value}")
    
    def register_retry_policy(self, 
                            category: ErrorCategory, 
                            max_retries: int, 
                            retry_delay: float = 1.0) -> None:
        """
        注册重试策略
        
        Args:
            category: 错误类别
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.retry_policies[category] = {
            'max_retries': max_retries,
            'retry_delay': retry_delay
        }
        logger.debug(f"注册重试策略: {category.value}, 最大重试次数: {max_retries}")
    
    def handle_error(self, 
                   exception: Exception, 
                   category: Optional[ErrorCategory] = None, 
                   context: Optional[Dict[str, Any]] = None,
                   is_fatal: bool = False) -> ErrorInfo:
        """
        处理错误
        
        Args:
            exception: 异常对象
            category: 错误类别，None表示自动判断
            context: 错误上下文
            is_fatal: 是否致命错误
            
        Returns:
            ErrorInfo: 错误信息
        """
        # 自动判断错误类别
        if category is None:
            category = self._categorize_error(exception)
        
        # 创建错误信息
        error_id = f"error_{len(self.errors) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        error_info = ErrorInfo(
            error_id=error_id,
            timestamp=datetime.now(),
            category=category,
            message=str(exception),
            exception=exception,
            traceback=traceback.format_exc(),
            context=context or {},
            is_fatal=is_fatal,
            is_handled=False,
            retry_count=0,
            max_retries=self.retry_policies.get(category, {}).get('max_retries', 0)
        )
        
        # 记录错误
        self.errors.append(error_info)
        self.error_counts[category] += 1
        
        # 如果是致命错误，记录
        if is_fatal:
            self.fatal_error = error_info
        
        # 调用对应的错误处理器
        if category in self.error_handlers:
            try:
                self.error_handlers[category](error_info)
                error_info.is_handled = True
            except Exception as e:
                logger.error(f"错误处理器执行失败: {e}")
        
        logger.debug(f"处理错误: {error_id}, 类别: {category.value}, 消息: {error_info.message}")
        return error_info
    
    def _categorize_error(self, exception: Exception) -> ErrorCategory:
        """
        自动分类错误
        
        Args:
            exception: 异常对象
            
        Returns:
            ErrorCategory: 错误类别
        """
        error_type = type(exception).__name__
        error_msg = str(exception).lower()
        
        # 配置错误
        if error_type in ['ConfigError', 'ValidationError'] or 'config' in error_msg:
            return ErrorCategory.CONFIGURATION
        
        # 数据访问错误
        if error_type in ['DatabaseError', 'ConnectionError', 'ClickHouseError'] or any(
            keyword in error_msg for keyword in ['database', 'connection', 'query', 'sql']
        ):
            return ErrorCategory.DATA_ACCESS
        
        # 指标错误
        if 'indicator' in error_msg:
            return ErrorCategory.INDICATOR
        
        # 形态错误
        if 'pattern' in error_msg:
            return ErrorCategory.PATTERN
        
        # 验证错误
        if 'verification' in error_msg or 'validate' in error_msg:
            return ErrorCategory.VERIFICATION
        
        # 超时错误
        if error_type == 'TimeoutError' or 'timeout' in error_msg:
            return ErrorCategory.TIMEOUT
        
        # 内存错误
        if error_type == 'MemoryError' or 'memory' in error_msg:
            return ErrorCategory.MEMORY
        
        # 系统错误
        if error_type in ['SystemError', 'OSError', 'IOError']:
            return ErrorCategory.SYSTEM
        
        # 默认为未知错误
        return ErrorCategory.UNKNOWN
    
    def should_retry(self, error_info: ErrorInfo) -> bool:
        """
        判断是否应该重试
        
        Args:
            error_info: 错误信息
            
        Returns:
            bool: 是否应该重试
        """
        # 致命错误不重试
        if error_info.is_fatal:
            return False
        
        # 检查重试策略
        retry_policy = self.retry_policies.get(error_info.category)
        if not retry_policy:
            return False
        
        # 检查重试次数
        if error_info.retry_count >= retry_policy['max_retries']:
            return False
        
        return True
    
    def retry_operation(self, error_info: ErrorInfo) -> None:
        """
        重试操作
        
        Args:
            error_info: 错误信息
        """
        # 更新重试次数
        error_info.retry_count += 1
        
        # 获取重试延迟
        retry_delay = self.retry_policies.get(error_info.category, {}).get('retry_delay', 1.0)
        
        logger.info(f"重试操作: {error_info.error_id}, 第 {error_info.retry_count} 次重试, 延迟 {retry_delay} 秒")
        
        # 延迟
        time.sleep(retry_delay)
    
    def should_continue_testing(self, error_count: int, total_tests: int) -> bool:
        """
        判断是否应该继续测试
        
        Args:
            error_count: 错误数量
            total_tests: 总测试数
            
        Returns:
            bool: 是否应该继续测试
        """
        # 如果有致命错误，停止测试
        if self.fatal_error:
            return False
        
        # 如果错误率过高，停止测试
        if total_tests > 0 and error_count / total_tests > 0.5:
            return False
        
        return True
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        获取错误摘要
        
        Returns:
            Dict[str, Any]: 错误摘要
        """
        return {
            'total_errors': len(self.errors),
            'error_counts': {category.value: count for category, count in self.error_counts.items()},
            'fatal_error': self.fatal_error.error_id if self.fatal_error else None,
            'handled_errors': sum(1 for error in self.errors if error.is_handled),
            'unhandled_errors': sum(1 for error in self.errors if not error.is_handled)
        }
    
    def get_detailed_errors(self, 
                          category: Optional[ErrorCategory] = None, 
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取详细错误信息
        
        Args:
            category: 错误类别，None表示所有类别
            limit: 最大返回数量
            
        Returns:
            List[Dict[str, Any]]: 详细错误信息列表
        """
        filtered_errors = self.errors
        if category:
            filtered_errors = [error for error in self.errors if error.category == category]
        
        # 按时间倒序排序
        sorted_errors = sorted(filtered_errors, key=lambda x: x.timestamp, reverse=True)
        
        # 限制数量
        limited_errors = sorted_errors[:limit]
        
        # 转换为字典
        return [
            {
                'error_id': error.error_id,
                'timestamp': error.timestamp.isoformat(),
                'category': error.category.value,
                'message': error.message,
                'is_fatal': error.is_fatal,
                'is_handled': error.is_handled,
                'retry_count': error.retry_count,
                'context': error.context
            }
            for error in limited_errors
        ]
    
    def clear_errors(self) -> None:
        """清空错误记录"""
        self.errors = []
        self.error_counts = {category: 0 for category in ErrorCategory}
        self.fatal_error = None
        logger.debug("错误记录已清空")


# 装饰器：带重试的错误处理
def with_error_handling(error_handler: ErrorHandler, 
                       category: Optional[ErrorCategory] = None,
                       is_fatal: bool = False):
    """
    带错误处理的装饰器
    
    Args:
        error_handler: 错误处理器
        category: 错误类别，None表示自动判断
        is_fatal: 是否致命错误
        
    Returns:
        Callable: 装饰器
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 处理错误
                error_info = error_handler.handle_error(
                    exception=e,
                    category=category,
                    context={'args': args, 'kwargs': kwargs},
                    is_fatal=is_fatal
                )
                
                # 如果应该重试
                if error_handler.should_retry(error_info):
                    error_handler.retry_operation(error_info)
                    try:
                        return func(*args, **kwargs)
                    except Exception as retry_e:
                        # 重试失败
                        error_handler.handle_error(
                            exception=retry_e,
                            category=category,
                            context={'args': args, 'kwargs': kwargs, 'retry': True},
                            is_fatal=is_fatal
                        )
                
                # 返回默认值或重新抛出异常
                if is_fatal:
                    raise
                return None
        return wrapper
    return decorator


# 全局错误处理器实例
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """
    获取全局错误处理器实例
    
    Returns:
        ErrorHandler: 错误处理器实例
    """
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler

class TestErrorHandler:
    """测试错误处理器"""
    
    def __init__(self):
        """初始化测试错误处理器"""
        self.error_handler = get_error_handler()
        
        # 注册测试相关的错误处理器
        self._register_test_error_handlers()
        
        # 注册重试策略
        self._register_retry_policies()
        
        logger.info("测试错误处理器初始化完成")
    
    def _register_test_error_handlers(self):
        """注册测试相关的错误处理器"""
        # 指标错误处理器
        self.error_handler.register_error_handler(
            ErrorCategory.INDICATOR,
            self._handle_indicator_error
        )
        
        # 形态错误处理器
        self.error_handler.register_error_handler(
            ErrorCategory.PATTERN,
            self._handle_pattern_error
        )
        
        # 验证错误处理器
        self.error_handler.register_error_handler(
            ErrorCategory.VERIFICATION,
            self._handle_verification_error
        )
    
    def _register_retry_policies(self):
        """注册重试策略"""
        # 数据访问错误：最多重试3次，每次延迟2秒
        self.error_handler.register_retry_policy(
            ErrorCategory.DATA_ACCESS,
            max_retries=3,
            retry_delay=2.0
        )
        
        # 指标错误：最多重试2次，每次延迟1秒
        self.error_handler.register_retry_policy(
            ErrorCategory.INDICATOR,
            max_retries=2,
            retry_delay=1.0
        )
        
        # 验证错误：最多重试1次，每次延迟1秒
        self.error_handler.register_retry_policy(
            ErrorCategory.VERIFICATION,
            max_retries=1,
            retry_delay=1.0
        )
    
    def _handle_indicator_error(self, error_info: ErrorInfo):
        """
        处理指标错误
        
        Args:
            error_info: 错误信息
        """
        logger.error(f"指标错误: {error_info.message}")
        
        # 记录详细信息
        if 'indicator_name' in error_info.context:
            indicator_name = error_info.context['indicator_name']
            logger.error(f"错误指标: {indicator_name}")
            
            # 可以在这里添加指标特定的处理逻辑
            # 例如：尝试使用备用指标、记录到特定日志等
    
    def _handle_pattern_error(self, error_info: ErrorInfo):
        """
        处理形态错误
        
        Args:
            error_info: 错误信息
        """
        logger.error(f"形态错误: {error_info.message}")
        
        # 记录详细信息
        if 'pattern_id' in error_info.context:
            pattern_id = error_info.context['pattern_id']
            logger.error(f"错误形态: {pattern_id}")
            
            # 可以在这里添加形态特定的处理逻辑
            # 例如：尝试使用备用形态、记录到特定日志等
    
    def _handle_verification_error(self, error_info: ErrorInfo):
        """
        处理验证错误
        
        Args:
            error_info: 错误信息
        """
        logger.error(f"验证错误: {error_info.message}")
        
        # 记录详细信息
        if 'stock_code' in error_info.context and 'date' in error_info.context:
            stock_code = error_info.context['stock_code']
            date = error_info.context['date']
            logger.error(f"验证失败: 股票 {stock_code}, 日期 {date}")
            
            # 可以在这里添加验证特定的处理逻辑
            # 例如：尝试使用不同的验证方法、记录到特定日志等
    
    def handle_indicator_error(self, indicator_name: str, error: Exception) -> Dict[str, Any]:
        """
        处理指标错误
        
        Args:
            indicator_name: 指标名称
            error: 异常对象
            
        Returns:
            Dict[str, Any]: 错误结果
        """
        error_info = self.error_handler.handle_error(
            exception=error,
            category=ErrorCategory.INDICATOR,
            context={'indicator_name': indicator_name}
        )
        
        return {
            'error_id': error_info.error_id,
            'message': error_info.message,
            'indicator_name': indicator_name,
            'handled': error_info.is_handled
        }
    
    def handle_pattern_error(self, pattern_id: str, error: Exception) -> Dict[str, Any]:
        """
        处理形态错误
        
        Args:
            pattern_id: 形态ID
            error: 异常对象
            
        Returns:
            Dict[str, Any]: 错误结果
        """
        error_info = self.error_handler.handle_error(
            exception=error,
            category=ErrorCategory.PATTERN,
            context={'pattern_id': pattern_id}
        )
        
        return {
            'error_id': error_info.error_id,
            'message': error_info.message,
            'pattern_id': pattern_id,
            'handled': error_info.is_handled
        }
    
    def handle_verification_error(self, stock_code: str, date: str, error: Exception) -> Dict[str, Any]:
        """
        处理验证错误
        
        Args:
            stock_code: 股票代码
            date: 日期
            error: 异常对象
            
        Returns:
            Dict[str, Any]: 错误结果
        """
        error_info = self.error_handler.handle_error(
            exception=error,
            category=ErrorCategory.VERIFICATION,
            context={'stock_code': stock_code, 'date': date}
        )
        
        return {
            'error_id': error_info.error_id,
            'message': error_info.message,
            'stock_code': stock_code,
            'date': date,
            'handled': error_info.is_handled
        }
    
    def should_continue_testing(self, error_count: int, total_tests: int) -> bool:
        """
        判断是否应该继续测试
        
        Args:
            error_count: 错误数量
            total_tests: 总测试数
            
        Returns:
            bool: 是否应该继续测试
        """
        return self.error_handler.should_continue_testing(error_count, total_tests)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        获取错误摘要
        
        Returns:
            Dict[str, Any]: 错误摘要
        """
        return self.error_handler.get_error_summary()
    
    def get_detailed_errors(self, 
                          category: Optional[ErrorCategory] = None, 
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取详细错误信息
        
        Args:
            category: 错误类别，None表示所有类别
            limit: 最大返回数量
            
        Returns:
            List[Dict[str, Any]]: 详细错误信息列表
        """
        return self.error_handler.get_detailed_errors(category, limit)


def main():
    """测试错误处理器"""
    print("测试错误处理器...")
    
    # 创建错误处理器
    error_handler = get_error_handler()
    
    # 注册重试策略
    error_handler.register_retry_policy(
        ErrorCategory.DATA_ACCESS,
        max_retries=3,
        retry_delay=1.0
    )
    
    # 测试错误处理
    print("\n测试错误处理...")
    try:
        # 模拟数据访问错误
        raise ConnectionError("数据库连接失败")
    except Exception as e:
        error_info = error_handler.handle_error(e)
        print(f"错误ID: {error_info.error_id}")
        print(f"错误类别: {error_info.category.value}")
        print(f"错误消息: {error_info.message}")
        print(f"是否处理: {error_info.is_handled}")
    
    # 测试重试
    print("\n测试重试机制...")
    retry_count = [0]
    
    def test_function():
        retry_count[0] += 1
        if retry_count[0] <= 2:
            raise ConnectionError(f"连接失败，第 {retry_count[0]} 次尝试")
        return "成功"
    
    # 使用装饰器
    decorated_func = with_error_handling(error_handler, ErrorCategory.DATA_ACCESS)(test_function)
    
    result = decorated_func()
    print(f"重试结果: {result}")
    print(f"重试次数: {retry_count[0]}")
    
    # 获取错误摘要
    summary = error_handler.get_error_summary()
    print("\n错误摘要:")
    print(f"总错误数: {summary['total_errors']}")
    print(f"错误分布: {summary['error_counts']}")
    
    # 获取详细错误
    detailed_errors = error_handler.get_detailed_errors()
    print("\n详细错误:")
    for error in detailed_errors:
        print(f"  - {error['error_id']}: {error['message']} ({error['category']})")


if __name__ == "__main__":
    main()