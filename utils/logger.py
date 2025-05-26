"""
日志工具模块

提供统一的日志记录功能
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import datetime
from typing import Dict, Optional, Union, Any

from config import get_config


# 日志级别映射
_LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

# 日志格式
_LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 日志处理器缓存
_handlers_cache: Dict[str, logging.Handler] = {}

# 是否已初始化
_initialized = False


def _ensure_log_dir() -> str:
    """
    确保日志目录存在
    
    Returns:
        str: 日志目录路径
    """
    output_dir = get_config('paths.output')
    log_dir = os.path.join(output_dir, get_config('paths.logs', 'logs'))
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    return log_dir


def _create_file_handler(log_file: str) -> logging.Handler:
    """
    创建文件日志处理器
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        logging.Handler: 文件日志处理器
    """
    if log_file in _handlers_cache:
        return _handlers_cache[log_file]
    
    max_size = get_config('log.max_size_mb', 10) * 1024 * 1024  # 默认10MB
    backup_count = get_config('log.backup_count', 5)  # 默认5个备份
    
    # 确保日志文件目录存在
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    
    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)
    handler.setFormatter(formatter)
    
    _handlers_cache[log_file] = handler
    return handler


def _create_console_handler() -> logging.Handler:
    """
    创建控制台日志处理器
    
    Returns:
        logging.Handler: 控制台日志处理器
    """
    if 'console' in _handlers_cache:
        return _handlers_cache['console']
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)
    handler.setFormatter(formatter)
    
    _handlers_cache['console'] = handler
    return handler


def init_logging(level: Optional[Union[str, int]] = None, 
                 to_console: bool = True, 
                 to_file: bool = True) -> None:
    """
    初始化日志系统
    
    Args:
        level: 日志级别，可以是字符串或整数
        to_console: 是否输出到控制台
        to_file: 是否输出到文件
    """
    global _initialized
    
    if _initialized:
        return
    
    # 获取日志级别
    if level is None:
        level = get_config('log.level', 'info')
    
    if isinstance(level, str):
        level = _LOG_LEVELS.get(level.lower(), logging.INFO)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 移除所有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台处理器
    if to_console:
        console_handler = _create_console_handler()
        root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    if to_file:
        log_dir = _ensure_log_dir()
        today = datetime.datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(log_dir, f'freedom_{today}.log')
        
        file_handler = _create_file_handler(log_file)
        root_logger.addHandler(file_handler)
    
    _initialized = True
    logging.info("日志系统初始化完成")


def get_logger(name: str, level: Optional[Union[str, int]] = None,
              log_file: Optional[str] = None, console: bool = True) -> logging.Logger:
    """
    获取命名日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别，可以是字符串或整数
        log_file: 日志文件名，如果指定则写入单独的日志文件
        console: 是否输出到控制台
        
    Returns:
        logging.Logger: 日志记录器
    """
    if not _initialized:
        init_logging()
    
    logger = logging.getLogger(name)
    
    # 设置日志级别
    if level is not None:
        if isinstance(level, str):
            level = _LOG_LEVELS.get(level.lower(), logging.INFO)
        logger.setLevel(level)
    
    # 添加控制台处理器
    if console and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = _create_console_handler()
        logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_file is not None:
        log_dir = _ensure_log_dir()
        full_log_file = os.path.join(log_dir, log_file)
        file_handler = _create_file_handler(full_log_file)
        
        # 避免重复添加相同的处理器
        if not any(getattr(h, 'baseFilename', None) == full_log_file for h in logger.handlers):
            logger.addHandler(file_handler)
    
    return logger


# 初始化日志系统
init_logging()


def setup_app_logger() -> logging.Logger:
    """
    设置应用程序级别的日志器
    
    Returns:
        logging.Logger: 应用程序日志器
    """
    log_level = get_config('log.level', 'info')
    return get_logger(name='app', log_file='app.log', console=True)


def setup_sync_logger() -> logging.Logger:
    """
    设置数据同步日志器
    
    Returns:
        logging.Logger: 同步日志器
    """
    return get_logger(name='sync', log_file='sync.log', console=True)


def setup_stock_logger() -> logging.Logger:
    """
    设置股票操作日志器
    
    Returns:
        logging.Logger: 股票操作日志器
    """
    return get_logger(name='stock', log_file='stock.log', console=True)


# 预先创建常用日志器
app_logger = setup_app_logger()
sync_logger = setup_sync_logger()
stock_logger = setup_stock_logger()


# 便捷的日志记录方法
def debug(msg: Any, *args: Any, **kwargs: Any) -> None:
    """记录调试级别日志"""
    app_logger.debug(msg, *args, **kwargs)


def info(msg: Any, *args: Any, **kwargs: Any) -> None:
    """记录信息级别日志"""
    app_logger.info(msg, *args, **kwargs)


def warning(msg: Any, *args: Any, **kwargs: Any) -> None:
    """记录警告级别日志"""
    app_logger.warning(msg, *args, **kwargs)


def error(msg: Any, *args: Any, **kwargs: Any) -> None:
    """记录错误级别日志"""
    app_logger.error(msg, *args, **kwargs)


def critical(msg: Any, *args: Any, **kwargs: Any) -> None:
    """记录严重错误级别日志"""
    app_logger.critical(msg, *args, **kwargs) 