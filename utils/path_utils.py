"""
路径工具模块

提供文件路径管理的辅助函数
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

from config import get_config
from utils.logger import get_logger
from utils.file_utils import ensure_dir

logger = get_logger(__name__)


def get_project_root() -> str:
    """
    获取项目根目录
    
    Returns:
        str: 项目根目录路径
    """
    # 当前模块所在目录的上级目录
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_output_dir() -> str:
    """
    获取输出目录路径
    
    Returns:
        str: 输出目录路径
    """
    output_dir = get_config('paths.output')
    
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(get_project_root(), output_dir)
    
    # 替换~为用户目录
    output_dir = os.path.expanduser(output_dir)
    
    # 确保目录存在
    ensure_dir(output_dir)
    
    return output_dir


def get_config_dir() -> str:
    """
    获取配置目录路径
    
    Returns:
        str: 配置目录路径
    """
    config_dir = os.path.join(get_project_root(), 'config')
    ensure_dir(config_dir)
    return config_dir


def get_doc_dir() -> str:
    """
    获取文档目录路径
    
    Returns:
        str: 文档目录路径
    """
    doc_dir = os.path.join(get_project_root(), get_config('paths.doc', 'doc'))
    ensure_dir(doc_dir)
    return doc_dir


def get_logs_dir() -> str:
    """
    获取日志目录路径
    
    Returns:
        str: 日志目录路径
    """
    logs_dir = os.path.join(get_output_dir(), get_config('paths.logs', 'logs'))
    ensure_dir(logs_dir)
    return logs_dir


def get_data_dir() -> str:
    """
    获取数据目录
    
    Returns:
        str: 数据目录路径
    """
    return os.path.join(get_project_root(), 'data')


def get_result_dir() -> str:
    """
    获取结果目录
    
    Returns:
        str: 结果目录路径
    """
    result_dir = os.path.join(get_data_dir(), 'result')
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def get_stock_result_file() -> str:
    """
    获取股票结果文件路径
    
    Returns:
        str: 股票结果文件路径
    """
    return os.path.join(get_result_dir(), 'stock_result.txt')


def get_backtest_result_dir() -> str:
    """
    获取回测结果目录
    
    Returns:
        str: 回测结果目录路径
    """
    backtest_dir = os.path.join(get_result_dir(), '回测结果')
    os.makedirs(backtest_dir, exist_ok=True)
    return backtest_dir


def get_strategies_dir() -> str:
    """
    获取策略目录
    
    Returns:
        str: 策略目录路径
    """
    strategies_dir = os.path.join(get_backtest_result_dir(), 'strategies')
    os.makedirs(strategies_dir, exist_ok=True)
    return strategies_dir


def get_strategy_dir() -> str:
    """
    获取策略配置文件目录
    
    Returns:
        str: 策略配置文件目录路径
    """
    strategy_dir = os.path.join(get_data_dir(), 'strategies')
    os.makedirs(strategy_dir, exist_ok=True)
    return strategy_dir


def get_multi_period_dir() -> str:
    """
    获取多周期分析结果目录
    
    Returns:
        str: 多周期分析结果目录路径
    """
    multi_period_dir = os.path.join(get_backtest_result_dir(), 'multi_period')
    os.makedirs(multi_period_dir, exist_ok=True)
    return multi_period_dir


def get_log_dir() -> str:
    """
    获取日志目录
    
    Returns:
        str: 日志目录路径
    """
    log_dir = os.path.join(get_project_root(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_stock_code_name_file() -> str:
    """
    获取股票代码名称文件路径
    
    Returns:
        str: 股票代码名称文件路径
    """
    return os.path.join(get_doc_dir(), 'stock_code_name.csv')


def get_temp_dir() -> str:
    """
    获取临时文件目录路径
    
    Returns:
        str: 临时文件目录路径
    """
    temp_dir = os.path.join(get_output_dir(), 'temp')
    ensure_dir(temp_dir)
    return temp_dir


def get_formula_dir() -> str:
    """
    获取公式文件目录路径
    
    Returns:
        str: 公式文件目录路径
    """
    formula_dir = os.path.join(get_project_root(), 'formula')
    ensure_dir(formula_dir)
    return formula_dir


def get_indicators_dir() -> str:
    """
    获取技术指标模块目录路径
    
    Returns:
        str: 技术指标模块目录路径
    """
    indicators_dir = os.path.join(get_project_root(), 'indicators')
    ensure_dir(indicators_dir)
    return indicators_dir


def get_file_path(rel_path: str, base_dir: Optional[str] = None) -> str:
    """
    获取文件的绝对路径
    
    Args:
        rel_path: 相对路径
        base_dir: 基础目录，如果为None则使用项目根目录
        
    Returns:
        str: 文件的绝对路径
    """
    if base_dir is None:
        base_dir = get_project_root()
    
    # 如果已经是绝对路径，直接返回
    if os.path.isabs(rel_path):
        return rel_path
    
    return os.path.join(base_dir, rel_path)


def get_analysis_report_dir() -> str:
    """
    获取分析报告目录路径
    
    Returns:
        str: 分析报告目录路径
    """
    report_dir = os.path.join(get_backtest_result_dir(), 'reports')
    ensure_dir(report_dir)
    return report_dir


def ensure_dir_exists(dir_path: str) -> str:
    """
    确保目录存在，不存在则创建
    
    Args:
        dir_path: 目录路径
        
    Returns:
        str: 目录路径
    """
    os.makedirs(dir_path, exist_ok=True)
    return dir_path 