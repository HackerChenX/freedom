#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双向验证系统模块

提供完整的双向验证解决方案，包括：
- 前向验证器（Forward Validator）
- 后向验证器（Backward Validator）
- 验证报告生成器（Validation Report Generator）
- 综合验证系统控制器

符合PMO执行计划的质量标准：
- 验证覆盖率 100%
- 假阳性率 < 5%
- 报告生成时间 < 10秒
"""

from .bidirectional_validation_system import (
    BidirectionalValidationSystem,
    ForwardValidator,
    BackwardValidator,
    ValidationReportGenerator,
    BidirectionalValidationReport,
    ForwardValidationResult,
    BackwardValidationResult,
    ValidationMetrics
)

__version__ = "1.0.0"
__author__ = "Financial Expert Advisor"

# 导出主要类和函数
__all__ = [
    'BidirectionalValidationSystem',
    'ForwardValidator',
    'BackwardValidator',
    'ValidationReportGenerator',
    'BidirectionalValidationReport',
    'ForwardValidationResult',
    'BackwardValidationResult',
    'ValidationMetrics'
]

# 模块级别的配置
DEFAULT_VALIDATION_CONFIG = {
    'forward_validation': {
        'validation_threshold': 0.95,
        'false_positive_threshold': 0.05,
        'min_pattern_match_rate': 0.8
    },
    'backward_validation': {
        'confidence_threshold': 0.95,
        'historical_days': 252,
        'min_coverage_rate': 0.6
    },
    'report_generation': {
        'output_formats': ['json', 'txt'],
        'max_execution_time': 10.0,
        'include_visualizations': False
    },
    'quality_standards': {
        'coverage_rate': 1.0,
        'false_positive_rate': 0.05,
        'max_report_time': 10.0,
        'min_overall_score': 0.75
    }
}

def get_validation_system(config=None):
    """
    获取配置好的双向验证系统实例

    Args:
        config: 可选的配置字典

    Returns:
        BidirectionalValidationSystem: 配置好的验证系统实例
    """
    return BidirectionalValidationSystem()

def validate_strategy_quality(strategy, original_buypoints, selected_stocks,
                            output_dir="./reports"):
    """
    快速验证策略质量的便捷函数

    Args:
        strategy: 策略对象
        original_buypoints: 原始买点数据
        selected_stocks: 选中的股票
        output_dir: 输出目录

    Returns:
        Dict: 验证结果摘要
    """
    validation_system = get_validation_system()
    return validation_system.execute_bidirectional_validation(
        strategy, original_buypoints, selected_stocks, output_dir
    )