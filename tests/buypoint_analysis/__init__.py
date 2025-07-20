#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
买点分析测试模块

提供买点分析功能的综合测试套件，包括：
- 40+种技术形态的识别准确性测试
- 正面和负面测试案例
- 性能基准测试
- 集成测试
- 详细的测试报告生成

测试目标：
- 确保重构后的买点分析系统保持与之前版本相同的识别准确性
- 验证所有112个技术指标的正常工作
- 确保系统能够处理4000+股票的批量分析需求
"""

from .test_buypoint_comprehensive import BuyPointAnalysisTestSuite
from .enhanced_test_data_generator import EnhancedTestDataGenerator
from .run_buypoint_tests import BuyPointTestRunner

__version__ = "1.0.0"
__author__ = "Stock Selection System Team"

__all__ = [
    'BuyPointAnalysisTestSuite',
    'EnhancedTestDataGenerator', 
    'BuyPointTestRunner'
]

# 测试套件信息
TEST_SUITE_INFO = {
    'name': '买点分析功能综合测试套件',
    'version': __version__,
    'description': '验证重构后买点分析系统的形态识别准确性',
    'supported_patterns': 40,
    'supported_indicators': 112,
    'test_categories': [
        'trend_patterns',      # 趋势形态
        'oscillator_patterns', # 振荡器形态
        'momentum_patterns',   # 动量形态
        'volume_patterns',     # 成交量形态
        'volatility_patterns', # 波动性形态
        'candlestick_patterns' # K线形态
    ],
    'test_types': [
        'unit_tests',          # 单元测试
        'integration_tests',   # 集成测试
        'negative_tests',      # 负面测试
        'performance_tests'    # 性能测试
    ]
}

def get_test_suite_info():
    """获取测试套件信息"""
    return TEST_SUITE_INFO

def get_supported_patterns():
    """获取支持的技术形态列表"""
    from .test_buypoint_comprehensive import BuyPointAnalysisTestSuite
    
    patterns = []
    for category_patterns in BuyPointAnalysisTestSuite.pattern_categories.values():
        patterns.extend(category_patterns)
    
    return patterns

def get_test_categories():
    """获取测试类别"""
    return TEST_SUITE_INFO['test_categories']
