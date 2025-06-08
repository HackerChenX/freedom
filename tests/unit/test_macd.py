"""
MACD指标单元测试
"""

import unittest
import pandas as pd

from indicators.macd import MACD
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator


class TestMACD(unittest.TestCase, IndicatorTestMixin):
    """MACD指标单元测试类，继承自TestCase和IndicatorTestMixin。"""

    def setUp(self):
        """
        为所有测试准备数据和指标实例。
        这个方法在每个测试方法运行前都会被调用。
        """
        # 指标实例
        self.indicator = MACD(fast_period=12, slow_period=26, signal_period=9)
        
        # 定义预期输出列，供基类测试使用
        self.expected_columns = ['macd_line', 'macd_signal', 'macd_histogram']
        
        # 生成并赋值通用测试数据，供基类测试使用
        # 使用一个包含多种形态的复杂序列
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 30},
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 90, 'periods': 40},
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 10},
            {'type': 'm_shape', 'start_price': 100, 'top_price': 110, 'periods': 40},
        ])

    def test_golden_cross_pattern(self):
        """测试金叉形态的精确定位"""
        # 为金叉场景生成特定数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 30}, # 稳定EMA
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 10},  # 快速下跌
            {'type': 'trend', 'start_price': 90, 'end_price': 105, 'periods': 20}  # 快速反弹
        ])
        patterns = self.indicator.get_patterns(data)
        
        self.assertGreaterEqual(patterns['MACD_GOLDEN_CROSS'].sum(), 1, "金叉信号未被检测到")
        
        # 交叉点应该在反弹开始之后
        first_cross_date = patterns[patterns['MACD_GOLDEN_CROSS']].index[0]
        rebound_start_date = data.index[40]
        self.assertGreater(first_cross_date, rebound_start_date, "金叉发生在预期时间之前")

    def test_death_cross_pattern(self):
        """测试死叉形态的精确定位"""
        # 为死叉场景生成特定数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 30}, # 稳定EMA
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 10}, # 快速上涨
            {'type': 'trend', 'start_price': 110, 'end_price': 95, 'periods': 20}   # 快速下跌
        ])
        patterns = self.indicator.get_patterns(data)

        self.assertGreaterEqual(patterns['MACD_DEATH_CROSS'].sum(), 1, "死叉信号未被检测到")

        first_cross_date = patterns[patterns['MACD_DEATH_CROSS']].index[0]
        decline_start_date = data.index[40]
        self.assertGreater(first_cross_date, decline_start_date, "死叉发生在预期时间之前")

    def test_bullish_divergence_pattern(self):
        """测试底背离形态的检测"""
        # 为底背离场景生成特定数据
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 70}, # 稳定EMA
            {'type': 'trend', 'start_price': 100, 'end_price': 95, 'periods': 10},  # 第一个低点
            {'type': 'trend', 'start_price': 95, 'end_price': 98, 'periods': 5},    # 小幅反弹
            {'type': 'trend', 'start_price': 98, 'end_price': 94, 'periods': 10}   # 第二个更低的低点
        ])
        patterns = self.indicator.get_patterns(data)
        
        divergence_range = patterns.iloc[85:95]
        self.assertTrue(divergence_range['MACD_BULLISH_DIVERGENCE'].any(), "在预设区间未检测到底背离")


if __name__ == '__main__':
    unittest.main() 