import unittest
import pandas as pd

from indicators.kdj import KDJ
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator

class TestKDJ(unittest.TestCase, IndicatorTestMixin):
    """KDJ指标单元测试类"""

    def setUp(self):
        """准备数据和指标实例"""
        self.indicator = KDJ(n=9, m1=3, m2=3)
        self.expected_columns = ['K', 'D', 'J']
        # 使用一个包含多种走势的数据进行通用测试
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30},
            {'type': 'trend', 'start_price': 120, 'end_price': 100, 'periods': 30},
        ])

    def test_golden_cross(self):
        """测试KDJ金叉"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 10}, # 下跌
            {'type': 'trend', 'start_price': 90, 'end_price': 105, 'periods': 15}, # 反弹
        ])
        calculated_data = self.indicator.calculate(data)
        patterns = self.indicator.get_patterns(calculated_data)
        self.assertGreater(patterns['KDJ_GOLDEN_CROSS'].sum(), 0, "未检测到KDJ金叉")

    def test_death_cross(self):
        """测试KDJ死叉"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 20},
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 10}, # 上涨
            {'type': 'trend', 'start_price': 110, 'end_price': 95, 'periods': 15}, # 回调
        ])
        calculated_data = self.indicator.calculate(data)
        patterns = self.indicator.get_patterns(calculated_data)
        self.assertGreater(patterns['KDJ_DEATH_CROSS'].sum(), 0, "未检测到KDJ死叉")

    def test_j_value_overbought(self):
        """测试J值超买"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30}
        ])
        result = self.indicator.calculate(data)
        # J值在持续上涨后应超过100
        self.assertTrue((result['J'].iloc[-10:] > 100).any(), "J值未进入超买区")

    def test_j_value_oversold(self):
        """测试J值超卖"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 80, 'periods': 30}
        ])
        result = self.indicator.calculate(data)
        # J值在持续下跌后应低于0
        self.assertTrue((result['J'].iloc[-10:] < 0).any(), "J值未进入超卖区")

if __name__ == '__main__':
    unittest.main() 