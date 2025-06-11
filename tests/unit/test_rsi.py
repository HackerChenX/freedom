import unittest
import pandas as pd
from indicators.factory import IndicatorFactory
from indicators.rsi import RSI
from tests.helper.log_capture import LogCaptureMixin
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator


class TestRSI(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """测试RSI指标"""

    def setUp(self):
        """测试准备"""
        super().setUp()
        self.indicator = IndicatorFactory.create_indicator('RSI', period=14)
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'periods': 100, 'start_price': 100, 'volatility': 0.02}
        ])
        self.expected_columns = ['rsi']

    def test_rsi_overbought(self):
        """测试RSI超买"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'periods': 50, 'start_price': 100, 'end_price': 150}
        ])
        result = self.indicator.calculate(data)
        # 在强劲上涨趋势中，最后几个周期的RSI应该高于70
        self.assertTrue((result['rsi'].iloc[-5:] > 70).all())

    def test_rsi_oversold(self):
        """测试RSI超卖"""
        data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'periods': 50, 'start_price': 150, 'end_price': 100}
        ])
        result = self.indicator.calculate(data)
        # 在强劲下跌趋势中，最后几个周期的RSI应该低于30
        self.assertTrue((result['rsi'].iloc[-5:] < 30).all())

    def test_rsi_neutral_zone(self):
        """测试RSI在中性区域"""
        result = self.indicator.calculate(self.data)
        # 在横盘行情中，RSI应该在30和70之间波动
        self.assertTrue(((result['rsi'].dropna() > 30) & (result['rsi'].dropna() < 70)).all())

if __name__ == '__main__':
    unittest.main() 