import unittest
import pandas as pd

from indicators.rsi import RSI
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator

class TestRSI(unittest.TestCase, IndicatorTestMixin):
    """RSI指标单元测试类"""

    def setUp(self):
        """准备数据和指标实例"""
        self.indicator = RSI(period=14)
        self.expected_columns = ['rsi']
        self.data = TestDataGenerator.generate_sideways_channel(periods=100)

    def test_rsi_overbought(self):
        """测试RSI超买"""
        # 持续上涨应导致RSI进入超买区
        data = TestDataGenerator.generate_steady_trend(start_price=100, end_price=150, periods=50)
        result = self.indicator.calculate(data)
        # 在上涨趋势的末期，RSI应该高于70
        self.assertGreater(result['rsi'].iloc[-1], 70, "RSI在持续上涨后未进入超买区")

    def test_rsi_oversold(self):
        """测试RSI超卖"""
        # 持续下跌应导致RSI进入超卖区
        data = TestDataGenerator.generate_steady_trend(start_price=100, end_price=50, periods=50)
        result = self.indicator.calculate(data)
        # 在下跌趋势的末期，RSI应该低于30
        self.assertLess(result['rsi'].iloc[-1], 30, "RSI在持续下跌后未进入超卖区")

    def test_rsi_neutral_zone(self):
        """测试RSI在中性区域"""
        # 横盘震荡行情，RSI应在30-70之间波动
        # 使用一个波动性更小的数据来测试中性区域
        data = TestDataGenerator.generate_sideways_channel(price_level=100, volatility=1.0, periods=100)
        result = self.indicator.calculate(data)
        # 在震荡行情的中后期，RSI大部分时间应处于中性区
        neutral_rsi = result['rsi'].iloc[30:]
        in_zone_ratio = ((neutral_rsi > 30) & (neutral_rsi < 70)).mean()
        self.assertGreater(in_zone_ratio, 0.9,
                        "RSI在震荡行情中应有90%以上的时间保持在中性区域")

if __name__ == '__main__':
    unittest.main() 