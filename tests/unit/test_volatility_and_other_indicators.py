"""
波动率及其他指标单元测试
"""
import unittest
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from indicators.factory import IndicatorFactory

class TestIntradayVolatility(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('INTRADAYVOLATILITY')
        self.expected_columns = ['volatility', 'volatility_ma', 'relative_volatility']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'price': 100, 'periods': 100}
        ])

class TestStockVIX(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('STOCKVIX')
        self.expected_columns = ['stock_vix']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 100}
        ])

class TestVIX(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('VIX')
        self.expected_columns = ['vix']
        # VIX usually requires options data, here we test if it can run with stock data
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 120, 'periods': 100}
        ])

class TestVolumeRatio(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('VOLUMERATIO')
        self.expected_columns = ['volume_ratio']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 100}
        ])

class TestUnifiedMA(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('UNIFIEDMA')
        self.expected_columns = ['MA5', 'MA10', 'MA20', 'MA30', 'MA60']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'price': 100, 'periods': 100}
        ])

if __name__ == '__main__':
    unittest.main() 