"""
波动率及其他指标单元测试
"""
import unittest
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from indicators.complete_indicator_registry import complete_registry

class TestIntradayVolatility(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('VIX')
        self.expected_columns = ['vix', 'vix_signal']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'start_price': 100, 'volatility': 0.02, 'periods': 100}
        ])

class TestStockVIX_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('VIX')
        self.expected_columns = ['vix', 'vix_signal']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 100}
        ])

class TestVIX_Indicators_Test_Volatility_And_Other_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('VIX')
        self.expected_columns = ['vix', 'vix_signal']
        # VIX usually requires options data, here we test if it can run with stock data
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 120, 'periods': 100}
        ])

class TestVolumeRatio_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('VR')
        self.expected_columns = ['vr', 'vr_signal']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 100}
        ])

class TestUnifiedMA_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('MA', periods=[5, 10, 20, 30, 60])
        self.expected_columns = ['MA5', 'MA10', 'MA20', 'MA30', 'MA60']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'start_price': 100, 'volatility': 0.02, 'periods': 100}
        ])

if __name__ == '__main__':
    unittest.main() 