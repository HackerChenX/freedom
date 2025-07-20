"""
ZXM系列指标单元测试
"""
import unittest
from indicators.complete_indicator_registry import complete_registry
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator

class TestZXMBSAbsorb(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('ZXM_BS_ABSORB')
        self.expected_columns = ['XG']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])

class TestZXMDailyMACD(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        super().setUp()
        self.indicator = complete_registry.create_indicator('ZXM_DAILY_MACD')
        self.expected_columns = ['XG']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 100}
        ])

if __name__ == '__main__':
    unittest.main() 