"""
特殊指标单元测试
"""
import unittest
from indicators.complete_indicator_registry import complete_registry
from indicators.complete_indicator_registry import complete_registry
from indicators.complete_indicator_registry import complete_registry
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator

class TestVIX_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp_IndicatorsTestspecialindicators(self):
        self.indicator = VIX(period=14)
        self.expected_columns = ['vix', 'vix_smooth']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 50}
        ])

class TestSAR_Indicators(unittest.TestCase, IndicatorTestMixin):
    def setUp_IndicatorsTestspecialindicators(self):
        self.indicator = SAR(acceleration=0.02, maximum=0.2)
        self.expected_columns = ['sar']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 50}
        ])

class TestTRIX(unittest.TestCase, IndicatorTestMixin):
    def setUp_IndicatorsTestspecialindicators(self):
        self.indicator = TRIX(n=12, m=9)
        self.expected_columns = ['TRIX', 'MATRIX']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 120, 'periods': 50}
        ])

if __name__ == '__main__':
    unittest.main() 