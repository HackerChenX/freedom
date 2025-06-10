"""
震荡指标单元测试
"""
import unittest
from indicators.rsi import RSI
from indicators.kdj import KDJ
from indicators.cci import CCI
from indicators.wr import WR
from indicators.bias import BIAS
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator

class TestRSI(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = RSI(period=14)
        self.expected_columns = ['rsi']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 50}
        ])

class TestKDJ(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = KDJ(n=9, m1=3, m2=3)
        self.expected_columns = ['K', 'D', 'J']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 120, 'periods': 50}
        ])

class TestCCI(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = CCI(period=14)
        self.expected_columns = ['CCI']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 50}
        ])

class TestWR(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = WR(period=14)
        self.expected_columns = ['wr']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 50}
        ])

class TestBIAS(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = BIAS(periods=[6, 12, 24])
        self.expected_columns = ['BIAS6', 'BIAS12', 'BIAS24']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 50}
        ])

if __name__ == '__main__':
    unittest.main() 