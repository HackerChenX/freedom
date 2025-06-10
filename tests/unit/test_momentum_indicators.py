"""
动量指标单元测试
"""
import unittest
from indicators.momentum import Momentum
from indicators.mtm import MTM
from indicators.roc import ROC
from indicators.rsima import RSIMA
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator

class TestMomentum(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = Momentum(period=12, signal_period=6)
        self.expected_columns = ['mtm', 'signal']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 50}
        ])

class TestMTM(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = MTM(period=12, ma_period=6)
        self.expected_columns = ['mtm', 'mtmma']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 50}
        ])

class TestROC(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = ROC(period=12, ma_period=6)
        self.expected_columns = ['roc', 'rocma']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 120, 'periods': 50}
        ])

class TestRSIMA(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = RSIMA(rsi_period=14, ma_periods=[6])
        self.expected_columns = ['rsi', 'rsi_ma6']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 50}
        ])

if __name__ == '__main__':
    unittest.main() 