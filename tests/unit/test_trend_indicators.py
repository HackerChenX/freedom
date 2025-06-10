"""
趋势指标单元测试
"""
import unittest
from indicators.ma import MA
from indicators.ema import EMA
from indicators.wma import WMA
from indicators.dmi import DMI
from indicators.atr import ATR
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator

class TestMA(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = MA(periods=[5, 20])
        self.expected_columns = ['MA5', 'MA20']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])

class TestEMA(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = EMA(periods=[12, 26])
        self.expected_columns = ['EMA12', 'EMA26']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])

class TestWMA(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = WMA(periods=[5, 10])
        self.expected_columns = ['WMA5', 'WMA10']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])

class TestDMI(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = DMI(period=14, adx_period=6)
        self.expected_columns = ['PDI', 'MDI', 'ADX', 'ADXR']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 50}
        ])

class TestATR(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        # ATR arugment is period not periods
        self.indicator = ATR(params={'period': 14})
        self.expected_columns = ['TR', 'ATR14']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'price': 100, 'periods': 50}
        ])

if __name__ == '__main__':
    unittest.main() 