"""
成交量指标单元测试
"""
import unittest
from indicators.vol import VOL
from indicators.obv import OBV
from indicators.mfi import MFI
from indicators.ad import AD
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator

class TestVOL(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = VOL()
        self.expected_columns = ['vol', 'vol_ma5', 'vol_ma10']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])

class TestOBV(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = OBV()
        self.expected_columns = ['obv']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 90, 'periods': 50}
        ])

class TestMFI(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = MFI(period=14)
        self.expected_columns = ['mfi']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 110, 'periods': 50}
        ])

class TestAD(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = AD()
        self.expected_columns = ['AD', 'AD_MA']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 100, 'periods': 50}
        ])

if __name__ == '__main__':
    unittest.main() 