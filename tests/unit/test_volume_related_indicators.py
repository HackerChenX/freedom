"""
量价关系指标单元测试
"""
import unittest
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from indicators.obv import OBV
from indicators.mfi import MFI
from indicators.pvt import PVT
from indicators.vosc import VOSC
from indicators.vr import VR
from indicators.ad import AD


class TestOBV(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = OBV(ma_period=10)
        self.expected_columns = ['obv', 'obv_ma']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])

class TestMFI(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = MFI(period=14)
        self.expected_columns = ['mfi']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 50}
        ])

class TestPVT(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = PVT()
        self.expected_columns = ['pvt']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 50}
        ])

class TestVOSC(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = VOSC(short_period=12, long_period=26)
        self.expected_columns = ['vosc']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 120, 'periods': 50}
        ])

class TestVR(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = VR(period=26, ma_period=6)
        self.expected_columns = ['VR', 'VR_MA']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'price': 100, 'periods': 50}
        ])

class TestAD(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = AD()
        self.expected_columns = ['ad']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 50}
        ])

if __name__ == '__main__':
    unittest.main() 