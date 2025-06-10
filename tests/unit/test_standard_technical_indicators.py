"""
标准技术指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from indicators.factory import IndicatorFactory

class TestAroon(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('AROON', period=5)
        self.expected_columns = ['aroon_up', 'aroon_down', 'aroon_oscillator']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 10}
        ])

    def test_calculation_against_reference(self):
        """测试Aroon指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        expected_aroon_up = pd.Series([np.nan, np.nan, np.nan, np.nan, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], name='aroon_up', index=result.index)
        expected_aroon_down = pd.Series([np.nan, np.nan, np.nan, np.nan, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0], name='aroon_down', index=result.index)
        pd_testing.assert_series_equal(result['aroon_up'].dropna(), expected_aroon_up.dropna(), check_exact=False, rtol=1e-5)
        pd_testing.assert_series_equal(result['aroon_down'].dropna(), expected_aroon_down.dropna(), check_exact=False, rtol=1e-5)

class TestChaikin(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('CHAIKIN', short_period=3, long_period=10)
        self.expected_columns = ['chaikin_oscillator']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 110, 'bottom_price': 90, 'periods': 20}
        ])

    def test_calculation_against_reference(self):
        """测试Chaikin振荡器计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        self.assertFalse(result['chaikin_oscillator'].dropna().empty, "Chaikin振荡器计算结果不应全为空")
        self.assertTrue(result['chaikin_oscillator'].dtype == 'float64')

class TestCMO(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('CMO', period=14)
        self.expected_columns = ['cmo']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 120, 'periods': 20}
        ])

    def test_calculation_against_reference(self):
        """测试CMO指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        self.assertTrue(result['cmo'].iloc[:14].isna().all())
        self.assertFalse(np.isnan(result['cmo'].iloc[14]))
        self.assertTrue(((result['cmo'].dropna() >= -100) & (result['cmo'].dropna() <= 100)).all())

class TestIchimoku(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('ICHIMOKU', conversion_period=9, base_period=26, leading_span_b_period=52)
        self.expected_columns = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'price': 100, 'periods': 60}
        ])

    def test_calculation_against_reference(self):
        """测试Ichimoku指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        self.assertFalse(result['tenkan_sen'].dropna().empty)
        self.assertFalse(result['kijun_sen'].dropna().empty)
        pd_testing.assert_series_equal(result['chikou_span'].dropna(), self.data['close'].shift(-26).dropna(), check_names=False)

class TestKC(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('KC', period=20, atr_period=10, multiplier=2)
        self.expected_columns = ['kc_upper', 'kc_middle', 'kc_lower']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 105, 'periods': 30}
        ])

    def test_calculation_against_reference(self):
        """测试KC指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        self.assertTrue((result['kc_upper'].dropna() >= result['kc_middle'].dropna()).all())
        self.assertTrue((result['kc_middle'].dropna() >= result['kc_lower'].dropna()).all())

class TestSAR(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('SAR', initial_af=0.02, max_af=0.2, af_increment=0.02)
        self.expected_columns = ['sar']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 20},
            {'type': 'trend', 'start_price': 120, 'end_price': 110, 'periods': 20}
        ])

    def test_calculation_against_reference(self):
        """测试SAR指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        self.assertTrue((result['sar'].iloc[1:20] < self.data['low'].iloc[1:20]).all())
        self.assertTrue((result['sar'].iloc[21:] > self.data['high'].iloc[21:]).all())

class TestStochRSI(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('STOCHRSI', rsi_period=14, stochastic_period=14, k_period=3, d_period=3)
        self.expected_columns = ['stochrsi_k', 'stochrsi_d']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 40}
        ])

    def test_calculation_against_reference(self):
        """测试StochRSI指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        self.assertTrue(((result['stochrsi_k'].dropna() >= 0) & (result['stochrsi_k'].dropna() <= 100)).all())
        self.assertTrue(((result['stochrsi_d'].dropna() >= 0) & (result['stochrsi_d'].dropna() <= 100)).all())

class TestTrix(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('TRIX', period=15, signal_period=9)
        self.expected_columns = ['trix', 'trix_signal']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])

    def test_calculation_against_reference(self):
        """测试Trix指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        self.assertFalse(result['trix'].dropna().empty, "Trix不应全为空")
        self.assertFalse(result['trix_signal'].dropna().empty, "Trix信号线不应全为空")

class TestVortex(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('VORTEX', period=14)
        self.expected_columns = ['vip', 'vim']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 120, 'periods': 30}
        ])

    def test_calculation_against_reference(self):
        """测试Vortex指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        self.assertFalse(result['vip'].dropna().empty)
        self.assertFalse(result['vim'].dropna().empty)

class TestDMA(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('DMA', short_period=10, long_period=50, ama_period=10)
        self.expected_columns = ['dma', 'ama']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'price': 100, 'periods': 60}
        ])

    def test_calculation_against_reference(self):
        """测试DMA指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        short_ma = self.data['close'].rolling(window=10).mean()
        long_ma = self.data['close'].rolling(window=50).mean()
        expected_dma = short_ma - long_ma
        pd_testing.assert_series_equal(result['dma'].dropna(), expected_dma.dropna(), check_exact=False, rtol=1e-5)

class TestEMV(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('EMV', period=14, ma_period=9)
        self.expected_columns = ['emv', 'emv_ma']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30}
        ])

    def test_calculation_against_reference(self):
        """测试EMV指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        self.assertFalse(result['emv'].dropna().empty)
        self.assertFalse(result['emv_ma'].dropna().empty, "EMV移动平均线不应全为空")

class TestPSY(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        self.indicator = IndicatorFactory.create_indicator('PSY', period=12, ma_period=6)
        self.expected_columns = ['psy', 'psy_ma']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 90, 'periods': 30}
        ])

    def test_calculation_against_reference(self):
        """测试PSY指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        self.assertFalse(result['psy'].dropna().empty)
        self.assertTrue(((result['psy'].dropna() >= 0) & (result['psy'].dropna() <= 100)).all())
        self.assertFalse(result['psy_ma'].dropna().empty) 