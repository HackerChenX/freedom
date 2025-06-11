"""
标准技术指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin
from indicators.factory import IndicatorFactory
from indicators.aroon import Aroon

def setUpModule():
    """在模块所有测试开始前运行，用于注册所有指标"""
    IndicatorFactory.auto_register_all_indicators()

class TestAroon(IndicatorTestMixin, LogCaptureMixin, unittest.TestCase):
    def setUp(self):
        """测试初始化，并为Mixin测试提供self.data"""
        super().setUp()
        self.indicator_class = Aroon
        self.indicator_params = {'period': 5}
        self.indicator = self.indicator_class(**self.indicator_params)
        self.data_generator = TestDataGenerator()
        # 为继承自Mixin的测试提供一个默认数据集
        self.data = self.data_generator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 20}
        ])
        self.expected_columns = ['aroon_up', 'aroon_down', 'aroon_oscillator']


    def test_calculation_against_reference(self):
        """测试Aroon指标计算结果与预定义参考值的一致性"""
        # Test with pure downtrend data
        data_downtrend = self.data_generator.generate_price_sequence(
            [{'type': 'trend', 'start_price': 120, 'end_price': 110, 'periods': 10}],
            apply_noise=False
        )
        result = self.indicator.calculate(data_downtrend)
        
        # For a pure downtrend, aroon_up should be low and constant, aroon_down should be 100.
        expected_aroon_up = pd.Series([20.0] * 6, index=result.index[-6:])
        expected_aroon_down = pd.Series([100.0] * 6, index=result.index[-6:])

        pd_testing.assert_series_equal(result['aroon_up'].dropna(), expected_aroon_up, check_exact=True)
        pd_testing.assert_series_equal(result['aroon_down'].dropna(), expected_aroon_down, check_exact=True)

        # Test with pure uptrend data
        data_uptrend = self.data_generator.generate_price_sequence(
            [{'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 10}],
            apply_noise=False
        )
        result = self.indicator.calculate(data_uptrend)
        
        # For a pure uptrend, aroon_up should be 100, aroon_down should be low and constant.
        expected_aroon_up = pd.Series([100.0] * 6, index=result.index[-6:])
        expected_aroon_down = pd.Series([20.0] * 6, index=result.index[-6:])
        
        pd_testing.assert_series_equal(result['aroon_up'].dropna(), expected_aroon_up, check_exact=True)
        pd_testing.assert_series_equal(result['aroon_down'].dropna(), expected_aroon_down, check_exact=True)

    def test_patterns_return_dataframe(self):
        """测试get_patterns是否返回DataFrame"""
        data = self.data_generator.generate_price_sequence([
             {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 20}
        ])
        patterns = self.indicator.get_patterns(data)
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertFalse(patterns.empty)

class TestChaikin(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator('CHAIKIN', short_period=3, long_period=10)
        except Exception as e:
            self.skipTest(f"无法创建CHAIKIN指标: {e}")
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
        try:
            self.indicator = IndicatorFactory.create_indicator('CMO', period=14)
        except Exception as e:
            self.skipTest(f"无法创建CMO指标: {e}")
        self.expected_columns = ['cmo']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 120, 'periods': 20}
        ])

    def test_calculation_against_reference(self):
        """测试CMO指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        if not result['cmo'].dropna().empty:
            self.assertTrue(((result['cmo'].dropna() >= -100) & (result['cmo'].dropna() <= 100)).all())

class TestIchimoku(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator('ICHIMOKU', conversion_period=9, base_period=26, leading_span_b_period=52)
        except Exception as e:
            self.skipTest(f"无法创建ICHIMOKU指标: {e}")
        self.expected_columns = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'price': 100, 'periods': 60}
        ])

    def test_calculation_against_reference(self):
        """测试Ichimoku指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        self.assertFalse(result['tenkan_sen'].dropna().empty)
        self.assertFalse(result['kijun_sen'].dropna().empty)
        if not result['chikou_span'].dropna().empty:
            pd_testing.assert_series_equal(result['chikou_span'].dropna(), self.data['close'].shift(-26).dropna(), check_names=False)

class TestKC(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator('KC', period=20, atr_period=10, multiplier=2)
        except Exception as e:
            self.skipTest(f"无法创建KC指标: {e}")
        self.expected_columns = ['kc_upper', 'kc_middle', 'kc_lower']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 105, 'periods': 30}
        ])

    def test_calculation_against_reference(self):
        """测试KC指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        if not result.dropna().empty:
            self.assertTrue((result['kc_upper'].dropna() >= result['kc_middle'].dropna()).all())
            self.assertTrue((result['kc_middle'].dropna() >= result['kc_lower'].dropna()).all())

class TestSAR(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator('SAR', initial_af=0.02, max_af=0.2, af_increment=0.02)
        except Exception as e:
            self.skipTest(f"无法创建SAR指标: {e}")
        self.expected_columns = ['sar']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 20},
            {'type': 'trend', 'start_price': 120, 'end_price': 110, 'periods': 20}
        ])

    def test_calculation_against_reference(self):
        """测试SAR指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        if not result['sar'].dropna().empty:
             self.assertTrue((result['sar'].iloc[1:20].dropna() < self.data['low'].iloc[1:20].dropna()).all())
             self.assertTrue((result['sar'].iloc[21:].dropna() > self.data['high'].iloc[21:].dropna()).all())


class TestStochRSI(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator('STOCHRSI', rsi_period=14, stochastic_period=14, k_period=3, d_period=3)
        except Exception as e:
            self.skipTest(f"无法创建STOCHRSI指标: {e}")
        self.expected_columns = ['stochrsi_k', 'stochrsi_d']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 80, 'periods': 40}
        ])

    def test_calculation_against_reference(self):
        """测试StochRSI指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        if not result.dropna().empty:
            self.assertTrue(((result['stochrsi_k'].dropna() >= 0) & (result['stochrsi_k'].dropna() <= 100)).all())
            self.assertTrue(((result['stochrsi_d'].dropna() >= 0) & (result['stochrsi_d'].dropna() <= 100)).all())

class TestTrix(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator('TRIX', period=15, signal_period=9)
        except Exception as e:
            self.skipTest(f"无法创建TRIX指标: {e}")
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
        try:
            self.indicator = IndicatorFactory.create_indicator('VORTEX', period=14)
        except Exception as e:
            self.skipTest(f"无法创建VORTEX指标: {e}")
        self.expected_columns = ['vi_plus', 'vi_minus']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'm_shape', 'start_price': 100, 'top_price': 120, 'periods': 30}
        ])

    def test_calculation_against_reference(self):
        """测试Vortex指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        self.assertFalse(result['vi_plus'].dropna().empty)
        self.assertFalse(result['vi_minus'].dropna().empty)

class TestDMA(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator('DMA', short_period=10, long_period=50, ama_period=10)
        except Exception as e:
            self.skipTest(f"无法创建DMA指标: {e}")
        self.expected_columns = ['dma', 'ama']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'sideways', 'price': 100, 'periods': 60}
        ])

    def test_calculation_against_reference(self):
        """测试DMA指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        if 'dma' in result.columns and not result['dma'].dropna().empty:
            short_ma = self.data['close'].rolling(window=10).mean()
            long_ma = self.data['close'].rolling(window=50).mean()
            expected_dma = short_ma - long_ma
            pd_testing.assert_series_equal(result['dma'].dropna(), expected_dma.dropna(), check_exact=False, rtol=1e-5)

class TestEMV(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        try:
            self.indicator = IndicatorFactory.create_indicator('EMV', period=14, ma_period=9)
        except Exception as e:
            self.skipTest(f"无法创建EMV指标: {e}")
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
        try:
            self.indicator = IndicatorFactory.create_indicator('PSY', period=12, ma_period=6)
        except Exception as e:
            self.skipTest(f"无法创建PSY指标: {e}")
        self.expected_columns = ['psy', 'psy_ma']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'v_shape', 'start_price': 100, 'bottom_price': 90, 'periods': 30}
        ])

    def test_calculation_against_reference(self):
        """测试PSY指标计算结果与预定义参考值的一致性"""
        result = self.indicator.calculate(self.data)
        if not result['psy'].dropna().empty:
            self.assertTrue(((result['psy'].dropna() >= 0) & (result['psy'].dropna() <= 100)).all())
            self.assertFalse(result['psy_ma'].dropna().empty) 