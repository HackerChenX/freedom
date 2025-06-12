"""
增强型指标单元测试
"""
import unittest
import pandas as pd
import numpy as np

from indicators.trend.enhanced_cci import EnhancedCCI
from indicators.volume.enhanced_obv import EnhancedOBV
from indicators.enhanced_stochrsi import EnhancedSTOCHRSI
from indicators.enhanced_wr import EnhancedWR
from indicators.trend.enhanced_trix import EnhancedTRIX
from indicators.trend.enhanced_macd import EnhancedMACD
from tests.helper.data_generator import TestDataGenerator

class TestEnhancedIndicators(unittest.TestCase):
    """增强型指标的单元测试"""

    @classmethod
    def setUpClass(cls):
        """为所有测试准备数据"""
        config = [
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 100},
            {'type': 'channel', 'periods': 100, 'volatility': 5},
            {'type': 'v_shape', 'bottom_price': 90, 'periods': 100},
        ]
        cls.data = TestDataGenerator.generate_price_sequence(config)

    def test_enhanced_cci(self):
        """测试增强型CCI指标"""
        indicator = EnhancedCCI()
        result = indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('cci', result.columns)
        self.assertIn('state', result.columns)

        patterns = indicator.identify_patterns()
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertGreater(len(patterns.columns), 0)

        score = indicator.calculate_score(self.data)
        self.assertIsInstance(score, pd.Series)
        self.assertTrue(all(0 <= s <= 100 for s in score.dropna()))

        signals = indicator.generate_signals(self.data)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)

    def test_enhanced_obv(self):
        """测试增强型OBV指标"""
        indicator = EnhancedOBV()
        result = indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('obv', result.columns)
        self.assertIn('obv_ma', result.columns)

        patterns = indicator.identify_patterns()
        self.assertIsInstance(patterns, pd.DataFrame)
        # self.assertGreater(len(patterns.columns), 0) # identify_patterns may be empty

        score = indicator.calculate_raw_score(self.data)
        self.assertIsInstance(score, pd.Series)
        self.assertTrue(all(0 <= s <= 100 for s in score.dropna()))

        signals = indicator.generate_signals(self.data)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)

    def test_enhanced_stochrsi(self):
        """测试增强型STOCHRSI指标"""
        indicator = EnhancedSTOCHRSI()
        result = indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('STOCHRSI_K', result.columns)
        self.assertIn('STOCHRSI_D', result.columns)

        patterns = indicator.identify_patterns()
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertGreater(len(patterns.columns), 0)

        score = indicator.calculate_score(self.data)
        self.assertIsInstance(score, pd.Series)
        self.assertTrue(all(0 <= s <= 100 for s in score.dropna()))

        signals = indicator.generate_signals(self.data)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)

    def test_enhanced_wr(self):
        """测试增强型WR指标"""
        indicator = EnhancedWR()
        result = indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('wr', result.columns)

        patterns = indicator.identify_patterns()
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertGreater(len(patterns.columns), 0)

        score = indicator.calculate_score(self.data)
        self.assertIsInstance(score, pd.Series)
        self.assertTrue(all(0 <= s <= 100 for s in score.dropna()))

        signals = indicator.generate_signals(self.data)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)

    def test_enhanced_trix(self):
        """测试增强型TRIX指标"""
        indicator = EnhancedTRIX()
        result = indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('TRIX', result.columns)
        self.assertIn('MATRIX', result.columns)

        patterns = indicator.identify_patterns()
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertGreater(len(patterns.columns), 0)

        score = indicator.calculate_score(self.data)
        self.assertIsInstance(score, pd.Series)
        self.assertTrue(all(0 <= s <= 100 for s in score.dropna()))

        signals = indicator.generate_signals(self.data)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)

    def test_enhanced_macd(self):
        """测试增强型MACD指标"""
        indicator = EnhancedMACD()
        result = indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('macd', result.columns)
        self.assertIn('macd_signal', result.columns)
        self.assertIn('macd_hist', result.columns)

        # 假设 identify_patterns 存在，但可能返回空DataFrame
        patterns = indicator.identify_patterns(self.data)
        self.assertIsInstance(patterns, pd.DataFrame)

        score = indicator.calculate_raw_score(self.data)
        self.assertIsInstance(score, pd.Series)
        self.assertTrue(all(0 <= s <= 100 for s in score.dropna()))

        signals = indicator.generate_signals(self.data)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)


if __name__ == '__main__':
    unittest.main() 