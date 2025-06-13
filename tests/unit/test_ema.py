import unittest
import pandas as pd
import numpy as np
import logging

from indicators.ema import EMA
from indicators.pattern_registry import PatternRegistry

class TestEMA(unittest.TestCase):
    def setUp(self):
        # Suppress all logging outputs during tests
        logging.disable(logging.CRITICAL)
        
        # Create sample data for testing
        self.data = self._create_test_data()
        self.ema_indicator = EMA(periods=[5, 10])
        # Get a clean instance of the registry for testing
        self.registry = PatternRegistry()
        self.registry.clear_registry()

    def tearDown(self):
        # Re-enable logging after tests
        logging.disable(logging.NOTSET)

    def _create_test_data(self, trend='up'):
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=30))
        close_prices = pd.Series(np.linspace(100, 150, 30), index=dates)
        if trend == 'down':
            close_prices = pd.Series(np.linspace(150, 100, 30), index=dates)
        elif trend == 'sideways':
            close_prices = pd.Series(120 + np.sin(np.arange(30)) * 5, index=dates)
        
        data = pd.DataFrame({
            'open': close_prices - 2,
            'high': close_prices + 2,
            'low': close_prices - 2,
            'close': close_prices
        })
        return data

    def test_calculate(self):
        """Test the calculation of EMA values."""
        df = self.ema_indicator.calculate(self.data.copy())
        self.assertIn('EMA5', df.columns)
        self.assertIn('EMA10', df.columns)
        self.assertFalse(df['EMA5'].isnull().all())
        self.assertFalse(df['EMA10'].isnull().all())

    def test_calculate_raw_score(self):
        """Test the raw score calculation."""
        # Test with uptrend data
        uptrend_data = self._create_test_data('up')
        df_up = self.ema_indicator.calculate(uptrend_data)
        scores_up = self.ema_indicator.calculate_raw_score(df_up)
        self.assertTrue((scores_up >= 0).all() and (scores_up <= 100).all())
        self.assertGreater(scores_up.iloc[-1], 60, "Score should be high in an uptrend")

        # Test with downtrend data
        downtrend_data = self._create_test_data('down')
        df_down = self.ema_indicator.calculate(downtrend_data)
        scores_down = self.ema_indicator.calculate_raw_score(df_down)
        self.assertTrue((scores_down >= 0).all() and (scores_down <= 100).all())
        self.assertLess(scores_down.iloc[-1], 40, "Score should be low in a downtrend")

    def test_get_patterns(self):
        """Test pattern identification."""
        # Create data with a golden cross
        close_prices = [100, 99, 98, 97, 96, 98, 100, 102, 104, 106, 108, 110]
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(close_prices)))
        cross_data = pd.DataFrame({'close': close_prices, 'high': close_prices, 'low': close_prices}, index=dates)
        
        indicator = EMA(periods=[3, 6])
        df = indicator.calculate(cross_data)
        patterns = indicator.get_patterns(df)

        self.assertIn('EMA_3_6_GOLDEN_CROSS', patterns.columns)
        self.assertTrue(patterns['EMA_3_6_GOLDEN_CROSS'].any())

        # Create data with a death cross
        death_cross_prices = [110, 108, 106, 104, 102, 100, 98, 96, 95, 94, 93, 92]
        death_cross_dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(death_cross_prices)))
        cross_data = pd.DataFrame({'close': death_cross_prices, 'high': death_cross_prices, 'low': death_cross_prices}, index=death_cross_dates)
        
        df = indicator.calculate(cross_data)
        patterns = indicator.get_patterns(df)
        self.assertIn('EMA_3_6_DEATH_CROSS', patterns.columns)
        self.assertTrue(patterns['EMA_3_6_DEATH_CROSS'].any(), "Death cross was not detected")

    def test_register_patterns(self):
        """Test if patterns are registered correctly."""
        # The indicator should register its patterns upon instantiation
        indicator = EMA(periods=[5, 10])
        registered_patterns = self.registry.get_patterns_by_indicator('EMA')
        
        expected_patterns = [
            'EMA_5_10_GOLDEN_CROSS',
            'EMA_5_10_DEATH_CROSS',
            'EMA_BULLISH_ARRANGEMENT',
            'EMA_BEARISH_ARRANGEMENT'
        ]
        
        for pattern_id in expected_patterns:
            self.assertIn(pattern_id, registered_patterns)

if __name__ == '__main__':
    unittest.main() 