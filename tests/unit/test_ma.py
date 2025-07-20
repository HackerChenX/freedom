import unittest
import pandas as pd
import numpy as np

from indicators.complete_indicator_registry import complete_registry
from indicators.complete_indicator_registry import complete_registry

class Test_mAIndicator(unittest.TestCase):
    def setUp(self):
        """Set up test data and indicator."""
        # Longer data to avoid NaN issues with longer period MAs
        close_prices = [
            110, 108, 106, 104, 102, 100, 98, 96, 94, 92, # Downtrend
            90, 91, 93, 95, 98, 101, 104, 107, 110, 113, # Bottoming and uptrend
            116, 119, 122, 125, 128, 131, 134, 137, 140, 143 # Strong uptrend
        ]
        data = {
            'high': [p + 2 for p in close_prices],
            'low': [p - 2 for p in close_prices],
            'close': close_prices
        }
        self.df = pd.DataFrame(data)
        self.periods = [5, 10]
        self.ma_indicator = complete_registry.create_indicator('MA', periods=self.periods)
        self.indicator_df = self.ma_indicator.calculate(self.df)

    def test_initialization(self):
        """Test indicator initialization and pattern registration."""
        self.assertEqual(self.ma_indicator.name, "MA")
        self.assert_equal(self.ma_indicator.periods, self.periods)
        
        registry = Pattern_registry()
        p_short, p_medium = sorted(self.periods)[:2]
        # The normalization logic in PatternRegistry prepends the indicator name if not already present.
        # Since our pattern_id in MA.py does not start with "MA_", it gets prepended.
        # Let's check the base indicator register method. It gets indicator_id='MA'.
        # It calls registry.register with pattern_id=f"MA_{p_short...}", indicator_id='MA'.
        # The registry sees that pattern_id starts with indicator_id, so it does not prepend again.
        expected_pattern_id = f"MA_{p_short}_{p_medium}_GOLDEN_CROSS".upper()
        # Let's correct MA's registration to not include the prefix itself.
        # No, let's fix the test. The pattern in registry will be MA_5_10_GOLDEN_CROSS
        self.assert_in(expected_pattern_id, [p.upper() for p in registry.get_all_pattern_ids()])

    def test_calculate_ma(self):
        """Test calculation of moving averages."""
        self.assertIn('SMA5', self.indicator_df.columns)
        self.assertIn('SMA10', self.indicator_df.columns)
        
        expected_ma5_at_15 = self.df['close'].iloc[11:16].mean()
        self.assertAlmostEqual(self.indicator_df['SMA5'].iloc[15], expected_ma5_at_15)
        
        expected_ma10_at_20 = self.df['close'].iloc[11:21].mean()
        self.assertAlmostEqual(self.indicator_df['SMA10'].iloc[20], expected_ma10_at_20)

    def test_get_patterns_Ma(self):
        """Test pattern recognition."""
        patterns = self.ma_indicator.get_patterns(self.indicator_df)
        p_short, p_medium = sorted(self.periods)[:2]
        
        golden_cross_col = f"MA_{p_short}_{p_medium}_GOLDEN_CROSS"
        
        self.assert_in(golden_cross_col, patterns.columns)
        
        # Golden Cross at index 15
        self.assert_true(patterns[golden_cross_col].iloc[15])
        
        # Check for bullish arrangement at the end
        self.assertTrue(patterns["MA_BULLISH_ARRANGEMENT"].iloc[-1])

    def test_calculate_raw_score_Ma(self):
        """Test score calculation."""
        score = self.ma_indicator.calculate_raw_score(self.indicator_df)
        self.assert_is_instance(score, pd.Series)
        
        # At the end of the data (strong uptrend), we expect a high score
        self.assert_greater(score.iloc[-1], 80)

        # At the beginning of the data (downtrend), we expect a low score
        self.assert_less(score.iloc[9], 20)

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        score = self.ma_indicator.calculate_raw_score(self.indicator_df)
        patterns = self.ma_indicator.get_patterns(self.indicator_df)

        confidence = self.ma_indicator.calculate_confidence(score, patterns, {})
        self.assert_is_instance(confidence, float)

        # Confidence should be between 0 and 1
        self.assert_greater_equal(confidence, 0.0)
        self.assert_less_equal(confidence, 1.0)

    def test_set_parameters(self):
        """Test setting new parameters on an existing indicator."""
        ma_indicator = complete_registry.create_indicator('MA', periods=[5, 10])
        ma_indicator.set_parameters(periods=[20, 40])
        self.assert_equal(ma_indicator.periods, [20, 40])

        registry = Pattern_registry()
        all_patterns = [p.upper() for p in registry.get_all_pattern_ids()]
        self.assertIn('MA_20_40_GOLDEN_CROSS', all_patterns)
        
        # Use a long enough local dataframe to avoid NaN issues
        local_df = pd.DataFrame({'close': range(100), 'high': range(1, 101), 'low': range(-1, 99)})
        new_df = ma_indicator.calculate(local_df)

        self.assertIn('SMA20', new_df.columns)
        self.assertIn('SMA40', new_df.columns)
        self.assertNotIn('SMA5', new_df.columns)
        self.assertNotIn('SMA10', new_df.columns)

if __name__ == '__main__':
    unittest.main() 