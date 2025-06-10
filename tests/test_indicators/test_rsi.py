import unittest
import pandas as pd
import numpy as np
import os

# Add project root to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from indicators.rsi import RSI
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator

class TestRSI(unittest.TestCase, IndicatorTestMixin):
    def setUp(self):
        """Set up test data for RSI indicator."""
        # Create a sample DataFrame that can trigger various RSI patterns
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=50, freq='D'))
        
        # Base data
        close_prices = 100 + np.sin(np.linspace(0, 10, 50)) * 5 + np.linspace(0, 20, 50)
        
        data = {
            'date': dates,
            'open': close_prices - 1,
            'high': close_prices + 2,
            'low': close_prices - 2,
            'close': close_prices,
            'volume': np.random.randint(100, 200, size=50) * 100
        }
        
        self.data_df = pd.DataFrame(data).set_index('date')
        
        # --- Create specific scenarios ---
        
        # 1. Oversold and Golden Cross scenario
        oversold_data = self.data_df.copy()
        oversold_data.loc[oversold_data.index[15:20], 'close'] = [85, 80, 78, 82, 85]
        self.oversold_df = oversold_data

        # 2. Overbought and Death Cross scenario
        overbought_data = self.data_df.copy()
        overbought_data.loc[overbought_data.index[35:40], 'close'] = [130, 135, 138, 134, 130]
        self.overbought_df = overbought_data
        
        # 3. Bullish Divergence scenario
        divergence_data = self.data_df.copy()
        # Price makes a lower low, but RSI should make a higher low
        divergence_data.loc[divergence_data.index[20:25], 'close'] = [95, 90, 88, 89, 91] 
        divergence_data.loc[divergence_data.index[25:30], 'close'] = [92, 88, 86, 87, 89] # Price makes a lower low than before
        self.bullish_divergence_df = divergence_data
        
        self.rsi_indicator = RSI(period=14, overbought=70, oversold=30)

    def test_get_patterns_returns_dataframe(self):
        """Test that get_patterns returns a DataFrame."""
        patterns = self.rsi_indicator.get_patterns(self.data_df)
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertEqual(patterns.index.name, 'date')

    def test_oversold_pattern(self):
        """Test the RSI_OVERSOLD pattern."""
        rsi_df = self.rsi_indicator.calculate(self.oversold_df)
        patterns = self.rsi_indicator.get_patterns(rsi_df)
        
        # Check if OVERSOLD is detected where RSI is below 30
        oversold_days = rsi_df['rsi'] < 30
        self.assertTrue(patterns['RSI_OVERSOLD'][oversold_days].any())
        self.assertFalse(patterns['RSI_OVERSOLD'][~oversold_days].any())

    def test_overbought_pattern(self):
        """Test the RSI_OVERBOUGHT pattern."""
        rsi_df = self.rsi_indicator.calculate(self.overbought_df)
        patterns = self.rsi_indicator.get_patterns(rsi_df)
        
        # Check if OVERBOUGHT is detected where RSI is above 70
        overbought_days = rsi_df['rsi'] > 70
        self.assertTrue(patterns['RSI_OVERBOUGHT'][overbought_days].any())
        self.assertFalse(patterns['RSI_OVERBOUGHT'][~overbought_days].any())

    def test_cross_above_50_pattern(self):
        """Test the RSI_CROSS_ABOVE_50 pattern."""
        rsi_df = self.rsi_indicator.calculate(self.oversold_df) # Use data that crosses 50
        patterns = self.rsi_indicator.get_patterns(rsi_df)
        
        rsi_series = rsi_df['rsi']
        # Find where RSI crosses above 50
        cross_above_mask = (rsi_series > 50) & (rsi_series.shift(1) <= 50)
        
        self.assertTrue(patterns['RSI_CROSS_ABOVE_50'][cross_above_mask].all())
        # Ensure it's not True elsewhere
        self.assertFalse(patterns['RSI_CROSS_ABOVE_50'][~cross_above_mask].any())

    def test_bullish_divergence_pattern(self):
        """Test the RSI_BULLISH_DIVERGENCE pattern."""
        # Note: Vectorized divergence is an approximation and might be tricky to test precisely.
        # This test checks if the pattern is detected in a crafted scenario.
        rsi_df = self.rsi_indicator.calculate(self.bullish_divergence_df)
        patterns = self.rsi_indicator.get_patterns(rsi_df)
        
        # We expect a divergence signal somewhere in the manipulated range
        self.assertTrue(patterns['RSI_BULLISH_DIVERGENCE'].any(), "Bullish divergence was not detected.")

if __name__ == '__main__':
    unittest.main() 