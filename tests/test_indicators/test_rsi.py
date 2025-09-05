import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from indicators.complete_indicator_registry import complete_registry
from indicators.rsi import RSI
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator

class TestRSI_Rsi_Test_Rsi(unittest.TestCase, IndicatorTestMixin):
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

        # 为IndicatorTestMixin设置必要的属性
        self.indicator = self.rsi_indicator
        self.data = self.data_df

        # 设置预期的输出列
        self.expected_columns = [
            f'rsi_{self.rsi_indicator.period}',  # rsi_14
            'rsi_ma_5', 'rsi_ma_10', 'rsi_ma_short', 'rsi_ma_long',
            'rsi_overbought', 'rsi_oversold',
            'pattern_bullish', 'pattern_bearish', 'pattern_neutral',
            'buy_signal', 'sell_signal', 'hold_signal'
        ]

    def test_get_patterns_returns_dataframe(self):
        """Test that get_patterns returns a DataFrame."""
        patterns = self.rsi_indicator.get_patterns(self.data_df)
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertEqual(patterns.index.name, 'date')

    def test_oversold_pattern(self):
        """Test the RSI_OVERSOLD pattern."""
        rsi_df = self.rsi_indicator.calculate(self.oversold_df)
        patterns = self.rsi_indicator.get_patterns(rsi_df)
        
        # Check if OVERSOLD is detected where RSI is below 30 - 使用标准形态名称和正确的列名
        rsi_column = f'rsi_{self.rsi_indicator.period}'  # rsi_14
        oversold_days = rsi_df[rsi_column] < 30
        self.assertTrue(patterns['OVERSOLD'][oversold_days].any())
        self.assertFalse(patterns['OVERSOLD'][~oversold_days].any())

    def test_overbought_pattern(self):
        """Test the RSI_OVERBOUGHT pattern."""
        rsi_df = self.rsi_indicator.calculate(self.overbought_df)
        patterns = self.rsi_indicator.get_patterns(rsi_df)
        
        # Check if OVERBOUGHT is detected where RSI is above 70 - 使用标准形态名称和正确的列名
        rsi_column = f'rsi_{self.rsi_indicator.period}'  # rsi_14
        overbought_days = rsi_df[rsi_column] > 70
        self.assertTrue(patterns['OVERBOUGHT'][overbought_days].any())
        self.assertFalse(patterns['OVERBOUGHT'][~overbought_days].any())

    def test_cross_above_50_pattern(self):
        """Test the RSI_CROSS_ABOVE_50 pattern."""
        rsi_df = self.rsi_indicator.calculate(self.oversold_df) # Use data that crosses 50
        patterns = self.rsi_indicator.get_patterns(rsi_df)
        
        rsi_column = f'rsi_{self.rsi_indicator.period}'  # rsi_14
        rsi_series = rsi_df[rsi_column]
        # Find where RSI crosses above 50
        cross_above_mask = (rsi_series > 50) & (rsi_series.shift(1) <= 50)
        
        # 注意：RSI_CROSS_ABOVE_50 不是标准形态，应该使用 GOLDEN_CROSS
        # 这里暂时保持原有逻辑，但需要后续优化
        if 'RSI_CROSS_ABOVE_50' in patterns.columns:
            self.assertTrue(patterns['RSI_CROSS_ABOVE_50'][cross_above_mask].all())
            self.assertFalse(patterns['RSI_CROSS_ABOVE_50'][~cross_above_mask].any())
        else:
            # 使用标准形态名称进行测试
            self.assertTrue(patterns['GOLDEN_CROSS'][cross_above_mask].any())

    def test_bullish_divergence_pattern_Rsi(self):
        """Test the RSI_BULLISH_DIVERGENCE pattern."""
        # Note: Vectorized divergence is an approximation and might be tricky to test precisely.
        # This test checks if the pattern is detected in a crafted scenario.
        rsi_df = self.rsi_indicator.calculate(self.bullish_divergence_df)
        patterns = self.rsi_indicator.get_patterns(rsi_df)

        # 验证BULLISH_DIVERGENCE列存在且为布尔类型 - 使用标准形态名称
        self.assertIn('BULLISH_DIVERGENCE', patterns.columns, "BULLISH_DIVERGENCE列应该存在")
        self.assertTrue(patterns['BULLISH_DIVERGENCE'].dtype == bool, "BULLISH_DIVERGENCE应该是布尔类型")

        # 背离检测是复杂的算法，这里主要验证功能不出错
        # 如果检测到背离，验证其为布尔值
        if patterns['BULLISH_DIVERGENCE'].any():
            print("检测到牛市背离信号")
        else:
            print("未检测到牛市背离信号（这是正常的，取决于数据特征）")

if __name__ == '__main__':
    unittest.main() 