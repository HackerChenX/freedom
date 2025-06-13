import pandas as pd
from indicators.base_indicator import BaseIndicator
from typing import Dict

class ZXMDailyMACD(BaseIndicator):
    """
    主力资金日MACD (ZXM Daily MACD)
    
    分析主力资金的日线MACD，判断资金流向。
    """
    def __init__(self, short_period=12, long_period=26, mid_period=9):
        super().__init__(name="ZXMDailyMACD", description="主力资金日MACD")
        self.short_period = short_period
        self.long_period = long_period
        self.mid_period = mid_period

    def set_parameters(self, short_period=12, long_period=26, mid_period=9):
        self.short_period = short_period
        self.long_period = long_period
        self.mid_period = mid_period

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        return 0.5

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> list:
        return []

    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算主力资金日MACD
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column.") 