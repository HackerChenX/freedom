from indicators.base_indicator import BaseIndicator
import pandas as pd

class VOL(BaseIndicator):
    def __init__(self, period: int = 5):
        super().__init__()
        self.period = period

    def set_parameters(self, **kwargs):
        """设置指标参数，可设置 'period'"""
        if 'period' in kwargs:
            self.period = int(kwargs['period'])

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> list:
        """
        获取VOL指标的技术形态
        """
        return []

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        data['VOL'] = data['volume']
        return data 