from indicators.base_indicator import BaseIndicator
import pandas as pd

class VOL(BaseIndicator):
    def __init__(self, period: int = 5):
        super().__init__()
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        data['VOL'] = data['volume']
        return data 