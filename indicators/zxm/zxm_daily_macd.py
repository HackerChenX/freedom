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

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            'bullish': {'name': '看涨形态', 'description': '指标显示看涨信号', 'type': 'BULLISH'},
            'bearish': {'name': '看跌形态', 'description': '指标显示看跌信号', 'type': 'BEARISH'},
            'neutral': {'name': '中性形态', 'description': '指标显示中性信号', 'type': 'NEUTRAL'},
            
            # 通用形态
            'strong_signal': {'name': '强信号', 'description': '强烈的技术信号', 'type': 'STRONG'},
            'weak_signal': {'name': '弱信号', 'description': '较弱的技术信号', 'type': 'WEAK'},
            'trend_up': {'name': '上升趋势', 'description': '价格呈上升趋势', 'type': 'BULLISH'},
            'trend_down': {'name': '下降趋势', 'description': '价格呈下降趋势', 'type': 'BEARISH'},
        }
        
        # 默认形态信息
        default_pattern = {
            'name': pattern_id.replace('_', ' ').title(),
            'description': f'{pattern_id}形态',
            'type': 'UNKNOWN'
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)
