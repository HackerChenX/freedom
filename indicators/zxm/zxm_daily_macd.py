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

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取ZXMDailyMACD相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate(data, **kwargs)

        patterns = pd.DataFrame(index=data.index)

        # 如果没有计算结果，返回空DataFrame
        if self._result is None or self._result.empty:
            return patterns

        # 基于计算结果创建形态
        if 'macd_golden_cross' in self._result.columns:
            patterns['ZXM_DAILY_MACD_GOLDEN_CROSS'] = self._result['macd_golden_cross']
        else:
            patterns['ZXM_DAILY_MACD_GOLDEN_CROSS'] = False

        if 'macd_death_cross' in self._result.columns:
            patterns['ZXM_DAILY_MACD_DEATH_CROSS'] = self._result['macd_death_cross']
        else:
            patterns['ZXM_DAILY_MACD_DEATH_CROSS'] = False

        if 'macd_positive' in self._result.columns:
            patterns['ZXM_DAILY_MACD_POSITIVE'] = self._result['macd_positive']
        else:
            patterns['ZXM_DAILY_MACD_POSITIVE'] = False

        return patterns

    def register_patterns(self):
        """
        注册ZXMDailyMACD指标的形态到全局形态注册表
        """
        # 注册MACD金叉形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_GOLDEN_CROSS",
            display_name="ZXM日线MACD金叉",
            description="日线MACD金叉，表明多头力量增强",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        # 注册MACD死叉形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_DEATH_CROSS",
            display_name="ZXM日线MACD死叉",
            description="日线MACD死叉，表明空头力量增强",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册MACD为正形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_POSITIVE",
            display_name="ZXM日线MACD为正",
            description="日线MACD值为正，表明多头趋势",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算主力资金日MACD
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column.")

        # 初始化结果DataFrame
        result = data.copy()

        # 计算MACD指标
        close = data['close']

        # 计算EMA
        ema_short = close.ewm(span=self.short_period).mean()
        ema_long = close.ewm(span=self.long_period).mean()

        # 计算DIF和DEA
        dif = ema_short - ema_long
        dea = dif.ewm(span=self.mid_period).mean()
        macd = (dif - dea) * 2

        # 计算形态信号
        macd_golden_cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        macd_death_cross = (dif < dea) & (dif.shift(1) >= dea.shift(1))
        macd_positive = dif > 0

        result['dif'] = dif
        result['dea'] = dea
        result['macd'] = macd
        result['macd_golden_cross'] = macd_golden_cross
        result['macd_death_cross'] = macd_death_cross
        result['macd_positive'] = macd_positive

        self._result = result
        return result

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
