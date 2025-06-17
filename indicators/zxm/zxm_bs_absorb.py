from indicators.base_indicator import BaseIndicator
import pandas as pd
from typing import Dict

class ZXMBSAbsorb(BaseIndicator):
    """
    主力吸筹指标 (ZXM Buy/Sell Absorb)
    
    通过分析成交量和价格的变化，判断主力资金是否在吸筹或派发。
    """
    def __init__(self, short_period=12, long_period=26, mid_period=9):
        super().__init__(name="ZXMBSAbsorb", description="主力吸筹指标")
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
        获取ZXMBSAbsorb相关形态

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
        if 'absorb_signal' in self._result.columns:
            patterns['ZXM_BS_ABSORB_SIGNAL'] = self._result['absorb_signal']
        else:
            patterns['ZXM_BS_ABSORB_SIGNAL'] = False

        if 'buy_signal' in self._result.columns:
            patterns['ZXM_BS_BUY_SIGNAL'] = self._result['buy_signal']
        else:
            patterns['ZXM_BS_BUY_SIGNAL'] = False

        if 'sell_signal' in self._result.columns:
            patterns['ZXM_BS_SELL_SIGNAL'] = self._result['sell_signal']
        else:
            patterns['ZXM_BS_SELL_SIGNAL'] = False

        return patterns

    def register_patterns(self):
        """
        注册ZXMBSAbsorb指标的形态到全局形态注册表
        """
        # 注册主力吸筹信号
        self.register_pattern_to_registry(
            pattern_id="ZXM_BS_ABSORB_SIGNAL",
            display_name="ZXM主力吸筹信号",
            description="基于买卖力量分析的主力吸筹信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        # 注册买入信号
        self.register_pattern_to_registry(
            pattern_id="ZXM_BS_BUY_SIGNAL",
            display_name="ZXM主力买入信号",
            description="主力资金买入信号，表明资金流入",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        # 注册卖出信号
        self.register_pattern_to_registry(
            pattern_id="ZXM_BS_SELL_SIGNAL",
            display_name="ZXM主力卖出信号",
            description="主力资金卖出信号，表明资金流出",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算主力吸筹指标
        """
        if 'close' not in data.columns or 'volume' not in data.columns:
            raise ValueError("Data must contain 'close' and 'volume' columns.")

        # 初始化结果DataFrame
        result = data.copy()

        # 计算简单的主力吸筹信号
        # 基于价格和成交量的关系
        price_change = data['close'].pct_change()
        volume_ma = data['volume'].rolling(window=self.mid_period).mean()
        volume_ratio = data['volume'] / volume_ma

        # 吸筹信号：价格小幅下跌或横盘，成交量放大
        absorb_signal = (price_change.abs() < 0.02) & (volume_ratio > 1.2)

        # 买入信号：价格上涨，成交量放大
        buy_signal = (price_change > 0.01) & (volume_ratio > 1.5)

        # 卖出信号：价格下跌，成交量放大
        sell_signal = (price_change < -0.01) & (volume_ratio > 1.5)

        result['absorb_signal'] = absorb_signal
        result['buy_signal'] = buy_signal
        result['sell_signal'] = sell_signal
        result['volume_ratio'] = volume_ratio

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
