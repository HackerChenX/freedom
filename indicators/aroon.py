#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
阿隆指标(Aroon)

阿隆指标用于识别趋势的开始和结束，通过计算最高价和最低价距离当前的时间来判断趋势强度
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any

from indicators.base_indicator import BaseIndicator, PatternResult
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class Aroon(BaseIndicator):
    """
    阿隆指标(Aroon) (Aroon)
    
    分类：趋势类指标
    描述：阿隆指标用于识别趋势的开始和结束，通过计算最高价和最低价距离当前的时间来判断趋势强度
    """
    
    def __init__(self, period: int = 14):
        """
        初始化阿隆指标(Aroon)
        
        Args:
            period: 计算周期，默认为14
        """
        super().__init__(name="Aroon", description="阿隆指标，用于识别趋势的开始和结束")
        self.period = period
        
        # 注册Aroon形态
        self._register_aroon_patterns()
        
    def _register_aroon_patterns(self):
        """
        注册Aroon指标形态
        """
        from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册Aroon交叉形态
        registry.register(
            pattern_id="AROON_BULLISH_CROSS",
            display_name="Aroon多头交叉",
            description="Aroon Up上穿Aroon Down，表明上升趋势可能开始",
            indicator_id="AROON",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=15.0
        )
        
        registry.register(
            pattern_id="AROON_BEARISH_CROSS",
            display_name="Aroon空头交叉",
            description="Aroon Down上穿Aroon Up，表明下降趋势可能开始",
            indicator_id="AROON",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-15.0
        )
        
        # 注册Aroon极值形态
        registry.register(
            pattern_id="AROON_STRONG_UPTREND",
            display_name="Aroon强势上涨",
            description="Aroon Up > 70 且 Aroon Down < 30，表明强势上升趋势",
            indicator_id="AROON",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=20.0
        )
        
        registry.register(
            pattern_id="AROON_STRONG_DOWNTREND",
            display_name="Aroon强势下跌",
            description="Aroon Down > 70 且 Aroon Up < 30，表明强势下降趋势",
            indicator_id="AROON",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-20.0
        )
        
        # 注册Aroon震荡器穿越形态
        registry.register(
            pattern_id="AROON_OSC_CROSS_ABOVE_ZERO",
            display_name="Aroon震荡器上穿零轴",
            description="Aroon震荡器从下方穿越零轴，表明趋势转为向上",
            indicator_id="AROON",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=12.0
        )
        
        registry.register(
            pattern_id="AROON_OSC_CROSS_BELOW_ZERO",
            display_name="Aroon震荡器下穿零轴",
            description="Aroon震荡器从上方穿越零轴，表明趋势转为向下",
            indicator_id="AROON",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-12.0
        )
        
        # 注册Aroon震荡器极值形态
        registry.register(
            pattern_id="AROON_OSC_EXTREME_BULLISH",
            display_name="Aroon震荡器极度看涨",
            description="Aroon震荡器值 > 50，表明极强的上升趋势",
            indicator_id="AROON",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.VERY_STRONG,
            score_impact=25.0
        )
        
        registry.register(
            pattern_id="AROON_OSC_EXTREME_BEARISH",
            display_name="Aroon震荡器极度看跌",
            description="Aroon震荡器值 < -50，表明极强的下降趋势",
            indicator_id="AROON",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.VERY_STRONG,
            score_impact=-25.0
        )
        
        # 注册Aroon盘整形态
        registry.register(
            pattern_id="AROON_CONSOLIDATION",
            display_name="Aroon盘整",
            description="Aroon Up和Aroon Down都在30以下，表明市场处于盘整状态",
            indicator_id="AROON",
            pattern_type=PatternType.CONSOLIDATION,
            default_strength=PatternStrength.MEDIUM,
            score_impact=0.0
        )
        
        # 注册Aroon转折形态
        registry.register(
            pattern_id="AROON_REVERSAL_BULLISH",
            display_name="Aroon看涨反转",
            description="Aroon Down从高位快速下降，同时Aroon Up从低位快速上升",
            indicator_id="AROON",
            pattern_type=PatternType.REVERSAL,
            default_strength=PatternStrength.STRONG,
            score_impact=18.0
        )
        
        registry.register(
            pattern_id="AROON_REVERSAL_BEARISH",
            display_name="Aroon看跌反转",
            description="Aroon Up从高位快速下降，同时Aroon Down从低位快速上升",
            indicator_id="AROON",
            pattern_type=PatternType.REVERSAL,
            default_strength=PatternStrength.STRONG,
            score_impact=-18.0
        )
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 输入数据
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算Aroon指标
        
        Args:
            df: 包含OHLC数据的DataFrame
                
        Returns:
            包含Aroon指标的DataFrame
        """
        return self.calculate(df)
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算阿隆指标(Aroon)
        
        Args:
            df: 包含OHLC数据的DataFrame
                必须包含以下列：
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了Aroon指标列的DataFrame
        """
        if df.empty:
            return df
            
        required_columns = ['high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()

        # 使用 rolling window 和 idxmax/idxmin 来找到最高价/最低价的位置
        rolling_high = df_copy['high'].rolling(window=self.period, min_periods=self.period)
        rolling_low = df_copy['low'].rolling(window=self.period, min_periods=self.period)
        
        # argmax() 返回窗口内最大值的相对位置（0-indexed）
        # 如果最大值在窗口末尾（即当天），argmax返回 `period-1`，days_since_high 为 0
        days_since_high = (self.period - 1) - rolling_high.apply(np.argmax, raw=True)
        days_since_low = (self.period - 1) - rolling_low.apply(np.argmin, raw=True)

        # 计算Aroon值
        df_copy['aroon_up'] = ((self.period - days_since_high) / self.period) * 100
        df_copy['aroon_down'] = ((self.period - days_since_low) / self.period) * 100
        
        # 计算Aroon震荡器
        df_copy['aroon_oscillator'] = df_copy['aroon_up'] - df_copy['aroon_down']
        
        return df_copy

    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取Aroon指标的交易信号
        
        Args:
            df: 包含Aroon指标的DataFrame
            **kwargs: 其他参数
        
        Returns:
            pd.DataFrame: 包含交易信号的DataFrame
        """
        if 'aroon_up' not in df.columns or 'aroon_down' not in df.columns:
            df = self.calculate(df)
            
        signal_df = pd.DataFrame(index=df.index)
        
        # 买入信号：Aroon Up 上穿 Aroon Down
        signal_df['buy_signal'] = crossover(df['aroon_up'], df['aroon_down'])
        
        # 卖出信号：Aroon Down 上穿 Aroon Up
        signal_df['sell_signal'] = crossover(df['aroon_down'], df['aroon_up'])
        
        # 强趋势信号
        signal_df['strong_uptrend'] = (df['aroon_up'] > 70) & (df['aroon_down'] < 30)
        signal_df['strong_downtrend'] = (df['aroon_down'] > 70) & (df['aroon_up'] < 30)
        
        return signal_df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算Aroon指标的原始评分
        
        Args:
            data: 包含Aroon指标的DataFrame
            **kwargs: 其他参数
        
        Returns:
            pd.Series: 原始评分
        """
        if 'aroon_oscillator' not in data.columns:
            data = self.calculate(data)
            
        # 基础分数为50
        score = pd.Series(50, index=data.index)
        
        # 根据Aroon Oscillator的值调整分数
        # Oscillator > 0: 上升趋势，分数 > 50
        # Oscillator < 0: 下降趋势，分数 < 50
        score += data['aroon_oscillator'] * 0.5
        
        # 根据趋势强度调整分数
        strong_uptrend = (data['aroon_up'] > 70) & (data['aroon_down'] < 30)
        strong_downtrend = (data['aroon_down'] > 70) & (data['aroon_up'] < 30)
        
        score[strong_uptrend] = np.minimum(score[strong_uptrend] + 15, 100)
        score[strong_downtrend] = np.maximum(score[strong_downtrend] - 15, 0)
        
        # 返回总分
        return score.clip(0, 100)

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        识别所有已注册的Aroon形态

        Args:
            data (pd.DataFrame): 包含Aroon指标计算结果的DataFrame。

        Returns:
            pd.DataFrame: 一个DataFrame，其列为各种形态，值为布尔值表示形态是否出现。
        """
        # 确保aroon值已计算
        if 'aroon_up' not in data.columns or 'aroon_down' not in data.columns:
            data = self.calculate(data)

        aroon_up = data['aroon_up']
        aroon_down = data['aroon_down']
        aroon_osc = data['aroon_oscillator']

        patterns = pd.DataFrame(index=data.index)

        # 交叉形态
        patterns['AROON_BULLISH_CROSS'] = crossover(aroon_up, aroon_down)
        patterns['AROON_BEARISH_CROSS'] = crossunder(aroon_up, aroon_down)

        # 极值形态
        patterns['AROON_STRONG_UPTREND'] = (aroon_up > 70) & (aroon_down < 30)
        patterns['AROON_STRONG_DOWNTREND'] = (aroon_down > 70) & (aroon_up < 30)

        # 震荡器穿越
        patterns['AROON_OSC_CROSS_ABOVE_ZERO'] = crossover(aroon_osc, 0)
        patterns['AROON_OSC_CROSS_BELOW_ZERO'] = crossunder(aroon_osc, 0)

        # 震荡器极值
        patterns['AROON_OSC_EXTREME_BULLISH'] = aroon_osc > 50
        patterns['AROON_OSC_EXTREME_BEARISH'] = aroon_osc < -50
        
        # 盘整
        patterns['AROON_CONSOLIDATION'] = (aroon_up < 30) & (aroon_down < 30)

        # 反转形态 (简化版)
        down_declining_fast = aroon_down < aroon_down.shift(1)
        up_rising_fast = aroon_up > aroon_up.shift(1)
        patterns['AROON_REVERSAL_BULLISH'] = (aroon_down.shift(1) > 70) & down_declining_fast & (aroon_up.shift(1) < 30) & up_rising_fast

        up_declining_fast = aroon_up < aroon_up.shift(1)
        down_rising_fast = aroon_down > aroon_down.shift(1)
        patterns['AROON_REVERSAL_BEARISH'] = (aroon_up.shift(1) > 70) & up_declining_fast & (aroon_down.shift(1) < 30) & down_rising_fast

        return patterns

    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制阿隆指标(Aroon)
        
        Args:
            df: 包含Aroon指标的DataFrame
            ax: Matplotlib的Axes对象
            **kwargs: 其他绘图参数
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            price_ax = ax[0]
            indicator_ax = ax[1]
        else:
            price_ax = ax
            indicator_ax = price_ax.twinx()
            
        # 绘制价格
        price_ax.plot(df.index, df['close'], label='Close Price')
        price_ax.set_ylabel('Price')
        price_ax.set_title(f'Aroon ({self.period})')
        price_ax.legend(loc='upper left')
        
        # 绘制Aroon指标
        indicator_ax.plot(df.index, df['aroon_up'], label='Aroon Up', color='green')
        indicator_ax.plot(df.index, df['aroon_down'], label='Aroon Down', color='red')
        indicator_ax.set_ylabel('Aroon')
        indicator_ax.legend(loc='upper right')
        
        # 添加水平线
        indicator_ax.axhline(70, linestyle='--', color='gray', alpha=0.7)
        indicator_ax.axhline(30, linestyle='--', color='gray', alpha=0.7)
        
        plt.show()

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 包含Aroon指标的DataFrame
            
        Returns:
            pd.DataFrame: 包含交易信号的DataFrame
        """
        return self.get_signals(data)

    def calculate_score(self, data):
        return {
            'final_score': pd.Series([50] * len(data), index=data.index),
            'patterns': [],
            'market_environment': MagicMock(),
            'confidence': 50
        } 