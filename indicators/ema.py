#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
指数移动平均线(EMA)

对近期价格赋予更高权重
"""

import pandas as pd
import numpy as np
from typing import List

from indicators.base_indicator import BaseIndicator
from utils.indicator_utils import crossover, crossunder


class EMA(BaseIndicator):
    """
    指数移动平均线(EMA)
    
    分类：趋势类指标
    描述：对近期价格赋予更高权重
    """
    
    REQUIRED_COLUMNS = ['close', 'high', 'low']

    def __init__(self, periods: List[int] = None, ma_type: str = 'EMA'):
        """
        初始化指数移动平均线(EMA)指标
        Args:
            periods: 计算周期列表，默认为[5, 10, 20, 60]
            ma_type: 均线类型，默认为'EMA'
        """
        super().__init__(name="EMA", description="指数移动平均线")
        self.periods = periods if periods is not None else [5, 10, 20, 60]
        self.ma_type = ma_type
        self.ma_cols = [f'{self.ma_type}{p}' for p in self.periods]
        self.register_patterns()
        
    def set_parameters(self, periods: List[int] = None, ma_type: str = None):
        """设置指标参数"""
        if periods is not None:
            self.periods = periods
        if ma_type is not None:
            self.ma_type = ma_type
        self.ma_cols = [f'{self.ma_type}{p}' for p in self.periods]
        self.register_patterns()
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算指数移动平均线(EMA)指标
        """
        for p in self.periods:
            df[f'{self.ma_type}{p}'] = df['close'].ewm(span=p, adjust=True).mean()
        return df

    def calculate_raw_score(self, df: pd.DataFrame) -> pd.Series:
        """
        计算EMA原始评分。
        评分标准:
        1.  多头/空头排列: +40 (多头) / -40 (空头)
        2.  短期趋势: 向上+15, 向下-15
        3.  价格与短周期均线关系: 价格在均线上方+10, 下方-10
        4.  金叉/死叉: 最近2天内发生金叉+20, 死叉-20
        """
        if not self.ma_cols or not all(c in df.columns for c in self.ma_cols):
            return pd.Series(50, index=df.index)

        score = pd.Series(50.0, index=df.index)
        
        sorted_mas = [df[f'{self.ma_type}{p}'] for p in sorted(self.periods)]
        
        if len(sorted_mas) > 1:
            is_bullish_arrangement = (sorted_mas[0] > sorted_mas[-1])
            is_bearish_arrangement = (sorted_mas[0] < sorted_mas[-1])
            score[is_bullish_arrangement] += 25
            score[is_bearish_arrangement] -= 25

        short_ma = sorted_mas[0]
        trend = np.sign(short_ma.diff(2)).fillna(0)
        score[trend == 1] += 15
        score[trend == -1] -= 15

        close_price = df['close']
        score[close_price > short_ma] += 10
        score[close_price < short_ma] -= 10
        
        if len(sorted_mas) >= 2:
            short_ema = sorted_mas[0]
            medium_ema = sorted_mas[1]
            golden_cross = crossover(short_ema, medium_ema)
            death_cross = crossunder(short_ema, medium_ema)
            score[golden_cross.rolling(window=2, min_periods=1).max().fillna(0).astype(bool)] += 20
            score[death_cross.rolling(window=2, min_periods=1).max().fillna(0).astype(bool)] -= 20

        return score.clip(0, 100)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算置信度。
        """
        return 0.5

    def get_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        识别EMA技术形态
        """
        patterns = {}
        if len(self.periods) < 2 or not all(c in df.columns for c in self.ma_cols):
            return pd.DataFrame(patterns)

        p_short, p_long = sorted(self.periods)[:2]
        short_ema = df[f'{self.ma_type}{p_short}']
        long_ema = df[f'{self.ma_type}{p_long}']

        golden_cross_key = f"EMA_{p_short}_{p_long}_GOLDEN_CROSS"
        patterns[golden_cross_key] = crossover(short_ema, long_ema)

        death_cross_key = f"EMA_{p_short}_{p_long}_DEATH_CROSS"
        patterns[death_cross_key] = crossunder(short_ema, long_ema)

        bullish_arrangement_key = "EMA_BULLISH_ARRANGEMENT"
        patterns[bullish_arrangement_key] = short_ema > long_ema

        bearish_arrangement_key = "EMA_BEARISH_ARRANGEMENT"
        patterns[bearish_arrangement_key] = short_ema < long_ema
        
        return pd.DataFrame(patterns)

    def register_patterns(self):
        """
        注册与该指标相关的技术形态。
        """
        if len(self.periods) < 2:
            return
            
        p_short, p_long = sorted(self.periods)[:2]
        
        self.register_pattern_to_registry(
            pattern_id=f"EMA_{p_short}_{p_long}_GOLDEN_CROSS",
            display_name=f"EMA({p_short},{p_long})金叉",
            description=f"当短期EMA({p_short})上穿长期EMA({p_long})时，被视为看涨信号。",
            pattern_type="BULLISH"
        )
        self.register_pattern_to_registry(
            pattern_id=f"EMA_{p_short}_{p_long}_DEATH_CROSS",
            display_name=f"EMA({p_short},{p_long})死叉",
            description=f"当短期EMA({p_short})下穿长期EMA({p_long})时，被视为看跌信号。",
            pattern_type="BEARISH"
        )
        self.register_pattern_to_registry(
            pattern_id="EMA_BULLISH_ARRANGEMENT",
            display_name="EMA多头排列",
            description=f"短期EMA在长期EMA之上，表明市场处于上升趋势。",
            pattern_type="BULLISH"
        )
        self.register_pattern_to_registry(
            pattern_id="EMA_BEARISH_ARRANGEMENT",
            display_name="EMA空头排列",
            description=f"短期EMA在长期EMA之下，表明市场处于下降趋势。",
            pattern_type="BEARISH"
        )

