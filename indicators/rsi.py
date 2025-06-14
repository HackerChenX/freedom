#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
相对强弱指数(RSI)

通过比较一段时期内平均收盘涨数和平均收盘跌数来分析市场买卖盘的意向和实力
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class RSI(BaseIndicator):
    """
    相对强弱指数(RSI)
    """

    def __init__(self, period: int = 14, ma_periods: List[int] = None, overbought: float = 70.0, oversold: float = 30.0):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        super().__init__()
        self.name = "RSI"
        self.period = period
        self.ma_periods = ma_periods if ma_periods is not None else [5, 10]
        self.overbought = overbought
        self.oversold = oversold

    def set_parameters(self, period: int = 14, overbought: float = 70.0, oversold: float = 30.0, **kwargs):
        """
        设置RSI指标的参数
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        if 'ma_periods' in kwargs:
            self.ma_periods = kwargs['ma_periods']

    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算RSI指标，并包含均线和信号
        
        Args:
            df: 包含价格数据的DataFrame
            
        Returns:
            pd.DataFrame: 添加了RSI指标的DataFrame
        """
        if data.empty:
            return data
            
        # 确保数据包含所需的列
        if 'close' not in data.columns:
            raise ValueError("输入数据必须包含'close'列")
            
        result_df = pd.DataFrame(index=data.index)
        
        # 计算价格变动
        delta = data['close'].diff()
        
        # 计算上涨和下跌
        gain = delta.where(delta > 0, 0).ewm(span=self.period, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(span=self.period, adjust=False).mean()

        # 计算相对强度
        rs = gain / loss.replace(0, 1e-9)
        
        # 计算RSI
        result_df[f'rsi_{self.period}'] = 100 - (100 / (1 + rs))
        
        # 可选：计算RSI均线
        if self.ma_periods and len(self.ma_periods) >= 2:
            result_df[f'rsi_ma_{self.ma_periods[0]}'] = result_df[f'rsi_{self.period}'].rolling(window=self.ma_periods[0]).mean()
            result_df[f'rsi_ma_{self.ma_periods[1]}'] = result_df[f'rsi_{self.period}'].rolling(window=self.ma_periods[1]).mean()
            # For pattern detection
            result_df['rsi_ma_short'] = result_df[f'rsi_ma_{self.ma_periods[0]}']
            result_df['rsi_ma_long'] = result_df[f'rsi_ma_{self.ma_periods[1]}']

        result_df['rsi_overbought'] = result_df[f'rsi_{self.period}'] > self.overbought
        result_df['rsi_oversold'] = result_df[f'rsi_{self.period}'] < self.oversold

        return result_df

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取RSI相关形态
        """
        calculated_data = self._calculate(data)
        patterns_df = pd.DataFrame(index=data.index)

        # 确保列存在
        if f'rsi_{self.period}' not in calculated_data.columns or 'rsi_ma_short' not in calculated_data.columns or 'rsi_ma_long' not in calculated_data.columns:
            return patterns_df

        # 金叉和死叉
        from utils.indicator_utils import crossover, crossunder
        patterns_df['RSI_GOLDEN_CROSS'] = crossover(calculated_data['rsi_ma_short'], calculated_data['rsi_ma_long'])
        patterns_df['RSI_DEATH_CROSS'] = crossunder(calculated_data['rsi_ma_short'], calculated_data['rsi_ma_long'])

        # 超买和超卖
        patterns_df['RSI_OVERBOUGHT'] = calculated_data[f'rsi_{self.period}'] > self.overbought
        patterns_df['RSI_OVERSOLD'] = calculated_data[f'rsi_{self.period}'] < self.oversold
        
        return patterns_df

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成RSI交易信号
        """
        calculated_data = self._calculate(data)

        patterns = self.get_patterns(data)

        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = patterns['RSI_GOLDEN_CROSS'] | (patterns['RSI_OVERSOLD'])
        signals['sell_signal'] = patterns['RSI_DEATH_CROSS'] | (patterns['RSI_OVERBOUGHT'])

        return signals

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算RSI指标的原始评分 (0-100分)
        """
        calculated_data = self._calculate(data)
        
        if calculated_data is None or f'rsi_{self.period}' not in calculated_data.columns:
            return pd.Series(50.0, index=data.index)
            
        score = pd.Series(50.0, index=data.index)
        rsi_values = calculated_data[f'rsi_{self.period}']
        
        # 基于RSI值的评分
        score += (rsi_values - 50) * 0.4 # 20-80 映射到 42-58
        
        # 超买超卖区域评分
        score[rsi_values > self.overbought] -= 15
        score[rsi_values < self.oversold] += 15
        
        # 均线交叉评分
        if 'rsi_ma_short' in calculated_data.columns and 'rsi_ma_long' in calculated_data.columns:
            short_ma = calculated_data['rsi_ma_short']
            long_ma = calculated_data['rsi_ma_long']
            
            from utils.indicator_utils import crossover, crossunder
            score[crossover(short_ma, long_ma)] += 20
            score[crossunder(short_ma, long_ma)] -= 20
            
        return score.clip(0, 100)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: Dict[str, pd.Series]) -> float:
        """
        计算RSI指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 1. 基于得分的置信度
        last_score = score.iloc[-1]
        score_confidence = 0.5

        # 超买超卖区域置信度较高
        if last_score > 70 or last_score < 30:
            score_confidence = 0.8
        # 中性区域置信度中等
        elif 40 <= last_score <= 60:
            score_confidence = 0.6
        else:
            score_confidence = 0.7

        # 2. 基于形态的置信度
        pattern_confidence = 0.5
        if not patterns.empty:
            # 统计最近几个周期的形态数量
            recent_patterns = patterns.iloc[-5:].sum().sum() if len(patterns) >= 5 else patterns.sum().sum()

            if recent_patterns > 0:
                pattern_confidence = min(0.5 + recent_patterns * 0.1, 0.9)

        # 3. 基于信号的置信度
        signal_confidence = 0.5
        if signals:
            # 检查是否有强烈的买卖信号
            for signal_name, signal_series in signals.items():
                if isinstance(signal_series, pd.Series) and signal_series.iloc[-1]:
                    signal_confidence = 0.8
                    break

        # 综合置信度
        confidence = (score_confidence * 0.4 + pattern_confidence * 0.3 + signal_confidence * 0.3)

        return min(confidence, 1.0)

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        计算最终评分

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 包含评分和置信度的字典
        """
        try:
            # 1. 计算原始评分序列
            raw_scores = self.calculate_raw_score(data, **kwargs)

            # 如果数据不足，返回中性评分
            if len(raw_scores) < 3:
                return {'score': 50.0, 'confidence': 0.5}

            # 取最近的评分作为最终评分，但考虑近期趋势
            recent_scores = raw_scores.iloc[-3:]
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 最终评分 = 最新评分 + 趋势调整
            final_score = recent_scores.iloc[-1] + trend / 2

            # 确保评分在0-100范围内
            final_score = max(0, min(100, final_score))

            # 2. 获取形态和信号
            patterns = self.get_patterns(data, **kwargs)
            signals = self.generate_signals(data, **kwargs)

            # 3. 计算置信度
            confidence = self.calculate_confidence(raw_scores, patterns, signals.to_dict('series') if hasattr(signals, 'to_dict') else {})

            return {
                'score': final_score,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"为指标 {self.name} 计算评分时出错: {e}")
            return {'score': 50.0, 'confidence': 0.0}

    def register_patterns(self):
        """
        注册RSI指标的形态到全局形态注册表
        """
        # 注册RSI超买形态
        self.register_pattern_to_registry(
            pattern_id="RSI_OVERBOUGHT",
            display_name="RSI超买",
            description="RSI指标进入超买区域，市场可能过热",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0
        )

        # 注册RSI超卖形态
        self.register_pattern_to_registry(
            pattern_id="RSI_OVERSOLD",
            display_name="RSI超卖",
            description="RSI指标进入超卖区域，市场可能过冷",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0
        )

        # 注册RSI金叉形态
        self.register_pattern_to_registry(
            pattern_id="RSI_GOLDEN_CROSS",
            display_name="RSI金叉",
            description="RSI短期均线上穿长期均线，可能是买入信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0
        )

        # 注册RSI死叉形态
        self.register_pattern_to_registry(
            pattern_id="RSI_DEATH_CROSS",
            display_name="RSI死叉",
            description="RSI短期均线下穿长期均线，可能是卖出信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0
        )

        # 注册RSI极度超买形态
        self.register_pattern_to_registry(
            pattern_id="RSI_EXTREME_OVERBOUGHT",
            display_name="RSI极度超买",
            description="RSI指标进入极度超买区域(>80)，市场极度过热",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-25.0
        )

        # 注册RSI极度超卖形态
        self.register_pattern_to_registry(
            pattern_id="RSI_EXTREME_OVERSOLD",
            display_name="RSI极度超卖",
            description="RSI指标进入极度超卖区域(<20)，市场极度过冷",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0
        )

        # 注册RSI中性形态
        self.register_pattern_to_registry(
            pattern_id="RSI_NEUTRAL",
            display_name="RSI中性",
            description="RSI指标在中性区域(40-60)，市场相对平衡",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0
        )