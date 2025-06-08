#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
相对强弱指数(RSI)

通过分析价格波动的幅度和速度，判断市场的超买超卖情况
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union, Tuple

from indicators.base_indicator import BaseIndicator, PatternResult
from utils.logger import get_logger

logger = get_logger(__name__)


class RSI(BaseIndicator):
    """
    相对强弱指数(RSI)
    
    分类：震荡类指标
    计算方法：RSI = 100 * RS / (1 + RS)，其中RS是平均上涨幅度与平均下跌幅度的比值
    参数：N，一般取14，表示计算周期
    """
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化RSI指标
        
        Args:
            period: 计算周期，默认为14
            overbought: 超买线，默认为70
            oversold: 超卖线，默认为30
        """
        self.name = "RSI"
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
        # 初始化基类（会自动调用register_patterns方法）
        super().__init__()
    
    def _register_rsi_patterns(self):
        """
        注册RSI指标形态
        """
        from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册RSI超买/超卖形态
        registry.register(
            pattern_id="RSI_OVERBOUGHT",
            display_name="RSI超买",
            description=f"RSI值高于{self.overbought}，表明市场可能超买，存在回调风险",
            indicator_id="RSI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-15.0
        )
        
        registry.register(
            pattern_id="RSI_OVERSOLD",
            display_name="RSI超卖",
            description=f"RSI值低于{self.oversold}，表明市场可能超卖，存在反弹机会",
            indicator_id="RSI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=15.0
        )
        
        # 注册RSI背离形态
        registry.register(
            pattern_id="RSI_BULLISH_DIVERGENCE",
            display_name="RSI底背离",
            description="价格创新低，但RSI未创新低，表明下跌动能减弱，可能即将反转向上",
            indicator_id="RSI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=25.0
        )
        
        registry.register(
            pattern_id="RSI_BEARISH_DIVERGENCE",
            display_name="RSI顶背离",
            description="价格创新高，但RSI未创新高，表明上涨动能减弱，可能即将反转向下",
            indicator_id="RSI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-25.0
        )
        
        # 注册RSI突破形态
        registry.register(
            pattern_id="RSI_BREAKOUT_UP",
            display_name="RSI向上突破",
            description=f"RSI突破{self.overbought}水平线，表明强势上涨可能开始",
            indicator_id="RSI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="RSI_BREAKOUT_DOWN",
            display_name="RSI向下突破",
            description=f"RSI跌破{self.oversold}水平线，表明强势下跌可能开始",
            indicator_id="RSI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-10.0
        )
        
        # 注册RSI中轴穿越形态
        registry.register(
            pattern_id="RSI_CROSS_ABOVE_50",
            display_name="RSI上穿50",
            description="RSI从下方穿越50中轴线，表明多头力量增强",
            indicator_id="RSI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=12.0
        )
        
        registry.register(
            pattern_id="RSI_CROSS_BELOW_50",
            display_name="RSI下穿50",
            description="RSI从上方穿越50中轴线，表明空头力量增强",
            indicator_id="RSI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-12.0
        )
        
        # 注册RSI钝化形态
        registry.register(
            pattern_id="RSI_BULLISH_FAILURE_SWING",
            display_name="RSI看涨钝化",
            description="RSI在超卖区触底回升后回调，未再次跌入超卖区便重新上升，形成W底",
            indicator_id="RSI",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=20.0
        )
        
        registry.register(
            pattern_id="RSI_BEARISH_FAILURE_SWING",
            display_name="RSI看跌钝化",
            description="RSI在超买区触顶回落后反弹，未再次进入超买区便重新下跌，形成M顶",
            indicator_id="RSI",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-20.0
        )
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算RSI指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含RSI指标的DataFrame
        """
        if df.empty:
            return df
            
        df_copy = df.copy()
        
        # 计算价格变化
        delta = df_copy['close'].diff().fillna(0)
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均上涨和下跌
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # 处理初始值
        avg_gain.fillna(gain.iloc[:self.period].mean(), inplace=True)
        avg_loss.fillna(loss.iloc[:self.period].mean(), inplace=True)
        
        # 计算RS和RSI
        rs = avg_gain / avg_loss.replace(0, 1e-9)  # 避免除以零
        rsi = 100 - (100 / (1 + rs))
        
        # 添加结果到DataFrame
        df_copy['rsi'] = rsi
        
        # 添加超买超卖状态
        df_copy['rsi_overbought'] = rsi > self.overbought
        df_copy['rsi_oversold'] = rsi < self.oversold
        
        # 存储结果
        self._result = df_copy
        
        return df_copy
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取RSI相关形态。

        此方法遵循新的指标开发规范，通过向量化操作识别所有形态，
        并返回一个标准的布尔型DataFrame。

        Args:
            data: 输入数据，可以是原始K线数据或已计算RSI的DataFrame。
            **kwargs: 其他参数。

        Returns:
            pd.DataFrame: 索引为日期，列为形态ID，值为布尔值的形态矩阵。
        """
        # 确保已计算指标
        if not self.has_result() or 'rsi' not in data.columns:
            indicator_df = self.calculate(data, **kwargs)
        else:
            indicator_df = data

        if indicator_df is None or 'rsi' not in indicator_df.columns:
            return pd.DataFrame()

        patterns_df = pd.DataFrame(index=indicator_df.index)
        rsi = indicator_df['rsi']
        close = indicator_df['close']

        # 1. 超买/超卖
        patterns_df['RSI_OVERBOUGHT'] = rsi > self.overbought
        patterns_df['RSI_OVERSOLD'] = rsi < self.oversold

        # 2. 中轴穿越
        patterns_df['RSI_CROSS_ABOVE_50'] = self.crossover(rsi, 50)
        patterns_df['RSI_CROSS_BELOW_50'] = self.crossunder(rsi, 50)

        # 3. 向上/向下突破
        patterns_df['RSI_BREAKOUT_UP'] = self.crossover(rsi, self.overbought)
        patterns_df['RSI_BREAKOUT_DOWN'] = self.crossunder(rsi, self.oversold)
        
        # 4. 背离形态
        # 注意：向量化背离检测相对复杂，这里提供一个简化的实现。
        # 更精确的实现可能需要更复杂的逻辑，例如使用 argrelextrema。
        bullish_div, bearish_div = self._detect_divergence(close, rsi, window=self.period)
        patterns_df['RSI_BULLISH_DIVERGENCE'] = bullish_div
        patterns_df['RSI_BEARISH_DIVERGENCE'] = bearish_div
        
        return patterns_df

    def _detect_divergence(self, price: pd.Series, indicator: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        Using an improved (but still approximate) vectorized method to detect divergence.

        Logic:
        - Bullish Divergence: Price hits a new rolling low, but the indicator's value
          is higher than its rolling low from the previous period, suggesting indicator lows are rising.
        - Bearish Divergence: Price hits a new rolling high, but the indicator's value
          is lower than its rolling high from the previous period, suggesting indicator highs are falling.

        Args:
            price: Price series ('close').
            indicator: Indicator series ('rsi').
            window: The lookback window for detecting divergence.

        Returns:
            A tuple of (bullish_divergence, bearish_divergence) boolean Series.
        """
        # Calculate rolling minimums and maximums
        price_low = price.rolling(window=window).min()
        indicator_low = indicator.rolling(window=window).min()
        price_high = price.rolling(window=window).max()
        indicator_high = indicator.rolling(window=window).max()

        # Bullish divergence: price hits a new low, but the indicator's value is higher than its previous rolling low.
        bullish_div = (price == price_low) & (price < price.shift(1)) & \
                      (indicator > indicator_low.shift(1))

        # Bearish divergence: price hits a new high, but the indicator's value is lower than its previous rolling high.
        bearish_div = (price == price_high) & (price > price.shift(1)) & \
                      (indicator < indicator_high.shift(1))

        return bullish_div.fillna(False), bearish_div.fillna(False)

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 初始化信号
        signals = {}
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
    
        # 如果没有结果，返回空信号
        if self._result is None or 'rsi' not in self._result.columns:
            return signals
        
        # 获取RSI值
        rsi = self._result['rsi']
        
        # 生成超卖买入信号
        for i in range(1, len(rsi)):
            # RSI从超卖区域上穿，生成买入信号
            if rsi.iloc[i-1] < self.oversold and rsi.iloc[i] > self.oversold:
                signals['buy_signal'].iloc[i] = True
                # 信号强度基于RSI值
                signals['signal_strength'].iloc[i] = 50 + min((self.oversold - rsi.iloc[i-1]) * 2, 40)
        
        # 生成超买卖出信号
        for i in range(1, len(rsi)):
            # RSI从超买区域下穿，生成卖出信号
            if rsi.iloc[i-1] > self.overbought and rsi.iloc[i] < self.overbought:
                signals['sell_signal'].iloc[i] = True
                # 信号强度基于RSI值
                signals['signal_strength'].iloc[i] = 50 + min((rsi.iloc[i-1] - self.overbought) * 2, 40)
        
        # 基于RSI背离形态生成信号
        patterns = self.get_patterns(data, **kwargs)
        for pattern in patterns:
            if pattern['name'] == 'RSI底背离' and pattern['type'] == 'bullish':
                end_date = pattern['end_date']
                if end_date in signals['buy_signal'].index:
                    signals['buy_signal'].loc[end_date] = True
                    signals['signal_strength'].loc[end_date] = 60 + min(pattern['strength'] * 40, 30)
            
            elif pattern['name'] == 'RSI顶背离' and pattern['type'] == 'bearish':
                end_date = pattern['end_date']
                if end_date in signals['sell_signal'].index:
                    signals['sell_signal'].loc[end_date] = True
                    signals['signal_strength'].loc[end_date] = 60 + min(pattern['strength'] * 40, 30)
    
        return signals
        
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算指标原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分(0-100)
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None or 'rsi' not in self._result.columns:
            return pd.Series(50.0, index=data.index)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)
        
        # 获取RSI值
        rsi = self._result['rsi']
        
        # 根据RSI值计算评分
        # RSI在30-70之间时，得分随RSI线性变化
        # RSI低于30时，得分随RSI降低而提高
        # RSI高于70时，得分随RSI提高而降低
        for i in range(len(rsi)):
            if rsi.iloc[i] < self.oversold:
                # RSI在超卖区，得分为50-80，越低得分越高
                score.iloc[i] = 50 + (self.oversold - rsi.iloc[i]) / self.oversold * 30
            elif rsi.iloc[i] > self.overbought:
                # RSI在超买区，得分为50-20，越高得分越低
                score.iloc[i] = 50 - (rsi.iloc[i] - self.overbought) / (100 - self.overbought) * 30
            else:
                # RSI在中间区域，得分为40-60，线性变化
                normalized_rsi = (rsi.iloc[i] - self.oversold) / (self.overbought - self.oversold)
                score.iloc[i] = 40 + normalized_rsi * 20
        
        # 结合背离形态增强评分
        patterns = self.get_patterns(data, **kwargs)
        for pattern in patterns:
            if pattern['name'] == 'RSI底背离' and pattern['type'] == 'bullish':
                # 底背离增加评分
                end_idx = data.index.get_loc(pattern['end_date'])
                adjust_range = min(5, len(score) - end_idx - 1)
                for j in range(adjust_range):
                    idx = end_idx + j
                    strength_factor = pattern['strength'] * (1 - j/adjust_range)  # 随时间衰减
                    score.iloc[idx] = min(score.iloc[idx] + strength_factor * 20, 90)
            
            elif pattern['name'] == 'RSI顶背离' and pattern['type'] == 'bearish':
                # 顶背离降低评分
                end_idx = data.index.get_loc(pattern['end_date'])
                adjust_range = min(5, len(score) - end_idx - 1)
                for j in range(adjust_range):
                    idx = end_idx + j
                    strength_factor = pattern['strength'] * (1 - j/adjust_range)  # 随时间衰减
                    score.iloc[idx] = max(score.iloc[idx] - strength_factor * 20, 10)
        
        return score
    
    def calculate_score(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算最终评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
        
        Returns:
            float: 评分(0-100)
        """
        # 计算原始评分序列
        raw_scores = self.calculate_raw_score(data, **kwargs)
        
        # 如果数据不足，返回中性评分
        if len(raw_scores) < 3:
            return 50.0
        
        # 取最近的评分作为最终评分，但考虑近期趋势
        recent_scores = raw_scores.iloc[-3:]
        trend = recent_scores.iloc[-1] - recent_scores.iloc[0]
        
        # 最终评分 = 最新评分 + 趋势调整
        final_score = recent_scores.iloc[-1] + trend / 2
        
        # 确保评分在0-100范围内
        return max(0, min(100, final_score)) 