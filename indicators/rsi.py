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

    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算RSI指标
        
        Args:
            df: 包含价格数据的DataFrame
            
        Returns:
            pd.DataFrame: 添加了RSI指标的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含所需的列
        if 'close' not in df.columns:
            raise ValueError("输入数据必须包含'close'列")
            
        df_copy = df.copy()
        
        # 计算价格变动
        delta = df_copy['close'].diff()
        
        # 计算上涨和下跌
        gain = delta.where(delta > 0, 0).ewm(span=self.period, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(span=self.period, adjust=False).mean()

        # 计算相对强度
        rs = gain / loss.replace(0, 1e-9)
        
        # 计算RSI
        df_copy['rsi'] = 100 - (100 / (1 + rs))
        
        # 可选：计算RSI均线
        if self.ma_periods:
            for period in self.ma_periods:
                df_copy[f'rsi_ma_{period}'] = df_copy['rsi'].rolling(window=period).mean()
        
        # 保存结果
        self._result = df_copy
        
        # 确保基础数据列被保留
        df_copy = self._preserve_base_columns(df, df_copy)
        
        return df_copy

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算RSI指标，并包含均线和信号
        """
        if df.empty:
            return df

        rsi_df = self.calculate(df)
        
        if 'rsi' in rsi_df.columns and not rsi_df['rsi'].empty:
            rsi_df['rsi_ma_short'] = rsi_df['rsi'].rolling(window=self.ma_periods[0]).mean()
            rsi_df['rsi_ma_long'] = rsi_df['rsi'].rolling(window=self.ma_periods[1]).mean()
            rsi_df['rsi_overbought'] = rsi_df['rsi'] > self.overbought
            rsi_df['rsi_oversold'] = rsi_df['rsi'] < self.oversold

        result = df.join(rsi_df)
        self._result = result
        return result

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取RSI相关形态
        """
        if not self.has_result() or 'rsi_ma_short' not in self._result.columns:
            self.compute(data, **kwargs)

        result = self._result.copy()
        patterns_df = pd.DataFrame(index=data.index)

        # 确保列存在
        if 'rsi' not in result.columns or 'rsi_ma_short' not in result.columns or 'rsi_ma_long' not in result.columns:
            return patterns_df

        # 金叉和死叉
        from indicators.common import crossover, crossunder
        patterns_df['RSI_GOLDEN_CROSS'] = crossover(result['rsi_ma_short'], result['rsi_ma_long'])
        patterns_df['RSI_DEATH_CROSS'] = crossunder(result['rsi_ma_short'], result['rsi_ma_long'])

        # 超买和超卖
        patterns_df['RSI_OVERBOUGHT'] = result['rsi'] > self.overbought
        patterns_df['RSI_OVERSOLD'] = result['rsi'] < self.oversold
        
        return patterns_df

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成RSI交易信号
        """
        if not self.has_result():
            self.compute(data, **kwargs)
        
        patterns = self.get_patterns(self._result)
        
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = patterns['RSI_GOLDEN_CROSS'] | (patterns['RSI_OVERSOLD'])
        signals['sell_signal'] = patterns['RSI_DEATH_CROSS'] | (patterns['RSI_OVERBOUGHT'])

        return signals

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算RSI指标的原始评分 (0-100分)
        """
        if not self.has_result():
            self.compute(data, **kwargs)
        
        if self._result is None or 'rsi' not in self._result.columns:
            return pd.Series(50.0, index=data.index)
            
        score = pd.Series(50.0, index=data.index)
        rsi_values = self._result['rsi']
        
        # 基于RSI值的评分
        score += (rsi_values - 50) * 0.4 # 20-80 映射到 42-58
        
        # 超买超卖区域评分
        score[rsi_values > self.overbought] -= 15
        score[rsi_values < self.oversold] += 15
        
        # 均线交叉评分
        if 'rsi_ma_short' in self._result.columns and 'rsi_ma_long' in self._result.columns:
            short_ma = self._result['rsi_ma_short']
            long_ma = self._result['rsi_ma_long']
            
            from indicators.common import crossover, crossunder
            score[crossover(short_ma, long_ma)] += 20
            score[crossunder(short_ma, long_ma)] -= 20
            
        return score.clip(0, 100) 