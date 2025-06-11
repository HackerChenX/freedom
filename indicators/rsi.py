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
        计算RSI指标，并包含均线和信号
        
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
            
        result_df = pd.DataFrame(index=df.index)
        
        # 计算价格变动
        delta = df['close'].diff()
        
        # 计算上涨和下跌
        gain = delta.where(delta > 0, 0).ewm(span=self.period, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(span=self.period, adjust=False).mean()

        # 计算相对强度
        rs = gain / loss.replace(0, 1e-9)
        
        # 计算RSI
        result_df['rsi'] = 100 - (100 / (1 + rs))
        
        # 可选：计算RSI均线
        if self.ma_periods and len(self.ma_periods) >= 2:
            result_df[f'rsi_ma_{self.ma_periods[0]}'] = result_df['rsi'].rolling(window=self.ma_periods[0]).mean()
            result_df[f'rsi_ma_{self.ma_periods[1]}'] = result_df['rsi'].rolling(window=self.ma_periods[1]).mean()
            # For pattern detection
            result_df['rsi_ma_short'] = result_df[f'rsi_ma_{self.ma_periods[0]}']
            result_df['rsi_ma_long'] = result_df[f'rsi_ma_{self.ma_periods[1]}']

        result_df['rsi_overbought'] = result_df['rsi'] > self.overbought
        result_df['rsi_oversold'] = result_df['rsi'] < self.oversold

        return df.join(result_df)

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取RSI相关形态
        """
        calculated_data = self._calculate(data)
        patterns_df = pd.DataFrame(index=data.index)

        # 确保列存在
        if 'rsi' not in calculated_data.columns or 'rsi_ma_short' not in calculated_data.columns or 'rsi_ma_long' not in calculated_data.columns:
            return patterns_df

        # 金叉和死叉
        from indicators.common import crossover, crossunder
        patterns_df['RSI_GOLDEN_CROSS'] = crossover(calculated_data['rsi_ma_short'], calculated_data['rsi_ma_long'])
        patterns_df['RSI_DEATH_CROSS'] = crossunder(calculated_data['rsi_ma_short'], calculated_data['rsi_ma_long'])

        # 超买和超卖
        patterns_df['RSI_OVERBOUGHT'] = calculated_data['rsi'] > self.overbought
        patterns_df['RSI_OVERSOLD'] = calculated_data['rsi'] < self.oversold
        
        return patterns_df

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成RSI交易信号
        """
        calculated_data = self._calculate(data)
        
        patterns = self.get_patterns(calculated_data)
        
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = patterns['RSI_GOLDEN_CROSS'] | (patterns['RSI_OVERSOLD'])
        signals['sell_signal'] = patterns['RSI_DEATH_CROSS'] | (patterns['RSI_OVERBOUGHT'])

        return signals

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算RSI指标的原始评分 (0-100分)
        """
        calculated_data = self._calculate(data)
        
        if calculated_data is None or 'rsi' not in calculated_data.columns:
            return pd.Series(50.0, index=data.index)
            
        score = pd.Series(50.0, index=data.index)
        rsi_values = calculated_data['rsi']
        
        # 基于RSI值的评分
        score += (rsi_values - 50) * 0.4 # 20-80 映射到 42-58
        
        # 超买超卖区域评分
        score[rsi_values > self.overbought] -= 15
        score[rsi_values < self.oversold] += 15
        
        # 均线交叉评分
        if 'rsi_ma_short' in calculated_data.columns and 'rsi_ma_long' in calculated_data.columns:
            short_ma = calculated_data['rsi_ma_short']
            long_ma = calculated_data['rsi_ma_long']
            
            from indicators.common import crossover, crossunder
            score[crossover(short_ma, long_ma)] += 20
            score[crossunder(short_ma, long_ma)] -= 20
            
        return score.clip(0, 100) 