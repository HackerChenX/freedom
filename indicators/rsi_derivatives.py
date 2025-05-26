#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RSI衍生指标模块

实现基于RSI的多种衍生指标，包括StochRSI、RSI均线、RSI动量等
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder, highest, lowest
from utils.logger import get_logger

logger = get_logger(__name__)


class STOCHRSI(BaseIndicator):
    """
    随机相对强弱指数(StochRSI)
    
    分类：震荡类指标
    描述：将RSI指标标准化到0-100区间，增强短期超买超卖信号
    """
    
    def __init__(self, period: int = 14, k_period: int = 3, d_period: int = 3):
        """
        初始化随机相对强弱指数(StochRSI)指标
        
        Args:
            period: RSI计算周期，默认为14
            k_period: K值计算周期，默认为3
            d_period: D值计算周期，默认为3
        """
        super().__init__()
        self.period = period
        self.k_period = k_period
        self.d_period = d_period
        self.name = "STOCHRSI"
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 包含价格数据的DataFrame
            required_columns: 所需的列名列表
        
        Raises:
            ValueError: 如果DataFrame不包含所需的列，或者行数少于所需的最小行数
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
        
        min_rows = self.period + self.k_period + self.d_period
        if len(df) < min_rows:
            raise ValueError(f"DataFrame至少需要 {min_rows} 行数据，但只有 {len(df)} 行")
    
    def calculate(self, df: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        计算StochRSI指标
        
        Args:
            df: 包含价格数据的DataFrame
            price_column: 用于计算的价格列名，默认为'close'
        
        Returns:
            包含StochRSI指标结果的DataFrame
        """
        required_columns = [price_column]
        self._validate_dataframe(df, required_columns)
        
        result = pd.DataFrame(index=df.index)
        
        # 计算价格变化
        delta = df[price_column].diff()
        
        # 计算上涨和下跌
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # 计算平均上涨和平均下跌
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # 计算相对强度(RS)
        rs = avg_gain / avg_loss
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        
        # 计算StochRSI
        # StochRSI = (RSI - 最低RSI) / (最高RSI - 最低RSI)
        min_rsi = rsi.rolling(window=self.period).min()
        max_rsi = rsi.rolling(window=self.period).max()
        stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
        
        # 计算K和D值
        result['k'] = stoch_rsi.rolling(window=self.k_period).mean() * 100
        result['d'] = result['k'].rolling(window=self.d_period).mean()
        
        return result
    
    def generate_signals(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 原始数据DataFrame
            result: 计算指标后的DataFrame
        
        Returns:
            添加了交易信号的DataFrame
        """
        # 初始化信号列
        result['buy_signal'] = 0
        result['sell_signal'] = 0
        
        # K线上穿D线买入
        buy_cross = crossover(result['k'].values, result['d'].values)
        result.loc[buy_cross, 'buy_signal'] = 1
        
        # K线下穿D线卖出
        sell_cross = crossunder(result['k'].values, result['d'].values)
        result.loc[sell_cross, 'sell_signal'] = 1
        
        # 超卖区域反转买入(K线从低于20上升到高于20)
        oversold_bounce = (result['k'].shift(1) < 20) & (result['k'] > 20)
        result.loc[oversold_bounce, 'buy_signal'] = 1
        
        # 超买区域反转卖出(K线从高于80下降到低于80)
        overbought_drop = (result['k'].shift(1) > 80) & (result['k'] < 80)
        result.loc[overbought_drop, 'sell_signal'] = 1
        
        return result
    
    def plot(self, df: pd.DataFrame, result: pd.DataFrame, ax=None):
        """
        绘制StochRSI指标图表
        
        Args:
            df: 原始数据DataFrame
            result: 计算指标后的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
        
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 7))
        
        # 绘制K线和D线
        ax.plot(result['k'], label='K线')
        ax.plot(result['d'], label='D线')
        
        # 绘制超买超卖区域
        ax.axhline(y=80, color='r', linestyle='--', alpha=0.3)
        ax.axhline(y=20, color='g', linestyle='--', alpha=0.3)
        
        # 标记买卖信号
        buy_signals = result[result['buy_signal'] == 1].index
        sell_signals = result[result['sell_signal'] == 1].index
        
        ax.scatter(buy_signals, result.loc[buy_signals, 'k'], color='green', marker='^', s=100, label='买入信号')
        ax.scatter(sell_signals, result.loc[sell_signals, 'k'], color='red', marker='v', s=100, label='卖出信号')
        
        ax.set_title(f'随机相对强弱指数(StochRSI) - 周期:{self.period}')
        ax.set_ylabel('StochRSI值')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def compute(self, df: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        计算StochRSI指标并生成交易信号
        
        Args:
            df: 包含价格数据的DataFrame
            price_column: 用于计算的价格列名，默认为'close'
        
        Returns:
            包含StochRSI指标和交易信号的DataFrame
        """
        try:
            result = self.calculate(df, price_column)
            result = self.generate_signals(df, result)
            self._result = result
            return result
        except Exception as e:
            self._error = str(e)
            logger.error(f"计算指标 {self.name} 时出错: {str(e)}")
            raise


class RSIMA(BaseIndicator):
    """
    RSI均线系统(RSIMA)
    
    分类：趋势类指标
    描述：计算RSI的移动平均线系统，用于确认RSI趋势
    """
    
    def __init__(self, rsi_period: int = 14, ma_periods: List[int] = None):
        """
        初始化RSI均线系统(RSIMA)指标
        
        Args:
            rsi_period: RSI计算周期，默认为14
            ma_periods: RSI均线周期列表，默认为[5, 10, 20]
        """
        super().__init__()
        self.rsi_period = rsi_period
        self.ma_periods = ma_periods if ma_periods is not None else [5, 10, 20]
        self.name = "RSIMA"
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 包含价格数据的DataFrame
            required_columns: 所需的列名列表
        
        Raises:
            ValueError: 如果DataFrame不包含所需的列，或者行数少于所需的最小行数
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
        
        min_rows = self.rsi_period + max(self.ma_periods)
        if len(df) < min_rows:
            raise ValueError(f"DataFrame至少需要 {min_rows} 行数据，但只有 {len(df)} 行")
    
    def calculate(self, df: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        计算RSI均线系统
        
        Args:
            df: 包含价格数据的DataFrame
            price_column: 用于计算的价格列名，默认为'close'
        
        Returns:
            包含RSI均线系统结果的DataFrame
        """
        required_columns = [price_column]
        self._validate_dataframe(df, required_columns)
        
        result = pd.DataFrame(index=df.index)
        
        # 计算价格变化
        delta = df[price_column].diff()
        
        # 计算上涨和下跌
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # 计算平均上涨和平均下跌
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        # 计算相对强度(RS)
        rs = avg_gain / avg_loss
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        result['rsi'] = rsi
        
        # 计算RSI的各个均线
        for period in self.ma_periods:
            result[f'rsi_ma{period}'] = result['rsi'].rolling(window=period).mean()
        
        return result
    
    def generate_signals(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 原始数据DataFrame
            result: 计算指标后的DataFrame
        
        Returns:
            添加了交易信号的DataFrame
        """
        # 初始化信号列
        result['buy_signal'] = 0
        result['sell_signal'] = 0
        
        # 如果有多个均线，使用短期均线上穿/下穿长期均线作为信号
        if len(self.ma_periods) >= 2:
            # 按周期排序
            periods = sorted(self.ma_periods)
            short_period = periods[0]
            long_period = periods[-1]
            
            # RSI短期均线上穿长期均线买入
            buy_cross = crossover(
                result[f'rsi_ma{short_period}'].values, 
                result[f'rsi_ma{long_period}'].values
            )
            result.loc[buy_cross, 'buy_signal'] = 1
            
            # RSI短期均线下穿长期均线卖出
            sell_cross = crossunder(
                result[f'rsi_ma{short_period}'].values, 
                result[f'rsi_ma{long_period}'].values
            )
            result.loc[sell_cross, 'sell_signal'] = 1
        
        # RSI上穿50买入
        rsi_above_50 = crossover(result['rsi'].values, np.array([50] * len(result)))
        result.loc[rsi_above_50, 'buy_signal'] = 1
        
        # RSI下穿50卖出
        rsi_below_50 = crossunder(result['rsi'].values, np.array([50] * len(result)))
        result.loc[rsi_below_50, 'sell_signal'] = 1
        
        return result
    
    def plot(self, df: pd.DataFrame, result: pd.DataFrame, ax=None):
        """
        绘制RSI均线系统图表
        
        Args:
            df: 原始数据DataFrame
            result: 计算指标后的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
        
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 7))
        
        # 绘制RSI
        ax.plot(result['rsi'], label=f'RSI({self.rsi_period})')
        
        # 绘制RSI均线
        for period in self.ma_periods:
            ax.plot(result[f'rsi_ma{period}'], label=f'RSI MA{period}')
        
        # 绘制超买超卖区域
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        ax.axhline(y=30, color='g', linestyle='--', alpha=0.3)
        ax.axhline(y=50, color='k', linestyle='-', alpha=0.3)
        
        # 标记买卖信号
        buy_signals = result[result['buy_signal'] == 1].index
        sell_signals = result[result['sell_signal'] == 1].index
        
        ax.scatter(buy_signals, result.loc[buy_signals, 'rsi'], color='green', marker='^', s=100, label='买入信号')
        ax.scatter(sell_signals, result.loc[sell_signals, 'rsi'], color='red', marker='v', s=100, label='卖出信号')
        
        ax.set_title(f'RSI均线系统(RSIMA) - RSI周期:{self.rsi_period}')
        ax.set_ylabel('RSI值')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def compute(self, df: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        计算RSI均线系统并生成交易信号
        
        Args:
            df: 包含价格数据的DataFrame
            price_column: 用于计算的价格列名，默认为'close'
        
        Returns:
            包含RSI均线系统和交易信号的DataFrame
        """
        try:
            result = self.calculate(df, price_column)
            result = self.generate_signals(df, result)
            self._result = result
            return result
        except Exception as e:
            self._error = str(e)
            logger.error(f"计算指标 {self.name} 时出错: {str(e)}")
            raise 