#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RSI均线系统指标(RSIMA)

使用RSI的移动平均线系统来确认RSI趋势，增强趋势判断
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


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
        # 使用较小的默认周期，以便在数据量较少时也能计算
        self.ma_periods = ma_periods if ma_periods is not None else [3, 5, 10]
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
        
        # 数据行数至少要能计算RSI值
        min_rows = self.rsi_period + 1
        if len(df) < min_rows:
            raise ValueError(f"DataFrame至少需要 {min_rows} 行数据才能计算RSI，但只有 {len(df)} 行")
    
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
        
        df_copy = df.copy()
        
        # 计算价格变化
        delta = df_copy[price_column].diff()
        
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
        df_copy['rsi'] = rsi
        
        # 计算可用的均线周期
        available_periods = []
        for period in self.ma_periods:
            # 如果数据行数足够计算该周期的均线，则添加到可用周期列表
            if len(df_copy) >= period + self.rsi_period:
                available_periods.append(period)
                df_copy[f'rsi_ma{period}'] = df_copy['rsi'].rolling(window=period).mean()
        
        # 如果没有可用的均线周期，至少计算一个3日均线
        if not available_periods and len(df_copy) >= self.rsi_period + 3:
            df_copy['rsi_ma3'] = df_copy['rsi'].rolling(window=3).mean()
            available_periods.append(3)
        
        # 记录可用的均线周期，供信号生成时使用
        self._available_periods = available_periods
        
        return df_copy
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算RSI均线系统指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含RSI均线系统指标的DataFrame
        """
        try:
            result = self.calculate(df)
            result = self.get_signals(result)
            return result
        except Exception as e:
            logger.error(f"计算指标 {self.name} 时出错: {str(e)}")
            raise
    
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成RSI均线系统指标交易信号
        
        Args:
            df: 包含价格数据和RSIMA指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - rsima_buy_signal: 1=买入信号, 0=无信号
            - rsima_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['rsi']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['rsima_buy_signal'] = 0
        df_copy['rsima_sell_signal'] = 0
        
        # 使用可用的均线周期生成信号
        available_periods = getattr(self, '_available_periods', [])
        
        # 如果有多个均线，使用短期均线上穿/下穿长期均线作为信号
        if len(available_periods) >= 2:
            # 按周期排序
            periods = sorted(available_periods)
            short_period = periods[0]
            long_period = periods[-1]
            
            # RSI短期均线上穿长期均线买入
            for i in range(1, len(df_copy)):
                if df_copy[f'rsi_ma{short_period}'].iloc[i-1] < df_copy[f'rsi_ma{long_period}'].iloc[i-1] and \
                   df_copy[f'rsi_ma{short_period}'].iloc[i] > df_copy[f'rsi_ma{long_period}'].iloc[i]:
                    df_copy.iloc[i, df_copy.columns.get_loc('rsima_buy_signal')] = 1
                
                # RSI短期均线下穿长期均线卖出
                elif df_copy[f'rsi_ma{short_period}'].iloc[i-1] > df_copy[f'rsi_ma{long_period}'].iloc[i-1] and \
                     df_copy[f'rsi_ma{short_period}'].iloc[i] < df_copy[f'rsi_ma{long_period}'].iloc[i]:
                    df_copy.iloc[i, df_copy.columns.get_loc('rsima_sell_signal')] = 1
        
        # RSI上穿50买入
        for i in range(1, len(df_copy)):
            if df_copy['rsi'].iloc[i-1] < 50 and df_copy['rsi'].iloc[i] > 50:
                df_copy.iloc[i, df_copy.columns.get_loc('rsima_buy_signal')] = 1
            
            # RSI下穿50卖出
            elif df_copy['rsi'].iloc[i-1] > 50 and df_copy['rsi'].iloc[i] < 50:
                df_copy.iloc[i, df_copy.columns.get_loc('rsima_sell_signal')] = 1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制RSI均线系统指标图表
        
        Args:
            df: 包含RSIMA指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['rsi']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制RSI线
        ax.plot(df.index, df['rsi'], label=f'RSI({self.rsi_period})')
        
        # 绘制RSI均线
        available_periods = getattr(self, '_available_periods', [])
        for period in available_periods:
            if f'rsi_ma{period}' in df.columns:
                ax.plot(df.index, df[f'rsi_ma{period}'], label=f'RSI MA{period}', linestyle='--')
        
        # 添加参考线
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        ax.axhline(y=30, color='g', linestyle='--', alpha=0.3)
        ax.axhline(y=50, color='k', linestyle='--', alpha=0.3)
        
        ax.set_ylabel('RSI均线系统(RSIMA)')
        ax.set_ylim([0, 100])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax 