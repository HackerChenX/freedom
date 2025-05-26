#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
加权移动平均线(WMA)

对不同时期价格赋予不同权重
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class WMA(BaseIndicator):
    """
    加权移动平均线(WMA) (WMA)
    
    分类：趋势类指标
    描述：对不同时期价格赋予不同权重
    """
    
    def __init__(self, period: int = 14, periods: List[int] = None):
        """
        初始化加权移动平均线(WMA)指标
        
        Args:
            period: 计算周期，默认为14
            periods: 多个计算周期，如果提供，将计算多个周期的WMA
        """
        super().__init__()
        self.period = period
        self.periods = periods if periods is not None else [period]
        self.name = "WMA"
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 要验证的DataFrame
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算加权移动平均线(WMA)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了WMA指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算不同周期的WMA
        for p in self.periods:
            # 创建权重数组，权重与周期成正比
            weights = np.arange(1, p + 1)
            # 计算权重和
            weights_sum = weights.sum()
            
            # 应用加权移动平均计算
            df_copy[f'WMA{p}'] = df_copy['close'].rolling(window=p).apply(
                lambda x: np.sum(x * weights[-len(x):]) / np.sum(weights[-len(x):]), 
                raw=True
            )
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成加权移动平均线(WMA)指标交易信号
        
        Args:
            df: 包含价格数据和WMA指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值
                oversold: 超卖阈值
                
        Returns:
            添加了信号列的DataFrame:
            - wma_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy[f'wma_signal'] = 0
        
        # 如果有多个周期，可以检测金叉和死叉
        if len(self.periods) >= 2 and self.periods[0] < self.periods[1]:
            short_period = self.periods[0]
            long_period = self.periods[1]
            
            # 检查必要的指标列是否存在
            required_columns = [f'WMA{short_period}', f'WMA{long_period}']
            self._validate_dataframe(df_copy, required_columns)
            
            # 金叉信号（短期WMA上穿长期WMA）
            df_copy.loc[crossover(df_copy[f'WMA{short_period}'], df_copy[f'WMA{long_period}']), f'wma_signal'] = 1
            
            # 死叉信号（短期WMA下穿长期WMA）
            df_copy.loc[crossunder(df_copy[f'WMA{short_period}'], df_copy[f'WMA{long_period}']), f'wma_signal'] = -1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制加权移动平均线(WMA)指标图表
        
        Args:
            df: 包含WMA指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        
        # 绘制各个周期的WMA指标线
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for i, p in enumerate(self.periods):
            # 检查必要的指标列是否存在
            required_columns = [f'WMA{p}']
            self._validate_dataframe(df, required_columns)
            
            color = colors[i % len(colors)]
            ax.plot(df.index, df[f'WMA{p}'], label=f'WMA({p})', color=color)
        
        ax.set_ylabel(f'加权移动平均线(WMA)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax
        
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标并返回结果
        
        Args:
            df: 输入DataFrame
            
        Returns:
            包含计算结果的DataFrame
        """
        return self.calculate(df)

