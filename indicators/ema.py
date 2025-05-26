#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
指数移动平均线(EMA)

对近期价格赋予更高权重
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class EMA(BaseIndicator):
    """
    指数移动平均线(EMA) (EMA)
    
    分类：趋势类指标
    描述：对近期价格赋予更高权重
    """
    
    def __init__(self, period: int = 14, periods: List[int] = None):
        """
        初始化指数移动平均线(EMA)指标
        
        Args:
            period: 计算周期，默认为14
            periods: 多个计算周期，如果提供，将计算多个周期的EMA
        """
        super().__init__()
        self.period = period
        self.periods = periods if periods is not None else [period]
        self.name = "EMA"
        
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
        计算指数移动平均线(EMA)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了EMA指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算不同周期的EMA
        for p in self.periods:
            df_copy[f'EMA{p}'] = df_copy['close'].ewm(span=p, adjust=False).mean()
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成指数移动平均线(EMA)指标交易信号
        
        Args:
            df: 包含价格数据和EMA指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值
                oversold: 超卖阈值
                
        Returns:
            添加了信号列的DataFrame:
            - ema_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = [f'EMA{self.periods[0]}']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy[f'ema_signal'] = 0
        
        # 如果有多个周期，可以检测金叉和死叉
        if len(self.periods) >= 2 and self.periods[0] < self.periods[1]:
            short_period = self.periods[0]
            long_period = self.periods[1]
            
            # 金叉信号（短期EMA上穿长期EMA）
            df_copy.loc[crossover(df_copy[f'EMA{short_period}'], df_copy[f'EMA{long_period}']), f'ema_signal'] = 1
            
            # 死叉信号（短期EMA下穿长期EMA）
            df_copy.loc[crossunder(df_copy[f'EMA{short_period}'], df_copy[f'EMA{long_period}']), f'ema_signal'] = -1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制指数移动平均线(EMA)指标图表
        
        Args:
            df: 包含EMA指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        for p in self.periods:
            required_columns = [f'EMA{p}']
            self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制各个周期的EMA指标线
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
        for i, p in enumerate(self.periods):
            color = colors[i % len(colors)]
            ax.plot(df.index, df[f'EMA{p}'], label=f'EMA({p})', color=color)
        
        ax.set_ylabel(f'指数移动平均线(EMA)')
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

