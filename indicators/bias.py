#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
均线多空指标(BIAS)

(收盘价-MA)/MA×100%
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class BIAS(BaseIndicator):
    """
    均线多空指标(BIAS) (BIAS)
    
    分类：趋势类指标
    描述：(收盘价-MA)/MA×100%
    """
    
    def __init__(self, period: int = 14, periods: List[int] = None):
        """
        初始化均线多空指标(BIAS)指标
        
        Args:
            period: 计算周期，默认为14
            periods: 多个计算周期，如果提供，将计算多个周期的BIAS
        """
        super().__init__()
        self.period = period
        self.periods = periods if periods is not None else [period]
        self.name = "BIAS"
        
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
        计算均线多空指标(BIAS)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了BIAS指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算不同周期的BIAS
        for p in self.periods:
            # 计算移动平均线
            ma = df_copy['close'].rolling(window=p).mean()
            
            # 计算BIAS
            df_copy[f'BIAS{p}'] = (df_copy['close'] - ma) / ma * 100
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成均线多空指标(BIAS)指标交易信号
        
        Args:
            df: 包含价格数据和BIAS指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值
                oversold: 超卖阈值
                
        Returns:
            添加了信号列的DataFrame:
            - bias_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy[f'bias_signal'] = 0
        
        # 获取参数
        overbought = kwargs.get('overbought', 6)  # 超买阈值
        oversold = kwargs.get('oversold', -6)  # 超卖阈值
        
        # 使用主要周期的BIAS生成信号
        main_period = self.periods[0]
        bias_col = f'BIAS{main_period}'
        
        # 检查必要的指标列是否存在
        if bias_col in df_copy.columns:
            # 超卖区域上穿信号线为买入信号
            df_copy.loc[crossover(df_copy[bias_col], oversold), f'bias_signal'] = 1
            
            # 超买区域下穿信号线为卖出信号
            df_copy.loc[crossunder(df_copy[bias_col], overbought), f'bias_signal'] = -1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制均线多空指标(BIAS)指标图表
        
        Args:
            df: 包含BIAS指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        
        # 绘制各个周期的BIAS指标线
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for i, p in enumerate(self.periods):
            bias_col = f'BIAS{p}'
            if bias_col in df.columns:
                color = colors[i % len(colors)]
                ax.plot(df.index, df[bias_col], label=f'BIAS({p})', color=color)
        
        # 添加超买超卖参考线
        overbought = kwargs.get('overbought', 6)
        oversold = kwargs.get('oversold', -6)
        ax.axhline(y=overbought, color='r', linestyle='--', alpha=0.3)
        ax.axhline(y=oversold, color='g', linestyle='--', alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        ax.set_ylabel(f'均线多空指标(BIAS)')
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

