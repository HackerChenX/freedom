#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
威廉指标(WR)

与KDJ配合使用，确认超买超卖
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class WR(BaseIndicator):
    """
    威廉指标(WR) (WR)
    
    分类：震荡类指标
    描述：与KDJ配合使用，确认超买超卖
    """
    
    def __init__(self, period: int = 14):
        """
        初始化威廉指标(WR)指标
        
        Args:
            period: 计算周期，默认为14
        """
        super().__init__()
        self.period = period
        self.name = "WR"
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 输入数据
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算WR指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含WR指标的DataFrame
        """
        return self.calculate(df)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算威廉指标(WR)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了WR指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 实现威廉指标(WR)计算逻辑
        # WR = -100 * (HIGH(n) - CLOSE) / (HIGH(n) - LOW(n))
        # 其中HIGH(n)和LOW(n)分别为n周期内的最高价和最低价
        highest_high = df_copy['high'].rolling(window=self.period).max()
        lowest_low = df_copy['low'].rolling(window=self.period).min()
        
        # 计算WR值
        df_copy['wr'] = -100 * (highest_high - df_copy['close']) / (highest_high - lowest_low)
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成威廉指标(WR)指标交易信号
        
        Args:
            df: 包含价格数据和WR指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值
                oversold: 超卖阈值
                
        Returns:
            添加了信号列的DataFrame:
            - wr_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['wr']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 获取参数
        overbought = kwargs.get('overbought', -20)  # 超买阈值
        oversold = kwargs.get('oversold', -80)  # 超卖阈值
        
        # 实现信号生成逻辑
        df_copy['wr_signal'] = 0
        
        # 超卖区域上穿信号线为买入信号
        for i in range(1, len(df_copy)):
            if df_copy['wr'].iloc[i-1] < oversold and df_copy['wr'].iloc[i] > oversold:
                df_copy.iloc[i, df_copy.columns.get_loc('wr_signal')] = 1
            
            # 超买区域下穿信号线为卖出信号
            elif df_copy['wr'].iloc[i-1] > overbought and df_copy['wr'].iloc[i] < overbought:
                df_copy.iloc[i, df_copy.columns.get_loc('wr_signal')] = -1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制威廉指标(WR)指标图表
        
        Args:
            df: 包含WR指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['wr']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制WR指标线
        ax.plot(df.index, df['wr'], label='威廉指标(WR)')
        
        # 添加超买超卖参考线
        ax.axhline(y=-20, color='r', linestyle='--', alpha=0.3, label='超买区域(-20)')
        ax.axhline(y=-50, color='k', linestyle='--', alpha=0.3, label='中轴线(-50)')
        ax.axhline(y=-80, color='g', linestyle='--', alpha=0.3, label='超卖区域(-80)')
        
        ax.set_ylabel('威廉指标(WR)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

