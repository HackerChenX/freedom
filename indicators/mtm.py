#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量指标(MTM)

当日收盘价与N日前收盘价之差
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class MTM(BaseIndicator):
    """
    动量指标(MTM) (MTM)
    
    分类：震荡类指标
    描述：当日收盘价与N日前收盘价之差
    """
    
    def __init__(self, period: int = 14, signal_period: int = 6):
        """
        初始化动量指标(MTM)指标
        
        Args:
            period: 计算周期，默认为14
            signal_period: 信号线周期，默认为6
        """
        super().__init__()
        self.period = period
        self.signal_period = signal_period
        self.name = "MTM"
    
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
        计算MTM指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含MTM指标的DataFrame
        """
        return self.calculate(df)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量指标(MTM)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                
        Returns:
            添加了MTM指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 实现动量指标(MTM)计算逻辑
        # MTM = 当日收盘价 - N日前的收盘价
        df_copy['mtm'] = df_copy['close'] - df_copy['close'].shift(self.period)
        
        # 计算MTM的移动平均线作为信号线
        df_copy['signal'] = df_copy['mtm'].rolling(window=self.signal_period).mean()
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成动量指标(MTM)指标交易信号
        
        Args:
            df: 包含价格数据和MTM指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - mtm_buy_signal: 1=买入信号, 0=无信号
            - mtm_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['mtm', 'signal']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['mtm_buy_signal'] = 0
        df_copy['mtm_sell_signal'] = 0
        
        # MTM上穿信号线为买入信号
        for i in range(1, len(df_copy)):
            if df_copy['mtm'].iloc[i-1] < df_copy['signal'].iloc[i-1] and \
               df_copy['mtm'].iloc[i] > df_copy['signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('mtm_buy_signal')] = 1
            
            # MTM下穿信号线为卖出信号
            elif df_copy['mtm'].iloc[i-1] > df_copy['signal'].iloc[i-1] and \
                 df_copy['mtm'].iloc[i] < df_copy['signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('mtm_sell_signal')] = 1
                
        # MTM上穿0轴
        for i in range(1, len(df_copy)):
            if df_copy['mtm'].iloc[i-1] < 0 and df_copy['mtm'].iloc[i] > 0:
                df_copy.iloc[i, df_copy.columns.get_loc('mtm_buy_signal')] = 1
            
            # MTM下穿0轴
            elif df_copy['mtm'].iloc[i-1] > 0 and df_copy['mtm'].iloc[i] < 0:
                df_copy.iloc[i, df_copy.columns.get_loc('mtm_sell_signal')] = 1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制动量指标(MTM)指标图表
        
        Args:
            df: 包含MTM指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['mtm', 'signal']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制MTM指标线
        ax.plot(df.index, df['mtm'], label='MTM')
        ax.plot(df.index, df['signal'], label='信号线', linestyle='--')
        
        # 添加零轴线
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        ax.set_ylabel('动量指标(MTM)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

