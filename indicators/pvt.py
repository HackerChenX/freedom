#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
价格成交量趋势指标(PVT)

通过价格变化与成交量相结合，反映价格趋势的强度和持续性
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class PVT(BaseIndicator):
    """
    价格成交量趋势指标(PVT) (PVT)
    
    分类：量能类指标
    描述：通过价格变化与成交量相结合，反映价格趋势的强度和持续性
    """
    
    def __init__(self, ma_period: int = 12):
        """
        初始化价格成交量趋势指标(PVT)指标
        
        Args:
            ma_period: 移动平均周期，默认为12
        """
        super().__init__()
        self.ma_period = ma_period
        self.name = "PVT"
        
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
        计算PVT指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含PVT指标的DataFrame
        """
        return self.calculate(df)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格成交量趋势指标(PVT)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - volume: 成交量
                
        Returns:
            添加了PVT指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'volume']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算价格变化率
        price_change = df_copy['close'].pct_change()
        
        # 计算PVT
        # PVT = 昨日PVT + 今日成交量 * 价格变化率
        df_copy['pvt'] = df_copy['volume'] * price_change
        df_copy['pvt'] = df_copy['pvt'].cumsum()
        
        # 计算PVT的移动平均作为信号线
        df_copy['pvt_signal'] = df_copy['pvt'].rolling(window=self.ma_period).mean()
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成价格成交量趋势指标(PVT)指标交易信号
        
        Args:
            df: 包含价格数据和PVT指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - pvt_buy_signal: 1=买入信号, 0=无信号
            - pvt_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['pvt', 'pvt_signal']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['pvt_buy_signal'] = 0
        df_copy['pvt_sell_signal'] = 0
        
        # PVT上穿信号线为买入信号
        for i in range(1, len(df_copy)):
            if df_copy['pvt'].iloc[i-1] < df_copy['pvt_signal'].iloc[i-1] and \
               df_copy['pvt'].iloc[i] > df_copy['pvt_signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('pvt_buy_signal')] = 1
            
            # PVT下穿信号线为卖出信号
            elif df_copy['pvt'].iloc[i-1] > df_copy['pvt_signal'].iloc[i-1] and \
                 df_copy['pvt'].iloc[i] < df_copy['pvt_signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('pvt_sell_signal')] = 1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制价格成交量趋势指标(PVT)指标图表
        
        Args:
            df: 包含PVT指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['pvt', 'pvt_signal']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制PVT指标线
        ax.plot(df.index, df['pvt'], label='PVT')
        ax.plot(df.index, df['pvt_signal'], label='信号线', linestyle='--')
        
        # 添加零轴线
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        ax.set_ylabel('价格成交量趋势指标(PVT)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

