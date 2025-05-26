#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
成交量震荡指标(VOSC)

通过对成交量的长短期移动平均差值的百分比来衡量成交量的变化和趋势
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class VOSC(BaseIndicator):
    """
    成交量震荡指标(VOSC) (VOSC)
    
    分类：量能类指标
    描述：通过对成交量的长短期移动平均差值的百分比来衡量成交量的变化和趋势
    """
    
    def __init__(self, short_period: int = 12, long_period: int = 26):
        """
        初始化成交量震荡指标(VOSC)指标
        
        Args:
            short_period: 短期移动平均周期，默认为12
            long_period: 长期移动平均周期，默认为26
        """
        super().__init__()
        self.short_period = short_period
        self.long_period = long_period
        self.name = "VOSC"
        
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
        计算VOSC指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含VOSC指标的DataFrame
        """
        return self.calculate(df)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量震荡指标(VOSC)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - volume: 成交量
                
        Returns:
            添加了VOSC指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['volume']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算短期和长期成交量移动平均
        short_ma = df_copy['volume'].rolling(window=self.short_period).mean()
        long_ma = df_copy['volume'].rolling(window=self.long_period).mean()
        
        # 计算VOSC值
        # VOSC = (短期成交量均线 - 长期成交量均线) / 长期成交量均线 * 100
        df_copy['vosc'] = (short_ma - long_ma) / long_ma * 100
        
        # 计算VOSC的移动平均作为信号线
        df_copy['vosc_signal'] = df_copy['vosc'].rolling(window=9).mean()
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成成交量震荡指标(VOSC)指标交易信号
        
        Args:
            df: 包含价格数据和VOSC指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - vosc_buy_signal: 1=买入信号, 0=无信号
            - vosc_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['vosc', 'vosc_signal']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['vosc_buy_signal'] = 0
        df_copy['vosc_sell_signal'] = 0
        
        # VOSC上穿信号线为买入信号
        for i in range(1, len(df_copy)):
            if df_copy['vosc'].iloc[i-1] < df_copy['vosc_signal'].iloc[i-1] and \
               df_copy['vosc'].iloc[i] > df_copy['vosc_signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('vosc_buy_signal')] = 1
            
            # VOSC下穿信号线为卖出信号
            elif df_copy['vosc'].iloc[i-1] > df_copy['vosc_signal'].iloc[i-1] and \
                 df_copy['vosc'].iloc[i] < df_copy['vosc_signal'].iloc[i]:
                df_copy.iloc[i, df_copy.columns.get_loc('vosc_sell_signal')] = 1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制成交量震荡指标(VOSC)指标图表
        
        Args:
            df: 包含VOSC指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['vosc', 'vosc_signal']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制VOSC指标线
        ax.plot(df.index, df['vosc'], label='VOSC')
        ax.plot(df.index, df['vosc_signal'], label='信号线', linestyle='--')
        
        # 添加零轴线
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        ax.set_ylabel('成交量震荡指标(VOSC)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

