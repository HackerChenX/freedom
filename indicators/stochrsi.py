#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
随机相对强弱指数(STOCHRSI)

将RSI指标标准化到0-100区间，增强短期超买超卖信号
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class STOCHRSI(BaseIndicator):
    """
    随机相对强弱指数(STOCHRSI) (STOCHRSI)
    
    分类：震荡类指标
    描述：将RSI指标标准化到0-100区间，增强短期超买超卖信号
    """
    
    def __init__(self, period: int = 14, k_period: int = 3, d_period: int = 3):
        """
        初始化随机相对强弱指数(STOCHRSI)指标
        
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
            df: 输入数据
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算随机相对强弱指数(STOCHRSI)指标
        
        Args:
            df: 包含价格数据的DataFrame
                
        Returns:
            包含STOCHRSI指标的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 步骤1: 计算RSI
        delta = df_copy['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # 步骤2: 计算StochRSI
        min_rsi = rsi.rolling(window=self.period).min()
        max_rsi = rsi.rolling(window=self.period).max()
        stoch_rsi = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)
        
        # 步骤3: 计算K和D值
        k = stoch_rsi.rolling(window=self.k_period).mean()
        d = k.rolling(window=self.d_period).mean()
        
        # 保存结果
        df_copy['rsi'] = rsi
        df_copy['stochrsi'] = stoch_rsi
        df_copy['k'] = k
        df_copy['d'] = d
        
        return df_copy
    
    # 兼容新接口
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算随机相对强弱指数(STOCHRSI)指标 - 兼容新接口
        
        Args:
            df: 包含价格数据的DataFrame
                
        Returns:
            包含STOCHRSI指标的DataFrame
        """
        return self.calculate(df)
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成随机相对强弱指数(STOCHRSI)指标交易信号
        
        Args:
            df: 包含价格数据和STOCHRSI指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值，默认为80
                oversold: 超卖阈值，默认为20
                
        Returns:
            添加了信号列的DataFrame:
            - stochrsi_buy_signal: 1=买入信号, 0=无信号
            - stochrsi_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['k', 'd']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 获取参数
        overbought = kwargs.get('overbought', 80)  # 超买阈值
        oversold = kwargs.get('oversold', 20)  # 超卖阈值
        
        # 初始化信号列
        df_copy['stochrsi_buy_signal'] = 0
        df_copy['stochrsi_sell_signal'] = 0
        
        # K上穿D为买入信号
        df_copy.loc[crossover(df_copy['k'], df_copy['d']), 'stochrsi_buy_signal'] = 1
        
        # K下穿D为卖出信号
        df_copy.loc[crossunder(df_copy['k'], df_copy['d']), 'stochrsi_sell_signal'] = 1
        
        # 超卖区域上穿为买入信号
        for i in range(1, len(df_copy)):
            if df_copy['k'].iloc[i-1] < oversold and df_copy['k'].iloc[i] > oversold:
                df_copy.iloc[i, df_copy.columns.get_loc('stochrsi_buy_signal')] = 1
            
            # 超买区域下穿为卖出信号
            elif df_copy['k'].iloc[i-1] > overbought and df_copy['k'].iloc[i] < overbought:
                df_copy.iloc[i, df_copy.columns.get_loc('stochrsi_sell_signal')] = 1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制随机相对强弱指数(STOCHRSI)指标图表
        
        Args:
            df: 包含STOCHRSI指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['k', 'd']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制K和D线
        ax.plot(df.index, df['k'], label='%K', color='blue')
        ax.plot(df.index, df['d'], label='%D', color='red', linestyle='--')
        
        # 添加超买超卖参考线
        ax.axhline(y=80, color='r', linestyle='--', alpha=0.3, label='超买区域(80)')
        ax.axhline(y=20, color='g', linestyle='--', alpha=0.3, label='超卖区域(20)')
        ax.axhline(y=50, color='k', linestyle='--', alpha=0.3)
        
        ax.set_ylabel('随机相对强弱指数(STOCHRSI)')
        ax.set_ylim([0, 100])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

