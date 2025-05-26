#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
趋向指标(DMI)

判断趋势强度与方向
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class DMI(BaseIndicator):
    """
    趋向指标(DMI) (DMI)
    
    分类：趋势类指标
    描述：判断趋势强度与方向
    """
    
    def __init__(self, period: int = 14, adx_period: int = 14):
        """
        初始化趋向指标(DMI)指标
        
        Args:
            period: 计算周期，默认为14
            adx_period: ADX计算周期，默认为14
        """
        super().__init__()
        self.period = period
        self.adx_period = adx_period
        self.name = "DMI"
    
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
        计算趋向指标(DMI)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了DMI指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算真实波幅TR
        df_copy['high_low'] = df_copy['high'] - df_copy['low']
        df_copy['high_close'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['low_close'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        df_copy['TR'] = df_copy[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # 计算方向线DM
        df_copy['up_move'] = df_copy['high'] - df_copy['high'].shift(1)
        df_copy['down_move'] = df_copy['low'].shift(1) - df_copy['low']
        
        # 计算+DM和-DM
        df_copy['+DM'] = np.where((df_copy['up_move'] > df_copy['down_move']) & (df_copy['up_move'] > 0), 
                                df_copy['up_move'], 0)
        df_copy['-DM'] = np.where((df_copy['down_move'] > df_copy['up_move']) & (df_copy['down_move'] > 0), 
                                df_copy['down_move'], 0)
        
        # 计算平滑后的TR、+DM和-DM
        df_copy['TR_' + str(self.period)] = df_copy['TR'].rolling(window=self.period).sum()
        df_copy['+DM_' + str(self.period)] = df_copy['+DM'].rolling(window=self.period).sum()
        df_copy['-DM_' + str(self.period)] = df_copy['-DM'].rolling(window=self.period).sum()
        
        # 计算+DI和-DI
        df_copy['PDI'] = 100 * df_copy['+DM_' + str(self.period)] / df_copy['TR_' + str(self.period)]
        df_copy['MDI'] = 100 * df_copy['-DM_' + str(self.period)] / df_copy['TR_' + str(self.period)]
        
        # 计算方向指数DX
        df_copy['DX'] = 100 * abs(df_copy['PDI'] - df_copy['MDI']) / (df_copy['PDI'] + df_copy['MDI'])
        
        # 计算平均方向指数ADX
        df_copy['ADX'] = df_copy['DX'].rolling(window=self.adx_period).mean()
        
        # 计算平均方向指数评估ADXR
        df_copy['ADXR'] = (df_copy['ADX'] + df_copy['ADX'].shift(self.adx_period)) / 2
        
        # 删除中间计算列
        df_copy.drop(['high_low', 'high_close', 'low_close', 'TR', 'up_move', 'down_move',
                     '+DM', '-DM', 'TR_' + str(self.period), '+DM_' + str(self.period), 
                     '-DM_' + str(self.period), 'DX'], axis=1, inplace=True)
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成趋向指标(DMI)指标交易信号
        
        Args:
            df: 包含价格数据和DMI指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - dmi_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['PDI', 'MDI', 'ADX']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['dmi_signal'] = 0
        
        # +DI上穿-DI为买入信号
        df_copy.loc[crossover(df_copy['PDI'], df_copy['MDI']), 'dmi_signal'] = 1
        
        # -DI上穿+DI为卖出信号
        df_copy.loc[crossover(df_copy['MDI'], df_copy['PDI']), 'dmi_signal'] = -1
        
        # 强化信号：ADX > 25表示趋势显著
        df_copy.loc[(df_copy['dmi_signal'] == 1) & (df_copy['ADX'] < 25), 'dmi_signal'] = 0
        df_copy.loc[(df_copy['dmi_signal'] == -1) & (df_copy['ADX'] < 25), 'dmi_signal'] = 0
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制趋向指标(DMI)指标图表
        
        Args:
            df: 包含DMI指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['PDI', 'MDI', 'ADX', 'ADXR']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制指标线
        ax.plot(df.index, df['PDI'], label='+DI', color='g')
        ax.plot(df.index, df['MDI'], label='-DI', color='r')
        ax.plot(df.index, df['ADX'], label='ADX', color='b')
        ax.plot(df.index, df['ADXR'], label='ADXR', color='m', linestyle='--')
        
        # 添加参考线
        ax.axhline(y=25, color='k', linestyle='--', alpha=0.3, label='趋势阈值')
        
        ax.set_ylabel('DMI指标')
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

