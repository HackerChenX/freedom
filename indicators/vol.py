#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
成交量(VOL)

市场活跃度、参与度直观体现
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class VOL(BaseIndicator):
    """
    成交量(VOL) (VOL)
    
    分类：量能类指标
    描述：市场活跃度、参与度直观体现
    """
    
    def __init__(self, period: int = 14):
        """
        初始化成交量(VOL)指标
        
        Args:
            period: 计算周期，默认为14
        """
        super().__init__()
        self.period = period
        self.name = "VOL"
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含VOL指标的DataFrame
        """
        return self.calculate(df)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量(VOL)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - volume: 成交量
                
        Returns:
            添加了VOL指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['volume']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 添加原始成交量
        df_copy['vol'] = df_copy['volume']
        
        # 计算成交量移动平均
        df_copy['vol_ma5'] = df_copy['volume'].rolling(window=5).mean()
        df_copy['vol_ma10'] = df_copy['volume'].rolling(window=10).mean()
        df_copy['vol_ma20'] = df_copy['volume'].rolling(window=20).mean()
        
        # 计算相对成交量（当前成交量与N日平均成交量的比值）
        df_copy['vol_ratio'] = df_copy['volume'] / df_copy['vol_ma5']
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成成交量(VOL)指标交易信号
        
        Args:
            df: 包含价格数据和VOL指标的DataFrame
            **kwargs: 额外参数
                vol_ratio_threshold: 相对成交量阈值，默认为1.5
                
        Returns:
            添加了信号列的DataFrame:
            - vol_signal: 1=放量信号, -1=缩量信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['vol', 'vol_ma5']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 获取参数
        vol_ratio_threshold = kwargs.get('vol_ratio_threshold', 1.5)  # 相对成交量阈值
        
        # 生成信号
        df_copy['vol_signal'] = 0
        
        # 放量信号（成交量大于N日平均的1.5倍）
        df_copy.loc[df_copy['vol_ratio'] > vol_ratio_threshold, 'vol_signal'] = 1
        
        # 缩量信号（成交量小于N日平均的0.5倍）
        df_copy.loc[df_copy['vol_ratio'] < 0.5, 'vol_signal'] = -1
        
        return df_copy
    
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
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制成交量(VOL)指标图表
        
        Args:
            df: 包含VOL指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['vol', 'vol_ma5', 'vol_ma10']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制VOL指标线
        ax.bar(df.index, df['vol'], label='成交量', alpha=0.3, color='gray')
        ax.plot(df.index, df['vol_ma5'], label='5日均量', color='red')
        ax.plot(df.index, df['vol_ma10'], label='10日均量', color='blue')
        ax.plot(df.index, df['vol_ma20'], label='20日均量', color='green')
        
        ax.set_ylabel('成交量')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

