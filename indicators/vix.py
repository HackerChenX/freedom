#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VIX恐慌指数指标

通过价格波动幅度衡量市场恐慌程度
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class VIX(BaseIndicator):
    """
    VIX恐慌指数指标
    
    分类：波动类指标
    描述：通过价格波动幅度衡量市场恐慌程度
    """
    
    def __init__(self, period: int = 10, smooth_period: int = 5):
        """
        初始化VIX恐慌指数指标
        
        Args:
            period: 计算周期，默认为10
            smooth_period: 平滑周期，默认为5
        """
        super().__init__()
        self.period = period
        self.smooth_period = smooth_period
        self.name = "VIX"
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算VIX指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含VIX指标的DataFrame
        """
        return self.calculate(df)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算VIX恐慌指数指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                
        Returns:
            添加了VIX指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['high', 'low', 'close']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算日内波动率：(high-low)/close
        df_copy['daily_range'] = (df_copy['high'] - df_copy['low']) / df_copy['close'] * 100
        
        # 计算N日平均波动率
        df_copy['vix'] = df_copy['daily_range'].rolling(window=self.period).mean()
        
        # 计算平滑后的VIX
        df_copy['vix_smooth'] = df_copy['vix'].rolling(window=self.smooth_period).mean()
        
        return df_copy
        
    def generate_signals(self, df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 原始数据DataFrame
            result: 计算指标后的DataFrame
        
        Returns:
            添加了交易信号的DataFrame
        """
        # 初始化信号列
        result['buy_signal'] = 0
        result['sell_signal'] = 0
        result['vix_buy_signal'] = 0  # 添加与指标名称相关的信号列
        result['vix_sell_signal'] = 0  # 添加与指标名称相关的信号列
        
        # 提取指标数据
        vix = result['vix'].values
        vix_smooth = result['vix_smooth'].values
        
        # VIX见顶回落买入信号
        for i in range(2, len(vix)):
            if vix[i-2] < vix[i-1] and vix[i] < vix[i-1]:
                result.iloc[i, result.columns.get_loc('buy_signal')] = 1
                result.iloc[i, result.columns.get_loc('vix_buy_signal')] = 1
        
        # VIX处于低位的买入信号
        vix_avg = result['vix'].rolling(window=20).mean()
        for i in range(20, len(vix)):
            if vix[i] < vix_avg[i] * 0.7:  # VIX低于20日均值的70%
                result.iloc[i, result.columns.get_loc('buy_signal')] = 1
                result.iloc[i, result.columns.get_loc('vix_buy_signal')] = 1
        
        # VIX急剧上升的卖出信号
        for i in range(1, len(vix)):
            if vix[i] > vix[i-1] * 1.5:  # VIX上升超过50%
                result.iloc[i, result.columns.get_loc('sell_signal')] = 1
                result.iloc[i, result.columns.get_loc('vix_sell_signal')] = 1
        
        return result
    
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

