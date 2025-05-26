#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
顺势指标(CCI)

判断价格偏离度，寻找短线机会
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class CCI(BaseIndicator):
    """
    顺势指标(CCI) (CCI)
    
    分类：震荡类指标
    描述：判断价格偏离度，寻找短线机会
    """
    
    def __init__(self, period: int = 14):
        """
        初始化顺势指标(CCI)指标
        
        Args:
            period: 计算周期，默认为14
        """
        super().__init__()
        self.period = period
        self.name = "CCI"
    
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
        计算顺势指标(CCI)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了CCI指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算典型价格TP (Typical Price)
        df_copy['TP'] = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
        
        # 计算简单移动平均MA
        df_copy['MA'] = df_copy['TP'].rolling(window=self.period).mean()
        
        # 计算偏差绝对值
        df_copy['MD'] = df_copy['TP'].rolling(window=self.period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        # 计算CCI指标
        df_copy['CCI'] = (df_copy['TP'] - df_copy['MA']) / (0.015 * df_copy['MD'])
        
        # 删除中间计算列
        df_copy.drop(['TP', 'MA', 'MD'], axis=1, inplace=True)
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成顺势指标(CCI)指标交易信号
        
        Args:
            df: 包含价格数据和CCI指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值
                oversold: 超卖阈值
                
        Returns:
            添加了信号列的DataFrame:
            - cci_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['CCI']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 获取参数
        overbought = kwargs.get('overbought', 100)  # 超买阈值
        oversold = kwargs.get('oversold', -100)  # 超卖阈值
        
        # 初始化信号列
        df_copy['cci_signal'] = 0
        
        # CCI由超卖区上穿-100为买入信号
        df_copy.loc[crossover(df_copy['CCI'], oversold), 'cci_signal'] = 1
        
        # CCI由超买区下穿+100为卖出信号
        df_copy.loc[crossunder(df_copy['CCI'], overbought), 'cci_signal'] = -1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制顺势指标(CCI)指标图表
        
        Args:
            df: 包含CCI指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['CCI']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制CCI指标线
        ax.plot(df.index, df['CCI'], label='顺势指标(CCI)')
        
        # 添加超买超卖参考线
        overbought = kwargs.get('overbought', 100)
        oversold = kwargs.get('oversold', -100)
        ax.axhline(y=overbought, color='r', linestyle='--', alpha=0.3, label='超买线')
        ax.axhline(y=oversold, color='g', linestyle='--', alpha=0.3, label='超卖线')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        ax.set_ylabel('顺势指标(CCI)')
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

