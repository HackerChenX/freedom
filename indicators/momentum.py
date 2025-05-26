#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动量指标(Momentum)

衡量价格变化速度，预测趋势转折点
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class Momentum(BaseIndicator):
    """
    动量指标(Momentum) (MTM)
    
    分类：动量类指标
    描述：衡量价格变化速度，预测趋势转折点
    """
    
    def __init__(self, period: int = 10, signal_period: int = 6, calculation_method: str = "difference"):
        """
        初始化动量指标(Momentum)
        
        Args:
            period: 计算周期，默认为10
            signal_period: 信号线平滑周期，默认为6
            calculation_method: 计算方法，可选"difference"(差值法)或"ratio"(比率法)
        """
        super().__init__()
        self.period = period
        self.signal_period = signal_period
        self.calculation_method = calculation_method
        self.name = "Momentum"
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 包含价格数据的DataFrame
            required_columns: 所需的列名列表
        
        Raises:
            ValueError: 如果DataFrame不包含所需的列，或者行数少于所需的最小行数
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
        
        min_rows = self.period + self.signal_period
        if len(df) < min_rows:
            raise ValueError(f"DataFrame至少需要 {min_rows} 行数据，但只有 {len(df)} 行")
    
    def calculate(self, df: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        计算动量指标
        
        Args:
            df: 包含价格数据的DataFrame
            price_column: 用于计算动量的价格列名，默认为'close'
        
        Returns:
            包含动量指标结果的DataFrame
        """
        required_columns = [price_column]
        self._validate_dataframe(df, required_columns)
        
        result = pd.DataFrame(index=df.index)
        
        # 计算动量
        if self.calculation_method == "difference":
            # 差值法: 当前价格 - N期前价格
            result['mtm'] = df[price_column] - df[price_column].shift(self.period)
        else:
            # 比率法: (当前价格 / N期前价格) * 100
            result['mtm'] = (df[price_column] / df[price_column].shift(self.period)) * 100
        
        # 计算信号线(MTM的移动平均)
        result['signal'] = result['mtm'].rolling(window=self.signal_period).mean()
        
        return result
    
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
        result['mtm_buy_signal'] = 0  # 添加与指标名称相关的信号列
        result['mtm_sell_signal'] = 0  # 添加与指标名称相关的信号列
        
        # MTM上穿信号线买入
        buy_cross = crossover(result['mtm'].values, result['signal'].values)
        result.loc[buy_cross, 'buy_signal'] = 1
        result.loc[buy_cross, 'mtm_buy_signal'] = 1
        
        # MTM下穿信号线卖出
        sell_cross = crossunder(result['mtm'].values, result['signal'].values)
        result.loc[sell_cross, 'sell_signal'] = 1
        result.loc[sell_cross, 'mtm_sell_signal'] = 1
        
        return result
    
    def plot(self, df: pd.DataFrame, result: pd.DataFrame, ax=None):
        """
        绘制动量指标图表
        
        Args:
            df: 原始数据DataFrame
            result: 计算指标后的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
        
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 7))
        
        # 绘制MTM和信号线
        ax.plot(result['mtm'], label=f'MTM({self.period})')
        ax.plot(result['signal'], label=f'Signal({self.signal_period})')
        
        # 绘制零线
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # 标记买卖信号
        buy_signals = result[result['buy_signal'] == 1].index
        sell_signals = result[result['sell_signal'] == 1].index
        
        ax.scatter(buy_signals, result.loc[buy_signals, 'mtm'], color='green', marker='^', s=100, label='买入信号')
        ax.scatter(sell_signals, result.loc[sell_signals, 'mtm'], color='red', marker='v', s=100, label='卖出信号')
        
        ax.set_title(f'动量指标(MTM) - 周期:{self.period}')
        ax.set_ylabel('动量值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def compute(self, df: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        计算动量指标并生成交易信号
        
        Args:
            df: 包含价格数据的DataFrame
            price_column: 用于计算动量的价格列名，默认为'close'
        
        Returns:
            包含动量指标和交易信号的DataFrame
        """
        try:
            result = self.calculate(df, price_column)
            result = self.generate_signals(df, result)
            self._result = result
            return result
        except Exception as e:
            self._error = str(e)
            logger.error(f"计算指标 {self.name} 时出错: {str(e)}")
            raise 