#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
抛物线转向系统(SAR)

判断价格趋势反转信号，提供买卖点
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class SAR(BaseIndicator):
    """
    抛物线转向系统(SAR) (SAR)
    
    分类：趋势跟踪指标
    描述：判断价格趋势反转信号，提供买卖点
    """
    
    def __init__(self, acceleration: float = 0.02, maximum: float = 0.2):
        """
        初始化抛物线转向系统(SAR)指标
        
        Args:
            acceleration: 加速因子，默认为0.02
            maximum: 加速因子最大值，默认为0.2
        """
        super().__init__()
        self.acceleration = acceleration
        self.maximum = maximum
        self.name = "SAR"
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 数据帧
            required_columns: 所需的列名列表
        
        Raises:
            ValueError: 如果DataFrame不包含所需的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少所需的列: {missing_columns}")
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算抛物线转向系统(SAR)指标
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            包含SAR值的DataFrame
        """
        self._validate_dataframe(df, ['high', 'low', 'close'])
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        length = len(df)
        sar = np.zeros(length)
        trend = np.zeros(length)  # 1为上升趋势，-1为下降趋势
        ep = np.zeros(length)     # 极点价格
        af = np.zeros(length)     # 加速因子
        
        # 初始化
        trend[0] = 1  # 假设初始为上升趋势
        ep[0] = high[0]  # 极点价格初始为第一个最高价
        sar[0] = low[0]  # SAR初始为第一个最低价
        af[0] = self.acceleration
        
        # 计算SAR
        for i in range(1, length):
            # 上一个周期是上升趋势
            if trend[i-1] == 1:
                # 计算当前SAR值
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # 确保SAR不高于前两个周期的最低价
                if i >= 2:
                    sar[i] = min(sar[i], min(low[i-1], low[i-2]))
                
                # 判断趋势是否反转
                if low[i] < sar[i]:
                    # 趋势反转为下降
                    trend[i] = -1
                    sar[i] = ep[i-1]  # SAR值设为前期极点
                    ep[i] = low[i]     # 极点设为当前最低价
                    af[i] = self.acceleration  # 加速因子重置
                else:
                    # 继续上升趋势
                    trend[i] = 1
                    # 更新极点和加速因子
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + self.acceleration, self.maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            
            # 上一个周期是下降趋势
            else:
                # 计算当前SAR值
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # 确保SAR不低于前两个周期的最高价
                if i >= 2:
                    sar[i] = max(sar[i], max(high[i-1], high[i-2]))
                
                # 判断趋势是否反转
                if high[i] > sar[i]:
                    # 趋势反转为上升
                    trend[i] = 1
                    sar[i] = ep[i-1]  # SAR值设为前期极点
                    ep[i] = high[i]    # 极点设为当前最高价
                    af[i] = self.acceleration  # 加速因子重置
                else:
                    # 继续下降趋势
                    trend[i] = -1
                    # 更新极点和加速因子
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + self.acceleration, self.maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=df.index)
        result['sar'] = sar
        result['trend'] = trend
        result['ep'] = ep
        result['af'] = af
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据SAR生成买卖信号
        
        Args:
            df: 包含SAR值的DataFrame
            
        Returns:
            包含买卖信号的DataFrame
        """
        signals = pd.DataFrame(index=df.index)
        trend = df['trend'].values
        
        buy_signals = np.zeros(len(df))
        sell_signals = np.zeros(len(df))
        
        # 寻找趋势反转点作为信号
        for i in range(1, len(df)):
            # 趋势由下降转为上升，买入信号
            if trend[i] == 1 and trend[i-1] == -1:
                buy_signals[i] = 1
            
            # 趋势由上升转为下降，卖出信号
            if trend[i] == -1 and trend[i-1] == 1:
                sell_signals[i] = 1
        
        signals['buy'] = buy_signals
        signals['sell'] = sell_signals
        
        return signals
    
    def plot(self, df: pd.DataFrame, ax=None):
        """
        绘制SAR图表
        
        Args:
            df: 包含SAR值和OHLC数据的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制K线图
        ax.plot(df.index, df['close'], color='black', label='Close')
        
        # 绘制SAR点
        for i in range(len(df)):
            if df['trend'].iloc[i] == 1:  # 上升趋势
                ax.scatter(df.index[i], df['sar'].iloc[i], color='green', marker='v', s=20)
            else:  # 下降趋势
                ax.scatter(df.index[i], df['sar'].iloc[i], color='red', marker='^', s=20)
        
        # 添加买卖信号
        signals = self.generate_signals(df)
        buy_points = df.index[signals['buy'] == 1]
        sell_points = df.index[signals['sell'] == 1]
        
        if len(buy_points) > 0:
            buy_prices = df.loc[buy_points, 'close']
            ax.scatter(buy_points, buy_prices, color='green', marker='^', s=100, label='Buy')
        
        if len(sell_points) > 0:
            sell_prices = df.loc[sell_points, 'close']
            ax.scatter(sell_points, sell_prices, color='red', marker='v', s=100, label='Sell')
        
        ax.set_title(f'SAR (acceleration={self.acceleration}, maximum={self.maximum})')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算SAR指标
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            包含SAR值和信号的DataFrame
        """
        try:
            result = self.calculate(df)
            signals = self.generate_signals(result)
            # 合并结果
            result['buy_signal'] = signals['buy']
            result['sell_signal'] = signals['sell']
            
            return result
        except Exception as e:
            logger.error(f"计算指标 {self.name} 时出错: {str(e)}")
            raise

