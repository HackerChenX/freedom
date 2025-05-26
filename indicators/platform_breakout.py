"""
平台突破指标模块

实现平台整理后突破的识别功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class BreakoutDirection(Enum):
    """突破方向枚举"""
    UP = "向上突破"      # 向上突破
    DOWN = "向下突破"    # 向下突破
    NONE = "无突破"      # 无突破


class PlatformBreakout(BaseIndicator):
    """
    平台突破指标
    
    识别价格在一定区间整理后的突破行为
    """
    
    def __init__(self, platform_period: int = 20, max_volatility: float = 0.05):
        """
        初始化平台突破指标
        
        Args:
            platform_period: 平台检测周期，默认为20天
            max_volatility: 平台最大波动率，默认为5%
        """
        super().__init__(name="PlatformBreakout", description="平台突破指标，识别价格在一定区间整理后的突破行为")
        self.platform_period = platform_period
        self.max_volatility = max_volatility
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算平台突破指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含平台和突破标记
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low", "close", "volume"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算平台特征
        result = self._detect_platforms(data, result)
        
        # 计算突破特征
        result = self._detect_breakouts(data, result)
        
        return result
    
    def _detect_platforms(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        检测平台
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 价格数据
        close_prices = data["close"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        
        # 初始化平台标记
        is_platform = np.zeros(len(data), dtype=bool)
        
        # 平台检测
        for i in range(self.platform_period, len(data)):
            # 计算平台区间
            period_high = np.max(high_prices[i-self.platform_period:i])
            period_low = np.min(low_prices[i-self.platform_period:i])
            
            # 计算波动率
            platform_volatility = (period_high - period_low) / period_low
            
            # 判断是否为平台
            is_platform[i] = platform_volatility <= self.max_volatility
        
        # 添加到结果
        result["is_platform"] = is_platform
        
        # 计算平台上下边界
        result["platform_upper"] = np.nan
        result["platform_lower"] = np.nan
        
        for i in range(self.platform_period, len(data)):
            if result["is_platform"].iloc[i]:
                period_high = np.max(high_prices[i-self.platform_period:i])
                period_low = np.min(low_prices[i-self.platform_period:i])
                
                result.iloc[i, result.columns.get_loc("platform_upper")] = period_high
                result.iloc[i, result.columns.get_loc("platform_lower")] = period_low
        
        # 计算平台持续天数
        result["platform_days"] = 0
        
        for i in range(1, len(data)):
            if result["is_platform"].iloc[i]:
                result.iloc[i, result.columns.get_loc("platform_days")] = result["platform_days"].iloc[i-1] + 1
        
        return result
    
    def _detect_breakouts(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        检测突破
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 价格数据
        close_prices = data["close"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        volumes = data["volume"].values
        
        # 初始化突破标记
        up_breakout = np.zeros(len(data), dtype=bool)
        down_breakout = np.zeros(len(data), dtype=bool)
        
        # 突破检测
        for i in range(self.platform_period + 1, len(data)):
            # 只在平台形成后检测突破
            if result["platform_days"].iloc[i-1] >= self.platform_period / 2:
                # 向上突破：收盘价突破平台上边界，且成交量放大
                if (close_prices[i] > result["platform_upper"].iloc[i-1] and 
                    volumes[i] > np.mean(volumes[i-self.platform_period:i])):
                    up_breakout[i] = True
                
                # 向下突破：收盘价跌破平台下边界，且成交量放大
                elif (close_prices[i] < result["platform_lower"].iloc[i-1] and 
                      volumes[i] > np.mean(volumes[i-self.platform_period:i])):
                    down_breakout[i] = True
        
        # 添加到结果
        result["up_breakout"] = up_breakout
        result["down_breakout"] = down_breakout
        
        # 突破方向
        breakout_direction = np.array([BreakoutDirection.NONE.value] * len(data), dtype=object)
        breakout_direction[up_breakout] = BreakoutDirection.UP.value
        breakout_direction[down_breakout] = BreakoutDirection.DOWN.value
        
        result["breakout_direction"] = breakout_direction
        
        # 计算突破强度
        result["breakout_strength"] = 0.0
        
        for i in range(self.platform_period + 1, len(data)):
            if up_breakout[i]:
                # 向上突破强度：(收盘价 - 平台上边界) / 平台上边界
                strength = (close_prices[i] - result["platform_upper"].iloc[i-1]) / result["platform_upper"].iloc[i-1]
                result.iloc[i, result.columns.get_loc("breakout_strength")] = strength
            
            elif down_breakout[i]:
                # 向下突破强度：(平台下边界 - 收盘价) / 平台下边界
                strength = (result["platform_lower"].iloc[i-1] - close_prices[i]) / result["platform_lower"].iloc[i-1]
                result.iloc[i, result.columns.get_loc("breakout_strength")] = strength
        
        return result
    
    def get_signals(self, data: pd.DataFrame, min_platform_days: int = 10, 
                  min_breakout_strength: float = 0.02) -> pd.DataFrame:
        """
        生成平台突破信号
        
        Args:
            data: 输入数据，包含平台突破指标
            min_platform_days: 最小平台天数，默认为10
            min_breakout_strength: 最小突破强度，默认为2%
            
        Returns:
            pd.DataFrame: 包含突破信号的数据框
        """
        if "breakout_direction" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["valid_up_breakout"] = False
        data["valid_down_breakout"] = False
        
        # 生成有效突破信号
        for i in range(len(data)):
            # 向上有效突破：平台天数充分 + 向上突破 + 突破强度充分
            if (data["platform_days"].iloc[i] >= min_platform_days and 
                data["breakout_direction"].iloc[i] == BreakoutDirection.UP.value and 
                data["breakout_strength"].iloc[i] >= min_breakout_strength):
                data.iloc[i, data.columns.get_loc("valid_up_breakout")] = True
            
            # 向下有效突破：平台天数充分 + 向下突破 + 突破强度充分
            elif (data["platform_days"].iloc[i] >= min_platform_days and 
                  data["breakout_direction"].iloc[i] == BreakoutDirection.DOWN.value and 
                  data["breakout_strength"].iloc[i] >= min_breakout_strength):
                data.iloc[i, data.columns.get_loc("valid_down_breakout")] = True
        
        return data 