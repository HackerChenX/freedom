"""
斐波那契工具模块

实现斐波那契回调线、扩展线和时间序列等工具
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class FibonacciType(Enum):
    """斐波那契工具类型枚举"""
    RETRACEMENT = "回调线"     # 下跌后回调支撑位
    EXTENSION = "扩展线"       # 上涨后目标位
    TIME_SERIES = "时间序列"   # 时间周期推演


class FibonacciTools(BaseIndicator):
    """
    斐波那契工具指标
    
    计算斐波那契回调线、扩展线和时间序列
    """
    
    # 斐波那契回调线比例
    RETRACEMENT_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    # 斐波那契扩展线比例
    EXTENSION_LEVELS = [0, 0.618, 1.0, 1.618, 2.618, 4.236]
    
    # 斐波那契时间序列
    TIME_SERIES = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    def __init__(self):
        """初始化斐波那契工具指标"""
        super().__init__(name="FibonacciTools", description="斐波那契工具指标，计算回调线、扩展线和时间序列")
    
    def calculate(self, data: pd.DataFrame, swing_high_idx: int = None, swing_low_idx: int = None, 
                fib_type: FibonacciType = FibonacciType.RETRACEMENT, *args, **kwargs) -> pd.DataFrame:
        """
        计算斐波那契工具
        
        Args:
            data: 输入数据，包含OHLC数据
            swing_high_idx: 摆动高点索引，如果为None则自动检测
            swing_low_idx: 摆动低点索引，如果为None则自动检测
            fib_type: 斐波那契工具类型，默认为回调线
            
        Returns:
            pd.DataFrame: 计算结果，包含斐波那契水平线
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["high", "low", "close"])
        
        # 如果未指定摆动点，则自动检测
        if swing_high_idx is None or swing_low_idx is None:
            swing_high_idx, swing_low_idx = self._detect_swing_points(data)
        
        # 根据工具类型计算斐波那契水平线
        if fib_type == FibonacciType.RETRACEMENT:
            return self.calculate_retracement(data, swing_high_idx, swing_low_idx)
        elif fib_type == FibonacciType.EXTENSION:
            return self.calculate_extension(data, swing_high_idx, swing_low_idx)
        elif fib_type == FibonacciType.TIME_SERIES:
            return self.calculate_time_series(data, swing_low_idx)
        else:
            logger.warning(f"不支持的斐波那契工具类型: {fib_type}")
            return pd.DataFrame(index=data.index)
    
    def calculate_retracement(self, data: pd.DataFrame, swing_high_idx: int, swing_low_idx: int) -> pd.DataFrame:
        """
        计算斐波那契回调线
        
        Args:
            data: 输入数据
            swing_high_idx: 摆动高点索引
            swing_low_idx: 摆动低点索引
            
        Returns:
            pd.DataFrame: 回调线水平
        """
        # 确保高点和低点正确
        if swing_high_idx > swing_low_idx:
            # 下跌后的回调：从高点到低点
            high_price = data["high"].iloc[swing_high_idx]
            low_price = data["low"].iloc[swing_low_idx]
            price_range = high_price - low_price
        else:
            # 上涨后的回调：从低点到高点
            high_price = data["high"].iloc[swing_low_idx]
            low_price = data["low"].iloc[swing_high_idx]
            price_range = high_price - low_price
            # 交换索引，确保高点在前，低点在后
            swing_high_idx, swing_low_idx = swing_low_idx, swing_high_idx
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 记录起止点
        result["swing_high"] = np.nan
        result["swing_low"] = np.nan
        result.iloc[swing_high_idx, result.columns.get_loc("swing_high")] = high_price
        result.iloc[swing_low_idx, result.columns.get_loc("swing_low")] = low_price
        
        # 计算回调线
        for level in self.RETRACEMENT_LEVELS:
            level_name = f"fib_retracement_{level:.3f}".replace(".", "_")
            level_price = low_price + price_range * level
            result[level_name] = level_price
        
        return result
    
    def calculate_extension(self, data: pd.DataFrame, swing_high_idx: int, swing_low_idx: int) -> pd.DataFrame:
        """
        计算斐波那契扩展线
        
        Args:
            data: 输入数据
            swing_high_idx: 摆动高点索引
            swing_low_idx: 摆动低点索引
            
        Returns:
            pd.DataFrame: 扩展线水平
        """
        # 确保点的顺序正确：先高点，后低点，用于看跌扩展
        if swing_high_idx < swing_low_idx:
            # 先低点，后高点，用于看涨扩展
            high_price = data["high"].iloc[swing_low_idx]
            low_price = data["low"].iloc[swing_high_idx]
            price_range = high_price - low_price
            is_bullish = True
        else:
            # 先高点，后低点，用于看跌扩展
            high_price = data["high"].iloc[swing_high_idx]
            low_price = data["low"].iloc[swing_low_idx]
            price_range = high_price - low_price
            is_bullish = False
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 记录起止点
        result["swing_high"] = np.nan
        result["swing_low"] = np.nan
        result.iloc[swing_high_idx, result.columns.get_loc("swing_high" if not is_bullish else "swing_low")] = high_price if not is_bullish else low_price
        result.iloc[swing_low_idx, result.columns.get_loc("swing_low" if not is_bullish else "swing_high")] = low_price if not is_bullish else high_price
        
        # 计算扩展线
        for level in self.EXTENSION_LEVELS:
            level_name = f"fib_extension_{level:.3f}".replace(".", "_")
            if is_bullish:
                # 看涨扩展：从低点开始向上扩展
                level_price = high_price + price_range * level
            else:
                # 看跌扩展：从低点开始向下扩展
                level_price = low_price - price_range * level
            result[level_name] = level_price
        
        return result
    
    def calculate_time_series(self, data: pd.DataFrame, start_idx: int) -> pd.DataFrame:
        """
        计算斐波那契时间序列
        
        Args:
            data: 输入数据
            start_idx: 起始点索引
            
        Returns:
            pd.DataFrame: 时间序列点
        """
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 记录起始点
        result["fib_time_start"] = np.nan
        result.iloc[start_idx, result.columns.get_loc("fib_time_start")] = data["close"].iloc[start_idx]
        
        # 计算时间序列点
        for level in self.TIME_SERIES:
            level_name = f"fib_time_{level}"
            target_idx = start_idx + level
            if target_idx < len(data):
                result[level_name] = np.nan
                result.iloc[target_idx, result.columns.get_loc(level_name)] = data["close"].iloc[target_idx]
        
        return result
    
    def _detect_swing_points(self, data: pd.DataFrame, window: int = 10) -> Tuple[int, int]:
        """
        自动检测摆动高点和低点
        
        Args:
            data: 输入数据
            window: 检测窗口
            
        Returns:
            Tuple[int, int]: (高点索引, 低点索引)
        """
        n = len(data)
        if n < 2 * window:
            logger.warning(f"数据长度不足，无法检测摆动点: {n} < {2*window}")
            return 0, n-1
        
        # 在最近的窗口内寻找最高点和最低点
        recent_data = data.iloc[-2*window:]
        high_idx = recent_data["high"].idxmax()
        low_idx = recent_data["low"].idxmin()
        
        # 将索引转换为相对于原始数据的位置
        high_pos = data.index.get_loc(high_idx)
        low_pos = data.index.get_loc(low_idx)
        
        return high_pos, low_pos
    
    def plot_fibonacci_levels(self, data: pd.DataFrame, result: pd.DataFrame, 
                            fib_type: FibonacciType = FibonacciType.RETRACEMENT,
                            ax=None):
        """
        绘制斐波那契水平线
        
        Args:
            data: 输入数据
            result: 斐波那契计算结果
            fib_type: 斐波那契工具类型
            ax: matplotlib轴对象
            
        Returns:
            matplotlib.axes.Axes: 轴对象
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制K线图
            ax.plot(data.index, data["close"], 'k-', linewidth=1)
            
            # 绘制高点和低点
            if "swing_high" in result.columns and "swing_low" in result.columns:
                high_idx = result["swing_high"].first_valid_index()
                low_idx = result["swing_low"].first_valid_index()
                
                if high_idx is not None and low_idx is not None:
                    ax.scatter(high_idx, result.loc[high_idx, "swing_high"], 
                             marker='^', color='g', s=100, label='Swing High')
                    ax.scatter(low_idx, result.loc[low_idx, "swing_low"], 
                             marker='v', color='r', s=100, label='Swing Low')
            
            # 绘制斐波那契水平线
            if fib_type == FibonacciType.RETRACEMENT:
                levels = self.RETRACEMENT_LEVELS
                prefix = "fib_retracement_"
                colors = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'black']
            elif fib_type == FibonacciType.EXTENSION:
                levels = self.EXTENSION_LEVELS
                prefix = "fib_extension_"
                colors = ['red', 'orange', 'gold', 'green', 'blue', 'purple']
            else:
                # 时间序列不绘制水平线
                return ax
            
            for i, level in enumerate(levels):
                level_name = f"{prefix}{level:.3f}".replace(".", "_")
                if level_name in result.columns:
                    level_value = result[level_name].iloc[0]
                    ax.axhline(y=level_value, color=colors[i % len(colors)], 
                             linestyle='--', alpha=0.7,
                             label=f"{level:.3f} - {level_value:.2f}")
            
            # 设置图例
            ax.legend(loc='best')
            
            # 设置标题
            ax.set_title(f'Fibonacci {fib_type.value}')
            
            # 设置日期格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            return ax
        
        except ImportError:
            logger.error("绘图需要安装matplotlib库")
            return None 