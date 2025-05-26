"""
江恩理论工具模块

实现江恩角度线、江恩方格和时间周期工具
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class GannAngle(Enum):
    """江恩角度枚举"""
    ANGLE_1X8 = "1×8角度线"      # 1×8角度线，约为82.5度
    ANGLE_1X4 = "1×4角度线"      # 1×4角度线，约为75度
    ANGLE_1X3 = "1×3角度线"      # 1×3角度线，约为71.6度
    ANGLE_1X2 = "1×2角度线"      # 1×2角度线，约为63.4度
    ANGLE_1X1 = "1×1角度线"      # 1×1角度线，约为45度
    ANGLE_2X1 = "2×1角度线"      # 2×1角度线，约为26.6度
    ANGLE_3X1 = "3×1角度线"      # 3×1角度线，约为18.4度
    ANGLE_4X1 = "4×1角度线"      # 4×1角度线，约为15度
    ANGLE_8X1 = "8×1角度线"      # 8×1角度线，约为7.5度


class GannTimeCycle(Enum):
    """江恩时间周期枚举"""
    CYCLE_30 = "30日周期"
    CYCLE_45 = "45日周期"
    CYCLE_60 = "60日周期"
    CYCLE_90 = "90日周期"
    CYCLE_120 = "120日周期"
    CYCLE_144 = "144日周期"
    CYCLE_180 = "180日周期"
    CYCLE_270 = "270日周期"
    CYCLE_360 = "360日周期"


class GannTools(BaseIndicator):
    """
    江恩理论工具指标
    
    计算江恩角度线、江恩方格和时间周期
    """
    
    # 江恩角度线比例
    ANGLE_RATIOS = {
        GannAngle.ANGLE_1X8: (1, 8),
        GannAngle.ANGLE_1X4: (1, 4),
        GannAngle.ANGLE_1X3: (1, 3),
        GannAngle.ANGLE_1X2: (1, 2),
        GannAngle.ANGLE_1X1: (1, 1),
        GannAngle.ANGLE_2X1: (2, 1),
        GannAngle.ANGLE_3X1: (3, 1),
        GannAngle.ANGLE_4X1: (4, 1),
        GannAngle.ANGLE_8X1: (8, 1)
    }
    
    # 江恩时间周期
    TIME_CYCLES = {
        GannTimeCycle.CYCLE_30: 30,
        GannTimeCycle.CYCLE_45: 45,
        GannTimeCycle.CYCLE_60: 60,
        GannTimeCycle.CYCLE_90: 90,
        GannTimeCycle.CYCLE_120: 120,
        GannTimeCycle.CYCLE_144: 144,
        GannTimeCycle.CYCLE_180: 180,
        GannTimeCycle.CYCLE_270: 270,
        GannTimeCycle.CYCLE_360: 360
    }
    
    def __init__(self):
        """初始化江恩理论工具指标"""
        super().__init__(name="GannTools", description="江恩理论工具指标，计算角度线和时间周期")
    
    def calculate(self, data: pd.DataFrame, pivot_idx: int = None, price_unit: float = None, 
                time_unit: int = 1, *args, **kwargs) -> pd.DataFrame:
        """
        计算江恩理论工具
        
        Args:
            data: 输入数据，包含OHLC数据
            pivot_idx: 支点索引，如果为None则使用第一个点
            price_unit: 价格单位，如果为None则自动计算
            time_unit: 时间单位，默认为1天
            
        Returns:
            pd.DataFrame: 计算结果，包含江恩角度线和时间周期
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 如果未指定支点，则使用第一个点
        if pivot_idx is None:
            pivot_idx = 0
        
        # 获取支点价格
        pivot_price = data["close"].iloc[pivot_idx]
        
        # 如果未指定价格单位，则自动计算
        if price_unit is None:
            # 使用价格的0.1%作为基本单位
            price_unit = pivot_price * 0.001
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算角度线
        result = self._calculate_angle_lines(data, result, pivot_idx, pivot_price, price_unit, time_unit)
        
        # 计算时间周期
        result = self._calculate_time_cycles(data, result, pivot_idx)
        
        return result
    
    def _calculate_angle_lines(self, data: pd.DataFrame, result: pd.DataFrame, 
                             pivot_idx: int, pivot_price: float, 
                             price_unit: float, time_unit: int) -> pd.DataFrame:
        """
        计算江恩角度线
        
        Args:
            data: 输入数据
            result: 结果数据框
            pivot_idx: 支点索引
            pivot_price: 支点价格
            price_unit: 价格单位
            time_unit: 时间单位
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        n = len(data)
        
        # 初始化角度线列
        for angle in GannAngle:
            result[angle.value] = np.nan
        
        # 计算每个角度线
        for angle, (price_ratio, time_ratio) in self.ANGLE_RATIOS.items():
            # 计算角度线的斜率
            slope = (price_ratio * price_unit) / (time_ratio * time_unit)
            
            # 计算角度线上的点
            for i in range(n):
                # 计算与支点的时间差
                time_diff = i - pivot_idx
                
                # 计算角度线上的价格
                if time_diff >= 0:
                    # 向上角度线
                    angle_price = pivot_price + slope * time_diff
                else:
                    # 向下角度线
                    angle_price = pivot_price - slope * abs(time_diff)
                
                # 添加到结果
                result.iloc[i, result.columns.get_loc(angle.value)] = angle_price
        
        return result
    
    def _calculate_time_cycles(self, data: pd.DataFrame, result: pd.DataFrame, pivot_idx: int) -> pd.DataFrame:
        """
        计算江恩时间周期
        
        Args:
            data: 输入数据
            result: 结果数据框
            pivot_idx: 支点索引
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        n = len(data)
        
        # 初始化时间周期列
        result["time_cycle"] = np.nan
        
        # 计算每个时间周期
        for cycle, days in self.TIME_CYCLES.items():
            cycle_name = f"cycle_{days}"
            result[cycle_name] = np.nan
            
            # 计算周期点
            for i in range(days, n, days):
                target_idx = pivot_idx + i
                if target_idx < n:
                    result.iloc[target_idx, result.columns.get_loc(cycle_name)] = data["close"].iloc[target_idx]
                    result.iloc[target_idx, result.columns.get_loc("time_cycle")] = days
        
        return result
    
    def calculate_gann_square(self, data: pd.DataFrame, pivot_idx: int = None, 
                            levels: int = 9, *args, **kwargs) -> pd.DataFrame:
        """
        计算江恩方格
        
        Args:
            data: 输入数据
            pivot_idx: 支点索引，如果为None则使用第一个点
            levels: 方格级别数，默认为9
            
        Returns:
            pd.DataFrame: 江恩方格计算结果
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 如果未指定支点，则使用第一个点
        if pivot_idx is None:
            pivot_idx = 0
        
        # 获取支点价格
        pivot_price = data["close"].iloc[pivot_idx]
        
        # 计算方格价格单位
        # 使用价格的平方根作为基本单位
        price_unit = np.sqrt(pivot_price)
        
        # 初始化结果数据框
        result = pd.DataFrame(index=pd.RangeIndex(2*levels+1))
        result["level"] = np.arange(-levels, levels+1)
        
        # 计算价格和时间方格
        result["price"] = pivot_price + result["level"] * price_unit
        result["time_factor"] = result["level"].apply(lambda x: abs(x) if x != 0 else 1)
        
        return result
    
    def plot_gann_angles(self, data: pd.DataFrame, result: pd.DataFrame, 
                       angles: List[GannAngle] = None, ax=None):
        """
        绘制江恩角度线
        
        Args:
            data: 输入数据
            result: 角度线计算结果
            angles: 要绘制的角度线列表，如果为None则绘制所有角度线
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
            
            # 确定要绘制的角度线
            if angles is None:
                angles = list(GannAngle)
            
            # 绘制角度线
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'cyan', 'magenta', 'brown']
            for i, angle in enumerate(angles):
                angle_name = angle.value
                if angle_name in result.columns:
                    ax.plot(data.index, result[angle_name], 
                          linestyle='--', color=colors[i % len(colors)], 
                          alpha=0.7, label=angle_name)
            
            # 设置图例
            ax.legend(loc='best')
            
            # 设置标题
            ax.set_title('Gann Angle Lines')
            
            # 设置日期格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            return ax
        
        except ImportError:
            logger.error("绘图需要安装matplotlib库")
            return None
    
    def plot_gann_time_cycles(self, data: pd.DataFrame, result: pd.DataFrame, ax=None):
        """
        绘制江恩时间周期
        
        Args:
            data: 输入数据
            result: 时间周期计算结果
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
            
            # 获取所有周期点
            cycle_points = result[~result["time_cycle"].isna()]
            
            # 绘制周期点
            for idx, row in cycle_points.iterrows():
                cycle = int(row["time_cycle"])
                price = data.loc[idx, "close"]
                
                # 设置不同周期的颜色和标记
                if cycle <= 45:
                    color = 'blue'
                    marker = 'o'
                elif cycle <= 90:
                    color = 'green'
                    marker = 's'
                elif cycle <= 180:
                    color = 'orange'
                    marker = '^'
                else:
                    color = 'red'
                    marker = '*'
                
                ax.scatter(idx, price, color=color, marker=marker, s=80, 
                         label=f'{cycle}日周期' if f'{cycle}日周期' not in plt.gca().get_legend_handles_labels()[1] else "")
                
                # 添加垂直线
                ax.axvline(x=idx, color=color, linestyle='--', alpha=0.3)
            
            # 设置图例（去除重复项）
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best')
            
            # 设置标题
            ax.set_title('Gann Time Cycles')
            
            # 设置日期格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            return ax
        
        except ImportError:
            logger.error("绘图需要安装matplotlib库")
            return None
    
    def plot_gann_square(self, gann_square: pd.DataFrame, ax=None):
        """
        绘制江恩方格
        
        Args:
            gann_square: 江恩方格计算结果
            ax: matplotlib轴对象
            
        Returns:
            matplotlib.axes.Axes: 轴对象
        """
        try:
            import matplotlib.pyplot as plt
            
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 8))
            
            # 获取价格范围
            min_price = gann_square["price"].min()
            max_price = gann_square["price"].max()
            
            # 绘制水平线（价格方格）
            for _, row in gann_square.iterrows():
                price = row["price"]
                ax.axhline(y=price, color='blue', linestyle='-', alpha=0.3)
                ax.text(-0.5, price, f'{price:.2f}', verticalalignment='center')
            
            # 绘制垂直线（时间方格）
            levels = (len(gann_square) - 1) // 2
            for i in range(-levels, levels+1):
                ax.axvline(x=i, color='red', linestyle='-', alpha=0.3)
                ax.text(i, min_price - (max_price - min_price) * 0.05, str(i), 
                      horizontalalignment='center')
            
            # 绘制对角线（角度线）
            ax.plot([-levels, levels], [min_price, max_price], 'g-', linewidth=2, label='1×1线')
            
            # 设置坐标轴范围
            ax.set_xlim(-levels-1, levels+1)
            ax.set_ylim(min_price - (max_price - min_price) * 0.1, 
                      max_price + (max_price - min_price) * 0.1)
            
            # 设置图例和标题
            ax.legend(loc='best')
            ax.set_title('Gann Square')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            
            # 设置网格
            ax.grid(True, alpha=0.3)
            
            return ax
        
        except ImportError:
            logger.error("绘图需要安装matplotlib库")
            return None 