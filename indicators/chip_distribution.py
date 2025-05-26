"""
筹码分布指标模块

实现股票筹码分布分析功能，包括筹码集中度、套牢盘、解套盘等分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple, Callable

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class ChipDistribution(BaseIndicator):
    """
    筹码分布指标
    
    分析股票各价位买入持仓情况，计算筹码集中度、套牢盘比例等
    """
    
    def __init__(self):
        """初始化筹码分布指标"""
        super().__init__(name="ChipDistribution", description="筹码分布指标，分析各价位持仓情况")
    
    def calculate(self, data: pd.DataFrame, half_life: int = 60, price_precision: float = 0.01, 
                 *args, **kwargs) -> pd.DataFrame:
        """
        计算筹码分布指标
        
        Args:
            data: 输入数据，包含OHLCV和换手率数据
            half_life: 半衰期，用于计算筹码衰减，默认为60天
            price_precision: 价格精度，用于设置价格区间，默认为0.01元
            
        Returns:
            pd.DataFrame: 计算结果，包含筹码分布相关指标
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low", "close", "volume"])
        
        # 检查是否有换手率数据，如果没有则计算估算值
        if "turnover_rate" not in data.columns:
            logger.warning("数据中缺少换手率数据，将使用估算值")
            if "total_share" in data.columns:
                data["turnover_rate"] = data["volume"] / data["total_share"] * 100
            else:
                # 使用成交量相对值估算换手率
                data["turnover_rate"] = data["volume"] / data["volume"].rolling(window=20).mean() * 5
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算价格网格
        min_price = data["low"].min() * 0.9  # 留有余量
        max_price = data["high"].max() * 1.1  # 留有余量
        price_grid = np.arange(min_price, max_price + price_precision, price_precision)
        n_prices = len(price_grid)
        
        # 初始化筹码分布矩阵
        n_days = len(data)
        chip_matrix = np.zeros((n_days, n_prices))
        
        # 计算衰减系数
        decay_factor = np.exp(np.log(0.5) / half_life)
        
        # 计算每日的筹码分布
        for i in range(n_days):
            # 当日交易产生的筹码分布
            day_chip = self._calculate_day_chip_distribution(
                data.iloc[i], price_grid, price_precision
            )
            
            if i == 0:
                chip_matrix[i] = day_chip
            else:
                # 历史筹码衰减
                decayed_chip = chip_matrix[i-1] * decay_factor
                
                # 加入当日新增筹码
                chip_matrix[i] = decayed_chip + day_chip
                
                # 归一化
                if np.sum(chip_matrix[i]) > 0:
                    chip_matrix[i] = chip_matrix[i] / np.sum(chip_matrix[i])
        
        # 计算筹码分布指标
        result = self._calculate_chip_metrics(data, result, chip_matrix, price_grid)
        
        return result
    
    def _calculate_day_chip_distribution(self, day_data: pd.Series, 
                                       price_grid: np.ndarray, 
                                       price_precision: float) -> np.ndarray:
        """
        计算单日的筹码分布
        
        Args:
            day_data: 单日数据
            price_grid: 价格网格
            price_precision: 价格精度
            
        Returns:
            np.ndarray: 单日筹码分布
        """
        # 提取数据
        open_price = day_data["open"]
        high_price = day_data["high"]
        low_price = day_data["low"]
        close_price = day_data["close"]
        turnover_rate = day_data["turnover_rate"] / 100  # 转换为小数
        
        # 初始化分布
        n_prices = len(price_grid)
        day_chip = np.zeros(n_prices)
        
        # 简化模型：假设成交量在最高价和最低价之间均匀分布
        # 实际应用中可以使用更复杂的模型，如TWAP或VWAP分布
        for i, price in enumerate(price_grid):
            if low_price <= price <= high_price:
                # 均匀分布模型
                day_chip[i] = turnover_rate / ((high_price - low_price) / price_precision + 1)
        
        # 归一化
        if np.sum(day_chip) > 0:
            day_chip = day_chip / np.sum(day_chip) * turnover_rate
        
        return day_chip
    
    def _calculate_chip_metrics(self, data: pd.DataFrame, result: pd.DataFrame, 
                              chip_matrix: np.ndarray, price_grid: np.ndarray) -> pd.DataFrame:
        """
        计算筹码分布相关指标
        
        Args:
            data: 原始数据
            result: 结果数据框
            chip_matrix: 筹码分布矩阵
            price_grid: 价格网格
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        n_days = len(data)
        close_prices = data["close"].values
        
        # 计算筹码集中度
        chip_concentration = np.zeros(n_days)
        
        # 计算获利盘比例
        profit_ratio = np.zeros(n_days)
        
        # 计算90%筹码区间宽度
        chip_width_90pct = np.zeros(n_days)
        
        # 计算套牢盘成本
        avg_cost = np.zeros(n_days)
        
        for i in range(n_days):
            # 当日收盘价
            close = close_prices[i]
            
            # 找到收盘价对应的价格网格索引
            close_idx = np.argmin(np.abs(price_grid - close))
            
            # 计算筹码集中度：90%筹码所在的价格区间占全部价格区间的比例
            sorted_chip = np.sort(chip_matrix[i])[::-1]  # 降序排列
            cumsum_chip = np.cumsum(sorted_chip)
            chip_90pct_idx = np.argmax(cumsum_chip >= 0.9)
            if chip_90pct_idx > 0:
                chip_concentration[i] = chip_90pct_idx / len(price_grid)
            
            # 计算获利盘比例：当前价格以下的筹码比例
            profit_ratio[i] = np.sum(chip_matrix[i][:close_idx+1])
            
            # 计算90%筹码区间宽度
            chip_cumsum = np.cumsum(chip_matrix[i])
            lower_idx = np.argmax(chip_cumsum >= 0.05)
            upper_idx = np.argmax(chip_cumsum >= 0.95)
            if upper_idx > lower_idx:
                chip_width_90pct[i] = (price_grid[upper_idx] - price_grid[lower_idx]) / close
            
            # 计算平均成本
            avg_cost[i] = np.sum(price_grid * chip_matrix[i]) / np.sum(chip_matrix[i]) if np.sum(chip_matrix[i]) > 0 else 0
        
        # 添加到结果
        result["chip_concentration"] = chip_concentration  # 筹码集中度
        result["profit_ratio"] = profit_ratio              # 获利盘比例
        result["chip_width_90pct"] = chip_width_90pct      # 90%筹码区间宽度
        result["avg_cost"] = avg_cost                      # 平均成本
        
        # 计算解套难度：当前价格与平均成本的比值
        result["untrapped_difficulty"] = close_prices / avg_cost
        
        # 计算筹码松散度：筹码集中度的倒数
        result["chip_looseness"] = 1 / (chip_concentration + 0.0001)  # 避免除以零
        
        # 计算筹码变动率：当日获利盘比例与前一日的差值
        result["profit_ratio_change"] = result["profit_ratio"].diff()
        
        return result
    
    def get_distribution_at_price(self, data: pd.DataFrame, 
                                price_point: float, 
                                half_life: int = 60,
                                price_precision: float = 0.01) -> float:
        """
        获取特定价格点的筹码密度
        
        Args:
            data: 输入数据
            price_point: 价格点
            half_life: 半衰期
            price_precision: 价格精度
            
        Returns:
            float: 价格点的筹码密度
        """
        # 计算筹码分布
        result = self.calculate(data, half_life, price_precision)
        
        # 找到最后一天的收盘价附近的筹码密度
        min_price = data["low"].min() * 0.9
        price_idx = int((price_point - min_price) / price_precision)
        
        # 安全检查
        if price_idx < 0:
            return 0.0
        
        # 计算所需的值（这里简化处理，实际应用需要访问chip_matrix）
        # 由于我们没有直接存储chip_matrix，这里使用估算方法
        
        # 获取最近的价格点
        last_close = data["close"].iloc[-1]
        last_avg_cost = result["avg_cost"].iloc[-1]
        
        # 估算密度：基于正态分布假设，价格越接近平均成本，密度越大
        density = np.exp(-0.5 * ((price_point - last_avg_cost) / (0.1 * last_avg_cost))**2)
        
        return density
    
    def plot_chip_distribution(self, data: pd.DataFrame, 
                             ax=None, 
                             half_life: int = 60,
                             price_precision: float = 0.01):
        """
        绘制筹码分布图
        
        Args:
            data: 输入数据
            ax: matplotlib轴对象
            half_life: 半衰期
            price_precision: 价格精度
            
        Returns:
            matplotlib.axes.Axes: 轴对象
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
            
            # 计算筹码分布指标
            result = self.calculate(data, half_life, price_precision)
            
            # 创建价格网格
            min_price = data["low"].min() * 0.9
            max_price = data["high"].max() * 1.1
            price_grid = np.arange(min_price, max_price + price_precision, price_precision)
            
            # 创建时间网格
            time_grid = np.arange(len(data))
            
            # 创建筹码密度矩阵（简化版本，实际应用需要使用完整的chip_matrix）
            density_matrix = np.zeros((len(time_grid), len(price_grid)))
            
            for i, idx in enumerate(time_grid):
                # 使用正态分布模拟筹码分布
                if idx < len(result):
                    avg_cost = result["avg_cost"].iloc[idx]
                    std = result["chip_width_90pct"].iloc[idx] * avg_cost / 3.29  # 3.29是90%置信区间的宽度
                    
                    for j, price in enumerate(price_grid):
                        density_matrix[i, j] = np.exp(-0.5 * ((price - avg_cost) / (std + 0.0001))**2)
            
            # 归一化
            for i in range(len(time_grid)):
                if np.sum(density_matrix[i]) > 0:
                    density_matrix[i] = density_matrix[i] / np.max(density_matrix[i])
            
            # 创建自定义colormap
            colors = [(0, 0, 1, 0), (0, 0, 1, 0.5), (1, 0, 0, 0.5), (1, 0, 0, 1)]
            cmap = LinearSegmentedColormap.from_list('chip_cmap', colors, N=100)
            
            # 绘制筹码分布热力图
            im = ax.imshow(
                density_matrix.T, 
                aspect='auto', 
                origin='lower',
                extent=[0, len(data), min_price, max_price],
                cmap=cmap,
                alpha=0.7
            )
            
            # 绘制K线图
            ax.plot(time_grid, data["close"].values, 'k-', linewidth=1.5)
            
            # 设置坐标轴
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.set_title('Chip Distribution')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, label='Density')
            
            return ax
        
        except ImportError:
            logger.error("绘图需要安装matplotlib库")
            return None 