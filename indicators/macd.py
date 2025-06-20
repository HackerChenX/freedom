"""
MACD指标模块

实现MACD指标的计算和相关功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import macd as calc_macd, cross
from enums.indicator_types import CrossType


class MACD(BaseIndicator):
    """
    MACD(Moving Average Convergence Divergence)指标
    
    MACD是一种趋势跟踪的动量指标，通过计算两条不同周期的指数移动平均线之差，
    以及该差值的移动平均线来判断市场趋势和动量。
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        初始化MACD指标
        
        Args:
            fast_period: 快线周期，默认为12
            slow_period: 慢线周期，默认为26
            signal_period: 信号线周期，默认为9
        """
        super().__init__(name="MACD", description="移动平均线收敛散度指标")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, data: pd.DataFrame, price_col: str = 'close', 
                  add_prefix: bool = False, **kwargs) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            price_col: 价格列名，默认为'close'
            add_prefix: 是否在输出列名前添加指标名称前缀
            kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含MACD指标的DataFrame
            
        Raises:
            ValueError: 如果输入数据不包含价格列
        """
        # 确保数据包含价格列
        self.ensure_columns(data, [price_col])
        
        # 复制输入数据
        result = data.copy()
        
        # 使用统一的公共函数计算MACD
        dif, dea, macd_hist = calc_macd(
            result[price_col].values,
            self.fast_period,
            self.slow_period,
            self.signal_period
        )
        
        # 确保前N个值为NaN，其中N = max(fast_period, slow_period) - 1
        min_periods = max(self.fast_period, self.slow_period) - 1
        dif[:min_periods] = np.nan
        dea[:min_periods + self.signal_period - 1] = np.nan
        macd_hist[:min_periods + self.signal_period - 1] = np.nan
        
        # 设置列名
        if add_prefix:
            macd_col = self.get_column_name('DIF')
            signal_col = self.get_column_name('DEA')
            hist_col = self.get_column_name('MACD')
        else:
            macd_col = 'DIF'
            signal_col = 'DEA'
            hist_col = 'MACD'
        
        # 添加结果列
        result[macd_col] = dif
        result[signal_col] = dea
        result[hist_col] = macd_hist
        
        # 添加信号
        result = self.add_signals(result, macd_col, signal_col, hist_col)
        
        return result
    
    def add_signals(self, data: pd.DataFrame, macd_col: str = 'DIF', 
                   signal_col: str = 'DEA', hist_col: str = 'MACD') -> pd.DataFrame:
        """
        添加MACD交易信号
        
        Args:
            data: 包含MACD指标的DataFrame
            macd_col: MACD线列名(DIF)
            signal_col: 信号线列名(DEA)
            hist_col: 柱状图列名(MACD)
            
        Returns:
            pd.DataFrame: 添加了信号的DataFrame
        """
        result = data.copy()
        
        # 计算金叉和死叉信号
        result['macd_buy_signal'] = self.get_buy_signal(result, macd_col, signal_col)
        result['macd_sell_signal'] = self.get_sell_signal(result, macd_col, signal_col)
        
        # 计算零轴穿越信号
        result['macd_zero_cross_up'] = (result[macd_col] > 0) & (result[macd_col].shift(1) <= 0)
        result['macd_zero_cross_down'] = (result[macd_col] < 0) & (result[macd_col].shift(1) >= 0)
        
        # 计算柱状图趋势
        result['macd_hist_increasing'] = result[hist_col] > result[hist_col].shift(1)
        result['macd_hist_decreasing'] = result[hist_col] < result[hist_col].shift(1)
        
        # 计算背离指标
        result = self._add_divergence_signals(result, macd_col, price_col='close')
        
        return result
    
    def get_buy_signal(self, data: pd.DataFrame, macd_col: str = 'DIF', 
                      signal_col: str = 'DEA') -> pd.Series:
        """
        获取MACD买入信号
        
        Args:
            data: 包含MACD指标的DataFrame
            macd_col: MACD线列名(DIF)
            signal_col: 信号线列名(DEA)
            
        Returns:
            pd.Series: 买入信号序列（布尔值）
        """
        # 使用公共cross函数检测金叉
        return pd.Series(
            cross(data[macd_col].values, data[signal_col].values),
            index=data.index
        )
    
    def get_sell_signal(self, data: pd.DataFrame, macd_col: str = 'DIF', 
                       signal_col: str = 'DEA') -> pd.Series:
        """
        获取MACD卖出信号
        
        Args:
            data: 包含MACD指标的DataFrame
            macd_col: MACD线列名(DIF)
            signal_col: 信号线列名(DEA)
            
        Returns:
            pd.Series: 卖出信号序列（布尔值）
        """
        # 使用公共cross函数检测死叉
        return pd.Series(
            cross(data[signal_col].values, data[macd_col].values),
            index=data.index
        )
    
    def _add_divergence_signals(self, data: pd.DataFrame, macd_col: str = 'DIF', 
                              price_col: str = 'close', window: int = 20) -> pd.DataFrame:
        """
        添加MACD背离信号
        
        Args:
            data: 包含MACD指标的DataFrame
            macd_col: MACD线列名(DIF)
            price_col: 价格列名
            window: 寻找背离的窗口大小
            
        Returns:
            pd.DataFrame: 添加了背离信号的DataFrame
        """
        result = data.copy()
        
        # 初始化背离信号列
        result['macd_bullish_divergence'] = False
        result['macd_bearish_divergence'] = False
        
        # 循环检测背离
        for i in range(window, len(result)):
            # 只检查窗口内的数据
            window_data = result.iloc[i-window:i+1]
            
            # 计算价格的局部最低点
            price_lows = window_data[window_data[price_col] == window_data[price_col].min()]
            
            # 计算价格的局部最高点
            price_highs = window_data[window_data[price_col] == window_data[price_col].max()]
            
            # 计算MACD的局部最低点
            macd_lows = window_data[window_data[macd_col] == window_data[macd_col].min()]
            
            # 计算MACD的局部最高点
            macd_highs = window_data[window_data[macd_col] == window_data[macd_col].max()]
            
            # 检查是否有足够的点来比较
            if len(price_lows) > 1 and len(macd_lows) > 1:
                # 检查看涨背离：价格创新低但MACD没有创新低
                if (price_lows.iloc[-1][price_col] < price_lows.iloc[0][price_col] and 
                    macd_lows.iloc[-1][macd_col] > macd_lows.iloc[0][macd_col]):
                    result.loc[result.index[i], 'macd_bullish_divergence'] = True
            
            # 检查是否有足够的点来比较
            if len(price_highs) > 1 and len(macd_highs) > 1:
                # 检查看跌背离：价格创新高但MACD没有创新高
                if (price_highs.iloc[-1][price_col] > price_highs.iloc[0][price_col] and 
                    macd_highs.iloc[-1][macd_col] < macd_highs.iloc[0][macd_col]):
                    result.loc[result.index[i], 'macd_bearish_divergence'] = True
        
        return result
    
    def ensure_columns(self, data: pd.DataFrame, columns: List[str]) -> None:
        """
        确保DataFrame包含所需的列
        
        Args:
            data: 输入数据
            columns: 所需的列名列表
            
        Raises:
            ValueError: 如果数据不包含所需的列
        """
        missing_columns = [col for col in columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少所需的列: {', '.join(missing_columns)}")
    
    def get_column_name(self, suffix: str = "") -> str:
        """
        获取指标列名
        
        Args:
            suffix: 列名后缀
            
        Returns:
            str: 指标列名
        """
        if suffix:
            return f"{self.name.lower()}_{suffix}"
        return self.name.lower()
    
    def get_cross_points(self, data: pd.DataFrame, cross_type: CrossType = CrossType.GOLDEN_CROSS,
                        macd_col: str = 'DIF', signal_col: str = 'DEA') -> pd.DataFrame:
        """
        获取MACD交叉点
        
        Args:
            data: 包含MACD指标的DataFrame
            cross_type: 交叉类型，金叉或死叉
            macd_col: MACD线列名(DIF)
            signal_col: 信号线列名(DEA)
            
        Returns:
            pd.DataFrame: 交叉点DataFrame
        """
        if cross_type == CrossType.GOLDEN_CROSS:
            # 金叉：MACD从下方穿过信号线
            cross_points = data[self.get_buy_signal(data, macd_col, signal_col)]
        else:
            # 死叉：MACD从上方穿过信号线
            cross_points = data[self.get_sell_signal(data, macd_col, signal_col)]
        
        return cross_points

    def to_dict(self) -> Dict:
        """
        将指标转换为字典表示
        
        Returns:
            Dict: 指标的字典表示
        """
        base_dict = super().to_dict()
        base_dict.update({
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period
        })
        return base_dict 