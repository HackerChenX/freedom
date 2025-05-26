"""
RSI指标模块

实现RSI指标的计算和相关功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any

from indicators.base_indicator import BaseIndicator
from indicators.common import rsi as calc_rsi
from utils.decorators import exception_handler, validate_dataframe, log_calls
from utils.logger import get_logger

logger = get_logger(__name__)


class RSI(BaseIndicator):
    """
    RSI(Relative Strength Index)指标
    
    RSI是一种动量震荡指标，通过计算价格变动的相对强度来衡量市场超买或超卖情况。
    """
    
    def __init__(self, period: int = 14, periods: Optional[Union[int, List[int]]] = None, 
                overbought: float = 70.0, oversold: float = 30.0):
        """
        初始化RSI指标
        
        Args:
            period: 计算周期，默认为14
            periods: period的别名，为了兼容性；如果是列表，则取第一个值
            overbought: 超买阈值，默认为70
            oversold: 超卖阈值，默认为30
        """
        super().__init__(name="RSI", description="相对强弱指标")
        
        # 处理periods参数可能是列表的情况
        if periods is not None:
            if isinstance(periods, list) and len(periods) > 0:
                # 如果是列表，取第一个值
                self.period = periods[0]
            else:
                # 否则直接使用
                self.period = periods
        else:
            # 如果periods为None，则使用period参数
            self.period = period
            
        self.overbought = overbought
        self.oversold = oversold
    
    @validate_dataframe(required_columns=['close'], min_rows=15)
    @log_calls(level='debug')
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame, price_col: str = 'close', 
                  add_prefix: bool = False, **kwargs) -> pd.DataFrame:
        """
        计算RSI指标
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            price_col: 价格列名，默认为'close'
            add_prefix: 是否在输出列名前添加指标名称前缀
            kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含RSI指标的DataFrame
            
        Raises:
            ValueError: 如果输入数据不包含价格列
        """
        # 复制输入数据
        result = data.copy()
        
        # 处理周期列表
        periods = kwargs.get('periods', [self.period])
        if not isinstance(periods, list):
            periods = [periods]
        
        # 对每个周期计算RSI
        for period in periods:
            # 使用公共函数计算RSI
            rsi_values = calc_rsi(result[price_col].values, period)
            
            # 设置列名
            if add_prefix:
                rsi_col = self.get_column_name(str(period))
            else:
                rsi_col = f'RSI{period}'
            
            # 添加RSI列
            result[rsi_col] = rsi_values
            
            # 添加信号
            result = self.add_signals(result, rsi_col, period)
        
        # 保存结果
        self._result = result
        
        return result
    
    def get_buy_signal(self, data: pd.DataFrame, rsi_col: str = 'rsi') -> pd.Series:
        """
        获取RSI买入信号
        
        Args:
            data: 包含RSI指标的DataFrame
            rsi_col: RSI列名
            
        Returns:
            pd.Series: 买入信号序列，True表示买入信号
        """
        # 确保数据包含RSI列
        self.ensure_columns(data, [rsi_col])
        
        # 从超卖区域上穿超卖阈值
        cross_oversold = (data[rsi_col].shift(1) < self.oversold) & (data[rsi_col] >= self.oversold)
        
        # RSI从低位上升
        rising = data[rsi_col] > data[rsi_col].shift(1)
        
        # 综合信号：从超卖区域上穿 或 RSI在低位且上升
        buy_signal = cross_oversold | ((data[rsi_col] < 40) & rising)
        
        return buy_signal
    
    def get_sell_signal(self, data: pd.DataFrame, rsi_col: str = 'rsi') -> pd.Series:
        """
        获取RSI卖出信号
        
        Args:
            data: 包含RSI指标的DataFrame
            rsi_col: RSI列名
            
        Returns:
            pd.Series: 卖出信号序列，True表示卖出信号
        """
        # 确保数据包含RSI列
        self.ensure_columns(data, [rsi_col])
        
        # 从超买区域下穿超买阈值
        cross_overbought = (data[rsi_col].shift(1) > self.overbought) & (data[rsi_col] <= self.overbought)
        
        # RSI从高位下降
        falling = data[rsi_col] < data[rsi_col].shift(1)
        
        # 综合信号：从超买区域下穿 或 RSI在高位且下降
        sell_signal = cross_overbought | ((data[rsi_col] > 60) & falling)
        
        return sell_signal
    
    def add_signals(self, data: pd.DataFrame, rsi_col: str = 'rsi', period: int = None) -> pd.DataFrame:
        """
        添加RSI信号到数据
        
        Args:
            data: 包含RSI指标的DataFrame
            rsi_col: RSI列名
            period: RSI周期
            
        Returns:
            pd.DataFrame: 添加了RSI信号的DataFrame
        """
        # 确保数据包含RSI列
        self.ensure_columns(data, [rsi_col])
        
        # 复制输入数据
        result = data.copy()
        
        # 为多周期RSI添加后缀
        suffix = f"_{period}" if period is not None else ""
        
        # 添加信号列
        result[f'rsi_buy_signal{suffix}'] = self.get_buy_signal(result, rsi_col)
        result[f'rsi_sell_signal{suffix}'] = self.get_sell_signal(result, rsi_col)
        
        # 添加超买超卖区域标记
        result[f'rsi_overbought{suffix}'] = result[rsi_col] >= self.overbought
        result[f'rsi_oversold{suffix}'] = result[rsi_col] <= self.oversold
        
        # 添加趋势信息
        result[f'rsi_rising{suffix}'] = result[rsi_col] > result[rsi_col].shift(1)
        result[f'rsi_falling{suffix}'] = result[rsi_col] < result[rsi_col].shift(1)
        
        # 添加强度标签
        conditions = [
            (result[rsi_col] >= 80),
            (result[rsi_col] >= 60) & (result[rsi_col] < 80),
            (result[rsi_col] >= 40) & (result[rsi_col] < 60),
            (result[rsi_col] >= 20) & (result[rsi_col] < 40),
            (result[rsi_col] < 20)
        ]
        choices = ['极强', '强', '中性', '弱', '极弱']
        result[f'rsi_strength{suffix}'] = np.select(conditions, choices, default='未知')
        
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
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将指标转换为字典表示
        
        Returns:
            Dict[str, Any]: 指标的字典表示
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'period': self.period,
                'overbought': self.overbought,
                'oversold': self.oversold
            },
            'has_result': self.has_result(),
            'has_error': self.has_error(),
            'error': str(self._error) if self._error else None
        } 