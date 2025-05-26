"""
移动平均线指标模块

实现各种移动平均线(MA)指标计算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from indicators.common import ma, ema, wma, sma
from utils.logger import get_logger

logger = get_logger(__name__)


class MA(BaseIndicator):
    """
    移动平均线指标类
    
    计算各种类型的移动平均线
    """
    
    # MA类型枚举
    MA_TYPE_SIMPLE = 'simple'  # 简单移动平均
    MA_TYPE_EMA = 'ema'        # 指数移动平均
    MA_TYPE_WMA = 'wma'        # 加权移动平均
    MA_TYPE_SMA = 'sma'        # 平滑移动平均
    
    def __init__(self, name: str = "MA", description: str = "移动平均线指标", periods: List[int] = None):
        """
        初始化移动平均线指标
        
        Args:
            name: 指标名称
            description: 指标描述
            periods: 周期列表，默认为[5, 10, 20, 30, 60]
        """
        super().__init__(name, description)
        
        # 设置默认参数
        self._parameters = {
            'periods': periods or [5, 10, 20, 30, 60],  # 周期列表
            'ma_type': self.MA_TYPE_SIMPLE,  # MA类型
            'weight': 1                      # SMA的权重参数
        }
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """获取参数"""
        return self._parameters.copy()
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        设置参数
        
        Args:
            params: 参数字典
        """
        for key, value in params.items():
            if key in self._parameters:
                self._parameters[key] = value
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算移动平均线指标
        
        Args:
            data: 输入数据，必须包含'close'列
            args: 位置参数
            kwargs: 关键字参数，可包含periods、ma_type和weight
            
        Returns:
            pd.DataFrame: 包含各周期MA的DataFrame
        """
        # 确保数据包含close列
        self.ensure_columns(data, ['close'])
        
        # 获取参数
        periods = kwargs.get('periods', self._parameters['periods'])
        ma_type = kwargs.get('ma_type', self._parameters['ma_type'])
        weight = kwargs.get('weight', self._parameters['weight'])
        
        # 如果periods是单个值，转换为列表
        if not isinstance(periods, list):
            periods = [periods]
        
        # 根据MA类型选择对应的计算函数
        if ma_type == self.MA_TYPE_EMA:
            calc_func = ema
        elif ma_type == self.MA_TYPE_WMA:
            calc_func = wma
        elif ma_type == self.MA_TYPE_SMA:
            def calc_func(series, period):
                return sma(series, period, weight)
        else:  # 默认使用简单移动平均
            calc_func = ma
        
        # 计算各周期的移动平均线
        result = pd.DataFrame(index=data.index)
        for period in periods:
            ma_values = calc_func(data['close'], period)
            result[f'MA{period}'] = ma_values
        
        return result
    
    def is_uptrend(self, period: int) -> pd.Series:
        """
        判断指定周期的均线是否上升趋势
        
        Args:
            period: MA周期
            
        Returns:
            pd.Series: 趋势信号，True表示上升趋势
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        ma_col = f'MA{period}'
        if ma_col not in self._result.columns:
            raise ValueError(f"结果中不存在{ma_col}列")
            
        return self._result[ma_col] > self._result[ma_col].shift(1)
    
    def is_golden_cross(self, short_period: int, long_period: int) -> pd.Series:
        """
        判断是否形成金叉（短期均线上穿长期均线）
        
        Args:
            short_period: 短期均线周期
            long_period: 长期均线周期
            
        Returns:
            pd.Series: 金叉信号，True表示形成金叉
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        short_ma = f'MA{short_period}'
        long_ma = f'MA{long_period}'
        
        if short_ma not in self._result.columns or long_ma not in self._result.columns:
            raise ValueError(f"结果中不存在{short_ma}或{long_ma}列")
            
        return self.crossover(self._result[short_ma], self._result[long_ma])
    
    def is_death_cross(self, short_period: int, long_period: int) -> pd.Series:
        """
        判断是否形成死叉（短期均线下穿长期均线）
        
        Args:
            short_period: 短期均线周期
            long_period: 长期均线周期
            
        Returns:
            pd.Series: 死叉信号，True表示形成死叉
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        short_ma = f'MA{short_period}'
        long_ma = f'MA{long_period}'
        
        if short_ma not in self._result.columns or long_ma not in self._result.columns:
            raise ValueError(f"结果中不存在{short_ma}或{long_ma}列")
            
        return self.crossunder(self._result[short_ma], self._result[long_ma])
    
    def is_multi_uptrend(self, periods: List[int] = None) -> pd.Series:
        """
        判断是否多条均线多头排列（短期均线在上，长期均线在下）
        
        Args:
            periods: 均线周期列表，按从短到长排序
            
        Returns:
            pd.Series: 多头排列信号，True表示形成多头排列
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        if periods is None:
            # 使用默认周期列表，但要确保是有序的
            periods = sorted(self._parameters['periods'])
            
        # 确保所有需要的列都存在
        for period in periods:
            if f'MA{period}' not in self._result.columns:
                raise ValueError(f"结果中不存在MA{period}列")
        
        # 检查是否形成多头排列
        result = pd.Series(True, index=self._result.index)
        for i in range(len(periods) - 1):
            short_ma = f'MA{periods[i]}'
            long_ma = f'MA{periods[i+1]}'
            result &= (self._result[short_ma] > self._result[long_ma])
            
        return result
    
    def price_to_ma_ratio(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        计算价格与均线的比值
        
        Args:
            data: 包含close列的DataFrame
            period: 均线周期
            
        Returns:
            pd.Series: 价格与均线的比值
        """
        if not self.has_result():
            self.compute(data)
            
        ma_col = f'MA{period}'
        if ma_col not in self._result.columns:
            raise ValueError(f"结果中不存在{ma_col}列")
            
        return data['close'] / self._result[ma_col]
    
    def is_touching_ma(self, data: pd.DataFrame, period: int, threshold: float = 0.02) -> pd.Series:
        """
        判断最低价是否接触或接近均线
        
        Args:
            data: 包含low列的DataFrame
            period: 均线周期
            threshold: 接近阈值，如0.02表示在均线2%范围内都算接触
            
        Returns:
            pd.Series: 接触信号，True表示最低价接触或接近均线
        """
        if not self.has_result():
            self.compute(data)
            
        ma_col = f'MA{period}'
        if ma_col not in self._result.columns:
            raise ValueError(f"结果中不存在{ma_col}列")
            
        ma_values = self._result[ma_col]
        
        # 检查最低价是否在均线的阈值范围内
        lower_bound = ma_values * (1 - threshold)
        upper_bound = ma_values * (1 + threshold)
        
        return (data['low'] >= lower_bound) & (data['low'] <= upper_bound) 