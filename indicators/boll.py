"""
布林带指标模块

实现布林带(BOLL)指标计算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from indicators.common import boll as calc_boll
from utils.logger import get_logger

logger = get_logger(__name__)


class BOLL(BaseIndicator):
    """
    布林带指标类
    
    计算布林带上轨、中轨和下轨
    """
    
    def __init__(self, name: str = "BOLL", description: str = "布林带指标"):
        """
        初始化布林带指标
        
        Args:
            name: 指标名称
            description: 指标描述
        """
        super().__init__(name, description)
        
        # 设置默认参数
        self._parameters = {
            'periods': 20,      # 周期
            'std_dev': 2.0      # 标准差倍数
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
        计算布林带指标
        
        Args:
            data: 输入数据，必须包含'close'列
            args: 位置参数
            kwargs: 关键字参数，可包含periods和std_dev
            
        Returns:
            pd.DataFrame: 包含upper、middle、lower列的DataFrame
        """
        # 确保数据包含close列
        self.ensure_columns(data, ['close'])
        
        # 获取参数
        periods = kwargs.get('periods', self._parameters['periods'])
        std_dev = kwargs.get('std_dev', self._parameters['std_dev'])
        
        # 计算布林带
        upper, middle, lower = calc_boll(data['close'], periods, std_dev)
        
        # 构建结果DataFrame
        result = pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        }, index=data.index)
        
        return result
    
    def is_upper_breakout(self, data: pd.DataFrame) -> pd.Series:
        """
        判断是否突破上轨
        
        Args:
            data: 包含close列和计算结果的DataFrame
            
        Returns:
            pd.Series: 突破信号，True表示突破上轨
        """
        if not self.has_result():
            self.compute(data)
            
        return data['close'] > self._result['upper']
    
    def is_lower_breakout(self, data: pd.DataFrame) -> pd.Series:
        """
        判断是否突破下轨
        
        Args:
            data: 包含close列和计算结果的DataFrame
            
        Returns:
            pd.Series: 突破信号，True表示突破下轨
        """
        if not self.has_result():
            self.compute(data)
            
        return data['close'] < self._result['lower']
    
    def is_middle_crossover(self, data: pd.DataFrame) -> pd.Series:
        """
        判断是否上穿中轨
        
        Args:
            data: 包含close列和计算结果的DataFrame
            
        Returns:
            pd.Series: 上穿信号，True表示上穿中轨
        """
        if not self.has_result():
            self.compute(data)
            
        return self.crossover(data['close'], self._result['middle'])
    
    def is_middle_crossunder(self, data: pd.DataFrame) -> pd.Series:
        """
        判断是否下穿中轨
        
        Args:
            data: 包含close列和计算结果的DataFrame
            
        Returns:
            pd.Series: 下穿信号，True表示下穿中轨
        """
        if not self.has_result():
            self.compute(data)
            
        return self.crossunder(data['close'], self._result['middle'])
    
    def get_bandwidth(self) -> pd.Series:
        """
        获取带宽
        
        Returns:
            pd.Series: 带宽 (upper - lower) / middle
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        return (self._result['upper'] - self._result['lower']) / self._result['middle']
    
    def get_position(self, data: pd.DataFrame) -> pd.Series:
        """
        获取价格在带宽中的相对位置
        
        Args:
            data: 包含close列的DataFrame
            
        Returns:
            pd.Series: 相对位置 (close - lower) / (upper - lower)
        """
        if not self.has_result():
            self.compute(data)
            
        return (data['close'] - self._result['lower']) / (self._result['upper'] - self._result['lower']) 