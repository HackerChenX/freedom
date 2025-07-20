"""
指标工厂模块

提供统一的指标创建接口
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class IndicatorFactory:
    """指标工厂类"""
    
    @staticmethod
    def create_indicator(indicator_name: str, **kwargs) -> 'MockIndicator':
        """
        创建指标实例
        
        Args:
            indicator_name: 指标名称
            **kwargs: 指标参数
            
        Returns:
            MockIndicator: 模拟指标实例
        """
        return MockIndicator(indicator_name, **kwargs)


class MockIndicator:
    """模拟指标类，用于测试"""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
        # 添加常用的指标属性
        self.period = kwargs.get('period', 14)
        self.fast_period = kwargs.get('fast_period', 12)
        self.slow_period = kwargs.get('slow_period', 26)
        self.signal_period = kwargs.get('signal_period', 9)
        self.multiplier = kwargs.get('multiplier', 2.0)
        self.length = kwargs.get('length', 20)
        self.deviation = kwargs.get('deviation', 2)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算指标值"""
        # 返回简单的模拟数据
        result = pd.DataFrame()
        
        if 'close' in data.columns:
            close = data['close']
            
            if self.name == "MA":
                periods = self.params.get('periods', [20])
                for period in periods:
                    result[f'MA{period}'] = close.rolling(window=period).mean()
                    
            elif self.name == "BOLL":
                period = self.params.get('period', 20)
                std_dev = self.params.get('std_dev', 2)
                ma = close.rolling(window=period).mean()
                std = close.rolling(window=period).std()
                result['BOLL_UPPER'] = ma + (std * std_dev)
                result['BOLL_MIDDLE'] = ma
                result['BOLL_LOWER'] = ma - (std * std_dev)
                
            elif self.name == "MACD":
                result['MACD'] = close * 0  # 简单的零值
                result['MACD_SIGNAL'] = close * 0
                result['MACD_HIST'] = close * 0
                
            else:
                # 对于其他指标，返回简单的模拟值
                result[self.name] = close * 0.5
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        """获取信号"""
        if 'close' in data.columns:
            # 返回简单的随机信号
            return pd.Series([False] * len(data), index=data.index)
        return pd.Series(dtype=bool) 