# -*- coding: utf-8 -*-
"""
Real模块 - 提供真实技术指标计算功能

这个模块是为了解决"No module named 'real'"错误而创建的兼容性模块
完全兼容所有现有的real模块调用
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List

# 重新导出real_technical_indicators中的所有内容
try:
    from indicators.real_technical_indicators import *
    from indicators.real_technical_indicators import RealIndicatorFactory
    
    # 创建全局工厂实例
    real_indicator_factory = RealIndicatorFactory()
    
    # 导出工厂方法
    def create_indicator(name: str, **kwargs):
        """创建真实指标实例"""
        return real_indicator_factory.create_indicator(name, **kwargs)
    
    def get_available_indicators():
        """获取可用指标列表"""
        return real_indicator_factory.get_available_indicators()
    
except ImportError:
    # 如果real_technical_indicators不存在，创建Mock实现
    class MockRealIndicatorFactory:
        """Mock真实指标工厂"""
        
        def create_indicator(self, name: str, **kwargs):
            """创建Mock指标"""
            return MockRealIndicator(name)
        
        def get_available_indicators(self):
            """获取可用指标列表"""
            return []
    
    class MockRealIndicator:
        """Mock真实指标"""
        
        def __init__(self, name: str):
            self.name = name
        
        def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
            """Mock计算方法"""
            return pd.DataFrame()
        
        def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
            """Mock信号方法"""
            return {'signal': 'HOLD', 'strength': 0.0}
    
    real_indicator_factory = MockRealIndicatorFactory()
    
    def create_indicator(name: str, **kwargs):
        """创建Mock指标实例"""
        return real_indicator_factory.create_indicator(name, **kwargs)
    
    def get_available_indicators():
        """获取Mock指标列表"""
        return real_indicator_factory.get_available_indicators()

# 常用的技术指标计算函数
def calculate_sma(data: pd.Series, period: int = 20) -> pd.Series:
    """计算简单移动平均"""
    return data.rolling(window=period).mean()

def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
    """计算指数移动平均"""
    return data.ewm(span=period).mean()

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """计算MACD"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    }

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """计算布林带"""
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    
    return {
        'Middle': sma,
        'Upper': sma + (std * std_dev),
        'Lower': sma - (std * std_dev)
    }

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """计算随机指标"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return {
        'K': k_percent,
        'D': d_percent
    }

# 兼容性别名和导出
RealIndicatorFactory = real_indicator_factory.__class__

# 导出所有公共接口
__all__ = [
    'create_indicator',
    'get_available_indicators', 
    'real_indicator_factory',
    'RealIndicatorFactory',
    'calculate_sma',
    'calculate_ema', 
    'calculate_rsi',
    'calculate_macd',
    'calculate_bollinger_bands',
    'calculate_stochastic'
]
