"""
ZXM体系弹性指标模块

实现ZXM体系的2个弹性指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMAmplitudeElasticity(BaseIndicator):
    """
    ZXM弹性-振幅指标
    
    判断近120日内是否有日振幅超过8.1%的情况
    """
    
    def __init__(self):
        """初始化ZXM弹性-振幅指标"""
        super().__init__(name="ZXMAmplitudeElasticity", description="ZXM弹性-振幅指标，判断近期是否有较大振幅")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM弹性-振幅指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含弹性信号
            
        公式说明：
        a1:=100*(H-L)/L>8.1;
        COUNT(a1,120)>1
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["high", "low"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算日振幅
        amplitude = 100 * (data["high"] - data["low"]) / data["low"]
        
        # 计算日振幅超过8.1%的情况
        a1 = amplitude > 8.1
        
        # 计算120日内是否有超过1次日振幅大于8.1%
        xg = pd.Series(np.zeros(len(data), dtype=bool), index=data.index)
        
        for i in range(120, len(data)):
            xg.iloc[i] = np.sum(a1.iloc[i-119:i+1]) > 1
        
        # 添加计算结果到数据框
        result["Amplitude"] = amplitude
        result["A1"] = a1
        result["XG"] = xg
        
        return result


class ZXMRiseElasticity(BaseIndicator):
    """
    ZXM弹性-涨幅指标
    
    判断近80日内是否有日涨幅超过7%的情况
    """
    
    def __init__(self):
        """初始化ZXM弹性-涨幅指标"""
        super().__init__(name="ZXMRiseElasticity", description="ZXM弹性-涨幅指标，判断近期是否有较大涨幅")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM弹性-涨幅指标
        
        Args:
            data: 输入数据，包含收盘价数据
            
        Returns:
            pd.DataFrame: 计算结果，包含弹性信号
            
        公式说明：
        a1:=C/REF(C,1)>1.07;
        COUNT(a1,80)>0
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算日涨幅
        rise_ratio = data["close"] / data["close"].shift(1)
        
        # 计算日涨幅超过7%的情况
        a1 = rise_ratio > 1.07
        
        # 计算80日内是否有涨幅大于7%
        xg = pd.Series(np.zeros(len(data), dtype=bool), index=data.index)
        
        for i in range(80, len(data)):
            xg.iloc[i] = np.sum(a1.iloc[i-79:i+1]) > 0
        
        # 添加计算结果到数据框
        result["RiseRatio"] = rise_ratio
        result["A1"] = a1
        result["XG"] = xg
        
        return result 