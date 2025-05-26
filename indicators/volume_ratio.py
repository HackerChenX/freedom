"""
量比指标模块

实现量比指标计算功能，用于判断市场热度
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class VolumeRatio(BaseIndicator):
    """
    量比指标
    
    计算当日成交量与过去N日平均成交量的比值，用于判断市场热度
    """
    
    def __init__(self, reference_period: int = 5):
        """
        初始化量比指标
        
        Args:
            reference_period: 参考周期，默认为5日
        """
        super().__init__(name="VolumeRatio", description="量比指标，计算当日成交量与过去N日平均成交量的比值")
        self.reference_period = reference_period
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算量比指标
        
        Args:
            data: 输入数据，包含成交量数据
            
        Returns:
            pd.DataFrame: 计算结果，包含量比指标
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["volume"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算成交量的N日移动平均
        volume_ma = data["volume"].rolling(window=self.reference_period).mean()
        
        # 计算量比：当日成交量 / N日平均成交量
        result["volume_ratio"] = data["volume"] / volume_ma
        
        # 计算相对量比：当日量比 / N日量比均值
        relative_volume_ratio = result["volume_ratio"] / result["volume_ratio"].rolling(window=self.reference_period).mean()
        result["relative_volume_ratio"] = relative_volume_ratio
        
        return result
    
    def get_signals(self, data: pd.DataFrame, active_threshold: float = 1.5, 
                   quiet_threshold: float = 0.7) -> pd.DataFrame:
        """
        生成量比信号
        
        Args:
            data: 输入数据，包含量比指标
            active_threshold: 活跃阈值，默认为1.5
            quiet_threshold: 低迷阈值，默认为0.7
            
        Returns:
            pd.DataFrame: 包含量比信号的数据框
        """
        if "volume_ratio" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["volume_active"] = data["volume_ratio"] > active_threshold  # 成交活跃信号
        data["volume_quiet"] = data["volume_ratio"] < quiet_threshold    # 成交低迷信号
        
        # 连续活跃或低迷的天数
        data["active_days"] = 0
        data["quiet_days"] = 0
        
        # 计算连续活跃或低迷的天数
        for i in range(1, len(data)):
            if data["volume_active"].iloc[i]:
                data.iloc[i, data.columns.get_loc("active_days")] = data["active_days"].iloc[i-1] + 1
            
            if data["volume_quiet"].iloc[i]:
                data.iloc[i, data.columns.get_loc("quiet_days")] = data["quiet_days"].iloc[i-1] + 1
        
        return data
    
    def get_market_status(self, data: pd.DataFrame) -> str:
        """
        获取市场热度状态
        
        Args:
            data: 输入数据，包含量比指标
            
        Returns:
            str: 市场热度状态描述
        """
        if "volume_ratio" not in data.columns:
            data = self.calculate(data)
        
        # 获取最新的量比值
        latest_vr = data["volume_ratio"].iloc[-1]
        
        # 判断市场热度
        if latest_vr > 2.0:
            return "极度活跃"
        elif latest_vr > 1.5:
            return "活跃"
        elif latest_vr > 1.0:
            return "正常偏活跃"
        elif latest_vr > 0.7:
            return "正常偏低迷"
        else:
            return "低迷" 