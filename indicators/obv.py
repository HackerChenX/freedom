#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
能量潮(OBV)指标模块

实现能量潮指标的计算功能，用于判断资金流向与价格趋势的一致性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class OBV(BaseIndicator):
    """
    能量潮(On Balance Volume)指标
    
    根据价格变动方向，计算成交量的累计值，用于判断资金流向与价格趋势的一致性
    """
    
    def __init__(self, ma_period: int = 30):
        """
        初始化OBV指标
        
        Args:
            ma_period: OBV均线周期，默认为30日
        """
        super().__init__(name="OBV", description="能量潮指标，根据价格变动方向，计算成交量的累计值")
        self.ma_period = ma_period
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算OBV指标
        
        Args:
            data: 输入数据，包含价格和成交量数据
            
        Returns:
            pd.DataFrame: 计算结果，包含OBV及其均线
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "volume"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 价格变动方向
        price_direction = np.zeros(len(data))
        price_direction[1:] = np.sign(data["close"].values[1:] - data["close"].values[:-1])
        
        # 计算OBV
        obv = np.zeros(len(data))
        
        for i in range(1, len(data)):
            if price_direction[i] > 0:  # 价格上涨
                obv[i] = obv[i-1] + data["volume"].iloc[i]
            elif price_direction[i] < 0:  # 价格下跌
                obv[i] = obv[i-1] - data["volume"].iloc[i]
            else:  # 价格不变
                obv[i] = obv[i-1]
        
        # 添加到结果
        result["obv"] = obv
        
        # 计算OBV均线
        result["obv_ma"] = pd.Series(obv).rolling(window=self.ma_period).mean().values
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成OBV信号
        
        Args:
            data: 输入数据，包含OBV指标
            
        Returns:
            pd.DataFrame: 包含OBV信号的数据框
        """
        if "obv" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["obv_trend"] = np.nan
        data["obv_divergence"] = np.nan
        
        # 计算OBV趋势
        for i in range(5, len(data)):
            # 检查OBV短期趋势
            if data["obv"].iloc[i] > data["obv"].iloc[i-5]:
                data.iloc[i, data.columns.get_loc("obv_trend")] = 1  # OBV上升
            elif data["obv"].iloc[i] < data["obv"].iloc[i-5]:
                data.iloc[i, data.columns.get_loc("obv_trend")] = -1  # OBV下降
            else:
                data.iloc[i, data.columns.get_loc("obv_trend")] = 0  # OBV横盘
        
        # 计算OBV与价格的背离
        for i in range(20, len(data)):
            # 判断近期是否创新高或新低
            is_price_high = data["close"].iloc[i] >= np.max(data["close"].iloc[i-20:i])
            is_price_low = data["close"].iloc[i] <= np.min(data["close"].iloc[i-20:i])
            
            is_obv_high = data["obv"].iloc[i] >= np.max(data["obv"].iloc[i-20:i])
            is_obv_low = data["obv"].iloc[i] <= np.min(data["obv"].iloc[i-20:i])
            
            # 价格创新高但OBV未创新高 -> 负背离
            if is_price_high and not is_obv_high:
                data.iloc[i, data.columns.get_loc("obv_divergence")] = -1
            
            # 价格创新低但OBV未创新低 -> 正背离
            elif is_price_low and not is_obv_low:
                data.iloc[i, data.columns.get_loc("obv_divergence")] = 1
            
            # 无背离
            else:
                data.iloc[i, data.columns.get_loc("obv_divergence")] = 0
        
        # 检测OBV突破OBV均线
        data["obv_ma_cross"] = np.nan
        
        for i in range(1, len(data)):
            if pd.notna(data["obv_ma"].iloc[i]) and pd.notna(data["obv_ma"].iloc[i-1]):
                # OBV上穿均线
                if data["obv"].iloc[i] > data["obv_ma"].iloc[i] and data["obv"].iloc[i-1] <= data["obv_ma"].iloc[i-1]:
                    data.iloc[i, data.columns.get_loc("obv_ma_cross")] = 1
                
                # OBV下穿均线
                elif data["obv"].iloc[i] < data["obv_ma"].iloc[i] and data["obv"].iloc[i-1] >= data["obv_ma"].iloc[i-1]:
                    data.iloc[i, data.columns.get_loc("obv_ma_cross")] = -1
                
                # 无交叉
                else:
                    data.iloc[i, data.columns.get_loc("obv_ma_cross")] = 0
        
        return data
    
    def get_obv_strength(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        计算OBV强度
        
        Args:
            data: 输入数据，包含OBV指标
            window: 计算窗口期，默认为20日
            
        Returns:
            pd.DataFrame: 包含OBV强度的数据框
        """
        if "obv" not in data.columns:
            data = self.calculate(data)
        
        # 计算OBV变化率
        data["obv_change_rate"] = np.nan
        
        for i in range(window, len(data)):
            # OBV变化率 = (当前OBV - N日前OBV) / N日前OBV
            data.iloc[i, data.columns.get_loc("obv_change_rate")] = (
                (data["obv"].iloc[i] - data["obv"].iloc[i-window]) / 
                abs(data["obv"].iloc[i-window]) if data["obv"].iloc[i-window] != 0 else 0
            )
        
        # 计算OBV强度
        data["obv_strength"] = np.nan
        
        for i in range(window, len(data)):
            # OBV强度 = OBV变化率 / 价格变化率
            price_change_rate = (data["close"].iloc[i] - data["close"].iloc[i-window]) / data["close"].iloc[i-window]
            
            if price_change_rate != 0:
                data.iloc[i, data.columns.get_loc("obv_strength")] = data["obv_change_rate"].iloc[i] / price_change_rate
            else:
                data.iloc[i, data.columns.get_loc("obv_strength")] = 0
        
        return data

