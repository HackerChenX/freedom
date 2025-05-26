#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
资金流向指标(MFI)模块

实现资金流向指标计算功能，用于识别价格反转点
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class MFI(BaseIndicator):
    """
    资金流向指标(Money Flow Index)
    
    计算资金流入和流出比率，结合价格与成交量判断超买超卖情况，用于识别价格反转点
    """
    
    def __init__(self, period: int = 14):
        """
        初始化MFI指标
        
        Args:
            period: 计算周期，默认为14日
        """
        super().__init__(name="MFI", description="资金流向指标，计算资金流入和流出比率，判断超买超卖")
        self.period = period
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算MFI指标
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含MFI指标值
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["high", "low", "close", "volume"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算典型价格: (high + low + close) / 3
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        
        # 计算资金流: 典型价格 * 成交量
        money_flow = typical_price * data["volume"]
        
        # 初始化正向资金流和负向资金流
        positive_flow = np.zeros(len(data))
        negative_flow = np.zeros(len(data))
        
        # 计算正向和负向资金流
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:  # 价格上涨
                positive_flow[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:  # 价格下跌
                negative_flow[i] = money_flow.iloc[i]
            else:  # 价格不变
                if money_flow.iloc[i] > money_flow.iloc[i-1]:
                    positive_flow[i] = money_flow.iloc[i]
                else:
                    negative_flow[i] = money_flow.iloc[i]
        
        # 计算周期内的正向和负向资金流总和
        positive_flow_sum = pd.Series(positive_flow).rolling(window=self.period).sum()
        negative_flow_sum = pd.Series(negative_flow).rolling(window=self.period).sum()
        
        # 计算资金比率: 正向资金流 / 负向资金流
        money_ratio = np.where(negative_flow_sum != 0, positive_flow_sum / negative_flow_sum, 100)
        
        # 计算MFI: 100 - 100 / (1 + 资金比率)
        mfi = 100 - (100 / (1 + money_ratio))
        
        # 添加到结果
        result["mfi"] = mfi
        
        # 额外计算：MFI变化率
        result["mfi_change"] = result["mfi"].diff()
        
        return result
    
    def get_signals(self, data: pd.DataFrame, overbought: float = 80, oversold: float = 20) -> pd.DataFrame:
        """
        生成MFI信号
        
        Args:
            data: 输入数据，包含MFI指标
            overbought: 超买阈值，默认为80
            oversold: 超卖阈值，默认为20
            
        Returns:
            pd.DataFrame: 包含MFI信号的数据框
        """
        if "mfi" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["mfi_signal"] = np.nan
        
        # 生成信号
        for i in range(1, len(data)):
            if pd.notna(data["mfi"].iloc[i]) and pd.notna(data["mfi"].iloc[i-1]):
                # MFI下穿超买线：卖出信号
                if data["mfi"].iloc[i] < overbought and data["mfi"].iloc[i-1] >= overbought:
                    data.iloc[i, data.columns.get_loc("mfi_signal")] = -1
                
                # MFI上穿超卖线：买入信号
                elif data["mfi"].iloc[i] > oversold and data["mfi"].iloc[i-1] <= oversold:
                    data.iloc[i, data.columns.get_loc("mfi_signal")] = 1
                
                # 无信号
                else:
                    data.iloc[i, data.columns.get_loc("mfi_signal")] = 0
        
        # 检测MFI背离
        data["mfi_divergence"] = np.nan
        window = 20  # 背离检测窗口
        
        for i in range(window, len(data)):
            # 价格新高/新低检测
            price_high = data["close"].iloc[i] >= np.max(data["close"].iloc[i-window:i])
            price_low = data["close"].iloc[i] <= np.min(data["close"].iloc[i-window:i])
            
            # MFI新高/新低检测
            mfi_high = data["mfi"].iloc[i] >= np.max(data["mfi"].iloc[i-window:i])
            mfi_low = data["mfi"].iloc[i] <= np.min(data["mfi"].iloc[i-window:i])
            
            # 顶背离：价格新高但MFI未创新高
            if price_high and not mfi_high and data["mfi"].iloc[i] < data["mfi"].iloc[i-5]:
                data.iloc[i, data.columns.get_loc("mfi_divergence")] = -1
            
            # 底背离：价格新低但MFI未创新低
            elif price_low and not mfi_low and data["mfi"].iloc[i] > data["mfi"].iloc[i-5]:
                data.iloc[i, data.columns.get_loc("mfi_divergence")] = 1
            
            # 无背离
            else:
                data.iloc[i, data.columns.get_loc("mfi_divergence")] = 0
        
        return data
    
    def get_market_status(self, data: pd.DataFrame, overbought: float = 80, oversold: float = 20) -> pd.DataFrame:
        """
        获取市场状态
        
        Args:
            data: 输入数据，包含MFI指标
            overbought: 超买阈值，默认为80
            oversold: 超卖阈值，默认为20
            
        Returns:
            pd.DataFrame: 包含市场状态的数据框
        """
        if "mfi" not in data.columns:
            data = self.calculate(data)
        
        # 初始化状态列
        data["market_status"] = np.nan
        
        # 判断市场状态
        for i in range(len(data)):
            if pd.notna(data["mfi"].iloc[i]):
                # 超买区域
                if data["mfi"].iloc[i] > overbought:
                    data.iloc[i, data.columns.get_loc("market_status")] = "超买"
                
                # 超卖区域
                elif data["mfi"].iloc[i] < oversold:
                    data.iloc[i, data.columns.get_loc("market_status")] = "超卖"
                
                # 中性区域靠上
                elif data["mfi"].iloc[i] >= 50:
                    data.iloc[i, data.columns.get_loc("market_status")] = "中性偏多"
                
                # 中性区域靠下
                else:
                    data.iloc[i, data.columns.get_loc("market_status")] = "中性偏空"
        
        return data

