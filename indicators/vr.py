#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
成交量指标(VR)模块

实现成交量指标计算功能，用于判断多空力量对比
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class VR(BaseIndicator):
    """
    成交量指标(Volume Ratio)
    
    计算上涨成交量与下跌成交量的比值，判断多空力量对比
    """
    
    def __init__(self, period: int = 26, ma_period: int = 6):
        """
        初始化VR指标
        
        Args:
            period: VR计算周期，默认为26日
            ma_period: VR均线周期，默认为6日
        """
        super().__init__(name="VR", description="成交量指标，计算上涨成交量与下跌成交量的比值")
        self.period = period
        self.ma_period = ma_period
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算VR指标
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含VR指标值
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "volume"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 判断价格变动方向
        price_direction = np.zeros(len(data))
        price_direction[1:] = np.sign(data["close"].values[1:] - data["close"].values[:-1])
        
        # 初始化上涨、下跌和平盘成交量
        up_volume = np.zeros(len(data))
        down_volume = np.zeros(len(data))
        flat_volume = np.zeros(len(data))
        
        # 分类成交量
        for i in range(1, len(data)):
            if price_direction[i] > 0:  # 价格上涨
                up_volume[i] = data["volume"].iloc[i]
            elif price_direction[i] < 0:  # 价格下跌
                down_volume[i] = data["volume"].iloc[i]
            else:  # 价格不变
                flat_volume[i] = data["volume"].iloc[i]
        
        # 计算N日上涨、下跌和平盘成交量之和
        up_volume_sum = pd.Series(up_volume).rolling(window=self.period).sum()
        down_volume_sum = pd.Series(down_volume).rolling(window=self.period).sum()
        flat_volume_sum = pd.Series(flat_volume).rolling(window=self.period).sum()
        
        # 计算VR: (AVS+1/2SVS)/(BVS+1/2SVS)×100
        # 其中AVS为上涨成交量，BVS为下跌成交量，SVS为平盘成交量
        vr = ((up_volume_sum + 0.5 * flat_volume_sum) / 
             (down_volume_sum + 0.5 * flat_volume_sum)) * 100
        
        # 添加到结果
        result["vr"] = vr
        
        # 计算VR均线
        result["vr_ma"] = result["vr"].rolling(window=self.ma_period).mean()
        
        return result
    
    def get_signals(self, data: pd.DataFrame, overbought: float = 160, oversold: float = 70) -> pd.DataFrame:
        """
        生成VR信号
        
        Args:
            data: 输入数据，包含VR指标
            overbought: 超买阈值，默认为160
            oversold: 超卖阈值，默认为70
            
        Returns:
            pd.DataFrame: 包含VR信号的数据框
        """
        if "vr" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["vr_signal"] = np.nan
        
        # 生成信号
        for i in range(1, len(data)):
            if pd.notna(data["vr"].iloc[i]) and pd.notna(data["vr"].iloc[i-1]):
                # VR下穿超买线：卖出信号
                if data["vr"].iloc[i] < overbought and data["vr"].iloc[i-1] >= overbought:
                    data.iloc[i, data.columns.get_loc("vr_signal")] = -1
                
                # VR上穿超卖线：买入信号
                elif data["vr"].iloc[i] > oversold and data["vr"].iloc[i-1] <= oversold:
                    data.iloc[i, data.columns.get_loc("vr_signal")] = 1
                
                # 无信号
                else:
                    data.iloc[i, data.columns.get_loc("vr_signal")] = 0
        
        # 检测VR与VR均线的交叉
        data["vr_ma_cross"] = np.nan
        
        for i in range(1, len(data)):
            if (pd.notna(data["vr"].iloc[i]) and pd.notna(data["vr_ma"].iloc[i]) and 
                pd.notna(data["vr"].iloc[i-1]) and pd.notna(data["vr_ma"].iloc[i-1])):
                
                # VR上穿其均线：买入信号
                if data["vr"].iloc[i] > data["vr_ma"].iloc[i] and data["vr"].iloc[i-1] <= data["vr_ma"].iloc[i-1]:
                    data.iloc[i, data.columns.get_loc("vr_ma_cross")] = 1
                
                # VR下穿其均线：卖出信号
                elif data["vr"].iloc[i] < data["vr_ma"].iloc[i] and data["vr"].iloc[i-1] >= data["vr_ma"].iloc[i-1]:
                    data.iloc[i, data.columns.get_loc("vr_ma_cross")] = -1
                
                # 无交叉
                else:
                    data.iloc[i, data.columns.get_loc("vr_ma_cross")] = 0
        
        return data
    
    def get_market_sentiment(self, data: pd.DataFrame, overbought: float = 160, 
                           oversold: float = 70, neutral_upper: float = 120, 
                           neutral_lower: float = 90) -> pd.DataFrame:
        """
        获取市场情绪
        
        Args:
            data: 输入数据，包含VR指标
            overbought: 超买阈值，默认为160
            oversold: 超卖阈值，默认为70
            neutral_upper: 中性区间上限，默认为120
            neutral_lower: 中性区间下限，默认为90
            
        Returns:
            pd.DataFrame: 包含市场情绪的数据框
        """
        if "vr" not in data.columns:
            data = self.calculate(data)
        
        # 初始化情绪列
        data["market_sentiment"] = np.nan
        
        # 判断市场情绪
        for i in range(len(data)):
            if pd.notna(data["vr"].iloc[i]):
                # 多头区域
                if data["vr"].iloc[i] > overbought:
                    data.iloc[i, data.columns.get_loc("market_sentiment")] = "极度多头"
                
                # 多头倾向区域
                elif data["vr"].iloc[i] > neutral_upper:
                    data.iloc[i, data.columns.get_loc("market_sentiment")] = "多头"
                
                # 中性区域靠上
                elif data["vr"].iloc[i] > neutral_lower:
                    data.iloc[i, data.columns.get_loc("market_sentiment")] = "中性偏多"
                
                # 中性区域靠下
                elif data["vr"].iloc[i] > oversold:
                    data.iloc[i, data.columns.get_loc("market_sentiment")] = "中性偏空"
                
                # 空头区域
                else:
                    data.iloc[i, data.columns.get_loc("market_sentiment")] = "空头"
        
        return data
    
    def get_vr_change_rate(self, data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        计算VR变化率
        
        Args:
            data: 输入数据，包含VR指标
            window: 计算窗口期，默认为5日
            
        Returns:
            pd.DataFrame: 包含VR变化率的数据框
        """
        if "vr" not in data.columns:
            data = self.calculate(data)
        
        # 计算VR变化率
        data["vr_change_rate"] = data["vr"].pct_change(periods=window) * 100
        
        return data

