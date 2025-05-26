#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
岛型反转指标模块

实现岛型反转形态识别功能，用于识别跳空+反向跳空形成孤岛的形态
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class IslandReversal(BaseIndicator):
    """
    岛型反转指标
    
    识别跳空+反向跳空形成孤岛的价格形态，用于短期急剧反转信号
    """
    
    def __init__(self, gap_threshold: float = 0.01, island_max_days: int = 5):
        """
        初始化岛型反转指标
        
        Args:
            gap_threshold: 跳空阈值，默认为1%
            island_max_days: 岛型最大天数，默认为5日
        """
        super().__init__(name="IslandReversal", description="岛型反转指标，识别跳空+反向跳空形成孤岛的形态")
        self.gap_threshold = gap_threshold
        self.island_max_days = island_max_days
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算岛型反转指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含岛型反转信号
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low", "close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 初始化跳空和岛型反转标记
        result["up_gap"] = False  # 向上跳空
        result["down_gap"] = False  # 向下跳空
        result["top_island_reversal"] = False  # 顶部岛型反转
        result["bottom_island_reversal"] = False  # 底部岛型反转
        
        # 计算跳空
        for i in range(1, len(data)):
            # 向上跳空：当日最低价高于前日最高价
            if data["low"].iloc[i] > data["high"].iloc[i-1] * (1 + self.gap_threshold):
                result.iloc[i, result.columns.get_loc("up_gap")] = True
            
            # 向下跳空：当日最高价低于前日最低价
            if data["high"].iloc[i] < data["low"].iloc[i-1] * (1 - self.gap_threshold):
                result.iloc[i, result.columns.get_loc("down_gap")] = True
        
        # 识别岛型反转
        for i in range(self.island_max_days + 1, len(data)):
            # 查找前island_max_days天内的跳空
            for j in range(1, self.island_max_days + 1):
                # 顶部岛型反转：先向上跳空进入，然后向下跳空离开
                if (result["up_gap"].iloc[i-j] and 
                    result["down_gap"].iloc[i]):
                    
                    # 检查中间区域是否孤立（无与前后区间重叠）
                    island_min = data["low"].iloc[i-j:i].min()
                    island_max = data["high"].iloc[i-j:i].max()
                    
                    before_max = data["high"].iloc[i-j-1]
                    after_min = data["low"].iloc[i]
                    
                    # 增强条件：确保岛型区域明显孤立
                    if (island_min > before_max * (1 + self.gap_threshold*0.5) and 
                        island_max > after_min * (1 + self.gap_threshold*0.5) and
                        data["close"].iloc[i-j:i].mean() > data["close"].iloc[i-j-5:i-j].mean()):
                        result.iloc[i, result.columns.get_loc("top_island_reversal")] = True
                        break
                
                # 底部岛型反转：先向下跳空进入，然后向上跳空离开
                if (result["down_gap"].iloc[i-j] and 
                    result["up_gap"].iloc[i]):
                    
                    # 检查中间区域是否孤立（无与前后区间重叠）
                    island_min = data["low"].iloc[i-j:i].min()
                    island_max = data["high"].iloc[i-j:i].max()
                    
                    before_min = data["low"].iloc[i-j-1]
                    after_max = data["high"].iloc[i]
                    
                    # 增强条件：确保岛型区域明显孤立
                    if (island_max < before_min * (1 - self.gap_threshold*0.5) and 
                        island_min < after_max * (1 - self.gap_threshold*0.5) and
                        data["close"].iloc[i-j:i].mean() < data["close"].iloc[i-j-5:i-j].mean()):
                        result.iloc[i, result.columns.get_loc("bottom_island_reversal")] = True
                        break
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成岛型反转信号
        
        Args:
            data: 输入数据，包含岛型反转指标
            
        Returns:
            pd.DataFrame: 包含岛型反转交易信号的数据框
        """
        if "top_island_reversal" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["island_signal"] = 0
        
        # 生成交易信号
        for i in range(len(data)):
            # 顶部岛型反转：卖出信号
            if data["top_island_reversal"].iloc[i]:
                data.iloc[i, data.columns.get_loc("island_signal")] = -1
            
            # 底部岛型反转：买入信号
            elif data["bottom_island_reversal"].iloc[i]:
                data.iloc[i, data.columns.get_loc("island_signal")] = 1
        
        return data
    
    def get_island_details(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        获取岛型反转的详细信息
        
        Args:
            data: 输入数据，包含岛型反转指标
            
        Returns:
            List[Dict[str, Any]]: 岛型反转详细信息列表
        """
        if "top_island_reversal" not in data.columns:
            data = self.calculate(data)
        
        island_details = []
        
        # 查找所有岛型反转
        for i in range(self.island_max_days + 1, len(data)):
            if data["top_island_reversal"].iloc[i] or data["bottom_island_reversal"].iloc[i]:
                # 确定岛型类型
                island_type = "顶部岛型反转" if data["top_island_reversal"].iloc[i] else "底部岛型反转"
                
                # 查找岛型起始位置
                start_idx = i
                for j in range(1, self.island_max_days + 1):
                    if (island_type == "顶部岛型反转" and data["up_gap"].iloc[i-j]) or \
                       (island_type == "底部岛型反转" and data["down_gap"].iloc[i-j]):
                        start_idx = i - j
                        break
                
                # 计算岛型区域价格特征
                island_data = data.iloc[start_idx:i+1]
                island_high = island_data["high"].max()
                island_low = island_data["low"].min()
                island_days = len(island_data)
                
                # 计算前后跳空幅度
                if island_type == "顶部岛型反转":
                    entry_gap = (island_data["low"].iloc[0] - data["high"].iloc[start_idx-1]) / data["high"].iloc[start_idx-1]
                    exit_gap = (island_data["high"].iloc[-1] - data["low"].iloc[i]) / data["low"].iloc[i]
                else:
                    entry_gap = (data["low"].iloc[start_idx-1] - island_data["high"].iloc[0]) / island_data["high"].iloc[0]
                    exit_gap = (data["high"].iloc[i] - island_data["low"].iloc[-1]) / island_data["low"].iloc[-1]
                
                # 保存岛型信息
                island_info = {
                    "type": island_type,
                    "start_date": data.index[start_idx],
                    "end_date": data.index[i],
                    "days": island_days,
                    "high": island_high,
                    "low": island_low,
                    "entry_gap": abs(entry_gap),
                    "exit_gap": abs(exit_gap)
                }
                
                island_details.append(island_info)
        
        return island_details
    
    def get_gap_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        获取跳空统计信息
        
        Args:
            data: 输入数据，包含岛型反转指标
            
        Returns:
            Dict[str, float]: 跳空统计信息
        """
        if "up_gap" not in data.columns:
            data = self.calculate(data)
        
        # 计算跳空统计信息
        up_gaps = data["up_gap"].sum()
        down_gaps = data["down_gap"].sum()
        top_islands = data["top_island_reversal"].sum()
        bottom_islands = data["bottom_island_reversal"].sum()
        
        total_bars = len(data)
        
        statistics = {
            "up_gap_ratio": up_gaps / total_bars if total_bars > 0 else 0,
            "down_gap_ratio": down_gaps / total_bars if total_bars > 0 else 0,
            "top_island_ratio": top_islands / total_bars if total_bars > 0 else 0,
            "bottom_island_ratio": bottom_islands / total_bars if total_bars > 0 else 0,
            "island_to_gap_ratio": (top_islands + bottom_islands) / (up_gaps + down_gaps) if (up_gaps + down_gaps) > 0 else 0
        }
        
        return statistics 