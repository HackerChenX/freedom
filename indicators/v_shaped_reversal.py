#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
V形反转指标模块

实现V形反转形态识别功能，用于识别急速下跌后快速反弹的形态
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class VShapedReversal(BaseIndicator):
    """
    V形反转指标
    
    识别急速下跌后快速反弹的价格形态，用于短期超卖后的反弹信号
    """
    
    def __init__(self, decline_period: int = 5, rebound_period: int = 5,
               decline_threshold: float = 0.05, rebound_threshold: float = 0.05):
        """
        初始化V形反转指标
        
        Args:
            decline_period: 下跌周期，默认为5日
            rebound_period: 反弹周期，默认为5日
            decline_threshold: 下跌阈值，默认为5%
            rebound_threshold: 反弹阈值，默认为5%
        """
        super().__init__(name="VShapedReversal", description="V形反转指标，识别急速下跌后快速反弹的形态")
        self.decline_period = decline_period
        self.rebound_period = rebound_period
        self.decline_threshold = decline_threshold
        self.rebound_threshold = rebound_threshold
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算V形反转指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含V形反转指标的DataFrame
        """
        return self.calculate(df)
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算V形反转指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含V形反转信号
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算各周期的价格变化率
        result["decline_rate"] = data["close"].pct_change(periods=self.decline_period)
        result["rebound_rate"] = data["close"].pct_change(periods=self.rebound_period)
        
        # 初始化V形反转信号
        result["v_reversal"] = False
        
        # 最小索引值，确保有足够的数据计算
        min_idx = self.decline_period + self.rebound_period
        
        if len(data) > min_idx:
            close_values = data["close"].values
            
            # 使用滑动窗口计算下跌和反弹
            for i in range(min_idx, len(data)):
                # 计算下跌区间的起始和结束价格
                decline_start_idx = i - self.decline_period - self.rebound_period
                decline_end_idx = i - self.rebound_period
                
                if decline_start_idx >= 0:
                    decline_start_price = close_values[decline_start_idx]
                    decline_end_price = close_values[decline_end_idx]
                    
                    # 计算反弹区间的起始和结束价格
                    rebound_start_idx = i - self.rebound_period
                    rebound_end_idx = i
                    
                    rebound_start_price = close_values[rebound_start_idx]
                    rebound_end_price = close_values[rebound_end_idx]
                    
                    # 计算下跌和反弹幅度
                    if decline_start_price > 0 and rebound_start_price > 0:
                        decline_rate = (decline_end_price - decline_start_price) / decline_start_price
                        rebound_rate = (rebound_end_price - rebound_start_price) / rebound_start_price
                        
                        # 判断是否满足V形反转条件
                        if decline_rate <= -self.decline_threshold and rebound_rate >= self.rebound_threshold:
                            result.iloc[i, result.columns.get_loc("v_reversal")] = True
        
        # 计算V形底部位置
        result["v_bottom"] = False
        
        if len(data) > self.decline_period + self.rebound_period:
            close_values = data["close"].values
            
            # 使用滑动窗口检测V形底部
            for i in range(self.decline_period, len(data) - self.rebound_period):
                pre_window_start = i - self.decline_period
                pre_window_end = i + 1
                post_window_start = i
                post_window_end = i + self.rebound_period + 1
                
                if pre_window_start >= 0 and post_window_end <= len(close_values):
                    pre_window = close_values[pre_window_start:pre_window_end]
                    post_window = close_values[post_window_start:post_window_end]
                    
                    # 如果当前价格是前后窗口的最低点
                    if (close_values[i] <= np.min(pre_window) and
                        close_values[i] <= np.min(post_window)):
                        result.iloc[i, result.columns.get_loc("v_bottom")] = True
        
        return result
    
    def get_signals(self, data: pd.DataFrame, confirmation_days: int = 2) -> pd.DataFrame:
        """
        生成V形反转信号
        
        Args:
            data: 输入数据，包含V形反转指标
            confirmation_days: 确认天数，默认为2日
            
        Returns:
            pd.DataFrame: 包含V形反转信号的数据框
        """
        if "v_reversal" not in data.columns:
            data = self.calculate(data)
        
        # 创建结果DataFrame的副本
        result = data.copy()
        
        # 初始化买入信号列
        result["v_buy_signal"] = False
        
        # 使用滑动窗口检测前N天是否有V形反转
        if len(data) > confirmation_days:
            # 获取v_reversal列的值
            v_reversal_values = data["v_reversal"].values
            close_values = data["close"].values
            
            # 滑动窗口检测
            for i in range(confirmation_days, len(data)):
                # 检查前confirmation_days天内是否有V形反转
                has_reversal = np.any(v_reversal_values[i-confirmation_days:i])
                
                # 检查价格是否上涨
                if i >= confirmation_days:
                    price_rising = close_values[i] > close_values[i-confirmation_days]
                    
                    # 生成买入信号
                    if has_reversal and price_rising:
                        result.iloc[i, result.columns.get_loc("v_buy_signal")] = True
        
        return result
    
    def get_reversal_strength(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算V形反转强度
        
        Args:
            data: 输入数据，包含V形反转指标
            
        Returns:
            pd.DataFrame: 包含V形反转强度的数据框
        """
        if "v_reversal" not in data.columns:
            data = self.calculate(data)
        
        # 创建结果DataFrame的副本
        result = data.copy()
        
        # 初始化反转强度列
        result["reversal_strength"] = 0.0
        
        # 获取需要的列的值
        v_reversal_values = data["v_reversal"].values
        decline_rate_values = data["decline_rate"].values
        rebound_rate_values = data["rebound_rate"].values
        
        # 计算反转强度
        for i in range(len(data)):
            if v_reversal_values[i]:
                # 反转强度 = 下跌幅度与反弹幅度的乘积
                result.iloc[i, result.columns.get_loc("reversal_strength")] = (
                    abs(decline_rate_values[i]) * rebound_rate_values[i]
                ) * 100
        
        # 初始化反转分类列
        result["reversal_category"] = None
        
        # 根据强度阈值分类
        for i in range(len(result)):
            strength = result.iloc[i, result.columns.get_loc("reversal_strength")]
            
            if strength > 5:
                result.iloc[i, result.columns.get_loc("reversal_category")] = "强烈反转"
            elif strength > 2:
                result.iloc[i, result.columns.get_loc("reversal_category")] = "明显反转"
            elif strength > 0:
                result.iloc[i, result.columns.get_loc("reversal_category")] = "弱反转"
        
        return result
    
    def find_v_patterns(self, data: pd.DataFrame, window: int = 20) -> List[Tuple[int, str]]:
        """
        查找数据中的V形反转模式
        
        Args:
            data: 输入数据，包含价格数据
            window: 搜索窗口大小，默认为20日
            
        Returns:
            List[Tuple[int, str]]: V形反转位置及其类别的列表
        """
        if "v_reversal" not in data.columns:
            data = self.calculate(data)
        
        patterns = []
        
        # 获取所有强V形反转的位置
        v_indices = data.index[data["v_reversal"]].tolist()
        
        for idx in v_indices:
            # 获取对应的数据位置
            pos = data.index.get_loc(idx)
            
            # 检查是否有强度分类
            if "reversal_category" in data.columns and pd.notna(data.iloc[pos]["reversal_category"]):
                category = data.iloc[pos]["reversal_category"]
            else:
                category = "V形反转"
            
            # 添加到模式列表
            patterns.append((pos, category))
        
        return patterns 