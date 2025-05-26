#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日内波动率指标模块

实现日内波动率计算功能，用于评估价格日内波动幅度与开盘价的关系
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class IntradayVolatility(BaseIndicator):
    """
    日内波动率指标
    
    计算日内波动范围与开盘价的比值，评估市场波动性
    """
    
    def __init__(self, smooth_period: int = 5):
        """
        初始化日内波动率指标
        
        Args:
            smooth_period: 平滑周期，默认为5日
        """
        super().__init__(name="IntradayVolatility", description="日内波动率指标，评估价格日内波动幅度")
        self.smooth_period = smooth_period
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算日内波动率指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含日内波动率指标值
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算日内波动范围
        intraday_range = data["high"] - data["low"]
        
        # 计算相对于开盘价的波动率
        result["volatility"] = intraday_range / data["open"] * 100
        
        # 计算波动率的移动平均
        result["volatility_ma"] = result["volatility"].rolling(window=self.smooth_period).mean()
        
        # 计算相对波动率（当前波动率与平均波动率的比值）
        result["relative_volatility"] = result["volatility"] / result["volatility_ma"]
        
        # 计算波动率变化率
        result["volatility_change"] = result["volatility"].pct_change()
        
        return result
    
    def get_signals(self, data: pd.DataFrame, high_threshold: float = 1.5, 
                   low_threshold: float = 0.5) -> pd.DataFrame:
        """
        生成波动率信号
        
        Args:
            data: 输入数据，包含日内波动率指标
            high_threshold: 高波动率阈值，默认为1.5
            low_threshold: 低波动率阈值，默认为0.5
            
        Returns:
            pd.DataFrame: 包含波动率信号的数据框
        """
        if "volatility" not in data.columns:
            data = self.calculate(data)
        
        # 初始化波动率状态列
        data["volatility_state"] = "正常"
        
        # 高波动状态
        high_volatility = data["relative_volatility"] > high_threshold
        data.loc[high_volatility, "volatility_state"] = "高波动"
        
        # 低波动状态
        low_volatility = data["relative_volatility"] < low_threshold
        data.loc[low_volatility, "volatility_state"] = "低波动"
        
        # 初始化信号列
        data["volatility_signal"] = 0
        
        # 使用向量化操作计算信号
        # 计算当前和前一天的高波动条件
        current_high = data["volatility"] > data["volatility_ma"] * high_threshold
        prev_high = current_high.shift(1).fillna(False)
        
        # 计算当前和前一天的低波动条件
        current_low = data["volatility"] < data["volatility_ma"] * low_threshold
        prev_low = current_low.shift(1).fillna(False)
        
        # 波动率突然上升：当前高波动但前一天不是
        up_signal = current_high & ~prev_high
        
        # 波动率突然下降：当前低波动但前一天不是
        down_signal = current_low & ~prev_low
        
        # 应用信号
        data.loc[up_signal, "volatility_signal"] = 1
        data.loc[down_signal, "volatility_signal"] = -1
        
        return data
    
    def get_volatility_trend(self, data: pd.DataFrame, trend_period: int = 10) -> pd.DataFrame:
        """
        分析波动率趋势
        
        Args:
            data: 输入数据，包含日内波动率指标
            trend_period: 趋势分析周期，默认为10日
            
        Returns:
            pd.DataFrame: 包含波动率趋势的数据框
        """
        if "volatility" not in data.columns:
            data = self.calculate(data)
        
        # 计算波动率趋势 - 使用rolling应用线性回归
        def calc_slope(y):
            if len(y) < trend_period or y.isna().any():
                return np.nan
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        # 使用向量化操作计算趋势斜率
        data["volatility_trend"] = data["volatility"].rolling(window=trend_period).apply(
            calc_slope, raw=True
        )
        
        # 使用向量化操作对趋势进行分类
        data["trend_category"] = pd.NA
        
        # 根据斜率值分类趋势
        trend = data["volatility_trend"]
        data.loc[trend > 0.1, "trend_category"] = "强上升"
        data.loc[(trend > 0.01) & (trend <= 0.1), "trend_category"] = "上升"
        data.loc[(trend >= -0.01) & (trend <= 0.01), "trend_category"] = "平稳"
        data.loc[(trend < -0.01) & (trend >= -0.1), "trend_category"] = "下降"
        data.loc[trend < -0.1, "trend_category"] = "强下降"
        
        return data
    
    def get_market_phase(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        分析市场阶段
        
        Args:
            data: 输入数据，包含日内波动率指标
            
        Returns:
            pd.DataFrame: 包含市场阶段的数据框
        """
        if "volatility" not in data.columns:
            data = self.calculate(data)
        
        # 初始化市场阶段列
        data["market_phase"] = np.nan
        
        # 结合波动率和价格趋势分析市场阶段
        # 计算价格短期趋势（20日移动平均线方向）
        if "close" in data.columns:
            data["price_ma20"] = data["close"].rolling(window=20).mean()
            data["price_trend"] = np.nan
            
            for i in range(20, len(data)):
                if data["close"].iloc[i] > data["price_ma20"].iloc[i]:
                    data.iloc[i, data.columns.get_loc("price_trend")] = 1  # 上升趋势
                else:
                    data.iloc[i, data.columns.get_loc("price_trend")] = -1  # 下降趋势
            
            # 基于波动率和价格趋势确定市场阶段
            for i in range(20, len(data)):
                if pd.notna(data["price_trend"].iloc[i]) and pd.notna(data["relative_volatility"].iloc[i]):
                    price_trend = data["price_trend"].iloc[i]
                    rel_vol = data["relative_volatility"].iloc[i]
                    
                    if price_trend > 0 and rel_vol > 1.2:
                        data.iloc[i, data.columns.get_loc("market_phase")] = "强势上涨"
                    elif price_trend > 0 and rel_vol < 0.8:
                        data.iloc[i, data.columns.get_loc("market_phase")] = "稳步上涨"
                    elif price_trend < 0 and rel_vol > 1.2:
                        data.iloc[i, data.columns.get_loc("market_phase")] = "恐慌下跌"
                    elif price_trend < 0 and rel_vol < 0.8:
                        data.iloc[i, data.columns.get_loc("market_phase")] = "缓慢下跌"
                    else:
                        data.iloc[i, data.columns.get_loc("market_phase")] = "盘整"
        
        return data 