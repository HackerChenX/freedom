#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
指数平均数指标(EMV)模块

实现指数平均数指标计算功能，用于评估价格上涨下跌的难易程度
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class EMV(BaseIndicator):
    """
    指数平均数指标(Ease of Movement Value)
    
    评估价格上涨下跌的难易程度，结合价格变动和成交量进行分析
    """
    
    def __init__(self, period: int = 14, ma_period: int = 9):
        """
        初始化EMV指标
        
        Args:
            period: EMV计算周期，默认为14日
            ma_period: EMV均线周期，默认为9日
        """
        super().__init__(name="EMV", description="指数平均数指标，评估价格上涨下跌的难易程度")
        self.period = period
        self.ma_period = ma_period
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算EMV指标
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含EMV指标值
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["high", "low", "volume"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算中间价
        midpoint_price = (data["high"] + data["low"]) / 2
        
        # 计算中间价的变化
        midpoint_move = midpoint_price - midpoint_price.shift(1)
        
        # 计算价格区间
        price_range = data["high"] - data["low"]
        
        # 计算成交量调整值（使用成交量的标准化）
        volume_adjusted = data["volume"] / 10000
        
        # 计算单日EMV值
        emv_daily = (midpoint_move * price_range) / volume_adjusted
        
        # 计算N日EMV值
        result["emv"] = emv_daily.rolling(window=self.period).sum()
        
        # 计算EMV均线
        result["emv_ma"] = result["emv"].rolling(window=self.ma_period).mean()
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成EMV信号
        
        Args:
            data: 输入数据，包含EMV指标
            
        Returns:
            pd.DataFrame: 包含EMV信号的数据框
        """
        if "emv" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["emv_signal"] = 0
        data["emv_ma_cross"] = 0
        
        # 使用向量化操作生成EMV穿越零轴信号
        emv = data["emv"]
        emv_prev = emv.shift(1)
        
        # EMV上穿零轴：买入信号
        zero_cross_up = (emv > 0) & (emv_prev <= 0)
        data.loc[zero_cross_up, "emv_signal"] = 1
        
        # EMV下穿零轴：卖出信号
        zero_cross_down = (emv < 0) & (emv_prev >= 0)
        data.loc[zero_cross_down, "emv_signal"] = -1
        
        # 生成EMV与EMV均线交叉信号
        emv_ma = data["emv_ma"]
        emv_ma_prev = emv_ma.shift(1)
        
        # EMV上穿均线：买入信号
        ma_cross_up = (emv > emv_ma) & (emv_prev <= emv_ma_prev)
        data.loc[ma_cross_up, "emv_ma_cross"] = 1
        
        # EMV下穿均线：卖出信号
        ma_cross_down = (emv < emv_ma) & (emv_prev >= emv_ma_prev)
        data.loc[ma_cross_down, "emv_ma_cross"] = -1
        
        # 添加短期EMV均线用于双均线系统
        data["emv_ma_short"] = data["emv"].rolling(window=self.ma_period//2).mean()
        
        # 生成短期均线与长期均线交叉信号
        data["emv_dual_ma_signal"] = 0
        
        if "emv_ma_short" in data.columns:
            short_ma = data["emv_ma_short"]
            long_ma = data["emv_ma"]
            short_ma_prev = short_ma.shift(1)
            long_ma_prev = long_ma.shift(1)
            
            # 短期均线上穿长期均线：买入信号
            dual_cross_up = (short_ma > long_ma) & (short_ma_prev <= long_ma_prev)
            data.loc[dual_cross_up, "emv_dual_ma_signal"] = 1
            
            # 短期均线下穿长期均线：卖出信号
            dual_cross_down = (short_ma < long_ma) & (short_ma_prev >= long_ma_prev)
            data.loc[dual_cross_down, "emv_dual_ma_signal"] = -1
        
        # 综合信号：结合零轴穿越、均线交叉和双均线系统
        data["emv_combined_signal"] = 0
        
        # 买入信号：至少有两个信号同时为买入
        buy_signals = (data["emv_signal"] == 1).astype(int) + \
                      (data["emv_ma_cross"] == 1).astype(int) + \
                      (data["emv_dual_ma_signal"] == 1).astype(int)
        
        data.loc[buy_signals >= 2, "emv_combined_signal"] = 1
        
        # 卖出信号：至少有两个信号同时为卖出
        sell_signals = (data["emv_signal"] == -1).astype(int) + \
                       (data["emv_ma_cross"] == -1).astype(int) + \
                       (data["emv_dual_ma_signal"] == -1).astype(int)
        
        data.loc[sell_signals >= 2, "emv_combined_signal"] = -1
        
        return data
    
    def get_market_efficiency(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        评估市场效率
        
        Args:
            data: 输入数据，包含EMV指标
            
        Returns:
            pd.DataFrame: 包含市场效率评估的数据框
        """
        if "emv" not in data.columns:
            data = self.calculate(data)
        
        # 初始化市场效率列
        data["market_efficiency"] = np.nan
        
        # 计算EMV的绝对值均值作为效率度量
        emv_abs_mean = data["emv"].abs().rolling(window=self.period).mean()
        
        # 归一化至0-100范围，便于理解
        max_val = emv_abs_mean.max()
        min_val = emv_abs_mean.min()
        
        if max_val != min_val:  # 避免除以零
            normalized_efficiency = 100 * (emv_abs_mean - min_val) / (max_val - min_val)
            data["market_efficiency"] = normalized_efficiency
        
        # 使用向量化操作进行市场效率分类
        data["efficiency_category"] = pd.NA
        
        # 根据效率值分类
        efficiency = data["market_efficiency"]
        data.loc[efficiency > 80, "efficiency_category"] = "高效"
        data.loc[(efficiency > 50) & (efficiency <= 80), "efficiency_category"] = "中效"
        data.loc[(efficiency > 20) & (efficiency <= 50), "efficiency_category"] = "低效"
        data.loc[(efficiency <= 20) & efficiency.notna(), "efficiency_category"] = "无效"
        
        return data

