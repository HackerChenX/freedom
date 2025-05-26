"""
ZXM洗盘形态识别模块

实现ZXM体系中的洗盘形态识别功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class WashPlateType(Enum):
    """洗盘形态类型枚举"""
    SHOCK_WASH = "横盘震荡洗盘"           # 一定区间内的来回震荡，成交量忽大忽小
    PULLBACK_WASH = "回调洗盘"            # 短期快速回调后在重要支撑位止跌，成交量逐步萎缩
    FALSE_BREAK_WASH = "假突破洗盘"        # 向下突破重要支撑位后快速收复，突破时量能放大，收复时量能更大
    TIME_WASH = "时间洗盘"                # 价格小幅波动，但周期较长，整体呈萎缩趋势
    CONTINUOUS_YIN_WASH = "连续阴线洗盘"   # 连续3-5根中小阴线，实体不断缩小，下影线增多，量能逐步萎缩


class ZXMWashPlate(BaseIndicator):
    """
    ZXM洗盘形态识别指标
    
    识别ZXM体系中各种洗盘形态
    """
    
    def __init__(self):
        """初始化ZXM洗盘形态识别指标"""
        super().__init__(name="ZXMWashPlate", description="ZXM洗盘形态识别指标，识别ZXM体系中各种洗盘形态")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        识别ZXM洗盘形态
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含各种洗盘形态的标记
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low", "close", "volume"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算横盘震荡洗盘
        result = self._calculate_shock_wash(data, result)
        
        # 计算回调洗盘
        result = self._calculate_pullback_wash(data, result)
        
        # 计算假突破洗盘
        result = self._calculate_false_break_wash(data, result)
        
        # 计算时间洗盘
        result = self._calculate_time_wash(data, result)
        
        # 计算连续阴线洗盘
        result = self._calculate_continuous_yin_wash(data, result)
        
        return result
    
    def _calculate_shock_wash(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算横盘震荡洗盘
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
            
        公式说明：
        ZXM_SHOCK_WASH:=(HHV(CLOSE,10)-LLV(CLOSE,10))/LLV(CLOSE,10)<0.07 AND HHV(VOL,10)/LLV(VOL,10)>2;
        """
        # 提取数据
        close = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        shock_wash = np.zeros(n, dtype=bool)
        
        # 计算横盘震荡洗盘
        window = 10
        for i in range(window, n):
            # 价格区间相对波动小于7%
            price_range = np.max(close[i-window:i]) - np.min(close[i-window:i])
            price_range_ratio = price_range / np.min(close[i-window:i])
            
            # 成交量波动大于2倍
            vol_range_ratio = np.max(volume[i-window:i]) / np.min(volume[i-window:i])
            
            if price_range_ratio < 0.07 and vol_range_ratio > 2:
                shock_wash[i] = True
        
        # 添加到结果
        result[WashPlateType.SHOCK_WASH.value] = shock_wash
        
        return result
    
    def _calculate_pullback_wash(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算回调洗盘
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
            
        公式说明：
        ZXM_PULLBACK_WASH:=REF(CLOSE>OPEN,1) AND CLOSE<OPEN AND LOW>MA(CLOSE,20)*0.97 AND VOL<REF(VOL,1);
        """
        # 提取数据
        open_prices = data["open"].values
        low_prices = data["low"].values
        close_prices = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        pullback_wash = np.zeros(n, dtype=bool)
        
        # 计算20日均线
        ma20 = np.zeros(n)
        for i in range(20, n):
            ma20[i] = np.mean(close_prices[i-20:i])
        
        # 计算回调洗盘
        for i in range(1, n):
            # 前一日阳线，当日阴线
            prev_bullish = close_prices[i-1] > open_prices[i-1]
            current_bearish = close_prices[i] < open_prices[i]
            
            # 当日最低价高于20日均线的97%
            above_support = i >= 20 and low_prices[i] > ma20[i] * 0.97
            
            # 当日成交量小于前日
            vol_decreasing = volume[i] < volume[i-1]
            
            if prev_bullish and current_bearish and above_support and vol_decreasing:
                pullback_wash[i] = True
        
        # 添加到结果
        result[WashPlateType.PULLBACK_WASH.value] = pullback_wash
        
        return result
    
    def _calculate_false_break_wash(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算假突破洗盘
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
            
        公式说明：
        ZXM_FALSE_BREAK:=REF(LOW<LLV(LOW,20),1) AND CLOSE>REF(CLOSE,1) AND CLOSE>REF(LOW,1)*1.02 AND VOL>REF(VOL,1);
        """
        # 提取数据
        low_prices = data["low"].values
        close_prices = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        false_break = np.zeros(n, dtype=bool)
        
        # 计算假突破洗盘
        window = 20
        for i in range(window+1, n):
            # 前一日最低价低于20日最低价
            prev_low_break = low_prices[i-1] < np.min(low_prices[i-window-1:i-1])
            
            # 当日收盘价高于前日收盘价
            price_recovery = close_prices[i] > close_prices[i-1]
            
            # 当日收盘价高于前日最低价的1.02倍
            strong_recovery = close_prices[i] > low_prices[i-1] * 1.02
            
            # 当日成交量大于前日
            vol_increasing = volume[i] > volume[i-1]
            
            if prev_low_break and price_recovery and strong_recovery and vol_increasing:
                false_break[i] = True
        
        # 添加到结果
        result[WashPlateType.FALSE_BREAK_WASH.value] = false_break
        
        return result
    
    def _calculate_time_wash(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算时间洗盘
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 提取数据
        close_prices = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        time_wash = np.zeros(n, dtype=bool)
        
        # 计算时间洗盘
        # 特征：价格小幅波动，成交量整体萎缩，周期较长（15天以上）
        window = 15
        for i in range(2*window, n):
            # 当前区间价格波动小
            current_range = np.max(close_prices[i-window:i]) - np.min(close_prices[i-window:i])
            current_range_ratio = current_range / np.min(close_prices[i-window:i])
            
            # 前一区间价格波动相对较大
            prev_range = np.max(close_prices[i-2*window:i-window]) - np.min(close_prices[i-2*window:i-window])
            prev_range_ratio = prev_range / np.min(close_prices[i-2*window:i-window])
            
            # 当前区间成交量整体萎缩
            current_vol_avg = np.mean(volume[i-window:i])
            prev_vol_avg = np.mean(volume[i-2*window:i-window])
            vol_shrinking = current_vol_avg < 0.8 * prev_vol_avg
            
            if current_range_ratio < 0.05 and current_range_ratio < prev_range_ratio and vol_shrinking:
                time_wash[i] = True
        
        # 添加到结果
        result[WashPlateType.TIME_WASH.value] = time_wash
        
        return result
    
    def _calculate_continuous_yin_wash(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算连续阴线洗盘
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
            
        公式说明：
        ZXM_CONTINUOUS_YIN:=COUNT(CLOSE<OPEN,5)>=3 AND COUNT(MIN(OPEN,CLOSE)-LOW>ABS(CLOSE-OPEN),5)>=2 AND VOL/REF(VOL,5)<0.8;
        """
        # 提取数据
        open_prices = data["open"].values
        low_prices = data["low"].values
        close_prices = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        continuous_yin = np.zeros(n, dtype=bool)
        
        # 计算连续阴线洗盘
        window = 5
        for i in range(window, n):
            # 最近5天中至少有3天是阴线
            bearish_count = np.sum(close_prices[i-window:i] < open_prices[i-window:i])
            
            # 最近5天中至少有2天下影线长于实体
            lower_shadow_count = 0
            for j in range(i-window, i):
                body_size = abs(close_prices[j] - open_prices[j])
                lower_shadow = min(open_prices[j], close_prices[j]) - low_prices[j]
                if lower_shadow > body_size:
                    lower_shadow_count += 1
            
            # 当日成交量低于5日前的80%
            vol_shrinking = volume[i] < 0.8 * volume[i-window]
            
            if bearish_count >= 3 and lower_shadow_count >= 2 and vol_shrinking:
                continuous_yin[i] = True
        
        # 添加到结果
        result[WashPlateType.CONTINUOUS_YIN_WASH.value] = continuous_yin
        
        return result
    
    def get_recent_wash_plates(self, data: pd.DataFrame, lookback: int = 10) -> Dict[str, bool]:
        """
        获取最近的洗盘形态
        
        Args:
            data: 输入数据
            lookback: 回溯天数
            
        Returns:
            Dict[str, bool]: 最近的洗盘形态
        """
        # 计算所有洗盘形态
        result = self.calculate(data)
        
        # 截取最近的数据
        recent_result = result.iloc[-lookback:]
        
        # 提取每种形态的最新状态
        recent_wash_plates = {}
        for wash_type in WashPlateType:
            wash_name = wash_type.value
            if wash_name in recent_result.columns:
                recent_wash_plates[wash_name] = recent_result[wash_name].any()
        
        return recent_wash_plates 