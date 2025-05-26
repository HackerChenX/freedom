"""
K线形态识别模块

实现单日和组合K线形态的识别功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class PatternType(Enum):
    """K线形态类型枚举"""
    # 单日K线形态
    DOJI = "十字星"                  # 开盘价与收盘价接近，上下影线明显
    HAMMER = "锤头线"                # 小实体，长下影线，几乎无上影线
    HANGING_MAN = "吊颈线"           # 小实体，长上影线，几乎无下影线
    LONG_LEGGED_DOJI = "长腿十字"    # 十字星带长下影线
    GRAVESTONE_DOJI = "墓碑线"       # 十字星带长上影线
    SHOOTING_STAR = "射击之星"        # 小实体，长上影线，短下影线
    
    # 组合K线形态
    ENGULFING_BULLISH = "阳包阴"      # 阳线完全包含前一天阴线
    ENGULFING_BEARISH = "阴包阳"      # 阴线完全包含前一天阳线
    DARK_CLOUD_COVER = "乌云盖顶"     # 阳线后接长阴线，阴线开盘价高于前日最高价
    PIERCING_LINE = "曙光初现"        # 阴线后接长阳线，阳线开盘价低于前日最低价
    MORNING_STAR = "启明星"           # 长阴线+十字星+长阳线
    EVENING_STAR = "黄昏星"           # 长阳线+十字星+长阴线
    HARAMI_BULLISH = "好友反攻"       # 长阴线后第二天以低于前日收盘价开盘，收于前日开盘价之上
    SINGLE_NEEDLE_BOTTOM = "单针探底" # 长下影线，表明下方有买盘支撑
    
    # 复合形态
    HEAD_SHOULDERS_TOP = "头肩顶"     # 三个波峰，中间高于两侧
    HEAD_SHOULDERS_BOTTOM = "头肩底"  # 三个波谷，中间低于两侧
    DOUBLE_TOP = "双顶"               # M形价格形态
    DOUBLE_BOTTOM = "双底"            # W形价格形态
    TRIANGLE_ASCENDING = "上升三角形" # 水平上轨+上升下轨
    TRIANGLE_DESCENDING = "下降三角形" # 下降上轨+水平下轨
    TRIANGLE_SYMMETRICAL = "对称三角形" # 上轨下降+下轨上升
    RECTANGLE = "矩形整理"            # 价格在水平支撑压力间震荡
    FLAG_BULLISH = "牛旗形"           # 上升趋势中的小幅调整
    FLAG_BEARISH = "熊旗形"           # 下降趋势中的小幅反弹
    WEDGE_RISING = "上升楔形"         # 上升通道，逐渐收窄
    WEDGE_FALLING = "下降楔形"        # 下降通道，逐渐收窄
    CUP_WITH_HANDLE = "杯柄形态"      # U形底部+小幅回调形成柄部
    ISLAND_REVERSAL = "岛型反转"      # 跳空+反向跳空形成孤岛
    V_REVERSAL = "V形反转"            # 急速下跌后快速反弹


class CandlestickPatterns(BaseIndicator):
    """
    K线形态识别指标
    
    识别各种单日和组合K线形态
    """
    
    def __init__(self):
        """初始化K线形态识别指标"""
        super().__init__(name="CandlestickPatterns", description="K线形态识别指标，识别各种单日和组合K线形态")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        识别各种K线形态
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含各种K线形态的标记
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low", "close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算单日K线形态
        result = self._calculate_single_patterns(data, result)
        
        # 计算组合K线形态
        result = self._calculate_combined_patterns(data, result)
        
        # 计算复合形态（需要更多的历史数据）
        result = self._calculate_complex_patterns(data, result)
        
        return result
    
    def _calculate_single_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算单日K线形态
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 提取数据
        open_prices = data["open"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values
        
        # 计算实体大小和影线长度
        body_size = np.abs(close_prices - open_prices)
        body_to_range_ratio = body_size / (high_prices - low_prices)
        upper_shadow = high_prices - np.maximum(close_prices, open_prices)
        lower_shadow = np.minimum(close_prices, open_prices) - low_prices
        
        # 十字星：开盘价与收盘价接近，上下影线明显
        doji = body_to_range_ratio < 0.1
        result[PatternType.DOJI.value] = doji
        
        # 锤头线：小实体，长下影线，几乎无上影线
        hammer = (body_to_range_ratio < 0.3) & \
                (lower_shadow > 2 * body_size) & \
                (upper_shadow < 0.1 * (high_prices - low_prices))
        result[PatternType.HAMMER.value] = hammer
        
        # 吊颈线：小实体，长上影线，几乎无下影线
        hanging_man = (body_to_range_ratio < 0.3) & \
                      (upper_shadow > 2 * body_size) & \
                      (lower_shadow < 0.1 * (high_prices - low_prices))
        result[PatternType.HANGING_MAN.value] = hanging_man
        
        # 长腿十字：十字星带长下影线
        long_legged_doji = doji & (lower_shadow > 2 * upper_shadow) & \
                           (lower_shadow > 0.3 * (high_prices - low_prices))
        result[PatternType.LONG_LEGGED_DOJI.value] = long_legged_doji
        
        # 墓碑线：十字星带长上影线
        gravestone_doji = doji & (upper_shadow > 2 * lower_shadow) & \
                         (upper_shadow > 0.3 * (high_prices - low_prices))
        result[PatternType.GRAVESTONE_DOJI.value] = gravestone_doji
        
        # 射击之星：小实体，长上影线，短下影线
        shooting_star = (body_to_range_ratio < 0.3) & \
                        (upper_shadow > 2 * body_size) & \
                        (upper_shadow > 2 * lower_shadow)
        result[PatternType.SHOOTING_STAR.value] = shooting_star
        
        return result
    
    def _calculate_combined_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算组合K线形态
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 提取数据
        open_prices = data["open"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values
        
        # 计算K线涨跌
        bullish = close_prices > open_prices
        bearish = close_prices < open_prices
        
        # 初始化结果数组
        n = len(data)
        engulfing_bullish = np.zeros(n, dtype=bool)
        engulfing_bearish = np.zeros(n, dtype=bool)
        dark_cloud_cover = np.zeros(n, dtype=bool)
        piercing_line = np.zeros(n, dtype=bool)
        morning_star = np.zeros(n, dtype=bool)
        evening_star = np.zeros(n, dtype=bool)
        harami_bullish = np.zeros(n, dtype=bool)
        single_needle_bottom = np.zeros(n, dtype=bool)
        
        # 计算组合K线形态
        for i in range(2, n):
            # 阳包阴：当日阳线，前日阴线，当日开盘价低于前日收盘价，当日收盘价高于前日开盘价
            if bullish[i] and bearish[i-1] and \
               open_prices[i] <= close_prices[i-1] and \
               close_prices[i] >= open_prices[i-1]:
                engulfing_bullish[i] = True
            
            # 阴包阳：当日阴线，前日阳线，当日开盘价高于前日收盘价，当日收盘价低于前日开盘价
            if bearish[i] and bullish[i-1] and \
               open_prices[i] >= close_prices[i-1] and \
               close_prices[i] <= open_prices[i-1]:
                engulfing_bearish[i] = True
            
            # 乌云盖顶：前日阳线，当日阴线，当日开盘价高于前日最高价，当日收盘价位于前日实体中部以下
            if bearish[i] and bullish[i-1] and \
               open_prices[i] > high_prices[i-1] and \
               close_prices[i] < (open_prices[i-1] + close_prices[i-1]) / 2 and \
               close_prices[i] > open_prices[i-1]:
                dark_cloud_cover[i] = True
            
            # 曙光初现：前日阴线，当日阳线，当日开盘价低于前日最低价，当日收盘价位于前日实体中部以上
            if bullish[i] and bearish[i-1] and \
               open_prices[i] < low_prices[i-1] and \
               close_prices[i] > (open_prices[i-1] + close_prices[i-1]) / 2 and \
               close_prices[i] < close_prices[i-1]:
                piercing_line[i] = True
            
            # 启明星：三日K线组合，第一日阴线，第二日十字星，第三日阳线
            if i >= 3 and \
               bearish[i-2] and \
               abs(close_prices[i-1] - open_prices[i-1]) < 0.1 * (high_prices[i-1] - low_prices[i-1]) and \
               bullish[i] and \
               close_prices[i] > (open_prices[i-2] + close_prices[i-2]) / 2:
                morning_star[i] = True
            
            # 黄昏星：三日K线组合，第一日阳线，第二日十字星，第三日阴线
            if i >= 3 and \
               bullish[i-2] and \
               abs(close_prices[i-1] - open_prices[i-1]) < 0.1 * (high_prices[i-1] - low_prices[i-1]) and \
               bearish[i] and \
               close_prices[i] < (open_prices[i-2] + close_prices[i-2]) / 2:
                evening_star[i] = True
            
            # 好友反攻：前日阴线，当日阳线，当日开盘价低于前日收盘价，当日收盘价高于前日开盘价
            if bullish[i] and bearish[i-1] and \
               open_prices[i] < close_prices[i-1] and \
               close_prices[i] > open_prices[i-1]:
                harami_bullish[i] = True
            
            # 单针探底：长下影线，表明下方有买盘支撑
            lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]
            body_size = abs(close_prices[i] - open_prices[i])
            if lower_shadow > 2 * body_size and \
               lower_shadow > 0.6 * (high_prices[i] - low_prices[i]):
                single_needle_bottom[i] = True
        
        # 添加到结果
        result[PatternType.ENGULFING_BULLISH.value] = engulfing_bullish
        result[PatternType.ENGULFING_BEARISH.value] = engulfing_bearish
        result[PatternType.DARK_CLOUD_COVER.value] = dark_cloud_cover
        result[PatternType.PIERCING_LINE.value] = piercing_line
        result[PatternType.MORNING_STAR.value] = morning_star
        result[PatternType.EVENING_STAR.value] = evening_star
        result[PatternType.HARAMI_BULLISH.value] = harami_bullish
        result[PatternType.SINGLE_NEEDLE_BOTTOM.value] = single_needle_bottom
        
        return result
    
    def _calculate_complex_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算复合形态
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 提取数据
        n = len(data)
        if n < 30:  # 复合形态需要更多数据
            return result
            
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values
        
        # 初始化结果数组
        head_shoulders_top = np.zeros(n, dtype=bool)
        head_shoulders_bottom = np.zeros(n, dtype=bool)
        double_top = np.zeros(n, dtype=bool)
        double_bottom = np.zeros(n, dtype=bool)
        island_reversal = np.zeros(n, dtype=bool)
        v_reversal = np.zeros(n, dtype=bool)
        
        # 计算复合形态
        window = 20  # 形态识别窗口
        
        for i in range(window, n):
            # 头肩顶：三个波峰，中间高于两侧
            if i >= 2*window:
                left_window = high_prices[i-2*window:i-window]
                middle_window = high_prices[i-window:i]
                left_peak_idx = np.argmax(left_window)
                middle_peak_idx = np.argmax(middle_window)
                
                if middle_peak_idx > 2 and middle_peak_idx < window-3:  # 确保中间峰在中间位置
                    left_peak = left_window[left_peak_idx]
                    middle_peak = middle_window[middle_peak_idx]
                    
                    if middle_peak > left_peak and middle_peak > high_prices[i]:
                        # 检查颈线
                        neckline = min(low_prices[i-2*window+left_peak_idx], low_prices[i-window+middle_peak_idx])
                        if close_prices[i] < neckline:
                            head_shoulders_top[i] = True
            
            # 头肩底：三个波谷，中间低于两侧
            if i >= 2*window:
                left_window = low_prices[i-2*window:i-window]
                middle_window = low_prices[i-window:i]
                left_trough_idx = np.argmin(left_window)
                middle_trough_idx = np.argmin(middle_window)
                
                if middle_trough_idx > 2 and middle_trough_idx < window-3:  # 确保中间谷在中间位置
                    left_trough = left_window[left_trough_idx]
                    middle_trough = middle_window[middle_trough_idx]
                    
                    if middle_trough < left_trough and middle_trough < low_prices[i]:
                        # 检查颈线
                        neckline = max(high_prices[i-2*window+left_trough_idx], high_prices[i-window+middle_trough_idx])
                        if close_prices[i] > neckline:
                            head_shoulders_bottom[i] = True
            
            # 双顶：两个相近的高点，中间有明显的低点
            high_window = high_prices[i-window:i]
            if len(high_window) == window:
                # 找出窗口内的两个最高点
                sorted_idx = np.argsort(high_window)
                highest_idx = sorted_idx[-1]
                second_highest_idx = sorted_idx[-2]
                
                # 确保两个高点相隔一定距离，且高度相近
                if abs(highest_idx - second_highest_idx) > 3 and \
                   abs(high_window[highest_idx] - high_window[second_highest_idx]) / high_window[highest_idx] < 0.03:
                    # 找出两个高点之间的低点
                    between_low = np.min(high_window[min(highest_idx, second_highest_idx):max(highest_idx, second_highest_idx)])
                    
                    # 当前价格低于中间低点，确认双顶
                    if close_prices[i] < between_low:
                        double_top[i] = True
            
            # 双底：两个相近的低点，中间有明显的高点
            low_window = low_prices[i-window:i]
            if len(low_window) == window:
                # 找出窗口内的两个最低点
                sorted_idx = np.argsort(low_window)
                lowest_idx = sorted_idx[0]
                second_lowest_idx = sorted_idx[1]
                
                # 确保两个低点相隔一定距离，且高度相近
                if abs(lowest_idx - second_lowest_idx) > 3 and \
                   abs(low_window[lowest_idx] - low_window[second_lowest_idx]) / low_window[lowest_idx] < 0.03:
                    # 找出两个低点之间的高点
                    between_high = np.max(low_window[min(lowest_idx, second_lowest_idx):max(lowest_idx, second_lowest_idx)])
                    
                    # 当前价格高于中间高点，确认双底
                    if close_prices[i] > between_high:
                        double_bottom[i] = True
            
            # 岛型反转：向上跳空后又向下跳空，或向下跳空后又向上跳空
            if i >= 2:
                # 向上跳空后向下跳空（顶部岛型反转）
                if low_prices[i-1] > high_prices[i-2] and high_prices[i] < low_prices[i-1]:
                    island_reversal[i] = True
                # 向下跳空后向上跳空（底部岛型反转）
                elif high_prices[i-1] < low_prices[i-2] and low_prices[i] > high_prices[i-1]:
                    island_reversal[i] = True
            
            # V形反转：急速下跌后快速反弹
            if i >= 5:
                # 计算前5天的跌幅
                drop_pct = (close_prices[i-5] - low_prices[i-1]) / close_prices[i-5]
                # 计算当日的涨幅
                rise_pct = (close_prices[i] - low_prices[i-1]) / low_prices[i-1]
                
                if drop_pct > 0.05 and rise_pct > 0.03:
                    v_reversal[i] = True
        
        # 添加到结果
        result[PatternType.HEAD_SHOULDERS_TOP.value] = head_shoulders_top
        result[PatternType.HEAD_SHOULDERS_BOTTOM.value] = head_shoulders_bottom
        result[PatternType.DOUBLE_TOP.value] = double_top
        result[PatternType.DOUBLE_BOTTOM.value] = double_bottom
        result[PatternType.ISLAND_REVERSAL.value] = island_reversal
        result[PatternType.V_REVERSAL.value] = v_reversal
        
        return result
    
    def get_latest_patterns(self, data: pd.DataFrame, lookback: int = 5) -> Dict[str, bool]:
        """
        获取最近形成的K线形态
        
        Args:
            data: 输入数据
            lookback: 回溯天数
            
        Returns:
            Dict[str, bool]: 最近形成的K线形态
        """
        # 计算所有K线形态
        result = self.calculate(data)
        
        # 截取最近的数据
        recent_result = result.iloc[-lookback:]
        
        # 提取每种形态的最新状态
        latest_patterns = {}
        for pattern in PatternType:
            pattern_name = pattern.value
            if pattern_name in recent_result.columns:
                latest_patterns[pattern_name] = recent_result[pattern_name].any()
        
        return latest_patterns 