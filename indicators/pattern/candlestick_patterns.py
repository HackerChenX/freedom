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
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化K线形态识别指标"""
        super().__init__(name="CandlestickPatterns", description="K线形态识别指标，识别各种单日和组合K线形态")
    
    def set_parameters(self, **kwargs):
        """
        设置指标参数
        """
        # K线形态识别通常没有可变参数，但为了符合接口要求，提供此方法
        pass
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        识别各种K线形态
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含各种K线形态的标记
        """
        # 确保数据包含必需的列
        required_columns = ["open", "high", "low", "close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据必须包含'{col}'列")
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算单日K线形态
        result = self._calculate_single_patterns(data, result)
        
        # 计算组合K线形态
        result = self._calculate_combined_patterns(data, result)
        
        # 计算复合形态（需要更多的历史数据）
        result = self._calculate_complex_patterns(data, result)

        # 确保所有形态列都存在（即使数据不足）
        all_pattern_names = [pattern.name.lower() for pattern in PatternType]
        for pattern_name in all_pattern_names:
            if pattern_name not in result.columns:
                result[pattern_name] = False
        
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
        result[PatternType.DOJI.name.lower()] = doji
        
        # 锤头线：小实体，长下影线，几乎无上影线
        hammer = (body_to_range_ratio < 0.3) & \
                (lower_shadow > 2 * body_size) & \
                (upper_shadow < 0.1 * (high_prices - low_prices))
        result[PatternType.HAMMER.name.lower()] = hammer
        
        # 吊颈线：小实体，长上影线，几乎无下影线
        hanging_man = (body_to_range_ratio < 0.3) & \
                      (upper_shadow > 2 * body_size) & \
                      (lower_shadow < 0.1 * (high_prices - low_prices))
        result[PatternType.HANGING_MAN.name.lower()] = hanging_man
        
        # 长腿十字：十字星带长下影线
        long_legged_doji = doji & (lower_shadow > 2 * upper_shadow) & \
                           (lower_shadow > 0.3 * (high_prices - low_prices))
        result[PatternType.LONG_LEGGED_DOJI.name.lower()] = long_legged_doji
        
        # 墓碑线：十字星带长上影线
        gravestone_doji = doji & (upper_shadow > 2 * lower_shadow) & \
                         (upper_shadow > 0.3 * (high_prices - low_prices))
        result[PatternType.GRAVESTONE_DOJI.name.lower()] = gravestone_doji
        
        # 射击之星：小实体，长上影线，短下影线
        shooting_star = (body_to_range_ratio < 0.3) & \
                        (upper_shadow > 2 * body_size) & \
                        (upper_shadow > 2 * lower_shadow)
        result[PatternType.SHOOTING_STAR.name.lower()] = shooting_star
        
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
        result[PatternType.ENGULFING_BULLISH.name.lower()] = engulfing_bullish
        result[PatternType.ENGULFING_BEARISH.name.lower()] = engulfing_bearish
        result[PatternType.DARK_CLOUD_COVER.name.lower()] = dark_cloud_cover
        result[PatternType.PIERCING_LINE.name.lower()] = piercing_line
        result[PatternType.MORNING_STAR.name.lower()] = morning_star
        result[PatternType.EVENING_STAR.name.lower()] = evening_star
        result[PatternType.HARAMI_BULLISH.name.lower()] = harami_bullish
        result[PatternType.SINGLE_NEEDLE_BOTTOM.name.lower()] = single_needle_bottom
        
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
        result[PatternType.HEAD_SHOULDERS_TOP.name.lower()] = head_shoulders_top
        result[PatternType.HEAD_SHOULDERS_BOTTOM.name.lower()] = head_shoulders_bottom
        result[PatternType.DOUBLE_TOP.name.lower()] = double_top
        result[PatternType.DOUBLE_BOTTOM.name.lower()] = double_bottom
        result[PatternType.ISLAND_REVERSAL.name.lower()] = island_reversal
        result[PatternType.V_REVERSAL.name.lower()] = v_reversal
        
        return result
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        识别所有已定义的形态，并以DataFrame形式返回

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含所有形态信号的DataFrame
        """
        return self.calculate(data, **kwargs)

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
        
        # 获取最近形成的形态
        patterns = {}
        for pattern_type in PatternType:
            pattern_name = pattern_type.name.lower()
            if pattern_name in recent_result.columns:
                patterns[pattern_name] = bool(recent_result[pattern_name].any())
        
        return patterns
    
    def calculate_raw_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算K线形态识别指标的原始评分
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含原始评分的DataFrame
        """
        # 计算指标值
        indicator_data = self.calculate(data)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. 看涨形态评分（+15到+40分）
        # 单日看涨形态
        if PatternType.HAMMER.name.lower() in indicator_data.columns:
            hammer_mask = indicator_data[PatternType.HAMMER.name.lower()]
            score.loc[hammer_mask] += 20
        
        if PatternType.LONG_LEGGED_DOJI.name.lower() in indicator_data.columns:
            long_legged_doji_mask = indicator_data[PatternType.LONG_LEGGED_DOJI.name.lower()]
            score.loc[long_legged_doji_mask] += 15
        
        if PatternType.SINGLE_NEEDLE_BOTTOM.name.lower() in indicator_data.columns:
            single_needle_mask = indicator_data[PatternType.SINGLE_NEEDLE_BOTTOM.name.lower()]
            score.loc[single_needle_mask] += 25
        
        # 组合看涨形态
        if PatternType.ENGULFING_BULLISH.name.lower() in indicator_data.columns:
            engulfing_bullish_mask = indicator_data[PatternType.ENGULFING_BULLISH.name.lower()]
            score.loc[engulfing_bullish_mask] += 30
        
        if PatternType.PIERCING_LINE.name.lower() in indicator_data.columns:
            piercing_line_mask = indicator_data[PatternType.PIERCING_LINE.name.lower()]
            score.loc[piercing_line_mask] += 25
        
        if PatternType.MORNING_STAR.name.lower() in indicator_data.columns:
            morning_star_mask = indicator_data[PatternType.MORNING_STAR.name.lower()]
            score.loc[morning_star_mask] += 35
        
        if PatternType.HARAMI_BULLISH.name.lower() in indicator_data.columns:
            harami_bullish_mask = indicator_data[PatternType.HARAMI_BULLISH.name.lower()]
            score.loc[harami_bullish_mask] += 20
        
        # 复合看涨形态
        if PatternType.HEAD_SHOULDERS_BOTTOM.name.lower() in indicator_data.columns:
            head_shoulders_bottom_mask = indicator_data[PatternType.HEAD_SHOULDERS_BOTTOM.name.lower()]
            score.loc[head_shoulders_bottom_mask] += 40
        
        if PatternType.DOUBLE_BOTTOM.name.lower() in indicator_data.columns:
            double_bottom_mask = indicator_data[PatternType.DOUBLE_BOTTOM.name.lower()]
            score.loc[double_bottom_mask] += 35
        
        if PatternType.V_REVERSAL.name.lower() in indicator_data.columns:
            v_reversal_mask = indicator_data[PatternType.V_REVERSAL.name.lower()]
            score.loc[v_reversal_mask] += 30
        
        # 2. 看跌形态评分（-15到-40分）
        # 单日看跌形态
        if PatternType.HANGING_MAN.name.lower() in indicator_data.columns:
            hanging_man_mask = indicator_data[PatternType.HANGING_MAN.name.lower()]
            score.loc[hanging_man_mask] -= 20
        
        if PatternType.GRAVESTONE_DOJI.name.lower() in indicator_data.columns:
            gravestone_doji_mask = indicator_data[PatternType.GRAVESTONE_DOJI.name.lower()]
            score.loc[gravestone_doji_mask] -= 15
        
        if PatternType.SHOOTING_STAR.name.lower() in indicator_data.columns:
            shooting_star_mask = indicator_data[PatternType.SHOOTING_STAR.name.lower()]
            score.loc[shooting_star_mask] -= 25
        
        # 组合看跌形态
        if PatternType.ENGULFING_BEARISH.name.lower() in indicator_data.columns:
            engulfing_bearish_mask = indicator_data[PatternType.ENGULFING_BEARISH.name.lower()]
            score.loc[engulfing_bearish_mask] -= 30
        
        if PatternType.DARK_CLOUD_COVER.name.lower() in indicator_data.columns:
            dark_cloud_mask = indicator_data[PatternType.DARK_CLOUD_COVER.name.lower()]
            score.loc[dark_cloud_mask] -= 25
        
        if PatternType.EVENING_STAR.name.lower() in indicator_data.columns:
            evening_star_mask = indicator_data[PatternType.EVENING_STAR.name.lower()]
            score.loc[evening_star_mask] -= 35
        
        # 复合看跌形态
        if PatternType.HEAD_SHOULDERS_TOP.name.lower() in indicator_data.columns:
            head_shoulders_top_mask = indicator_data[PatternType.HEAD_SHOULDERS_TOP.name.lower()]
            score.loc[head_shoulders_top_mask] -= 40
        
        if PatternType.DOUBLE_TOP.name.lower() in indicator_data.columns:
            double_top_mask = indicator_data[PatternType.DOUBLE_TOP.name.lower()]
            score.loc[double_top_mask] -= 35
        
        # 3. 中性形态评分（-5到+5分）
        if PatternType.DOJI.name.lower() in indicator_data.columns:
            doji_mask = indicator_data[PatternType.DOJI.name.lower()]
            # 十字星在不同位置有不同含义
            if 'close' in data.columns and len(data) >= 20:
                close_price = data['close']
                ma20 = close_price.rolling(window=20).mean()
                
                # 在上升趋势中的十字星偏空
                uptrend_doji = doji_mask & (close_price > ma20)
                score.loc[uptrend_doji] -= 5
                
                # 在下降趋势中的十字星偏多
                downtrend_doji = doji_mask & (close_price < ma20)
                score.loc[downtrend_doji] += 5
        
        # 4. 岛型反转特殊评分（±30分）
        if PatternType.ISLAND_REVERSAL.name.lower() in indicator_data.columns:
            island_reversal_mask = indicator_data[PatternType.ISLAND_REVERSAL.name.lower()]
            
            # 需要结合价格趋势判断岛型反转的方向
            if 'close' in data.columns and len(data) >= 5:
                close_price = data['close']
                price_change_5d = close_price.pct_change(5)
                
                # 在上升趋势后的岛型反转（看跌）
                bearish_island = island_reversal_mask & (price_change_5d > 0.05)
                score.loc[bearish_island] -= 30
                
                # 在下降趋势后的岛型反转（看涨）
                bullish_island = island_reversal_mask & (price_change_5d < -0.05)
                score.loc[bullish_island] += 30
        
        # 5. 形态强度调整（±10分）
        # 根据成交量确认形态强度
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            vol_ratio = volume / vol_ma5
            
            # 任何形态如果伴随放量，增强信号强度
            high_volume_mask = vol_ratio > 1.5
            
            # 看涨形态+放量
            bullish_patterns = (
                indicator_data.get(PatternType.HAMMER.name.lower(), False) |
                indicator_data.get(PatternType.ENGULFING_BULLISH.name.lower(), False) |
                indicator_data.get(PatternType.MORNING_STAR.name.lower(), False) |
                indicator_data.get(PatternType.DOUBLE_BOTTOM.name.lower(), False)
            )
            if isinstance(bullish_patterns, pd.Series):
                bullish_volume_confirm = bullish_patterns & high_volume_mask
                score.loc[bullish_volume_confirm] += 10
            
            # 看跌形态+放量
            bearish_patterns = (
                indicator_data.get(PatternType.HANGING_MAN.name.lower(), False) |
                indicator_data.get(PatternType.ENGULFING_BEARISH.name.lower(), False) |
                indicator_data.get(PatternType.EVENING_STAR.name.lower(), False) |
                indicator_data.get(PatternType.DOUBLE_TOP.name.lower(), False)
            )
            if isinstance(bearish_patterns, pd.Series):
                bearish_volume_confirm = bearish_patterns & high_volume_mask
                score.loc[bearish_volume_confirm] -= 10
        
        # 6. 形态位置调整（±15分）
        # 在关键技术位置的形态更重要
        if 'close' in data.columns and len(data) >= 60:
            close_price = data['close']
            
            # 计算支撑阻力位
            high_60 = close_price.rolling(window=60).max()
            low_60 = close_price.rolling(window=60).min()
            
            # 在阻力位附近的看跌形态
            near_resistance = close_price > high_60 * 0.95
            bearish_at_resistance = (
                (indicator_data.get(PatternType.HANGING_MAN.name.lower(), False) |
                 indicator_data.get(PatternType.EVENING_STAR.name.lower(), False) |
                 indicator_data.get(PatternType.SHOOTING_STAR.name.lower(), False)) &
                near_resistance
            )
            if isinstance(bearish_at_resistance, pd.Series):
                score.loc[bearish_at_resistance] -= 15
            
            # 在支撑位附近的看涨形态
            near_support = close_price < low_60 * 1.05
            bullish_at_support = (
                (indicator_data.get(PatternType.HAMMER.name.lower(), False) |
                 indicator_data.get(PatternType.MORNING_STAR.name.lower(), False) |
                 indicator_data.get(PatternType.SINGLE_NEEDLE_BOTTOM.name.lower(), False)) &
                near_support
            )
            if isinstance(bullish_at_support, pd.Series):
                score.loc[bullish_at_support] += 15
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return pd.DataFrame({'score': score}, index=data.index)
    
    def identify_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别最新的K线形态
        
        Args:
            data: 输入数据
            
        Returns:
            List[str]: 识别到的形态列表
        """
        # 计算K线形态
        result = self.calculate(data)
        
        # 获取最新的形态
        latest_patterns = {}
        for column in result.columns:
            if result[column].iloc[-1]:
                latest_patterns[column] = True
        
        return list(latest_patterns.keys())
        
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        根据识别到的K线形态生成交易信号
        
        Args:
            data: 输入数据，包含OHLC数据
            *args, **kwargs: 附加参数
            
        Returns:
            pd.DataFrame: 包含标准化信号的DataFrame
        """
        # 计算K线形态
        pattern_results = self.calculate(data)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True
        signals['trend'] = 0
        signals['score'] = 50
        signals['signal_type'] = ''
        signals['signal_desc'] = ''
        signals['confidence'] = 0
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = 0.0
        signals['market_env'] = '未知'
        signals['volume_confirmation'] = False
        
        # 定义看涨形态
        bullish_patterns = [
            PatternType.HAMMER.name.lower(),
            PatternType.MORNING_STAR.name.lower(),
            PatternType.PIERCING_LINE.name.lower(),
            PatternType.ENGULFING_BULLISH.name.lower(),
            PatternType.HARAMI_BULLISH.name.lower(),
            PatternType.SINGLE_NEEDLE_BOTTOM.name.lower(),
            PatternType.HEAD_SHOULDERS_BOTTOM.name.lower(),
            PatternType.DOUBLE_BOTTOM.name.lower(),
            PatternType.TRIANGLE_ASCENDING.name.lower(),
            PatternType.WEDGE_FALLING.name.lower(),
            PatternType.CUP_WITH_HANDLE.name.lower(),
            PatternType.V_REVERSAL.name.lower()
        ]
        
        # 定义看跌形态
        bearish_patterns = [
            PatternType.HANGING_MAN.name.lower(),
            PatternType.EVENING_STAR.name.lower(),
            PatternType.DARK_CLOUD_COVER.name.lower(),
            PatternType.ENGULFING_BEARISH.name.lower(),
            PatternType.SHOOTING_STAR.name.lower(),
            PatternType.HEAD_SHOULDERS_TOP.name.lower(),
            PatternType.DOUBLE_TOP.name.lower(),
            PatternType.TRIANGLE_DESCENDING.name.lower(),
            PatternType.WEDGE_RISING.name.lower()
        ]
        
        # 强看涨形态
        strong_bullish_patterns = [
            PatternType.MORNING_STAR.name.lower(),
            PatternType.ENGULFING_BULLISH.name.lower(),
            PatternType.DOUBLE_BOTTOM.name.lower(),
            PatternType.HEAD_SHOULDERS_BOTTOM.name.lower(),
            PatternType.V_REVERSAL.name.lower()
        ]
        
        # 强看跌形态
        strong_bearish_patterns = [
            PatternType.EVENING_STAR.name.lower(),
            PatternType.ENGULFING_BEARISH.name.lower(),
            PatternType.DOUBLE_TOP.name.lower(),
            PatternType.HEAD_SHOULDERS_TOP.name.lower()
        ]
        
        # 生成信号
        for i in range(len(data)):
            # 初始化信号描述
            pattern_desc = []
            
            # 检查看涨形态
            bullish_found = False
            for pattern in bullish_patterns:
                if pattern in pattern_results.columns and pattern_results[pattern].iloc[i]:
                    bullish_found = True
                    pattern_desc.append(pattern)
                    # 强看涨形态
                    if pattern in strong_bullish_patterns:
                        signals.loc[data.index[i], 'score'] = 75
                        signals.loc[data.index[i], 'confidence'] = 80
                    else:
                        signals.loc[data.index[i], 'score'] = 65
                        signals.loc[data.index[i], 'confidence'] = 70
            
            # 检查看跌形态
            bearish_found = False
            for pattern in bearish_patterns:
                if pattern in pattern_results.columns and pattern_results[pattern].iloc[i]:
                    bearish_found = True
                    pattern_desc.append(pattern)
                    # 强看跌形态
                    if pattern in strong_bearish_patterns:
                        signals.loc[data.index[i], 'score'] = 25
                        signals.loc[data.index[i], 'confidence'] = 80
                    else:
                        signals.loc[data.index[i], 'score'] = 35
                        signals.loc[data.index[i], 'confidence'] = 70
            
            # 设置信号标志
            if bullish_found and not bearish_found:
                signals.loc[data.index[i], 'buy_signal'] = True
                signals.loc[data.index[i], 'sell_signal'] = False
                signals.loc[data.index[i], 'neutral_signal'] = False
                signals.loc[data.index[i], 'trend'] = 1
                signals.loc[data.index[i], 'signal_type'] = '看涨形态'
            elif bearish_found and not bullish_found:
                signals.loc[data.index[i], 'buy_signal'] = False
                signals.loc[data.index[i], 'sell_signal'] = True
                signals.loc[data.index[i], 'neutral_signal'] = False
                signals.loc[data.index[i], 'trend'] = -1
                signals.loc[data.index[i], 'signal_type'] = '看跌形态'
            
            # 当出现多个信号时，可能有冲突
            if bullish_found and bearish_found:
                # 这种情况我们保持中性，但仍然记录形态
                signals.loc[data.index[i], 'neutral_signal'] = True
                signals.loc[data.index[i], 'score'] = 50
                signals.loc[data.index[i], 'signal_type'] = '混合形态'
            
            # 设置信号描述
            if pattern_desc:
                signals.loc[data.index[i], 'signal_desc'] = ', '.join(pattern_desc)
            
            # 设置止损位
            if bullish_found:
                # 设置在当前K线的最低点下方
                signals.loc[data.index[i], 'stop_loss'] = data['low'].iloc[i] * 0.98
                # 设置仓位
                signals.loc[data.index[i], 'position_size'] = 0.3 if signals.loc[data.index[i], 'score'] > 70 else 0.2
                # 设置风险级别
                signals.loc[data.index[i], 'risk_level'] = '低' if signals.loc[data.index[i], 'score'] > 70 else '中'
            elif bearish_found:
                # 设置在当前K线的最高点上方
                signals.loc[data.index[i], 'stop_loss'] = data['high'].iloc[i] * 1.02
                # 设置仓位
                signals.loc[data.index[i], 'position_size'] = 0.3 if signals.loc[data.index[i], 'score'] < 30 else 0.2
                # 设置风险级别
                signals.loc[data.index[i], 'risk_level'] = '低' if signals.loc[data.index[i], 'score'] < 30 else '中'
            
            # 分析市场环境
            if i >= 20:  # 需要一定的历史数据
                # 简单的趋势判断
                recent_trend = (data['close'].iloc[i] - data['close'].iloc[i-20]) / data['close'].iloc[i-20]
                if recent_trend > 0.05:
                    signals.loc[data.index[i], 'market_env'] = '上升趋势'
                elif recent_trend < -0.05:
                    signals.loc[data.index[i], 'market_env'] = '下降趋势'
                else:
                    signals.loc[data.index[i], 'market_env'] = '横盘整理'
        
        # 添加成交量确认
        if 'volume' in data.columns:
            for i in range(1, len(data)):
                if data['volume'].iloc[i] > data['volume'].iloc[i-1] * 1.2:  # 成交量放大20%
                    signals.loc[data.index[i], 'volume_confirmation'] = True
                    # 成交量确认增加信号置信度
                    signals.loc[data.index[i], 'confidence'] = min(100, signals.loc[data.index[i], 'confidence'] + 10)
        
        return signals

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算CandlestickPatterns指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于数据质量的置信度
        if hasattr(self, '_result') and self._result is not None:
            # 检查是否有形态数据
            pattern_columns = [col for col in self._result.columns
                             if any(pattern.name.lower() in col for pattern in PatternType)]
            if pattern_columns:
                # 形态数据越完整，置信度越高
                data_completeness = len(pattern_columns) / len(PatternType)
                confidence += data_completeness * 0.1

        # 3. 基于形态的置信度
        if not patterns.empty:
            # 检查CandlestickPatterns形态（只计算布尔列）
            bool_columns = patterns.select_dtypes(include=[bool]).columns
            if len(bool_columns) > 0:
                pattern_count = patterns[bool_columns].sum().sum()
                if pattern_count > 0:
                    confidence += min(pattern_count * 0.02, 0.15)

        # 4. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.05, 0.1)

        # 5. 基于数据长度的置信度
        if len(score) >= 60:  # 两个月数据
            confidence += 0.1
        elif len(score) >= 30:  # 一个月数据
            confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def register_patterns(self):
        """
        注册CandlestickPatterns指标的形态到全局形态注册表
        """
        # 注册单日看涨形态
        self.register_pattern_to_registry(
            pattern_id="HAMMER",
            display_name="锤头线",
            description="小实体，长下影线，几乎无上影线，底部反转信号",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0
        )

        self.register_pattern_to_registry(
            pattern_id="LONG_LEGGED_DOJI",
            display_name="长腿十字",
            description="十字星带长下影线，表明买卖力量均衡但下方有支撑",
            pattern_type="BULLISH",
            default_strength="WEAK",
            score_impact=15.0
        )

        self.register_pattern_to_registry(
            pattern_id="SINGLE_NEEDLE_BOTTOM",
            display_name="单针探底",
            description="长下影线，表明下方有强力买盘支撑",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0
        )

        # 注册单日看跌形态
        self.register_pattern_to_registry(
            pattern_id="HANGING_MAN",
            display_name="吊颈线",
            description="小实体，长上影线，几乎无下影线，顶部反转信号",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-20.0
        )

        self.register_pattern_to_registry(
            pattern_id="GRAVESTONE_DOJI",
            display_name="墓碑线",
            description="十字星带长上影线，表明上方抛压沉重",
            pattern_type="BEARISH",
            default_strength="WEAK",
            score_impact=-15.0
        )

        self.register_pattern_to_registry(
            pattern_id="SHOOTING_STAR",
            display_name="射击之星",
            description="小实体，长上影线，短下影线，顶部反转信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0
        )

        # 注册组合看涨形态
        self.register_pattern_to_registry(
            pattern_id="ENGULFING_BULLISH",
            display_name="阳包阴",
            description="阳线完全包含前一天阴线，强烈的底部反转信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0
        )

        self.register_pattern_to_registry(
            pattern_id="PIERCING_LINE",
            display_name="曙光初现",
            description="阴线后接长阳线，阳线开盘价低于前日最低价",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0
        )

        self.register_pattern_to_registry(
            pattern_id="MORNING_STAR",
            display_name="启明星",
            description="长阴线+十字星+长阳线，经典的底部反转形态",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=35.0
        )

        self.register_pattern_to_registry(
            pattern_id="HARAMI_BULLISH",
            display_name="好友反攻",
            description="长阴线后第二天以低于前日收盘价开盘，收于前日开盘价之上",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0
        )

        # 注册组合看跌形态
        self.register_pattern_to_registry(
            pattern_id="ENGULFING_BEARISH",
            display_name="阴包阳",
            description="阴线完全包含前一天阳线，强烈的顶部反转信号",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-30.0
        )

        self.register_pattern_to_registry(
            pattern_id="DARK_CLOUD_COVER",
            display_name="乌云盖顶",
            description="阳线后接长阴线，阴线开盘价高于前日最高价",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0
        )

        self.register_pattern_to_registry(
            pattern_id="EVENING_STAR",
            display_name="黄昏星",
            description="长阳线+十字星+长阴线，经典的顶部反转形态",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-35.0
        )

        # 注册复合形态
        self.register_pattern_to_registry(
            pattern_id="HEAD_SHOULDERS_BOTTOM",
            display_name="头肩底",
            description="三个波谷，中间低于两侧，强烈的底部反转形态",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=40.0
        )

        self.register_pattern_to_registry(
            pattern_id="HEAD_SHOULDERS_TOP",
            display_name="头肩顶",
            description="三个波峰，中间高于两侧，强烈的顶部反转形态",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-40.0
        )

        self.register_pattern_to_registry(
            pattern_id="DOUBLE_BOTTOM",
            display_name="双底",
            description="W形价格形态，强烈的底部反转信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=35.0
        )

        self.register_pattern_to_registry(
            pattern_id="DOUBLE_TOP",
            display_name="双顶",
            description="M形价格形态，强烈的顶部反转信号",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-35.0
        )

        self.register_pattern_to_registry(
            pattern_id="V_REVERSAL",
            display_name="V形反转",
            description="急速下跌后快速反弹，快速反转形态",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0
        )

        self.register_pattern_to_registry(
            pattern_id="ISLAND_REVERSAL",
            display_name="岛型反转",
            description="跳空+反向跳空形成孤岛，强烈的反转信号",
            pattern_type="NEUTRAL",
            default_strength="VERY_STRONG",
            score_impact=30.0
        )

        # 注册中性形态
        self.register_pattern_to_registry(
            pattern_id="DOJI",
            display_name="十字星",
            description="开盘价与收盘价接近，上下影线明显，表明市场犹豫",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> dict:
        """
        生成CandlestickPatterns交易信号

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            dict: 包含买卖信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate(data, **kwargs)

        if self._result is None or self._result.empty:
            return {
                'buy_signal': pd.Series(False, index=data.index),
                'sell_signal': pd.Series(False, index=data.index),
                'signal_strength': pd.Series(0.0, index=data.index)
            }

        # 初始化信号
        buy_signal = pd.Series(False, index=data.index)
        sell_signal = pd.Series(False, index=data.index)
        signal_strength = pd.Series(0.0, index=data.index)

        # 定义看涨形态
        bullish_patterns = [
            PatternType.HAMMER.name.lower(),
            PatternType.MORNING_STAR.name.lower(),
            PatternType.PIERCING_LINE.name.lower(),
            PatternType.ENGULFING_BULLISH.name.lower(),
            PatternType.HARAMI_BULLISH.name.lower(),
            PatternType.SINGLE_NEEDLE_BOTTOM.name.lower(),
            PatternType.HEAD_SHOULDERS_BOTTOM.name.lower(),
            PatternType.DOUBLE_BOTTOM.name.lower(),
            PatternType.V_REVERSAL.name.lower()
        ]

        # 定义看跌形态
        bearish_patterns = [
            PatternType.HANGING_MAN.name.lower(),
            PatternType.EVENING_STAR.name.lower(),
            PatternType.DARK_CLOUD_COVER.name.lower(),
            PatternType.ENGULFING_BEARISH.name.lower(),
            PatternType.SHOOTING_STAR.name.lower(),
            PatternType.HEAD_SHOULDERS_TOP.name.lower(),
            PatternType.DOUBLE_TOP.name.lower()
        ]

        # 强形态权重
        strong_patterns = {
            PatternType.MORNING_STAR.name.lower(): 0.9,
            PatternType.EVENING_STAR.name.lower(): -0.9,
            PatternType.ENGULFING_BULLISH.name.lower(): 0.8,
            PatternType.ENGULFING_BEARISH.name.lower(): -0.8,
            PatternType.HEAD_SHOULDERS_BOTTOM.name.lower(): 0.9,
            PatternType.HEAD_SHOULDERS_TOP.name.lower(): -0.9,
            PatternType.DOUBLE_BOTTOM.name.lower(): 0.8,
            PatternType.DOUBLE_TOP.name.lower(): -0.8
        }

        # 生成信号
        for pattern in bullish_patterns:
            if pattern in self._result.columns:
                pattern_mask = self._result[pattern]
                buy_signal |= pattern_mask

                # 设置信号强度
                if pattern in strong_patterns:
                    signal_strength[pattern_mask] = strong_patterns[pattern]
                else:
                    signal_strength[pattern_mask] = 0.6

        for pattern in bearish_patterns:
            if pattern in self._result.columns:
                pattern_mask = self._result[pattern]
                sell_signal |= pattern_mask

                # 设置信号强度
                if pattern in strong_patterns:
                    signal_strength[pattern_mask] = strong_patterns[pattern]
                else:
                    signal_strength[pattern_mask] = -0.6

        # 处理岛型反转（需要结合趋势判断）
        if PatternType.ISLAND_REVERSAL.name.lower() in self._result.columns:
            island_mask = self._result[PatternType.ISLAND_REVERSAL.name.lower()]
            if island_mask.any() and len(data) >= 5:
                # 简单趋势判断
                price_change_5d = data['close'].pct_change(5)

                # 在上升趋势后的岛型反转（看跌）
                bearish_island = island_mask & (price_change_5d > 0.05)
                sell_signal |= bearish_island
                signal_strength[bearish_island] = -0.8

                # 在下降趋势后的岛型反转（看涨）
                bullish_island = island_mask & (price_change_5d < -0.05)
                buy_signal |= bullish_island
                signal_strength[bullish_island] = 0.8

        # 标准化信号强度
        signal_strength = signal_strength.clip(-1, 1)

        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'signal_strength': signal_strength
        }

    def get_indicator_type(self) -> str:
        """
        获取指标类型

        Returns:
            str: 指标类型
        """
        return "CANDLESTICKPATTERNS"