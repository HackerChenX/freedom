#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pattern Detector - 形态检测器

提供通用的技术指标形态检测算法，用于识别各种常见的技术形态
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from utils.logger import get_logger

logger = get_logger(__name__)


def detect_crossover(series1: pd.Series, series2: pd.Series, 
                    lookback: int = 1) -> bool:
    """
    检测序列1是否从下方上穿序列2
    
    Args:
        series1: 序列1
        series2: 序列2
        lookback: 回溯周期数
        
    Returns:
        bool: 是否发生上穿
    """
    if len(series1) < lookback + 1 or len(series2) < lookback + 1:
        return False
    
    # 检查当前值关系
    current_above = series1.iloc[-1] > series2.iloc[-1]
    
    # 检查前一周期值关系
    prev_above = False
    for i in range(1, lookback + 1):
        if series1.iloc[-1-i] <= series2.iloc[-1-i]:
            prev_above = False
            break
        prev_above = True
    
    # 如果当前在上方，且前一周期在下方，则发生上穿
    return current_above and not prev_above


def detect_crossunder(series1: pd.Series, series2: pd.Series, 
                     lookback: int = 1) -> bool:
    """
    检测序列1是否从上方下穿序列2
    
    Args:
        series1: 序列1
        series2: 序列2
        lookback: 回溯周期数
        
    Returns:
        bool: 是否发生下穿
    """
    if len(series1) < lookback + 1 or len(series2) < lookback + 1:
        return False
    
    # 检查当前值关系
    current_below = series1.iloc[-1] < series2.iloc[-1]
    
    # 检查前一周期值关系
    prev_below = False
    for i in range(1, lookback + 1):
        if series1.iloc[-1-i] >= series2.iloc[-1-i]:
            prev_below = False
            break
        prev_below = True
    
    # 如果当前在下方，且前一周期在上方，则发生下穿
    return current_below and not prev_below


def detect_value_crossover(series: pd.Series, value: float, 
                          lookback: int = 1) -> bool:
    """
    检测序列是否从下方上穿某个值
    
    Args:
        series: 序列
        value: 阈值
        lookback: 回溯周期数
        
    Returns:
        bool: 是否发生上穿
    """
    if len(series) < lookback + 1:
        return False
    
    # 检查当前值关系
    current_above = series.iloc[-1] > value
    
    # 检查前一周期值关系
    prev_above = True
    for i in range(1, lookback + 1):
        if series.iloc[-1-i] <= value:
            prev_above = False
            break
    
    # 如果当前在上方，且前一周期在下方，则发生上穿
    return current_above and not prev_above


def detect_value_crossunder(series: pd.Series, value: float, 
                           lookback: int = 1) -> bool:
    """
    检测序列是否从上方下穿某个值
    
    Args:
        series: 序列
        value: 阈值
        lookback: 回溯周期数
        
    Returns:
        bool: 是否发生下穿
    """
    if len(series) < lookback + 1:
        return False
    
    # 检查当前值关系
    current_below = series.iloc[-1] < value
    
    # 检查前一周期值关系
    prev_below = True
    for i in range(1, lookback + 1):
        if series.iloc[-1-i] >= value:
            prev_below = False
            break
    
    # 如果当前在下方，且前一周期在上方，则发生下穿
    return current_below and not prev_below


def detect_divergence(price: pd.Series, indicator: pd.Series, 
                     bullish: bool = True, 
                     lookback: int = 20,
                     min_distance: int = 5) -> bool:
    """
    检测价格和指标之间是否存在背离
    
    Args:
        price: 价格序列
        indicator: 指标序列
        bullish: 是否为底背离（True为底背离，False为顶背离）
        lookback: 回溯周期数
        min_distance: 两个峰/谷之间的最小距离
        
    Returns:
        bool: 是否存在背离
    """
    if len(price) < lookback or len(indicator) < lookback:
        return False
    
    # 截取回溯窗口的数据
    price_window = price.iloc[-lookback:]
    indicator_window = indicator.iloc[-lookback:]
    
    if bullish:
        # 底背离：价格创新低，但指标未创新低
        # 寻找价格的低点
        price_min_idx = []
        indicator_min_idx = []
        
        # 使用简单的方法查找局部最小值
        for i in range(1, len(price_window) - 1):
            if (price_window.iloc[i] < price_window.iloc[i-1] and 
                price_window.iloc[i] < price_window.iloc[i+1]):
                price_min_idx.append(i)
                
            if (indicator_window.iloc[i] < indicator_window.iloc[i-1] and 
                indicator_window.iloc[i] < indicator_window.iloc[i+1]):
                indicator_min_idx.append(i)
        
        # 至少需要两个低点才能形成背离
        if len(price_min_idx) < 2 or len(indicator_min_idx) < 2:
            return False
        
        # 检查两个最近的价格低点
        recent_price_mins = sorted(price_min_idx, reverse=True)[:2]
        if len(recent_price_mins) < 2:
            return False
            
        # 确保两个低点之间有足够的距离
        if abs(recent_price_mins[0] - recent_price_mins[1]) < min_distance:
            return False
            
        # 价格创新低
        if price_window.iloc[recent_price_mins[0]] >= price_window.iloc[recent_price_mins[1]]:
            return False
            
        # 找到对应时间的指标值
        recent_indicator_values = [indicator_window.iloc[i] for i in recent_price_mins]
        
        # 指标未创新低（形成底背离）
        return recent_indicator_values[0] > recent_indicator_values[1]
        
    else:
        # 顶背离：价格创新高，但指标未创新高
        # 寻找价格的高点
        price_max_idx = []
        indicator_max_idx = []
        
        # 使用简单的方法查找局部最大值
        for i in range(1, len(price_window) - 1):
            if (price_window.iloc[i] > price_window.iloc[i-1] and 
                price_window.iloc[i] > price_window.iloc[i+1]):
                price_max_idx.append(i)
                
            if (indicator_window.iloc[i] > indicator_window.iloc[i-1] and 
                indicator_window.iloc[i] > indicator_window.iloc[i+1]):
                indicator_max_idx.append(i)
        
        # 至少需要两个高点才能形成背离
        if len(price_max_idx) < 2 or len(indicator_max_idx) < 2:
            return False
        
        # 检查两个最近的价格高点
        recent_price_maxs = sorted(price_max_idx, reverse=True)[:2]
        if len(recent_price_maxs) < 2:
            return False
            
        # 确保两个高点之间有足够的距离
        if abs(recent_price_maxs[0] - recent_price_maxs[1]) < min_distance:
            return False
            
        # 价格创新高
        if price_window.iloc[recent_price_maxs[0]] <= price_window.iloc[recent_price_maxs[1]]:
            return False
            
        # 找到对应时间的指标值
        recent_indicator_values = [indicator_window.iloc[i] for i in recent_price_maxs]
        
        # 指标未创新高（形成顶背离）
        return recent_indicator_values[0] < recent_indicator_values[1]


def detect_overbought(series: pd.Series, threshold: float = 70, 
                     consecutive: int = 1) -> bool:
    """
    检测指标是否处于超买状态
    
    Args:
        series: 指标序列
        threshold: 超买阈值
        consecutive: 连续超买的周期数
        
    Returns:
        bool: 是否处于超买状态
    """
    if len(series) < consecutive:
        return False
    
    # 检查最近几个周期是否都超过阈值
    return all(value > threshold for value in series.iloc[-consecutive:])


def detect_oversold(series: pd.Series, threshold: float = 30, 
                   consecutive: int = 1) -> bool:
    """
    检测指标是否处于超卖状态
    
    Args:
        series: 指标序列
        threshold: 超卖阈值
        consecutive: 连续超卖的周期数
        
    Returns:
        bool: 是否处于超卖状态
    """
    if len(series) < consecutive:
        return False
    
    # 检查最近几个周期是否都低于阈值
    return all(value < threshold for value in series.iloc[-consecutive:])


def detect_trend(series: pd.Series, window: int = 5, 
               threshold: float = 0.0, up: bool = True) -> bool:
    """
    检测序列是否处于趋势状态
    
    Args:
        series: 序列
        window: 窗口大小
        threshold: 趋势确认阈值
        up: 是否为上升趋势
        
    Returns:
        bool: 是否处于趋势状态
    """
    if len(series) < window:
        return False
    
    # 计算窗口内的变化率
    pct_change = series.pct_change(periods=window).iloc[-1]
    
    # 根据趋势方向判断
    if up:
        return pct_change > threshold
    else:
        return pct_change < -threshold


def detect_consolidation(series: pd.Series, window: int = 5, 
                       threshold: float = 0.03) -> bool:
    """
    检测序列是否处于盘整状态
    
    Args:
        series: 序列
        window: 窗口大小
        threshold: 盘整确认阈值
        
    Returns:
        bool: 是否处于盘整状态
    """
    if len(series) < window:
        return False
    
    # 获取窗口内的数据
    window_data = series.iloc[-window:]
    
    # 计算最大值和最小值
    max_value = window_data.max()
    min_value = window_data.min()
    
    # 计算波动范围
    if min_value == 0:
        return False
        
    range_pct = (max_value - min_value) / min_value
    
    # 如果波动范围小于阈值，则认为是盘整
    return range_pct < threshold


def detect_volume_spike(volume: pd.Series, window: int = 5, 
                       threshold: float = 2.0) -> bool:
    """
    检测成交量是否出现放量
    
    Args:
        volume: 成交量序列
        window: 窗口大小
        threshold: 放量确认阈值
        
    Returns:
        bool: 是否出现放量
    """
    if len(volume) < window + 1:
        return False
    
    # 计算最近一个周期的成交量
    current_volume = volume.iloc[-1]
    
    # 计算窗口内的平均成交量
    avg_volume = volume.iloc[-window-1:-1].mean()
    
    # 如果当前成交量超过平均成交量的threshold倍，则认为是放量
    return current_volume > avg_volume * threshold


def detect_volume_shrink(volume: pd.Series, window: int = 5, 
                        threshold: float = 0.5) -> bool:
    """
    检测成交量是否出现缩量
    
    Args:
        volume: 成交量序列
        window: 窗口大小
        threshold: 缩量确认阈值
        
    Returns:
        bool: 是否出现缩量
    """
    if len(volume) < window + 1:
        return False
    
    # 计算最近一个周期的成交量
    current_volume = volume.iloc[-1]
    
    # 计算窗口内的平均成交量
    avg_volume = volume.iloc[-window-1:-1].mean()
    
    # 如果当前成交量低于平均成交量的threshold倍，则认为是缩量
    return current_volume < avg_volume * threshold


def detect_breakout(price: pd.Series, window: int = 20, 
                  threshold: float = 0.03, up: bool = True) -> bool:
    """
    检测价格是否突破
    
    Args:
        price: 价格序列
        window: 窗口大小
        threshold: 突破确认阈值
        up: 是否为向上突破
        
    Returns:
        bool: 是否发生突破
    """
    if len(price) < window + 1:
        return False
    
    # 计算窗口内的最高价和最低价
    if up:
        # 向上突破
        resistance = price.iloc[-window-1:-1].max()
        return price.iloc[-1] > resistance * (1 + threshold)
    else:
        # 向下突破
        support = price.iloc[-window-1:-1].min()
        return price.iloc[-1] < support * (1 - threshold)


def detect_ma_alignment(ma_short: pd.Series, ma_mid: pd.Series, 
                       ma_long: pd.Series, up: bool = True) -> bool:
    """
    检测均线是否多头/空头排列
    
    Args:
        ma_short: 短期均线
        ma_mid: 中期均线
        ma_long: 长期均线
        up: 是否为多头排列
        
    Returns:
        bool: 是否形成多头/空头排列
    """
    if len(ma_short) < 1 or len(ma_mid) < 1 or len(ma_long) < 1:
        return False
    
    # 获取最新值
    short_val = ma_short.iloc[-1]
    mid_val = ma_mid.iloc[-1]
    long_val = ma_long.iloc[-1]
    
    # 判断排列
    if up:
        # 多头排列：短期均线 > 中期均线 > 长期均线
        return short_val > mid_val > long_val
    else:
        # 空头排列：短期均线 < 中期均线 < 长期均线
        return short_val < mid_val < long_val


def detect_double_top(price: pd.Series, window: int = 30, 
                     threshold: float = 0.03) -> bool:
    """
    检测价格是否形成双顶形态
    
    Args:
        price: 价格序列
        window: 窗口大小
        threshold: 顶部偏差阈值
        
    Returns:
        bool: 是否形成双顶
    """
    if len(price) < window:
        return False
    
    # 获取窗口内的数据
    window_data = price.iloc[-window:]
    
    # 查找局部最大值
    peaks = []
    for i in range(1, len(window_data) - 1):
        if (window_data.iloc[i] > window_data.iloc[i-1] and 
            window_data.iloc[i] > window_data.iloc[i+1]):
            peaks.append((i, window_data.iloc[i]))
    
    # 至少需要两个峰才能形成双顶
    if len(peaks) < 2:
        return False
    
    # 取最高的两个峰
    top_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:2]
    
    # 确保两个峰之间有足够的距离
    idx1, val1 = top_peaks[0]
    idx2, val2 = top_peaks[1]
    
    if abs(idx1 - idx2) < 5:  # 至少间隔5个周期
        return False
    
    # 检查两个峰的高度是否接近
    height_diff = abs(val1 - val2) / val1
    
    # 如果两个峰的高度接近，则认为是双顶
    return height_diff < threshold


def detect_double_bottom(price: pd.Series, window: int = 30, 
                        threshold: float = 0.03) -> bool:
    """
    检测价格是否形成双底形态
    
    Args:
        price: 价格序列
        window: 窗口大小
        threshold: 底部偏差阈值
        
    Returns:
        bool: 是否形成双底
    """
    if len(price) < window:
        return False
    
    # 获取窗口内的数据
    window_data = price.iloc[-window:]
    
    # 查找局部最小值
    troughs = []
    for i in range(1, len(window_data) - 1):
        if (window_data.iloc[i] < window_data.iloc[i-1] and 
            window_data.iloc[i] < window_data.iloc[i+1]):
            troughs.append((i, window_data.iloc[i]))
    
    # 至少需要两个谷才能形成双底
    if len(troughs) < 2:
        return False
    
    # 取最低的两个谷
    bottom_troughs = sorted(troughs, key=lambda x: x[1])[:2]
    
    # 确保两个谷之间有足够的距离
    idx1, val1 = bottom_troughs[0]
    idx2, val2 = bottom_troughs[1]
    
    if abs(idx1 - idx2) < 5:  # 至少间隔5个周期
        return False
    
    # 检查两个谷的高度是否接近
    if val1 == 0:
        return False
        
    height_diff = abs(val1 - val2) / val1
    
    # 如果两个谷的高度接近，则认为是双底
    return height_diff < threshold


def detect_head_and_shoulders(price: pd.Series, window: int = 50, 
                             threshold: float = 0.1) -> bool:
    """
    检测价格是否形成头肩顶形态
    
    Args:
        price: 价格序列
        window: 窗口大小
        threshold: 肩部高度偏差阈值
        
    Returns:
        bool: 是否形成头肩顶
    """
    if len(price) < window:
        return False
    
    # 获取窗口内的数据
    window_data = price.iloc[-window:]
    
    # 查找局部最大值
    peaks = []
    for i in range(1, len(window_data) - 1):
        if (window_data.iloc[i] > window_data.iloc[i-1] and 
            window_data.iloc[i] > window_data.iloc[i+1]):
            peaks.append((i, window_data.iloc[i]))
    
    # 至少需要三个峰才能形成头肩顶
    if len(peaks) < 3:
        return False
    
    # 按时间顺序排序
    peaks.sort(key=lambda x: x[0])
    
    # 检查是否有三个峰，且中间的峰最高
    if len(peaks) >= 3:
        for i in range(len(peaks) - 2):
            left_idx, left_val = peaks[i]
            head_idx, head_val = peaks[i+1]
            right_idx, right_val = peaks[i+2]
            
            # 确保头部高于两个肩部
            if head_val > left_val and head_val > right_val:
                # 检查两个肩部的高度是否接近
                shoulder_diff = abs(left_val - right_val) / left_val
                
                # 确保两个肩部之间的距离合适
                if shoulder_diff < threshold and right_idx - left_idx > 10:
                    return True
    
    return False


def detect_inverse_head_and_shoulders(price: pd.Series, window: int = 50, 
                                    threshold: float = 0.1) -> bool:
    """
    检测价格是否形成头肩底形态
    
    Args:
        price: 价格序列
        window: 窗口大小
        threshold: 肩部高度偏差阈值
        
    Returns:
        bool: 是否形成头肩底
    """
    if len(price) < window:
        return False
    
    # 获取窗口内的数据
    window_data = price.iloc[-window:]
    
    # 查找局部最小值
    troughs = []
    for i in range(1, len(window_data) - 1):
        if (window_data.iloc[i] < window_data.iloc[i-1] and 
            window_data.iloc[i] < window_data.iloc[i+1]):
            troughs.append((i, window_data.iloc[i]))
    
    # 至少需要三个谷才能形成头肩底
    if len(troughs) < 3:
        return False
    
    # 按时间顺序排序
    troughs.sort(key=lambda x: x[0])
    
    # 检查是否有三个谷，且中间的谷最低
    if len(troughs) >= 3:
        for i in range(len(troughs) - 2):
            left_idx, left_val = troughs[i]
            head_idx, head_val = troughs[i+1]
            right_idx, right_val = troughs[i+2]
            
            # 确保头部低于两个肩部
            if head_val < left_val and head_val < right_val:
                # 检查两个肩部的高度是否接近
                if left_val == 0:
                    return False
                    
                shoulder_diff = abs(left_val - right_val) / left_val
                
                # 确保两个肩部之间的距离合适
                if shoulder_diff < threshold and right_idx - left_idx > 10:
                    return True
    
    return False 