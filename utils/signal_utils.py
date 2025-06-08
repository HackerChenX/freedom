#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
信号处理工具模块

提供信号处理和识别相关的功能函数
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from scipy.signal import find_peaks as scipy_find_peaks

def find_peaks(series: Union[pd.Series, np.ndarray],
               height: Optional[Union[float, np.ndarray]] = None,
               threshold: Optional[Union[float, np.ndarray]] = None,
               distance: Optional[int] = None,
               prominence: Optional[Union[float, np.ndarray]] = None,
               width: Optional[Union[float, np.ndarray]] = None,
               wlen: Optional[int] = None,
               rel_height: Optional[float] = 0.5) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    在序列中寻找峰值 (scipy.signal.find_peaks的封装)

    Args:
        series: 包含数据的一维序列
        height: 峰值的最小高度
        threshold: 峰值与其邻近样本之间的最小垂直距离
        distance: 峰值之间的最小水平距离（样本数）
        prominence: 峰值的最小凸起程度
        width: 峰值的最小宽度
        wlen: 用于计算峰值凸起度的窗口大小
        rel_height: 用于计算峰值宽度的相对高度

    Returns:
        Tuple[np.ndarray, Dict[str, np.ndarray]]:
            - 峰值索引数组
            - 包含峰值属性的字典
    """
    if isinstance(series, pd.Series):
        series = series.values

    # 确保series是一维的
    if series.ndim > 1:
        raise ValueError("输入序列必须是一维的")

    # 调用scipy的find_peaks函数
    peaks, properties = scipy_find_peaks(
        series,
        height=height,
        threshold=threshold,
        distance=distance,
        prominence=prominence,
        width=width,
        wlen=wlen,
        rel_height=rel_height
    )
    return peaks, properties

def detect_cross_over(series1: np.ndarray, series2: np.ndarray) -> List[int]:
    """
    检测两个序列的交叉点
    
    Args:
        series1: 第一个序列
        series2: 第二个序列
        
    Returns:
        List[int]: 交叉点位置列表，正值表示上穿，负值表示下穿
    """
    if len(series1) != len(series2):
        raise ValueError("两个序列长度必须相同")
    
    cross_points = []
    for i in range(1, len(series1)):
        # 上穿
        if series1[i-1] < series2[i-1] and series1[i] >= series2[i]:
            cross_points.append(i)
        # 下穿
        elif series1[i-1] > series2[i-1] and series1[i] <= series2[i]:
            cross_points.append(-i)
    
    return cross_points

def detect_golden_cross(fast_ma: np.ndarray, slow_ma: np.ndarray) -> List[int]:
    """
    检测金叉信号
    
    Args:
        fast_ma: 快速移动平均线
        slow_ma: 慢速移动平均线
        
    Returns:
        List[int]: 金叉位置列表
    """
    cross_points = detect_cross_over(fast_ma, slow_ma)
    return [i for i in cross_points if i > 0]  # 只返回上穿点

def detect_death_cross(fast_ma: np.ndarray, slow_ma: np.ndarray) -> List[int]:
    """
    检测死叉信号
    
    Args:
        fast_ma: 快速移动平均线
        slow_ma: 慢速移动平均线
        
    Returns:
        List[int]: 死叉位置列表
    """
    cross_points = detect_cross_over(fast_ma, slow_ma)
    return [-i for i in cross_points if i < 0]  # 只返回下穿点，并转为正值

def detect_trend_change(series: np.ndarray, window: int = 5) -> List[int]:
    """
    检测趋势变化点
    
    Args:
        series: 数据序列
        window: 窗口大小
        
    Returns:
        List[int]: 趋势变化点位置列表，正值表示上升趋势开始，负值表示下降趋势开始
    """
    if len(series) <= window:
        return []
    
    change_points = []
    # 计算窗口内的线性回归斜率
    for i in range(window, len(series)):
        window_data = series[i-window:i+1]
        x = np.arange(len(window_data))
        slope = np.polyfit(x, window_data, 1)[0]
        
        # 判断前一个窗口的斜率
        if i > window:
            prev_window_data = series[i-window-1:i]
            prev_x = np.arange(len(prev_window_data))
            prev_slope = np.polyfit(prev_x, prev_window_data, 1)[0]
            
            # 趋势变化判断
            if prev_slope <= 0 and slope > 0:
                change_points.append(i)  # 上升趋势开始
            elif prev_slope >= 0 and slope < 0:
                change_points.append(-i)  # 下降趋势开始
    
    return change_points

def detect_overbought(series: np.ndarray, threshold: float = 70) -> List[int]:
    """
    检测超买信号
    
    Args:
        series: 数据序列（如RSI）
        threshold: 超买阈值
        
    Returns:
        List[int]: 超买信号位置列表
    """
    overbought_points = []
    for i in range(1, len(series)):
        if series[i-1] < threshold and series[i] >= threshold:
            overbought_points.append(i)
    
    return overbought_points

def detect_oversold(series: np.ndarray, threshold: float = 30) -> List[int]:
    """
    检测超卖信号
    
    Args:
        series: 数据序列（如RSI）
        threshold: 超卖阈值
        
    Returns:
        List[int]: 超卖信号位置列表
    """
    oversold_points = []
    for i in range(1, len(series)):
        if series[i-1] > threshold and series[i] <= threshold:
            oversold_points.append(i)
    
    return oversold_points

def detect_divergence(price: np.ndarray, indicator: np.ndarray, window: int = 10) -> List[Tuple[int, str, float]]:
    """
    检测背离信号
    
    Args:
        price: 价格序列
        indicator: 指标序列
        window: 寻找局部极值的窗口大小
        
    Returns:
        List[Tuple[int, str, float]]: 背离信号列表，每个元素为(位置, 类型, 强度)
    """
    if len(price) != len(indicator):
        raise ValueError("价格序列和指标序列长度必须相同")
    
    if len(price) <= window*2:
        return []
    
    # 寻找局部极值
    price_peaks = []
    price_troughs = []
    indicator_peaks = []
    indicator_troughs = []
    
    for i in range(window, len(price) - window):
        # 价格局部极大值
        if price[i] == max(price[i-window:i+window+1]):
            price_peaks.append(i)
        # 价格局部极小值
        if price[i] == min(price[i-window:i+window+1]):
            price_troughs.append(i)
        # 指标局部极大值
        if indicator[i] == max(indicator[i-window:i+window+1]):
            indicator_peaks.append(i)
        # 指标局部极小值
        if indicator[i] == min(indicator[i-window:i+window+1]):
            indicator_troughs.append(i)
    
    divergence_signals = []
    
    # 检测顶背离
    for i in range(len(price_peaks)-1):
        p1 = price_peaks[i]
        p2 = price_peaks[i+1]
        
        # 找到两个价格峰值之间的指标峰值
        matching_indicator_peaks = [idx for idx in indicator_peaks if p1 <= idx <= p2]
        
        if matching_indicator_peaks:
            # 取最接近第二个价格峰值的指标峰值
            indicator_peak = max(matching_indicator_peaks, key=lambda x: abs(x - p2))
            
            # 检测顶背离：价格创新高，但指标没有创新高
            if price[p2] > price[p1] and indicator[indicator_peak] < indicator[indicator_peaks[0]]:
                strength = (price[p2] - price[p1]) / price[p1] * (indicator[indicator_peaks[0]] - indicator[indicator_peak]) / indicator[indicator_peaks[0]]
                divergence_signals.append((p2, "bearish", strength))
    
    # 检测底背离
    for i in range(len(price_troughs)-1):
        p1 = price_troughs[i]
        p2 = price_troughs[i+1]
        
        # 找到两个价格谷值之间的指标谷值
        matching_indicator_troughs = [idx for idx in indicator_troughs if p1 <= idx <= p2]
        
        if matching_indicator_troughs:
            # 取最接近第二个价格谷值的指标谷值
            indicator_trough = max(matching_indicator_troughs, key=lambda x: abs(x - p2))
            
            # 检测底背离：价格创新低，但指标没有创新低
            if price[p2] < price[p1] and indicator[indicator_trough] > indicator[indicator_troughs[0]]:
                strength = (price[p1] - price[p2]) / price[p1] * (indicator[indicator_trough] - indicator[indicator_troughs[0]]) / indicator[indicator_troughs[0]]
                divergence_signals.append((p2, "bullish", strength))
    
    return divergence_signals

def generate_signal_mask(signals: List[int], length: int) -> np.ndarray:
    """
    根据信号位置生成信号掩码数组
    
    Args:
        signals: 信号位置列表
        length: 数组长度
        
    Returns:
        np.ndarray: 信号掩码数组，有信号的位置为1，其余为0
    """
    mask = np.zeros(length)
    for pos in signals:
        if 0 <= pos < length:
            mask[pos] = 1
    return mask

def count_signals_in_window(signals: List[int], window_start: int, window_end: int) -> int:
    """
    统计窗口内的信号数量
    
    Args:
        signals: 信号位置列表
        window_start: 窗口起始位置
        window_end: 窗口结束位置
        
    Returns:
        int: 窗口内的信号数量
    """
    return sum(1 for pos in signals if window_start <= pos <= window_end)

def signal_density(signals: List[int], length: int, window: int = 20) -> np.ndarray:
    """
    计算信号密度
    
    Args:
        signals: 信号位置列表
        length: 数组长度
        window: 滑动窗口大小
        
    Returns:
        np.ndarray: 信号密度数组
    """
    density = np.zeros(length)
    for i in range(length):
        window_start = max(0, i - window)
        window_end = min(length - 1, i + window)
        density[i] = count_signals_in_window(signals, window_start, window_end) / (window_end - window_start + 1)
    return density

def weighted_signal_strength(price: np.ndarray, signals: List[Tuple[int, float]]) -> np.ndarray:
    """
    计算加权信号强度
    
    Args:
        price: 价格序列
        signals: 信号列表，每个元素为(位置, 强度)
        
    Returns:
        np.ndarray: 加权信号强度数组
    """
    strength = np.zeros(len(price))
    for pos, weight in signals:
        if 0 <= pos < len(price):
            strength[pos] = weight
    return strength

def smooth_signals(signals: np.ndarray, window: int = 5) -> np.ndarray:
    """
    平滑信号序列
    
    Args:
        signals: 信号序列
        window: 滑动窗口大小
        
    Returns:
        np.ndarray: 平滑后的信号序列
    """
    return np.convolve(signals, np.ones(window)/window, mode='same')

def filter_false_signals(signals: List[int], confirmation: List[int], max_distance: int = 3) -> List[int]:
    """
    过滤假信号，保留有确认信号的信号
    
    Args:
        signals: 原始信号位置列表
        confirmation: 确认信号位置列表
        max_distance: 最大距离，超过此距离的信号将被过滤
        
    Returns:
        List[int]: 过滤后的信号位置列表
    """
    filtered_signals = []
    for sig in signals:
        # 寻找最近的确认信号
        closest_conf = min(confirmation, key=lambda x: abs(x - sig), default=None)
        if closest_conf is not None and abs(sig - closest_conf) <= max_distance:
            filtered_signals.append(sig)
    return filtered_signals

def combine_signals(signal_lists: List[List[int]], logic: str = 'union') -> List[int]:
    """
    组合多个信号列表
    
    Args:
        signal_lists: 信号列表的列表
        logic: 组合逻辑，'union'表示并集，'intersection'表示交集
        
    Returns:
        List[int]: 组合后的信号位置列表
    """
    if not signal_lists:
        return []
    
    if logic == 'union':
        # 并集，合并所有信号并去重
        combined = set()
        for signals in signal_lists:
            combined.update(signals)
        return sorted(list(combined))
    elif logic == 'intersection':
        # 交集，只保留所有列表都有的信号
        if not signal_lists:
            return []
        result = set(signal_lists[0])
        for signals in signal_lists[1:]:
            result.intersection_update(signals)
        return sorted(list(result))
    else:
        raise ValueError(f"不支持的组合逻辑: {logic}")

def signal_to_dataframe(signals: List[Tuple[int, str, float]], dates: List = None) -> pd.DataFrame:
    """
    将信号列表转换为DataFrame
    
    Args:
        signals: 信号列表，每个元素为(位置, 类型, 强度)
        dates: 日期列表，用于添加日期信息
        
    Returns:
        pd.DataFrame: 包含信号信息的DataFrame
    """
    df = pd.DataFrame(signals, columns=['position', 'type', 'strength'])
    if dates and len(dates) > df['position'].max():
        df['date'] = df['position'].apply(lambda x: dates[x] if 0 <= x < len(dates) else None)
    return df

def crossunder(series1, series2):
    """
    判断series1是否下穿series2
    
    Args:
        series1: 第一个序列
        series2: 第二个序列
        
    Returns:
        bool: 如果series1下穿series2返回True，否则返回False
    """
    if len(series1) < 2 or len(series2) < 2:
        return False
        
    # 判断当前值是否小于等于series2，且前一个值大于series2
    return (series1.iloc[-1] <= series2.iloc[-1]) and (series1.iloc[-2] > series2.iloc[-2])

def crossover(series1, series2):
    """
    判断series1是否上穿series2
    
    Args:
        series1: 第一个序列
        series2: 第二个序列
        
    Returns:
        bool: 如果series1上穿series2返回True，否则返回False
    """
    if len(series1) < 2 or len(series2) < 2:
        return False
        
    # 判断当前值是否大于等于series2，且前一个值小于series2
    return (series1.iloc[-1] >= series2.iloc[-1]) and (series1.iloc[-2] < series2.iloc[-2]) 