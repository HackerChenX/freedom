#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术分析工具模块

提供技术分析相关的功能函数
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple

def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    计算简单移动平均线
    
    Args:
        data: 数据序列
        window: 窗口大小
        
    Returns:
        np.ndarray: 移动平均线
    """
    if len(data) < window:
        # 返回全部为NaN的数组
        return np.full_like(data, np.nan, dtype=float)
    
    weights = np.ones(window) / window
    # 使用convolve计算移动平均，valid模式确保只返回完全在窗口内的值
    ma = np.convolve(data, weights, mode='valid')
    # 填充前面的值为NaN
    return np.concatenate([np.full(window-1, np.nan), ma])

def exponential_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    计算指数移动平均线
    
    Args:
        data: 数据序列
        window: 窗口大小
        
    Returns:
        np.ndarray: 指数移动平均线
    """
    if len(data) < window:
        return np.full_like(data, np.nan, dtype=float)
    
    # 指数衰减因子
    alpha = 2 / (window + 1)
    
    # 初始化EMA数组，前window-1个值为NaN
    ema = np.full_like(data, np.nan, dtype=float)
    
    # 初始值使用window个数据的简单平均
    ema[window-1] = np.mean(data[:window])
    
    # 计算后续值
    for i in range(window, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    
    return ema

def weighted_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    计算加权移动平均线
    
    Args:
        data: 数据序列
        window: 窗口大小
        
    Returns:
        np.ndarray: 加权移动平均线
    """
    if len(data) < window:
        return np.full_like(data, np.nan, dtype=float)
    
    # 构建加权系数，权重与数据的位置成正比
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    
    # 初始化WMA数组
    wma = np.full_like(data, np.nan, dtype=float)
    
    # 计算加权平均
    for i in range(window-1, len(data)):
        wma[i] = np.sum(data[i-window+1:i+1] * weights)
    
    return wma

def macd(data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算MACD指标
    
    Args:
        data: 数据序列
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (MACD线, 信号线, 柱状图)
    """
    # 计算快线和慢线
    fast_ema = exponential_moving_average(data, fast_period)
    slow_ema = exponential_moving_average(data, slow_period)
    
    # MACD线 = 快线 - 慢线
    macd_line = fast_ema - slow_ema
    
    # 信号线是MACD的EMA
    signal_line = exponential_moving_average(macd_line, signal_period)
    
    # 柱状图 = MACD线 - 信号线
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
    """
    计算相对强弱指标(RSI)
    
    Args:
        data: 数据序列
        period: 周期
        
    Returns:
        np.ndarray: RSI值
    """
    if len(data) <= period:
        return np.full_like(data, np.nan, dtype=float)
    
    # 计算价格变化
    deltas = np.diff(data)
    deltas = np.append([0], deltas)  # 第一个值为0
    
    # 分离上涨和下跌
    up = np.where(deltas > 0, deltas, 0)
    down = np.where(deltas < 0, -deltas, 0)
    
    # 计算平均上涨和平均下跌
    avg_up = np.full_like(data, np.nan, dtype=float)
    avg_down = np.full_like(data, np.nan, dtype=float)
    
    # 初始值
    avg_up[period] = np.mean(up[1:period+1])
    avg_down[period] = np.mean(down[1:period+1])
    
    # 计算后续值（使用WilderSmoothing方法）
    for i in range(period+1, len(data)):
        avg_up[i] = (avg_up[i-1] * (period-1) + up[i]) / period
        avg_down[i] = (avg_down[i-1] * (period-1) + down[i]) / period
    
    # 计算相对强度
    rs = avg_up / (avg_down + 1e-10)  # 防止除以0
    
    # 计算RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def bollinger_bands(data: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算布林带
    
    Args:
        data: 数据序列
        window: 窗口大小
        num_std: 标准差倍数
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (中轨, 上轨, 下轨)
    """
    if len(data) < window:
        empty = np.full_like(data, np.nan, dtype=float)
        return empty, empty, empty
    
    # 中轨(SMA)
    middle = moving_average(data, window)
    
    # 计算标准差
    rolling_std = np.full_like(data, np.nan, dtype=float)
    for i in range(window-1, len(data)):
        rolling_std[i] = np.std(data[i-window+1:i+1])
    
    # 上轨和下轨
    upper = middle + (rolling_std * num_std)
    lower = middle - (rolling_std * num_std)
    
    return middle, upper, lower

def stochastic_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算随机振荡器
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        k_period: %K周期
        d_period: %D周期
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (%K, %D)
    """
    if len(close) < k_period:
        empty = np.full_like(close, np.nan, dtype=float)
        return empty, empty
    
    # 初始化%K数组
    k = np.full_like(close, np.nan, dtype=float)
    
    # 计算%K
    for i in range(k_period-1, len(close)):
        window_high = np.max(high[i-k_period+1:i+1])
        window_low = np.min(low[i-k_period+1:i+1])
        
        # 如果最高价等于最低价，则%K为50
        if window_high == window_low:
            k[i] = 50
        else:
            k[i] = 100 * (close[i] - window_low) / (window_high - window_low)
    
    # %D是%K的移动平均
    d = moving_average(k, d_period)
    
    return k, d

def average_directional_index(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算平均趋向指数(ADX)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (ADX, +DI, -DI, DX)
    """
    if len(close) < period + 1:
        empty = np.full_like(close, np.nan, dtype=float)
        return empty, empty, empty, empty
    
    # 计算真实范围TR
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]  # 第一个值
    
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # 计算方向性指标
    up_move = np.zeros(len(close))
    down_move = np.zeros(len(close))
    
    for i in range(1, len(close)):
        up_move[i] = high[i] - high[i-1]
        down_move[i] = low[i-1] - low[i]
    
    # 修正方向性指标
    for i in range(1, len(close)):
        if up_move[i] < 0 or up_move[i] < down_move[i]:
            up_move[i] = 0
        if down_move[i] < 0 or down_move[i] < up_move[i]:
            down_move[i] = 0
    
    # 计算平滑值
    atr = np.full_like(close, np.nan, dtype=float)
    plus_di = np.full_like(close, np.nan, dtype=float)
    minus_di = np.full_like(close, np.nan, dtype=float)
    
    # 初始值
    atr[period] = np.mean(tr[1:period+1])
    plus_di[period] = 100 * np.mean(up_move[1:period+1]) / atr[period]
    minus_di[period] = 100 * np.mean(down_move[1:period+1]) / atr[period]
    
    # 计算后续值
    for i in range(period+1, len(close)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_di[i] = 100 * ((plus_di[i-1] * (period-1) + up_move[i]) / period) / atr[i]
        minus_di[i] = 100 * ((minus_di[i-1] * (period-1) + down_move[i]) / period) / atr[i]
    
    # 计算方向性指数DX
    dx = np.full_like(close, np.nan, dtype=float)
    for i in range(period, len(close)):
        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i] + 1e-10)
    
    # 计算ADX（DX的平均值）
    adx = np.full_like(close, np.nan, dtype=float)
    adx[2*period-1] = np.mean(dx[period:2*period])
    
    for i in range(2*period, len(close)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx, plus_di, minus_di, dx

def average_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    计算平均真实范围(ATR)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期
        
    Returns:
        np.ndarray: ATR值
    """
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    # 计算真实范围TR
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]  # 第一个值
    
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # 计算ATR
    atr = np.full_like(close, np.nan, dtype=float)
    
    # 初始值
    atr[period] = np.mean(tr[1:period+1])
    
    # 计算后续值（使用WilderSmoothing方法）
    for i in range(period+1, len(close)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def on_balance_volume(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    计算能量潮指标(OBV)
    
    Args:
        close: 收盘价序列
        volume: 成交量序列
        
    Returns:
        np.ndarray: OBV值
    """
    if len(close) != len(volume):
        raise ValueError("收盘价和成交量序列长度必须相同")
    
    obv = np.zeros_like(close)
    
    # 第一个值等于第一个成交量
    obv[0] = volume[0]
    
    # 计算后续值
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    return obv

def money_flow_index(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
    """
    计算资金流量指标(MFI)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
        period: 周期
        
    Returns:
        np.ndarray: MFI值
    """
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    # 计算典型价格
    typical_price = (high + low + close) / 3
    
    # 计算资金流量
    money_flow = typical_price * volume
    
    # 分离正向和负向资金流量
    positive_flow = np.zeros_like(close)
    negative_flow = np.zeros_like(close)
    
    for i in range(1, len(close)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow[i] = money_flow[i]
            negative_flow[i] = 0
        elif typical_price[i] < typical_price[i-1]:
            positive_flow[i] = 0
            negative_flow[i] = money_flow[i]
        else:
            positive_flow[i] = 0
            negative_flow[i] = 0
    
    # 计算正向和负向资金流量的移动总和
    positive_sum = np.full_like(close, np.nan, dtype=float)
    negative_sum = np.full_like(close, np.nan, dtype=float)
    
    for i in range(period-1, len(close)):
        positive_sum[i] = np.sum(positive_flow[i-period+1:i+1])
        negative_sum[i] = np.sum(negative_flow[i-period+1:i+1])
    
    # 计算资金比率
    money_ratio = positive_sum / (negative_sum + 1e-10)  # 防止除以0
    
    # 计算MFI
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi

def rate_of_change(data: np.ndarray, period: int = 10) -> np.ndarray:
    """
    计算变动率指标(ROC)
    
    Args:
        data: 数据序列
        period: 周期
        
    Returns:
        np.ndarray: ROC值
    """
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    
    roc = np.full_like(data, np.nan, dtype=float)
    
    for i in range(period, len(data)):
        roc[i] = 100 * (data[i] - data[i-period]) / data[i-period]
    
    return roc

def relative_strength_index(data: np.ndarray, period: int = 14) -> np.ndarray:
    """
    计算相对强弱指标(RSI)，与rsi函数相同，但实现方式略有不同
    
    Args:
        data: 数据序列
        period: 周期
        
    Returns:
        np.ndarray: RSI值
    """
    return rsi(data, period)

def standard_deviation(data: np.ndarray, window: int = 20) -> np.ndarray:
    """
    计算标准差
    
    Args:
        data: 数据序列
        window: 窗口大小
        
    Returns:
        np.ndarray: 标准差值
    """
    if len(data) < window:
        return np.full_like(data, np.nan, dtype=float)
    
    std = np.full_like(data, np.nan, dtype=float)
    
    for i in range(window-1, len(data)):
        std[i] = np.std(data[i-window+1:i+1])
    
    return std

def linear_regression(data: np.ndarray, window: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算线性回归
    
    Args:
        data: 数据序列
        window: 窗口大小
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (预测值, 斜率, R^2)
    """
    if len(data) < window:
        empty = np.full_like(data, np.nan, dtype=float)
        return empty, empty, empty
    
    predicted = np.full_like(data, np.nan, dtype=float)
    slope = np.full_like(data, np.nan, dtype=float)
    r_squared = np.full_like(data, np.nan, dtype=float)
    
    for i in range(window-1, len(data)):
        y = data[i-window+1:i+1]
        x = np.arange(window)
        
        # 计算线性回归参数
        n = window
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)
        
        # 计算斜率和截距
        slope[i] = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        intercept = (sum_y - slope[i] * sum_x) / n
        
        # 计算预测值
        predicted[i] = intercept + slope[i] * (window - 1)
        
        # 计算R^2
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - (intercept + slope[i] * x)) ** 2)
        
        if ss_total == 0:
            r_squared[i] = 1.0  # 如果数据完全平坦，R^2设为1
        else:
            r_squared[i] = 1 - (ss_residual / ss_total)
    
    return predicted, slope, r_squared

def fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
    """
    计算斐波那契回调水平
    
    Args:
        high: 最高价
        low: 最低价
        
    Returns:
        Dict[str, float]: 斐波那契回调水平
    """
    diff = high - low
    
    return {
        "0.0": low,
        "0.236": low + 0.236 * diff,
        "0.382": low + 0.382 * diff,
        "0.5": low + 0.5 * diff,
        "0.618": low + 0.618 * diff,
        "0.786": low + 0.786 * diff,
        "1.0": high
    }

def zigzag(data: np.ndarray, min_change: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    """
    计算ZigZag线
    
    Args:
        data: 数据序列
        min_change: 最小变化百分比
        
    Returns:
        Tuple[np.ndarray, List[int]]: (ZigZag线, 转折点位置)
    """
    if len(data) < 2:
        return np.full_like(data, np.nan, dtype=float), []
    
    # 初始化
    zigzag = np.full_like(data, np.nan, dtype=float)
    turning_points = []
    
    # 设置第一个点
    zigzag[0] = data[0]
    turning_points.append(0)
    
    # 当前趋势（1为上升，-1为下降）
    trend = 0
    last_turning_point = 0
    
    for i in range(1, len(data)):
        change = (data[i] - data[last_turning_point]) / data[last_turning_point]
        
        if trend == 0:
            # 初始化趋势
            if change > min_change:
                trend = 1  # 上升趋势
            elif change < -min_change:
                trend = -1  # 下降趋势
            
            if trend != 0:
                zigzag[i] = data[i]
                turning_points.append(i)
                last_turning_point = i
        
        elif trend == 1:  # 上升趋势
            if change > 0 and data[i] > data[last_turning_point]:
                # 如果继续上升，更新最后的转折点
                zigzag[last_turning_point] = np.nan
                turning_points.pop()
                zigzag[i] = data[i]
                turning_points.append(i)
                last_turning_point = i
            elif change < -min_change:
                # 如果下降超过阈值，转为下降趋势
                trend = -1
                zigzag[i] = data[i]
                turning_points.append(i)
                last_turning_point = i
        
        elif trend == -1:  # 下降趋势
            if change < 0 and data[i] < data[last_turning_point]:
                # 如果继续下降，更新最后的转折点
                zigzag[last_turning_point] = np.nan
                turning_points.pop()
                zigzag[i] = data[i]
                turning_points.append(i)
                last_turning_point = i
            elif change > min_change:
                # 如果上升超过阈值，转为上升趋势
                trend = 1
                zigzag[i] = data[i]
                turning_points.append(i)
                last_turning_point = i
    
    return zigzag, turning_points

def find_peaks_and_troughs(data: np.ndarray, window: int = 5) -> Tuple[list, list]:
    """
    查找局部极大值和极小值（占位实现，防止import错误）
    Args:
        data: 数据序列
        window: 窗口大小
    Returns:
        Tuple[list, list]: (极大值索引列表, 极小值索引列表)
    """
    return [], []

def calculate_ma(data: pd.Series, period: int) -> pd.Series:
    """
    计算移动平均线
    
    Args:
        data: 价格数据序列
        period: 周期
        
    Returns:
        pd.Series: 移动平均线序列
    """
    return data.rolling(window=period).mean()

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """
    计算指数移动平均线
    
    Args:
        data: 价格数据序列
        period: 周期
        
    Returns:
        pd.Series: 指数移动平均线序列
    """
    return data.ewm(span=period, adjust=False).mean()

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, 
                  signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算MACD指标
    
    Args:
        data: 价格数据序列
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (DIF, DEA, MACD)
    """
    # 计算快线和慢线的EMA
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    
    # 计算DIF
    dif = ema_fast - ema_slow
    
    # 计算DEA
    dea = calculate_ema(dif, signal_period)
    
    # 计算MACD
    macd = (dif - dea) * 2
    
    return dif, dea, macd

def calculate_kdj(high: pd.Series, low: pd.Series, close: pd.Series,
                 k_period: int = 9, d_period: int = 3, j_period: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算KDJ指标
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        k_period: K值周期
        d_period: D值周期
        j_period: J值周期
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (K, D, J)
    """
    # 计算RSV
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    rsv = (close - low_min) / (high_max - low_min) * 100
    
    # 计算K值
    k = pd.Series(0.0, index=close.index)
    for i in range(len(close)):
        if i == 0:
            k.iloc[i] = 50.0
        else:
            k.iloc[i] = (2/3) * k.iloc[i-1] + (1/3) * rsv.iloc[i]
    
    # 计算D值
    d = pd.Series(0.0, index=close.index)
    for i in range(len(close)):
        if i == 0:
            d.iloc[i] = 50.0
        else:
            d.iloc[i] = (2/3) * d.iloc[i-1] + (1/3) * k.iloc[i]
    
    # 计算J值
    j = 3 * k - 2 * d
    
    return k, d, j

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    计算RSI指标
    
    Args:
        data: 价格数据序列
        period: 周期
        
    Returns:
        pd.Series: RSI序列
    """
    # 计算价格变化
    delta = data.diff()
    
    # 分离上涨和下跌
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # 计算RS和RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(data: pd.Series, period: int = 20, 
                            num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算布林带
    
    Args:
        data: 价格数据序列
        period: 周期
        num_std: 标准差倍数
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (中轨, 上轨, 下轨)
    """
    # 计算中轨
    middle_band = calculate_ma(data, period)
    
    # 计算标准差
    std = data.rolling(window=period).std()
    
    # 计算上下轨
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return middle_band, upper_band, lower_band

def find_local_extrema(data: np.ndarray, window: int = 5) -> Tuple[List[int], List[int]]:
    """
    查找局部极值点 (波峰和波谷)
    
    Args:
        data: 数据序列
        window: 窗口大小，用于判断极值
        
    Returns:
        Tuple[List[int], List[int]]: (波峰索引列表, 波谷索引列表)
    """
    peaks = []
    troughs = []
    
    # 确保窗口大小为奇数
    if window % 2 == 0:
        window += 1
        
    half_window = window // 2
    
    for i in range(half_window, len(data) - half_window):
        # 当前点
        current_point = data[i]
        
        # 窗口内的数据
        window_data = data[i-half_window:i+half_window+1]
        
        # 判断是否为波峰
        if current_point == np.max(window_data):
            peaks.append(i)
        
        # 判断是否为波谷
        if current_point == np.min(window_data):
            troughs.append(i)
            
    return peaks, troughs

def calculate_slope(points: List[Tuple[int, float]]) -> float:
    """
    计算一系列点的斜率
    
    Args:
        points: 点的列表，每个点是(索引, 值)的元组
        
    Returns:
        float: 斜率
    """
    if len(points) < 2:
        return 0.0
        
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    
    # 使用线性回归计算斜率
    slope, _, _, _, _ = np.polyfit(x, y, 1, full=True)
    
    return slope[0] 