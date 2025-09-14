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
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return np.full(len(data), np.nan)
        
    df = pd.DataFrame(data)
    ema = df.ewm(span=window, adjust=False, min_periods=window).mean()
    return ema.values.flatten()

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

def macd_Utils(data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def rsi_Utils(data: np.ndarray, period: int = 14) -> np.ndarray:
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
    return rsi_Utils(data, period)

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
    计算Zig_zag线
    
    Args:
        data: 数据序列
        min_change: 最小变化百分比
        
    Returns:
        Tuple[np.ndarray, List[int]]: (Zig_zag线, 转折点位置)
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

def calculate_ma_Utils(data: pd.Series, period: int) -> pd.Series:
    """
    计算移动平均线
    
    Args:
        data: 价格数据序列
        period: 周期
        
    Returns:
        pd.Series: 移动平均线序列
    """
    return data.rolling(window=period).mean()

def calculate_ema_Utils(data: pd.Series, period: int, method: str = 'standard') -> pd.Series:
    """
    计算指数移动平均线(EMA) - 金融级高精度版本

    专业金融量化交易系统要求：
    - 计算精度：6位小数
    - 数值稳定性：处理极值和边界情况
    - 金融标准：符合行业计算标准

    Args:
        data: 价格数据序列
        period: 周期
        method: 计算方法 ('standard', 'sma_init', 'pandas')

    Returns:
        pd.Series: 指数移动平均线序列（6位小数精度）
    """
    import numpy as np
    from decimal import Decimal, getcontext

    # 设置高精度计算上下文（8位小数内部计算，输出6位）
    getcontext().prec = 28

    # 边界情况处理
    if data is None or len(data) == 0:
        return pd.Series(index=pd.Index([]), dtype='float64')

    if period <= 0:
        raise ValueError(f"EMA周期必须大于0，当前值: {period}")

    if method == 'pandas':
        # 原始pandas方法（保持向后兼容）
        result = data.ewm(span=period, adjust=False).mean()
        # 保持6位小数精度
        return result.round(6)

    elif method == 'sma_init':
        # SMA初始化方法（金融行业标准 - 高精度版本）
        if len(data) < period:
            return pd.Series(index=data.index, dtype='float64').fillna(np.nan)

        result = pd.Series(index=data.index, dtype='float64')

        # 高精度乘数计算
        multiplier = Decimal(2) / Decimal(period + 1)
        multiplier_float = float(multiplier)
        complement = 1.0 - multiplier_float

        # 使用SMA作为初始值（高精度计算）
        valid_data = data.iloc[:period].dropna()
        if len(valid_data) < period:
            # 如果数据不足，使用可用数据的平均值
            sma_init = valid_data.mean()
        else:
            sma_init = valid_data.mean()

        result.iloc[period-1] = round(float(sma_init), 6)

        # 从第period个值开始计算EMA（高精度迭代）
        for i in range(period, len(data)):
            if pd.isna(data.iloc[i]) or pd.isna(result.iloc[i-1]):
                result.iloc[i] = np.nan
            else:
                # 高精度EMA计算
                current_value = Decimal(str(data.iloc[i]))
                previous_ema = Decimal(str(result.iloc[i-1]))

                new_ema = (current_value * multiplier) + (previous_ema * Decimal(str(complement)))
                result.iloc[i] = round(float(new_ema), 6)

        return result

    else:  # 'standard' - 金融级高精度标准方法
        if len(data) == 0:
            return pd.Series(index=data.index, dtype='float64')

        result = pd.Series(index=data.index, dtype='float64')

        # 高精度乘数计算
        multiplier = Decimal(2) / Decimal(period + 1)
        multiplier_float = float(multiplier)
        complement = 1.0 - multiplier_float

        # 第一个非NaN值作为初始值
        first_valid_idx = data.first_valid_index()
        if first_valid_idx is None:
            return result.fillna(np.nan)

        first_valid_pos = data.index.get_loc(first_valid_idx)
        result.iloc[first_valid_pos] = round(float(data.iloc[first_valid_pos]), 6)

        # 计算后续EMA值（高精度迭代）
        for i in range(first_valid_pos + 1, len(data)):
            if pd.isna(data.iloc[i]):
                # 处理缺失数据：保持前一个EMA值
                result.iloc[i] = result.iloc[i-1]
            elif pd.isna(result.iloc[i-1]):
                # 如果前一个EMA值缺失，使用当前值
                result.iloc[i] = round(float(data.iloc[i]), 6)
            else:
                # 高精度EMA计算
                current_value = Decimal(str(data.iloc[i]))
                previous_ema = Decimal(str(result.iloc[i-1]))

                new_ema = (current_value * multiplier) + (previous_ema * Decimal(str(complement)))
                result.iloc[i] = round(float(new_ema), 6)

        return result

def calculate_macd_Utils(data: pd.Series, fast_period: int = 12, slow_period: int = 26,
                  signal_period: int = 9, method: str = 'standard') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算MACD指标 - 金融级高精度版本

    专业量化交易系统要求：
    - 计算精度：6位小数
    - 数值稳定性：处理数值溢出和边界情况
    - 金融标准：符合专业交易平台标准

    Args:
        data: 价格数据序列
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期
        method: EMA计算方法 ('standard', 'sma_init', 'pandas')

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (DIF, DEA, MACD) - 所有精度6位小数
    """
    import numpy as np
    from decimal import Decimal, getcontext

    # 设置高精度计算上下文
    getcontext().prec = 28

    # 边界情况处理
    if data is None or len(data) == 0:
        empty_series = pd.Series(index=pd.Index([]), dtype='float64')
        return empty_series, empty_series, empty_series

    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError(f"MACD周期必须大于0: fast={fast_period}, slow={slow_period}, signal={signal_period}")

    if fast_period >= slow_period:
        raise ValueError(f"快线周期必须小于慢线周期: fast={fast_period}, slow={slow_period}")

    # 使用金融级高精度EMA计算方法
    ema_fast = calculate_ema_Utils(data, fast_period, method)
    ema_slow = calculate_ema_Utils(data, slow_period, method)

    # 计算DIF（MACD线）- 高精度减法运算
    dif = pd.Series(index=data.index, dtype='float64')
    for i in range(len(data)):
        if pd.isna(ema_fast.iloc[i]) or pd.isna(ema_slow.iloc[i]):
            dif.iloc[i] = np.nan
        else:
            # 高精度减法计算
            fast_val = Decimal(str(ema_fast.iloc[i]))
            slow_val = Decimal(str(ema_slow.iloc[i]))
            dif.iloc[i] = round(float(fast_val - slow_val), 6)

    # 计算DEA（信号线）- 对DIF进行EMA平滑
    if method == 'sma_init':
        # 对于SMA初始化方法，从慢线有效开始计算信号线
        valid_start = slow_period - 1
        if len(dif) > valid_start:
            valid_dif = dif.iloc[valid_start:].dropna()
            if len(valid_dif) > 0:
                dea_partial = calculate_ema_Utils(valid_dif, signal_period, method)
                dea = pd.Series(index=dif.index, dtype='float64')
                dea.iloc[valid_start:valid_start+len(dea_partial)] = dea_partial.values
            else:
                dea = pd.Series(index=dif.index, dtype='float64').fillna(np.nan)
        else:
            dea = pd.Series(index=dif.index, dtype='float64').fillna(np.nan)
    else:
        # 标准方法直接计算
        dea = calculate_ema_Utils(dif, signal_period, method)

    # 计算MACD柱状图 - 高精度乘法运算（金融标准：MACD = (DIF - DEA) * 2）
    macd = pd.Series(index=data.index, dtype='float64')
    for i in range(len(data)):
        if pd.isna(dif.iloc[i]) or pd.isna(dea.iloc[i]):
            macd.iloc[i] = np.nan
        else:
            # 高精度减法和乘法计算
            dif_val = Decimal(str(dif.iloc[i]))
            dea_val = Decimal(str(dea.iloc[i]))
            macd_val = (dif_val - dea_val) * Decimal('2')
            macd.iloc[i] = round(float(macd_val), 6)

    # 数值稳定性检查和修正（使用专业稳定性管理器）
    from utils.numerical_stability_manager import get_stability_manager
    stability_manager = get_stability_manager()

    dif = stability_manager.check_and_fix_extreme_values(dif, "MACD DIF")
    dea = stability_manager.check_and_fix_extreme_values(dea, "MACD DEA")
    macd = stability_manager.check_and_fix_extreme_values(macd, "MACD Histogram")

    # 确保最终精度
    dif = stability_manager.ensure_series_precision(dif)
    dea = stability_manager.ensure_series_precision(dea)
    macd = stability_manager.ensure_series_precision(macd)

    return dif, dea, macd

def calculate_kdj_Utils(high: pd.Series, low: pd.Series, close: pd.Series,
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

def calculate_rsi_Utils(data: pd.Series, period: int = 14) -> pd.Series:
    """
    计算RSI指标 - 金融级高精度版本（标准Wilder平滑方法）

    专业量化交易系统要求：
    - 计算精度：6位小数
    - Wilder平滑算法：金融行业标准
    - 数值稳定性：处理边界情况和极值

    Args:
        data: 价格数据序列
        period: 周期

    Returns:
        pd.Series: RSI序列（6位小数精度）
    """
    import numpy as np
    from decimal import Decimal, getcontext
    from utils.dependency_injection import get_logger

    logger = get_logger(__name__)

    # 设置高精度计算上下文
    getcontext().prec = 28

    # 边界情况处理
    if data is None or len(data) == 0:
        return pd.Series(index=pd.Index([]), dtype='float64')

    if period <= 0:
        raise ValueError(f"RSI周期必须大于0，当前值: {period}")

    if len(data) <= period:
        logger.warning(f"RSI数据长度不足: 需要至少{period+1}个数据点，实际{len(data)}个")
        return pd.Series(index=data.index, dtype='float64').fillna(np.nan)

    # 计算价格变化
    delta = data.diff()

    # 分离上涨和下跌（高精度处理）
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # 使用标准Wilder平滑方法进行高精度计算
    rsi_values = pd.Series(index=data.index, dtype='float64')

    if len(gains) >= period:
        # 计算初始平均值（前period个值的简单平均，跳过第一个NaN值）
        initial_gains = gains.iloc[1:period+1].dropna()
        initial_losses = losses.iloc[1:period+1].dropna()

        if len(initial_gains) < period - 1 or len(initial_losses) < period - 1:
            logger.warning(f"RSI初始计算数据不足，使用可用数据")

        # 高精度初始平均值计算
        initial_avg_gain = Decimal(str(initial_gains.mean())) if len(initial_gains) > 0 else Decimal('0')
        initial_avg_loss = Decimal(str(initial_losses.mean())) if len(initial_losses) > 0 else Decimal('0')

        # 设置初始RSI值
        if initial_avg_loss == 0:
            if initial_avg_gain == 0:
                rsi_values.iloc[period] = 50.0  # 中性值
            else:
                rsi_values.iloc[period] = 100.0  # 全部上涨
        else:
            # 高精度RS计算
            rs = initial_avg_gain / initial_avg_loss
            rsi_val = 100 - (100 / (1 + rs))
            rsi_values.iloc[period] = round(float(rsi_val), 6)

        # 使用Wilder平滑方法计算后续值（高精度迭代）
        avg_gain = initial_avg_gain
        avg_loss = initial_avg_loss
        period_decimal = Decimal(str(period))

        for i in range(period + 1, len(gains)):
            if pd.isna(gains.iloc[i]) or pd.isna(losses.iloc[i]):
                # 处理缺失数据
                rsi_values.iloc[i] = rsi_values.iloc[i-1] if pd.notna(rsi_values.iloc[i-1]) else np.nan
                continue

            # 高精度Wilder平滑公式
            current_gain = Decimal(str(gains.iloc[i]))
            current_loss = Decimal(str(losses.iloc[i]))

            avg_gain = (avg_gain * (period_decimal - 1) + current_gain) / period_decimal
            avg_loss = (avg_loss * (period_decimal - 1) + current_loss) / period_decimal

            # 计算RSI值
            if avg_loss == 0:
                if avg_gain == 0:
                    rsi_values.iloc[i] = 50.0  # 中性值
                else:
                    rsi_values.iloc[i] = 100.0  # 全部上涨
            else:
                # 高精度RS和RSI计算
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
                rsi_values.iloc[i] = round(float(rsi_val), 6)

    # 数值稳定性检查和修正（使用专业稳定性管理器）
    from utils.numerical_stability_manager import get_stability_manager
    stability_manager = get_stability_manager()

    rsi_values = stability_manager.validate_rsi_range(rsi_values)
    rsi_values = stability_manager.check_and_fix_extreme_values(rsi_values, "RSI")

    # 检测计算异常
    anomalies = stability_manager.detect_calculation_anomalies(rsi_values, "RSI")

    return stability_manager.ensure_series_precision(rsi_values)

def calculate_bollinger_bands_Utils(data: pd.Series, period: int = 20, 
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
    middle_band = calculate_ma_Utils(data, period)
    
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
    """计算一系列点的斜率"""
    n = len(points)
    if n < 2:
        return 0.0
    
    x_sum = sum(p[0] for p in points)
    y_sum = sum(p[1] for p in points)
    xy_sum = sum(p[0] * p[1] for p in points)
    x_sq_sum = sum(p[0]**2 for p in points)
    
    numerator = n * xy_sum - x_sum * y_sum
    denominator = n * x_sq_sum - x_sum**2
    
    return numerator / denominator if denominator != 0 else 0.0

# === 从 BaseIndicator 迁移的通用函数 ===

def crossover_Utils(series1: pd.Series, series2: Union[pd.Series, float, int]) -> pd.Series:
    """
    判断series1上穿series2
    """
    return (series1 > series2) & (series1.shift(1) <= series2)


def crossunder_Utils(series1: pd.Series, series2: Union[pd.Series, float, int]) -> pd.Series:
    """
    判断series1下穿series2
    """
    return (series1 < series2) & (series1.shift(1) >= series2)


def sma_Utils(series: pd.Series, periods: int) -> pd.Series:
    """
    计算简单移动平均 (SMA)
    """
    return series.rolling(window=periods, min_periods=periods).mean()


def ema_Utils(series: pd.Series, periods: int) -> pd.Series:
    """
    计算指数移动平均 (EMA)
    """
    return series.ewm(span=periods, adjust=False).mean()


def highest_Utils(series: pd.Series, periods: int) -> pd.Series:
    """
    获取N周期内的最高价
    """
    return series.rolling(window=periods, min_periods=periods).max()


def lowest_Utils(series: pd.Series, periods: int) -> pd.Series:
    """
    获取N周期内的最低价
    """
    return series.rolling(window=periods, min_periods=periods).min()


def atr_Utils(high: pd.Series, low: pd.Series, close: pd.Series, periods: int) -> pd.Series:
    """
    计算平均真实波幅 (ATR)
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    return ema_Utils(tr, periods)


def ensure_columns_Utils(data: pd.DataFrame, required_columns: List[str]) -> None:
    """
    确保DataFrame中存在所需的列
    
    Args:
        data: 输入的DataFrame
        required_columns: 必需的列名列表
        
    Raises:
        ValueError: 如果缺少任何必需的列
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"数据中缺少以下必需列: {', '.join(missing_columns)}")

# 添加简单别名以便其他模块使用
calculate_macd = calculate_macd_Utils
calculate_kdj = calculate_kdj_Utils
crossover = crossover_Utils
crossunder = crossunder_Utils 