"""
技术指标公共函数模块

提供统一的技术指标计算功能，用于被各个具体指标类调用
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Optional, Any

# 类型别名
NumericArray = Union[List[float], np.ndarray, pd.Series]


def ma(series: NumericArray, periods: int) -> np.ndarray:
    """
    计算简单移动平均线
    
    Args:
        series: 输入序列
        periods: 周期
        
    Returns:
        np.ndarray: 简单移动平均线
    """
    return pd.Series(series).rolling(periods).mean().values


def ema(series: NumericArray, periods: int) -> np.ndarray:
    """
    计算指数移动平均线
    
    Args:
        series: 输入序列
        periods: 周期
        
    Returns:
        np.ndarray: 指数移动平均线
    """
    return pd.Series(series).ewm(span=periods, adjust=False).mean().values


def sma(series: NumericArray, periods: int, weight: float = 1) -> np.ndarray:
    """
    计算中国式的SMA平滑移动平均线
    
    Args:
        series: 输入序列
        periods: 周期
        weight: 权重
        
    Returns:
        np.ndarray: 平滑移动平均线
    """
    return pd.Series(series).ewm(alpha=weight/periods, adjust=False).mean().values


def wma(series: NumericArray, periods: int) -> np.ndarray:
    """
    计算加权移动平均线
    
    Args:
        series: 输入序列
        periods: 周期
        
    Returns:
        np.ndarray: 加权移动平均线
    """
    def weighted_average(x):
        weights = np.arange(1, len(x) + 1)
        return np.sum(x * weights) / np.sum(weights)
    
    return pd.Series(series).rolling(periods).apply(
        lambda x: weighted_average(x.values), raw=True
    ).values


def dma(series: NumericArray, alpha: Union[float, NumericArray]) -> np.ndarray:
    """
    计算动态移动平均线
    
    Args:
        series: 输入序列
        alpha: 平滑因子，必须在0-1之间，可以是单一值或者与series等长的序列
        
    Returns:
        np.ndarray: 动态移动平均线
    """
    if isinstance(alpha, (int, float)):
        return pd.Series(series).ewm(alpha=alpha, adjust=False).mean().values
    
    alpha_array = np.array(alpha)
    alpha_array[np.isnan(alpha_array)] = 1.0
    
    series_array = np.array(series)
    result = np.zeros(len(series_array))
    result[0] = series_array[0]
    
    for i in range(1, len(series_array)):
        result[i] = alpha_array[i] * series_array[i] + (1 - alpha_array[i]) * result[i-1]
    
    return result


def highest(series: NumericArray, periods: int) -> np.ndarray:
    """
    计算周期内最高值
    
    Args:
        series: 输入序列
        periods: 周期
        
    Returns:
        np.ndarray: 周期内最高值
    """
    return pd.Series(series).rolling(periods).max().values


def lowest(series: NumericArray, periods: int) -> np.ndarray:
    """
    计算周期内最低值
    
    Args:
        series: 输入序列
        periods: 周期
        
    Returns:
        np.ndarray: 周期内最低值
    """
    return pd.Series(series).rolling(periods).min().values


def ref(series: NumericArray, periods: int = 1) -> np.ndarray:
    """
    计算序列向前移动周期数
    
    Args:
        series: 输入序列
        periods: 周期数，默认为1
        
    Returns:
        np.ndarray: 移动后的序列
    """
    return pd.Series(series).shift(periods).values


def std(series: NumericArray, periods: int) -> np.ndarray:
    """
    计算标准差
    
    Args:
        series: 输入序列
        periods: 周期
        
    Returns:
        np.ndarray: 标准差
    """
    return pd.Series(series).rolling(periods).std(ddof=0).values


def sum(series: NumericArray, periods: int) -> np.ndarray:
    """
    计算周期内求和
    
    Args:
        series: 输入序列
        periods: 周期，如果为0则计算累计和
        
    Returns:
        np.ndarray: 周期内和
    """
    if periods <= 0:
        return pd.Series(series).cumsum().values
    else:
        return pd.Series(series).rolling(periods).sum().values


def diff(series: NumericArray, periods: int = 1) -> np.ndarray:
    """
    计算序列差分
    
    Args:
        series: 输入序列
        periods: 差分周期
        
    Returns:
        np.ndarray: 差分序列
    """
    return pd.Series(series).diff(periods).values


def macd(close: NumericArray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算MACD指标
    
    Args:
        close: 收盘价序列
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (DIF, DEA, MACD)
    """
    fast_ema = ema(close, fast_period)
    slow_ema = ema(close, slow_period)
    
    # 确保前N个值为NaN，其中N = max(fast_period, slow_period) - 1
    min_periods = max(fast_period, slow_period) - 1
    dif = fast_ema - slow_ema
    dif[:min_periods] = np.nan
    
    dea = ema(dif, signal_period)
    # 确保DEA的前N+signal_period-1个值为NaN
    dea[:min_periods + signal_period - 1] = np.nan
    
    # 计算MACD柱状图
    macd_value = (dif - dea) * 2
    macd_value[:min_periods + signal_period - 1] = np.nan
    
    return dif, dea, macd_value


def kdj(close: NumericArray, high: NumericArray, low: NumericArray, n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算KDJ指标
    
    Args:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        n: RSV周期
        m1: K值平滑因子
        m2: D值平滑因子
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (K, D, J)
    """
    rsv = (close - lowest(low, n)) / (highest(high, n) - lowest(low, n)) * 100
    k = ema(rsv, (m1*2-1))
    d = ema(k, (m2*2-1))
    j = k*3 - d*2
    return k, d, j


def boll(close: NumericArray, periods: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算布林带指标
    
    Args:
        close: 收盘价序列
        periods: 周期
        std_dev: 标准差倍数
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (上轨, 中轨, 下轨)
    """
    mid = ma(close, periods)
    stdev = std(close, periods)
    upper = mid + stdev * std_dev
    lower = mid - stdev * std_dev
    return upper, mid, lower


def rsi(close: NumericArray, periods: int = 14) -> np.ndarray:
    """
    计算RSI指标
    
    Args:
        close: 收盘价序列
        periods: 周期
        
    Returns:
        np.ndarray: RSI值
    """
    delta = diff(close)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    avg_up = sma(up, periods)
    avg_down = sma(down, periods)
    
    rs = np.divide(avg_up, avg_down, out=np.zeros_like(avg_up), where=avg_down != 0)
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def atr(close: NumericArray, high: NumericArray, low: NumericArray, periods: int = 14) -> np.ndarray:
    """
    计算ATR指标
    
    Args:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        periods: 周期
        
    Returns:
        np.ndarray: ATR值
    """
    prev_close = ref(close, 1)
    tr = np.maximum(
        np.maximum(high - low, np.abs(high - prev_close)),
        np.abs(low - prev_close)
    )
    return ma(tr, periods)


def obv(close: NumericArray, volume: NumericArray) -> np.ndarray:
    """
    计算OBV(On-Balance Volume)指标
    
    Args:
        close: 收盘价序列
        volume: 成交量序列
        
    Returns:
        np.ndarray: OBV值
    """
    close_diff = diff(close)
    obv_values = np.zeros_like(volume, dtype=float)
    
    for i in range(1, len(close)):
        if close_diff[i] > 0:
            obv_values[i] = obv_values[i-1] + volume[i]
        elif close_diff[i] < 0:
            obv_values[i] = obv_values[i-1] - volume[i]
        else:
            obv_values[i] = obv_values[i-1]
            
    return obv_values


def cross(series1: NumericArray, series2: NumericArray) -> np.ndarray:
    """
    判断两条线是否交叉（金叉）
    
    Args:
        series1: 第一条线
        series2: 第二条线
        
    Returns:
        np.ndarray: 布尔数组，True表示当前位置发生了金叉
    """
    series1 = np.array(series1)
    series2 = np.array(series2)
    
    if len(series1) < 2 or len(series2) < 2:
        return np.array([False] * len(series1))
        
    # 判断向上交叉
    # 前一个周期series1 <= series2，当前周期series1 > series2
    cond1 = np.concatenate(([False], series1[:-1] <= series2[:-1]))
    cond2 = series1 > series2
    
    return cond1 & cond2


def crossover(series1: NumericArray, series2: NumericArray) -> np.ndarray:
    """
    判断向上穿越情况（金叉）
    
    Args:
        series1: 第一个序列
        series2: 第二个序列或固定值
        
    Returns:
        np.ndarray: 布尔数组，True表示series1从下方穿过series2
    """
    # 确保输入是numpy数组
    s1 = np.array(series1)
    
    # 如果series2是标量值，创建相同长度的常数数组
    if np.isscalar(series2):
        s2 = np.full_like(s1, series2)
    else:
        s2 = np.array(series2)
    
    # 前一个时刻series1小于等于series2，当前时刻series1大于series2
    prev_leq = np.roll(s1 <= s2, 1)
    curr_gt = s1 > s2
    
    # 组合判断
    crossover_result = prev_leq & curr_gt
    
    # 第一个元素设为False，因为没有前一个时刻的数据
    crossover_result[0] = False
    
    return crossover_result


def crossunder(series1: NumericArray, series2: NumericArray) -> np.ndarray:
    """
    判断向下穿越情况（死叉）
    
    Args:
        series1: 第一个序列
        series2: 第二个序列或固定值
        
    Returns:
        np.ndarray: 布尔数组，True表示series1从上方穿过series2
    """
    # 确保输入是numpy数组
    s1 = np.array(series1)
    
    # 如果series2是标量值，创建相同长度的常数数组
    if np.isscalar(series2):
        s2 = np.full_like(s1, series2)
    else:
        s2 = np.array(series2)
    
    # 前一个时刻series1大于等于series2，当前时刻series1小于series2
    prev_geq = np.roll(s1 >= s2, 1)
    curr_lt = s1 < s2
    
    # 组合判断
    crossunder_result = prev_geq & curr_lt
    
    # 第一个元素设为False，因为没有前一个时刻的数据
    crossunder_result[0] = False
    
    return crossunder_result


def barslast(condition: NumericArray) -> np.ndarray:
    """
    计算上一次条件成立到当前的周期数
    
    Args:
        condition: 条件序列
        
    Returns:
        np.ndarray: 距离上次条件成立的周期数
    """
    condition = np.array(condition, dtype=bool)
    result = np.zeros(len(condition))
    
    # 初始化计数器
    count = -1
    
    for i in range(len(condition)):
        if condition[i]:
            # 条件成立，重置计数器
            count = 0
        else:
            # 条件不成立，计数器加1
            if count >= 0:
                count += 1
        
        result[i] = count
        
    return result 