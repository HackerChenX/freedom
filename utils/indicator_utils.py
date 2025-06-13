#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用指标计算工具函数
"""
import pandas as pd
from typing import Union, List


def crossover(series1: pd.Series, series2: Union[pd.Series, float, int]) -> pd.Series:
    """
    判断series1上穿series2
    """
    series2 = pd.Series(series2, index=series1.index) if isinstance(series2, (int, float)) else series2
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def crossunder(series1: pd.Series, series2: Union[pd.Series, float, int]) -> pd.Series:
    """
    判断series1下穿series2
    """
    series2 = pd.Series(series2, index=series1.index) if isinstance(series2, (int, float)) else series2
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


def sma(series: pd.Series, periods: int) -> pd.Series:
    """
    计算简单移动平均 (SMA)
    """
    return series.rolling(window=periods, min_periods=periods).mean()


def ema(series: pd.Series, periods: int) -> pd.Series:
    """
    计算指数移动平均 (EMA)
    """
    return series.ewm(span=periods, adjust=False).mean()


def highest(series: pd.Series, periods: int) -> pd.Series:
    """
    获取N周期内的最高价
    """
    return series.rolling(window=periods, min_periods=periods).max()


def lowest(series: pd.Series, periods: int) -> pd.Series:
    """
    获取N周期内的最低价
    """
    return series.rolling(window=periods, min_periods=periods).min()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, periods: int) -> pd.Series:
    """
    计算平均真实波幅 (ATR)
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    return ema(tr, periods)


def ensure_columns(data: pd.DataFrame, required_columns: List[str]) -> None:
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