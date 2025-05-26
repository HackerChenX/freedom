"""
公式工具模块

提供各种技术指标计算的辅助函数
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)


def mergeList(list1, list2, n, func):
    """
    合并两个列表，应用给定的函数，返回最后n个元素
    
    Args:
        list1: 第一个列表
        list2: 第二个列表
        n: 要返回的元素数量
        func: 应用于两个列表对应元素的函数
        
    Returns:
        list: 结果列表的最后n个元素
    """
    if len(list1) != len(list2):
        raise Exception("列表长度不一致")
    result = []
    for i in range(len(list1)):
        result.append(func(list1[i], list2[i]))
    n = n if n < len(result) else len(result)
    return result[-n:]


def mergeAndGetLast(list1, list2, func):
    """
    合并两个列表，并返回应用函数后的最后一个元素
    
    Args:
        list1: 第一个列表
        list2: 第二个列表
        func: 应用于两个列表对应元素的函数
        
    Returns:
        最后一个元素的计算结果，或者在处理NaN时返回False
    """
    if len(list1) != len(list2):
        raise Exception("列表长度不一致")
    for i in range(len(list1)):
        if np.isnan(list1[i]) or np.isnan(list2[i]):
            continue
        if i + 1 == len(list1):
            return func(list1[i], list2[i])
    return False


def countListAllMatch(lst, n, m, func):
    """
    检查列表中最后n个元素是否有至少m个满足条件
    
    Args:
        lst: 要检查的列表
        n: 要检查的最后n个元素
        m: 要满足的最小数量
        func: 判断函数
        
    Returns:
        bool: 如果满足条件，返回True，否则返回False
    """
    count = 0
    for i in range(len(lst) - n, len(lst)):
        if func(lst[i]):
            count += 1
    return count >= m


def countListAnyMatch(lst, n, func):
    """
    检查列表中最后n个元素是否有任意一个满足条件
    
    Args:
        lst: 要检查的列表
        n: 要检查的最后n个元素
        func: 判断函数
        
    Returns:
        bool: 如果有任意元素满足条件，返回True，否则返回False
    """
    n = n if n < len(lst) else len(lst)
    for i in range(len(lst) - n, len(lst)):
        if func(lst[i]):
            return True
    return False


def ltn(n):
    """小于n的函数"""
    return lambda x: x < n


def lt():
    """小于函数"""
    return lambda x, y: x < y


def ge():
    """大于等于函数"""
    return lambda x, y: x >= y


def gt(n):
    """大于n的函数"""
    return lambda x: x > n


def countTrue(lst, n):
    """
    检查列表中True的数量是否大于等于n
    
    Args:
        lst: 要检查的列表
        n: 阈值
        
    Returns:
        bool: 如果True的数量大于等于n，返回True，否则返回False
    """
    return lst.count(True) >= n


def 合并K线(stock_data):
    """
    合并K线数据
    
    Args:
        stock_data: StockData对象
        
    Returns:
        tuple: (close, high, low, volume)
    """
    low_level_close = stock_data.close
    low_level_high = stock_data.high
    low_level_low = stock_data.low
    high_level_close = low_level_close[1::2]
    high = np.maximum(low_level_high[::2], low_level_high[1::2])
    close = low_level_close[1::2]
    low = np.minimum(low_level_low[::2], low_level_low[1::2])
    volume = low_level_low[::2]
    return close, high, low, volume


def ma(series: pd.Series, periods: Union[int, List[int]]) -> pd.DataFrame:
    """
    计算移动平均线
    
    Args:
        series: 数据序列，通常是收盘价
        periods: 周期或周期列表
        
    Returns:
        pd.DataFrame: 包含MA值的DataFrame
    """
    if isinstance(periods, int):
        periods = [periods]
    
    result = pd.DataFrame()
    
    for period in periods:
        column_name = f'MA{period}'
        result[column_name] = series.rolling(window=period).mean()
    
    return result


def ema(series: pd.Series, periods: Union[int, List[int]]) -> pd.DataFrame:
    """
    计算指数移动平均线
    
    Args:
        series: 数据序列，通常是收盘价
        periods: 周期或周期列表
        
    Returns:
        pd.DataFrame: 包含EMA值的DataFrame
    """
    if isinstance(periods, int):
        periods = [periods]
    
    result = pd.DataFrame()
    
    for period in periods:
        column_name = f'EMA{period}'
        result[column_name] = series.ewm(span=period, adjust=False).mean()
    
    return result


def macd(series: pd.Series, fast_period: int = 12, slow_period: int = 26, 
        signal_period: int = 9) -> pd.DataFrame:
    """
    计算MACD指标
    
    Args:
        series: 数据序列，通常是收盘价
        fast_period: 快线周期，默认为12
        slow_period: 慢线周期，默认为26
        signal_period: 信号线周期，默认为9
        
    Returns:
        pd.DataFrame: 包含MACD线、信号线和柱状图的DataFrame
    """
    # 计算快线和慢线
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    
    # 计算MACD线、信号线和柱状图
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    # 构建结果DataFrame
    result = pd.DataFrame({
        'MACD': macd_line,
        'SIGNAL': signal_line,
        'HISTOGRAM': histogram
    })
    
    return result


def rsi(series: pd.Series, periods: Union[int, List[int]] = 14) -> pd.DataFrame:
    """
    计算RSI指标
    
    Args:
        series: 数据序列，通常是收盘价
        periods: 周期或周期列表，默认为14
        
    Returns:
        pd.DataFrame: 包含RSI值的DataFrame
    """
    if isinstance(periods, int):
        periods = [periods]
    
    result = pd.DataFrame()
    
    for period in periods:
        # 计算价格变化
        delta = series.diff()
        
        # 分离上涨和下跌
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss
        
        # 计算平均上涨和平均下跌
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 计算相对强度
        rs = avg_gain / avg_loss
        
        # 计算RSI
        column_name = f'RSI{period}'
        result[column_name] = 100 - (100 / (1 + rs))
    
    return result


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    计算布林带指标
    
    Args:
        series: 数据序列，通常是收盘价
        period: 周期，默认为20
        std_dev: 标准差倍数，默认为2.0
        
    Returns:
        pd.DataFrame: 包含布林带上轨、中轨和下轨的DataFrame
    """
    # 计算中轨（简单移动平均线）
    middle_band = series.rolling(window=period).mean()
    
    # 计算标准差
    rolling_std = series.rolling(window=period).std()
    
    # 计算上轨和下轨
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    # 构建结果DataFrame
    result = pd.DataFrame({
        'BB_UPPER': upper_band,
        'BB_MIDDLE': middle_band,
        'BB_LOWER': lower_band,
        'BB_WIDTH': (upper_band - lower_band) / middle_band,
        'BB_PCT': (series - lower_band) / (upper_band - lower_band)
    })
    
    return result


def kdj(high: pd.Series, low: pd.Series, close: pd.Series, 
       n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """
    计算KDJ指标
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        n: RSV计算周期，默认为9
        m1: K值平滑因子，默认为3
        m2: D值平滑因子，默认为3
        
    Returns:
        pd.DataFrame: 包含K、D、J值的DataFrame
    """
    # 计算RSV
    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()
    rsv = 100 * ((close - low_n) / (high_n - low_n))
    
    # 计算K值（RSV的m1日EMA）
    k = rsv.ewm(span=m1, adjust=False).mean()
    
    # 计算D值（K的m2日EMA）
    d = k.ewm(span=m2, adjust=False).mean()
    
    # 计算J值
    j = 3 * k - 2 * d
    
    # 构建结果DataFrame
    result = pd.DataFrame({
        'K': k,
        'D': d,
        'J': j
    })
    
    return result


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算平均真实范围(ATR)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期，默认为14
        
    Returns:
        pd.Series: ATR值
    """
    # 计算真实范围
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # 计算ATR
    atr = tr.rolling(window=period).mean()
    
    return atr


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    计算能量潮(OBV)指标
    
    Args:
        close: 收盘价序列
        volume: 成交量序列
        
    Returns:
        pd.Series: OBV值
    """
    # 计算价格变化方向
    price_change = close.diff()
    
    # 根据价格变化方向给成交量赋予正负号
    obv_volume = np.where(price_change > 0, volume, 
                          np.where(price_change < 0, -volume, 0))
    
    # 计算OBV（累积求和）
    obv = pd.Series(obv_volume).cumsum()
    
    return obv


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算威廉指标(%R)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期，默认为14
        
    Returns:
        pd.Series: %R值
    """
    # 计算周期内最高价和最低价
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    # 计算%R
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    return williams_r


def calculate_indicator(df: pd.DataFrame, indicator: str, **kwargs) -> pd.DataFrame:
    """
    根据指标名称计算相应的技术指标
    
    Args:
        df: 输入数据，包含价格和成交量数据的DataFrame
        indicator: 指标名称
        kwargs: 指标参数
        
    Returns:
        pd.DataFrame: 包含指标值的DataFrame
        
    Raises:
        ValueError: 如果指标名称不支持
    """
    indicator = indicator.lower()
    
    try:
        if indicator == 'ma':
            periods = kwargs.get('periods', [5, 10, 20, 60])
            return ma(df['close'], periods)
        
        elif indicator == 'ema':
            periods = kwargs.get('periods', [5, 10, 20, 60])
            return ema(df['close'], periods)
        
        elif indicator == 'macd':
            fast_period = kwargs.get('fast_period', 12)
            slow_period = kwargs.get('slow_period', 26)
            signal_period = kwargs.get('signal_period', 9)
            return macd(df['close'], fast_period, slow_period, signal_period)
        
        elif indicator == 'rsi':
            periods = kwargs.get('periods', [14])
            return rsi(df['close'], periods)
        
        elif indicator == 'bollinger':
            period = kwargs.get('period', 20)
            std_dev = kwargs.get('std_dev', 2.0)
            return bollinger_bands(df['close'], period, std_dev)
        
        elif indicator == 'kdj':
            n = kwargs.get('n', 9)
            m1 = kwargs.get('m1', 3)
            m2 = kwargs.get('m2', 3)
            return kdj(df['high'], df['low'], df['close'], n, m1, m2)
        
        elif indicator == 'atr':
            period = kwargs.get('period', 14)
            return pd.DataFrame({'ATR': atr(df['high'], df['low'], df['close'], period)})
        
        elif indicator == 'obv':
            return pd.DataFrame({'OBV': obv(df['close'], df['volume'])})
        
        elif indicator == 'williams_r':
            period = kwargs.get('period', 14)
            return pd.DataFrame({'WILLIAMS_R': williams_r(df['high'], df['low'], df['close'], period)})
        
        else:
            raise ValueError(f"不支持的指标: {indicator}")
    
    except Exception as e:
        logger.error(f"计算指标 {indicator} 时出错: {e}")
        raise 