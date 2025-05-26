"""
ZXM体系趋势识别指标模块

实现ZXM体系的7个趋势识别指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMDailyTrendUp(BaseIndicator):
    """
    ZXM趋势-日线上移指标
    
    判断60日或120日均线是否向上移动
    """
    
    def __init__(self):
        """初始化ZXM趋势-日线上移指标"""
        super().__init__(name="ZXMDailyTrendUp", description="ZXM趋势-日线上移指标，判断日线均线是否向上")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-日线上移指标
        
        Args:
            data: 输入数据，包含收盘价数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        j1:MA(C,60)>=REF(MA(C,60),1);
        j2:MA(C,120)>=REF(MA(C,120),1);
        xg:j1 OR j2;
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算60日均线和120日均线
        ma60 = data["close"].rolling(window=60).mean()
        ma120 = data["close"].rolling(window=120).mean()
        
        # 计算均线是否上移
        j1 = ma60 >= ma60.shift(1)
        j2 = ma120 >= ma120.shift(1)
        
        # 计算趋势信号
        xg = j1 | j2
        
        # 添加计算结果到数据框
        result["MA60"] = ma60
        result["MA120"] = ma120
        result["J1"] = j1
        result["J2"] = j2
        result["XG"] = xg
        
        return result


class ZXMWeeklyTrendUp(BaseIndicator):
    """
    ZXM趋势-周线上移指标
    
    判断周线10周、20周或30周均线是否向上移动
    """
    
    def __init__(self):
        """初始化ZXM趋势-周线上移指标"""
        super().__init__(name="ZXMWeeklyTrendUp", description="ZXM趋势-周线上移指标，判断周线均线是否向上")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-周线上移指标
        
        Args:
            data: 输入数据，包含收盘价数据，需为周线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        a1:MA(C,10)>=REF(MA(C,10),1);
        b1:MA(C,20)>=REF(MA(C,20),1);
        c1:MA(C,30)>=REF(MA(C,30),1);
        xg:a1 OR b1 OR c1;
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算10周、20周和30周均线
        ma10 = data["close"].rolling(window=10).mean()
        ma20 = data["close"].rolling(window=20).mean()
        ma30 = data["close"].rolling(window=30).mean()
        
        # 计算均线是否上移
        a1 = ma10 >= ma10.shift(1)
        b1 = ma20 >= ma20.shift(1)
        c1 = ma30 >= ma30.shift(1)
        
        # 计算趋势信号
        xg = a1 | b1 | c1
        
        # 添加计算结果到数据框
        result["MA10"] = ma10
        result["MA20"] = ma20
        result["MA30"] = ma30
        result["A1"] = a1
        result["B1"] = b1
        result["C1"] = c1
        result["XG"] = xg
        
        return result


class ZXMMonthlyKDJTrendUp(BaseIndicator):
    """
    ZXM趋势-月KDJ·D及K上移指标
    
    判断月线KDJ指标的D值和K值是否同时向上移动
    """
    
    def __init__(self):
        """初始化ZXM趋势-月KDJ·D及K上移指标"""
        super().__init__(name="ZXMMonthlyKDJTrendUp", description="ZXM趋势-月KDJ·D及K上移指标，判断月线KDJ·D和K值是否向上")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-月KDJ·D及K上移指标
        
        Args:
            data: 输入数据，包含OHLC数据，需为月线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
        K:=SMA(RSV,3,1);
        D:=SMA(K,3,1);
        J:=3*K-2*D;
        xg:D>=REF(D,1) AND K>=REF(K,1);
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算KDJ指标
        low_9 = data["low"].rolling(window=9).min()
        high_9 = data["high"].rolling(window=9).max()
        
        # 计算RSV，处理除零情况
        rsv = pd.Series(np.zeros(len(data)), index=data.index)
        divisor = high_9 - low_9
        valid_idx = divisor > 0
        rsv[valid_idx] = ((data["close"] - low_9) / divisor * 100)[valid_idx]
        
        # 计算K、D、J值
        k = self._sma(rsv, 3, 1)
        d = self._sma(k, 3, 1)
        j = 3 * k - 2 * d
        
        # 计算趋势信号
        xg = (d >= d.shift(1)) & (k >= k.shift(1))
        
        # 添加计算结果到数据框
        result["RSV"] = rsv
        result["K"] = k
        result["D"] = d
        result["J"] = j
        result["XG"] = xg
        
        return result
    
    def _sma(self, series: pd.Series, n: int, m: int) -> pd.Series:
        """
        计算SMA(简单移动平均)
        
        Args:
            series: 输入序列
            n: 周期
            m: 权重
            
        Returns:
            pd.Series: SMA结果
        """
        result = pd.Series(np.zeros(len(series)), index=series.index)
        result.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
        
        return result


class ZXMWeeklyKDJDOrDEATrendUp(BaseIndicator):
    """
    ZXM趋势-周KDJ·D/DEA上移指标
    
    判断周线KDJ指标的D值或MACD的DEA值是否有一个向上移动
    """
    
    def __init__(self):
        """初始化ZXM趋势-周KDJ·D/DEA上移指标"""
        super().__init__(name="ZXMWeeklyKDJDOrDEATrendUp", description="ZXM趋势-周KDJ·D/DEA上移指标，判断周线KDJ·D或DEA值是否向上")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-周KDJ·D/DEA上移指标
        
        Args:
            data: 输入数据，包含OHLC数据，需为周线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
        K:=SMA(RSV,3,1);
        D:=SMA(K,3,1);
        J:=3*K-2*D;
        DIFF:=EMA(CLOSE,12)-EMA(CLOSE,26);
        DEA:=EMA(DIFF,9);
        xg:D>=REF(D,1) OR DEA>=REF(DEA,1);
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算KDJ指标
        low_9 = data["low"].rolling(window=9).min()
        high_9 = data["high"].rolling(window=9).max()
        
        # 计算RSV，处理除零情况
        rsv = pd.Series(np.zeros(len(data)), index=data.index)
        divisor = high_9 - low_9
        valid_idx = divisor > 0
        rsv[valid_idx] = ((data["close"] - low_9) / divisor * 100)[valid_idx]
        
        # 计算K、D、J值
        k = self._sma(rsv, 3, 1)
        d = self._sma(k, 3, 1)
        j = 3 * k - 2 * d
        
        # 计算MACD指标
        ema12 = data["close"].ewm(span=12, adjust=False).mean()
        ema26 = data["close"].ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        
        # 计算趋势信号
        xg = (d >= d.shift(1)) | (dea >= dea.shift(1))
        
        # 添加计算结果到数据框
        result["RSV"] = rsv
        result["K"] = k
        result["D"] = d
        result["J"] = j
        result["DIFF"] = diff
        result["DEA"] = dea
        result["XG"] = xg
        
        return result
    
    def _sma(self, series: pd.Series, n: int, m: int) -> pd.Series:
        """
        计算SMA(简单移动平均)
        
        Args:
            series: 输入序列
            n: 周期
            m: 权重
            
        Returns:
            pd.Series: SMA结果
        """
        result = pd.Series(np.zeros(len(series)), index=series.index)
        result.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
        
        return result


class ZXMWeeklyKDJDTrendUp(BaseIndicator):
    """
    ZXM趋势-周KDJ·D上移指标
    
    判断周线KDJ指标的D值是否向上移动
    """
    
    def __init__(self):
        """初始化ZXM趋势-周KDJ·D上移指标"""
        super().__init__(name="ZXMWeeklyKDJDTrendUp", description="ZXM趋势-周KDJ·D上移指标，判断周线KDJ·D值是否向上")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-周KDJ·D上移指标
        
        Args:
            data: 输入数据，包含OHLC数据，需为周线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
        K:=SMA(RSV,3,1);
        D:=SMA(K,3,1);
        J:=3*K-2*D;
        xg:D>=REF(D,1);
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算KDJ指标
        low_9 = data["low"].rolling(window=9).min()
        high_9 = data["high"].rolling(window=9).max()
        
        # 计算RSV，处理除零情况
        rsv = pd.Series(np.zeros(len(data)), index=data.index)
        divisor = high_9 - low_9
        valid_idx = divisor > 0
        rsv[valid_idx] = ((data["close"] - low_9) / divisor * 100)[valid_idx]
        
        # 计算K、D、J值
        k = self._sma(rsv, 3, 1)
        d = self._sma(k, 3, 1)
        j = 3 * k - 2 * d
        
        # 计算趋势信号
        xg = d >= d.shift(1)
        
        # 添加计算结果到数据框
        result["RSV"] = rsv
        result["K"] = k
        result["D"] = d
        result["J"] = j
        result["XG"] = xg
        
        return result
    
    def _sma(self, series: pd.Series, n: int, m: int) -> pd.Series:
        """
        计算SMA(简单移动平均)
        
        Args:
            series: 输入序列
            n: 周期
            m: 权重
            
        Returns:
            pd.Series: SMA结果
        """
        result = pd.Series(np.zeros(len(series)), index=series.index)
        result.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
        
        return result


class ZXMMonthlyMACD(BaseIndicator):
    """
    ZXM趋势-月MACD<1.5指标
    
    判断月线MACD指标是否小于1.5
    """
    
    def __init__(self):
        """初始化ZXM趋势-月MACD<1.5指标"""
        super().__init__(name="ZXMMonthlyMACD", description="ZXM趋势-月MACD<1.5指标，判断月线MACD值是否小于1.5")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-月MACD<1.5指标
        
        Args:
            data: 输入数据，包含收盘价数据，需为月线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        DIFF:=EMA(CLOSE,12)-EMA(CLOSE,26);
        DEA:=EMA(DIFF,9);
        MACD:=2*(DIFF-DEA);
        xg:MACD<1.5
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算MACD指标
        ema12 = data["close"].ewm(span=12, adjust=False).mean()
        ema26 = data["close"].ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd = 2 * (diff - dea)
        
        # 计算趋势信号
        xg = macd < 1.5
        
        # 添加计算结果到数据框
        result["EMA12"] = ema12
        result["EMA26"] = ema26
        result["DIFF"] = diff
        result["DEA"] = dea
        result["MACD"] = macd
        result["XG"] = xg
        
        return result


class ZXMWeeklyMACD(BaseIndicator):
    """
    ZXM趋势-周MACD<2指标
    
    判断周线MACD指标是否小于2
    """
    
    def __init__(self):
        """初始化ZXM趋势-周MACD<2指标"""
        super().__init__(name="ZXMWeeklyMACD", description="ZXM趋势-周MACD<2指标，判断周线MACD值是否小于2")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-周MACD<2指标
        
        Args:
            data: 输入数据，包含收盘价数据，需为周线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        DIFF:=EMA(CLOSE,12)-EMA(CLOSE,26);
        DEA:=EMA(DIFF,9);
        MACD:=2*(DIFF-DEA);
        xg:MACD<2
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算MACD指标
        ema12 = data["close"].ewm(span=12, adjust=False).mean()
        ema26 = data["close"].ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd = 2 * (diff - dea)
        
        # 计算趋势信号
        xg = macd < 2
        
        # 添加计算结果到数据框
        result["EMA12"] = ema12
        result["EMA26"] = ema26
        result["DIFF"] = diff
        result["DEA"] = dea
        result["MACD"] = macd
        result["XG"] = xg
        
        return result 