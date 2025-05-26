"""
ZXM体系买点指标模块

实现ZXM体系的5个买点指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMDailyMACD(BaseIndicator):
    """
    ZXM买点-日MACD指标
    
    判断日线MACD指标是否小于0.9
    """
    
    def __init__(self):
        """初始化ZXM买点-日MACD指标"""
        super().__init__(name="ZXMDailyMACD", description="ZXM买点-日MACD指标，判断日线MACD值是否小于0.9")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM买点-日MACD指标
        
        Args:
            data: 输入数据，包含收盘价数据
            
        Returns:
            pd.DataFrame: 计算结果，包含买点信号
            
        公式说明：
        DIFF:=EMA(CLOSE,12)-EMA(CLOSE,26);
        DEA:=EMA(DIFF,9);
        MACD:=2*(DIFF-DEA);
        xg:MACD<0.9
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
        
        # 计算买点信号
        xg = macd < 0.9
        
        # 添加计算结果到数据框
        result["EMA12"] = ema12
        result["EMA26"] = ema26
        result["DIFF"] = diff
        result["DEA"] = dea
        result["MACD"] = macd
        result["XG"] = xg
        
        return result


class ZXMTurnover(BaseIndicator):
    """
    ZXM买点-换手率指标
    
    判断日线换手率是否大于0.7%
    """
    
    def __init__(self):
        """初始化ZXM买点-换手率指标"""
        super().__init__(name="ZXMTurnover", description="ZXM买点-换手率指标，判断日线换手率是否大于0.7%")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM买点-换手率指标
        
        Args:
            data: 输入数据，包含成交量和流通股本数据
            
        Returns:
            pd.DataFrame: 计算结果，包含买点信号
            
        公式说明：
        换手:=VOL*100/CAPITAL>0.7;
        xg:换手;
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["volume", "capital"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算换手率
        turnover = data["volume"] * 100 / data["capital"]
        
        # 计算买点信号
        xg = turnover > 0.7
        
        # 添加计算结果到数据框
        result["Turnover"] = turnover
        result["XG"] = xg
        
        return result


class ZXMVolumeShrink(BaseIndicator):
    """
    ZXM买点-缩量指标
    
    判断成交量是否较2日平均成交量缩减10%以上
    """
    
    def __init__(self):
        """初始化ZXM买点-缩量指标"""
        super().__init__(name="ZXMVolumeShrink", description="ZXM买点-缩量指标，判断成交量是否明显缩量")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM买点-缩量指标
        
        Args:
            data: 输入数据，包含成交量数据
            
        Returns:
            pd.DataFrame: 计算结果，包含买点信号
            
        公式说明：
        VOL/MA(VOL,2)<0.9;
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["volume"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算2日均量
        ma_vol_2 = data["volume"].rolling(window=2).mean()
        
        # 计算量比
        vol_ratio = data["volume"] / ma_vol_2
        
        # 计算买点信号
        xg = vol_ratio < 0.9
        
        # 添加计算结果到数据框
        result["MA_VOL_2"] = ma_vol_2
        result["VOL_RATIO"] = vol_ratio
        result["XG"] = xg
        
        return result


class ZXMMACallback(BaseIndicator):
    """
    ZXM买点-回踩均线指标
    
    判断收盘价是否回踩至20日、30日、60日或120日均线的N%以内
    """
    
    def __init__(self, callback_percent: float = 4.0):
        """
        初始化ZXM买点-回踩均线指标
        
        Args:
            callback_percent: 回踩百分比，默认为4%
        """
        super().__init__(name="ZXMMACallback", description="ZXM买点-回踩均线指标，判断价格是否回踩至关键均线附近")
        self.callback_percent = callback_percent
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM买点-回踩均线指标
        
        Args:
            data: 输入数据，包含收盘价数据
            
        Returns:
            pd.DataFrame: 计算结果，包含买点信号
            
        公式说明：
        A20:=ABS((C/MA(C,20)-1)*100)<= N;
        A30:=ABS((C/MA(C,30)-1)*100)<= N;
        A60:=ABS((C/MA(C,60)-1)*100)<= N;
        A120:=ABS((C/MA(C,120)-1)*100)<= N;
        XG:A20 OR A30 OR A60 OR A120;
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算各均线
        ma20 = data["close"].rolling(window=20).mean()
        ma30 = data["close"].rolling(window=30).mean()
        ma60 = data["close"].rolling(window=60).mean()
        ma120 = data["close"].rolling(window=120).mean()
        
        # 计算收盘价与各均线的偏离百分比
        a20 = abs((data["close"] / ma20 - 1) * 100) <= self.callback_percent
        a30 = abs((data["close"] / ma30 - 1) * 100) <= self.callback_percent
        a60 = abs((data["close"] / ma60 - 1) * 100) <= self.callback_percent
        a120 = abs((data["close"] / ma120 - 1) * 100) <= self.callback_percent
        
        # 计算买点信号
        xg = a20 | a30 | a60 | a120
        
        # 添加计算结果到数据框
        result["MA20"] = ma20
        result["MA30"] = ma30
        result["MA60"] = ma60
        result["MA120"] = ma120
        result["A20"] = a20
        result["A30"] = a30
        result["A60"] = a60
        result["A120"] = a120
        result["XG"] = xg
        
        return result
    
    
class ZXMBSAbsorb(BaseIndicator):
    """
    ZXM买点-BS吸筹指标
    
    判断60分钟级别是否存在低位吸筹特征
    """
    
    def __init__(self):
        """初始化ZXM买点-BS吸筹指标"""
        super().__init__(name="ZXMBSAbsorb", description="ZXM买点-BS吸筹指标，判断60分钟级别是否存在低位吸筹特征")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM买点-BS吸筹指标
        
        Args:
            data: 输入数据，包含OHLC数据，需为60分钟级别数据
            
        Returns:
            pd.DataFrame: 计算结果，包含买点信号
            
        公式说明：
        V11:=3*SMA((C-LLV(L,55))/(HHV(H,55)-LLV(L,55))*100,5,1)-2*SMA(SMA((C-LLV(L,55))/(HHV(H,55)-LLV(L,55))*100,5,1),3,1);
        V12:=(EMA(V11,3)-REF(EMA(V11,3),1))/REF(EMA(V11,3),1)*100;
        AA:=(EMA(V11,3)<=13) AND FILTER((EMA(V11,3)<=13),15);
        BB:=(EMA(V11,3)<=13 AND V12>13) AND FILTER((EMA(V11,3)<=13 AND V12>13),10);
        XG:COUNT(AA OR BB,6)
        
        注意：这里的FILTER函数表示在过去N周期内至少出现一次该条件
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算LLV和HHV
        llv_55 = data["low"].rolling(window=55).min()
        hhv_55 = data["high"].rolling(window=55).max()
        
        # 计算RSV变种
        rsv_55 = pd.Series(np.zeros(len(data)), index=data.index)
        divisor = hhv_55 - llv_55
        valid_idx = divisor > 0
        rsv_55[valid_idx] = ((data["close"] - llv_55) / divisor * 100)[valid_idx]
        
        # 计算V11
        sma_rsv_5 = self._sma(rsv_55, 5, 1)
        sma_sma_3 = self._sma(sma_rsv_5, 3, 1)
        v11 = 3 * sma_rsv_5 - 2 * sma_sma_3
        
        # 计算V11的EMA
        ema_v11_3 = v11.ewm(span=3, adjust=False).mean()
        
        # 计算V12
        v12 = pd.Series(np.zeros(len(data)), index=data.index)
        valid_idx = ema_v11_3.shift(1) != 0
        v12[valid_idx] = ((ema_v11_3 - ema_v11_3.shift(1)) / ema_v11_3.shift(1) * 100)[valid_idx]
        
        # 计算AA和BB条件
        aa_base = ema_v11_3 <= 13
        aa_filter = pd.Series(np.zeros(len(data), dtype=bool), index=data.index)
        for i in range(15, len(data)):
            aa_filter.iloc[i] = np.any(aa_base.iloc[i-14:i+1])
        aa = aa_base & aa_filter
        
        bb_base = (ema_v11_3 <= 13) & (v12 > 13)
        bb_filter = pd.Series(np.zeros(len(data), dtype=bool), index=data.index)
        for i in range(10, len(data)):
            bb_filter.iloc[i] = np.any(bb_base.iloc[i-9:i+1])
        bb = bb_base & bb_filter
        
        # 计算XG：近6周期内AA或BB条件满足的次数
        xg = pd.Series(np.zeros(len(data), dtype=int), index=data.index)
        for i in range(6, len(data)):
            xg.iloc[i] = np.sum((aa | bb).iloc[i-5:i+1])
        
        # 添加计算结果到数据框
        result["V11"] = v11
        result["EMA_V11_3"] = ema_v11_3
        result["V12"] = v12
        result["AA"] = aa
        result["BB"] = bb
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