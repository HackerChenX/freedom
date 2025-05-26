"""
TRIX三重指数平滑移动平均线模块

实现TRIX指标计算，用于过滤短期波动，捕捉中长期趋势
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class TRIX(BaseIndicator):
    """
    TRIX三重指数平滑移动平均线指标
    
    TRIX = (TR - REF(TR, 1)) / REF(TR, 1) × 100，其中TR = EMA(EMA(EMA(Close, N), N), N)
    过滤短期波动，捕捉中长期趋势变化
    """
    
    def __init__(self, n: int = 12, m: int = 9):
        """
        初始化TRIX指标
        
        Args:
            n: TRIX计算周期，默认为12
            m: 信号线周期，默认为9
        """
        super().__init__(name="TRIX", description="TRIX三重指数平滑移动平均线")
        self.n = n
        self.m = m
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算TRIX指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含TRIX指标的DataFrame
        """
        result = self.calculate(df, self.n, self.m)
        # 重命名列以符合标准
        result['trix'] = result['TRIX']
        result['signal'] = result['MATRIX']
        return result
    
    def sma(self, series: np.ndarray, n: int) -> np.ndarray:
        """
        计算简单移动平均
        
        Args:
            series: 输入序列
            n: 周期
            
        Returns:
            np.ndarray: SMA结果
        """
        result = np.zeros_like(series)
        
        for i in range(len(series)):
            if i < n:
                result[i] = np.mean(series[:i+1])
            else:
                result[i] = np.mean(series[i-n+1:i+1])
        
        return result
    
    def calculate(self, data: pd.DataFrame, n: int = 12, m: int = 9, *args, **kwargs) -> pd.DataFrame:
        """
        计算TRIX指标
        
        Args:
            data: 输入数据，包含收盘价
            n: TRIX计算周期，默认为12
            m: MATRIX信号线周期，默认为9
            
        Returns:
            pd.DataFrame: 计算结果，包含TRIX和MATRIX
            
        公式说明：
        TR:=EMA(EMA(EMA(CLOSE,12),12),12);
        TRIX:(TR-REF(TR,1))/REF(TR,1)*100;
        MATRIX:MA(TRIX,9);
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 提取数据
        close = data["close"].values
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算TR = EMA(EMA(EMA(Close, N), N), N)
        ema1 = self._ema(close, n)
        ema2 = self._ema(ema1, n)
        tr = self._ema(ema2, n)
        
        # 计算TRIX = (TR - REF(TR, 1)) / REF(TR, 1) × 100
        trix = np.zeros_like(close)
        for i in range(1, len(tr)):
            if tr[i-1] != 0:  # 防止除以零
                trix[i] = (tr[i] - tr[i-1]) / tr[i-1] * 100
        
        # 计算MATRIX = MA(TRIX, M)
        matrix = self.sma(trix, m)
        
        # 添加计算结果到数据框
        result["TR"] = tr
        result["TRIX"] = trix
        result["MATRIX"] = matrix
        
        return result
    
    def _ema(self, series: np.ndarray, n: int) -> np.ndarray:
        """
        计算指数移动平均
        
        Args:
            series: 输入序列
            n: 周期
            
        Returns:
            np.ndarray: EMA结果
        """
        alpha = 2 / (n + 1)
        result = np.zeros_like(series)
        result[0] = series[0]
        
        for i in range(1, len(series)):
            result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
        
        return result 