"""
ZXM核心吸筹公式模块

实现ZXM体系的核心吸筹信号识别算法
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXM_ABSORB(BaseIndicator):
    """
    ZXM核心吸筹公式指标
    
    基于KDJ衍生指标的低位区域信号，识别主力在低位吸筹动作
    """
    
    def __init__(self):
        """初始化ZXM核心吸筹指标"""
        super().__init__(name="ZXM_ABSORB", description="ZXM核心吸筹指标，用于识别主力低位吸筹动作")
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算ZXM核心吸筹指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含ZXM核心吸筹指标的DataFrame
        """
        result = self.calculate(df)
        result['absorb_signal'] = result['BUY']
        result['absorb_strength'] = result['XG']
        
        # 修改调试日志，避免百分号
        if len(result) > 0:
            logger.info(f"absorb_strength值: {result['absorb_strength'].iloc[-1]}, 类型: {type(result['absorb_strength'].iloc[-1])}")
            
        return result
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM核心吸筹指标
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含核心吸筹信号
            
        公式说明：
        V11:=3*SMA((CLOSE-LLV(LOW,55))/(HHV(HIGH,55)-LLV(LOW,55))*100,5,1)-2*SMA(SMA((CLOSE-LLV(LOW,55))/(HHV(HIGH,55)-LLV(LOW,55))*100,5,1),3,1);
        V12:=(EMA(V11,3)-REF(EMA(V11,3),1))/REF(EMA(V11,3),1)*100;
        AA:=EMA(V11,3)<=13;
        BB:=EMA(V11,3)<=13 AND V12>13;
        XC:=COUNT(AA,15)>=10; // 满足低位条件
        XD:=COUNT(BB,10)>=5;  // 满足低位回升条件
        XG:=COUNT(AA OR BB,6); // 近期满足吸筹条件次数
        BUY:=XG>=3; // 满足3次以上视为有效吸筹信号
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 提取数据
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算V11 - KDJ衍生指标
        llv_55 = self._llv(low, 55)
        hhv_55 = self._hhv(high, 55)
        
        # 计算RSV变种
        rsv_55 = np.zeros_like(close)
        for i in range(len(close)):
            if hhv_55[i] - llv_55[i] != 0:
                rsv_55[i] = (close[i] - llv_55[i]) / (hhv_55[i] - llv_55[i]) * 100
            else:
                rsv_55[i] = 50  # 防止除以零
        
        # 计算SMA
        sma_rsv_5 = self._sma(rsv_55, 5, 1)
        sma_sma_3 = self._sma(sma_rsv_5, 3, 1)
        
        # 计算V11
        v11 = 3 * sma_rsv_5 - 2 * sma_sma_3
        
        # 计算V11的EMA
        ema_v11_3 = self._ema(v11, 3)
        
        # 计算V12 - 动量指标
        v12 = np.zeros_like(close)
        for i in range(1, len(ema_v11_3)):
            if ema_v11_3[i-1] != 0:
                v12[i] = (ema_v11_3[i] - ema_v11_3[i-1]) / ema_v11_3[i-1] * 100
        
        # 定义AA和BB条件
        aa = ema_v11_3 <= 13
        bb = (ema_v11_3 <= 13) & (v12 > 13)
        
        # 计算满足条件的次数
        xc = np.zeros_like(close, dtype=bool)
        xd = np.zeros_like(close, dtype=bool)
        xg = np.zeros_like(close, dtype=int)
        buy = np.zeros_like(close, dtype=bool)
        
        for i in range(15, len(close)):
            # 计算XC：近15天内AA条件满足10次以上
            xc[i] = np.sum(aa[i-14:i+1]) >= 10
            
            # 计算XD：近10天内BB条件满足5次以上
            if i >= 10:
                xd[i] = np.sum(bb[i-9:i+1]) >= 5
            
            # 计算XG：近6天内AA或BB条件满足的次数
            if i >= 6:
                xg[i] = np.sum(aa[i-5:i+1] | bb[i-5:i+1])
            
            # 计算BUY：XG >= 3
            buy[i] = xg[i] >= 3
        
        # 添加计算结果到数据框
        result["V11"] = v11
        result["V12"] = v12
        result["EMA_V11_3"] = ema_v11_3
        result["AA"] = aa
        result["BB"] = bb
        result["XC"] = xc
        result["XD"] = xd
        result["XG"] = xg
        result["BUY"] = buy
        
        return result
    
    def _sma(self, series: np.ndarray, n: int, m: int) -> np.ndarray:
        """
        计算简单移动平均
        
        Args:
            series: 输入序列
            n: 周期
            m: 权重
            
        Returns:
            np.ndarray: SMA结果
        """
        result = np.zeros_like(series)
        result[0] = series[0]
        
        for i in range(1, len(series)):
            result[i] = (m * series[i] + (n - m) * result[i-1]) / n
        
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
    
    def _llv(self, series: np.ndarray, n: int) -> np.ndarray:
        """
        计算n周期内最低值
        
        Args:
            series: 输入序列
            n: 周期
            
        Returns:
            np.ndarray: 最低值序列
        """
        result = np.zeros_like(series)
        
        for i in range(len(series)):
            if i < n:
                result[i] = np.min(series[:i+1])
            else:
                result[i] = np.min(series[i-n+1:i+1])
        
        return result
    
    def _hhv(self, series: np.ndarray, n: int) -> np.ndarray:
        """
        计算n周期内最高值
        
        Args:
            series: 输入序列
            n: 周期
            
        Returns:
            np.ndarray: 最高值序列
        """
        result = np.zeros_like(series)
        
        for i in range(len(series)):
            if i < n:
                result[i] = np.max(series[:i+1])
            else:
                result[i] = np.max(series[i-n+1:i+1])
        
        return result 