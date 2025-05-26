"""
量价背离指标模块

实现量价背离识别和分析功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class DivergenceType(Enum):
    """背离类型枚举"""
    NONE = 0             # 无背离
    POSITIVE = 1         # 正背离（底背离）：价格创新低，指标未创新低，看涨信号
    NEGATIVE = 2         # 负背离（顶背离）：价格创新高，指标未创新高，看跌信号
    HIDDEN_POSITIVE = 3  # 隐藏正背离：价格未创新低，指标创新低，看涨信号
    HIDDEN_NEGATIVE = 4  # 隐藏负背离：价格未创新高，指标创新高，看跌信号


class DIVERGENCE(BaseIndicator):
    """
    量价背离指标
    
    用于识别价格与技术指标之间的背离，预示可能的趋势反转
    """
    
    def __init__(self, lookback_period: int = 20, confirm_period: int = 5):
        """
        初始化量价背离指标
        
        Args:
            lookback_period: 回溯周期，默认为20
            confirm_period: 确认周期，默认为5
        """
        super().__init__(name="DIVERGENCE", description="量价背离指标")
        self.lookback_period = lookback_period
        self.confirm_period = confirm_period
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算量价背离指标
        
        Args:
            df: 输入数据，包含价格和成交量数据
                
        Returns:
            包含量价背离指标的DataFrame
        """
        # 计算价格与成交量背离
        return self.price_volume_divergence(df, self.lookback_period, self.confirm_period)
    
    def price_volume_divergence(self, data: pd.DataFrame, lookback_period: int = 20, confirm_period: int = 5) -> pd.DataFrame:
        """
        计算价格与成交量的背离
        
        Args:
            data: 输入数据，包含价格和成交量数据
            lookback_period: 回溯周期，默认为20
            confirm_period: 确认周期，默认为5
            
        Returns:
            pd.DataFrame: 计算结果
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "volume"])
        
        # 提取数据
        close = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数据框
        result = data.copy()
        
        # 初始化背离标志
        price_volume_divergence = np.zeros(len(close), dtype=bool)
        divergence_type = np.full(len(close), "none", dtype=object)
        
        # 计算价格与成交量背离
        for i in range(lookback_period, len(close)):
            # 价格上涨但成交量下降，负背离
            if i >= 5 and close[i] > close[i-5] and volume[i] < volume[i-5]:
                price_volume_divergence[i] = True
                divergence_type[i] = "negative"
            
            # 价格下跌但成交量上升，正背离
            elif i >= 5 and close[i] < close[i-5] and volume[i] > volume[i-5]:
                price_volume_divergence[i] = True
                divergence_type[i] = "positive"
        
        # 添加计算结果到数据框
        result["price_volume_divergence"] = price_volume_divergence
        result["divergence_type"] = divergence_type
        
        return result
    
    def calculate(self, data: pd.DataFrame, indicator_name: str, 
                  lookback_period: int = 20, confirm_period: int = 5, 
                  *args, **kwargs) -> pd.DataFrame:
        """
        计算量价背离指标
        
        Args:
            data: 输入数据，包含价格和技术指标数据
            indicator_name: 用于对比的技术指标列名
            lookback_period: 回溯周期，默认为20
            confirm_period: 确认周期，默认为5
            
        Returns:
            pd.DataFrame: 计算结果，包含各类背离信号
            
        公式说明：
        PRICE_NEWLOW:=LOW=LLV(LOW,N);
        MACD_NO_NEWLOW:=MACD>LLV(MACD,N);
        POSITIVE_DIVERGENCE:=PRICE_NEWLOW AND MACD_NO_NEWLOW;

        PRICE_NEWHIGH:=HIGH=HHV(HIGH,N);
        MACD_NO_NEWHIGH:=MACD<HHV(MACD,N);
        NEGATIVE_DIVERGENCE:=PRICE_NEWHIGH AND MACD_NO_NEWHIGH;
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low", indicator_name])
        
        # 提取数据
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        indicator = data[indicator_name].values
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算价格新高新低
        price_newlow = np.zeros(len(close), dtype=bool)
        price_newhigh = np.zeros(len(close), dtype=bool)
        
        # 计算指标新高新低
        indicator_newlow = np.zeros(len(close), dtype=bool)
        indicator_newhigh = np.zeros(len(close), dtype=bool)
        
        # 初始化各类背离
        positive_divergence = np.zeros(len(close), dtype=bool)  # 正背离（底背离）
        negative_divergence = np.zeros(len(close), dtype=bool)  # 负背离（顶背离）
        hidden_positive_divergence = np.zeros(len(close), dtype=bool)  # 隐藏正背离
        hidden_negative_divergence = np.zeros(len(close), dtype=bool)  # 隐藏负背离
        
        # 计算各类背离
        for i in range(lookback_period, len(close)):
            # 计算价格新低：当前低点是否是lookback_period周期内的最低点
            price_newlow[i] = low[i] == np.min(low[i-lookback_period+1:i+1])
            
            # 计算价格新高：当前高点是否是lookback_period周期内的最高点
            price_newhigh[i] = high[i] == np.max(high[i-lookback_period+1:i+1])
            
            # 计算指标新低：当前指标是否是lookback_period周期内的最低点
            indicator_newlow[i] = indicator[i] == np.min(indicator[i-lookback_period+1:i+1])
            
            # 计算指标新高：当前指标是否是lookback_period周期内的最高点
            indicator_newhigh[i] = indicator[i] == np.max(indicator[i-lookback_period+1:i+1])
            
            # 计算正背离（底背离）：价格创新低，但指标未创新低
            if price_newlow[i]:
                # 检查指标是否背离（未创新低）
                indicator_value = indicator[i]
                min_indicator = np.min(indicator[i-confirm_period:i])
                if indicator_value > min_indicator:
                    positive_divergence[i] = True
            
            # 计算负背离（顶背离）：价格创新高，但指标未创新高
            if price_newhigh[i]:
                # 检查指标是否背离（未创新高）
                indicator_value = indicator[i]
                max_indicator = np.max(indicator[i-confirm_period:i])
                if indicator_value < max_indicator:
                    negative_divergence[i] = True
            
            # 计算隐藏正背离：价格未创新低，但指标创新低
            if not price_newlow[i] and indicator_newlow[i]:
                # 检查价格是否在上升趋势中（近期低点高于前期低点）
                current_low = low[i]
                previous_low = np.min(low[i-confirm_period:i])
                if current_low > previous_low:
                    hidden_positive_divergence[i] = True
            
            # 计算隐藏负背离：价格未创新高，但指标创新高
            if not price_newhigh[i] and indicator_newhigh[i]:
                # 检查价格是否在下降趋势中（近期高点低于前期高点）
                current_high = high[i]
                previous_high = np.max(high[i-confirm_period:i])
                if current_high < previous_high:
                    hidden_negative_divergence[i] = True
        
        # 添加计算结果到数据框
        result["price_newlow"] = price_newlow
        result["price_newhigh"] = price_newhigh
        result["indicator_newlow"] = indicator_newlow
        result["indicator_newhigh"] = indicator_newhigh
        result["positive_divergence"] = positive_divergence  # 正背离（底背离）
        result["negative_divergence"] = negative_divergence  # 负背离（顶背离）
        result["hidden_positive_divergence"] = hidden_positive_divergence  # 隐藏正背离
        result["hidden_negative_divergence"] = hidden_negative_divergence  # 隐藏负背离
        
        # 计算背离类型
        divergence_type = np.zeros(len(close), dtype=int)
        for i in range(len(close)):
            if positive_divergence[i]:
                divergence_type[i] = DivergenceType.POSITIVE.value
            elif negative_divergence[i]:
                divergence_type[i] = DivergenceType.NEGATIVE.value
            elif hidden_positive_divergence[i]:
                divergence_type[i] = DivergenceType.HIDDEN_POSITIVE.value
            elif hidden_negative_divergence[i]:
                divergence_type[i] = DivergenceType.HIDDEN_NEGATIVE.value
            else:
                divergence_type[i] = DivergenceType.NONE.value
        
        result["divergence_type"] = divergence_type
        
        return result
    
    def macd_divergence(self, data: pd.DataFrame, lookback_period: int = 20, confirm_period: int = 5) -> pd.DataFrame:
        """
        计算MACD与价格的背离
        
        Args:
            data: 输入数据，包含价格和MACD数据
            lookback_period: 回溯周期，默认为20
            confirm_period: 确认周期，默认为5
            
        Returns:
            pd.DataFrame: 计算结果
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 如果数据中没有MACD，计算MACD
        if "MACD" not in data.columns:
            # 计算MACD
            close = data["close"].values
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            dif = ema12 - ema26
            dea = self._ema(dif, 9)
            macd = 2 * (dif - dea)
            
            # 添加MACD到数据
            data_with_macd = data.copy()
            data_with_macd["MACD"] = macd
        else:
            data_with_macd = data
        
        return self.calculate(data_with_macd, "MACD", lookback_period, confirm_period)
    
    def rsi_divergence(self, data: pd.DataFrame, period: int = 14, 
                      lookback_period: int = 20, confirm_period: int = 5) -> pd.DataFrame:
        """
        计算RSI与价格的背离
        
        Args:
            data: 输入数据，包含价格数据
            period: RSI周期，默认为14
            lookback_period: 回溯周期，默认为20
            confirm_period: 确认周期，默认为5
            
        Returns:
            pd.DataFrame: 计算结果
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 如果数据中没有RSI，计算RSI
        if f"RSI_{period}" not in data.columns:
            # 计算RSI
            close = data["close"].values
            delta = np.zeros(len(close))
            delta[1:] = close[1:] - close[:-1]
            
            up = np.zeros_like(delta)
            down = np.zeros_like(delta)
            up[delta > 0] = delta[delta > 0]
            down[delta < 0] = -delta[delta < 0]
            
            roll_up = np.zeros_like(up)
            roll_down = np.zeros_like(down)
            roll_up[0] = up[0]
            roll_down[0] = down[0]
            
            alpha = 1 / period
            for i in range(1, len(close)):
                roll_up[i] = roll_up[i-1] * (1 - alpha) + up[i] * alpha
                roll_down[i] = roll_down[i-1] * (1 - alpha) + down[i] * alpha
            
            rs = np.zeros_like(close)
            for i in range(len(close)):
                if roll_down[i] != 0:
                    rs[i] = roll_up[i] / roll_down[i]
                else:
                    rs[i] = 100.0
            
            rsi = 100 - (100 / (1 + rs))
            
            # 添加RSI到数据
            data_with_rsi = data.copy()
            data_with_rsi[f"RSI_{period}"] = rsi
        else:
            data_with_rsi = data
        
        return self.calculate(data_with_rsi, f"RSI_{period}", lookback_period, confirm_period)
    
    def obv_divergence(self, data: pd.DataFrame, 
                      lookback_period: int = 20, confirm_period: int = 5) -> pd.DataFrame:
        """
        计算OBV与价格的背离
        
        Args:
            data: 输入数据，包含价格和成交量数据
            lookback_period: 回溯周期，默认为20
            confirm_period: 确认周期，默认为5
            
        Returns:
            pd.DataFrame: 计算结果
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low", "volume"])
        
        # 如果数据中没有OBV，计算OBV
        if "OBV" not in data.columns:
            # 计算OBV
            close = data["close"].values
            volume = data["volume"].values
            
            obv = np.zeros_like(close)
            
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            
            # 添加OBV到数据
            data_with_obv = data.copy()
            data_with_obv["OBV"] = obv
        else:
            data_with_obv = data
        
        return self.calculate(data_with_obv, "OBV", lookback_period, confirm_period)
    
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