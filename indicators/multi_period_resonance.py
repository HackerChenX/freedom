"""
多周期共振分析模块

实现不同周期指标共振信号的识别和评级
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple, Callable
from enum import Enum

from indicators.base_indicator import BaseIndicator
from enums.kline_period import KlinePeriod
from utils.logger import get_logger

logger = get_logger(__name__)


class ResonanceLevel(Enum):
    """共振等级枚举"""
    NONE = 0        # 无共振
    WEAK = 1        # 弱共振：2个周期
    MEDIUM = 2      # 中等共振：3个周期
    STRONG = 3      # 强共振：4个及以上周期


class MULTI_PERIOD_RESONANCE(BaseIndicator):
    """
    多周期共振分析指标
    
    分析不同周期的技术指标是否产生同步信号，提高信号可靠性
    """
    
    def __init__(self):
        """初始化多周期共振分析指标"""
        super().__init__(name="MULTI_PERIOD_RESONANCE", description="多周期共振分析指标")
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算多周期共振指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含多周期共振指标的DataFrame
        """
        # 由于多周期共振需要多个周期的数据，我们在这里进行简化处理
        # 使用同一数据模拟不同周期
        result = pd.DataFrame(index=df.index)
        
        # 生成MA金叉信号
        ma_signal = self.ma_golden_cross_signal(df)
        
        # 生成MACD金叉信号
        macd_signal = self.macd_golden_cross_signal(df)
        
        # 生成KDJ金叉信号
        kdj_signal = self.kdj_golden_cross_signal(df)
        
        # 计算共振强度
        buy_strength = np.zeros(len(df))
        for i in range(len(df)):
            signals = [ma_signal[i], macd_signal[i], kdj_signal[i]]
            buy_strength[i] = sum(signals) / len(signals)
        
        # 生成买入信号
        buy_signal = buy_strength >= 0.5
        
        # 添加结果
        result['ma_signal'] = ma_signal
        result['macd_signal'] = macd_signal
        result['kdj_signal'] = kdj_signal
        result['buy_strength'] = buy_strength
        result['buy_signal'] = buy_signal
        
        return result
    
    def calculate(self, data_dict: Dict[KlinePeriod, pd.DataFrame], 
                  signal_func: Callable[[pd.DataFrame], np.ndarray], 
                  periods: List[KlinePeriod] = None, 
                  *args, **kwargs) -> pd.DataFrame:
        """
        计算多周期共振分析指标
        
        Args:
            data_dict: 不同周期的数据字典，键为周期枚举，值为对应周期的数据框
            signal_func: 信号生成函数，接受一个DataFrame，返回一个布尔型numpy数组
            periods: 要分析的周期列表，默认为[日线、60分钟、30分钟、15分钟]
            
        Returns:
            pd.DataFrame: 计算结果，包含共振信号和共振等级
        """
        # 设置默认周期
        if periods is None:
            periods = [KlinePeriod.DAILY, KlinePeriod.MINUTE_60, KlinePeriod.MINUTE_30, KlinePeriod.MINUTE_15]
        
        # 确保所有周期数据都存在
        available_periods = []
        for period in periods:
            if period in data_dict:
                available_periods.append(period)
            else:
                logger.warning(f"周期 {period.value} 的数据不存在，将被忽略")
        
        if not available_periods:
            raise ValueError("没有可用的周期数据")
        
        # 获取日线周期数据作为基准
        daily_data = data_dict.get(KlinePeriod.DAILY)
        if daily_data is None:
            # 使用可用周期的第一个作为基准
            base_period = available_periods[0]
            daily_data = data_dict[base_period]
            logger.warning(f"日线数据不存在，使用 {base_period.value} 作为基准")
        
        # 初始化结果数据框
        result = pd.DataFrame(index=daily_data.index)
        
        # 计算每个周期的信号
        period_signals = {}
        for period in available_periods:
            try:
                period_data = data_dict[period]
                # 计算该周期的信号
                signal = signal_func(period_data)
                # 确保信号长度与该周期数据长度一致
                if len(signal) != len(period_data):
                    logger.error(f"周期 {period.value} 的信号长度与数据长度不一致")
                    continue
                
                # 存储该周期的信号
                period_signals[period] = signal
                # 添加该周期的信号到结果
                result[f"signal_{period.value}"] = np.zeros(len(daily_data), dtype=bool)
                
                # 如果不是日线周期，需要将信号映射到日线周期
                if period != KlinePeriod.DAILY and period != available_periods[0]:
                    # 这里需要实现一个复杂的周期映射算法
                    # 简化起见，我们假设较小周期的最后一个信号可以作为当天的信号
                    # 实际应用中，可能需要根据时间戳进行更精确的映射
                    # 此处仅为示例，实际项目中需要根据具体数据结构调整
                    logger.warning(f"使用简化的周期映射方法，可能不够准确")
                    
                    # 示例映射：将当日最后一个信号作为日线信号
                    # 实际应用中应根据实际数据结构调整此处的映射逻辑
                    for i, date in enumerate(daily_data.index):
                        # 假设可以通过日期找到对应的分钟级数据
                        # 实际应用中需要根据具体的数据组织方式调整
                        # 此处仅为演示
                        if i < len(signal):
                            result.loc[date, f"signal_{period.value}"] = signal[i]
                else:
                    # 日线周期直接赋值
                    result[f"signal_{period.value}"] = signal
            except Exception as e:
                logger.error(f"计算周期 {period.value} 的信号时出错: {e}")
        
        # 计算共振信号
        resonance_count = np.zeros(len(daily_data), dtype=int)
        for period in period_signals:
            period_col = f"signal_{period.value}"
            if period_col in result.columns:
                resonance_count += result[period_col].astype(int)
        
        # 计算共振等级
        resonance_level = np.zeros(len(daily_data), dtype=int)
        for i in range(len(daily_data)):
            if resonance_count[i] == 0:
                resonance_level[i] = ResonanceLevel.NONE.value
            elif resonance_count[i] == 1:
                resonance_level[i] = ResonanceLevel.NONE.value  # 单周期不算共振
            elif resonance_count[i] == 2:
                resonance_level[i] = ResonanceLevel.WEAK.value
            elif resonance_count[i] == 3:
                resonance_level[i] = ResonanceLevel.MEDIUM.value
            else:
                resonance_level[i] = ResonanceLevel.STRONG.value
        
        # 添加共振信号和等级到结果
        result["resonance_count"] = resonance_count
        result["resonance_level"] = resonance_level
        # 生成总的共振信号：至少是弱共振
        result["resonance_signal"] = resonance_level >= ResonanceLevel.WEAK.value
        
        return result
    
    def ma_golden_cross_signal(self, data: pd.DataFrame, short_period: int = 5, long_period: int = 10) -> np.ndarray:
        """
        生成均线金叉信号
        
        Args:
            data: 输入数据
            short_period: 短期均线周期，默认为5
            long_period: 长期均线周期，默认为10
            
        Returns:
            np.ndarray: 金叉信号布尔数组
        """
        close = data["close"].values
        
        # 计算均线
        ma_short = self.sma(close, short_period)
        ma_long = self.sma(close, long_period)
        
        # 计算金叉信号
        cross_signal = np.zeros(len(close), dtype=bool)
        for i in range(1, len(close)):
            if ma_short[i-1] <= ma_long[i-1] and ma_short[i] > ma_long[i]:
                cross_signal[i] = True
        
        return cross_signal
    
    def macd_golden_cross_signal(self, data: pd.DataFrame) -> np.ndarray:
        """
        生成MACD金叉信号
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: MACD金叉信号布尔数组
        """
        close = data["close"].values
        
        # 计算MACD
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        dif = ema12 - ema26
        dea = self._ema(dif, 9)
        
        # 计算金叉信号
        cross_signal = np.zeros(len(close), dtype=bool)
        for i in range(1, len(close)):
            if dif[i-1] <= dea[i-1] and dif[i] > dea[i]:
                cross_signal[i] = True
        
        return cross_signal
    
    def kdj_golden_cross_signal(self, data: pd.DataFrame) -> np.ndarray:
        """
        生成KDJ金叉信号
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: KDJ金叉信号布尔数组
        """
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        
        # 计算KDJ
        n = 9
        m1 = 3
        m2 = 3
        
        # 计算RSV
        rsv = np.zeros_like(close)
        for i in range(len(close)):
            if i < n - 1:
                rsv[i] = 50
            else:
                llv = np.min(low[i-n+1:i+1])
                hhv = np.max(high[i-n+1:i+1])
                if hhv == llv:
                    rsv[i] = 50
                else:
                    rsv[i] = (close[i] - llv) / (hhv - llv) * 100
        
        # 计算K、D值
        k = self._sma(rsv, m1, 1)
        d = self._sma(k, m2, 1)
        
        # 计算金叉信号
        cross_signal = np.zeros(len(close), dtype=bool)
        for i in range(1, len(close)):
            if k[i-1] <= d[i-1] and k[i] > d[i]:
                cross_signal[i] = True
        
        return cross_signal
    
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