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
    
    def calculate(self, data: Union[pd.DataFrame, Dict[KlinePeriod, pd.DataFrame]], 
                  signal_func: Callable[[pd.DataFrame], np.ndarray] = None, 
                  periods: List[KlinePeriod] = None, 
                  *args, **kwargs) -> pd.DataFrame:
        """
        计算多周期共振分析指标
        
        Args:
            data: 可以是单一DataFrame或不同周期的数据字典
            signal_func: 信号生成函数，接受一个DataFrame，返回一个布尔型numpy数组
            periods: 要分析的周期列表，默认为[日线、60分钟、30分钟、15分钟]
            
        Returns:
            pd.DataFrame: 计算结果，包含共振信号和共振等级
        """
        # 如果输入是单一DataFrame，使用简化的分析方法
        if isinstance(data, pd.DataFrame):
            return self._calculate_single_dataframe(data, **kwargs)
        
        # 如果没有提供信号函数，抛出异常
        if signal_func is None:
            raise ValueError("多周期分析需要提供signal_func参数")
            
        # 处理多周期数据字典
        data_dict = data
        
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
        
        # 保存结果
        self._result = result
        
        return result
    
    def _calculate_single_dataframe(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        使用单一数据框计算简化的共振指标
        
        Args:
            data: 输入的K线数据DataFrame
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 计算结果
        """
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算不同指标的信号
        # 1. MA金叉信号
        ma_signal = self.ma_golden_cross_signal(data)
        
        # 2. MACD金叉信号 
        macd_signal = self.macd_golden_cross_signal(data)
        
        # 3. KDJ金叉信号
        kdj_signal = self.kdj_golden_cross_signal(data)
        
        # 4. RSI信号 (RSI < 30 为超卖信号)
        rsi_signal = np.zeros(len(data), dtype=bool)
        if 'close' in data.columns:
            close = data['close'].values
            rsi = self._calculate_rsi(close)
            rsi_signal = rsi < 30
        
        # 添加各个信号到结果
        result['ma_signal'] = ma_signal
        result['macd_signal'] = macd_signal
        result['kdj_signal'] = kdj_signal
        result['rsi_signal'] = rsi_signal
        
        # 计算共振信号
        signals = [ma_signal, macd_signal, kdj_signal, rsi_signal]
        resonance_count = np.zeros(len(data))
        for signal in signals:
            resonance_count += signal
        
        # 计算共振等级
        resonance_level = np.zeros(len(data), dtype=int)
        for i in range(len(data)):
            if resonance_count[i] == 0:
                resonance_level[i] = ResonanceLevel.NONE.value
            elif resonance_count[i] == 1:
                resonance_level[i] = ResonanceLevel.NONE.value  # 单指标不算共振
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
        
        # 保存结果
        self._result = result
        
        return result
        
    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        计算RSI指标
        
        Args:
            close: 收盘价数组
            period: 周期，默认为14
            
        Returns:
            np.ndarray: RSI值数组
        """
        # 计算价格变化
        delta = np.zeros(len(close))
        delta[1:] = close[1:] - close[:-1]
        
        # 分离上涨和下跌
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # 计算平均上涨和下跌
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))
        
        # 初始平均值
        if len(close) > period:
            avg_gain[period] = np.mean(gain[1:period+1])
            avg_loss[period] = np.mean(loss[1:period+1])
            
            # 后续值使用Wilder平滑方法
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        # 计算相对强度
        rs = np.zeros(len(close))
        valid_indices = (avg_loss != 0)
        rs[valid_indices] = avg_gain[valid_indices] / avg_loss[valid_indices]
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
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
    
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 初始化信号
        signals = {}
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
    
        # 在这里实现指标特定的信号生成逻辑
        # 此处提供默认实现
        
        if self._result is not None and "resonance_signal" in self._result.columns:
            # 使用共振信号作为买入信号
            for date, row in self._result.iterrows():
                if date in signals['buy_signal'].index:
                    signals['buy_signal'].loc[date] = row["resonance_signal"]
                    
                    # 设置信号强度，基于共振等级
                    if "resonance_level" in self._result.columns:
                        signals['signal_strength'].loc[date] = row["resonance_level"] * 25  # 0-75分
    
        return signals
        
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算指标原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分(0-100)
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)
    
        # 基于共振等级计算评分
        if "resonance_level" in self._result.columns:
            for date, row in self._result.iterrows():
                if date in score.index:
                    level = row["resonance_level"]
                    # 转换共振等级为评分（0-100）
                    if level == ResonanceLevel.NONE.value:
                        score.loc[date] = 25.0  # 无共振
                    elif level == ResonanceLevel.WEAK.value:
                        score.loc[date] = 50.0  # 弱共振
                    elif level == ResonanceLevel.MEDIUM.value:
                        score.loc[date] = 75.0  # 中等共振
                    elif level == ResonanceLevel.STRONG.value:
                        score.loc[date] = 90.0  # 强共振
        
        return score
    
    def calculate_score(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算最终评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
        
        Returns:
            float: 评分(0-100)
        """
        # 计算原始评分序列
        raw_scores = self.calculate_raw_score(data, **kwargs)
        
        # 取最近的评分作为最终评分
        if not raw_scores.empty:
            return raw_scores.iloc[-1]
        
        # 默认评分
        return 50.0
    
    def identify_partial_resonance(self, data_dict: Dict[KlinePeriod, pd.DataFrame], 
                                 signal_func: Callable[[pd.DataFrame], np.ndarray], 
                                 periods: List[KlinePeriod] = None,
                                 *args, **kwargs) -> pd.DataFrame:
        """
        识别部分周期共振
        
        Args:
            data_dict: 不同周期的数据字典
            signal_func: 信号生成函数
            periods: 要分析的周期列表
            
        Returns:
            pd.DataFrame: 部分共振结果
        """
        # 设置默认周期
        if periods is None:
            periods = [KlinePeriod.DAILY, KlinePeriod.MINUTE_60, KlinePeriod.MINUTE_30, KlinePeriod.MINUTE_15]
        
        # 确保所有周期数据都存在
        available_periods = []
        for period in periods:
            if period in data_dict:
                available_periods.append(period)
        
        if len(available_periods) < 2:
            logger.warning("周期数量不足，无法识别部分共振")
            return pd.DataFrame()
        
        # 获取日线周期数据作为基准
        daily_data = data_dict.get(KlinePeriod.DAILY)
        if daily_data is None:
            base_period = available_periods[0]
            daily_data = data_dict[base_period]
        
        # 初始化结果数据框
        result = pd.DataFrame(index=daily_data.index)
        
        # 计算每个周期的信号
        signals_by_period = {}
        for period in available_periods:
            try:
                period_data = data_dict[period]
                signal = signal_func(period_data)
                signals_by_period[period] = signal
            except Exception as e:
                logger.error(f"计算周期 {period.value} 的信号时出错: {e}")
        
        # 初始化部分共振信号
        result["partial_signal"] = False
        
        # 部分共振规则：
        # 1. 至少有两个周期产生同向信号
        # 2. 包含日线周期
        daily_period = KlinePeriod.DAILY
        
        if daily_period in signals_by_period:
            daily_signal = signals_by_period[daily_period]
            
            # 遍历每个交易日
            for i, date in enumerate(result.index):
                if i < len(daily_signal) and daily_signal[i]:
                    # 日线有信号，检查其他周期
                    matching_periods = 1  # 日线已经计入
                    
                    for period in signals_by_period:
                        if period != daily_period:
                            period_signal = signals_by_period[period]
                            if i < len(period_signal) and period_signal[i]:
                                matching_periods += 1
                    
                    # 至少有两个周期共振
                    if matching_periods >= 2:
                        result.loc[date, "partial_signal"] = True
        
        return result
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        获取指标形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 形态列表
        """
        patterns = []
        
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 分析共振形态
        if "resonance_level" in self._result.columns:
            # 寻找强共振形态
            for i, (date, row) in enumerate(self._result.iterrows()):
                level = row["resonance_level"]
                
                if level >= ResonanceLevel.MEDIUM.value:
                    # 计算持续天数
                    duration = 1
                    for j in range(i+1, len(self._result)):
                        next_level = self._result.iloc[j]["resonance_level"]
                        if next_level >= ResonanceLevel.MEDIUM.value:
                            duration += 1
                        else:
                            break
                    
                    # 创建形态
                    pattern = {
                        "name": f"多周期{'强' if level == ResonanceLevel.STRONG.value else '中等'}共振",
                        "start_date": date,
                        "end_date": self._result.index[min(i + duration - 1, len(self._result) - 1)],
                        "duration": duration,
                        "strength": level / ResonanceLevel.STRONG.value,  # 0-1之间的强度
                        "description": f"多周期{'强' if level == ResonanceLevel.STRONG.value else '中等'}共振持续{duration}天",
                        "type": "bullish" if level == ResonanceLevel.STRONG.value else "neutral"
                    }
                    
                    patterns.append(pattern)
                    
                    # 跳过已经处理的形态时间段
                    i += duration
        
        return patterns

# 创建符合驼峰命名法的别名，供导入使用
MultiPeriodResonance = MULTI_PERIOD_RESONANCE 