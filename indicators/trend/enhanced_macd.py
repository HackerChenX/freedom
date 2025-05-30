"""
增强型MACD指标模块

实现改进版的MACD指标，优化计算方法和信号质量，增加多周期适应能力和市场环境感知
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator, MarketEnvironment, SignalStrength
from indicators.macd import MACD
from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedMACD(MACD):
    """
    增强型MACD指标
    
    在标准MACD基础上增加了动态周期调整、多周期分析、柱状体变化率分析、趋势强度评估等功能
    """
    
    def __init__(self, 
                fast_period: int = 12, 
                slow_period: int = 26, 
                signal_period: int = 9,
                sensitivity: float = 1.0,
                multi_periods: List[Tuple[int, int, int]] = None,
                volume_weighted: bool = False,
                adapt_to_volatility: bool = True):
        """
        初始化增强型MACD指标
        
        Args:
            fast_period: 快速EMA周期，默认为12
            slow_period: 慢速EMA周期，默认为26
            signal_period: 信号线周期，默认为9
            sensitivity: 灵敏度参数，控制对价格变化的响应程度，默认为1.0
            multi_periods: 多周期分析参数，默认为[(8, 17, 9), (12, 26, 9), (24, 52, 18)]
            volume_weighted: 是否使用成交量加权，默认为False
            adapt_to_volatility: 是否根据波动率自适应调整参数，默认为True
        """
        super().__init__(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
        self.name = "EnhancedMACD"
        self.description = "增强型MACD指标，优化计算方法和信号质量，增加多周期适应和市场环境感知"
        self.sensitivity = sensitivity
        self.multi_periods = multi_periods or [(8, 17, 9), (12, 26, 9), (24, 52, 18)]
        self.volume_weighted = volume_weighted
        self.adapt_to_volatility = adapt_to_volatility
        self.indicator_type = "trend"  # 指标类型：趋势类
    
    def get_indicator_type(self) -> str:
        """
        获取指标类型
        
        Returns:
            str: 指标类型
        """
        return self.indicator_type
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算增强型MACD指标
        
        Args:
            data: 输入数据，包含OHLC和成交量数据
            
        Returns:
            pd.DataFrame: 计算结果，包含MACD及其多周期指标
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 复制输入数据
        result = data.copy()
        
        # 如果需要自适应波动率，调整参数
        fast_period, slow_period, signal_period = self.fast_period, self.slow_period, self.signal_period
        
        if self.adapt_to_volatility:
            # 计算价格波动率
            volatility = self._calculate_volatility(data["close"])
            # 调整参数
            fast_period, slow_period, signal_period = self._adapt_parameters_to_volatility(
                volatility, self.fast_period, self.slow_period, self.signal_period
            )
        
        # 计算标准MACD（可能使用调整后的参数）
        if self.volume_weighted and "volume" in data.columns:
            self._calculate_volume_weighted_macd(result, fast_period, slow_period, signal_period)
        else:
            self._calculate_macd(result, fast_period, slow_period, signal_period)
        
        # 计算多周期MACD
        for fast, slow, signal in self.multi_periods:
            # 避免重复计算
            if fast == fast_period and slow == slow_period and signal == signal_period:
                continue
                
            if self.volume_weighted and "volume" in data.columns:
                self._calculate_multi_period_volume_weighted_macd(result, fast, slow, signal)
            else:
                self._calculate_multi_period_macd(result, fast, slow, signal)
        
        # 计算MACD柱状体变化率
        result["hist_change_rate"] = self._calculate_histogram_change_rate(result["macd_hist"])
        
        # 计算MACD趋势强度
        result["trend_strength"] = self._calculate_trend_strength(result["macd_hist"])
        
        # 计算MACD零线交叉角度
        result["zero_cross_angle"] = self._calculate_zero_cross_angle(result["macd"])
        
        # 计算MACD与信号线交叉角度
        result["signal_cross_angle"] = self._calculate_signal_cross_angle(result["macd"], result["macd_signal"])
        
        # 计算MACD偏离度
        result["macd_deviation"] = self._calculate_macd_deviation(result["macd"], result["macd_signal"])
        
        # 保存结果
        self._result = result
        
        return result
    
    def _calculate_macd(self, data: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> None:
        """
        计算MACD指标
        
        Args:
            data: 输入数据
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            signal_period: 信号线周期
        """
        # 计算快速和慢速EMA
        fast_ema = data["close"].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data["close"].ewm(span=slow_period, adjust=False).mean()
        
        # 应用灵敏度调整
        if self.sensitivity != 1.0:
            # 调整快速EMA的响应度
            center = slow_ema
            fast_ema = center + (fast_ema - center) * self.sensitivity
        
        # 计算MACD线 = 快速EMA - 慢速EMA
        macd = fast_ema - slow_ema
        
        # 计算信号线 = MACD的EMA
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图 = MACD线 - 信号线
        hist = macd - signal
        
        # 保存结果
        data["fast_ema"] = fast_ema
        data["slow_ema"] = slow_ema
        data["macd"] = macd
        data["macd_signal"] = signal
        data["macd_hist"] = hist
    
    def _calculate_volume_weighted_macd(self, data: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> None:
        """
        计算成交量加权MACD指标
        
        Args:
            data: 输入数据
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            signal_period: 信号线周期
        """
        # 确保数据包含成交量列
        if "volume" not in data.columns:
            logger.warning("数据中不包含成交量列，使用标准MACD计算")
            self._calculate_macd(data, fast_period, slow_period, signal_period)
            return
        
        # 计算成交量归一化（确保成交量变化不会过大影响价格）
        volume_normalized = data["volume"] / data["volume"].rolling(window=20).mean()
        volume_normalized = volume_normalized.clip(0.5, 2.0)  # 限制范围，避免极端值
        
        # 价格乘以归一化成交量
        price_volume = data["close"] * volume_normalized
        
        # 计算快速和慢速EMA（使用成交量加权价格）
        fast_ema = price_volume.ewm(span=fast_period, adjust=False).mean()
        slow_ema = price_volume.ewm(span=slow_period, adjust=False).mean()
        
        # 应用灵敏度调整
        if self.sensitivity != 1.0:
            center = slow_ema
            fast_ema = center + (fast_ema - center) * self.sensitivity
        
        # 计算MACD线
        macd = fast_ema - slow_ema
        
        # 计算信号线
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图
        hist = macd - signal
        
        # 保存结果
        data["volume_fast_ema"] = fast_ema
        data["volume_slow_ema"] = slow_ema
        data["macd"] = macd
        data["macd_signal"] = signal
        data["macd_hist"] = hist
    
    def _calculate_multi_period_macd(self, data: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> None:
        """
        计算多周期MACD指标
        
        Args:
            data: 输入数据
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            signal_period: 信号线周期
        """
        # 计算快速和慢速EMA
        fast_ema = data["close"].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data["close"].ewm(span=slow_period, adjust=False).mean()
        
        # 应用灵敏度调整
        if self.sensitivity != 1.0:
            center = slow_ema
            fast_ema = center + (fast_ema - center) * self.sensitivity
        
        # 计算MACD线
        macd = fast_ema - slow_ema
        
        # 计算信号线
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图
        hist = macd - signal
        
        # 保存结果 - 使用周期作为后缀
        period_suffix = f"_{fast_period}_{slow_period}_{signal_period}"
        data[f"fast_ema{period_suffix}"] = fast_ema
        data[f"slow_ema{period_suffix}"] = slow_ema
        data[f"macd{period_suffix}"] = macd
        data[f"macd_signal{period_suffix}"] = signal
        data[f"macd_hist{period_suffix}"] = hist
    
    def _calculate_multi_period_volume_weighted_macd(self, data: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> None:
        """
        计算多周期成交量加权MACD指标
        
        Args:
            data: 输入数据
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            signal_period: 信号线周期
        """
        # 确保数据包含成交量列
        if "volume" not in data.columns:
            logger.warning("数据中不包含成交量列，使用标准MACD计算")
            self._calculate_multi_period_macd(data, fast_period, slow_period, signal_period)
            return
        
        # 计算成交量归一化
        volume_normalized = data["volume"] / data["volume"].rolling(window=20).mean()
        volume_normalized = volume_normalized.clip(0.5, 2.0)
        
        # 价格乘以归一化成交量
        price_volume = data["close"] * volume_normalized
        
        # 计算快速和慢速EMA
        fast_ema = price_volume.ewm(span=fast_period, adjust=False).mean()
        slow_ema = price_volume.ewm(span=slow_period, adjust=False).mean()
        
        # 应用灵敏度调整
        if self.sensitivity != 1.0:
            center = slow_ema
            fast_ema = center + (fast_ema - center) * self.sensitivity
        
        # 计算MACD线
        macd = fast_ema - slow_ema
        
        # 计算信号线
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图
        hist = macd - signal
        
        # 保存结果 - 使用周期作为后缀
        period_suffix = f"_{fast_period}_{slow_period}_{signal_period}"
        data[f"volume_fast_ema{period_suffix}"] = fast_ema
        data[f"volume_slow_ema{period_suffix}"] = slow_ema
        data[f"macd{period_suffix}"] = macd
        data[f"macd_signal{period_suffix}"] = signal
        data[f"macd_hist{period_suffix}"] = hist
    
    def _calculate_volatility(self, price_series: pd.Series, window: int = 20) -> float:
        """
        计算价格波动率
        
        Args:
            price_series: 价格序列
            window: 计算窗口
            
        Returns:
            float: 波动率
        """
        # 计算价格百分比变化
        returns = price_series.pct_change()
        
        # 计算波动率（标准差）
        volatility = returns.rolling(window=window).std().iloc[-1]
        
        return volatility
    
    def _adapt_parameters_to_volatility(self, volatility: float, 
                                      fast_period: int, 
                                      slow_period: int, 
                                      signal_period: int) -> Tuple[int, int, int]:
        """
        根据波动率调整MACD参数
        
        Args:
            volatility: 波动率
            fast_period: 原快速EMA周期
            slow_period: 原慢速EMA周期
            signal_period: 原信号线周期
            
        Returns:
            Tuple[int, int, int]: 调整后的参数 (fast_period, slow_period, signal_period)
        """
        # 定义波动率阈值
        low_volatility = 0.01
        high_volatility = 0.03
        
        # 针对不同波动率环境调整参数
        if volatility < low_volatility:
            # 低波动率环境 - 使用较短周期，增加灵敏度
            adjusted_fast = max(6, int(fast_period * 0.7))
            adjusted_slow = max(13, int(slow_period * 0.8))
            adjusted_signal = max(5, int(signal_period * 0.8))
        elif volatility > high_volatility:
            # 高波动率环境 - 使用较长周期，减少噪音
            adjusted_fast = int(fast_period * 1.3)
            adjusted_slow = int(slow_period * 1.2)
            adjusted_signal = int(signal_period * 1.2)
        else:
            # 中等波动率 - 使用原始参数
            adjusted_fast = fast_period
            adjusted_slow = slow_period
            adjusted_signal = signal_period
        
        logger.debug(f"根据波动率({volatility:.4f})调整MACD参数: {fast_period}->{adjusted_fast}, {slow_period}->{adjusted_slow}, {signal_period}->{adjusted_signal}")
        
        return adjusted_fast, adjusted_slow, adjusted_signal
    
    def _calculate_histogram_change_rate(self, hist: pd.Series, window: int = 3) -> pd.Series:
        """
        计算MACD柱状体变化率
        
        Args:
            hist: MACD柱状体序列
            window: 计算窗口
            
        Returns:
            pd.Series: 柱状体变化率
        """
        # 计算柱状体一阶差分
        hist_diff = hist.diff(periods=1)
        
        # 计算变化率（相对于柱状体绝对值）
        hist_abs = hist.abs()
        change_rate = hist_diff / (hist_abs + 1e-10)  # 避免除以0
        
        # 使用移动平均平滑变化率
        change_rate_smooth = change_rate.rolling(window=window).mean()
        
        return change_rate_smooth
    
    def _calculate_trend_strength(self, hist: pd.Series, window: int = 14) -> pd.Series:
        """
        计算MACD趋势强度
        
        Args:
            hist: MACD柱状体序列
            window: 计算窗口
            
        Returns:
            pd.Series: 趋势强度
        """
        # 计算柱状体在窗口内的一致性
        # 正值表示上升趋势，负值表示下降趋势，绝对值表示强度
        trend_strength = pd.Series(0.0, index=hist.index)
        
        for i in range(window, len(hist)):
            window_hist = hist.iloc[i-window:i]
            
            # 计算正柱状体和负柱状体的比例
            positive_ratio = (window_hist > 0).mean()
            negative_ratio = (window_hist < 0).mean()
            
            # 计算趋势强度
            if positive_ratio > 0.5:  # 上升趋势
                strength = positive_ratio * 2 - 1  # 映射到 0~1 范围
                trend_strength.iloc[i] = strength
            elif negative_ratio > 0.5:  # 下降趋势
                strength = negative_ratio * 2 - 1  # 映射到 0~1 范围
                trend_strength.iloc[i] = -strength
            else:  # 无明显趋势
                trend_strength.iloc[i] = 0
        
        return trend_strength
    
    def _calculate_zero_cross_angle(self, macd: pd.Series) -> pd.Series:
        """
        计算MACD零线交叉角度
        
        Args:
            macd: MACD序列
            
        Returns:
            pd.Series: 零线交叉角度
        """
        # 计算MACD斜率
        macd_slope = macd.diff(periods=1)
        
        # 计算角度（近似值，用斜率表示）
        angle = macd_slope.copy()
        
        # 只关注零线交叉点附近
        zero_cross = (macd > 0) != (macd.shift(1) > 0)
        non_cross_indices = ~zero_cross
        angle.loc[non_cross_indices] = 0
        
        return angle
    
    def _calculate_signal_cross_angle(self, macd: pd.Series, signal: pd.Series) -> pd.Series:
        """
        计算MACD与信号线交叉角度
        
        Args:
            macd: MACD序列
            signal: 信号线序列
            
        Returns:
            pd.Series: 信号线交叉角度
        """
        # 计算MACD和信号线的斜率
        macd_slope = macd.diff(periods=1)
        signal_slope = signal.diff(periods=1)
        
        # 计算交叉角度（用斜率差表示）
        angle = macd_slope - signal_slope
        
        return angle
    
    def _calculate_macd_deviation(self, macd: pd.Series, signal: pd.Series, window: int = 20) -> pd.Series:
        """
        计算MACD偏离度（MACD与信号线的偏离程度）
        
        Args:
            macd: MACD序列
            signal: 信号线序列
            window: 计算窗口
            
        Returns:
            pd.Series: MACD偏离度
        """
        # 计算MACD与信号线的差距
        diff = macd - signal
        
        # 计算历史标准差
        std = diff.rolling(window=window).std()
        
        # 计算偏离度（标准化）
        deviation = diff / (std + 1e-10)
        
        return deviation 