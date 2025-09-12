"""
增强布林带指标模块

实现增强版的布林带指标，提供自适应带宽、动态标准差和多周期分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class EnhancedBoll(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    增强布林带指标
    
    在标准布林带基础上增加自适应带宽、动态标准差调整和多周期分析
    """
    
    def __init__(self, **kwargs):
        """
        初始化增强布林带指标
        
        Args:
            **kwargs: 指标参数
        """
        # 移除super().__init__调用，直接设置属性
        self.name = "EnhancedBoll"
        self.description = "增强布林带指标，提供自适应带宽和动态标准差调整"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_enhancedboll()
        
        # 应用用户参数
        self.set_parameters_Enhanced_Boll(**kwargs)
    
    def _get_default_parameters_enhancedboll(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "period": 20,
            "std_dev": 2.0,
            "adaptive_std": True,
            "volatility_window": 60,
            "squeeze_threshold": 0.1,
            "expansion_threshold": 0.3
        }
    
    def set_parameters_Enhanced_Boll(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        self.period = kwargs.get('period', 20)
        self.std_dev = kwargs.get('std_dev', 2.0)
        self.adaptive_std = kwargs.get('adaptive_std', True)
        self.volatility_window = kwargs.get('volatility_window', 60)
        self.squeeze_threshold = kwargs.get('squeeze_threshold', 0.1)
        self.expansion_threshold = kwargs.get('expansion_threshold', 0.3)

    @property
    def minimum_periods(self) -> int:
        """
        增强布林带指标所需的最少数据周期数
        
        Returns:
            int: 最少需要的数据周期数
        """
        return max(self.period, self.volatility_window) + 10

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算增强布林带指标的主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含增强布林带指标的DataFrame
        """
        # 处理空数据
        if data.empty:
            return pd.DataFrame()

        # 检查必需列
        required_columns = ['close']
        if not all(col in data.columns for col in required_columns):
            return data.copy()

        # 检查数据长度
        if len(data) < self.minimum_periods:
            return data.copy()

        result = data.copy()

        # 计算基础布林带
        result = self._calculate_basic_bollinger_bands(result)

        # 计算自适应标准差
        if self.adaptive_std:
            result = self._calculate_adaptive_std(result)

        # 计算带宽指标
        result = self._calculate_bandwidth_indicators(result)

        # 计算%B和相对位置
        result = self._calculate_percent_b_and_position(result)

        # 生成增强信号
        result = self._generate_enhanced_signals(result)

        return result

    def _calculate_basic_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算基础布林带"""
        result = data.copy()
        
        close = result['close']
        
        # 计算移动平均（中轨）
        middle = close.rolling(window=self.period).mean()
        
        # 计算标准差
        rolling_std = close.rolling(window=self.period).std()
        
        # 计算上轨和下轨
        upper = middle + (rolling_std * self.std_dev)
        lower = middle - (rolling_std * self.std_dev)
        
        result['Middle'] = middle
        result['Upper'] = upper
        result['Lower'] = lower
        result['RollingStd'] = rolling_std
        
        return result

    def _calculate_adaptive_std(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算自适应标准差"""
        result = data.copy()
        
        close = result['close']
        
        # 计算历史波动率
        returns = close.pct_change()
        historical_volatility = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
        
        # 计算当前波动率
        current_volatility = returns.rolling(window=self.period).std() * np.sqrt(252)
        
        # 自适应标准差倍数
        volatility_ratio = current_volatility / historical_volatility.rolling(window=self.volatility_window).mean()
        adaptive_multiplier = self.std_dev * (1 + np.tanh(volatility_ratio - 1) * 0.5)
        adaptive_multiplier = np.clip(adaptive_multiplier, self.std_dev * 0.5, self.std_dev * 2.0)
        
        # 重新计算自适应布林带
        middle = result['Middle']
        rolling_std = result['RollingStd']
        
        adaptive_upper = middle + (rolling_std * adaptive_multiplier)
        adaptive_lower = middle - (rolling_std * adaptive_multiplier)
        
        result['AdaptiveUpper'] = adaptive_upper
        result['AdaptiveLower'] = adaptive_lower
        result['AdaptiveMultiplier'] = adaptive_multiplier
        result['VolatilityRatio'] = volatility_ratio
        
        return result

    def _calculate_bandwidth_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算带宽指标"""
        result = data.copy()
        
        upper = result['Upper']
        lower = result['Lower']
        middle = result['Middle']
        
        # 带宽
        bandwidth = (upper - lower) / middle
        
        # 带宽百分位数
        bandwidth_percentile = bandwidth.rolling(window=self.volatility_window).rank(pct=True)
        
        # 带宽变化率
        bandwidth_change = bandwidth.pct_change()
        
        # 挤压和扩张检测
        squeeze_signal = bandwidth < bandwidth.rolling(window=self.volatility_window).quantile(self.squeeze_threshold)
        expansion_signal = bandwidth > bandwidth.rolling(window=self.volatility_window).quantile(1 - self.expansion_threshold)
        
        result['Bandwidth'] = bandwidth
        result['BandwidthPercentile'] = bandwidth_percentile
        result['BandwidthChange'] = bandwidth_change
        result['BandwidthSqueeze'] = squeeze_signal
        result['BandwidthExpansion'] = expansion_signal
        
        return result

    def _calculate_percent_b_and_position(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算%B和相对位置"""
        result = data.copy()
        
        close = result['close']
        upper = result['Upper']
        lower = result['Lower']
        middle = result['Middle']
        
        # %B值
        percent_b = (close - lower) / (upper - lower)
        
        # 相对于中轨的位置
        middle_position = (close - middle) / (upper - middle)
        
        # %B的移动平均
        percent_b_ma = percent_b.rolling(window=5).mean()
        
        # %B的标准差
        percent_b_std = percent_b.rolling(window=20).std()
        
        result['PercentB'] = percent_b
        result['MiddlePosition'] = middle_position
        result['PercentB_MA'] = percent_b_ma
        result['PercentB_Std'] = percent_b_std
        
        return result

    def _generate_enhanced_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成增强信号"""
        result = data.copy()
        
        close = result['close']
        upper = result['Upper']
        lower = result['Lower']
        middle = result['Middle']
        percent_b = result['PercentB']
        bandwidth = result['Bandwidth']
        
        # 基础突破信号
        upper_breakout = close > upper
        lower_breakout = close < lower
        
        # 回归信号
        upper_pullback = (close.shift(1) > upper.shift(1)) & (close <= upper)
        lower_pullback = (close.shift(1) < lower.shift(1)) & (close >= lower)
        
        # %B极值信号
        percent_b_overbought = percent_b > 1.0
        percent_b_oversold = percent_b < 0.0
        
        # 中轨支撑阻力信号
        middle_support = (close > middle) & (close.shift(1) <= middle.shift(1))
        middle_resistance = (close < middle) & (close.shift(1) >= middle.shift(1))
        
        # 带宽信号
        squeeze_signal = result['BandwidthSqueeze']
        expansion_signal = result['BandwidthExpansion']
        
        # 综合信号
        bull_signal = lower_pullback | middle_support | (percent_b < 0.2)
        bear_signal = upper_pullback | middle_resistance | (percent_b > 0.8)
        
        result['UpperBreakout'] = upper_breakout
        result['LowerBreakout'] = lower_breakout
        result['UpperPullback'] = upper_pullback
        result['LowerPullback'] = lower_pullback
        result['PercentB_Overbought'] = percent_b_overbought
        result['PercentB_Oversold'] = percent_b_oversold
        result['MiddleSupport'] = middle_support
        result['MiddleResistance'] = middle_resistance
        result['BullSignal'] = bull_signal
        result['BearSignal'] = bear_signal
        result['BollSignal'] = bull_signal | bear_signal
        result['BreakoutSignal'] = upper_breakout | lower_breakout
        
        return result

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数
            
        Returns:
            包含增强布林带指标的DataFrame
        """
        return self.calculate(data, **kwargs)
    
    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        BaseIndicator要求的置信度计算方法
        
        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典
            
        Returns:
            float: 置信度分数 (0-1)
        """
        return 0.78  # 增强布林带指标置信度
    
    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        BaseIndicator要求的原始评分计算方法
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列
        """
        result = self.calculate(data, **kwargs)
        
        # 基于%B值和带宽计算评分
        percent_b = result['PercentB']
        bandwidth_percentile = result['BandwidthPercentile']
        
        # %B评分（0-1范围内得高分）
        percent_b_score = 100 * (1 - abs(percent_b - 0.5) * 2)
        
        # 带宽评分（适中带宽得高分）
        bandwidth_score = 100 * (1 - abs(bandwidth_percentile - 0.5) * 2)
        
        # 综合评分
        raw_score = (percent_b_score * 0.6 + bandwidth_score * 0.4)
        
        return raw_score.fillna(50)
    
    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的形态获取方法
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 形态DataFrame
        """
        result = self.calculate(data, **kwargs)
        patterns = pd.DataFrame(index=data.index)
        
        # 布林带形态
        patterns['ENHANCED_BOLL_SQUEEZE'] = result['BandwidthSqueeze']
        patterns['ENHANCED_BOLL_EXPANSION'] = result['BandwidthExpansion']
        patterns['ENHANCED_BOLL_UPPER_BREAKOUT'] = result['UpperBreakout']
        patterns['ENHANCED_BOLL_LOWER_BREAKOUT'] = result['LowerBreakout']
        patterns['ENHANCED_BOLL_MIDDLE_SUPPORT'] = result['MiddleSupport']
        patterns['ENHANCED_BOLL_MIDDLE_RESISTANCE'] = result['MiddleResistance']
        patterns['ENHANCED_BOLL_OVERBOUGHT'] = result['PercentB_Overbought']
        patterns['ENHANCED_BOLL_OVERSOLD'] = result['PercentB_Oversold']
        
        return patterns
    
    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Enhanced_Boll(**kwargs)

    def _get_default_parameters(self) -> Dict[str, Any]:
        """
        BaseIndicator要求的默认参数获取方法

        Returns:
            dict: 默认参数字典
        """
        return self._get_default_parameters_enhancedboll()

    def set_parameters(self, **kwargs):
        """
        标准参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Enhanced_Boll(**kwargs)
