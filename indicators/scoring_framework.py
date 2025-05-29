"""
统一指标评分框架

为所有技术指标提供标准化的打分机制和形态识别能力
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from abc import ABC, abstractmethod
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)


class SignalStrength(Enum):
    """信号强度枚举"""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NEUTRAL = 0
    VERY_WEAK_NEGATIVE = -1
    WEAK_NEGATIVE = -2
    MODERATE_NEGATIVE = -3
    STRONG_NEGATIVE = -4
    VERY_STRONG_NEGATIVE = -5


class MarketEnvironment(Enum):
    """市场环境枚举"""
    BULL_MARKET = "牛市"
    BEAR_MARKET = "熊市"
    SIDEWAYS_MARKET = "震荡市"
    VOLATILE_MARKET = "高波动市"


class IndicatorScoreBase(ABC):
    """
    指标评分基类
    
    所有指标评分都应继承此类，实现统一的评分接口
    """
    
    def __init__(self, name: str, weight: float = 1.0):
        """
        初始化指标评分基类
        
        Args:
            name: 指标名称
            weight: 指标权重，默认为1.0
        """
        self.name = name
        self.weight = weight
        self.market_weights = {
            MarketEnvironment.BULL_MARKET: 1.0,
            MarketEnvironment.BEAR_MARKET: 0.8,
            MarketEnvironment.SIDEWAYS_MARKET: 1.2,
            MarketEnvironment.VOLATILE_MARKET: 0.9
        }
    
    @abstractmethod
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算原始评分（0-100分）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列
        """
        pass
    
    @abstractmethod
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, pd.Series]: 信号字典
        """
        pass
    
    def detect_market_environment(self, data: pd.DataFrame, window: int = 20) -> MarketEnvironment:
        """
        检测市场环境
        
        Args:
            data: 输入数据
            window: 检测窗口期
            
        Returns:
            MarketEnvironment: 市场环境
        """
        if len(data) < window:
            return MarketEnvironment.SIDEWAYS_MARKET
        
        # 计算趋势强度
        close_prices = data['close'].tail(window)
        price_change = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
        
        # 计算波动率
        returns = close_prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        # 判断市场环境
        if volatility > 0.3:  # 高波动
            return MarketEnvironment.VOLATILE_MARKET
        elif price_change > 0.1:  # 上涨超过10%
            return MarketEnvironment.BULL_MARKET
        elif price_change < -0.1:  # 下跌超过10%
            return MarketEnvironment.BEAR_MARKET
        else:
            return MarketEnvironment.SIDEWAYS_MARKET
    
    def calculate_final_score(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        计算最终评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 包含评分、信号、形态的完整结果
        """
        # 计算原始评分
        raw_score = self.calculate_raw_score(data, **kwargs)
        
        # 检测市场环境
        market_env = self.detect_market_environment(data)
        
        # 应用市场环境权重
        env_weight = self.market_weights.get(market_env, 1.0)
        adjusted_score = raw_score * env_weight * self.weight
        
        # 确保评分在0-100范围内
        final_score = np.clip(adjusted_score, 0, 100)
        
        # 识别形态
        patterns = self.identify_patterns(data, **kwargs)
        
        # 生成信号
        signals = self.generate_signals(data, **kwargs)
        
        # 计算信号强度
        signal_strength = self._calculate_signal_strength(final_score)
        
        return {
            'raw_score': raw_score,
            'final_score': final_score,
            'market_environment': market_env.value,
            'patterns': patterns,
            'signals': signals,
            'signal_strength': signal_strength,
            'weight': self.weight,
            'env_weight': env_weight
        }
    
    def _calculate_signal_strength(self, score: pd.Series) -> pd.Series:
        """
        根据评分计算信号强度
        
        Args:
            score: 评分序列
            
        Returns:
            pd.Series: 信号强度序列
        """
        def score_to_strength(s):
            if s >= 90:
                return SignalStrength.VERY_STRONG.value
            elif s >= 80:
                return SignalStrength.STRONG.value
            elif s >= 60:
                return SignalStrength.MODERATE.value
            elif s >= 40:
                return SignalStrength.WEAK.value
            elif s >= 20:
                return SignalStrength.VERY_WEAK.value
            elif s >= 10:
                return SignalStrength.NEUTRAL.value
            elif s >= 5:
                return SignalStrength.VERY_WEAK_NEGATIVE.value
            elif s >= 2:
                return SignalStrength.WEAK_NEGATIVE.value
            elif s >= 1:
                return SignalStrength.MODERATE_NEGATIVE.value
            else:
                return SignalStrength.STRONG_NEGATIVE.value
        
        return score.apply(score_to_strength)


class PatternRecognitionMixin:
    """
    形态识别混入类
    
    提供通用的形态识别方法
    """
    
    def detect_golden_cross(self, fast_line: pd.Series, slow_line: pd.Series) -> pd.Series:
        """检测金叉"""
        return (fast_line > slow_line) & (fast_line.shift(1) <= slow_line.shift(1))
    
    def detect_death_cross(self, fast_line: pd.Series, slow_line: pd.Series) -> pd.Series:
        """检测死叉"""
        return (fast_line < slow_line) & (fast_line.shift(1) >= slow_line.shift(1))
    
    def detect_divergence(self, price: pd.Series, indicator: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """
        检测背离
        
        Args:
            price: 价格序列
            indicator: 指标序列
            window: 检测窗口
            
        Returns:
            Dict[str, pd.Series]: 包含顶背离和底背离的字典
        """
        # 寻找价格和指标的峰值和谷值
        price_peaks = self._find_peaks(price, window)
        price_troughs = self._find_troughs(price, window)
        indicator_peaks = self._find_peaks(indicator, window)
        indicator_troughs = self._find_troughs(indicator, window)
        
        # 检测顶背离（价格创新高，指标不创新高）
        top_divergence = pd.Series(False, index=price.index)
        for i in range(len(price)):
            if price_peaks.iloc[i]:
                # 寻找前一个价格峰值
                prev_peak_idx = self._find_previous_peak(price_peaks, i)
                if prev_peak_idx is not None:
                    if (price.iloc[i] > price.iloc[prev_peak_idx] and 
                        indicator.iloc[i] < indicator.iloc[prev_peak_idx]):
                        top_divergence.iloc[i] = True
        
        # 检测底背离（价格创新低，指标不创新低）
        bottom_divergence = pd.Series(False, index=price.index)
        for i in range(len(price)):
            if price_troughs.iloc[i]:
                # 寻找前一个价格谷值
                prev_trough_idx = self._find_previous_trough(price_troughs, i)
                if prev_trough_idx is not None:
                    if (price.iloc[i] < price.iloc[prev_trough_idx] and 
                        indicator.iloc[i] > indicator.iloc[prev_trough_idx]):
                        bottom_divergence.iloc[i] = True
        
        return {
            'top_divergence': top_divergence,
            'bottom_divergence': bottom_divergence
        }
    
    def _find_peaks(self, series: pd.Series, window: int) -> pd.Series:
        """寻找峰值"""
        peaks = pd.Series(False, index=series.index)
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                peaks.iloc[i] = True
        return peaks
    
    def _find_troughs(self, series: pd.Series, window: int) -> pd.Series:
        """寻找谷值"""
        troughs = pd.Series(False, index=series.index)
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].min():
                troughs.iloc[i] = True
        return troughs
    
    def _find_previous_peak(self, peaks: pd.Series, current_idx: int) -> Optional[int]:
        """寻找前一个峰值索引"""
        for i in range(current_idx - 1, -1, -1):
            if peaks.iloc[i]:
                return i
        return None
    
    def _find_previous_trough(self, troughs: pd.Series, current_idx: int) -> Optional[int]:
        """寻找前一个谷值索引"""
        for i in range(current_idx - 1, -1, -1):
            if troughs.iloc[i]:
                return i
        return None


class IndicatorScoreManager:
    """
    指标评分管理器
    
    管理多个指标的评分和综合评估
    """
    
    def __init__(self):
        """初始化指标评分管理器"""
        self.indicators = {}
        self.weights = {}
    
    def register_indicator(self, indicator: IndicatorScoreBase, weight: float = 1.0):
        """
        注册指标
        
        Args:
            indicator: 指标评分实例
            weight: 指标权重
        """
        self.indicators[indicator.name] = indicator
        self.weights[indicator.name] = weight
    
    def calculate_comprehensive_score(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        计算综合评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 综合评分结果
        """
        if not self.indicators:
            raise ValueError("没有注册任何指标")
        
        # 计算各指标评分
        indicator_results = {}
        total_weight = 0
        weighted_score = pd.Series(0.0, index=data.index)
        all_patterns = []
        all_signals = {}
        
        for name, indicator in self.indicators.items():
            try:
                result = indicator.calculate_final_score(data, **kwargs)
                indicator_results[name] = result
                
                # 累加权重评分
                weight = self.weights[name]
                weighted_score += result['final_score'] * weight
                total_weight += weight
                
                # 收集形态
                all_patterns.extend([f"{name}_{pattern}" for pattern in result['patterns']])
                
                # 收集信号
                for signal_name, signal_series in result['signals'].items():
                    all_signals[f"{name}_{signal_name}"] = signal_series
                
            except Exception as e:
                logger.error(f"计算指标 {name} 评分时出错: {e}")
                continue
        
        # 计算最终综合评分
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = pd.Series(0.0, index=data.index)
        
        # 生成综合信号
        comprehensive_signal = self._generate_comprehensive_signal(final_score)
        
        return {
            'comprehensive_score': final_score,
            'indicator_results': indicator_results,
            'patterns': list(set(all_patterns)),  # 去重
            'signals': all_signals,
            'comprehensive_signal': comprehensive_signal,
            'total_weight': total_weight
        }
    
    def _generate_comprehensive_signal(self, score: pd.Series) -> Dict[str, pd.Series]:
        """
        生成综合信号
        
        Args:
            score: 综合评分序列
            
        Returns:
            Dict[str, pd.Series]: 综合信号字典
        """
        return {
            'strong_buy': score >= 80,
            'buy': score >= 60,
            'hold': (score >= 40) & (score < 60),
            'sell': (score >= 20) & (score < 40),
            'strong_sell': score < 20
        } 