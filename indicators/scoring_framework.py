"""
统一指标评分框架

为所有技术指标提供标准化的打分机制和形态识别能力
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple, Callable
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
    """指标评分管理器"""
    
    def __init__(self):
        """初始化评分管理器"""
        self.indicators: List[IndicatorScoreBase] = []
        self.weights: Dict[str, float] = {}
        self.market_environment = MarketEnvironment.SIDEWAYS_MARKET
        self.pattern_registry = None  # 将在first_time_setup中初始化
        self._setup_done = False
        
    def first_time_setup(self):
        """首次设置，延迟导入避免循环依赖"""
        if self._setup_done:
            return
            
        from indicators.pattern_registry import PatternRegistry
        self.pattern_registry = PatternRegistry
        self._setup_done = True
    
    def add_indicator(self, indicator: IndicatorScoreBase, weight: float = 1.0) -> None:
        """
        添加指标
        
        Args:
            indicator: 指标实例
            weight: 权重
        """
        self.indicators.append(indicator)
        self.weights[indicator.name] = weight
        logger.info(f"已添加指标: {indicator.name}，权重: {weight}")
        
    def score_pattern(self, pattern_name: str, pattern_data: Dict[str, Any]) -> float:
        """
        为单个形态评分
        
        Args:
            pattern_name: 形态名称
            pattern_data: 形态相关数据
            
        Returns:
            float: 形态得分
        """
        # 内置评分规则，可以扩展
        pattern_scores = {
            # K线形态
            'bullish_engulfing': 75,
            'hammer': 70,
            'morning_star': 80,
            'bearish_engulfing': 25,
            'hanging_man': 30,
            'evening_star': 20,
            # 指标交叉
            'AD_GOLDEN_CROSS': 70,
            'AD_DEATH_CROSS': 30,
            'MACD_GOLDEN_CROSS': 75,
            'MACD_DEATH_CROSS': 25,
            # 背离
            'POSITIVE_DIVERGENCE': 85,
            'NEGATIVE_DIVERGENCE': 15,
            'HIDDEN_POSITIVE_DIVERGENCE': 80,
            'HIDDEN_NEGATIVE_DIVERGENCE': 20,
            # 筹码分布相关
            'CHIP_LOW_PROFIT': 65,
            'CHIP_TIGHT': 60,
            'CHIP_BOTTOM_ACCUMULATION': 75,
            'CHIP_HIGH_CONCENTRATION': 40,
            'PRICE_NEAR_COST': 70,
            # 布林带相关
            'BOLL_SQUEEZE': 65,
            'BOLL_BREAKOUT_UP': 80,
            'BOLL_BREAKOUT_DOWN': 20,
            # 成交量相关
            'VOL_BREAKOUT_UP': 75,
            'VOL_HIGH': 60,
            'VOL_RISING': 65,
            # 威廉指标相关
            'WR_EXTREME_OVERBOUGHT': 30,
            'WR_RISING': 60,
            'WR_UPTREND': 70,
            # 机构行为
            'INST_ABSORPTION_PHASE': 75,
            'INST_GENTLE_ABSORPTION': 65,
            'INST_LOW_PROFIT': 70,
            # MACD相关
            'MACD_BULLISH_DIVERGENCE': 80,
            'MACD_ABOVE_ZERO': 65,
            'MACD_RISING': 60,
            # 其他常见指标
            'TRIX_ABOVE_SIGNAL': 65,
            'TRIX_RISING': 60,
            'CCI_OVERBOUGHT': 35,
            'CCI_RISING_STRONG': 70,
            'OBV_RISING': 65,
            'OBV_BREAKOUT': 75,
            'MFI_BULLISH_DIVERGENCE': 80,
            'MFI_OVERBOUGHT': 35,
            'MFI_RISING': 60,
            'ADX_UPTREND': 70,
            'ADX_STRONG_RISING': 75,
            'VIX_VERY_LOW_VOLATILITY': 60,
            'VIX_EXTREME_LOW': 65,
            'GENERIC_MA5_GT_MA10': 65
        }
        
        # 移除指标名称前缀，如 'CDLHAMMER' -> 'HAMMER'
        clean_pattern_name = pattern_name.upper().replace('CDL', '')
        
        score = pattern_scores.get(clean_pattern_name, 50.0) # 默认为中性分
        
        # 可以根据pattern_data中的详细信息进行微调
        # 例如，如果形态强度很高，可以增加分数
        if 'strength' in pattern_data and pattern_data['strength'] > 0.8:
            if score > 50:
                score = min(100, score + 10)
            else:
                score = max(0, score - 10)
        
        return score
    
    def set_weight(self, indicator_name: str, weight: float) -> None:
        """
        设置指标权重
        
        Args:
            indicator_name: 指标名称
            weight: 权重
        """
        self.weights[indicator_name] = weight
        
    def set_market_environment(self, environment: MarketEnvironment) -> None:
        """
        设置市场环境
        
        Args:
            environment: 市场环境
        """
        self.market_environment = environment
        # 同时更新所有指标的市场环境
        for indicator in self.indicators:
            indicator.set_market_environment(environment)
            
    def calculate_combined_score(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        计算综合评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 综合评分结果
        """
        if not self._setup_done:
            self.first_time_setup()
            
        if not self.indicators:
            return {
                'combined_score': pd.Series(50.0, index=data.index),
                'indicator_scores': {},
                'patterns': [],
                'signals': {},
                'confidence': 0.0
            }
            
        # 计算每个指标的评分
        indicator_scores = {}
        pattern_results = []
        signals = {}
        confidences = []
        
        for indicator in self.indicators:
            # 设置市场环境
            indicator.set_market_environment(self.market_environment)
            
            # 计算指标评分
            score_result = indicator.calculate_final_score(data, **kwargs)
            
            indicator_name = indicator.name
            indicator_scores[indicator_name] = score_result
            
            # 收集所有指标的形态
            if 'patterns' in score_result:
                pattern_results.extend(score_result['patterns'])
                
            # 合并信号
            if 'signals' in score_result:
                signals[indicator_name] = score_result['signals']
                
            # 收集置信度
            if 'confidence' in score_result:
                confidences.append(score_result['confidence'])
        
        # 汇总所有指标形态ID
        pattern_ids = [p.pattern_id for p in pattern_results if hasattr(p, 'pattern_id')]
        
        # 计算模式对评分的影响
        pattern_impact = 0.0
        if self.pattern_registry and pattern_ids:
            pattern_impact = self.pattern_registry.calculate_combined_score_impact(pattern_ids)
        
        # 计算加权评分
        total_weight = sum(self.weights.values())
        combined_score = pd.Series(0.0, index=data.index)
        
        for indicator in self.indicators:
            indicator_name = indicator.name
            indicator_result = indicator_scores[indicator_name]
            indicator_weight = self.weights.get(indicator_name, 1.0)
            
            # 获取最终评分（可能是score或final_score）
            if 'score' in indicator_result:
                indicator_score = indicator_result['score']
            elif 'final_score' in indicator_result:
                indicator_score = indicator_result['final_score']
            else:
                indicator_score = pd.Series(50.0, index=data.index)
                
            # 应用权重
            weight_ratio = indicator_weight / total_weight
            combined_score += indicator_score * weight_ratio
        
        # 应用形态影响
        combined_score += pattern_impact
        
        # 确保评分范围在0-100
        combined_score = np.clip(combined_score, 0, 100)
        
        # 计算综合置信度
        confidence = np.mean(confidences) if confidences else 0.5
        
        # 构建返回结果
        result = {
            'combined_score': combined_score,
            'indicator_scores': indicator_scores,
            'patterns': pattern_results,
            'pattern_ids': pattern_ids,
            'pattern_impact': pattern_impact,
            'signals': signals,
            'confidence': confidence,
            'weights': self.weights.copy(),
            'market_environment': self.market_environment,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
        
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, pd.Series]: 交易信号
        """
        # 计算综合评分
        result = self.calculate_combined_score(data, **kwargs)
        combined_score = result['combined_score']
        
        # 初始化信号字典
        signals = {
            'buy_signal': pd.Series(False, index=data.index),
            'sell_signal': pd.Series(False, index=data.index),
            'buy_strength': pd.Series(0, index=data.index),
            'sell_strength': pd.Series(0, index=data.index)
        }
        
        # 根据综合评分生成信号
        # 1. 强烈买入信号：评分 >= 80
        strong_buy = combined_score >= 80
        signals['buy_signal'] |= strong_buy
        signals['buy_strength'] = np.where(strong_buy, 5, signals['buy_strength'])
        
        # 2. 买入信号：评分 >= 70 且 < 80
        buy = (combined_score >= 70) & (combined_score < 80)
        signals['buy_signal'] |= buy
        signals['buy_strength'] = np.where(buy, 4, signals['buy_strength'])
        
        # 3. 弱买入信号：评分 >= 60 且 < 70
        weak_buy = (combined_score >= 60) & (combined_score < 70)
        signals['buy_signal'] |= weak_buy
        signals['buy_strength'] = np.where(weak_buy, 3, signals['buy_strength'])
        
        # 4. 强烈卖出信号：评分 <= 20
        strong_sell = combined_score <= 20
        signals['sell_signal'] |= strong_sell
        signals['sell_strength'] = np.where(strong_sell, 5, signals['sell_strength'])
        
        # 5. 卖出信号：评分 > 20 且 <= 30
        sell = (combined_score > 20) & (combined_score <= 30)
        signals['sell_signal'] |= sell
        signals['sell_strength'] = np.where(sell, 4, signals['sell_strength'])
        
        # 6. 弱卖出信号：评分 > 30 且 <= 40
        weak_sell = (combined_score > 30) & (combined_score <= 40)
        signals['sell_signal'] |= weak_sell
        signals['sell_strength'] = np.where(weak_sell, 3, signals['sell_strength'])
        
        # 如果有特定形态，增强或减弱信号
        if 'patterns' in result and result['patterns']:
            pattern_scores = {}
            
            if not self._setup_done:
                self.first_time_setup()
                
            # 对每个形态，根据其信号类型和强度调整信号
            for pattern in result['patterns']:
                if not hasattr(pattern, 'pattern_id') or not self.pattern_registry:
                    continue
                    
                pattern_id = pattern.pattern_id
                pattern_info = self.pattern_registry.get_pattern_info(pattern_id)
                
                if not pattern_info:
                    continue
                    
                signal_type = pattern_info.get('signal_type')
                score_impact = pattern_info.get('score_impact', 0.0)
                
                # 累积每个形态的评分影响
                if pattern_id not in pattern_scores:
                    pattern_scores[pattern_id] = 0.0
                pattern_scores[pattern_id] += score_impact
                
                # 根据形态的信号类型和强度调整信号
                if signal_type == 'bullish' and score_impact > 0:
                    # 看涨形态增强买入信号
                    signals['buy_signal'] |= True
                    # 根据形态强度增加信号强度
                    strength_boost = min(int(abs(score_impact) / 5), 2)
                    signals['buy_strength'] += strength_boost
                    
                elif signal_type == 'bearish' and score_impact < 0:
                    # 看跌形态增强卖出信号
                    signals['sell_signal'] |= True
                    # 根据形态强度增加信号强度
                    strength_boost = min(int(abs(score_impact) / 5), 2)
                    signals['sell_strength'] += strength_boost
            
            # 限制信号强度范围
            signals['buy_strength'] = np.clip(signals['buy_strength'], 0, 5)
            signals['sell_strength'] = np.clip(signals['sell_strength'], 0, 5)
            
            # 添加形态评分到信号字典
            signals['pattern_scores'] = pattern_scores
        
        # 合并各指标的原始信号
        if 'signals' in result:
            indicator_signals = result['signals']
            signals['indicator_signals'] = indicator_signals
        
        return signals
        
    def get_signal_summary(self, data: pd.DataFrame, index: int = -1, **kwargs) -> Dict[str, Any]:
        """
        获取指定位置的信号摘要
        
        Args:
            data: 输入数据
            index: 索引位置，默认为最后一个位置
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 信号摘要
        """
        # 生成信号
        signals = self.generate_signals(data, **kwargs)
        
        # 获取指定位置的信号
        buy_signal = signals['buy_signal'].iloc[index]
        sell_signal = signals['sell_signal'].iloc[index]
        buy_strength = signals['buy_strength'].iloc[index]
        sell_strength = signals['sell_strength'].iloc[index]
        
        # 获取综合评分
        result = self.calculate_combined_score(data, **kwargs)
        combined_score = result['combined_score'].iloc[index]
        
        # 获取形态信息
        patterns = result.get('patterns', [])
        active_patterns = [p for p in patterns if hasattr(p, 'pattern_id')]
        
        # 构建形态摘要
        pattern_summary = []
        if active_patterns and self.pattern_registry:
            for pattern in active_patterns:
                pattern_id = pattern.pattern_id
                pattern_info = self.pattern_registry.get_pattern_info(pattern_id)
                
                if pattern_info:
                    pattern_summary.append({
                        'pattern_id': pattern_id,
                        'display_name': pattern_info.get('display_name', pattern_id),
                        'signal_type': pattern_info.get('signal_type', 'neutral'),
                        'score_impact': pattern_info.get('score_impact', 0.0),
                        'strength': getattr(pattern, 'strength', 1.0)
                    })
        
        # 构建指标评分摘要
        indicator_summary = {}
        for indicator in self.indicators:
            indicator_name = indicator.name
            if indicator_name in result['indicator_scores']:
                indicator_result = result['indicator_scores'][indicator_name]
                
                # 获取评分
                if 'score' in indicator_result:
                    score = indicator_result['score'].iloc[index]
                elif 'final_score' in indicator_result:
                    score = indicator_result['final_score'].iloc[index]
                else:
                    score = 50.0
                    
                # 获取信号
                indicator_signals = {}
                if indicator_name in signals.get('indicator_signals', {}):
                    ind_sigs = signals['indicator_signals'][indicator_name]
                    for sig_name, sig_series in ind_sigs.items():
                        indicator_signals[sig_name] = sig_series.iloc[index]
                
                indicator_summary[indicator_name] = {
                    'score': score,
                    'weight': self.weights.get(indicator_name, 1.0),
                    'signals': indicator_signals
                }
        
        # 确定整体信号类型
        if buy_signal and not sell_signal:
            signal_type = 'BUY'
        elif sell_signal and not buy_signal:
            signal_type = 'SELL'
        elif buy_signal and sell_signal:
            # 如果同时有买入和卖出信号，比较强度
            if buy_strength > sell_strength:
                signal_type = 'BUY'
            elif sell_strength > buy_strength:
                signal_type = 'SELL'
            else:
                signal_type = 'NEUTRAL'
        else:
            signal_type = 'NEUTRAL'
            
        # 确定信号强度
        if signal_type == 'BUY':
            signal_strength = buy_strength
        elif signal_type == 'SELL':
            signal_strength = sell_strength
        else:
            signal_strength = 0
            
        # 构建返回结果
        summary = {
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'combined_score': combined_score,
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength,
            'patterns': pattern_summary,
            'indicators': indicator_summary,
            'confidence': result.get('confidence', 0.5),
            'market_environment': str(self.market_environment),
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """
        将管理器转换为字典
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'indicators': [indicator.name for indicator in self.indicators],
            'weights': self.weights,
            'market_environment': str(self.market_environment)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], indicator_factory: Callable = None) -> 'IndicatorScoreManager':
        """
        从字典创建管理器
        
        Args:
            data: 字典数据
            indicator_factory: 指标创建工厂函数
            
        Returns:
            IndicatorScoreManager: 管理器实例
        """
        manager = cls()
        
        # 设置市场环境
        if 'market_environment' in data:
            manager.set_market_environment(MarketEnvironment(data['market_environment']))
            
        # 如果没有提供指标工厂，则不加载指标
        if not indicator_factory:
            return manager
            
        # 加载指标
        if 'indicators' in data and 'weights' in data:
            for indicator_name in data['indicators']:
                indicator = indicator_factory(indicator_name)
                if indicator:
                    weight = data['weights'].get(indicator_name, 1.0)
                    manager.add_indicator(indicator, weight)
                    
        return manager 