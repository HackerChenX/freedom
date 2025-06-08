#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CompositeIndicator - 组合指标

将多个技术指标组合在一起进行计算和形态识别
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

from indicators.base_indicator import BaseIndicator, PatternResult
from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength
from utils.logger import get_logger

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)


class CompositeIndicator(BaseIndicator):
    """
    组合指标类
    
    将多个指标组合在一起进行计算和形态识别
    """
    
    def __init__(self, 
                 name: str = "CompositeIndicator",
                 description: str = "组合多个技术指标",
                 indicators: List[BaseIndicator] = None,
                 weights: Dict[str, float] = None):
        """
        初始化组合指标
        
        Args:
            name: 组合指标名称
            description: 组合指标描述
            indicators: 要组合的指标列表
            weights: 各指标的权重字典，键为指标名，值为权重
        """
        super().__init__(name=name, description=description)
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        
        self.indicators = indicators or []
        self.weights = weights or {}
        
        # 为未指定权重的指标设置默认权重
        for indicator in self.indicators:
            if indicator.name not in self.weights:
                self.weights[indicator.name] = 1.0
                
        # 标准化权重，使总和为1
        if self.weights:
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for name in self.weights:
                    self.weights[name] /= total_weight
                    
        # 注册组合指标形态
        self._register_composite_patterns()
        
    def add_indicator(self, indicator: BaseIndicator, weight: float = 1.0):
        """
        添加指标到组合中
        
        Args:
            indicator: 要添加的指标
            weight: 指标权重
        """
        self.indicators.append(indicator)
        self.weights[indicator.name] = weight
        
        # 重新标准化权重
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for name in self.weights:
                self.weights[name] /= total_weight
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算组合指标
        
        Args:
            data: 输入K线数据
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            包含组合指标计算结果的DataFrame
        """
        if not self.indicators:
            logger.warning("没有指标可供组合")
            return data.copy()
        
        # 创建结果DataFrame
        result = data.copy()
        
        # 存储各指标的计算结果
        indicator_results = {}
        
        # 计算每个指标
        for indicator in self.indicators:
            try:
                indicator_result = indicator.calculate(data, *args, **kwargs)
                indicator_results[indicator.name] = indicator_result
                
                # 将指标结果合并到结果DataFrame
                # 为避免列名冲突，添加指标名前缀
                for col in indicator_result.columns:
                    if col not in data.columns:  # 避免覆盖原始数据列
                        result[f"{indicator.name}_{col}"] = indicator_result[col]
            except Exception as e:
                logger.error(f"计算指标 {indicator.name} 时出错: {e}")
        
        # 计算组合指标评分
        self._calculate_composite_score(result, indicator_results)
        
        # 存储结果
        self._result = result
        
        return result
    
    def _calculate_composite_score(self, result: pd.DataFrame, 
                                   indicator_results: Dict[str, pd.DataFrame]):
        """
        计算组合指标评分
        
        Args:
            result: 结果DataFrame
            indicator_results: 各指标的计算结果
        """
        # 创建组合评分列
        result['composite_score'] = 0.0
        
        # 收集每个指标的评分
        indicator_scores = {}
        
        # 计算每个指标的评分并加权
        for indicator in self.indicators:
            try:
                # 获取指标评分
                raw_score = indicator.calculate_raw_score(indicator_results.get(indicator.name, result))
                weight = self.weights.get(indicator.name, 1.0)
                
                # 存储指标评分
                indicator_scores[indicator.name] = raw_score
                
                # 加权累加到组合评分
                result['composite_score'] += raw_score * weight
            except Exception as e:
                logger.error(f"计算指标 {indicator.name} 评分时出错: {e}")
        
        # 存储各指标的原始评分
        for name, score in indicator_scores.items():
            result[f"{name}_score"] = score
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        识别组合指标形态
        
        Args:
            data: 输入K线数据
            kwargs: 额外参数
            
        Returns:
            组合指标形态列表
        """
        if not self.has_result():
            self.calculate(data)
            
        if not self.indicators:
            return []
            
        patterns = []
        
        # 收集各指标识别的形态
        indicator_patterns = {}
        
        # 获取每个指标的形态
        for indicator in self.indicators:
            try:
                # 使用每个指标的get_patterns方法获取形态
                ind_patterns = indicator.get_patterns(data, **kwargs)
                indicator_patterns[indicator.name] = ind_patterns
                
                # 将每个指标的形态添加到结果中，并带上指标名前缀
                for pattern in ind_patterns:
                    # 复制形态信息，避免修改原始对象
                    modified_pattern = pattern.copy()
                    
                    # 修改形态ID和显示名称，添加指标名前缀
                    if 'pattern_id' in modified_pattern:
                        modified_pattern['pattern_id'] = f"{indicator.name}_{modified_pattern['pattern_id']}"
                    
                    if 'display_name' in modified_pattern:
                        modified_pattern['display_name'] = f"{indicator.name}: {modified_pattern['display_name']}"
                    
                    # 调整形态强度，乘以指标权重
                    if 'strength' in modified_pattern:
                        weight = self.weights.get(indicator.name, 1.0)
                        modified_pattern['strength'] *= weight
                    
                    patterns.append(modified_pattern)
            except Exception as e:
                logger.error(f"获取指标 {indicator.name} 形态时出错: {e}")
        
        # 识别指标组合特有的形态
        composite_patterns = self._identify_composite_patterns(indicator_patterns)
        patterns.extend(composite_patterns)
        
        return patterns
    
    def _identify_composite_patterns(self, indicator_patterns: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        识别指标组合特有的形态
        
        Args:
            indicator_patterns: 各指标识别的形态字典
            
        Returns:
            组合特有的形态列表
        """
        composite_patterns = []
        
        # 如果没有足够的指标形态，返回空列表
        if len(indicator_patterns) < 2:
            return composite_patterns
        
        # 检测多指标共振形态
        resonance_patterns = self._detect_resonance_patterns(indicator_patterns)
        composite_patterns.extend(resonance_patterns)
        
        # 检测趋势确认形态
        trend_confirmation_patterns = self._detect_trend_confirmation_patterns(indicator_patterns)
        composite_patterns.extend(trend_confirmation_patterns)
        
        # 检测背离形态
        divergence_patterns = self._detect_divergence_patterns(indicator_patterns)
        composite_patterns.extend(divergence_patterns)
        
        return composite_patterns
    
    def _detect_resonance_patterns(self, indicator_patterns: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        检测多指标共振形态
        
        Args:
            indicator_patterns: 各指标识别的形态字典
            
        Returns:
            共振形态列表
        """
        resonance_patterns = []
        
        # 收集所有看涨和看跌形态
        bullish_indicators = []
        bearish_indicators = []
        
        # 遍历所有指标形态
        for indicator_name, patterns in indicator_patterns.items():
            for pattern in patterns:
                # 根据形态强度判断看涨还是看跌
                if 'strength' in pattern:
                    # 假设强度>60表示看涨形态，<40表示看跌形态
                    if pattern['strength'] > 60:
                        bullish_indicators.append(indicator_name)
                    elif pattern['strength'] < 40:
                        bearish_indicators.append(indicator_name)
        
        # 去重
        bullish_indicators = list(set(bullish_indicators))
        bearish_indicators = list(set(bearish_indicators))
        
        # 检测看涨共振 - 多个指标同时看涨
        if len(bullish_indicators) >= 2:
            bullish_strength = min(90, 50 + len(bullish_indicators) * 10)
            resonance_patterns.append(PatternResult(
                pattern_id="BULLISH_RESONANCE",
                display_name="多指标看涨共振",
                strength=bullish_strength,
                duration=1,
                details={"indicators": bullish_indicators}
            ).to_dict())
        
        # 检测看跌共振 - 多个指标同时看跌
        if len(bearish_indicators) >= 2:
            bearish_strength = min(90, 50 + len(bearish_indicators) * 10)
            resonance_patterns.append(PatternResult(
                pattern_id="BEARISH_RESONANCE",
                display_name="多指标看跌共振",
                strength=bearish_strength,
                duration=1,
                details={"indicators": bearish_indicators}
            ).to_dict())
        
        return resonance_patterns
    
    def _detect_trend_confirmation_patterns(self, indicator_patterns: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        检测趋势确认形态
        
        Args:
            indicator_patterns: 各指标识别的形态字典
            
        Returns:
            趋势确认形态列表
        """
        confirmation_patterns = []
        
        # 趋势指标和震荡指标分类
        trend_indicators = {'MA', 'EMA', 'MACD', 'ADX'}
        oscillator_indicators = {'RSI', 'KDJ', 'STOCH', 'CCI'}
        
        # 检查是否有趋势指标和震荡指标
        has_trend_indicator = any(ind.split('_')[0] in trend_indicators for ind in indicator_patterns.keys())
        has_oscillator_indicator = any(ind.split('_')[0] in oscillator_indicators for ind in indicator_patterns.keys())
        
        if not (has_trend_indicator and has_oscillator_indicator):
            return confirmation_patterns
        
        # 收集趋势指标和震荡指标的看涨/看跌信号
        trend_bullish = []
        trend_bearish = []
        oscillator_bullish = []
        oscillator_bearish = []
        
        # 遍历所有指标形态
        for indicator_name, patterns in indicator_patterns.items():
            ind_type = indicator_name.split('_')[0]
            
            for pattern in patterns:
                # 根据形态强度判断看涨还是看跌
                if 'strength' in pattern:
                    if ind_type in trend_indicators:
                        if pattern['strength'] > 60:
                            trend_bullish.append(indicator_name)
                        elif pattern['strength'] < 40:
                            trend_bearish.append(indicator_name)
                    elif ind_type in oscillator_indicators:
                        if pattern['strength'] > 60:
                            oscillator_bullish.append(indicator_name)
                        elif pattern['strength'] < 40:
                            oscillator_bearish.append(indicator_name)
        
        # 检测趋势和震荡指标同时看涨 - 强烈的买入信号
        if trend_bullish and oscillator_bullish:
            confirmation_patterns.append(PatternResult(
                pattern_id="TREND_OSCILLATOR_BULLISH_CONFIRMATION",
                display_name="趋势与震荡指标看涨确认",
                strength=85,
                duration=1,
                details={"trend_indicators": trend_bullish, "oscillator_indicators": oscillator_bullish}
            ).to_dict())
        
        # 检测趋势和震荡指标同时看跌 - 强烈的卖出信号
        if trend_bearish and oscillator_bearish:
            confirmation_patterns.append(PatternResult(
                pattern_id="TREND_OSCILLATOR_BEARISH_CONFIRMATION",
                display_name="趋势与震荡指标看跌确认",
                strength=85,
                duration=1,
                details={"trend_indicators": trend_bearish, "oscillator_indicators": oscillator_bearish}
            ).to_dict())
        
        return confirmation_patterns
    
    def _detect_divergence_patterns(self, indicator_patterns: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        检测背离形态
        
        Args:
            indicator_patterns: 各指标识别的形态字典
            
        Returns:
            背离形态列表
        """
        divergence_patterns = []
        
        # 如果结果中没有价格数据，无法检测背离
        if not self.has_result() or 'close' not in self._result.columns:
            return divergence_patterns
            
        # 获取近期价格数据
        close_prices = self._result['close']
        if len(close_prices) < 10:
            return divergence_patterns
            
        # 计算价格趋势 - 简单使用10日价格变化判断
        price_trend = 'up' if close_prices.iloc[-1] > close_prices.iloc[-10] else 'down'
        
        # 收集指标趋势信息
        indicator_trends = {}
        
        # 遍历所有指标形态，检测与价格的背离
        for indicator_name, patterns in indicator_patterns.items():
            # 尝试从形态信息中推断指标趋势
            indicator_trend = None
            
            for pattern in patterns:
                pattern_id = pattern.get('pattern_id', '').lower()
                if 'bullish' in pattern_id or 'uptrend' in pattern_id:
                    indicator_trend = 'up'
                    break
                elif 'bearish' in pattern_id or 'downtrend' in pattern_id:
                    indicator_trend = 'down'
                    break
            
            if indicator_trend:
                indicator_trends[indicator_name] = indicator_trend
        
        # 检测价格与指标的背离
        bullish_divergence = []  # 价格下跌但指标上涨
        bearish_divergence = []  # 价格上涨但指标下跌
        
        for indicator_name, trend in indicator_trends.items():
            if price_trend == 'down' and trend == 'up':
                bullish_divergence.append(indicator_name)
            elif price_trend == 'up' and trend == 'down':
                bearish_divergence.append(indicator_name)
        
        # 创建看涨背离形态
        if bullish_divergence:
            divergence_patterns.append(PatternResult(
                pattern_id="BULLISH_DIVERGENCE",
                display_name="看涨背离",
                strength=75,
                duration=1,
                details={"indicators": bullish_divergence}
            ).to_dict())
        
        # 创建看跌背离形态
        if bearish_divergence:
            divergence_patterns.append(PatternResult(
                pattern_id="BEARISH_DIVERGENCE",
                display_name="看跌背离",
                strength=75,
                duration=1,
                details={"indicators": bearish_divergence}
            ).to_dict())
        
        return divergence_patterns
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算组合指标原始评分
        
        Args:
            data: 输入K线数据
            kwargs: 额外参数
            
        Returns:
            评分序列
        """
        if not self.has_result():
            self.calculate(data)
            
        if 'composite_score' in self._result:
            return self._result['composite_score']
        
        # 如果没有预先计算的组合评分，则现在计算
        composite_score = pd.Series(50.0, index=data.index)  # 默认中性评分
        
        # 计算每个指标的评分并加权
        for indicator in self.indicators:
            try:
                # 获取指标评分
                raw_score = indicator.calculate_raw_score(data, **kwargs)
                weight = self.weights.get(indicator.name, 1.0)
                
                # 加权累加到组合评分
                composite_score += raw_score * weight
            except Exception as e:
                logger.error(f"计算指标 {indicator.name} 评分时出错: {e}")
        
        # 确保评分在0-100范围内
        composite_score = np.clip(composite_score, 0, 100)
        
        return composite_score
    
    def _register_composite_patterns(self):
        """
        注册组合指标特有的形态
        """
        # 注册共振形态
        PatternRegistry.register_indicator_pattern(
            indicator_type="COMPOSITE",
            pattern_id="BULLISH_RESONANCE",
            display_name="多指标看涨共振",
            description="多个技术指标同时出现看涨信号，确认强烈的买入机会",
            score_impact=20.0,
            signal_type="bullish"
        )
        
        PatternRegistry.register_indicator_pattern(
            indicator_type="COMPOSITE",
            pattern_id="BEARISH_RESONANCE",
            display_name="多指标看跌共振",
            description="多个技术指标同时出现看跌信号，确认强烈的卖出机会",
            score_impact=-20.0,
            signal_type="bearish"
        )
        
        # 注册趋势确认形态
        PatternRegistry.register_indicator_pattern(
            indicator_type="COMPOSITE",
            pattern_id="TREND_OSCILLATOR_BULLISH_CONFIRMATION",
            display_name="趋势与震荡指标看涨确认",
            description="趋势指标和震荡指标同时出现看涨信号，提供更可靠的买入机会",
            score_impact=25.0,
            signal_type="bullish"
        )
        
        PatternRegistry.register_indicator_pattern(
            indicator_type="COMPOSITE",
            pattern_id="TREND_OSCILLATOR_BEARISH_CONFIRMATION",
            display_name="趋势与震荡指标看跌确认",
            description="趋势指标和震荡指标同时出现看跌信号，提供更可靠的卖出机会",
            score_impact=-25.0,
            signal_type="bearish"
        )
        
        # 注册背离形态
        PatternRegistry.register_indicator_pattern(
            indicator_type="COMPOSITE",
            pattern_id="BULLISH_DIVERGENCE",
            display_name="看涨背离",
            description="价格创新低但指标未创新低，可能预示反转向上",
            score_impact=15.0,
            signal_type="bullish"
        )
        
        PatternRegistry.register_indicator_pattern(
            indicator_type="COMPOSITE",
            pattern_id="BEARISH_DIVERGENCE",
            display_name="看跌背离",
            description="价格创新高但指标未创新高，可能预示反转向下",
            score_impact=-15.0,
            signal_type="bearish"
        ) 