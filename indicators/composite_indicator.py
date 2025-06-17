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
        self.custom_columns = {}  # 自定义列计算函数
        self.calculate_score_automatically = True  # 是否自动计算评分
        
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
        
    def set_parameters(self, indicators: List[BaseIndicator] = None, weights: Dict[str, float] = None, **kwargs):
        """
        设置组合指标的参数
        
        Args:
            indicators: 新的指标列表
            weights: 新的权重字典
        """
        if indicators is not None:
            self.indicators = indicators
        
        if weights is not None:
            self.weights = weights
        
        # 为未指定权重的指标设置默认权重并重新标准化
        temp_weights = self.weights.copy() if self.weights is not None else {}
        for indicator in self.indicators:
            if indicator.name not in temp_weights:
                temp_weights[indicator.name] = 1.0
        
        if temp_weights:
            total_weight = sum(temp_weights.values())
            if total_weight > 0:
                self.weights = {name: weight / total_weight for name, weight in temp_weights.items()}
            else:
                self.weights = temp_weights
    
    def add_indicator(self, indicator: BaseIndicator, weight: float = 1.0):
        """
        添加指标到组合中

        Args:
            indicator: 要添加的指标
            weight: 指标权重
        """
        self.indicators.append(indicator)
        self.weights[indicator.name] = weight

        # 不自动标准化权重，保持用户设置的权重
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算复合指标
        
        Args:
            data: 包含价格数据的DataFrame
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 添加了指标的DataFrame
        """
        if data.empty:
            return data
            
        result = data.copy()
        
        # 计算每个子指标
        for indicator in self.indicators:
            try:
                # 计算指标
                indicator_result = indicator.calculate(result, *args, **kwargs)
                
                # 合并结果
                if indicator_result is not None:
                    # 使用indicator_result作为结果，但保留result中的列
                    for col in result.columns:
                        if col not in indicator_result.columns:
                            indicator_result[col] = result[col]
                    
                    result = indicator_result
            except Exception as e:
                logger.error(f"计算指标 {indicator.name} 时出错: {e}")
        
        # 添加自定义列
        for name, func in self.custom_columns.items():
            try:
                result[name] = func(result)
            except Exception as e:
                logger.error(f"计算自定义列 {name} 时出错: {e}")
        
        # 计算复合指标评分
        if self.calculate_score_automatically:
            try:
                result['composite_score'] = self.calculate_composite_score(result)

                # 计算各子指标的评分
                for indicator in self.indicators:
                    try:
                        indicator_score = indicator.calculate_raw_score(result)
                        result[f"{indicator.name}_score"] = indicator_score
                    except Exception as e:
                        logger.error(f"计算指标 {indicator.name} 评分时出错: {e}")
                        result[f"{indicator.name}_score"] = 50.0
            except Exception as e:
                logger.error(f"计算复合指标评分时出错: {e}")

        # 保存结果
        self._result = result
        
        # 确保基础数据列被保留
        result = self._preserve_base_columns(data, result)
        
        return result
    
    def _calculate_composite_score(self, result: pd.DataFrame, 
                                   indicator_results: Dict[str, pd.DataFrame],
                                   data: pd.DataFrame):
        """
        计算组合指标评分
        
        Args:
            result: 结果DataFrame
            indicator_results: 各指标的计算结果
            data: 输入K线数据
        """
        # 创建组合评分列
        result['composite_score'] = 0.0
        
        # 收集每个指标的评分
        indicator_scores = {}
        
        # 计算每个指标的评分并加权
        for indicator in self.indicators:
            try:
                # 为每个指标的评分计算准备一个合并了原始数据和该指标结果的DataFrame
                indicator_specific_data = data.copy()
                indicator_result_df = indicator_results.get(indicator.name)
                
                if indicator_result_df is not None:
                    # 合并时要处理重复列，以指标自己的结果为准
                    # 使用 update 和 concat 确保所有列都被添加
                    indicator_specific_data.update(indicator_result_df)
                    for col in indicator_result_df.columns:
                         if col not in indicator_specific_data.columns:
                            indicator_specific_data[col] = indicator_result_df[col]

                # 获取指标评分
                raw_score = indicator.calculate_raw_score(indicator_specific_data)
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
    
    def calculate_composite_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算组合指标的综合评分
        
        Args:
            data: 包含指标数据的DataFrame
            
        Returns:
            pd.Series: 综合评分
        """
        # 初始化评分为零
        score = pd.Series(0.0, index=data.index)
        total_weight = 0.0
        
        # 收集每个指标的评分
        for indicator in self.indicators:
            try:
                # 获取指标评分
                raw_score = indicator.calculate_raw_score(data)
                weight = self.weights.get(indicator.name, 1.0)
                
                # 加权累加到组合评分
                score += raw_score * weight
                total_weight += weight
            except Exception as e:
                logger.error(f"计算指标 {indicator.name} 评分时出错: {e}")
        
        # 标准化评分
        if total_weight > 0:
            score = score / total_weight
            
        return score
    
    def _register_composite_patterns(self):
        """
        注册组合指标特有的形态
        """
        # 注册共振形态
        self.register_pattern_to_registry(
            pattern_id="BULLISH_RESONANCE",
            display_name="多指标看涨共振",
            description="多个技术指标同时出现看涨信号，确认强烈的买入机会",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="BEARISH_RESONANCE",
            display_name="多指标看跌共振",
            description="多个技术指标同时出现看跌信号，确认强烈的卖出机会",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册趋势确认形态
        self.register_pattern_to_registry(
            pattern_id="TREND_OSCILLATOR_BULLISH_CONFIRMATION",
            display_name="趋势与震荡指标看涨确认",
            description="趋势指标和震荡指标同时出现看涨信号，提供更可靠的买入机会",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="TREND_OSCILLATOR_BEARISH_CONFIRMATION",
            display_name="趋势与震荡指标看跌确认",
            description="趋势指标和震荡指标同时出现看跌信号，提供更可靠的卖出机会",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册背离形态
        self.register_pattern_to_registry(
            pattern_id="BULLISH_DIVERGENCE",
            display_name="看涨背离",
            description="价格创新低但指标未创新低，可能预示反转向上",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="BEARISH_DIVERGENCE",
            display_name="看跌背离",
            description="价格创新高但指标未创新高，可能预示反转向下",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算CompositeIndicator指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于指标数量的置信度
        num_indicators = len(self.indicators)
        if num_indicators >= 3:
            confidence += 0.15  # 多指标组合置信度更高
        elif num_indicators >= 2:
            confidence += 0.1

        # 3. 基于形态的置信度
        if not patterns.empty:
            # 检查CompositeIndicator形态
            pattern_count = patterns.sum().sum()
            if pattern_count > 0:
                confidence += min(pattern_count * 0.05, 0.2)

        # 4. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.1, 0.15)

        # 5. 基于评分趋势的置信度
        if len(score) >= 3:
            recent_scores = score.iloc[-3:]
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 明确的趋势增加置信度
            if abs(trend) > 10:
                confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取CompositeIndicator相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data)

        if not self.indicators:
            return pd.DataFrame(index=data.index)

        patterns = pd.DataFrame(index=data.index)

        # 收集各指标的形态
        all_patterns = []
        for indicator in self.indicators:
            try:
                # 获取指标形态
                ind_patterns = indicator.get_patterns(data, **kwargs)
                if isinstance(ind_patterns, pd.DataFrame):
                    # 为每个形态列添加指标名前缀
                    for col in ind_patterns.columns:
                        patterns[f"{indicator.name}_{col}"] = ind_patterns[col]
                elif isinstance(ind_patterns, list):
                    # 处理返回列表的情况
                    all_patterns.extend(ind_patterns)
            except Exception as e:
                logger.error(f"获取指标 {indicator.name} 形态时出错: {e}")

        # 添加组合指标特有的形态
        if len(self.indicators) >= 2:
            # 多指标共振形态
            bullish_count = 0
            bearish_count = 0

            for indicator in self.indicators:
                try:
                    # 计算指标评分
                    score = indicator.calculate_raw_score(data)
                    if not score.empty:
                        last_score = score.iloc[-1]
                        if last_score > 70:
                            bullish_count += 1
                        elif last_score < 30:
                            bearish_count += 1
                except:
                    continue

            # 多指标看涨共振
            patterns['COMPOSITE_BULLISH_RESONANCE'] = bullish_count >= 2
            # 多指标看跌共振
            patterns['COMPOSITE_BEARISH_RESONANCE'] = bearish_count >= 2
            # 指标分歧
            patterns['COMPOSITE_DIVERGENCE'] = (bullish_count >= 1) & (bearish_count >= 1)

        return patterns

    def register_patterns(self):
        """
        注册CompositeIndicator指标的形态到全局形态注册表
        """
        # 调用已有的注册方法
        self._register_composite_patterns()

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成CompositeIndicator交易信号

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            Dict[str, pd.Series]: 包含买卖信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data)

        if not self.indicators:
            return {
                'buy_signal': pd.Series(False, index=data.index),
                'sell_signal': pd.Series(False, index=data.index),
                'signal_strength': pd.Series(0.0, index=data.index)
            }

        # 生成信号
        buy_signal = pd.Series(False, index=data.index)
        sell_signal = pd.Series(False, index=data.index)
        signal_strength = pd.Series(0.0, index=data.index)

        # 收集各指标的信号
        buy_votes = pd.Series(0, index=data.index)
        sell_votes = pd.Series(0, index=data.index)

        for indicator in self.indicators:
            try:
                # 获取指标信号
                ind_signals = indicator.generate_trading_signals(data, **kwargs)
                weight = self.weights.get(indicator.name, 1.0)

                if isinstance(ind_signals, dict):
                    if 'buy_signal' in ind_signals:
                        buy_votes += ind_signals['buy_signal'].astype(int) * weight
                    if 'sell_signal' in ind_signals:
                        sell_votes += ind_signals['sell_signal'].astype(int) * weight
                    if 'signal_strength' in ind_signals:
                        signal_strength += ind_signals['signal_strength'] * weight
            except Exception as e:
                logger.error(f"获取指标 {indicator.name} 信号时出错: {e}")

        # 基于投票生成最终信号
        total_weight = sum(self.weights.values()) if self.weights else len(self.indicators)

        # 需要超过一半的权重支持才产生信号
        threshold = total_weight * 0.5

        buy_signal = buy_votes > threshold
        sell_signal = sell_votes > threshold

        # 标准化信号强度
        if total_weight > 0:
            signal_strength = signal_strength / total_weight

        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'signal_strength': signal_strength
        }

    def add_custom_column(self, name: str, func):
        """
        添加自定义列计算函数

        Args:
            name: 列名
            func: 计算函数，接受DataFrame参数，返回Series
        """
        self.custom_columns[name] = func

    def remove_indicator(self, indicator_name: str):
        """
        从组合中移除指标

        Args:
            indicator_name: 要移除的指标名称
        """
        # 移除指标
        self.indicators = [ind for ind in self.indicators if ind.name != indicator_name]

        # 移除权重
        if indicator_name in self.weights:
            del self.weights[indicator_name]

        # 重新标准化权重
        if self.weights:
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for name in self.weights:
                    self.weights[name] /= total_weight

    def get_indicator_names(self) -> List[str]:
        """
        获取所有指标名称

        Returns:
            List[str]: 指标名称列表
        """
        return [indicator.name for indicator in self.indicators]

    def get_indicator_weights(self) -> Dict[str, float]:
        """
        获取指标权重

        Returns:
            Dict[str, float]: 指标权重字典
        """
        return self.weights.copy()

    def get_indicator_type(self) -> str:
        """
        获取指标类型

        Returns:
            str: 指标类型
        """
        return "COMPOSITE"

    def set_market_environment(self, environment: str):
        """
        设置市场环境

        Args:
            environment: 市场环境字符串
        """
        valid_environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        if environment not in valid_environments:
            raise ValueError(f"无效的市场环境: {environment}. 有效值: {valid_environments}")

        self.market_environment = environment

        # 同时设置所有子指标的市场环境
        for indicator in self.indicators:
            if hasattr(indicator, 'set_market_environment'):
                try:
                    indicator.set_market_environment(environment)
                except:
                    pass  # 忽略不支持的指标

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            'bullish': {'name': '看涨形态', 'description': '指标显示看涨信号', 'type': 'BULLISH'},
            'bearish': {'name': '看跌形态', 'description': '指标显示看跌信号', 'type': 'BEARISH'},
            'neutral': {'name': '中性形态', 'description': '指标显示中性信号', 'type': 'NEUTRAL'},
            
            # 通用形态
            'strong_signal': {'name': '强信号', 'description': '强烈的技术信号', 'type': 'STRONG'},
            'weak_signal': {'name': '弱信号', 'description': '较弱的技术信号', 'type': 'WEAK'},
            'trend_up': {'name': '上升趋势', 'description': '价格呈上升趋势', 'type': 'BULLISH'},
            'trend_down': {'name': '下降趋势', 'description': '价格呈下降趋势', 'type': 'BEARISH'},
        }
        
        # 默认形态信息
        default_pattern = {
            'name': pattern_id.replace('_', ' ').title(),
            'description': f'{pattern_id}形态',
            'type': 'UNKNOWN'
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)
