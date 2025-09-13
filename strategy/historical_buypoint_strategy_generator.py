#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
历史买点策略生成器

从历史买点数据自动生成选股策略，实现用户设想的核心闭环功能：
历史买点输入 → 技术形态分析 → 策略生成 → 选股执行 → 双向验证

遵循六层架构规范，使用真实数据和标准技术指标算法
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from enum import Enum

from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container
from db.interfaces.data_access_interface import DataAccessInterface
from indicators.complete_indicator_registry import CompleteIndicatorRegistry

logger = get_logger(__name__)


class StrategyGenerationMode(Enum):
    """策略生成模式"""
    CONSERVATIVE = "conservative"  # 保守型：要求严格条件匹配
    BALANCED = "balanced"          # 平衡型：适中的条件要求
    AGGRESSIVE = "aggressive"      # 激进型：较宽松的条件
    ADAPTIVE = "adaptive"          # 自适应：根据数据自动调整


class PatternRecognitionMethod(Enum):
    """模式识别方法"""
    STATISTICAL = "statistical"     # 统计分析方法
    FREQUENT_PATTERN = "frequent"   # 频繁模式挖掘
    CLUSTERING = "clustering"       # 聚类分析
    HYBRID = "hybrid"              # 混合方法


@dataclass
class BuyPointInput:
    """买点输入数据结构"""
    stock_code: str
    buypoint_date: str
    expected_return: Optional[float] = None
    holding_days: Optional[int] = None
    note: Optional[str] = None

    def __post_init__(self):
        """数据验证"""
        if not self.stock_code or len(self.stock_code) != 6:
            raise ValueError(f"无效的股票代码: {self.stock_code}")

        try:
            datetime.strptime(self.buypoint_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"无效的日期格式: {self.buypoint_date}")


@dataclass
class TechnicalPattern:
    """技术形态模式"""
    indicator_name: str
    condition_type: str  # '>', '<', '>=', '<=', 'between'
    threshold_value: Union[float, Tuple[float, float]]
    confidence: float    # 置信度 [0, 1]
    frequency: int       # 在历史买点中出现的频率
    avg_return: Optional[float] = None  # 平均收益

    def to_condition_string(self) -> str:
        """转换为条件字符串"""
        if self.condition_type == 'between':
            low, high = self.threshold_value
            return f"{self.indicator_name} BETWEEN {low:.4f} AND {high:.4f}"
        else:
            return f"{self.indicator_name} {self.condition_type} {self.threshold_value:.4f}"


@dataclass
class GeneratedStrategy:
    """生成的策略"""
    strategy_name: str
    description: str
    technical_patterns: List[TechnicalPattern]
    expected_success_rate: float
    expected_return: float
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'

    # 元数据
    source_buypoints_count: int
    generation_time: datetime
    generation_mode: StrategyGenerationMode
    pattern_method: PatternRecognitionMethod

    # 验证结果
    validation_results: Optional[Dict[str, Any]] = None


class TechnicalFeatureExtractor:
    """技术特征提取器 - 核心算法组件"""

    def __init__(self):
        """初始化特征提取器"""
        self.data_manager = get_container().resolve(DataAccessInterface)
        # Create indicator registry instance directly as it may not be registered in container
        self.indicator_registry = CompleteIndicatorRegistry()
        self.cache = {}

    @exception_handler(reraise=False)
    @performance_monitor(threshold=5.0)
    def extract_buypoint_features(self,
                                 buypoint_input: BuyPointInput,
                                 analysis_days_before: int = 5,
                                 analysis_days_after: int = 20) -> Dict[str, Any]:
        """
        提取单个买点的技术特征

        Args:
            buypoint_input: 买点输入数据
            analysis_days_before: 买点前分析天数
            analysis_days_after: 买点后分析天数（用于验证收益）

        Returns:
            Dict: 技术特征字典
        """
        stock_code = buypoint_input.stock_code
        buypoint_date = buypoint_input.buypoint_date

        # 计算日期范围
        buypoint_dt = datetime.strptime(buypoint_date, '%Y-%m-%d')
        start_date = (buypoint_dt - timedelta(days=analysis_days_before + 30)).strftime('%Y-%m-%d')
        end_date = (buypoint_dt + timedelta(days=analysis_days_after + 5)).strftime('%Y-%m-%d')

        # 获取股票历史数据
        stock_data = self._get_stock_data(stock_code, start_date, end_date)
        if stock_data is None or len(stock_data) < 30:
            logger.warning(f"股票 {stock_code} 在 {buypoint_date} 附近数据不足")
            return {}

        # 找到买点在数据中的索引
        buypoint_idx = self._find_buypoint_index(stock_data, buypoint_date)
        if buypoint_idx == -1:
            logger.warning(f"无法在数据中找到买点日期 {buypoint_date}")
            return {}

        # 提取技术指标特征
        features = self._extract_technical_indicators(stock_data, buypoint_idx)

        # 提取K线形态特征
        pattern_features = self._extract_pattern_features(stock_data, buypoint_idx)
        features.update(pattern_features)

        # 计算买点后的收益表现
        returns = self._calculate_future_returns(stock_data, buypoint_idx, analysis_days_after)
        features.update(returns)

        # 添加元数据
        features.update({
            'stock_code': stock_code,
            'buypoint_date': buypoint_date,
            'buypoint_price': stock_data.iloc[buypoint_idx]['close'] if buypoint_idx < len(stock_data) else None,
            'extraction_time': datetime.now().isoformat()
        })

        return features

    def _get_stock_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        cache_key = f"{stock_code}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            data = self.data_manager.get_stock_data_data_access_manager(
                stock_code,
                start_date,
                end_date
            )

            if data is not None and len(data) > 0:
                self.cache[cache_key] = data
                return data
            else:
                logger.warning(f"从ClickHouse获取的股票数据为空: {stock_code}")
                return None

        except Exception as e:
            logger.error(f"获取股票数据失败: {stock_code}, {e}")
            return None

    def _find_buypoint_index(self, stock_data: pd.DataFrame, buypoint_date: str) -> int:
        """在股票数据中找到买点日期的索引"""
        try:
            # 尝试精确匹配
            if 'date' in stock_data.columns:
                date_match = stock_data[stock_data['date'] == buypoint_date]
            elif stock_data.index.name == 'date':
                date_match = stock_data[stock_data.index == buypoint_date]
            else:
                return -1

            if not date_match.empty:
                return date_match.index[0]

            # 如果精确匹配失败，找最近的交易日
            buypoint_dt = datetime.strptime(buypoint_date, '%Y-%m-%d')
            if 'date' in stock_data.columns:
                stock_data['date_dt'] = pd.to_datetime(stock_data['date'])
                date_diffs = (stock_data['date_dt'] - buypoint_dt).abs()
            else:
                stock_data['date_dt'] = pd.to_datetime(stock_data.index)
                date_diffs = (stock_data['date_dt'] - buypoint_dt).abs()

            closest_idx = date_diffs.idxmin()
            return closest_idx

        except Exception as e:
            logger.error(f"查找买点索引失败: {buypoint_date}, {e}")
            return -1

    def _extract_technical_indicators(self, stock_data: pd.DataFrame, buypoint_idx: int) -> Dict[str, Any]:
        """提取技术指标特征"""
        features = {}

        try:
            # 获取买点当日的技术指标值
            price = stock_data.iloc[buypoint_idx]['close']
            volume = stock_data.iloc[buypoint_idx]['volume']

            # 价格相关指标
            features['price'] = price
            features['volume'] = volume

            # 使用现有指标系统计算常用指标
            indicator_names = ['RSI', 'MACD', 'KDJ', 'BOLL', 'ATR', 'MA5', 'MA10', 'MA20']

            for indicator_name in indicator_names:
                try:
                    if hasattr(self.indicator_registry, 'get_indicator'):
                        indicator = self.indicator_registry.get_indicator(indicator_name)
                        if indicator:
                            result = indicator.calculate(stock_data.iloc[:buypoint_idx + 1])
                            if isinstance(result, dict) and result:
                                # 取最后一个值作为买点当日的指标值
                                for key, values in result.items():
                                    if hasattr(values, '__iter__') and len(values) > 0:
                                        features[f"{indicator_name}_{key}"] = values[-1]
                                    else:
                                        features[f"{indicator_name}"] = values
                except Exception as e:
                    logger.debug(f"计算指标 {indicator_name} 失败: {e}")

            # 添加价格位置相关特征
            recent_high = stock_data.iloc[max(0, buypoint_idx-20):buypoint_idx+1]['high'].max()
            recent_low = stock_data.iloc[max(0, buypoint_idx-20):buypoint_idx+1]['low'].min()

            if recent_high > recent_low:
                features['price_position'] = (price - recent_low) / (recent_high - recent_low)
            else:
                features['price_position'] = 0.5

        except Exception as e:
            logger.error(f"提取技术指标特征失败: {e}")

        return features

    def _extract_pattern_features(self, stock_data: pd.DataFrame, buypoint_idx: int) -> Dict[str, Any]:
        """提取K线形态特征"""
        features = {}

        try:
            # 确保有足够的数据
            if buypoint_idx < 5:
                return features

            # 获取最近几日的OHLC数据
            recent_data = stock_data.iloc[buypoint_idx-4:buypoint_idx+1]
            opens = recent_data['open'].values
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            closes = recent_data['close'].values

            # 基础形态特征
            features['is_green_candle'] = closes[-1] > opens[-1]  # 是否为阳线
            features['upper_shadow_ratio'] = (highs[-1] - max(opens[-1], closes[-1])) / (highs[-1] - lows[-1]) if highs[-1] > lows[-1] else 0
            features['lower_shadow_ratio'] = (min(opens[-1], closes[-1]) - lows[-1]) / (highs[-1] - lows[-1]) if highs[-1] > lows[-1] else 0

            # 价格变化模式
            if len(closes) >= 3:
                features['price_trend_3d'] = 'up' if closes[-1] > closes[-3] else 'down'
                features['volume_trend_3d'] = 'up' if recent_data['volume'].iloc[-1] > recent_data['volume'].iloc[-3] else 'down'

            # 突破模式
            if buypoint_idx >= 20:
                ma20_data = stock_data.iloc[buypoint_idx-19:buypoint_idx+1]['close'].rolling(20).mean()
                if len(ma20_data) > 0:
                    features['above_ma20'] = closes[-1] > ma20_data.iloc[-1]
                    features['ma20_breakthrough'] = closes[-1] > ma20_data.iloc[-1] and closes[-2] <= ma20_data.iloc[-2]

        except Exception as e:
            logger.error(f"提取形态特征失败: {e}")

        return features

    def _calculate_future_returns(self, stock_data: pd.DataFrame, buypoint_idx: int, days_after: int) -> Dict[str, Any]:
        """计算买点后的收益表现"""
        returns = {}

        try:
            buypoint_price = stock_data.iloc[buypoint_idx]['close']

            # 计算未来不同周期的收益
            for days in [5, 10, 20]:
                if buypoint_idx + days < len(stock_data):
                    future_price = stock_data.iloc[buypoint_idx + days]['close']
                    return_pct = (future_price - buypoint_price) / buypoint_price * 100
                    returns[f'return_{days}d'] = return_pct

            # 计算最大收益和最大回撤
            max_days = min(days_after, len(stock_data) - buypoint_idx - 1)
            if max_days > 0:
                future_prices = stock_data.iloc[buypoint_idx+1:buypoint_idx+1+max_days]['close'].values
                max_return = ((future_prices.max() - buypoint_price) / buypoint_price * 100) if len(future_prices) > 0 else 0
                min_return = ((future_prices.min() - buypoint_price) / buypoint_price * 100) if len(future_prices) > 0 else 0

                returns['max_return'] = max_return
                returns['max_drawdown'] = min_return

        except Exception as e:
            logger.error(f"计算收益表现失败: {e}")

        return returns


class PatternRecognitionEngine:
    """模式识别引擎 - 从特征中识别技术模式"""

    def __init__(self, method: PatternRecognitionMethod = PatternRecognitionMethod.STATISTICAL):
        """初始化模式识别引擎"""
        self.method = method
        self.min_confidence = 0.6
        self.min_frequency = 3

    @exception_handler(reraise=False)
    @performance_monitor(threshold=10.0)
    def recognize_patterns(self, all_features: List[Dict[str, Any]]) -> List[TechnicalPattern]:
        """
        从所有买点特征中识别共同的技术模式

        Args:
            all_features: 所有买点的特征列表

        Returns:
            List[TechnicalPattern]: 识别出的技术模式列表
        """
        if len(all_features) < 3:
            logger.warning("买点数据太少，无法进行有效的模式识别")
            return []

        # 根据选择的方法进行模式识别
        if self.method == PatternRecognitionMethod.STATISTICAL:
            return self._statistical_pattern_recognition(all_features)
        elif self.method == PatternRecognitionMethod.FREQUENT_PATTERN:
            return self._frequent_pattern_recognition(all_features)
        elif self.method == PatternRecognitionMethod.HYBRID:
            # 混合方法：结合统计和频繁模式
            stat_patterns = self._statistical_pattern_recognition(all_features)
            freq_patterns = self._frequent_pattern_recognition(all_features)
            return self._merge_patterns(stat_patterns, freq_patterns)
        else:
            return self._statistical_pattern_recognition(all_features)

    def _statistical_pattern_recognition(self, all_features: List[Dict[str, Any]]) -> List[TechnicalPattern]:
        """基于统计分析的模式识别"""
        patterns = []

        try:
            # 收集所有数值型特征
            numeric_features = {}
            for features in all_features:
                for key, value in features.items():
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        if key not in numeric_features:
                            numeric_features[key] = []
                        numeric_features[key].append(value)

            # 对每个特征进行统计分析
            for feature_name, values in numeric_features.items():
                if len(values) < self.min_frequency:
                    continue

                values_array = np.array(values)

                # 计算统计特征
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                q25, q75 = np.percentile(values_array, [25, 75])

                # 判断是否有明显的集中区间
                if std_val > 0:
                    cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')

                    # 如果变异系数较小，说明值比较集中
                    if cv < 0.5:  # 经验阈值
                        # 使用四分位数作为区间
                        pattern = TechnicalPattern(
                            indicator_name=feature_name,
                            condition_type='between',
                            threshold_value=(q25, q75),
                            confidence=max(0.0, 1.0 - cv),  # 变异系数越小，置信度越高
                            frequency=len(values),
                            avg_return=self._calculate_avg_return_for_feature(all_features, feature_name, q25, q75)
                        )
                        patterns.append(pattern)

                # 寻找明显的阈值模式（大于某个值或小于某个值）
                median_val = np.median(values_array)

                # 检查是否大部分值都大于中位数
                above_median_ratio = np.sum(values_array > median_val) / len(values_array)
                if above_median_ratio >= 0.7:  # 70%以上的值都大于中位数
                    pattern = TechnicalPattern(
                        indicator_name=feature_name,
                        condition_type='>=',
                        threshold_value=median_val,
                        confidence=above_median_ratio,
                        frequency=int(len(values) * above_median_ratio),
                        avg_return=self._calculate_avg_return_for_threshold(all_features, feature_name, '>=', median_val)
                    )
                    patterns.append(pattern)

        except Exception as e:
            logger.error(f"统计模式识别失败: {e}")

        # 按置信度排序
        patterns.sort(key=lambda x: x.confidence, reverse=True)

        # 过滤低置信度的模式
        return [p for p in patterns if p.confidence >= self.min_confidence]

    def _frequent_pattern_recognition(self, all_features: List[Dict[str, Any]]) -> List[TechnicalPattern]:
        """基于频繁模式挖掘的模式识别"""
        patterns = []

        try:
            # 将数值型特征离散化
            discretized_features = []
            feature_ranges = {}

            # 首先计算每个特征的离散化区间
            all_numeric_features = {}
            for features in all_features:
                for key, value in features.items():
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        if key not in all_numeric_features:
                            all_numeric_features[key] = []
                        all_numeric_features[key].append(value)

            for feature_name, values in all_numeric_features.items():
                if len(values) >= self.min_frequency:
                    values_array = np.array(values)
                    # 使用三分位数进行离散化
                    q33, q67 = np.percentile(values_array, [33, 67])
                    feature_ranges[feature_name] = {
                        'low': values_array.min(),
                        'q33': q33,
                        'q67': q67,
                        'high': values_array.max()
                    }

            # 离散化所有特征
            for features in all_features:
                discretized = {}
                for feature_name, ranges in feature_ranges.items():
                    if feature_name in features:
                        value = features[feature_name]
                        if isinstance(value, (int, float)) and not pd.isna(value):
                            if value <= ranges['q33']:
                                discretized[feature_name] = 'low'
                            elif value <= ranges['q67']:
                                discretized[feature_name] = 'medium'
                            else:
                                discretized[feature_name] = 'high'
                discretized_features.append(discretized)

            # 计算每个特征值组合的频率
            feature_combinations = {}
            for discretized in discretized_features:
                for feature_name, category in discretized.items():
                    key = f"{feature_name}_{category}"
                    if key not in feature_combinations:
                        feature_combinations[key] = 0
                    feature_combinations[key] += 1

            # 识别高频模式
            total_samples = len(all_features)
            for combination, frequency in feature_combinations.items():
                if frequency >= self.min_frequency:
                    confidence = frequency / total_samples
                    if confidence >= self.min_confidence:
                        feature_name, category = combination.rsplit('_', 1)

                        if feature_name in feature_ranges:
                            ranges = feature_ranges[feature_name]

                            if category == 'low':
                                condition_type = '<='
                                threshold_value = ranges['q33']
                            elif category == 'high':
                                condition_type = '>='
                                threshold_value = ranges['q67']
                            else:  # medium
                                condition_type = 'between'
                                threshold_value = (ranges['q33'], ranges['q67'])

                            pattern = TechnicalPattern(
                                indicator_name=feature_name,
                                condition_type=condition_type,
                                threshold_value=threshold_value,
                                confidence=confidence,
                                frequency=frequency,
                                avg_return=self._calculate_avg_return_for_pattern(all_features, feature_name, condition_type, threshold_value)
                            )
                            patterns.append(pattern)

        except Exception as e:
            logger.error(f"频繁模式识别失败: {e}")

        # 按置信度和频率排序
        patterns.sort(key=lambda x: (x.confidence, x.frequency), reverse=True)

        return patterns

    def _merge_patterns(self, stat_patterns: List[TechnicalPattern], freq_patterns: List[TechnicalPattern]) -> List[TechnicalPattern]:
        """合并统计和频繁模式的结果"""
        merged_patterns = []

        # 创建指标名称到模式的映射
        stat_dict = {p.indicator_name: p for p in stat_patterns}
        freq_dict = {p.indicator_name: p for p in freq_patterns}

        all_indicators = set(stat_dict.keys()) | set(freq_dict.keys())

        for indicator_name in all_indicators:
            stat_pattern = stat_dict.get(indicator_name)
            freq_pattern = freq_dict.get(indicator_name)

            if stat_pattern and freq_pattern:
                # 选择置信度更高的模式
                if stat_pattern.confidence >= freq_pattern.confidence:
                    merged_patterns.append(stat_pattern)
                else:
                    merged_patterns.append(freq_pattern)
            elif stat_pattern:
                merged_patterns.append(stat_pattern)
            elif freq_pattern:
                merged_patterns.append(freq_pattern)

        return merged_patterns

    def _calculate_avg_return_for_feature(self, all_features: List[Dict[str, Any]],
                                        feature_name: str, min_val: float, max_val: float) -> Optional[float]:
        """计算特定特征区间的平均收益"""
        try:
            returns = []
            for features in all_features:
                if feature_name in features:
                    value = features[feature_name]
                    if isinstance(value, (int, float)) and min_val <= value <= max_val:
                        # 尝试获取收益数据
                        for return_key in ['return_20d', 'return_10d', 'return_5d']:
                            if return_key in features:
                                returns.append(features[return_key])
                                break

            return np.mean(returns) if returns else None
        except:
            return None

    def _calculate_avg_return_for_threshold(self, all_features: List[Dict[str, Any]],
                                          feature_name: str, condition: str, threshold: float) -> Optional[float]:
        """计算特定阈值条件的平均收益"""
        try:
            returns = []
            for features in all_features:
                if feature_name in features:
                    value = features[feature_name]
                    if isinstance(value, (int, float)):
                        meets_condition = False
                        if condition == '>=' and value >= threshold:
                            meets_condition = True
                        elif condition == '<=' and value <= threshold:
                            meets_condition = True

                        if meets_condition:
                            for return_key in ['return_20d', 'return_10d', 'return_5d']:
                                if return_key in features:
                                    returns.append(features[return_key])
                                    break

            return np.mean(returns) if returns else None
        except:
            return None

    def _calculate_avg_return_for_pattern(self, all_features: List[Dict[str, Any]],
                                        feature_name: str, condition_type: str, threshold_value) -> Optional[float]:
        """计算特定模式的平均收益"""
        if condition_type == 'between':
            min_val, max_val = threshold_value
            return self._calculate_avg_return_for_feature(all_features, feature_name, min_val, max_val)
        else:
            return self._calculate_avg_return_for_threshold(all_features, feature_name, condition_type, threshold_value)


class HistoricalBuyPointStrategyGenerator:
    """历史买点策略生成器 - 核心控制器"""

    def __init__(self,
                 generation_mode: StrategyGenerationMode = StrategyGenerationMode.BALANCED,
                 pattern_method: PatternRecognitionMethod = PatternRecognitionMethod.STATISTICAL):
        """
        初始化策略生成器

        Args:
            generation_mode: 策略生成模式
            pattern_method: 模式识别方法
        """
        self.generation_mode = generation_mode
        self.pattern_method = pattern_method

        # 初始化组件
        self.feature_extractor = TechnicalFeatureExtractor()
        self.pattern_engine = PatternRecognitionEngine(pattern_method)

        # 配置参数
        self._configure_parameters()

    def _configure_parameters(self):
        """根据生成模式配置参数"""
        if self.generation_mode == StrategyGenerationMode.CONSERVATIVE:
            self.pattern_engine.min_confidence = 0.8
            self.pattern_engine.min_frequency = 5
            self.min_patterns = 3
            self.max_patterns = 5
        elif self.generation_mode == StrategyGenerationMode.BALANCED:
            self.pattern_engine.min_confidence = 0.6
            self.pattern_engine.min_frequency = 3
            self.min_patterns = 2
            self.max_patterns = 6
        elif self.generation_mode == StrategyGenerationMode.AGGRESSIVE:
            self.pattern_engine.min_confidence = 0.5
            self.pattern_engine.min_frequency = 2
            self.min_patterns = 1
            self.max_patterns = 8
        else:  # ADAPTIVE
            # 自适应模式会根据数据动态调整
            self.pattern_engine.min_confidence = 0.6
            self.pattern_engine.min_frequency = 3
            self.min_patterns = 2
            self.max_patterns = 6

    @exception_handler(reraise=False)
    @performance_monitor(threshold=30.0)
    def generate_strategy_from_buypoints(self,
                                       buypoint_inputs: List[BuyPointInput],
                                       strategy_name: Optional[str] = None) -> Optional[GeneratedStrategy]:
        """
        从历史买点生成策略

        Args:
            buypoint_inputs: 历史买点输入列表
            strategy_name: 策略名称

        Returns:
            GeneratedStrategy: 生成的策略，如果失败返回None
        """
        if not buypoint_inputs or len(buypoint_inputs) < 3:
            logger.error("买点数据不足，至少需要3个买点才能生成策略")
            return None

        logger.info(f"开始从 {len(buypoint_inputs)} 个历史买点生成策略...")

        # 第一步：提取所有买点的技术特征
        all_features = []
        successful_extractions = 0

        for buypoint_input in buypoint_inputs:
            try:
                features = self.feature_extractor.extract_buypoint_features(buypoint_input)
                if features:
                    all_features.append(features)
                    successful_extractions += 1
                    logger.debug(f"成功提取 {buypoint_input.stock_code} {buypoint_input.buypoint_date} 的特征")
                else:
                    logger.warning(f"无法提取 {buypoint_input.stock_code} {buypoint_input.buypoint_date} 的特征")
            except Exception as e:
                logger.error(f"提取买点特征失败: {buypoint_input.stock_code} {buypoint_input.buypoint_date}, {e}")

        if successful_extractions < 3:
            logger.error(f"成功提取特征的买点数量不足: {successful_extractions}/3")
            return None

        logger.info(f"成功提取 {successful_extractions} 个买点的技术特征")

        # 自适应模式：根据数据质量调整参数
        if self.generation_mode == StrategyGenerationMode.ADAPTIVE:
            self._adaptive_parameter_adjustment(all_features)

        # 第二步：识别技术模式
        technical_patterns = self.pattern_engine.recognize_patterns(all_features)

        if not technical_patterns:
            logger.warning("未能识别出有效的技术模式")
            return None

        logger.info(f"识别出 {len(technical_patterns)} 个技术模式")

        # 第三步：选择最佳模式组合
        selected_patterns = self._select_best_patterns(technical_patterns, all_features)

        if len(selected_patterns) < self.min_patterns:
            logger.warning(f"有效模式数量不足: {len(selected_patterns)}/{self.min_patterns}")
            return None

        # 第四步：生成策略
        strategy = self._create_strategy(
            selected_patterns,
            all_features,
            strategy_name,
            len(buypoint_inputs)
        )

        logger.info(f"成功生成策略: {strategy.strategy_name}")
        return strategy

    def _adaptive_parameter_adjustment(self, all_features: List[Dict[str, Any]]):
        """自适应参数调整"""
        try:
            # 评估数据质量
            feature_count = len(all_features)
            feature_completeness = self._calculate_feature_completeness(all_features)

            # 根据数据质量调整参数
            if feature_count >= 10 and feature_completeness >= 0.8:
                # 高质量数据，可以使用严格标准
                self.pattern_engine.min_confidence = 0.7
                self.pattern_engine.min_frequency = 4
            elif feature_count >= 5 and feature_completeness >= 0.6:
                # 中等质量数据，使用平衡标准
                self.pattern_engine.min_confidence = 0.6
                self.pattern_engine.min_frequency = 3
            else:
                # 低质量数据，使用宽松标准
                self.pattern_engine.min_confidence = 0.5
                self.pattern_engine.min_frequency = 2

            logger.info(f"自适应调整参数: 置信度={self.pattern_engine.min_confidence}, 频率={self.pattern_engine.min_frequency}")

        except Exception as e:
            logger.error(f"自适应参数调整失败: {e}")

    def _calculate_feature_completeness(self, all_features: List[Dict[str, Any]]) -> float:
        """计算特征完整性"""
        if not all_features:
            return 0.0

        total_possible_features = 0
        total_actual_features = 0

        expected_feature_keys = {
            'price', 'volume', 'RSI', 'MACD', 'KDJ',
            'is_green_candle', 'price_position', 'return_20d'
        }

        for features in all_features:
            total_possible_features += len(expected_feature_keys)
            actual_features = sum(1 for key in expected_feature_keys if key in features and features[key] is not None)
            total_actual_features += actual_features

        return total_actual_features / total_possible_features if total_possible_features > 0 else 0.0

    def _select_best_patterns(self, technical_patterns: List[TechnicalPattern], all_features: List[Dict[str, Any]]) -> List[TechnicalPattern]:
        """选择最佳的技术模式组合"""
        if not technical_patterns:
            return []

        # 按综合得分排序
        scored_patterns = []
        for pattern in technical_patterns:
            score = self._calculate_pattern_score(pattern, all_features)
            scored_patterns.append((pattern, score))

        # 按得分排序
        scored_patterns.sort(key=lambda x: x[1], reverse=True)

        # 选择前N个模式，但要避免重复指标
        selected_patterns = []
        used_indicators = set()

        for pattern, score in scored_patterns:
            if len(selected_patterns) >= self.max_patterns:
                break

            # 避免同一指标的重复模式
            base_indicator = pattern.indicator_name.split('_')[0]
            if base_indicator not in used_indicators:
                selected_patterns.append(pattern)
                used_indicators.add(base_indicator)

        return selected_patterns

    def _calculate_pattern_score(self, pattern: TechnicalPattern, all_features: List[Dict[str, Any]]) -> float:
        """计算模式得分"""
        try:
            # 基础得分：置信度 * 频率权重
            confidence_score = pattern.confidence
            frequency_score = min(1.0, pattern.frequency / len(all_features))

            # 收益得分
            return_score = 0.0
            if pattern.avg_return is not None:
                # 将收益转换为得分（正收益加分，负收益减分）
                return_score = max(0.0, min(1.0, (pattern.avg_return + 10) / 20))  # 归一化到[0,1]

            # 综合得分
            total_score = (confidence_score * 0.4 + frequency_score * 0.3 + return_score * 0.3)

            return total_score

        except Exception as e:
            logger.error(f"计算模式得分失败: {e}")
            return 0.0

    def _create_strategy(self,
                        patterns: List[TechnicalPattern],
                        all_features: List[Dict[str, Any]],
                        strategy_name: Optional[str],
                        source_count: int) -> GeneratedStrategy:
        """创建策略对象"""
        # 生成策略名称
        if not strategy_name:
            strategy_name = f"历史买点策略_{datetime.now().strftime('%Y%m%d_%H%M')}"

        # 计算预期成功率和收益
        expected_success_rate = self._estimate_success_rate(patterns, all_features)
        expected_return = self._estimate_expected_return(patterns, all_features)

        # 评估风险级别
        risk_level = self._assess_risk_level(patterns, all_features)

        # 生成策略描述
        description = self._generate_strategy_description(patterns)

        strategy = GeneratedStrategy(
            strategy_name=strategy_name,
            description=description,
            technical_patterns=patterns,
            expected_success_rate=expected_success_rate,
            expected_return=expected_return,
            risk_level=risk_level,
            source_buypoints_count=source_count,
            generation_time=datetime.now(),
            generation_mode=self.generation_mode,
            pattern_method=self.pattern_method
        )

        return strategy

    def _estimate_success_rate(self, patterns: List[TechnicalPattern], all_features: List[Dict[str, Any]]) -> float:
        """估算策略成功率"""
        try:
            if not patterns:
                return 0.0

            # 基于模式置信度的加权平均
            total_confidence = sum(p.confidence for p in patterns)
            weighted_confidence = total_confidence / len(patterns)

            # 考虑模式数量的影响（更多模式可能意味着更严格的条件）
            pattern_count_factor = min(1.0, len(patterns) / 5.0)  # 5个模式为满分

            # 基于历史收益的调整
            positive_return_count = 0
            total_return_count = 0

            for features in all_features:
                for return_key in ['return_20d', 'return_10d', 'return_5d']:
                    if return_key in features and features[return_key] is not None:
                        total_return_count += 1
                        if features[return_key] > 0:
                            positive_return_count += 1
                        break

            historical_success_rate = positive_return_count / total_return_count if total_return_count > 0 else 0.5

            # 综合计算
            estimated_rate = (weighted_confidence * 0.4 + pattern_count_factor * 0.3 + historical_success_rate * 0.3)

            return min(0.95, max(0.1, estimated_rate))  # 限制在合理范围内

        except Exception as e:
            logger.error(f"估算成功率失败: {e}")
            return 0.5

    def _estimate_expected_return(self, patterns: List[TechnicalPattern], all_features: List[Dict[str, Any]]) -> float:
        """估算预期收益"""
        try:
            # 收集所有模式的平均收益
            pattern_returns = [p.avg_return for p in patterns if p.avg_return is not None]

            if pattern_returns:
                # 使用模式收益的平均值
                pattern_avg_return = np.mean(pattern_returns)
            else:
                pattern_avg_return = 0.0

            # 收集历史买点的实际收益
            historical_returns = []
            for features in all_features:
                for return_key in ['return_20d', 'return_10d', 'return_5d']:
                    if return_key in features and features[return_key] is not None:
                        historical_returns.append(features[return_key])
                        break

            historical_avg_return = np.mean(historical_returns) if historical_returns else 0.0

            # 综合估算（偏保守）
            estimated_return = (pattern_avg_return * 0.6 + historical_avg_return * 0.4) * 0.8

            return max(-50.0, min(100.0, estimated_return))  # 限制在合理范围内

        except Exception as e:
            logger.error(f"估算预期收益失败: {e}")
            return 0.0

    def _assess_risk_level(self, patterns: List[TechnicalPattern], all_features: List[Dict[str, Any]]) -> str:
        """评估风险级别"""
        try:
            # 基于模式数量：更多模式意味着更严格的条件，相对低风险
            pattern_risk_score = max(0, 5 - len(patterns)) / 5.0  # 模式越多，风险越低

            # 基于收益波动性
            returns = []
            for features in all_features:
                for return_key in ['return_20d', 'return_10d', 'return_5d']:
                    if return_key in features and features[return_key] is not None:
                        returns.append(features[return_key])
                        break

            volatility_risk_score = 0.5  # 默认中等风险
            if len(returns) > 3:
                return_std = np.std(returns)
                if return_std < 5:
                    volatility_risk_score = 0.2  # 低波动，低风险
                elif return_std > 15:
                    volatility_risk_score = 0.8  # 高波动，高风险

            # 基于最大回撤
            max_drawdowns = [features.get('max_drawdown', 0) for features in all_features if 'max_drawdown' in features]
            drawdown_risk_score = 0.5
            if max_drawdowns:
                avg_max_drawdown = abs(np.mean(max_drawdowns))
                if avg_max_drawdown < 5:
                    drawdown_risk_score = 0.2
                elif avg_max_drawdown > 15:
                    drawdown_risk_score = 0.8

            # 综合风险评估
            overall_risk_score = (pattern_risk_score * 0.3 + volatility_risk_score * 0.4 + drawdown_risk_score * 0.3)

            if overall_risk_score < 0.4:
                return 'LOW'
            elif overall_risk_score < 0.7:
                return 'MEDIUM'
            else:
                return 'HIGH'

        except Exception as e:
            logger.error(f"评估风险级别失败: {e}")
            return 'MEDIUM'

    def _generate_strategy_description(self, patterns: List[TechnicalPattern]) -> str:
        """生成策略描述"""
        try:
            if not patterns:
                return "无有效技术模式的策略"

            description_parts = ["基于历史买点分析生成的选股策略，主要技术条件包括："]

            for i, pattern in enumerate(patterns, 1):
                condition_desc = pattern.to_condition_string()
                confidence_desc = f"(置信度: {pattern.confidence:.2f})"
                description_parts.append(f"{i}. {condition_desc} {confidence_desc}")

            description_parts.append(f"\n该策略基于 {patterns[0].frequency} 个历史买点的共性特征生成。")

            return "\n".join(description_parts)

        except Exception as e:
            logger.error(f"生成策略描述失败: {e}")
            return "策略描述生成失败"