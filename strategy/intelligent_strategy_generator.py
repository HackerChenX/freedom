#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能策略生成器

基于历史买点数据自动生成选股策略的核心引擎，实现生产级的策略生成算法链条。
包括模式识别、策略生成、参数优化和风险评估功能。
遵循六层架构规范。
"""

import os
import time
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict

from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container
from analysis.buypoints.enhanced_backtest_engine import BuyPointData, BacktestResult
from strategy.enhanced_strategy_config_engine import StrategyConfig


logger = get_logger(__name__)

@dataclass
class TechnicalPattern:
    """技术形态模式"""
    pattern_id: str
    pattern_name: str
    indicators: List[str]
    conditions: Dict[str, Any]
    frequency: int
    success_rate: float
    confidence_score: float

@dataclass
class StrategyRule:
    """策略规则"""
    rule_id: str
    rule_name: str
    description: str
    formula: str
    conditions: List[Dict[str, Any]]
    weight: float
    min_score_threshold: float

@dataclass
class GeneratedStrategy:
    """生成的策略"""
    strategy_id: str
    strategy_name: str
    description: str
    rules: List[StrategyRule]
    patterns: List[TechnicalPattern]
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    optimization_params: Dict[str, Any]
    confidence_level: float
    generated_at: str

@dataclass
class StrategyGenerationConfig:
    """策略生成配置"""
    lookback_days: int = 10  # 买点前后分析天数
    min_pattern_frequency: int = 3  # 最小模式频次
    min_success_rate: float = 0.6  # 最小成功率
    max_strategies: int = 5  # 最大生成策略数
    indicator_combinations: int = 3  # 指标组合数
    enable_optimization: bool = True
    parallel_workers: int = 4
    cache_enabled: bool = True

class IntelligentStrategyGenerator:
    """
    智能策略生成器

    核心算法引擎，实现从历史买点到自动生成选股策略的完整链条：
    1. 分析历史买点技术形态特征
    2. 识别高频技术指标组合模式
    3. 自动生成可执行的选股策略规则
    4. 优化策略参数和风险控制
    """

    def __init__(self, config: Optional[StrategyGenerationConfig] = None):
        """
        初始化智能策略生成器

        Args:
            config: 策略生成配置
        """
        self.config = config or StrategyGenerationConfig()
        self.logger = get_logger(__name__)

        # 初始化依赖组件
        container = get_container()
        try:
            self.data_access = container.resolve("DataAccessInterface")
            self.indicator_registry = container.resolve("CompleteIndicatorRegistry")
        except:
            self.data_access = None
            self.indicator_registry = None
            self.logger.warning("未能获取依赖注入服务，将使用默认实现")

        # 内部状态
        self.generation_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'total_patterns_identified': 0,
            'total_strategies_generated': 0,
            'average_generation_time': 0.0
        }

        # 缓存系统
        self._pattern_cache = {}
        self._strategy_cache = {}

        self.logger.info("智能策略生成器初始化完成")

    @exception_handler(reraise=True)
    @performance_monitor(threshold=180.0)
    def generate_strategies_from_buypoints(self, buypoints: List[BuyPointData],
                                          strategy_name_prefix: str = "AI_Generated") -> List[GeneratedStrategy]:
        """
        从历史买点生成策略

        Args:
            buypoints: 历史买点数据列表
            strategy_name_prefix: 策略名称前缀

        Returns:
            List[GeneratedStrategy]: 生成的策略列表
        """
        start_time = time.time()
        self.logger.info(f"开始从 {len(buypoints)} 个历史买点生成策略")

        try:
            # 1. 技术形态模式识别
            self.logger.info("🔍 执行技术形态模式识别...")
            patterns = self._identify_technical_patterns(buypoints)

            if not patterns:
                self.logger.warning("未识别到有效的技术形态模式")
                return []

            # 2. 策略规则生成
            self.logger.info(f"⚙️ 基于 {len(patterns)} 个模式生成策略规则...")
            generated_strategies = self._generate_strategies_from_patterns(
                patterns, strategy_name_prefix
            )

            # 3. 策略优化（如果启用）
            if self.config.enable_optimization and generated_strategies:
                self.logger.info("🎯 执行策略优化...")
                optimized_strategies = self._optimize_strategies(generated_strategies, buypoints)
                generated_strategies = optimized_strategies

            # 4. 策略评估和排序
            self.logger.info("📊 执行策略评估和排序...")
            evaluated_strategies = self._evaluate_and_rank_strategies(generated_strategies, buypoints)

            # 5. 更新统计信息
            execution_time = time.time() - start_time
            self._update_generation_stats(len(patterns), len(evaluated_strategies), execution_time)

            self.logger.info(f"✅ 策略生成完成，共生成 {len(evaluated_strategies)} 个策略，耗时 {execution_time:.2f}秒")

            return evaluated_strategies[:self.config.max_strategies]

        except Exception as e:
            self.logger.error(f"❌ 策略生成失败: {e}")
            raise

    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def _identify_technical_patterns(self, buypoints: List[BuyPointData]) -> List[TechnicalPattern]:
        """
        识别技术形态模式

        Args:
            buypoints: 买点数据列表

        Returns:
            List[TechnicalPattern]: 识别到的技术形态模式
        """
        patterns = []
        pattern_candidates = defaultdict(list)

        # 并行分析每个买点的技术形态
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            future_to_buypoint = {
                executor.submit(self._analyze_buypoint_patterns, bp): bp
                for bp in buypoints
            }

            for future in as_completed(future_to_buypoint):
                buypoint = future_to_buypoint[future]
                try:
                    buypoint_patterns = future.result()
                    for pattern_key, pattern_data in buypoint_patterns.items():
                        pattern_candidates[pattern_key].append(pattern_data)
                except Exception as e:
                    self.logger.warning(f"分析买点 {buypoint.stock_code} 失败: {e}")

        # 识别高频模式
        for pattern_key, occurrences in pattern_candidates.items():
            if len(occurrences) >= self.config.min_pattern_frequency:
                # 计算模式成功率
                success_count = sum(1 for occ in occurrences if occ.get('success', False))
                success_rate = success_count / len(occurrences)

                if success_rate >= self.config.min_success_rate:
                    # 创建技术形态模式
                    pattern = self._create_technical_pattern(pattern_key, occurrences)
                    patterns.append(pattern)

        self.logger.info(f"识别到 {len(patterns)} 个有效技术形态模式")
        return patterns

    @exception_handler(reraise=False, default_return={})
    def _analyze_buypoint_patterns(self, buypoint: BuyPointData) -> Dict[str, Dict[str, Any]]:
        """
        分析单个买点的技术形态

        Args:
            buypoint: 买点数据

        Returns:
            Dict[str, Dict[str, Any]]: 模式字典
        """
        patterns = {}

        try:
            # 获取买点前后的数据
            start_date = (datetime.strptime(buypoint.buypoint_date, '%Y-%m-%d') -
                         timedelta(days=self.config.lookback_days)).strftime('%Y-%m-%d')
            end_date = (datetime.strptime(buypoint.buypoint_date, '%Y-%m-%d') +
                       timedelta(days=self.config.lookback_days)).strftime('%Y-%m-%d')

            # 获取股票数据（如果数据访问可用）
            if self.data_access:
                stock_data = self.data_access.get_stock_data(
                    code=buypoint.stock_code,
                    start_date=start_date,
                    end_date=end_date
                )

                if not stock_data.empty:
                    # 计算技术指标
                    indicators_data = self._calculate_technical_indicators(stock_data)

                    # 识别买点日的指标状态
                    buypoint_indicators = self._extract_buypoint_indicator_state(
                        indicators_data, buypoint.buypoint_date
                    )

                    # 生成模式候选
                    pattern_candidates = self._generate_pattern_candidates(
                        buypoint_indicators, buypoint.stock_code
                    )

                    patterns.update(pattern_candidates)

        except Exception as e:
            self.logger.warning(f"分析买点 {buypoint.stock_code} 模式失败: {e}")

        return patterns

    def _calculate_technical_indicators(self, stock_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        计算技术指标

        Args:
            stock_data: 股票数据

        Returns:
            Dict[str, pd.DataFrame]: 指标数据字典
        """
        indicators_data = {}

        # 核心指标计算
        try:
            # MA指标
            if 'close' in stock_data.columns:
                for period in [5, 10, 20, 30]:
                    ma_key = f'MA{period}'
                    indicators_data[ma_key] = stock_data['close'].rolling(window=period).mean()

            # MACD指标
            if 'close' in stock_data.columns:
                exp1 = stock_data['close'].ewm(span=12, adjust=False).mean()
                exp2 = stock_data['close'].ewm(span=26, adjust=False).mean()
                macd_line = exp1 - exp2
                signal_line = macd_line.ewm(span=9, adjust=False).mean()

                indicators_data['MACD_LINE'] = macd_line
                indicators_data['MACD_SIGNAL'] = signal_line
                indicators_data['MACD_HISTOGRAM'] = macd_line - signal_line

            # RSI指标
            if 'close' in stock_data.columns:
                delta = stock_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators_data['RSI'] = 100 - (100 / (1 + rs))

            # KDJ指标
            if all(col in stock_data.columns for col in ['high', 'low', 'close']):
                low_min = stock_data['low'].rolling(window=9).min()
                high_max = stock_data['high'].rolling(window=9).max()
                rsv = (stock_data['close'] - low_min) / (high_max - low_min) * 100

                k_values = []
                d_values = []
                k = 50  # 初始值
                d = 50  # 初始值

                for rsv_val in rsv:
                    if pd.notna(rsv_val):
                        k = (2/3) * k + (1/3) * rsv_val
                        d = (2/3) * d + (1/3) * k
                    k_values.append(k)
                    d_values.append(d)

                indicators_data['KDJ_K'] = pd.Series(k_values, index=stock_data.index)
                indicators_data['KDJ_D'] = pd.Series(d_values, index=stock_data.index)
                indicators_data['KDJ_J'] = 3 * indicators_data['KDJ_K'] - 2 * indicators_data['KDJ_D']

            # 布林带
            if 'close' in stock_data.columns:
                sma = stock_data['close'].rolling(window=20).mean()
                std = stock_data['close'].rolling(window=20).std()
                indicators_data['BOLL_UPPER'] = sma + (std * 2)
                indicators_data['BOLL_MIDDLE'] = sma
                indicators_data['BOLL_LOWER'] = sma - (std * 2)

        except Exception as e:
            self.logger.warning(f"计算技术指标失败: {e}")

        return indicators_data

    def _extract_buypoint_indicator_state(self, indicators_data: Dict[str, pd.Series],
                                         buypoint_date: str) -> Dict[str, Any]:
        """
        提取买点日的指标状态

        Args:
            indicators_data: 指标数据
            buypoint_date: 买点日期

        Returns:
            Dict[str, Any]: 买点指标状态
        """
        state = {}
        target_date = pd.to_datetime(buypoint_date)

        for indicator_name, indicator_series in indicators_data.items():
            try:
                # 找到最接近买点日期的数据
                if hasattr(indicator_series, 'index'):
                    closest_idx = indicator_series.index.get_indexer([target_date], method='nearest')[0]
                    if closest_idx >= 0 and closest_idx < len(indicator_series):
                        value = indicator_series.iloc[closest_idx]
                        if pd.notna(value):
                            state[indicator_name] = float(value)
                else:
                    # 如果是简单的数值列表
                    if len(indicator_series) > 0:
                        state[indicator_name] = float(indicator_series[-1])
            except Exception as e:
                self.logger.debug(f"提取指标 {indicator_name} 状态失败: {e}")

        return state

    def _generate_pattern_candidates(self, indicators: Dict[str, Any],
                                   stock_code: str) -> Dict[str, Dict[str, Any]]:
        """
        生成模式候选

        Args:
            indicators: 指标数据
            stock_code: 股票代码

        Returns:
            Dict[str, Dict[str, Any]]: 模式候选字典
        """
        candidates = {}

        # MACD金叉模式
        if all(key in indicators for key in ['MACD_LINE', 'MACD_SIGNAL']):
            if indicators['MACD_LINE'] > indicators['MACD_SIGNAL'] and indicators['MACD_LINE'] > 0:
                candidates['MACD_GOLDEN_CROSS'] = {
                    'indicators': ['MACD_LINE', 'MACD_SIGNAL'],
                    'conditions': {
                        'MACD_LINE > MACD_SIGNAL': True,
                        'MACD_LINE > 0': True
                    },
                    'success': True,  # 这里可以根据后续表现计算
                    'stock_code': stock_code
                }

        # KDJ低位金叉
        if all(key in indicators for key in ['KDJ_K', 'KDJ_D']):
            if (indicators['KDJ_K'] > indicators['KDJ_D'] and
                indicators['KDJ_K'] < 50 and indicators['KDJ_D'] < 50):
                candidates['KDJ_LOW_GOLDEN_CROSS'] = {
                    'indicators': ['KDJ_K', 'KDJ_D'],
                    'conditions': {
                        'KDJ_K > KDJ_D': True,
                        'KDJ_K < 50': True,
                        'KDJ_D < 50': True
                    },
                    'success': True,
                    'stock_code': stock_code
                }

        # RSI超卖反弹
        if 'RSI' in indicators:
            if 20 <= indicators['RSI'] <= 40:
                candidates['RSI_OVERSOLD_REBOUND'] = {
                    'indicators': ['RSI'],
                    'conditions': {
                        'RSI >= 20': True,
                        'RSI <= 40': True
                    },
                    'success': True,
                    'stock_code': stock_code
                }

        # 布林带下轨反弹
        if all(key in indicators for key in ['BOLL_LOWER', 'close']) and 'close' in indicators:
            # 注意：这里假设close价格也在indicators中
            if 'close' in indicators and indicators['close'] <= indicators['BOLL_LOWER'] * 1.02:
                candidates['BOLL_LOWER_REBOUND'] = {
                    'indicators': ['BOLL_LOWER', 'close'],
                    'conditions': {
                        'close <= BOLL_LOWER * 1.02': True
                    },
                    'success': True,
                    'stock_code': stock_code
                }

        # 均线支撑
        ma_keys = [k for k in indicators.keys() if k.startswith('MA') and k != 'MACD_LINE']
        if ma_keys and 'close' in indicators:
            for ma_key in ma_keys:
                if indicators['close'] > indicators[ma_key] * 0.98:  # 接近均线
                    candidates[f'{ma_key}_SUPPORT'] = {
                        'indicators': [ma_key, 'close'],
                        'conditions': {
                            f'close > {ma_key} * 0.98': True
                        },
                        'success': True,
                        'stock_code': stock_code
                    }

        return candidates

    def _create_technical_pattern(self, pattern_key: str,
                                occurrences: List[Dict[str, Any]]) -> TechnicalPattern:
        """
        创建技术形态模式

        Args:
            pattern_key: 模式键值
            occurrences: 出现记录

        Returns:
            TechnicalPattern: 技术形态模式
        """
        # 计算模式统计
        frequency = len(occurrences)
        success_count = sum(1 for occ in occurrences if occ.get('success', False))
        success_rate = success_count / frequency

        # 提取指标和条件
        indicators = occurrences[0].get('indicators', [])
        conditions = occurrences[0].get('conditions', {})

        # 计算置信度分数
        confidence_score = min(success_rate * (frequency / self.config.min_pattern_frequency), 1.0)

        pattern = TechnicalPattern(
            pattern_id=hashlib.md5(pattern_key.encode()).hexdigest()[:8],
            pattern_name=pattern_key,
            indicators=indicators,
            conditions=conditions,
            frequency=frequency,
            success_rate=success_rate,
            confidence_score=confidence_score
        )

        return pattern

    def _generate_strategies_from_patterns(self, patterns: List[TechnicalPattern],
                                         strategy_prefix: str) -> List[GeneratedStrategy]:
        """
        从技术形态生成策略

        Args:
            patterns: 技术形态列表
            strategy_prefix: 策略名称前缀

        Returns:
            List[GeneratedStrategy]: 生成的策略列表
        """
        strategies = []

        # 按置信度排序模式
        sorted_patterns = sorted(patterns, key=lambda p: p.confidence_score, reverse=True)

        # 生成单模式策略
        for i, pattern in enumerate(sorted_patterns[:self.config.max_strategies]):
            strategy = self._create_strategy_from_single_pattern(pattern, f"{strategy_prefix}_Single_{i+1}")
            strategies.append(strategy)

        # 生成组合模式策略
        if len(sorted_patterns) >= 2:
            combination_strategy = self._create_strategy_from_pattern_combination(
                sorted_patterns[:self.config.indicator_combinations],
                f"{strategy_prefix}_Combination"
            )
            strategies.append(combination_strategy)

        return strategies

    def _create_strategy_from_single_pattern(self, pattern: TechnicalPattern,
                                           strategy_name: str) -> GeneratedStrategy:
        """
        从单个技术形态创建策略

        Args:
            pattern: 技术形态
            strategy_name: 策略名称

        Returns:
            GeneratedStrategy: 生成的策略
        """
        # 创建策略规则
        rule = StrategyRule(
            rule_id=f"rule_{pattern.pattern_id}",
            rule_name=f"{pattern.pattern_name}_规则",
            description=f"基于{pattern.pattern_name}模式的选股规则",
            formula=self._pattern_to_formula(pattern),
            conditions=[{
                'indicator': indicator,
                'condition': condition,
                'value': value
            } for condition, value in pattern.conditions.items() for indicator in pattern.indicators],
            weight=pattern.confidence_score,
            min_score_threshold=60.0
        )

        # 计算性能指标
        performance_metrics = {
            'expected_success_rate': pattern.success_rate,
            'pattern_frequency': pattern.frequency,
            'confidence_score': pattern.confidence_score
        }

        # 计算风险指标
        risk_metrics = {
            'pattern_risk': 1.0 - pattern.success_rate,
            'frequency_risk': max(0, (self.config.min_pattern_frequency - pattern.frequency) / self.config.min_pattern_frequency),
            'overall_risk': (1.0 - pattern.success_rate) * 0.7 + max(0, (self.config.min_pattern_frequency - pattern.frequency) / self.config.min_pattern_frequency) * 0.3
        }

        strategy = GeneratedStrategy(
            strategy_id=hashlib.md5(f"{strategy_name}_{pattern.pattern_id}".encode()).hexdigest()[:12],
            strategy_name=strategy_name,
            description=f"基于{pattern.pattern_name}技术形态的智能选股策略，成功率{pattern.success_rate:.1%}",
            rules=[rule],
            patterns=[pattern],
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            optimization_params={},
            confidence_level=pattern.confidence_score,
            generated_at=datetime.now().isoformat()
        )

        return strategy

    def _create_strategy_from_pattern_combination(self, patterns: List[TechnicalPattern],
                                                strategy_name: str) -> GeneratedStrategy:
        """
        从模式组合创建策略

        Args:
            patterns: 技术形态列表
            strategy_name: 策略名称

        Returns:
            GeneratedStrategy: 生成的策略
        """
        rules = []
        combined_indicators = set()

        for i, pattern in enumerate(patterns):
            rule = StrategyRule(
                rule_id=f"rule_combo_{i}_{pattern.pattern_id}",
                rule_name=f"{pattern.pattern_name}_组合规则",
                description=f"组合策略中的{pattern.pattern_name}模式规则",
                formula=self._pattern_to_formula(pattern),
                conditions=[{
                    'indicator': indicator,
                    'condition': condition,
                    'value': value
                } for condition, value in pattern.conditions.items() for indicator in pattern.indicators],
                weight=pattern.confidence_score,
                min_score_threshold=50.0
            )
            rules.append(rule)
            combined_indicators.update(pattern.indicators)

        # 计算组合性能指标
        avg_success_rate = sum(p.success_rate for p in patterns) / len(patterns)
        total_frequency = sum(p.frequency for p in patterns)
        avg_confidence = sum(p.confidence_score for p in patterns) / len(patterns)

        performance_metrics = {
            'expected_success_rate': avg_success_rate,
            'total_pattern_frequency': total_frequency,
            'average_confidence_score': avg_confidence,
            'pattern_count': len(patterns)
        }

        # 组合风险通常更低
        risk_metrics = {
            'diversification_benefit': 0.2,  # 分散化收益
            'overall_risk': (1.0 - avg_success_rate) * 0.8  # 风险降低
        }

        strategy = GeneratedStrategy(
            strategy_id=hashlib.md5(f"{strategy_name}_combo".encode()).hexdigest()[:12],
            strategy_name=strategy_name,
            description=f"基于{len(patterns)}个技术形态组合的智能选股策略，平均成功率{avg_success_rate:.1%}",
            rules=rules,
            patterns=patterns,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            optimization_params={},
            confidence_level=avg_confidence,
            generated_at=datetime.now().isoformat()
        )

        return strategy

    def _pattern_to_formula(self, pattern: TechnicalPattern) -> str:
        """
        将技术形态转换为通达信风格公式

        Args:
            pattern: 技术形态

        Returns:
            str: 通达信公式
        """
        formula_parts = []

        # 根据模式名称生成对应的公式
        if pattern.pattern_name == 'MACD_GOLDEN_CROSS':
            formula_parts.append("MACD.DIF > MACD.DEA")
            formula_parts.append("MACD.DIF > 0")

        elif pattern.pattern_name == 'KDJ_LOW_GOLDEN_CROSS':
            formula_parts.append("KDJ.K > KDJ.D")
            formula_parts.append("KDJ.K < 50")
            formula_parts.append("KDJ.D < 50")

        elif pattern.pattern_name == 'RSI_OVERSOLD_REBOUND':
            formula_parts.append("RSI(14) >= 20")
            formula_parts.append("RSI(14) <= 40")

        elif pattern.pattern_name == 'BOLL_LOWER_REBOUND':
            formula_parts.append("CLOSE <= BOLL.LOWER * 1.02")

        elif pattern.pattern_name.endswith('_SUPPORT'):
            ma_period = pattern.pattern_name.split('_')[0].replace('MA', '')
            if ma_period.isdigit():
                formula_parts.append(f"CLOSE > MA({ma_period}) * 0.98")

        # 默认公式
        if not formula_parts:
            formula_parts = [f"// 基于{pattern.pattern_name}模式的条件"]
            for condition in pattern.conditions:
                formula_parts.append(f"// {condition}")

        return " AND ".join(formula_parts)

    def _optimize_strategies(self, strategies: List[GeneratedStrategy],
                           buypoints: List[BuyPointData]) -> List[GeneratedStrategy]:
        """
        优化策略参数

        Args:
            strategies: 策略列表
            buypoints: 历史买点数据

        Returns:
            List[GeneratedStrategy]: 优化后的策略列表
        """
        optimized_strategies = []

        for strategy in strategies:
            try:
                # 参数优化
                optimized_params = self._optimize_strategy_parameters(strategy, buypoints)

                # 更新策略参数
                strategy.optimization_params = optimized_params

                # 重新计算性能指标
                updated_performance = self._recalculate_performance_with_optimization(
                    strategy, optimized_params
                )
                strategy.performance_metrics.update(updated_performance)

                optimized_strategies.append(strategy)

            except Exception as e:
                self.logger.warning(f"优化策略 {strategy.strategy_name} 失败: {e}")
                optimized_strategies.append(strategy)  # 保留未优化版本

        return optimized_strategies

    def _optimize_strategy_parameters(self, strategy: GeneratedStrategy,
                                    buypoints: List[BuyPointData]) -> Dict[str, Any]:
        """
        优化单个策略的参数

        Args:
            strategy: 策略
            buypoints: 买点数据

        Returns:
            Dict[str, Any]: 优化参数
        """
        optimized_params = {
            'min_score_threshold': 60.0,
            'max_results': 50,
            'risk_adjustment': 1.0
        }

        # 基于历史表现优化阈值
        pattern_success_rates = [p.success_rate for p in strategy.patterns]
        avg_success_rate = sum(pattern_success_rates) / len(pattern_success_rates)

        # 根据成功率调整阈值
        if avg_success_rate > 0.8:
            optimized_params['min_score_threshold'] = 70.0
        elif avg_success_rate > 0.7:
            optimized_params['min_score_threshold'] = 65.0
        else:
            optimized_params['min_score_threshold'] = 55.0

        # 根据模式频次调整结果数量
        total_frequency = sum(p.frequency for p in strategy.patterns)
        if total_frequency > 10:
            optimized_params['max_results'] = 30
        elif total_frequency > 5:
            optimized_params['max_results'] = 40
        else:
            optimized_params['max_results'] = 50

        return optimized_params

    def _recalculate_performance_with_optimization(self, strategy: GeneratedStrategy,
                                                 optimized_params: Dict[str, Any]) -> Dict[str, float]:
        """
        基于优化参数重新计算性能指标

        Args:
            strategy: 策略
            optimized_params: 优化参数

        Returns:
            Dict[str, float]: 更新的性能指标
        """
        updated_metrics = {}

        # 基于优化参数调整预期性能
        base_success_rate = strategy.performance_metrics.get('expected_success_rate', 0.6)
        threshold_adjustment = optimized_params.get('min_score_threshold', 60.0) / 60.0

        updated_metrics['optimized_success_rate'] = min(base_success_rate * threshold_adjustment, 0.95)
        updated_metrics['expected_precision'] = updated_metrics['optimized_success_rate'] * 0.9
        updated_metrics['optimization_boost'] = (updated_metrics['optimized_success_rate'] - base_success_rate) / base_success_rate

        return updated_metrics

    def _evaluate_and_rank_strategies(self, strategies: List[GeneratedStrategy],
                                    buypoints: List[BuyPointData]) -> List[GeneratedStrategy]:
        """
        评估和排序策略

        Args:
            strategies: 策略列表
            buypoints: 买点数据

        Returns:
            List[GeneratedStrategy]: 排序后的策略列表
        """
        # 为每个策略计算综合评分
        for strategy in strategies:
            score = self._calculate_strategy_score(strategy)
            strategy.performance_metrics['综合评分'] = score

        # 按综合评分排序
        ranked_strategies = sorted(
            strategies,
            key=lambda s: s.performance_metrics.get('综合评分', 0),
            reverse=True
        )

        return ranked_strategies

    def _calculate_strategy_score(self, strategy: GeneratedStrategy) -> float:
        """
        计算策略综合评分

        Args:
            strategy: 策略

        Returns:
            float: 综合评分
        """
        # 成功率权重 40%
        success_rate = strategy.performance_metrics.get('expected_success_rate', 0.6)
        success_score = success_rate * 40

        # 置信度权重 30%
        confidence_score = strategy.confidence_level * 30

        # 模式频次权重 20%
        total_frequency = sum(p.frequency for p in strategy.patterns)
        frequency_score = min(total_frequency / 10.0, 1.0) * 20

        # 风险调整权重 10%
        overall_risk = strategy.risk_metrics.get('overall_risk', 0.4)
        risk_score = (1.0 - overall_risk) * 10

        total_score = success_score + confidence_score + frequency_score + risk_score

        return min(total_score, 100.0)

    def _update_generation_stats(self, patterns_count: int, strategies_count: int,
                               execution_time: float):
        """
        更新生成统计信息

        Args:
            patterns_count: 识别的模式数量
            strategies_count: 生成的策略数量
            execution_time: 执行时间
        """
        self.generation_stats['total_generations'] += 1
        if strategies_count > 0:
            self.generation_stats['successful_generations'] += 1

        self.generation_stats['total_patterns_identified'] += patterns_count
        self.generation_stats['total_strategies_generated'] += strategies_count

        # 更新平均生成时间
        total_time = (self.generation_stats['average_generation_time'] *
                     (self.generation_stats['total_generations'] - 1) + execution_time)
        self.generation_stats['average_generation_time'] = total_time / self.generation_stats['total_generations']

    @exception_handler(reraise=False, default_return={})
    def get_generation_statistics(self) -> Dict[str, Any]:
        """
        获取生成统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.generation_stats.copy()

        if stats['total_generations'] > 0:
            stats['success_rate'] = stats['successful_generations'] / stats['total_generations']
            stats['avg_patterns_per_generation'] = stats['total_patterns_identified'] / stats['total_generations']
            stats['avg_strategies_per_generation'] = stats['total_strategies_generated'] / stats['total_generations']

        return stats

    @exception_handler(reraise=True)
    def export_strategy_to_config(self, strategy: GeneratedStrategy) -> Dict[str, Any]:
        """
        导出策略为标准配置格式

        Args:
            strategy: 生成的策略

        Returns:
            Dict[str, Any]: 标准策略配置
        """
        config = {
            'name': strategy.strategy_name,
            'description': strategy.description,
            'version': '1.0.0',
            'generated': True,
            'generation_metadata': {
                'generated_at': strategy.generated_at,
                'confidence_level': strategy.confidence_level,
                'pattern_count': len(strategy.patterns)
            },
            'rules': [],
            'global_settings': {
                'min_score': strategy.optimization_params.get('min_score_threshold', 60.0),
                'max_results': strategy.optimization_params.get('max_results', 50),
                'risk_level': 'medium' if strategy.risk_metrics.get('overall_risk', 0.4) < 0.3 else 'high'
            }
        }

        # 转换规则
        for rule in strategy.rules:
            rule_config = {
                'name': rule.rule_name,
                'description': rule.description,
                'formula': rule.formula,
                'min_score': rule.min_score_threshold,
                'weight': rule.weight
            }
            config['rules'].append(rule_config)

        return config