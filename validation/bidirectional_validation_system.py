#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双向验证系统

实现PMO计划中的双向验证核心逻辑：
1. 前向验证器（Forward Validator）：策略 → 买点验证
2. 后向验证器（Backward Validator）：买点 → 策略验证
3. 验证报告生成器（Validation Report Generator）

质量标准：
- 验证覆盖率 100%
- 假阳性率 < 5%
- 报告生成时间 < 10秒
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container
from db.interfaces.data_access_interface import DataAccessInterface

logger = get_logger(__name__)


@dataclass
class ValidationMetrics:
    """验证指标数据类"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    coverage_rate: float


@dataclass
class ForwardValidationResult:
    """前向验证结果"""
    strategy_name: str
    validation_time: datetime
    total_patterns: int
    matched_buypoints: int
    validation_metrics: ValidationMetrics
    pattern_match_details: List[Dict[str, Any]]
    signal_consistency_score: float
    performance_indicators: Dict[str, Any]
    recommendations: List[str]


@dataclass
class BackwardValidationResult:
    """后向验证结果"""
    buypoint_count: int
    validation_time: datetime
    strategy_coverage: float
    historical_replay_results: Dict[str, Any]
    statistical_tests: Dict[str, float]
    buypoint_pattern_analysis: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]


@dataclass
class BidirectionalValidationReport:
    """双向验证报告"""
    report_id: str
    generation_time: datetime
    forward_validation: ForwardValidationResult
    backward_validation: BackwardValidationResult
    overall_validation_score: float
    consistency_analysis: Dict[str, Any]
    risk_profile: Dict[str, Any]
    performance_summary: Dict[str, Any]
    improvement_suggestions: List[str]
    validation_passed: bool
    execution_time: float


class ForwardValidator:
    """前向验证器：策略 → 买点验证"""

    def __init__(self, data_manager: DataAccessInterface):
        """初始化前向验证器"""
        self.data_manager = data_manager
        self.validation_threshold = 0.95  # 验证通过阈值
        self.false_positive_threshold = 0.05  # 假阳性率阈值

    @exception_handler(reraise=False)
    @performance_monitor(threshold=30.0)
    def validate_strategy_against_buypoints(self,
                                          strategy,
                                          original_buypoints: List[Dict[str, Any]],
                                          validation_period_days: int = 252) -> ForwardValidationResult:
        """
        验证生成的策略是否能准确识别原始买点

        Args:
            strategy: 生成的策略对象
            original_buypoints: 原始买点列表
            validation_period_days: 验证时间窗口（交易日）

        Returns:
            ForwardValidationResult: 前向验证结果
        """
        logger.info(f"开始前向验证：策略'{strategy.strategy_name}' → {len(original_buypoints)}个买点")

        validation_start_time = datetime.now()
        pattern_match_details = []
        matched_buypoints = 0
        signal_consistency_scores = []

        try:
            # 遍历每个原始买点，验证策略是否能识别
            for i, buypoint in enumerate(original_buypoints):
                stock_code = buypoint.get('stock_code', '')
                buypoint_date = buypoint.get('buypoint_date', '')

                if not stock_code or not buypoint_date:
                    logger.warning(f"买点数据不完整: {buypoint}")
                    continue

                # 获取买点周围的数据
                match_result = self._validate_single_buypoint(
                    strategy, stock_code, buypoint_date, validation_period_days
                )

                pattern_match_details.append(match_result)

                if match_result['is_matched']:
                    matched_buypoints += 1
                    signal_consistency_scores.append(match_result['consistency_score'])

                # 进度日志
                if (i + 1) % 10 == 0:
                    logger.info(f"已验证 {i + 1}/{len(original_buypoints)} 个买点")

            # 计算验证指标
            validation_metrics = self._calculate_validation_metrics(
                len(original_buypoints), matched_buypoints, pattern_match_details
            )

            # 计算信号一致性得分
            signal_consistency_score = np.mean(signal_consistency_scores) if signal_consistency_scores else 0.0

            # 生成性能指标
            performance_indicators = self._generate_performance_indicators(
                pattern_match_details, validation_metrics
            )

            # 生成建议
            recommendations = self._generate_forward_recommendations(
                validation_metrics, signal_consistency_score, performance_indicators
            )

            result = ForwardValidationResult(
                strategy_name=strategy.strategy_name,
                validation_time=validation_start_time,
                total_patterns=len(strategy.technical_patterns),
                matched_buypoints=matched_buypoints,
                validation_metrics=validation_metrics,
                pattern_match_details=pattern_match_details,
                signal_consistency_score=signal_consistency_score,
                performance_indicators=performance_indicators,
                recommendations=recommendations
            )

            logger.info(f"前向验证完成: 匹配率 {validation_metrics.accuracy:.2%}, "
                       f"信号一致性 {signal_consistency_score:.3f}")

            return result

        except Exception as e:
            logger.error(f"前向验证失败: {e}")
            # 返回空结果但包含错误信息
            empty_metrics = ValidationMetrics(0, 0, 0, 0, 1.0, 0, (0, 0), 0)
            return ForwardValidationResult(
                strategy_name=strategy.strategy_name,
                validation_time=validation_start_time,
                total_patterns=0,
                matched_buypoints=0,
                validation_metrics=empty_metrics,
                pattern_match_details=[],
                signal_consistency_score=0.0,
                performance_indicators={},
                recommendations=[f"验证过程出现错误: {str(e)}"]
            )

    def _validate_single_buypoint(self, strategy, stock_code: str, buypoint_date: str,
                                validation_period_days: int) -> Dict[str, Any]:
        """验证单个买点"""
        try:
            # 计算数据获取范围
            buypoint_dt = datetime.strptime(buypoint_date, '%Y-%m-%d')
            start_date = (buypoint_dt - timedelta(days=validation_period_days)).strftime('%Y-%m-%d')
            end_date = (buypoint_dt + timedelta(days=30)).strftime('%Y-%m-%d')

            # 获取股票数据
            stock_data = self.data_manager.get_stock_data_data_access_manager(
                stock_code, start_date, end_date
            )

            if stock_data.empty:
                logger.warning(f"无法获取股票数据: {stock_code} {buypoint_date}")
                return {
                    'stock_code': stock_code,
                    'buypoint_date': buypoint_date,
                    'is_matched': False,
                    'consistency_score': 0.0,
                    'pattern_matches': {},
                    'error': '数据获取失败'
                }

            # 找到买点在数据中的位置
            buypoint_idx = self._find_buypoint_index(stock_data, buypoint_date)
            if buypoint_idx == -1:
                return {
                    'stock_code': stock_code,
                    'buypoint_date': buypoint_date,
                    'is_matched': False,
                    'consistency_score': 0.0,
                    'pattern_matches': {},
                    'error': '买点日期不在数据范围内'
                }

            # 验证每个技术模式
            pattern_matches = {}
            total_patterns = len(strategy.technical_patterns)
            matched_patterns = 0

            for pattern in strategy.technical_patterns:
                match_result = self._check_pattern_match(
                    stock_data, buypoint_idx, pattern
                )
                pattern_matches[pattern.indicator_name] = match_result
                if match_result['is_match']:
                    matched_patterns += 1

            # 计算一致性得分
            consistency_score = matched_patterns / total_patterns if total_patterns > 0 else 0.0

            # 判断是否匹配（需要80%以上的模式匹配）
            is_matched = consistency_score >= 0.8

            return {
                'stock_code': stock_code,
                'buypoint_date': buypoint_date,
                'is_matched': is_matched,
                'consistency_score': consistency_score,
                'pattern_matches': pattern_matches,
                'matched_patterns': matched_patterns,
                'total_patterns': total_patterns,
                'error': None
            }

        except Exception as e:
            logger.error(f"验证单个买点失败: {stock_code} {buypoint_date}, {e}")
            return {
                'stock_code': stock_code,
                'buypoint_date': buypoint_date,
                'is_matched': False,
                'consistency_score': 0.0,
                'pattern_matches': {},
                'error': str(e)
            }

    def _find_buypoint_index(self, stock_data: pd.DataFrame, buypoint_date: str) -> int:
        """在股票数据中找到买点日期的索引"""
        try:
            if 'date' in stock_data.columns:
                date_match = stock_data[stock_data['date'] == buypoint_date]
                if not date_match.empty:
                    return date_match.index[0]

            # 如果精确匹配失败，找最近的交易日
            buypoint_dt = datetime.strptime(buypoint_date, '%Y-%m-%d')
            if 'date' in stock_data.columns:
                stock_data['date_dt'] = pd.to_datetime(stock_data['date'])
                date_diffs = (stock_data['date_dt'] - buypoint_dt).abs()
                closest_idx = date_diffs.idxmin()
                return closest_idx

            return -1

        except Exception as e:
            logger.error(f"查找买点索引失败: {buypoint_date}, {e}")
            return -1

    def _check_pattern_match(self, stock_data: pd.DataFrame, buypoint_idx: int,
                           pattern) -> Dict[str, Any]:
        """检查单个技术模式是否匹配"""
        try:
            # 获取指标值（这里需要根据实际的指标系统来实现）
            indicator_value = self._get_indicator_value(
                stock_data, buypoint_idx, pattern.indicator_name
            )

            if indicator_value is None:
                return {
                    'is_match': False,
                    'indicator_value': None,
                    'expected_range': pattern.threshold_value,
                    'condition_type': pattern.condition_type,
                    'error': f'无法获取指标 {pattern.indicator_name} 的值'
                }

            # 检查条件是否满足
            is_match = self._evaluate_condition(
                indicator_value, pattern.condition_type, pattern.threshold_value
            )

            return {
                'is_match': is_match,
                'indicator_value': indicator_value,
                'expected_range': pattern.threshold_value,
                'condition_type': pattern.condition_type,
                'confidence': pattern.confidence,
                'error': None
            }

        except Exception as e:
            return {
                'is_match': False,
                'indicator_value': None,
                'expected_range': pattern.threshold_value,
                'condition_type': pattern.condition_type,
                'error': str(e)
            }

    def _get_indicator_value(self, stock_data: pd.DataFrame, index: int,
                           indicator_name: str) -> Optional[float]:
        """获取指标值（简化实现）"""
        try:
            # 基础价格指标
            if indicator_name == 'price':
                return stock_data.iloc[index]['close']
            elif indicator_name == 'volume':
                return stock_data.iloc[index]['volume']

            # 移动平均线
            if 'MA' in indicator_name:
                period = int(indicator_name.replace('MA', '').replace('_', ''))
                if index >= period - 1:
                    ma_values = stock_data.iloc[index-period+1:index+1]['close'].mean()
                    return ma_values

            # RSI (简化计算)
            if 'RSI' in indicator_name:
                if index >= 14:
                    recent_data = stock_data.iloc[index-13:index+1]['close']
                    deltas = recent_data.diff().dropna()
                    gains = deltas.where(deltas > 0, 0)
                    losses = -deltas.where(deltas < 0, 0)
                    avg_gain = gains.rolling(14).mean().iloc[-1]
                    avg_loss = losses.rolling(14).mean().iloc[-1]
                    rs = avg_gain / avg_loss if avg_loss != 0 else 100
                    rsi = 100 - (100 / (1 + rs))
                    return rsi

            # 价格位置
            if indicator_name == 'price_position':
                if index >= 20:
                    recent_data = stock_data.iloc[index-19:index+1]
                    high_20 = recent_data['high'].max()
                    low_20 = recent_data['low'].min()
                    current_price = stock_data.iloc[index]['close']
                    if high_20 > low_20:
                        return (current_price - low_20) / (high_20 - low_20)

            # 其他指标的占位符实现
            logger.debug(f"指标 {indicator_name} 暂未实现具体计算")
            return None

        except Exception as e:
            logger.error(f"计算指标 {indicator_name} 失败: {e}")
            return None

    def _evaluate_condition(self, value: float, condition_type: str,
                          threshold_value: Union[float, Tuple[float, float]]) -> bool:
        """评估条件是否满足"""
        try:
            if condition_type == '>=':
                return value >= threshold_value
            elif condition_type == '<=':
                return value <= threshold_value
            elif condition_type == '>':
                return value > threshold_value
            elif condition_type == '<':
                return value < threshold_value
            elif condition_type == 'between':
                if isinstance(threshold_value, (tuple, list)) and len(threshold_value) == 2:
                    low, high = threshold_value
                    return low <= value <= high

            return False

        except Exception as e:
            logger.error(f"评估条件失败: {e}")
            return False

    def _calculate_validation_metrics(self, total_buypoints: int, matched_buypoints: int,
                                    pattern_details: List[Dict[str, Any]]) -> ValidationMetrics:
        """计算验证指标"""
        try:
            if total_buypoints == 0:
                return ValidationMetrics(0, 0, 0, 0, 1.0, 0, (0, 0), 0)

            # 基础指标
            accuracy = matched_buypoints / total_buypoints
            precision = accuracy  # 在这个场景下，精确率等于准确率
            recall = accuracy     # 召回率也等于准确率

            # F1分数
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0

            # 假阳性率
            false_positive_rate = 1.0 - accuracy

            # 统计显著性检验 (使用二项式检验)
            if total_buypoints >= 10:
                # 零假设：匹配率 = 0.5 (随机)
                p_value = stats.binom_test(matched_buypoints, total_buypoints, 0.5)
                statistical_significance = 1 - p_value
            else:
                statistical_significance = 0.0

            # 置信区间 (95%置信区间)
            if total_buypoints >= 5:
                confidence_interval = stats.binom.interval(
                    0.95, total_buypoints, accuracy
                )
                confidence_interval = (
                    confidence_interval[0] / total_buypoints,
                    confidence_interval[1] / total_buypoints
                )
            else:
                confidence_interval = (0, 1)

            # 覆盖率 (有多少买点成功进行了验证)
            valid_validations = sum(1 for detail in pattern_details if detail.get('error') is None)
            coverage_rate = valid_validations / total_buypoints if total_buypoints > 0 else 0

            return ValidationMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                false_positive_rate=false_positive_rate,
                statistical_significance=statistical_significance,
                confidence_interval=confidence_interval,
                coverage_rate=coverage_rate
            )

        except Exception as e:
            logger.error(f"计算验证指标失败: {e}")
            return ValidationMetrics(0, 0, 0, 0, 1.0, 0, (0, 0), 0)

    def _generate_performance_indicators(self, pattern_details: List[Dict[str, Any]],
                                       metrics: ValidationMetrics) -> Dict[str, Any]:
        """生成性能指标"""
        try:
            # 模式匹配统计
            pattern_success_rates = {}
            consistency_scores = [detail.get('consistency_score', 0) for detail in pattern_details]

            # 分析各个模式的成功率
            for detail in pattern_details:
                pattern_matches = detail.get('pattern_matches', {})
                for pattern_name, match_info in pattern_matches.items():
                    if pattern_name not in pattern_success_rates:
                        pattern_success_rates[pattern_name] = []
                    pattern_success_rates[pattern_name].append(match_info.get('is_match', False))

            # 计算每个模式的成功率
            pattern_stats = {}
            for pattern_name, matches in pattern_success_rates.items():
                success_rate = sum(matches) / len(matches) if matches else 0
                pattern_stats[pattern_name] = {
                    'success_rate': success_rate,
                    'sample_count': len(matches)
                }

            return {
                'pattern_statistics': pattern_stats,
                'average_consistency_score': np.mean(consistency_scores) if consistency_scores else 0,
                'consistency_std': np.std(consistency_scores) if consistency_scores else 0,
                'validation_quality_score': metrics.accuracy * metrics.coverage_rate,
                'reliability_index': metrics.statistical_significance * metrics.f1_score,
                'error_rate': sum(1 for detail in pattern_details if detail.get('error')) / len(pattern_details) if pattern_details else 0
            }

        except Exception as e:
            logger.error(f"生成性能指标失败: {e}")
            return {}

    def _generate_forward_recommendations(self, metrics: ValidationMetrics,
                                        consistency_score: float,
                                        performance_indicators: Dict[str, Any]) -> List[str]:
        """生成前向验证建议"""
        recommendations = []

        try:
            # 准确率建议
            if metrics.accuracy < 0.7:
                recommendations.append("策略准确率偏低，建议调整技术模式的阈值参数")
            elif metrics.accuracy >= 0.9:
                recommendations.append("策略准确率优秀，可以考虑在实盘中验证")

            # 假阳性率建议
            if metrics.false_positive_rate > 0.1:
                recommendations.append("假阳性率过高，建议增加更严格的过滤条件")

            # 一致性建议
            if consistency_score < 0.8:
                recommendations.append("信号一致性偏低，建议优化模式组合")

            # 覆盖率建议
            if metrics.coverage_rate < 0.95:
                recommendations.append("验证覆盖率不足，建议检查数据完整性")

            # 统计显著性建议
            if metrics.statistical_significance < 0.95:
                recommendations.append("统计显著性不足，需要更多样本进行验证")

            # 根据模式统计给出具体建议
            pattern_stats = performance_indicators.get('pattern_statistics', {})
            for pattern_name, stats in pattern_stats.items():
                if stats['success_rate'] < 0.5:
                    recommendations.append(f"技术模式 {pattern_name} 效果不佳，建议调整或移除")

            if not recommendations:
                recommendations.append("策略验证表现良好，可以进入下一阶段测试")

        except Exception as e:
            logger.error(f"生成前向验证建议失败: {e}")
            recommendations.append("建议生成过程出现错误，请检查验证结果")

        return recommendations


class BackwardValidator:
    """后向验证器：买点 → 策略验证"""

    def __init__(self, data_manager: DataAccessInterface):
        """初始化后向验证器"""
        self.data_manager = data_manager
        self.confidence_threshold = 0.95

    @exception_handler(reraise=False)
    @performance_monitor(threshold=45.0)
    def validate_buypoints_against_strategy(self,
                                          selected_stocks: List[Dict[str, Any]],
                                          strategy,
                                          historical_days: int = 252) -> BackwardValidationResult:
        """
        验证选出的股票是否具备买点特征

        Args:
            selected_stocks: 策略选出的股票列表
            strategy: 生成的策略
            historical_days: 历史回放天数

        Returns:
            BackwardValidationResult: 后向验证结果
        """
        logger.info(f"开始后向验证：{len(selected_stocks)}只股票 ← 策略'{strategy.strategy_name}'")

        validation_start_time = datetime.now()

        try:
            # 历史数据回放分析
            historical_replay_results = self._perform_historical_replay(
                selected_stocks, strategy, historical_days
            )

            # 买点模式分析
            buypoint_pattern_analysis = self._analyze_buypoint_patterns(
                selected_stocks, strategy
            )

            # 统计检验
            statistical_tests = self._perform_statistical_tests(
                historical_replay_results, buypoint_pattern_analysis
            )

            # 风险评估
            risk_assessment = self._assess_investment_risks(
                selected_stocks, historical_replay_results
            )

            # 计算策略覆盖率
            strategy_coverage = self._calculate_strategy_coverage(
                buypoint_pattern_analysis, strategy
            )

            # 生成建议
            recommendations = self._generate_backward_recommendations(
                strategy_coverage, statistical_tests, risk_assessment
            )

            result = BackwardValidationResult(
                buypoint_count=len(selected_stocks),
                validation_time=validation_start_time,
                strategy_coverage=strategy_coverage,
                historical_replay_results=historical_replay_results,
                statistical_tests=statistical_tests,
                buypoint_pattern_analysis=buypoint_pattern_analysis,
                risk_assessment=risk_assessment,
                recommendations=recommendations
            )

            logger.info(f"后向验证完成: 策略覆盖率 {strategy_coverage:.2%}")
            return result

        except Exception as e:
            logger.error(f"后向验证失败: {e}")
            return BackwardValidationResult(
                buypoint_count=len(selected_stocks),
                validation_time=validation_start_time,
                strategy_coverage=0.0,
                historical_replay_results={},
                statistical_tests={'error': str(e)},
                buypoint_pattern_analysis=[],
                risk_assessment={'error': str(e)},
                recommendations=[f"验证过程出现错误: {str(e)}"]
            )

    def _perform_historical_replay(self, selected_stocks: List[Dict[str, Any]],
                                 strategy, historical_days: int) -> Dict[str, Any]:
        """执行历史数据回放"""
        try:
            replay_results = {
                'total_stocks_analyzed': len(selected_stocks),
                'successful_replays': 0,
                'average_pattern_match_rate': 0.0,
                'historical_performance': {},
                'pattern_stability': {}
            }

            pattern_match_rates = []
            historical_returns = []

            for stock in selected_stocks:
                stock_code = stock.get('stock_code', '')
                if not stock_code:
                    continue

                # 获取历史数据
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=historical_days)).strftime('%Y-%m-%d')

                stock_data = self.data_manager.get_stock_data_data_access_manager(
                    stock_code, start_date, end_date
                )

                if stock_data.empty:
                    continue

                # 分析该股票在历史期间的模式匹配情况
                stock_analysis = self._analyze_stock_historical_patterns(
                    stock_data, strategy, stock_code
                )

                if stock_analysis:
                    replay_results['successful_replays'] += 1
                    pattern_match_rates.append(stock_analysis['pattern_match_rate'])
                    historical_returns.extend(stock_analysis['historical_returns'])

                    # 存储个股分析结果
                    replay_results['historical_performance'][stock_code] = stock_analysis

            # 计算整体指标
            if pattern_match_rates:
                replay_results['average_pattern_match_rate'] = np.mean(pattern_match_rates)

            if historical_returns:
                replay_results['average_historical_return'] = np.mean(historical_returns)
                replay_results['historical_return_std'] = np.std(historical_returns)
                replay_results['positive_return_rate'] = sum(1 for r in historical_returns if r > 0) / len(historical_returns)

            return replay_results

        except Exception as e:
            logger.error(f"历史回放分析失败: {e}")
            return {'error': str(e)}

    def _analyze_stock_historical_patterns(self, stock_data: pd.DataFrame,
                                         strategy, stock_code: str) -> Optional[Dict[str, Any]]:
        """分析单只股票的历史模式"""
        try:
            if len(stock_data) < 50:  # 至少需要50个交易日
                return None

            pattern_matches = []
            historical_returns = []

            # 滑动窗口分析
            window_size = 20
            for i in range(window_size, len(stock_data) - 5):  # 留5天计算收益
                # 检查当前时点的模式匹配
                current_matches = 0
                total_patterns = len(strategy.technical_patterns)

                for pattern in strategy.technical_patterns:
                    # 简化的模式匹配检查
                    if self._check_historical_pattern_match(stock_data, i, pattern):
                        current_matches += 1

                match_rate = current_matches / total_patterns if total_patterns > 0 else 0
                pattern_matches.append(match_rate)

                # 如果匹配率高，计算未来收益
                if match_rate >= 0.7:  # 70%以上模式匹配
                    current_price = stock_data.iloc[i]['close']
                    future_price = stock_data.iloc[i + 5]['close']  # 5天后价格
                    return_pct = (future_price - current_price) / current_price * 100
                    historical_returns.append(return_pct)

            return {
                'stock_code': stock_code,
                'pattern_match_rate': np.mean(pattern_matches) if pattern_matches else 0,
                'pattern_stability': np.std(pattern_matches) if pattern_matches else 0,
                'historical_returns': historical_returns,
                'total_signals': len([r for r in pattern_matches if r >= 0.7]),
                'data_quality': len(stock_data)
            }

        except Exception as e:
            logger.error(f"分析股票历史模式失败 {stock_code}: {e}")
            return None

    def _check_historical_pattern_match(self, stock_data: pd.DataFrame, index: int,
                                      pattern) -> bool:
        """检查历史时点的模式匹配（简化版）"""
        try:
            # 这里使用与前向验证器相同的逻辑
            indicator_value = self._get_historical_indicator_value(
                stock_data, index, pattern.indicator_name
            )

            if indicator_value is None:
                return False

            # 评估条件
            if pattern.condition_type == '>=':
                return indicator_value >= pattern.threshold_value
            elif pattern.condition_type == '<=':
                return indicator_value <= pattern.threshold_value
            elif pattern.condition_type == 'between':
                if isinstance(pattern.threshold_value, (tuple, list)) and len(pattern.threshold_value) == 2:
                    low, high = pattern.threshold_value
                    return low <= indicator_value <= high

            return False

        except Exception as e:
            logger.debug(f"检查历史模式匹配失败: {e}")
            return False

    def _get_historical_indicator_value(self, stock_data: pd.DataFrame, index: int,
                                      indicator_name: str) -> Optional[float]:
        """获取历史指标值（复用前向验证器的逻辑）"""
        try:
            # 基础价格指标
            if indicator_name == 'price':
                return stock_data.iloc[index]['close']
            elif indicator_name == 'volume':
                return stock_data.iloc[index]['volume']

            # 移动平均线
            if 'MA' in indicator_name:
                try:
                    period = int(indicator_name.replace('MA', '').replace('_', ''))
                    if index >= period - 1:
                        ma_values = stock_data.iloc[index-period+1:index+1]['close'].mean()
                        return ma_values
                except:
                    pass

            # 价格位置
            if indicator_name == 'price_position':
                if index >= 20:
                    recent_data = stock_data.iloc[index-19:index+1]
                    high_20 = recent_data['high'].max()
                    low_20 = recent_data['low'].min()
                    current_price = stock_data.iloc[index]['close']
                    if high_20 > low_20:
                        return (current_price - low_20) / (high_20 - low_20)

            # 其他指标返回None
            return None

        except Exception as e:
            logger.debug(f"计算历史指标 {indicator_name} 失败: {e}")
            return None

    def _analyze_buypoint_patterns(self, selected_stocks: List[Dict[str, Any]],
                                 strategy) -> List[Dict[str, Any]]:
        """分析买点模式"""
        try:
            pattern_analysis = []

            for stock in selected_stocks:
                stock_code = stock.get('stock_code', '')
                if not stock_code:
                    continue

                # 获取最近的数据进行分析
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

                stock_data = self.data_manager.get_stock_data_data_access_manager(
                    stock_code, start_date, end_date
                )

                if stock_data.empty:
                    continue

                # 分析当前买点特征
                current_analysis = self._analyze_current_buypoint_features(
                    stock_data, strategy, stock_code
                )

                if current_analysis:
                    pattern_analysis.append(current_analysis)

            return pattern_analysis

        except Exception as e:
            logger.error(f"分析买点模式失败: {e}")
            return []

    def _analyze_current_buypoint_features(self, stock_data: pd.DataFrame,
                                         strategy, stock_code: str) -> Optional[Dict[str, Any]]:
        """分析当前买点特征"""
        try:
            if len(stock_data) < 20:
                return None

            # 使用最新的数据点
            latest_idx = len(stock_data) - 1

            # 检查各个模式的匹配情况
            pattern_scores = {}
            total_score = 0

            for pattern in strategy.technical_patterns:
                indicator_value = self._get_historical_indicator_value(
                    stock_data, latest_idx, pattern.indicator_name
                )

                if indicator_value is not None:
                    # 计算模式得分（基于置信度和匹配程度）
                    match_score = self._calculate_pattern_match_score(
                        indicator_value, pattern
                    )
                    pattern_scores[pattern.indicator_name] = {
                        'score': match_score,
                        'confidence': pattern.confidence,
                        'indicator_value': indicator_value,
                        'threshold': pattern.threshold_value
                    }
                    total_score += match_score * pattern.confidence

            # 计算综合买点质量分数
            avg_confidence = np.mean([p.confidence for p in strategy.technical_patterns])
            buypoint_quality = total_score / len(strategy.technical_patterns) if strategy.technical_patterns else 0

            return {
                'stock_code': stock_code,
                'buypoint_quality_score': buypoint_quality,
                'pattern_scores': pattern_scores,
                'recommendation_strength': buypoint_quality * avg_confidence,
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'data_points_analyzed': len(stock_data)
            }

        except Exception as e:
            logger.error(f"分析当前买点特征失败 {stock_code}: {e}")
            return None

    def _calculate_pattern_match_score(self, indicator_value: float, pattern) -> float:
        """计算模式匹配得分"""
        try:
            if pattern.condition_type == '>=':
                if indicator_value >= pattern.threshold_value:
                    return 1.0
                else:
                    # 渐进得分
                    ratio = indicator_value / pattern.threshold_value
                    return max(0, min(1, ratio))

            elif pattern.condition_type == '<=':
                if indicator_value <= pattern.threshold_value:
                    return 1.0
                else:
                    ratio = pattern.threshold_value / indicator_value
                    return max(0, min(1, ratio))

            elif pattern.condition_type == 'between':
                if isinstance(pattern.threshold_value, (tuple, list)) and len(pattern.threshold_value) == 2:
                    low, high = pattern.threshold_value
                    if low <= indicator_value <= high:
                        return 1.0
                    else:
                        # 计算距离最近边界的得分
                        if indicator_value < low:
                            distance = (low - indicator_value) / (high - low)
                        else:
                            distance = (indicator_value - high) / (high - low)
                        return max(0, 1 - distance)

            return 0.0

        except Exception as e:
            logger.debug(f"计算模式匹配得分失败: {e}")
            return 0.0

    def _perform_statistical_tests(self, historical_results: Dict[str, Any],
                                 pattern_analysis: List[Dict[str, Any]]) -> Dict[str, float]:
        """执行统计检验"""
        try:
            tests = {}

            # 检验1: 历史收益率的显著性
            historical_returns = historical_results.get('historical_performance', {})
            all_returns = []
            for stock_code, analysis in historical_returns.items():
                all_returns.extend(analysis.get('historical_returns', []))

            if len(all_returns) >= 10:
                # t检验：检验平均收益是否显著大于0
                t_stat, p_value = stats.ttest_1samp(all_returns, 0)
                tests['return_significance'] = 1 - p_value if t_stat > 0 else p_value
            else:
                tests['return_significance'] = 0.0

            # 检验2: 买点质量分数的一致性
            quality_scores = [analysis.get('buypoint_quality_score', 0)
                            for analysis in pattern_analysis if analysis]

            if len(quality_scores) >= 5:
                # 检验质量分数是否显著高于随机水平(0.5)
                if np.std(quality_scores) > 0:
                    t_stat, p_value = stats.ttest_1samp(quality_scores, 0.5)
                    tests['quality_consistency'] = 1 - p_value if t_stat > 0 else p_value
                else:
                    tests['quality_consistency'] = 1.0 if np.mean(quality_scores) > 0.5 else 0.0
            else:
                tests['quality_consistency'] = 0.0

            # 检验3: 模式匹配率的稳定性
            pattern_match_rates = []
            for stock_code, analysis in historical_results.get('historical_performance', {}).items():
                rate = analysis.get('pattern_match_rate', 0)
                if rate > 0:
                    pattern_match_rates.append(rate)

            if len(pattern_match_rates) >= 5:
                # 检验模式匹配率的变异系数
                mean_rate = np.mean(pattern_match_rates)
                std_rate = np.std(pattern_match_rates)
                cv = std_rate / mean_rate if mean_rate > 0 else 1.0
                tests['pattern_stability'] = max(0, 1 - cv)  # 变异系数越小，稳定性越好
            else:
                tests['pattern_stability'] = 0.0

            # 综合显著性得分
            test_values = [v for v in tests.values() if isinstance(v, float)]
            tests['overall_significance'] = np.mean(test_values) if test_values else 0.0

            return tests

        except Exception as e:
            logger.error(f"统计检验失败: {e}")
            return {'error': str(e)}

    def _assess_investment_risks(self, selected_stocks: List[Dict[str, Any]],
                               historical_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估投资风险"""
        try:
            risk_assessment = {
                'portfolio_size': len(selected_stocks),
                'diversification_risk': 'LOW',
                'volatility_risk': 'MEDIUM',
                'liquidity_risk': 'LOW',
                'strategy_risk': 'MEDIUM',
                'overall_risk_score': 0.5
            }

            # 组合规模风险
            if len(selected_stocks) < 5:
                risk_assessment['diversification_risk'] = 'HIGH'
                diversification_score = 0.3
            elif len(selected_stocks) < 15:
                risk_assessment['diversification_risk'] = 'MEDIUM'
                diversification_score = 0.6
            else:
                risk_assessment['diversification_risk'] = 'LOW'
                diversification_score = 0.9

            # 波动性风险评估
            historical_returns = []
            for stock_code, analysis in historical_results.get('historical_performance', {}).items():
                historical_returns.extend(analysis.get('historical_returns', []))

            if historical_returns:
                return_volatility = np.std(historical_returns)
                if return_volatility > 20:  # 高波动
                    risk_assessment['volatility_risk'] = 'HIGH'
                    volatility_score = 0.3
                elif return_volatility > 10:  # 中等波动
                    risk_assessment['volatility_risk'] = 'MEDIUM'
                    volatility_score = 0.6
                else:  # 低波动
                    risk_assessment['volatility_risk'] = 'LOW'
                    volatility_score = 0.9
            else:
                volatility_score = 0.5

            # 策略风险评估
            avg_pattern_match_rate = historical_results.get('average_pattern_match_rate', 0)
            if avg_pattern_match_rate > 0.8:
                risk_assessment['strategy_risk'] = 'LOW'
                strategy_score = 0.9
            elif avg_pattern_match_rate > 0.6:
                risk_assessment['strategy_risk'] = 'MEDIUM'
                strategy_score = 0.6
            else:
                risk_assessment['strategy_risk'] = 'HIGH'
                strategy_score = 0.3

            # 综合风险评分
            overall_score = (diversification_score * 0.3 +
                           volatility_score * 0.4 +
                           strategy_score * 0.3)
            risk_assessment['overall_risk_score'] = overall_score

            # 风险建议
            risk_suggestions = []
            if diversification_score < 0.6:
                risk_suggestions.append("建议增加持仓股票数量以降低集中风险")
            if volatility_score < 0.6:
                risk_suggestions.append("注意控制单笔投资金额，降低波动风险")
            if strategy_score < 0.6:
                risk_suggestions.append("策略有效性存疑，建议进一步验证")

            risk_assessment['risk_suggestions'] = risk_suggestions

            return risk_assessment

        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            return {'error': str(e)}

    def _calculate_strategy_coverage(self, pattern_analysis: List[Dict[str, Any]],
                                   strategy) -> float:
        """计算策略覆盖率"""
        try:
            if not pattern_analysis:
                return 0.0

            total_stocks = len(pattern_analysis)
            covered_stocks = 0

            for analysis in pattern_analysis:
                buypoint_quality = analysis.get('buypoint_quality_score', 0)
                if buypoint_quality >= 0.7:  # 质量阈值
                    covered_stocks += 1

            coverage_rate = covered_stocks / total_stocks if total_stocks > 0 else 0.0
            return coverage_rate

        except Exception as e:
            logger.error(f"计算策略覆盖率失败: {e}")
            return 0.0

    def _generate_backward_recommendations(self, strategy_coverage: float,
                                         statistical_tests: Dict[str, float],
                                         risk_assessment: Dict[str, Any]) -> List[str]:
        """生成后向验证建议"""
        recommendations = []

        try:
            # 策略覆盖率建议
            if strategy_coverage < 0.5:
                recommendations.append("策略覆盖率偏低，建议调整选股条件")
            elif strategy_coverage > 0.8:
                recommendations.append("策略覆盖率良好，选股质量较高")

            # 统计显著性建议
            overall_significance = statistical_tests.get('overall_significance', 0)
            if overall_significance < 0.8:
                recommendations.append("统计检验结果不够显著，建议扩大样本或优化策略")
            elif overall_significance > 0.95:
                recommendations.append("统计检验结果优秀，策略具有良好的统计学基础")

            # 风险评估建议
            overall_risk_score = risk_assessment.get('overall_risk_score', 0.5)
            if overall_risk_score < 0.5:
                recommendations.append("整体风险较高，建议谨慎操作并做好风控")
            elif overall_risk_score > 0.8:
                recommendations.append("风险控制良好，可以考虑适当增加仓位")

            # 添加风险建议
            risk_suggestions = risk_assessment.get('risk_suggestions', [])
            recommendations.extend(risk_suggestions)

            if not recommendations:
                recommendations.append("后向验证通过，策略表现符合预期")

        except Exception as e:
            logger.error(f"生成后向验证建议失败: {e}")
            recommendations.append("建议生成过程出现错误，请检查验证结果")

        return recommendations


class ValidationReportGenerator:
    """验证报告生成器"""

    def __init__(self):
        """初始化报告生成器"""
        self.report_template_path = None
        self.output_formats = ['json', 'html', 'txt']

    @exception_handler(reraise=False)
    @performance_monitor(threshold=10.0)
    def generate_bidirectional_report(self,
                                    forward_result: ForwardValidationResult,
                                    backward_result: BackwardValidationResult,
                                    strategy,
                                    output_dir: str = "./reports") -> BidirectionalValidationReport:
        """
        生成双向验证报告

        Args:
            forward_result: 前向验证结果
            backward_result: 后向验证结果
            strategy: 策略对象
            output_dir: 输出目录

        Returns:
            BidirectionalValidationReport: 双向验证报告
        """
        logger.info("开始生成双向验证报告...")

        report_start_time = datetime.now()
        report_id = f"validation_report_{report_start_time.strftime('%Y%m%d_%H%M%S')}"

        try:
            # 计算整体验证得分
            overall_validation_score = self._calculate_overall_validation_score(
                forward_result, backward_result
            )

            # 一致性分析
            consistency_analysis = self._perform_consistency_analysis(
                forward_result, backward_result
            )

            # 风险概况
            risk_profile = self._generate_risk_profile(
                forward_result, backward_result
            )

            # 性能摘要
            performance_summary = self._generate_performance_summary(
                forward_result, backward_result
            )

            # 改进建议
            improvement_suggestions = self._generate_improvement_suggestions(
                forward_result, backward_result, consistency_analysis
            )

            # 验证通过判定
            validation_passed = self._determine_validation_pass(
                overall_validation_score, forward_result, backward_result
            )

            # 计算执行时间
            execution_time = (datetime.now() - report_start_time).total_seconds()

            # 创建报告对象
            report = BidirectionalValidationReport(
                report_id=report_id,
                generation_time=report_start_time,
                forward_validation=forward_result,
                backward_validation=backward_result,
                overall_validation_score=overall_validation_score,
                consistency_analysis=consistency_analysis,
                risk_profile=risk_profile,
                performance_summary=performance_summary,
                improvement_suggestions=improvement_suggestions,
                validation_passed=validation_passed,
                execution_time=execution_time
            )

            # 保存报告到文件
            self._save_report_to_files(report, output_dir)

            logger.info(f"双向验证报告生成完成: {report_id}, "
                       f"整体得分: {overall_validation_score:.3f}, "
                       f"验证{'通过' if validation_passed else '失败'}")

            return report

        except Exception as e:
            logger.error(f"生成双向验证报告失败: {e}")
            # 返回错误报告
            return BidirectionalValidationReport(
                report_id=report_id,
                generation_time=report_start_time,
                forward_validation=forward_result,
                backward_validation=backward_result,
                overall_validation_score=0.0,
                consistency_analysis={'error': str(e)},
                risk_profile={'error': str(e)},
                performance_summary={'error': str(e)},
                improvement_suggestions=[f"报告生成失败: {str(e)}"],
                validation_passed=False,
                execution_time=(datetime.now() - report_start_time).total_seconds()
            )

    def _calculate_overall_validation_score(self, forward_result: ForwardValidationResult,
                                          backward_result: BackwardValidationResult) -> float:
        """计算整体验证得分"""
        try:
            # 前向验证得分 (权重0.6)
            forward_score = (
                forward_result.validation_metrics.accuracy * 0.4 +
                forward_result.validation_metrics.f1_score * 0.3 +
                forward_result.signal_consistency_score * 0.3
            )

            # 后向验证得分 (权重0.4)
            backward_score = (
                backward_result.strategy_coverage * 0.4 +
                backward_result.statistical_tests.get('overall_significance', 0) * 0.3 +
                backward_result.risk_assessment.get('overall_risk_score', 0.5) * 0.3
            )

            # 综合得分
            overall_score = forward_score * 0.6 + backward_score * 0.4

            return min(1.0, max(0.0, overall_score))

        except Exception as e:
            logger.error(f"计算整体验证得分失败: {e}")
            return 0.0

    def _perform_consistency_analysis(self, forward_result: ForwardValidationResult,
                                    backward_result: BackwardValidationResult) -> Dict[str, Any]:
        """执行一致性分析"""
        try:
            consistency = {
                'forward_backward_correlation': 0.0,
                'pattern_consistency': 0.0,
                'performance_consistency': 0.0,
                'risk_consistency': 0.0,
                'overall_consistency': 0.0
            }

            # 前后向结果相关性
            forward_accuracy = forward_result.validation_metrics.accuracy
            backward_coverage = backward_result.strategy_coverage
            consistency['forward_backward_correlation'] = min(forward_accuracy, backward_coverage)

            # 模式一致性
            forward_patterns = len(forward_result.pattern_match_details)
            backward_patterns = len(backward_result.buypoint_pattern_analysis)
            if forward_patterns > 0 and backward_patterns > 0:
                pattern_ratio = min(forward_patterns, backward_patterns) / max(forward_patterns, backward_patterns)
                consistency['pattern_consistency'] = pattern_ratio

            # 性能一致性
            forward_perf = forward_result.performance_indicators.get('validation_quality_score', 0)
            backward_sig = backward_result.statistical_tests.get('overall_significance', 0)
            consistency['performance_consistency'] = (forward_perf + backward_sig) / 2

            # 风险一致性
            forward_fpr = 1 - forward_result.validation_metrics.false_positive_rate
            backward_risk = backward_result.risk_assessment.get('overall_risk_score', 0.5)
            consistency['risk_consistency'] = (forward_fpr + backward_risk) / 2

            # 整体一致性
            consistency_values = [v for v in consistency.values() if isinstance(v, float) and v > 0]
            consistency['overall_consistency'] = np.mean(consistency_values) if consistency_values else 0.0

            return consistency

        except Exception as e:
            logger.error(f"一致性分析失败: {e}")
            return {'error': str(e)}

    def _generate_risk_profile(self, forward_result: ForwardValidationResult,
                             backward_result: BackwardValidationResult) -> Dict[str, Any]:
        """生成风险概况"""
        try:
            risk_profile = {
                'validation_risk': 'MEDIUM',
                'strategy_risk': 'MEDIUM',
                'implementation_risk': 'MEDIUM',
                'overall_risk': 'MEDIUM',
                'risk_factors': [],
                'mitigation_suggestions': []
            }

            # 验证风险评估
            fpr = forward_result.validation_metrics.false_positive_rate
            if fpr > 0.1:
                risk_profile['validation_risk'] = 'HIGH'
                risk_profile['risk_factors'].append('假阳性率过高')
                risk_profile['mitigation_suggestions'].append('增加验证样本，优化模式参数')
            elif fpr < 0.05:
                risk_profile['validation_risk'] = 'LOW'

            # 策略风险评估
            strategy_coverage = backward_result.strategy_coverage
            if strategy_coverage < 0.5:
                risk_profile['strategy_risk'] = 'HIGH'
                risk_profile['risk_factors'].append('策略覆盖率偏低')
                risk_profile['mitigation_suggestions'].append('重新审视策略设计逻辑')
            elif strategy_coverage > 0.8:
                risk_profile['strategy_risk'] = 'LOW'

            # 实施风险评估
            consistency_score = forward_result.signal_consistency_score
            if consistency_score < 0.7:
                risk_profile['implementation_risk'] = 'HIGH'
                risk_profile['risk_factors'].append('信号一致性不足')
                risk_profile['mitigation_suggestions'].append('加强策略实施的监控机制')
            elif consistency_score > 0.9:
                risk_profile['implementation_risk'] = 'LOW'

            # 整体风险
            risk_scores = []
            for risk_type in ['validation_risk', 'strategy_risk', 'implementation_risk']:
                if risk_profile[risk_type] == 'LOW':
                    risk_scores.append(0.2)
                elif risk_profile[risk_type] == 'MEDIUM':
                    risk_scores.append(0.5)
                else:  # HIGH
                    risk_scores.append(0.8)

            overall_risk_score = np.mean(risk_scores)
            if overall_risk_score < 0.4:
                risk_profile['overall_risk'] = 'LOW'
            elif overall_risk_score > 0.7:
                risk_profile['overall_risk'] = 'HIGH'

            return risk_profile

        except Exception as e:
            logger.error(f"生成风险概况失败: {e}")
            return {'error': str(e)}

    def _generate_performance_summary(self, forward_result: ForwardValidationResult,
                                    backward_result: BackwardValidationResult) -> Dict[str, Any]:
        """生成性能摘要"""
        try:
            summary = {
                'validation_metrics': {
                    'accuracy': forward_result.validation_metrics.accuracy,
                    'precision': forward_result.validation_metrics.precision,
                    'recall': forward_result.validation_metrics.recall,
                    'f1_score': forward_result.validation_metrics.f1_score,
                    'false_positive_rate': forward_result.validation_metrics.false_positive_rate,
                    'coverage_rate': forward_result.validation_metrics.coverage_rate
                },
                'strategy_performance': {
                    'strategy_coverage': backward_result.strategy_coverage,
                    'statistical_significance': backward_result.statistical_tests.get('overall_significance', 0),
                    'pattern_stability': backward_result.statistical_tests.get('pattern_stability', 0),
                    'risk_score': backward_result.risk_assessment.get('overall_risk_score', 0.5)
                },
                'consistency_metrics': {
                    'signal_consistency': forward_result.signal_consistency_score,
                    'pattern_match_rate': len([d for d in forward_result.pattern_match_details if d.get('is_matched', False)]) / len(forward_result.pattern_match_details) if forward_result.pattern_match_details else 0
                },
                'quality_indicators': {
                    'data_quality': forward_result.validation_metrics.coverage_rate,
                    'model_reliability': forward_result.validation_metrics.statistical_significance,
                    'strategy_robustness': backward_result.statistical_tests.get('pattern_stability', 0)
                }
            }

            return summary

        except Exception as e:
            logger.error(f"生成性能摘要失败: {e}")
            return {'error': str(e)}

    def _generate_improvement_suggestions(self, forward_result: ForwardValidationResult,
                                        backward_result: BackwardValidationResult,
                                        consistency_analysis: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        suggestions = []

        try:
            # 基于前向验证结果的建议
            if forward_result.validation_metrics.accuracy < 0.8:
                suggestions.append("前向验证准确率偏低，建议：1) 调整技术指标参数 2) 增加更多验证样本 3) 优化策略逻辑")

            if forward_result.validation_metrics.false_positive_rate > 0.1:
                suggestions.append("假阳性率过高，建议增加更严格的过滤条件和阈值设置")

            if forward_result.signal_consistency_score < 0.8:
                suggestions.append("信号一致性不足，建议重新评估技术模式的有效性和相关性")

            # 基于后向验证结果的建议
            if backward_result.strategy_coverage < 0.6:
                suggestions.append("策略覆盖率不足，建议：1) 扩大选股范围 2) 调整策略条件 3) 考虑多策略组合")

            overall_significance = backward_result.statistical_tests.get('overall_significance', 0)
            if overall_significance < 0.8:
                suggestions.append("统计显著性不足，建议扩大历史数据样本并进行更长期的回测验证")

            # 基于一致性分析的建议
            overall_consistency = consistency_analysis.get('overall_consistency', 0)
            if overall_consistency < 0.7:
                suggestions.append("前后向验证结果一致性偏低，建议检查策略设计的逻辑连贯性")

            # 通用建议
            if not suggestions:
                suggestions.append("验证结果总体良好，建议进入小规模实盘测试阶段")
            else:
                suggestions.append("建议在解决上述问题后重新进行验证测试")

        except Exception as e:
            logger.error(f"生成改进建议失败: {e}")
            suggestions.append("建议生成过程出现错误，请人工审查验证结果")

        return suggestions

    def _determine_validation_pass(self, overall_score: float,
                                 forward_result: ForwardValidationResult,
                                 backward_result: BackwardValidationResult) -> bool:
        """判定验证是否通过"""
        try:
            # 主要条件
            accuracy_pass = forward_result.validation_metrics.accuracy >= 0.75
            fpr_pass = forward_result.validation_metrics.false_positive_rate <= 0.1
            coverage_pass = backward_result.strategy_coverage >= 0.6
            consistency_pass = forward_result.signal_consistency_score >= 0.7
            overall_pass = overall_score >= 0.7

            # 所有条件都要满足
            validation_passed = all([
                accuracy_pass, fpr_pass, coverage_pass,
                consistency_pass, overall_pass
            ])

            return validation_passed

        except Exception as e:
            logger.error(f"判定验证通过失败: {e}")
            return False

    def _save_report_to_files(self, report: BidirectionalValidationReport,
                            output_dir: str):
        """保存报告到文件"""
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 保存JSON格式
            json_file = os.path.join(output_dir, f"{report.report_id}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=str)

            # 保存文本格式报告
            txt_file = os.path.join(output_dir, f"{report.report_id}.txt")
            self._generate_text_report(report, txt_file)

            logger.info(f"验证报告已保存: {json_file}, {txt_file}")

        except Exception as e:
            logger.error(f"保存报告文件失败: {e}")

    def _generate_text_report(self, report: BidirectionalValidationReport,
                            file_path: str):
        """生成文本格式报告"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("双向验证系统报告\n")
                f.write("="*80 + "\n\n")

                f.write(f"报告ID: {report.report_id}\n")
                f.write(f"生成时间: {report.generation_time}\n")
                f.write(f"执行时间: {report.execution_time:.2f}秒\n")
                f.write(f"验证结果: {'通过' if report.validation_passed else '失败'}\n")
                f.write(f"整体得分: {report.overall_validation_score:.3f}\n\n")

                f.write("前向验证结果\n")
                f.write("-"*40 + "\n")
                f.write(f"策略名称: {report.forward_validation.strategy_name}\n")
                f.write(f"匹配买点数: {report.forward_validation.matched_buypoints}/{report.forward_validation.total_patterns}\n")
                f.write(f"准确率: {report.forward_validation.validation_metrics.accuracy:.3f}\n")
                f.write(f"假阳性率: {report.forward_validation.validation_metrics.false_positive_rate:.3f}\n")
                f.write(f"信号一致性: {report.forward_validation.signal_consistency_score:.3f}\n\n")

                f.write("后向验证结果\n")
                f.write("-"*40 + "\n")
                f.write(f"验证股票数: {report.backward_validation.buypoint_count}\n")
                f.write(f"策略覆盖率: {report.backward_validation.strategy_coverage:.3f}\n")
                f.write(f"统计显著性: {report.backward_validation.statistical_tests.get('overall_significance', 0):.3f}\n\n")

                f.write("风险概况\n")
                f.write("-"*40 + "\n")
                f.write(f"整体风险等级: {report.risk_profile.get('overall_risk', 'UNKNOWN')}\n")
                risk_factors = report.risk_profile.get('risk_factors', [])
                if risk_factors:
                    f.write("风险因素:\n")
                    for factor in risk_factors:
                        f.write(f"  - {factor}\n")
                f.write("\n")

                f.write("改进建议\n")
                f.write("-"*40 + "\n")
                for i, suggestion in enumerate(report.improvement_suggestions, 1):
                    f.write(f"{i}. {suggestion}\n")

                f.write("\n" + "="*80 + "\n")
                f.write("报告结束\n")

        except Exception as e:
            logger.error(f"生成文本报告失败: {e}")


class BidirectionalValidationSystem:
    """双向验证系统主控制器"""

    def __init__(self, data_manager: Optional[DataAccessInterface] = None):
        """初始化双向验证系统"""
        if data_manager:
            self.data_manager = data_manager
        else:
            # 从容器获取数据管理器
            try:
                container = get_container()
                self.data_manager = container.resolve(DataAccessInterface)
            except:
                logger.warning("无法从容器获取DataAccessInterface，将使用模拟实现")
                self.data_manager = self._create_mock_data_manager()

        # 初始化验证组件
        self.forward_validator = ForwardValidator(self.data_manager)
        self.backward_validator = BackwardValidator(self.data_manager)
        self.report_generator = ValidationReportGenerator()

        # 系统统计
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'average_execution_time': 0.0
        }

    @exception_handler(reraise=False)
    @performance_monitor(threshold=60.0)
    def execute_bidirectional_validation(self,
                                       strategy,
                                       original_buypoints: List[Dict[str, Any]],
                                       selected_stocks: List[Dict[str, Any]],
                                       output_dir: str = "./reports") -> Dict[str, Any]:
        """
        执行完整的双向验证流程

        Args:
            strategy: 生成的策略对象
            original_buypoints: 原始买点数据
            selected_stocks: 策略选出的股票
            output_dir: 报告输出目录

        Returns:
            Dict: 验证结果摘要
        """
        logger.info("开始执行双向验证系统...")

        start_time = datetime.now()
        self.validation_stats['total_validations'] += 1

        try:
            # 步骤1: 前向验证 (策略 → 买点)
            logger.info("执行前向验证...")
            forward_result = self.forward_validator.validate_strategy_against_buypoints(
                strategy, original_buypoints
            )

            # 步骤2: 后向验证 (买点 → 策略)
            logger.info("执行后向验证...")
            backward_result = self.backward_validator.validate_buypoints_against_strategy(
                selected_stocks, strategy
            )

            # 步骤3: 生成综合报告
            logger.info("生成双向验证报告...")
            validation_report = self.report_generator.generate_bidirectional_report(
                forward_result, backward_result, strategy, output_dir
            )

            # 更新统计信息
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_validation_stats(execution_time, validation_report.validation_passed)

            # 构建返回结果
            result = {
                'status': 'success' if validation_report.validation_passed else 'warning',
                'message': f'双向验证{"通过" if validation_report.validation_passed else "未通过"}',
                'details': {
                    'report_id': validation_report.report_id,
                    'overall_score': validation_report.overall_validation_score,
                    'forward_validation': {
                        'accuracy': forward_result.validation_metrics.accuracy,
                        'matched_buypoints': forward_result.matched_buypoints,
                        'total_patterns': forward_result.total_patterns,
                        'signal_consistency': forward_result.signal_consistency_score
                    },
                    'backward_validation': {
                        'strategy_coverage': backward_result.strategy_coverage,
                        'analyzed_stocks': backward_result.buypoint_count,
                        'statistical_significance': backward_result.statistical_tests.get('overall_significance', 0)
                    },
                    'risk_assessment': validation_report.risk_profile.get('overall_risk', 'MEDIUM'),
                    'execution_time': execution_time
                },
                'recommendations': validation_report.improvement_suggestions,
                'report_files': {
                    'json': f"{output_dir}/{validation_report.report_id}.json",
                    'text': f"{output_dir}/{validation_report.report_id}.txt"
                }
            }

            logger.info(f"双向验证完成: 整体得分 {validation_report.overall_validation_score:.3f}, "
                       f"验证{'通过' if validation_report.validation_passed else '失败'}")

            return result

        except Exception as e:
            logger.error(f"双向验证执行失败: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                'status': 'failed',
                'message': f'双向验证失败: {str(e)}',
                'details': {
                    'error': str(e),
                    'execution_time': execution_time
                },
                'recommendations': ['系统错误，建议检查数据源和网络连接']
            }

    def _update_validation_stats(self, execution_time: float, success: bool):
        """更新验证统计"""
        try:
            if success:
                self.validation_stats['successful_validations'] += 1

            # 更新平均执行时间
            current_avg = self.validation_stats['average_execution_time']
            total_count = self.validation_stats['total_validations']

            if total_count > 1:
                new_avg = ((current_avg * (total_count - 1)) + execution_time) / total_count
                self.validation_stats['average_execution_time'] = new_avg
            else:
                self.validation_stats['average_execution_time'] = execution_time

        except Exception as e:
            logger.error(f"更新验证统计失败: {e}")

    def get_validation_statistics(self) -> Dict[str, Any]:
        """获取验证系统统计信息"""
        return {
            'total_validations': self.validation_stats['total_validations'],
            'successful_validations': self.validation_stats['successful_validations'],
            'success_rate': self.validation_stats['successful_validations'] / max(1, self.validation_stats['total_validations']),
            'average_execution_time': self.validation_stats['average_execution_time']
        }

    def _create_mock_data_manager(self):
        """创建模拟数据管理器（用于测试）"""
        class MockDataManager:
            def get_stock_data_data_access_manager(self, code, start_date, end_date):
                # 返回模拟数据
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                data = pd.DataFrame({
                    'date': dates.strftime('%Y-%m-%d'),
                    'open': 10.0,
                    'high': 11.0,
                    'low': 9.0,
                    'close': 10.5,
                    'volume': 1000000
                })
                return data

        return MockDataManager()


# 导出主要类和函数
__all__ = [
    'BidirectionalValidationSystem',
    'ForwardValidator',
    'BackwardValidator',
    'ValidationReportGenerator',
    'BidirectionalValidationReport',
    'ForwardValidationResult',
    'BackwardValidationResult',
    'ValidationMetrics'
]