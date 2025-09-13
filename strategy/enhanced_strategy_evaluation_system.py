"""
增强策略评估系统

提供多维度策略评估，包括历史回测、实时评估、风险分析等
遵循六层架构规范，实现全面的策略评估能力
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

from utils.dependency_injection import get_logger, get_service
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container
from db.interfaces.data_access_interface import DataAccessInterface
from strategy.enhanced_strategy_config_engine import StrategyConfig
from strategy.enhanced_stock_selection_engine import EnhancedStockSelectionEngine

logger = get_logger(__name__)


@dataclass
class StrategyPerformanceMetrics:
    """策略性能指标"""
    strategy_id: str
    evaluation_period: str
    total_selections: int
    successful_selections: int
    win_rate: float
    average_return: float
    max_return: float
    min_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    alpha: float
    beta: float
    information_ratio: float


@dataclass
class RiskMetrics:
    """风险指标"""
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    expected_shortfall: float
    maximum_drawdown: float
    downside_deviation: float
    calmar_ratio: float
    sortino_ratio: float


@dataclass
class StrategyEvaluationResult:
    """策略评估结果"""
    strategy_id: str
    strategy_name: str
    evaluation_date: str
    performance_metrics: StrategyPerformanceMetrics
    risk_metrics: RiskMetrics
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]
    overall_score: float
    grade: str


class EnhancedStrategyEvaluationSystem:
    """
    增强策略评估系统
    
    提供全面的策略评估和分析功能
    """
    
    def __init__(self):
        """初始化策略评估系统"""
        self.logger = logger
        self.data_access = get_service(DataAccessInterface)
        self.selection_engine = EnhancedStockSelectionEngine()
        
        # 评估配置
        self.evaluation_config = {
            'min_evaluation_period': 30,  # 最小评估期（天）
            'benchmark_return': 0.08,     # 基准收益率
            'risk_free_rate': 0.03,       # 无风险利率
            'confidence_levels': [0.95, 0.99],  # 置信水平
            'evaluation_weights': {
                'performance': 0.4,
                'risk': 0.3,
                'stability': 0.2,
                'consistency': 0.1
            }
        }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def evaluate_strategy(self, strategy_config: StrategyConfig,
                         evaluation_period: Optional[Tuple[str, str]] = None,
                         benchmark_data: Optional[pd.DataFrame] = None) -> StrategyEvaluationResult:
        """
        全面评估策略
        
        Args:
            strategy_config: 策略配置
            evaluation_period: 评估期间 (start_date, end_date)
            benchmark_data: 基准数据
            
        Returns:
            StrategyEvaluationResult: 评估结果
        """
        self.logger.info(f"开始评估策略: {strategy_config.name}")
        
        # 设置评估期间
        if evaluation_period is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # 默认3个月
            evaluation_period = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # 获取历史选股结果
        historical_results = self._get_historical_selection_results(strategy_config, evaluation_period)
        
        # 计算性能指标
        performance_metrics = self._calculate_performance_metrics(
            strategy_config, historical_results, evaluation_period
        )
        
        # 计算风险指标
        risk_metrics = self._calculate_risk_metrics(historical_results, benchmark_data)
        
        # 详细分析
        detailed_analysis = self._perform_detailed_analysis(
            strategy_config, historical_results, performance_metrics, risk_metrics
        )
        
        # 生成建议
        recommendations = self._generate_recommendations(performance_metrics, risk_metrics, detailed_analysis)
        
        # 计算综合评分
        overall_score = self._calculate_overall_score(performance_metrics, risk_metrics, detailed_analysis)
        
        # 确定评级
        grade = self._determine_grade(overall_score)
        
        result = StrategyEvaluationResult(
            strategy_id=strategy_config.strategy_id,
            strategy_name=strategy_config.name,
            evaluation_date=datetime.now().isoformat(),
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations,
            overall_score=overall_score,
            grade=grade
        )
        
        self.logger.info(f"策略评估完成，综合评分: {overall_score:.2f}, 评级: {grade}")
        return result
    
    def _get_historical_selection_results(self, strategy_config: StrategyConfig,
                                        evaluation_period: Tuple[str, str]) -> List[Dict[str, Any]]:
        """获取历史选股结果"""
        try:
            # 模拟历史选股结果
            # 实际实现中应该从数据库或历史记录中获取
            results = []
            
            start_date = datetime.strptime(evaluation_period[0], '%Y-%m-%d')
            end_date = datetime.strptime(evaluation_period[1], '%Y-%m-%d')
            
            # 每周执行一次选股
            current_date = start_date
            while current_date <= end_date:
                # 模拟选股结果
                selection_result = {
                    'date': current_date.strftime('%Y-%m-%d'),
                    'selected_stocks': [
                        {'stock_code': '000001', 'score': 85.5, 'return_1w': 0.05, 'return_1m': 0.12},
                        {'stock_code': '000002', 'score': 78.2, 'return_1w': 0.03, 'return_1m': 0.08},
                        {'stock_code': '600519', 'score': 92.1, 'return_1w': 0.08, 'return_1m': 0.15}
                    ]
                }
                results.append(selection_result)
                current_date += timedelta(days=7)
            
            return results
            
        except Exception as e:
            self.logger.error(f"获取历史选股结果时出错: {e}")
            return []
    
    def _calculate_performance_metrics(self, strategy_config: StrategyConfig,
                                     historical_results: List[Dict[str, Any]],
                                     evaluation_period: Tuple[str, str]) -> StrategyPerformanceMetrics:
        """计算性能指标"""
        try:
            if not historical_results:
                return self._create_empty_performance_metrics(strategy_config.strategy_id, evaluation_period)
            
            # 统计基本指标
            total_selections = sum(len(result['selected_stocks']) for result in historical_results)
            
            # 计算收益率
            all_returns = []
            successful_count = 0
            
            for result in historical_results:
                for stock in result['selected_stocks']:
                    return_1m = stock.get('return_1m', 0.0)
                    all_returns.append(return_1m)
                    if return_1m > 0:
                        successful_count += 1
            
            if not all_returns:
                return self._create_empty_performance_metrics(strategy_config.strategy_id, evaluation_period)
            
            # 计算统计指标
            returns_array = np.array(all_returns)
            win_rate = successful_count / len(all_returns) if all_returns else 0.0
            average_return = np.mean(returns_array)
            max_return = np.max(returns_array)
            min_return = np.min(returns_array)
            volatility = np.std(returns_array)
            
            # 计算风险调整指标
            sharpe_ratio = (average_return - self.evaluation_config['risk_free_rate']) / volatility if volatility > 0 else 0.0
            
            # 计算最大回撤
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
            
            # Alpha和Beta（简化计算）
            alpha = average_return - self.evaluation_config['benchmark_return']
            beta = 1.0  # 简化假设
            
            # 信息比率
            excess_return = average_return - self.evaluation_config['benchmark_return']
            tracking_error = volatility  # 简化
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0.0
            
            return StrategyPerformanceMetrics(
                strategy_id=strategy_config.strategy_id,
                evaluation_period=f"{evaluation_period[0]} to {evaluation_period[1]}",
                total_selections=total_selections,
                successful_selections=successful_count,
                win_rate=win_rate,
                average_return=average_return,
                max_return=max_return,
                min_return=min_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=abs(max_drawdown),
                volatility=volatility,
                alpha=alpha,
                beta=beta,
                information_ratio=information_ratio
            )
            
        except Exception as e:
            self.logger.error(f"计算性能指标时出错: {e}")
            return self._create_empty_performance_metrics(strategy_config.strategy_id, evaluation_period)
    
    def _calculate_risk_metrics(self, historical_results: List[Dict[str, Any]],
                              benchmark_data: Optional[pd.DataFrame] = None) -> RiskMetrics:
        """计算风险指标"""
        try:
            # 提取收益率数据
            all_returns = []
            for result in historical_results:
                for stock in result['selected_stocks']:
                    all_returns.append(stock.get('return_1m', 0.0))
            
            if not all_returns:
                return RiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            returns_array = np.array(all_returns)
            
            # 计算VaR
            var_95 = np.percentile(returns_array, 5)  # 95% VaR
            var_99 = np.percentile(returns_array, 1)  # 99% VaR
            
            # 期望损失（ES）
            expected_shortfall = np.mean(returns_array[returns_array <= var_95])
            
            # 最大回撤
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            maximum_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            
            # 下行偏差
            negative_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0.0
            
            # Calmar比率
            average_return = np.mean(returns_array)
            calmar_ratio = average_return / maximum_drawdown if maximum_drawdown > 0 else 0.0
            
            # Sortino比率
            sortino_ratio = (average_return - self.evaluation_config['risk_free_rate']) / downside_deviation if downside_deviation > 0 else 0.0
            
            return RiskMetrics(
                var_95=abs(var_95),
                var_99=abs(var_99),
                expected_shortfall=abs(expected_shortfall),
                maximum_drawdown=maximum_drawdown,
                downside_deviation=downside_deviation,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio
            )
            
        except Exception as e:
            self.logger.error(f"计算风险指标时出错: {e}")
            return RiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _create_empty_performance_metrics(self, strategy_id: str, evaluation_period: Tuple[str, str]) -> StrategyPerformanceMetrics:
        """创建空的性能指标"""
        return StrategyPerformanceMetrics(
            strategy_id=strategy_id,
            evaluation_period=f"{evaluation_period[0]} to {evaluation_period[1]}",
            total_selections=0,
            successful_selections=0,
            win_rate=0.0,
            average_return=0.0,
            max_return=0.0,
            min_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            alpha=0.0,
            beta=0.0,
            information_ratio=0.0
        )

    def _perform_detailed_analysis(self, strategy_config: StrategyConfig,
                                 historical_results: List[Dict[str, Any]],
                                 performance_metrics: StrategyPerformanceMetrics,
                                 risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """执行详细分析"""
        try:
            analysis = {
                'strategy_characteristics': self._analyze_strategy_characteristics(strategy_config),
                'performance_analysis': self._analyze_performance_trends(historical_results),
                'risk_analysis': self._analyze_risk_profile(risk_metrics),
                'consistency_analysis': self._analyze_consistency(historical_results),
                'market_adaptation': self._analyze_market_adaptation(historical_results),
                'optimization_suggestions': self._generate_optimization_suggestions(
                    performance_metrics, risk_metrics
                )
            }

            return analysis

        except Exception as e:
            self.logger.error(f"执行详细分析时出错: {e}")
            return {}

    def _analyze_strategy_characteristics(self, strategy_config: StrategyConfig) -> Dict[str, Any]:
        """分析策略特征"""
        characteristics = {
            'rule_count': len(strategy_config.rules),
            'complexity_score': 0.0,
            'indicator_diversity': set(),
            'period_coverage': set(),
            'condition_types': []
        }

        total_conditions = 0
        for rule in strategy_config.rules:
            total_conditions += len(rule.conditions)
            for condition in rule.conditions:
                characteristics['indicator_diversity'].add(condition.indicator)
                characteristics['period_coverage'].add(condition.period)
                characteristics['condition_types'].append(condition.pattern)

        # 计算复杂度分数
        characteristics['complexity_score'] = min(100.0, total_conditions * 10)
        characteristics['indicator_diversity'] = len(characteristics['indicator_diversity'])
        characteristics['period_coverage'] = len(characteristics['period_coverage'])
        characteristics['total_conditions'] = total_conditions

        return characteristics

    def _analyze_performance_trends(self, historical_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析性能趋势"""
        if not historical_results:
            return {'trend': 'insufficient_data', 'stability': 0.0}

        # 计算每期收益率
        period_returns = []
        for result in historical_results:
            if result['selected_stocks']:
                period_return = np.mean([stock.get('return_1m', 0.0) for stock in result['selected_stocks']])
                period_returns.append(period_return)

        if len(period_returns) < 2:
            return {'trend': 'insufficient_data', 'stability': 0.0}

        # 趋势分析
        x = np.arange(len(period_returns))
        trend_slope = np.polyfit(x, period_returns, 1)[0]

        # 稳定性分析
        stability = 1.0 / (1.0 + np.std(period_returns)) if period_returns else 0.0

        return {
            'trend': 'improving' if trend_slope > 0 else 'declining',
            'trend_slope': trend_slope,
            'stability': stability,
            'period_count': len(period_returns),
            'best_period': max(period_returns) if period_returns else 0.0,
            'worst_period': min(period_returns) if period_returns else 0.0
        }

    def _analyze_risk_profile(self, risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """分析风险特征"""
        risk_level = 'low'
        if risk_metrics.maximum_drawdown > 0.2:
            risk_level = 'high'
        elif risk_metrics.maximum_drawdown > 0.1:
            risk_level = 'medium'

        return {
            'risk_level': risk_level,
            'risk_score': min(100.0, risk_metrics.maximum_drawdown * 500),
            'var_analysis': {
                'var_95_level': 'high' if risk_metrics.var_95 > 0.05 else 'acceptable',
                'var_99_level': 'high' if risk_metrics.var_99 > 0.1 else 'acceptable'
            },
            'risk_adjusted_performance': {
                'calmar_rating': 'excellent' if risk_metrics.calmar_ratio > 2.0 else 'good' if risk_metrics.calmar_ratio > 1.0 else 'poor',
                'sortino_rating': 'excellent' if risk_metrics.sortino_ratio > 2.0 else 'good' if risk_metrics.sortino_ratio > 1.0 else 'poor'
            }
        }

    def _analyze_consistency(self, historical_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析一致性"""
        if not historical_results:
            return {'consistency_score': 0.0, 'volatility_level': 'unknown'}

        # 计算选股数量的一致性
        selection_counts = [len(result['selected_stocks']) for result in historical_results]
        count_consistency = 1.0 / (1.0 + np.std(selection_counts)) if selection_counts else 0.0

        # 计算收益率的一致性
        all_returns = []
        for result in historical_results:
            for stock in result['selected_stocks']:
                all_returns.append(stock.get('return_1m', 0.0))

        return_consistency = 1.0 / (1.0 + np.std(all_returns)) if all_returns else 0.0

        overall_consistency = (count_consistency + return_consistency) / 2

        return {
            'consistency_score': overall_consistency * 100,
            'selection_count_std': np.std(selection_counts) if selection_counts else 0.0,
            'return_volatility': np.std(all_returns) if all_returns else 0.0,
            'volatility_level': 'low' if np.std(all_returns) < 0.1 else 'high' if np.std(all_returns) > 0.2 else 'medium'
        }

    def _analyze_market_adaptation(self, historical_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析市场适应性"""
        # 简化的市场适应性分析
        return {
            'adaptation_score': 75.0,  # 模拟分数
            'market_conditions': 'mixed',
            'performance_in_bull_market': 'good',
            'performance_in_bear_market': 'moderate',
            'adaptability_rating': 'good'
        }

    def _generate_optimization_suggestions(self, performance_metrics: StrategyPerformanceMetrics,
                                         risk_metrics: RiskMetrics) -> List[str]:
        """生成优化建议"""
        suggestions = []

        # 基于胜率的建议
        if performance_metrics.win_rate < 0.5:
            suggestions.append("胜率偏低，建议优化选股条件，提高准确性")

        # 基于风险的建议
        if risk_metrics.maximum_drawdown > 0.15:
            suggestions.append("最大回撤过大，建议增加风险控制措施")

        # 基于收益的建议
        if performance_metrics.average_return < 0.05:
            suggestions.append("平均收益偏低，建议调整策略参数或增加新的选股因子")

        # 基于夏普比率的建议
        if performance_metrics.sharpe_ratio < 1.0:
            suggestions.append("风险调整收益不佳，建议平衡收益与风险")

        if not suggestions:
            suggestions.append("策略表现良好，建议继续监控并适时微调")

        return suggestions

    def _generate_recommendations(self, performance_metrics: StrategyPerformanceMetrics,
                                risk_metrics: RiskMetrics,
                                detailed_analysis: Dict[str, Any]) -> List[str]:
        """生成投资建议"""
        recommendations = []

        # 基于综合表现的建议
        if performance_metrics.win_rate > 0.6 and performance_metrics.sharpe_ratio > 1.5:
            recommendations.append("策略表现优秀，建议增加资金配置")
        elif performance_metrics.win_rate > 0.5 and risk_metrics.maximum_drawdown < 0.1:
            recommendations.append("策略表现稳健，建议保持当前配置")
        else:
            recommendations.append("策略需要优化，建议降低资金配置或暂停使用")

        # 风险管理建议
        if risk_metrics.maximum_drawdown > 0.2:
            recommendations.append("风险过高，建议设置止损机制")

        # 市场环境建议
        recommendations.append("建议根据市场环境动态调整策略参数")

        return recommendations

    def _calculate_overall_score(self, performance_metrics: StrategyPerformanceMetrics,
                               risk_metrics: RiskMetrics,
                               detailed_analysis: Dict[str, Any]) -> float:
        """计算综合评分"""
        try:
            weights = self.evaluation_config['evaluation_weights']

            # 性能得分 (0-100)
            performance_score = min(100.0, max(0.0, (
                performance_metrics.win_rate * 40 +
                min(performance_metrics.average_return * 200, 30) +
                min(performance_metrics.sharpe_ratio * 15, 30)
            )))

            # 风险得分 (0-100, 风险越低得分越高)
            risk_score = min(100.0, max(0.0, (
                100 - risk_metrics.maximum_drawdown * 300 -
                risk_metrics.var_95 * 200
            )))

            # 稳定性得分
            stability_score = detailed_analysis.get('consistency_analysis', {}).get('consistency_score', 50.0)

            # 一致性得分
            consistency_score = detailed_analysis.get('performance_analysis', {}).get('stability', 0.5) * 100

            # 加权计算综合得分
            overall_score = (
                performance_score * weights['performance'] +
                risk_score * weights['risk'] +
                stability_score * weights['stability'] +
                consistency_score * weights['consistency']
            )

            return min(100.0, max(0.0, overall_score))

        except Exception as e:
            self.logger.error(f"计算综合评分时出错: {e}")
            return 50.0  # 默认中等评分

    def _determine_grade(self, overall_score: float) -> str:
        """确定评级"""
        if overall_score >= 90:
            return "A+"
        elif overall_score >= 80:
            return "A"
        elif overall_score >= 70:
            return "B+"
        elif overall_score >= 60:
            return "B"
        elif overall_score >= 50:
            return "C+"
        elif overall_score >= 40:
            return "C"
        else:
            return "D"

    @exception_handler(reraise=False, default_return={})
    @performance_monitor(threshold=1.0)
    def export_evaluation_report(self, evaluation_result: StrategyEvaluationResult,
                                format_type: str = 'json') -> str:
        """导出评估报告"""
        try:
            report_data = asdict(evaluation_result)

            if format_type.lower() == 'json':
                return json.dumps(report_data, indent=2, ensure_ascii=False, default=str)
            else:
                raise ValueError(f"不支持的导出格式: {format_type}")

        except Exception as e:
            self.logger.error(f"导出评估报告时出错: {e}")
            return "{}"
