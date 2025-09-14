#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
事后风险评估与绩效归因系统

专业量化交易系统的事后分析模块，提供全面的风险评估和绩效归因分析。
用于交易后的风险复盘、收益分解、风险来源识别和策略优化建议。

核心功能：
1. 风险评估 - VaR、回撤、波动率等风险指标计算
2. 绩效归因 - 收益来源分解和策略效果评估
3. 风险归因 - 识别主要风险来源和风险贡献
4. 策略评估 - 策略有效性和风险调整收益分析
5. 优化建议 - 基于历史数据的策略优化建议

技术特性：
- 高精度数值计算（6位小数）
- 多维度风险分析
- 实时绩效追踪
- 智能异常检测
- 自动化报告生成
"""

import time
import json
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import statistics

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from utils.unified_container import get_container
from utils.numerical_stability_manager import get_stability_manager

logger = get_logger(__name__)


class AnalysisPeriod(Enum):
    """分析周期"""
    DAILY = "日度"
    WEEKLY = "周度"
    MONTHLY = "月度"
    QUARTERLY = "季度"
    YEARLY = "年度"
    CUSTOM = "自定义"


class RiskAttributionType(Enum):
    """风险归因类型"""
    MARKET_FACTOR = "市场因子"
    SECTOR_FACTOR = "行业因子"
    STYLE_FACTOR = "风格因子"
    STOCK_SPECIFIC = "个股特有"
    CURRENCY_RISK = "汇率风险"
    INTEREST_RATE_RISK = "利率风险"


class PerformanceMetricType(Enum):
    """绩效指标类型"""
    RETURN = "收益率"
    RISK = "风险"
    RISK_ADJUSTED = "风险调整"
    DRAWDOWN = "回撤"
    EFFICIENCY = "效率"


@dataclass
class RiskMetrics:
    """风险指标数据类"""
    # 基础风险指标
    var_1d: float = 0.0                         # 1日VaR
    var_5d: float = 0.0                         # 5日VaR
    var_10d: float = 0.0                        # 10日VaR
    cvar_1d: float = 0.0                        # 1日CVaR

    # 回撤指标
    max_drawdown: float = 0.0                   # 最大回撤
    current_drawdown: float = 0.0               # 当前回撤
    avg_drawdown: float = 0.0                   # 平均回撤
    drawdown_duration: int = 0                  # 回撤持续期

    # 波动率指标
    volatility_daily: float = 0.0               # 日波动率
    volatility_annualized: float = 0.0          # 年化波动率
    volatility_rolling_30d: float = 0.0         # 30日滚动波动率

    # 风险分解
    systematic_risk: float = 0.0                # 系统性风险
    specific_risk: float = 0.0                  # 特有风险
    correlation_risk: float = 0.0               # 相关性风险


@dataclass
class PerformanceMetrics:
    """绩效指标数据类"""
    # 收益指标
    total_return: float = 0.0                   # 总收益率
    annualized_return: float = 0.0              # 年化收益率
    cumulative_return: float = 0.0              # 累计收益率

    # 风险调整收益
    sharpe_ratio: float = 0.0                   # 夏普比率
    sortino_ratio: float = 0.0                  # 索提诺比率
    calmar_ratio: float = 0.0                   # 卡玛比率
    information_ratio: float = 0.0              # 信息比率

    # 效率指标
    win_rate: float = 0.0                       # 胜率
    profit_loss_ratio: float = 0.0              # 盈亏比
    average_win: float = 0.0                    # 平均盈利
    average_loss: float = 0.0                   # 平均亏损

    # 基准比较
    alpha: float = 0.0                          # 超额收益
    beta: float = 0.0                           # 市场敏感度
    tracking_error: float = 0.0                 # 跟踪误差
    active_return: float = 0.0                  # 主动收益


@dataclass
class AttributionAnalysis:
    """归因分析结果"""
    # 收益归因
    stock_selection_effect: float = 0.0         # 选股效果
    sector_allocation_effect: float = 0.0       # 行业配置效果
    timing_effect: float = 0.0                  # 择时效果
    interaction_effect: float = 0.0             # 交互效果

    # 风险归因
    market_risk_contribution: float = 0.0       # 市场风险贡献
    sector_risk_contribution: float = 0.0       # 行业风险贡献
    stock_risk_contribution: float = 0.0        # 个股风险贡献

    # 因子归因
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    factor_exposures: Dict[str, float] = field(default_factory=dict)


@dataclass
class PostTradeAnalysisResult:
    """事后分析结果"""
    # 基础信息
    analysis_date: datetime
    period_start: datetime
    period_end: datetime
    analysis_period: AnalysisPeriod

    # 风险评估
    risk_metrics: RiskMetrics

    # 绩效评估
    performance_metrics: PerformanceMetrics

    # 归因分析
    attribution_analysis: AttributionAnalysis

    # 异常检测
    anomalies_detected: List[Dict[str, Any]] = field(default_factory=list)

    # 优化建议
    optimization_suggestions: List[str] = field(default_factory=list)

    # 风险预警
    risk_warnings: List[str] = field(default_factory=list)


class PostTradeRiskAnalysisSystem:
    """
    事后风险评估与绩效归因系统

    提供全面的交易后风险分析和绩效评估功能
    """

    def __init__(self,
                 data_manager=None,
                 precision: int = 6):
        """
        初始化事后分析系统

        Args:
            data_manager: 数据管理器
            precision: 计算精度
        """
        self.data_manager = data_manager or get_container().resolve("data_manager")
        self.precision = precision
        self.stability_manager = get_stability_manager()

        # 分析缓存
        self._analysis_cache = {}
        self._cache_timestamps = {}

        # 风险模型参数
        self.confidence_levels = [0.95, 0.99]   # VaR置信度
        self.risk_free_rate = 0.03              # 无风险利率
        self.market_benchmark = "000001"        # 市场基准

        logger.info("事后风险评估与绩效归因系统初始化完成")

    @performance_monitor
    @exception_handler
    def analyze_post_trade_performance(self,
                                     start_date: datetime,
                                     end_date: datetime,
                                     analysis_period: AnalysisPeriod = AnalysisPeriod.DAILY,
                                     benchmark_code: Optional[str] = None) -> PostTradeAnalysisResult:
        """
        执行事后绩效分析

        Args:
            start_date: 分析开始日期
            end_date: 分析结束日期
            analysis_period: 分析周期
            benchmark_code: 基准代码

        Returns:
            PostTradeAnalysisResult: 分析结果
        """
        try:
            logger.info(f"开始事后分析: {start_date} 至 {end_date}")

            # 获取组合数据
            portfolio_data = self._get_portfolio_data(start_date, end_date)
            benchmark_data = self._get_benchmark_data(start_date, end_date, benchmark_code)

            # 计算风险指标
            risk_metrics = self._calculate_risk_metrics(portfolio_data)

            # 计算绩效指标
            performance_metrics = self._calculate_performance_metrics(
                portfolio_data, benchmark_data
            )

            # 执行归因分析
            attribution_analysis = self._perform_attribution_analysis(
                portfolio_data, benchmark_data, start_date, end_date
            )

            # 异常检测
            anomalies = self._detect_anomalies(portfolio_data)

            # 生成优化建议
            suggestions = self._generate_optimization_suggestions(
                risk_metrics, performance_metrics, attribution_analysis
            )

            # 生成风险预警
            warnings = self._generate_risk_warnings(risk_metrics, performance_metrics)

            result = PostTradeAnalysisResult(
                analysis_date=datetime.now(),
                period_start=start_date,
                period_end=end_date,
                analysis_period=analysis_period,
                risk_metrics=risk_metrics,
                performance_metrics=performance_metrics,
                attribution_analysis=attribution_analysis,
                anomalies_detected=anomalies,
                optimization_suggestions=suggestions,
                risk_warnings=warnings
            )

            logger.info(f"事后分析完成 - 总收益率: {performance_metrics.total_return:.2%}, "
                       f"夏普比率: {performance_metrics.sharpe_ratio:.3f}, "
                       f"最大回撤: {risk_metrics.max_drawdown:.2%}")

            return result

        except Exception as e:
            logger.error(f"事后分析异常: {e}")
            raise

    def _calculate_risk_metrics(self, portfolio_data: pd.DataFrame) -> RiskMetrics:
        """计算风险指标"""
        try:
            returns = portfolio_data['daily_return'].dropna()

            if len(returns) == 0:
                return RiskMetrics()

            # VaR计算（历史模拟法）
            var_1d_95 = self._calculate_var(returns, confidence_level=0.95, horizon=1)
            var_5d_95 = self._calculate_var(returns, confidence_level=0.95, horizon=5)
            var_10d_95 = self._calculate_var(returns, confidence_level=0.95, horizon=10)

            # CVaR计算
            cvar_1d_95 = self._calculate_cvar(returns, confidence_level=0.95)

            # 回撤计算
            drawdown_metrics = self._calculate_drawdown_metrics(portfolio_data['cumulative_return'])

            # 波动率计算
            volatility_metrics = self._calculate_volatility_metrics(returns)

            # 风险分解
            risk_decomposition = self._decompose_risk(portfolio_data)

            return RiskMetrics(
                var_1d=self.stability_manager.round_to_precision(var_1d_95, self.precision),
                var_5d=self.stability_manager.round_to_precision(var_5d_95, self.precision),
                var_10d=self.stability_manager.round_to_precision(var_10d_95, self.precision),
                cvar_1d=self.stability_manager.round_to_precision(cvar_1d_95, self.precision),
                max_drawdown=self.stability_manager.round_to_precision(drawdown_metrics['max_drawdown'], self.precision),
                current_drawdown=self.stability_manager.round_to_precision(drawdown_metrics['current_drawdown'], self.precision),
                avg_drawdown=self.stability_manager.round_to_precision(drawdown_metrics['avg_drawdown'], self.precision),
                drawdown_duration=drawdown_metrics['drawdown_duration'],
                volatility_daily=self.stability_manager.round_to_precision(volatility_metrics['daily'], self.precision),
                volatility_annualized=self.stability_manager.round_to_precision(volatility_metrics['annualized'], self.precision),
                volatility_rolling_30d=self.stability_manager.round_to_precision(volatility_metrics['rolling_30d'], self.precision),
                systematic_risk=self.stability_manager.round_to_precision(risk_decomposition['systematic'], self.precision),
                specific_risk=self.stability_manager.round_to_precision(risk_decomposition['specific'], self.precision),
                correlation_risk=self.stability_manager.round_to_precision(risk_decomposition['correlation'], self.precision)
            )

        except Exception as e:
            logger.error(f"风险指标计算异常: {e}")
            return RiskMetrics()

    def _calculate_performance_metrics(self,
                                     portfolio_data: pd.DataFrame,
                                     benchmark_data: Optional[pd.DataFrame]) -> PerformanceMetrics:
        """计算绩效指标"""
        try:
            returns = portfolio_data['daily_return'].dropna()

            if len(returns) == 0:
                return PerformanceMetrics()

            # 基础收益指标
            total_return = (portfolio_data['cumulative_return'].iloc[-1] - 1.0) if len(portfolio_data) > 0 else 0.0
            trading_days = len(returns)
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0.0

            # 风险调整收益
            excess_returns = returns - self.risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0

            # 下行风险指标
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0
            sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0.0

            # 最大回撤相关
            max_dd = abs(self._calculate_drawdown_metrics(portfolio_data['cumulative_return'])['max_drawdown'])
            calmar_ratio = annualized_return / max_dd if max_dd > 0 else 0.0

            # 交易效率指标
            win_loss_stats = self._calculate_win_loss_stats(returns)

            # 与基准比较
            benchmark_comparison = self._compare_with_benchmark(returns, benchmark_data)

            return PerformanceMetrics(
                total_return=self.stability_manager.round_to_precision(total_return, self.precision),
                annualized_return=self.stability_manager.round_to_precision(annualized_return, self.precision),
                cumulative_return=self.stability_manager.round_to_precision(total_return, self.precision),
                sharpe_ratio=self.stability_manager.round_to_precision(sharpe_ratio, self.precision),
                sortino_ratio=self.stability_manager.round_to_precision(sortino_ratio, self.precision),
                calmar_ratio=self.stability_manager.round_to_precision(calmar_ratio, self.precision),
                information_ratio=self.stability_manager.round_to_precision(benchmark_comparison.get('information_ratio', 0.0), self.precision),
                win_rate=self.stability_manager.round_to_precision(win_loss_stats['win_rate'], self.precision),
                profit_loss_ratio=self.stability_manager.round_to_precision(win_loss_stats['profit_loss_ratio'], self.precision),
                average_win=self.stability_manager.round_to_precision(win_loss_stats['average_win'], self.precision),
                average_loss=self.stability_manager.round_to_precision(win_loss_stats['average_loss'], self.precision),
                alpha=self.stability_manager.round_to_precision(benchmark_comparison.get('alpha', 0.0), self.precision),
                beta=self.stability_manager.round_to_precision(benchmark_comparison.get('beta', 1.0), self.precision),
                tracking_error=self.stability_manager.round_to_precision(benchmark_comparison.get('tracking_error', 0.0), self.precision),
                active_return=self.stability_manager.round_to_precision(benchmark_comparison.get('active_return', 0.0), self.precision)
            )

        except Exception as e:
            logger.error(f"绩效指标计算异常: {e}")
            return PerformanceMetrics()

    def _perform_attribution_analysis(self,
                                    portfolio_data: pd.DataFrame,
                                    benchmark_data: Optional[pd.DataFrame],
                                    start_date: datetime,
                                    end_date: datetime) -> AttributionAnalysis:
        """执行归因分析"""
        try:
            # 获取持仓数据
            holdings = self._get_holdings_data(start_date, end_date)

            # 选股效果分析
            stock_selection = self._analyze_stock_selection_effect(holdings, benchmark_data)

            # 行业配置效果分析
            sector_allocation = self._analyze_sector_allocation_effect(holdings, benchmark_data)

            # 择时效果分析
            timing_effect = self._analyze_timing_effect(portfolio_data, benchmark_data)

            # 风险归因
            risk_attribution = self._analyze_risk_attribution(holdings)

            # 因子分析
            factor_analysis = self._perform_factor_analysis(portfolio_data, benchmark_data)

            return AttributionAnalysis(
                stock_selection_effect=self.stability_manager.round_to_precision(stock_selection, self.precision),
                sector_allocation_effect=self.stability_manager.round_to_precision(sector_allocation, self.precision),
                timing_effect=self.stability_manager.round_to_precision(timing_effect, self.precision),
                interaction_effect=self.stability_manager.round_to_precision(0.0, self.precision),  # 简化处理
                market_risk_contribution=self.stability_manager.round_to_precision(risk_attribution['market'], self.precision),
                sector_risk_contribution=self.stability_manager.round_to_precision(risk_attribution['sector'], self.precision),
                stock_risk_contribution=self.stability_manager.round_to_precision(risk_attribution['stock'], self.precision),
                factor_contributions=factor_analysis['contributions'],
                factor_exposures=factor_analysis['exposures']
            )

        except Exception as e:
            logger.error(f"归因分析异常: {e}")
            return AttributionAnalysis()

    def _detect_anomalies(self, portfolio_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """检测异常情况"""
        anomalies = []

        try:
            returns = portfolio_data['daily_return'].dropna()

            # 异常收益检测（3倍标准差）
            return_threshold = returns.std() * 3
            extreme_returns = returns[abs(returns) > return_threshold]

            for date, return_value in extreme_returns.items():
                anomalies.append({
                    'type': '异常收益',
                    'date': date.strftime('%Y-%m-%d'),
                    'value': self.stability_manager.round_to_precision(return_value, self.precision),
                    'threshold': self.stability_manager.round_to_precision(return_threshold, self.precision),
                    'severity': '高' if abs(return_value) > return_threshold * 2 else '中'
                })

            # 连续下跌检测
            consecutive_losses = self._detect_consecutive_losses(returns)
            for loss_period in consecutive_losses:
                anomalies.append({
                    'type': '连续亏损',
                    'start_date': loss_period['start_date'],
                    'end_date': loss_period['end_date'],
                    'duration': loss_period['duration'],
                    'cumulative_loss': self.stability_manager.round_to_precision(loss_period['cumulative_loss'], self.precision),
                    'severity': '高' if loss_period['duration'] > 5 else '中'
                })

            # 波动率突增检测
            volatility_spikes = self._detect_volatility_spikes(returns)
            anomalies.extend(volatility_spikes)

        except Exception as e:
            logger.error(f"异常检测异常: {e}")

        return anomalies

    def _generate_optimization_suggestions(self,
                                         risk_metrics: RiskMetrics,
                                         performance_metrics: PerformanceMetrics,
                                         attribution_analysis: AttributionAnalysis) -> List[str]:
        """生成优化建议"""
        suggestions = []

        try:
            # 风险优化建议
            if risk_metrics.max_drawdown > 0.10:  # 最大回撤超过10%
                suggestions.append(f"建议加强风控措施，当前最大回撤 {risk_metrics.max_drawdown:.2%} 较高")

            if risk_metrics.volatility_annualized > 0.25:  # 年化波动率超过25%
                suggestions.append(f"建议降低组合波动率，当前年化波动率 {risk_metrics.volatility_annualized:.2%}")

            # 绩效优化建议
            if performance_metrics.sharpe_ratio < 1.0:
                suggestions.append(f"建议提高风险调整收益，当前夏普比率 {performance_metrics.sharpe_ratio:.3f}")

            if performance_metrics.win_rate < 0.5:
                suggestions.append(f"建议提高交易胜率，当前胜率 {performance_metrics.win_rate:.2%}")

            # 归因分析建议
            if attribution_analysis.stock_selection_effect < 0:
                suggestions.append(f"选股效果为负 {attribution_analysis.stock_selection_effect:.2%}，建议优化选股策略")

            if attribution_analysis.sector_allocation_effect < 0:
                suggestions.append(f"行业配置效果为负 {attribution_analysis.sector_allocation_effect:.2%}，建议调整行业配置")

            # 风险结构建议
            if risk_metrics.specific_risk > risk_metrics.systematic_risk * 2:
                suggestions.append("特有风险过高，建议增加分散化程度")

            # 基准比较建议
            if performance_metrics.tracking_error > 0.05:
                suggestions.append(f"跟踪误差较大 {performance_metrics.tracking_error:.2%}，建议调整策略或基准")

        except Exception as e:
            logger.error(f"优化建议生成异常: {e}")

        return suggestions

    def _generate_risk_warnings(self,
                               risk_metrics: RiskMetrics,
                               performance_metrics: PerformanceMetrics) -> List[str]:
        """生成风险预警"""
        warnings = []

        try:
            # 回撤预警
            if risk_metrics.current_drawdown > 0.05:
                warnings.append(f"当前回撤 {risk_metrics.current_drawdown:.2%} 需要关注")

            if risk_metrics.max_drawdown > 0.15:
                warnings.append(f"最大回撤 {risk_metrics.max_drawdown:.2%} 过高，风险较大")

            # VaR预警
            portfolio_value = 1000000  # 假设组合价值100万
            var_amount = portfolio_value * risk_metrics.var_1d
            if var_amount > 50000:  # VaR金额超过5万
                warnings.append(f"日VaR金额 {var_amount:,.0f} 元较高")

            # 波动率预警
            if risk_metrics.volatility_rolling_30d > risk_metrics.volatility_daily * 1.5:
                warnings.append("近期波动率上升，需要密切关注市场变化")

            # 收益质量预警
            if performance_metrics.sortino_ratio < performance_metrics.sharpe_ratio * 0.7:
                warnings.append("下行风险较高，收益质量需要改善")

            # Beta预警
            if abs(performance_metrics.beta) > 1.3:
                warnings.append(f"市场敏感度 {performance_metrics.beta:.2f} 较高，系统性风险需要关注")

        except Exception as e:
            logger.error(f"风险预警生成异常: {e}")

        return warnings

    # 辅助计算方法
    def _calculate_var(self, returns: pd.Series, confidence_level: float = 0.95, horizon: int = 1) -> float:
        """计算VaR（历史模拟法）"""
        if len(returns) == 0:
            return 0.0

        # 计算分位数
        percentile = (1 - confidence_level) * 100
        var_1d = np.percentile(returns, percentile)

        # 时间调整
        var_horizon = var_1d * np.sqrt(horizon)

        return abs(var_horizon)

    def _calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算CVaR（条件VaR）"""
        if len(returns) == 0:
            return 0.0

        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        tail_returns = returns[returns <= var_threshold]

        return abs(tail_returns.mean()) if len(tail_returns) > 0 else 0.0

    def _calculate_drawdown_metrics(self, cumulative_returns: pd.Series) -> Dict[str, Any]:
        """计算回撤指标"""
        if len(cumulative_returns) == 0:
            return {'max_drawdown': 0.0, 'current_drawdown': 0.0, 'avg_drawdown': 0.0, 'drawdown_duration': 0}

        # 计算历史最高点
        peak = cumulative_returns.expanding().max()

        # 计算回撤
        drawdown = (cumulative_returns - peak) / peak

        # 最大回撤
        max_drawdown = drawdown.min()

        # 当前回撤
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0

        # 平均回撤
        negative_dd = drawdown[drawdown < 0]
        avg_drawdown = negative_dd.mean() if len(negative_dd) > 0 else 0.0

        # 回撤持续期（简化计算）
        current_dd_duration = 0
        for i in range(len(drawdown) - 1, -1, -1):
            if drawdown.iloc[i] < 0:
                current_dd_duration += 1
            else:
                break

        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'avg_drawdown': avg_drawdown,
            'drawdown_duration': current_dd_duration
        }

    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """计算波动率指标"""
        if len(returns) == 0:
            return {'daily': 0.0, 'annualized': 0.0, 'rolling_30d': 0.0}

        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)

        # 30日滚动波动率
        if len(returns) >= 30:
            rolling_30d_vol = returns.rolling(30).std().iloc[-1] * np.sqrt(252)
        else:
            rolling_30d_vol = annualized_vol

        return {
            'daily': daily_vol,
            'annualized': annualized_vol,
            'rolling_30d': rolling_30d_vol
        }

    def _decompose_risk(self, portfolio_data: pd.DataFrame) -> Dict[str, float]:
        """风险分解（简化版）"""
        returns = portfolio_data['daily_return'].dropna()

        if len(returns) == 0:
            return {'systematic': 0.0, 'specific': 0.0, 'correlation': 0.0}

        total_variance = returns.var()

        # 简化的风险分解模型
        systematic_risk = total_variance * 0.6   # 假设60%为系统性风险
        specific_risk = total_variance * 0.3     # 假设30%为特有风险
        correlation_risk = total_variance * 0.1  # 假设10%为相关性风险

        return {
            'systematic': systematic_risk,
            'specific': specific_risk,
            'correlation': correlation_risk
        }

    def _calculate_win_loss_stats(self, returns: pd.Series) -> Dict[str, float]:
        """计算胜负统计"""
        if len(returns) == 0:
            return {'win_rate': 0.0, 'profit_loss_ratio': 0.0, 'average_win': 0.0, 'average_loss': 0.0}

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        win_rate = len(wins) / len(returns)
        average_win = wins.mean() if len(wins) > 0 else 0.0
        average_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        profit_loss_ratio = average_win / average_loss if average_loss > 0 else 0.0

        return {
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'average_win': average_win,
            'average_loss': average_loss
        }

    def _compare_with_benchmark(self, returns: pd.Series, benchmark_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """与基准比较"""
        if benchmark_data is None or len(benchmark_data) == 0:
            return {'alpha': 0.0, 'beta': 1.0, 'tracking_error': 0.0, 'information_ratio': 0.0, 'active_return': 0.0}

        benchmark_returns = benchmark_data['daily_return'].dropna()

        # 对齐数据
        aligned_data = pd.concat([returns, benchmark_returns], axis=1, join='inner')
        aligned_data.columns = ['portfolio', 'benchmark']

        if len(aligned_data) < 2:
            return {'alpha': 0.0, 'beta': 1.0, 'tracking_error': 0.0, 'information_ratio': 0.0, 'active_return': 0.0}

        # 计算Beta和Alpha
        covariance = aligned_data['portfolio'].cov(aligned_data['benchmark'])
        benchmark_variance = aligned_data['benchmark'].var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

        portfolio_mean = aligned_data['portfolio'].mean()
        benchmark_mean = aligned_data['benchmark'].mean()
        alpha = portfolio_mean - beta * benchmark_mean

        # 主动收益和跟踪误差
        active_returns = aligned_data['portfolio'] - aligned_data['benchmark']
        active_return = active_returns.mean()
        tracking_error = active_returns.std()
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0.0

        return {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'active_return': active_return
        }

    def _detect_consecutive_losses(self, returns: pd.Series) -> List[Dict[str, Any]]:
        """检测连续亏损期"""
        consecutive_losses = []

        if len(returns) == 0:
            return consecutive_losses

        current_loss_start = None
        current_loss_sum = 0.0

        for date, return_value in returns.items():
            if return_value < 0:
                if current_loss_start is None:
                    current_loss_start = date
                    current_loss_sum = return_value
                else:
                    current_loss_sum += return_value
            else:
                if current_loss_start is not None:
                    # 结束当前连续亏损期
                    duration = (date - current_loss_start).days
                    if duration >= 3:  # 至少连续3个交易日
                        consecutive_losses.append({
                            'start_date': current_loss_start.strftime('%Y-%m-%d'),
                            'end_date': date.strftime('%Y-%m-%d'),
                            'duration': duration,
                            'cumulative_loss': current_loss_sum
                        })

                    current_loss_start = None
                    current_loss_sum = 0.0

        # 处理以亏损结尾的情况
        if current_loss_start is not None:
            end_date = returns.index[-1]
            duration = (end_date - current_loss_start).days
            if duration >= 3:
                consecutive_losses.append({
                    'start_date': current_loss_start.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'duration': duration,
                    'cumulative_loss': current_loss_sum
                })

        return consecutive_losses

    def _detect_volatility_spikes(self, returns: pd.Series) -> List[Dict[str, Any]]:
        """检测波动率突增"""
        volatility_spikes = []

        if len(returns) < 20:
            return volatility_spikes

        # 计算20日滚动波动率
        rolling_vol = returns.rolling(20).std()
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()

        # 检测超过均值+2倍标准差的情况
        threshold = vol_mean + 2 * vol_std

        spikes = rolling_vol[rolling_vol > threshold]
        for date, vol_value in spikes.items():
            volatility_spikes.append({
                'type': '波动率突增',
                'date': date.strftime('%Y-%m-%d'),
                'value': self.stability_manager.round_to_precision(vol_value, self.precision),
                'threshold': self.stability_manager.round_to_precision(threshold, self.precision),
                'severity': '高' if vol_value > threshold * 1.5 else '中'
            })

        return volatility_spikes

    # 数据获取方法（模拟实现）
    def _get_portfolio_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """获取组合数据（模拟）"""
        # 生成模拟的组合数据
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日

        np.random.seed(42)  # 确保结果可重复
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 日均收益0.05%，波动率2%

        # 添加一些趋势和波动
        trend = np.linspace(0, 0.1, len(dates)) / len(dates)
        returns = returns + trend

        cumulative_return = (1 + pd.Series(returns, index=dates)).cumprod()

        portfolio_data = pd.DataFrame({
            'date': dates,
            'daily_return': returns,
            'cumulative_return': cumulative_return
        })
        portfolio_data.set_index('date', inplace=True)

        return portfolio_data

    def _get_benchmark_data(self, start_date: datetime, end_date: datetime, benchmark_code: Optional[str]) -> Optional[pd.DataFrame]:
        """获取基准数据（模拟）"""
        if benchmark_code is None:
            return None

        dates = pd.date_range(start=start_date, end=end_date, freq='B')

        np.random.seed(123)  # 不同的随机种子
        returns = np.random.normal(0.0003, 0.015, len(dates))  # 基准收益稍低，波动率稍小

        cumulative_return = (1 + pd.Series(returns, index=dates)).cumprod()

        benchmark_data = pd.DataFrame({
            'date': dates,
            'daily_return': returns,
            'cumulative_return': cumulative_return
        })
        benchmark_data.set_index('date', inplace=True)

        return benchmark_data

    def _get_holdings_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """获取持仓数据（模拟）"""
        # 模拟持仓数据
        stocks = ['000001', '000002', '300001', '300002', '600001']
        sectors = ['金融', '金融', '科技', '科技', '制造业']

        holdings = []
        for i, stock in enumerate(stocks):
            holdings.append({
                'stock_code': stock,
                'sector': sectors[i],
                'weight': 0.2,  # 等权重
                'return_contribution': np.random.normal(0.0001, 0.005)
            })

        return pd.DataFrame(holdings)

    def _analyze_stock_selection_effect(self, holdings: pd.DataFrame, benchmark_data: Optional[pd.DataFrame]) -> float:
        """分析选股效果（简化版）"""
        if len(holdings) == 0:
            return 0.0

        # 简化的选股效果计算
        stock_excess_returns = holdings['return_contribution'].sum()
        return stock_excess_returns * 252  # 年化

    def _analyze_sector_allocation_effect(self, holdings: pd.DataFrame, benchmark_data: Optional[pd.DataFrame]) -> float:
        """分析行业配置效果（简化版）"""
        if len(holdings) == 0:
            return 0.0

        # 简化的行业配置效果
        sector_weights = holdings.groupby('sector')['weight'].sum()
        allocation_effect = np.random.normal(0.0, 0.01)  # 模拟配置效果
        return allocation_effect * 252  # 年化

    def _analyze_timing_effect(self, portfolio_data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame]) -> float:
        """分析择时效果（简化版）"""
        # 简化的择时效果分析
        if benchmark_data is None or len(portfolio_data) == 0:
            return 0.0

        timing_effect = np.random.normal(0.0, 0.005)  # 模拟择时效果
        return timing_effect * 252  # 年化

    def _analyze_risk_attribution(self, holdings: pd.DataFrame) -> Dict[str, float]:
        """分析风险归因（简化版）"""
        total_risk = 0.04  # 假设总风险4%

        return {
            'market': total_risk * 0.6,
            'sector': total_risk * 0.25,
            'stock': total_risk * 0.15
        }

    def _perform_factor_analysis(self, portfolio_data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """执行因子分析（简化版）"""
        factors = ['市场因子', '价值因子', '成长因子', '质量因子', '动量因子']

        contributions = {}
        exposures = {}

        for factor in factors:
            contributions[factor] = self.stability_manager.round_to_precision(
                np.random.normal(0.0, 0.02), self.precision
            )
            exposures[factor] = self.stability_manager.round_to_precision(
                np.random.normal(0.0, 0.5), self.precision
            )

        return {
            'contributions': contributions,
            'exposures': exposures
        }

    def generate_analysis_report(self, analysis_result: PostTradeAnalysisResult) -> str:
        """生成分析报告"""
        report = f"""
# 事后风险评估与绩效归因报告

## 分析概览
- 分析日期: {analysis_result.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}
- 分析期间: {analysis_result.period_start.strftime('%Y-%m-%d')} 至 {analysis_result.period_end.strftime('%Y-%m-%d')}
- 分析周期: {analysis_result.analysis_period.value}

## 绩效指标
- 总收益率: {analysis_result.performance_metrics.total_return:.2%}
- 年化收益率: {analysis_result.performance_metrics.annualized_return:.2%}
- 夏普比率: {analysis_result.performance_metrics.sharpe_ratio:.3f}
- 索提诺比率: {analysis_result.performance_metrics.sortino_ratio:.3f}
- 卡玛比率: {analysis_result.performance_metrics.calmar_ratio:.3f}
- 胜率: {analysis_result.performance_metrics.win_rate:.2%}
- 盈亏比: {analysis_result.performance_metrics.profit_loss_ratio:.2f}

## 风险指标
- 最大回撤: {analysis_result.risk_metrics.max_drawdown:.2%}
- 当前回撤: {analysis_result.risk_metrics.current_drawdown:.2%}
- 日VaR(95%): {analysis_result.risk_metrics.var_1d:.2%}
- 年化波动率: {analysis_result.risk_metrics.volatility_annualized:.2%}
- 系统性风险: {analysis_result.risk_metrics.systematic_risk:.2%}
- 特有风险: {analysis_result.risk_metrics.specific_risk:.2%}

## 归因分析
- 选股效果: {analysis_result.attribution_analysis.stock_selection_effect:.2%}
- 行业配置效果: {analysis_result.attribution_analysis.sector_allocation_effect:.2%}
- 择时效果: {analysis_result.attribution_analysis.timing_effect:.2%}
- 市场风险贡献: {analysis_result.attribution_analysis.market_risk_contribution:.2%}
- 行业风险贡献: {analysis_result.attribution_analysis.sector_risk_contribution:.2%}

## 异常检测
"""

        if analysis_result.anomalies_detected:
            for anomaly in analysis_result.anomalies_detected:
                report += f"- {anomaly['type']}: {anomaly.get('date', '')} - {anomaly.get('message', '无详细信息')}\n"
        else:
            report += "- 未检测到异常情况\n"

        report += "\n## 优化建议\n"
        if analysis_result.optimization_suggestions:
            for suggestion in analysis_result.optimization_suggestions:
                report += f"- {suggestion}\n"
        else:
            report += "- 当前组合表现良好，无特别建议\n"

        report += "\n## 风险预警\n"
        if analysis_result.risk_warnings:
            for warning in analysis_result.risk_warnings:
                report += f"- ⚠️  {warning}\n"
        else:
            report += "- 无风险预警\n"

        return report


# 全局实例管理
_post_trade_analysis_system = None

def get_post_trade_analysis_system(data_manager=None) -> PostTradeRiskAnalysisSystem:
    """
    获取事后分析系统实例（单例模式）

    Args:
        data_manager: 数据管理器

    Returns:
        PostTradeRiskAnalysisSystem: 事后分析系统实例
    """
    global _post_trade_analysis_system

    if _post_trade_analysis_system is None:
        _post_trade_analysis_system = PostTradeRiskAnalysisSystem(
            data_manager=data_manager
        )

    return _post_trade_analysis_system


if __name__ == "__main__":
    # 演示使用
    print("=== 事后风险评估与绩效归因系统演示 ===")

    # 创建分析系统
    analysis_system = get_post_trade_analysis_system()

    # 设置分析期间
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 最近90天

    # 执行事后分析
    result = analysis_system.analyze_post_trade_performance(
        start_date=start_date,
        end_date=end_date,
        analysis_period=AnalysisPeriod.DAILY,
        benchmark_code="000001"
    )

    # 生成报告
    report = analysis_system.generate_analysis_report(result)
    print(report)

    # 输出关键指标
    print(f"\n=== 关键指标摘要 ===")
    print(f"总收益率: {result.performance_metrics.total_return:.2%}")
    print(f"夏普比率: {result.performance_metrics.sharpe_ratio:.3f}")
    print(f"最大回撤: {result.risk_metrics.max_drawdown:.2%}")
    print(f"年化波动率: {result.risk_metrics.volatility_annualized:.2%}")
    print(f"异常检测数量: {len(result.anomalies_detected)}")
    print(f"优化建议数量: {len(result.optimization_suggestions)}")
    print(f"风险预警数量: {len(result.risk_warnings)}")

    print("演示完成")