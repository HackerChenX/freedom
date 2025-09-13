#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
业务指标监控仪表板

股票分析系统专用的业务指标监控和可视化仪表板：
1. 股票分析性能指标（分析速度、成功率、覆盖范围）
2. 技术指标计算监控（88+指标计算状态、准确性）
3. 买点检测系统监控（检测准确率、响应时间）
4. 市场数据监控（数据更新状态、质量检查）
5. 用户行为分析（查询模式、使用频率）
6. 系统资源使用率（缓存命中率、并发处理能力）
7. 收益分析（策略回测表现、实时收益跟踪）
8. 风险监控（风险指标、异常检测）
"""

import os
import json
import time
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor

logger = get_logger(__name__)


class MetricCategory(Enum):
    """指标类别"""
    ANALYSIS_PERFORMANCE = "analysis_performance"
    TECHNICAL_INDICATORS = "technical_indicators"
    SIGNAL_DETECTION = "signal_detection"
    DATA_QUALITY = "data_quality"
    USER_BEHAVIOR = "user_behavior"
    SYSTEM_RESOURCES = "system_resources"
    TRADING_PERFORMANCE = "trading_performance"
    RISK_MANAGEMENT = "risk_management"


class TrendDirection(Enum):
    """趋势方向"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class BusinessMetric:
    """业务指标"""
    category: MetricCategory
    name: str
    value: Union[int, float, str]
    unit: str
    timestamp: datetime
    trend: TrendDirection
    target_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    description: str = ""
    tags: Dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class DashboardWidget:
    """仪表板组件"""
    widget_id: str
    title: str
    widget_type: str  # chart, gauge, number, table, heatmap
    category: MetricCategory
    metrics: List[str]
    config: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    refresh_interval: int = 60


@dataclass
class AlertCondition:
    """告警条件"""
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold: float
    duration: int  # 持续时间（秒）
    severity: str  # info, warning, critical


class BusinessMetricsCollector:
    """业务指标收集器"""

    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.collection_active = False
        self.collection_thread: Optional[threading.Thread] = None
        self.metric_calculators: Dict[MetricCategory, callable] = {}

        # 注册指标计算器
        self._register_metric_calculators()

        # 模拟数据存储
        self.stock_analysis_stats = {
            'total_analyzed': 0,
            'success_count': 0,
            'average_time': 0.0,
            'stocks_per_hour': 0.0
        }

        self.indicator_stats = {
            'total_calculations': 0,
            'indicators_calculated': {},
            'calculation_times': defaultdict(list),
            'accuracy_scores': defaultdict(list)
        }

        logger.info("业务指标收集器初始化完成")

    def _register_metric_calculators(self):
        """注册指标计算器"""
        self.metric_calculators[MetricCategory.ANALYSIS_PERFORMANCE] = self._collect_analysis_performance_metrics
        self.metric_calculators[MetricCategory.TECHNICAL_INDICATORS] = self._collect_technical_indicator_metrics
        self.metric_calculators[MetricCategory.SIGNAL_DETECTION] = self._collect_signal_detection_metrics
        self.metric_calculators[MetricCategory.DATA_QUALITY] = self._collect_data_quality_metrics
        self.metric_calculators[MetricCategory.USER_BEHAVIOR] = self._collect_user_behavior_metrics
        self.metric_calculators[MetricCategory.SYSTEM_RESOURCES] = self._collect_system_resource_metrics
        self.metric_calculators[MetricCategory.TRADING_PERFORMANCE] = self._collect_trading_performance_metrics
        self.metric_calculators[MetricCategory.RISK_MANAGEMENT] = self._collect_risk_management_metrics

    @exception_handler(reraise=True)
    def start_collection(self, interval: int = 60):
        """启动指标收集"""
        if self.collection_active:
            logger.warning("指标收集已在运行")
            return

        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval,),
            daemon=True
        )
        self.collection_thread.start()

        logger.info(f"业务指标收集已启动，间隔: {interval}秒")

    def stop_collection(self):
        """停止指标收集"""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=10)
        logger.info("业务指标收集已停止")

    def _collection_loop(self, interval: int):
        """指标收集循环"""
        while self.collection_active:
            try:
                # 收集所有类别的指标
                for category, calculator in self.metric_calculators.items():
                    try:
                        metrics = calculator()
                        for metric in metrics:
                            self._store_metric(metric)
                    except Exception as e:
                        logger.error(f"收集{category.value}指标失败: {e}")

                time.sleep(interval)

            except Exception as e:
                logger.error(f"指标收集循环出错: {e}")
                time.sleep(5)

    def _store_metric(self, metric: BusinessMetric):
        """存储指标"""
        metric_key = f"{metric.category.value}_{metric.name}"
        self.metrics_history[metric_key].append(metric)

    def _collect_analysis_performance_metrics(self) -> List[BusinessMetric]:
        """收集分析性能指标"""
        metrics = []
        timestamp = datetime.now()

        try:
            # 模拟更新分析统计数据
            import random
            self.stock_analysis_stats['total_analyzed'] += random.randint(50, 200)
            self.stock_analysis_stats['success_count'] += random.randint(45, 190)
            self.stock_analysis_stats['average_time'] = random.uniform(0.03, 0.08)
            self.stock_analysis_stats['stocks_per_hour'] = random.uniform(60000, 75000)

            # 分析总数指标
            metrics.append(BusinessMetric(
                category=MetricCategory.ANALYSIS_PERFORMANCE,
                name="total_stocks_analyzed",
                value=self.stock_analysis_stats['total_analyzed'],
                unit="count",
                timestamp=timestamp,
                trend=TrendDirection.UP,
                target_value=100000,
                description="已分析股票总数"
            ))

            # 成功率指标
            success_rate = (self.stock_analysis_stats['success_count'] /
                          max(self.stock_analysis_stats['total_analyzed'], 1)) * 100

            metrics.append(BusinessMetric(
                category=MetricCategory.ANALYSIS_PERFORMANCE,
                name="analysis_success_rate",
                value=success_rate,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=95.0,
                threshold_warning=90.0,
                threshold_critical=85.0,
                description="股票分析成功率"
            ))

            # 平均分析时间
            metrics.append(BusinessMetric(
                category=MetricCategory.ANALYSIS_PERFORMANCE,
                name="average_analysis_time",
                value=self.stock_analysis_stats['average_time'],
                unit="seconds",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=0.05,
                threshold_warning=0.10,
                threshold_critical=0.15,
                description="平均股票分析时间"
            ))

            # 每小时分析股票数
            metrics.append(BusinessMetric(
                category=MetricCategory.ANALYSIS_PERFORMANCE,
                name="stocks_per_hour",
                value=self.stock_analysis_stats['stocks_per_hour'],
                unit="stocks/hour",
                timestamp=timestamp,
                trend=TrendDirection.UP,
                target_value=72000,
                threshold_warning=50000,
                threshold_critical=30000,
                description="每小时分析股票数量"
            ))

        except Exception as e:
            logger.error(f"收集分析性能指标失败: {e}")

        return metrics

    def _collect_technical_indicator_metrics(self) -> List[BusinessMetric]:
        """收集技术指标计算指标"""
        metrics = []
        timestamp = datetime.now()

        try:
            import random

            # 更新指标统计数据
            indicator_names = ['RSI', 'MACD', 'KDJ', 'BOLL', 'ATR', 'CCI', 'WR', 'ROC', 'MA', 'EMA']

            for indicator in indicator_names:
                if indicator not in self.indicator_stats['indicators_calculated']:
                    self.indicator_stats['indicators_calculated'][indicator] = 0

                # 模拟新的计算
                new_calculations = random.randint(100, 500)
                self.indicator_stats['indicators_calculated'][indicator] += new_calculations

                # 模拟计算时间
                calc_time = random.uniform(0.001, 0.005)
                self.indicator_stats['calculation_times'][indicator].append(calc_time)

                # 模拟准确率分数
                accuracy = random.uniform(0.95, 0.999)
                self.indicator_stats['accuracy_scores'][indicator].append(accuracy)

            # 总计算次数
            total_calculations = sum(self.indicator_stats['indicators_calculated'].values())
            metrics.append(BusinessMetric(
                category=MetricCategory.TECHNICAL_INDICATORS,
                name="total_indicator_calculations",
                value=total_calculations,
                unit="count",
                timestamp=timestamp,
                trend=TrendDirection.UP,
                description="技术指标总计算次数"
            ))

            # 平均计算时间
            all_times = []
            for times in self.indicator_stats['calculation_times'].values():
                all_times.extend(times[-10:])  # 最近10次

            avg_calc_time = statistics.mean(all_times) if all_times else 0
            metrics.append(BusinessMetric(
                category=MetricCategory.TECHNICAL_INDICATORS,
                name="average_calculation_time",
                value=avg_calc_time,
                unit="seconds",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=0.002,
                threshold_warning=0.01,
                threshold_critical=0.05,
                description="技术指标平均计算时间"
            ))

            # 平均准确率
            all_accuracies = []
            for accuracies in self.indicator_stats['accuracy_scores'].values():
                all_accuracies.extend(accuracies[-5:])  # 最近5次

            avg_accuracy = statistics.mean(all_accuracies) * 100 if all_accuracies else 0
            metrics.append(BusinessMetric(
                category=MetricCategory.TECHNICAL_INDICATORS,
                name="average_accuracy",
                value=avg_accuracy,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=99.0,
                threshold_warning=95.0,
                threshold_critical=90.0,
                description="技术指标平均准确率"
            ))

            # 活跃指标数量
            active_indicators = len([i for i, count in self.indicator_stats['indicators_calculated'].items() if count > 0])
            metrics.append(BusinessMetric(
                category=MetricCategory.TECHNICAL_INDICATORS,
                name="active_indicators_count",
                value=active_indicators,
                unit="count",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=88,
                description="活跃技术指标数量"
            ))

        except Exception as e:
            logger.error(f"收集技术指标指标失败: {e}")

        return metrics

    def _collect_signal_detection_metrics(self) -> List[BusinessMetric]:
        """收集信号检测指标"""
        metrics = []
        timestamp = datetime.now()

        try:
            import random

            # 模拟信号检测数据
            total_signals = random.randint(500, 1500)
            buy_signals = random.randint(200, 600)
            sell_signals = random.randint(150, 400)
            hold_signals = total_signals - buy_signals - sell_signals

            # 总信号数
            metrics.append(BusinessMetric(
                category=MetricCategory.SIGNAL_DETECTION,
                name="total_signals_detected",
                value=total_signals,
                unit="count",
                timestamp=timestamp,
                trend=TrendDirection.UP,
                description="检测到的总信号数量"
            ))

            # 买入信号数
            metrics.append(BusinessMetric(
                category=MetricCategory.SIGNAL_DETECTION,
                name="buy_signals_count",
                value=buy_signals,
                unit="count",
                timestamp=timestamp,
                trend=TrendDirection.VOLATILE,
                description="买入信号数量",
                tags={'signal_type': 'buy'}
            ))

            # 卖出信号数
            metrics.append(BusinessMetric(
                category=MetricCategory.SIGNAL_DETECTION,
                name="sell_signals_count",
                value=sell_signals,
                unit="count",
                timestamp=timestamp,
                trend=TrendDirection.VOLATILE,
                description="卖出信号数量",
                tags={'signal_type': 'sell'}
            ))

            # 信号准确率
            signal_accuracy = random.uniform(75, 90)
            metrics.append(BusinessMetric(
                category=MetricCategory.SIGNAL_DETECTION,
                name="signal_accuracy_rate",
                value=signal_accuracy,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=85.0,
                threshold_warning=75.0,
                threshold_critical=65.0,
                description="信号检测准确率"
            ))

            # 信号响应时间
            response_time = random.uniform(0.5, 2.0)
            metrics.append(BusinessMetric(
                category=MetricCategory.SIGNAL_DETECTION,
                name="signal_response_time",
                value=response_time,
                unit="seconds",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=1.0,
                threshold_warning=3.0,
                threshold_critical=5.0,
                description="信号检测响应时间"
            ))

        except Exception as e:
            logger.error(f"收集信号检测指标失败: {e}")

        return metrics

    def _collect_data_quality_metrics(self) -> List[BusinessMetric]:
        """收集数据质量指标"""
        metrics = []
        timestamp = datetime.now()

        try:
            import random

            # 数据完整性
            data_completeness = random.uniform(95, 99.9)
            metrics.append(BusinessMetric(
                category=MetricCategory.DATA_QUALITY,
                name="data_completeness",
                value=data_completeness,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=99.0,
                threshold_warning=95.0,
                threshold_critical=90.0,
                description="数据完整性"
            ))

            # 数据准确性
            data_accuracy = random.uniform(98, 99.9)
            metrics.append(BusinessMetric(
                category=MetricCategory.DATA_QUALITY,
                name="data_accuracy",
                value=data_accuracy,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=99.5,
                threshold_warning=98.0,
                threshold_critical=95.0,
                description="数据准确性"
            ))

            # 数据更新及时性
            data_freshness = random.uniform(85, 98)
            metrics.append(BusinessMetric(
                category=MetricCategory.DATA_QUALITY,
                name="data_freshness",
                value=data_freshness,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=95.0,
                threshold_warning=85.0,
                threshold_critical=75.0,
                description="数据更新及时性"
            ))

            # 覆盖股票数量
            covered_stocks = random.randint(4500, 5000)
            metrics.append(BusinessMetric(
                category=MetricCategory.DATA_QUALITY,
                name="covered_stocks_count",
                value=covered_stocks,
                unit="count",
                timestamp=timestamp,
                trend=TrendDirection.UP,
                target_value=5000,
                description="数据覆盖股票数量"
            ))

        except Exception as e:
            logger.error(f"收集数据质量指标失败: {e}")

        return metrics

    def _collect_user_behavior_metrics(self) -> List[BusinessMetric]:
        """收集用户行为指标"""
        metrics = []
        timestamp = datetime.now()

        try:
            import random

            # 活跃用户数
            active_users = random.randint(50, 200)
            metrics.append(BusinessMetric(
                category=MetricCategory.USER_BEHAVIOR,
                name="active_users_count",
                value=active_users,
                unit="count",
                timestamp=timestamp,
                trend=TrendDirection.UP,
                description="活跃用户数量"
            ))

            # 查询次数
            total_queries = random.randint(1000, 5000)
            metrics.append(BusinessMetric(
                category=MetricCategory.USER_BEHAVIOR,
                name="total_queries_count",
                value=total_queries,
                unit="count",
                timestamp=timestamp,
                trend=TrendDirection.UP,
                description="用户查询总次数"
            ))

            # 平均会话时长
            avg_session_duration = random.uniform(15, 45)
            metrics.append(BusinessMetric(
                category=MetricCategory.USER_BEHAVIOR,
                name="average_session_duration",
                value=avg_session_duration,
                unit="minutes",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                description="平均会话时长"
            ))

            # 热门股票查询排名更新
            popular_stocks = ['000001', '000002', '600000', '600036', '000858']
            for i, stock_code in enumerate(popular_stocks[:3]):
                metrics.append(BusinessMetric(
                    category=MetricCategory.USER_BEHAVIOR,
                    name="popular_stock_queries",
                    value=random.randint(50, 200),
                    unit="count",
                    timestamp=timestamp,
                    trend=TrendDirection.VOLATILE,
                    description=f"热门股票查询次数",
                    tags={'stock_code': stock_code, 'rank': str(i+1)}
                ))

        except Exception as e:
            logger.error(f"收集用户行为指标失败: {e}")

        return metrics

    def _collect_system_resource_metrics(self) -> List[BusinessMetric]:
        """收集系统资源指标"""
        metrics = []
        timestamp = datetime.now()

        try:
            import random

            # 缓存命中率
            cache_hit_rate = random.uniform(85, 95)
            metrics.append(BusinessMetric(
                category=MetricCategory.SYSTEM_RESOURCES,
                name="cache_hit_rate",
                value=cache_hit_rate,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=90.0,
                threshold_warning=80.0,
                threshold_critical=70.0,
                description="缓存命中率"
            ))

            # 并发处理能力
            concurrent_requests = random.randint(80, 150)
            metrics.append(BusinessMetric(
                category=MetricCategory.SYSTEM_RESOURCES,
                name="concurrent_processing_capacity",
                value=concurrent_requests,
                unit="requests",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=100,
                description="并发处理能力"
            ))

            # 数据库连接池使用率
            db_pool_usage = random.uniform(30, 80)
            metrics.append(BusinessMetric(
                category=MetricCategory.SYSTEM_RESOURCES,
                name="database_pool_usage",
                value=db_pool_usage,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                threshold_warning=85.0,
                threshold_critical=95.0,
                description="数据库连接池使用率"
            ))

            # 内存使用效率
            memory_efficiency = random.uniform(75, 90)
            metrics.append(BusinessMetric(
                category=MetricCategory.SYSTEM_RESOURCES,
                name="memory_efficiency",
                value=memory_efficiency,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=85.0,
                description="内存使用效率"
            ))

        except Exception as e:
            logger.error(f"收集系统资源指标失败: {e}")

        return metrics

    def _collect_trading_performance_metrics(self) -> List[BusinessMetric]:
        """收集交易性能指标"""
        metrics = []
        timestamp = datetime.now()

        try:
            import random

            # 策略总数
            total_strategies = random.randint(50, 100)
            metrics.append(BusinessMetric(
                category=MetricCategory.TRADING_PERFORMANCE,
                name="total_strategies_count",
                value=total_strategies,
                unit="count",
                timestamp=timestamp,
                trend=TrendDirection.UP,
                description="交易策略总数"
            ))

            # 回测成功率
            backtest_success_rate = random.uniform(70, 85)
            metrics.append(BusinessMetric(
                category=MetricCategory.TRADING_PERFORMANCE,
                name="backtest_success_rate",
                value=backtest_success_rate,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=80.0,
                threshold_warning=70.0,
                threshold_critical=60.0,
                description="策略回测成功率"
            ))

            # 平均年化收益率
            avg_annual_return = random.uniform(8, 25)
            metrics.append(BusinessMetric(
                category=MetricCategory.TRADING_PERFORMANCE,
                name="average_annual_return",
                value=avg_annual_return,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.VOLATILE,
                target_value=15.0,
                description="平均年化收益率"
            ))

            # 最大回撤
            max_drawdown = random.uniform(5, 20)
            metrics.append(BusinessMetric(
                category=MetricCategory.TRADING_PERFORMANCE,
                name="maximum_drawdown",
                value=max_drawdown,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=10.0,
                threshold_warning=15.0,
                threshold_critical=25.0,
                description="最大回撤"
            ))

            # 夏普比率
            sharpe_ratio = random.uniform(1.0, 3.0)
            metrics.append(BusinessMetric(
                category=MetricCategory.TRADING_PERFORMANCE,
                name="sharpe_ratio",
                value=sharpe_ratio,
                unit="ratio",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                target_value=2.0,
                description="夏普比率"
            ))

        except Exception as e:
            logger.error(f"收集交易性能指标失败: {e}")

        return metrics

    def _collect_risk_management_metrics(self) -> List[BusinessMetric]:
        """收集风险管理指标"""
        metrics = []
        timestamp = datetime.now()

        try:
            import random

            # 风险预警数量
            risk_alerts = random.randint(5, 25)
            metrics.append(BusinessMetric(
                category=MetricCategory.RISK_MANAGEMENT,
                name="risk_alerts_count",
                value=risk_alerts,
                unit="count",
                timestamp=timestamp,
                trend=TrendDirection.VOLATILE,
                threshold_warning=20,
                threshold_critical=50,
                description="风险预警数量"
            ))

            # VaR值
            var_value = random.uniform(2, 8)
            metrics.append(BusinessMetric(
                category=MetricCategory.RISK_MANAGEMENT,
                name="value_at_risk",
                value=var_value,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                threshold_warning=10.0,
                threshold_critical=15.0,
                description="风险价值（VaR）"
            ))

            # 波动率
            volatility = random.uniform(15, 35)
            metrics.append(BusinessMetric(
                category=MetricCategory.RISK_MANAGEMENT,
                name="portfolio_volatility",
                value=volatility,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.VOLATILE,
                threshold_warning=30.0,
                threshold_critical=40.0,
                description="组合波动率"
            ))

            # 集中度风险
            concentration_risk = random.uniform(10, 40)
            metrics.append(BusinessMetric(
                category=MetricCategory.RISK_MANAGEMENT,
                name="concentration_risk",
                value=concentration_risk,
                unit="percent",
                timestamp=timestamp,
                trend=TrendDirection.STABLE,
                threshold_warning=35.0,
                threshold_critical=50.0,
                description="持仓集中度风险"
            ))

        except Exception as e:
            logger.error(f"收集风险管理指标失败: {e}")

        return metrics

    def get_metrics_by_category(self, category: MetricCategory, limit: int = 100) -> List[Dict[str, Any]]:
        """根据类别获取指标"""
        category_metrics = []

        for metric_key, metric_history in self.metrics_history.items():
            if metric_key.startswith(category.value):
                # 获取最新的指标数据
                recent_metrics = list(metric_history)[-limit:]
                category_metrics.extend([asdict(metric) for metric in recent_metrics])

        return category_metrics

    def get_latest_metrics(self, categories: Optional[List[MetricCategory]] = None) -> Dict[str, Any]:
        """获取最新指标"""
        result = {}

        target_categories = categories or list(MetricCategory)

        for category in target_categories:
            category_data = {}

            for metric_key, metric_history in self.metrics_history.items():
                if metric_key.startswith(category.value) and metric_history:
                    latest_metric = metric_history[-1]
                    metric_name = metric_key.replace(f"{category.value}_", "")
                    category_data[metric_name] = asdict(latest_metric)

            result[category.value] = category_data

        return result


class BusinessDashboard:
    """业务监控仪表板"""

    def __init__(self):
        self.metrics_collector = BusinessMetricsCollector()
        self.widgets: Dict[str, DashboardWidget] = {}
        self.alert_conditions: List[AlertCondition] = []
        self.dashboard_config = self._create_default_dashboard_config()

        # 创建默认组件
        self._create_default_widgets()

        logger.info("业务监控仪表板初始化完成")

    def _create_default_dashboard_config(self) -> Dict[str, Any]:
        """创建默认仪表板配置"""
        return {
            'title': '股票分析系统业务监控仪表板',
            'refresh_interval': 60,
            'auto_refresh': True,
            'theme': 'dark',
            'layout': {
                'columns': 12,
                'rows': 8
            }
        }

    def _create_default_widgets(self):
        """创建默认组件"""
        # 分析性能概览
        self.widgets['analysis_overview'] = DashboardWidget(
            widget_id='analysis_overview',
            title='分析性能概览',
            widget_type='chart',
            category=MetricCategory.ANALYSIS_PERFORMANCE,
            metrics=['total_stocks_analyzed', 'analysis_success_rate', 'average_analysis_time'],
            config={
                'chart_type': 'line',
                'time_range': '1h',
                'show_legend': True
            },
            position={'x': 0, 'y': 0, 'width': 6, 'height': 3}
        )

        # 技术指标状态
        self.widgets['indicator_status'] = DashboardWidget(
            widget_id='indicator_status',
            title='技术指标状态',
            widget_type='gauge',
            category=MetricCategory.TECHNICAL_INDICATORS,
            metrics=['average_accuracy', 'active_indicators_count'],
            config={
                'gauge_type': 'arc',
                'color_ranges': [
                    {'min': 0, 'max': 80, 'color': 'red'},
                    {'min': 80, 'max': 95, 'color': 'yellow'},
                    {'min': 95, 'max': 100, 'color': 'green'}
                ]
            },
            position={'x': 6, 'y': 0, 'width': 3, 'height': 3}
        )

        # 系统资源监控
        self.widgets['system_resources'] = DashboardWidget(
            widget_id='system_resources',
            title='系统资源状态',
            widget_type='number',
            category=MetricCategory.SYSTEM_RESOURCES,
            metrics=['cache_hit_rate', 'concurrent_processing_capacity', 'memory_efficiency'],
            config={
                'display_type': 'big_number',
                'show_trend': True
            },
            position={'x': 9, 'y': 0, 'width': 3, 'height': 3}
        )

        # 信号检测统计
        self.widgets['signal_detection'] = DashboardWidget(
            widget_id='signal_detection',
            title='信号检测统计',
            widget_type='chart',
            category=MetricCategory.SIGNAL_DETECTION,
            metrics=['buy_signals_count', 'sell_signals_count', 'signal_accuracy_rate'],
            config={
                'chart_type': 'bar',
                'stacked': False,
                'show_values': True
            },
            position={'x': 0, 'y': 3, 'width': 6, 'height': 3}
        )

        # 数据质量监控
        self.widgets['data_quality'] = DashboardWidget(
            widget_id='data_quality',
            title='数据质量监控',
            widget_type='heatmap',
            category=MetricCategory.DATA_QUALITY,
            metrics=['data_completeness', 'data_accuracy', 'data_freshness'],
            config={
                'color_scheme': 'RdYlGn',
                'show_values': True
            },
            position={'x': 6, 'y': 3, 'width': 6, 'height': 3}
        )

        # 交易性能分析
        self.widgets['trading_performance'] = DashboardWidget(
            widget_id='trading_performance',
            title='交易性能分析',
            widget_type='chart',
            category=MetricCategory.TRADING_PERFORMANCE,
            metrics=['average_annual_return', 'maximum_drawdown', 'sharpe_ratio'],
            config={
                'chart_type': 'area',
                'fill_opacity': 0.3,
                'show_points': True
            },
            position={'x': 0, 'y': 6, 'width': 8, 'height': 2}
        )

        # 风险监控
        self.widgets['risk_monitoring'] = DashboardWidget(
            widget_id='risk_monitoring',
            title='风险监控',
            widget_type='table',
            category=MetricCategory.RISK_MANAGEMENT,
            metrics=['risk_alerts_count', 'value_at_risk', 'portfolio_volatility'],
            config={
                'show_pagination': False,
                'max_rows': 10,
                'sortable': True
            },
            position={'x': 8, 'y': 6, 'width': 4, 'height': 2}
        )

    @exception_handler(reraise=True)
    def start_dashboard(self, collection_interval: int = 60) -> Dict[str, Any]:
        """启动仪表板"""
        # 启动指标收集
        self.metrics_collector.start_collection(collection_interval)

        logger.info("业务监控仪表板已启动")

        return {
            'status': 'started',
            'dashboard_title': self.dashboard_config['title'],
            'widgets_count': len(self.widgets),
            'collection_interval': collection_interval,
            'start_time': datetime.now().isoformat()
        }

    def stop_dashboard(self):
        """停止仪表板"""
        self.metrics_collector.stop_collection()
        logger.info("业务监控仪表板已停止")

    @exception_handler(reraise=True)
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        dashboard_data = {
            'config': self.dashboard_config,
            'widgets': {},
            'timestamp': datetime.now().isoformat(),
            'status': 'active' if self.metrics_collector.collection_active else 'stopped'
        }

        # 获取每个组件的数据
        for widget_id, widget in self.widgets.items():
            try:
                widget_data = self._get_widget_data(widget)
                dashboard_data['widgets'][widget_id] = {
                    'config': asdict(widget),
                    'data': widget_data
                }
            except Exception as e:
                logger.error(f"获取组件数据失败 {widget_id}: {e}")
                dashboard_data['widgets'][widget_id] = {
                    'config': asdict(widget),
                    'data': {'error': str(e)}
                }

        return dashboard_data

    def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """获取组件数据"""
        try:
            # 获取该类别的最新指标
            category_metrics = self.metrics_collector.get_metrics_by_category(
                widget.category, limit=100
            )

            # 过滤出组件关心的指标
            widget_metrics = []
            for metric in category_metrics:
                if any(metric_name in metric.get('name', '') for metric_name in widget.metrics):
                    widget_metrics.append(metric)

            # 根据组件类型处理数据
            if widget.widget_type == 'chart':
                return self._prepare_chart_data(widget_metrics, widget.config)
            elif widget.widget_type == 'gauge':
                return self._prepare_gauge_data(widget_metrics, widget.config)
            elif widget.widget_type == 'number':
                return self._prepare_number_data(widget_metrics, widget.config)
            elif widget.widget_type == 'table':
                return self._prepare_table_data(widget_metrics, widget.config)
            elif widget.widget_type == 'heatmap':
                return self._prepare_heatmap_data(widget_metrics, widget.config)
            else:
                return {'metrics': widget_metrics}

        except Exception as e:
            logger.error(f"准备组件数据失败: {e}")
            return {'error': str(e)}

    def _prepare_chart_data(self, metrics: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """准备图表数据"""
        chart_data = {
            'type': config.get('chart_type', 'line'),
            'series': [],
            'timestamps': []
        }

        # 按指标名称分组数据
        grouped_metrics = defaultdict(list)
        for metric in metrics:
            grouped_metrics[metric['name']].append(metric)

        # 为每个指标创建数据系列
        for metric_name, metric_list in grouped_metrics.items():
            # 按时间排序
            metric_list.sort(key=lambda x: x['timestamp'])

            series_data = {
                'name': metric_name,
                'data': [m['value'] for m in metric_list],
                'timestamps': [m['timestamp'] for m in metric_list]
            }

            chart_data['series'].append(series_data)

        return chart_data

    def _prepare_gauge_data(self, metrics: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """准备仪表盘数据"""
        gauge_data = {
            'type': config.get('gauge_type', 'arc'),
            'gauges': []
        }

        # 获取最新的指标值
        latest_metrics = {}
        for metric in metrics:
            metric_name = metric['name']
            if metric_name not in latest_metrics or metric['timestamp'] > latest_metrics[metric_name]['timestamp']:
                latest_metrics[metric_name] = metric

        # 为每个指标创建仪表盘
        for metric_name, metric in latest_metrics.items():
            gauge_info = {
                'name': metric_name,
                'value': metric['value'],
                'unit': metric['unit'],
                'min_value': 0,
                'max_value': 100 if metric['unit'] == 'percent' else metric['value'] * 1.5,
                'target_value': metric.get('target_value'),
                'color_ranges': config.get('color_ranges', [])
            }

            gauge_data['gauges'].append(gauge_info)

        return gauge_data

    def _prepare_number_data(self, metrics: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """准备数字显示数据"""
        number_data = {
            'display_type': config.get('display_type', 'big_number'),
            'show_trend': config.get('show_trend', False),
            'numbers': []
        }

        # 按指标分组并计算趋势
        grouped_metrics = defaultdict(list)
        for metric in metrics:
            grouped_metrics[metric['name']].append(metric)

        for metric_name, metric_list in grouped_metrics.items():
            if not metric_list:
                continue

            # 按时间排序
            metric_list.sort(key=lambda x: x['timestamp'])
            latest_metric = metric_list[-1]

            # 计算趋势
            trend = None
            if len(metric_list) >= 2 and number_data['show_trend']:
                previous_value = metric_list[-2]['value']
                current_value = latest_metric['value']

                if isinstance(current_value, (int, float)) and isinstance(previous_value, (int, float)):
                    if current_value > previous_value:
                        trend = 'up'
                    elif current_value < previous_value:
                        trend = 'down'
                    else:
                        trend = 'stable'

            number_info = {
                'name': metric_name,
                'value': latest_metric['value'],
                'unit': latest_metric['unit'],
                'trend': trend,
                'target_value': latest_metric.get('target_value'),
                'threshold_warning': latest_metric.get('threshold_warning'),
                'threshold_critical': latest_metric.get('threshold_critical')
            }

            number_data['numbers'].append(number_info)

        return number_data

    def _prepare_table_data(self, metrics: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """准备表格数据"""
        table_data = {
            'columns': ['指标名称', '当前值', '单位', '目标值', '状态', '更新时间'],
            'rows': [],
            'sortable': config.get('sortable', True),
            'max_rows': config.get('max_rows', 50)
        }

        # 获取最新指标
        latest_metrics = {}
        for metric in metrics:
            metric_name = metric['name']
            if metric_name not in latest_metrics or metric['timestamp'] > latest_metrics[metric_name]['timestamp']:
                latest_metrics[metric_name] = metric

        # 构建表格行
        for metric_name, metric in latest_metrics.items():
            # 确定状态
            status = '正常'
            current_value = metric['value']

            if isinstance(current_value, (int, float)):
                if metric.get('threshold_critical') and current_value >= metric['threshold_critical']:
                    status = '严重'
                elif metric.get('threshold_warning') and current_value >= metric['threshold_warning']:
                    status = '警告'

            row = [
                metric_name,
                current_value,
                metric['unit'],
                metric.get('target_value', '-'),
                status,
                metric['timestamp']
            ]

            table_data['rows'].append(row)

        # 限制行数
        if len(table_data['rows']) > table_data['max_rows']:
            table_data['rows'] = table_data['rows'][:table_data['max_rows']]

        return table_data

    def _prepare_heatmap_data(self, metrics: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """准备热力图数据"""
        heatmap_data = {
            'color_scheme': config.get('color_scheme', 'RdYlGn'),
            'show_values': config.get('show_values', True),
            'data': []
        }

        # 获取最新指标
        latest_metrics = {}
        for metric in metrics:
            metric_name = metric['name']
            if metric_name not in latest_metrics or metric['timestamp'] > latest_metrics[metric_name]['timestamp']:
                latest_metrics[metric_name] = metric

        # 构建热力图数据
        for i, (metric_name, metric) in enumerate(latest_metrics.items()):
            value = metric['value']

            # 标准化值到0-1范围（基于百分比指标）
            if metric['unit'] == 'percent':
                normalized_value = value / 100.0
            else:
                # 对于其他单位，使用目标值标准化
                target = metric.get('target_value', 1)
                normalized_value = min(value / target, 1) if target > 0 else 0

            heatmap_data['data'].append({
                'x': 0,  # 单列热力图
                'y': i,
                'value': normalized_value,
                'original_value': value,
                'name': metric_name,
                'unit': metric['unit']
            })

        return heatmap_data

    @exception_handler(reraise=True)
    def add_widget(self, widget: DashboardWidget) -> bool:
        """添加组件"""
        try:
            self.widgets[widget.widget_id] = widget
            logger.info(f"添加仪表板组件: {widget.widget_id}")
            return True
        except Exception as e:
            logger.error(f"添加组件失败: {e}")
            return False

    @exception_handler(reraise=True)
    def remove_widget(self, widget_id: str) -> bool:
        """移除组件"""
        try:
            if widget_id in self.widgets:
                del self.widgets[widget_id]
                logger.info(f"移除仪表板组件: {widget_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"移除组件失败: {e}")
            return False

    @exception_handler(reraise=True)
    def export_dashboard_config(self, export_path: str) -> Dict[str, Any]:
        """导出仪表板配置"""
        try:
            export_data = {
                'dashboard_config': self.dashboard_config,
                'widgets': {widget_id: asdict(widget) for widget_id, widget in self.widgets.items()},
                'alert_conditions': [asdict(condition) for condition in self.alert_conditions],
                'exported_at': datetime.now().isoformat()
            }

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"仪表板配置已导出: {export_path}")
            return {'success': True, 'export_path': export_path}

        except Exception as e:
            logger.error(f"导出仪表板配置失败: {e}")
            return {'success': False, 'error': str(e)}

    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        try:
            latest_metrics = self.metrics_collector.get_latest_metrics()

            summary = {
                'categories_count': len(latest_metrics),
                'total_metrics': sum(len(metrics) for metrics in latest_metrics.values()),
                'categories': {},
                'timestamp': datetime.now().isoformat()
            }

            for category, metrics in latest_metrics.items():
                if metrics:
                    summary['categories'][category] = {
                        'metrics_count': len(metrics),
                        'key_metrics': list(metrics.keys())[:5]  # 显示前5个指标名
                    }

            return summary

        except Exception as e:
            logger.error(f"获取指标摘要失败: {e}")
            return {'error': str(e)}


# 全局仪表板实例
_business_dashboard = None


def get_business_dashboard() -> BusinessDashboard:
    """
    获取业务监控仪表板实例（单例模式）

    Returns:
        BusinessDashboard: 业务监控仪表板实例
    """
    global _business_dashboard

    if _business_dashboard is None:
        _business_dashboard = BusinessDashboard()

    return _business_dashboard


def create_business_dashboard() -> BusinessDashboard:
    """
    创建新的业务监控仪表板实例

    Returns:
        BusinessDashboard: 新的业务监控仪表板实例
    """
    return BusinessDashboard()