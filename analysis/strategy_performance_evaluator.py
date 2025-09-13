#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略性能评估框架核心模块

提供全面的策略性能评估功能，包括：
- 多维度性能指标计算（风险调整收益、最大回撤、夏普比率、信息比率、胜率、盈亏比等）
- 基准比较分析（相对于市场指数的超额收益）
- 风险分析模块（波动率分析、VaR计算、压力测试）
- 时间序列分析（滚动窗口性能、季节性分析）

设计目标：
- 指标计算准确度>99.95%
- 评估报告生成时间<30秒
- 支持1000+策略并行评估
- 内存使用<2GB
"""

import os
import sys
import time
import json
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from scipy import stats
import hashlib

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.dependency_injection import get_logger, get_service
from db.interfaces.data_access_interface import DataAccessInterface
from strategy.unified_base_strategy import UnifiedBaseStrategy, PeriodConfig
from utils.decorators import performance_monitor, exception_handler
from enums.kline_period import KlinePeriod

# 忽略pandas性能警告
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """策略性能指标"""

    # 收益指标
    total_return: float = 0.0          # 总收益率
    annualized_return: float = 0.0     # 年化收益率
    excess_return: float = 0.0         # 超额收益率
    alpha: float = 0.0                 # Alpha值
    beta: float = 0.0                  # Beta值

    # 风险指标
    volatility: float = 0.0            # 波动率（年化）
    max_drawdown: float = 0.0          # 最大回撤
    downside_deviation: float = 0.0    # 下行偏差
    var_95: float = 0.0                # 95% VaR
    cvar_95: float = 0.0               # 95% CVaR

    # 风险调整收益指标
    sharpe_ratio: float = 0.0          # 夏普比率
    sortino_ratio: float = 0.0         # 索提诺比率
    calmar_ratio: float = 0.0          # 卡玛比率
    information_ratio: float = 0.0     # 信息比率
    treynor_ratio: float = 0.0         # 特雷纳比率

    # 交易统计指标
    win_rate: float = 0.0              # 胜率
    profit_loss_ratio: float = 0.0     # 盈亏比
    total_trades: int = 0              # 总交易次数
    profitable_trades: int = 0         # 盈利交易次数
    losing_trades: int = 0             # 亏损交易次数
    avg_profit: float = 0.0            # 平均盈利
    avg_loss: float = 0.0              # 平均亏损

    # 时间统计指标
    evaluation_period_days: int = 0    # 评估期间天数
    active_trading_days: int = 0       # 活跃交易天数
    avg_holding_period: float = 0.0    # 平均持仓天数

    # 基准比较指标
    benchmark_total_return: float = 0.0      # 基准总收益率
    benchmark_volatility: float = 0.0        # 基准波动率
    tracking_error: float = 0.0              # 跟踪误差
    correlation_with_benchmark: float = 0.0   # 与基准相关性

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


@dataclass
class RiskMetrics:
    """风险分析指标"""

    # 波动率分析
    daily_volatility: float = 0.0     # 日波动率
    weekly_volatility: float = 0.0    # 周波动率
    monthly_volatility: float = 0.0   # 月波动率
    annual_volatility: float = 0.0    # 年化波动率

    # VaR分析
    var_1d_95: float = 0.0            # 1日95% VaR
    var_1d_99: float = 0.0            # 1日99% VaR
    var_1w_95: float = 0.0            # 1周95% VaR
    var_1w_99: float = 0.0            # 1周99% VaR

    # CVaR分析
    cvar_1d_95: float = 0.0           # 1日95% CVaR
    cvar_1d_99: float = 0.0           # 1日99% CVaR

    # 回撤分析
    max_drawdown: float = 0.0         # 最大回撤
    max_drawdown_duration: int = 0    # 最大回撤持续期
    current_drawdown: float = 0.0     # 当前回撤
    drawdown_frequency: float = 0.0   # 回撤频率
    avg_drawdown: float = 0.0         # 平均回撤

    # 压力测试
    stress_test_results: Dict[str, float] = None  # 压力测试结果

    def __post_init__(self):
        if self.stress_test_results is None:
            self.stress_test_results = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


@dataclass
class TimeSeriesAnalysis:
    """时间序列分析结果"""

    # 滚动窗口分析
    rolling_returns: pd.Series = None          # 滚动收益率
    rolling_sharpe: pd.Series = None           # 滚动夏普比率
    rolling_volatility: pd.Series = None       # 滚动波动率
    rolling_max_drawdown: pd.Series = None     # 滚动最大回撤

    # 季节性分析
    monthly_returns: Dict[int, float] = None   # 月度收益分布
    quarterly_returns: Dict[int, float] = None # 季度收益分布
    yearly_returns: Dict[int, float] = None    # 年度收益分布

    # 周期性分析
    period_performance: Dict[str, PerformanceMetrics] = None  # 不同周期性能

    # 相关性分析
    autocorrelation: Dict[int, float] = None   # 收益率自相关性
    partial_autocorr: Dict[int, float] = None  # 偏自相关性

    def __post_init__(self):
        if self.monthly_returns is None:
            self.monthly_returns = {}
        if self.quarterly_returns is None:
            self.quarterly_returns = {}
        if self.yearly_returns is None:
            self.yearly_returns = {}
        if self.period_performance is None:
            self.period_performance = {}
        if self.autocorrelation is None:
            self.autocorrelation = {}
        if self.partial_autocorr is None:
            self.partial_autocorr = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, pd.Series):
                result[key] = value.to_dict() if value is not None else None
            elif isinstance(value, dict) and key == 'period_performance':
                result[key] = {k: v.to_dict() for k, v in value.items()} if value else {}
            else:
                result[key] = value
        return result


@dataclass
class EvaluationConfig:
    """性能评估配置"""

    # 基本配置
    benchmark_code: str = "000001"              # 基准指数代码
    risk_free_rate: float = 0.03               # 无风险利率
    confidence_levels: List[float] = None       # 置信水平

    # 时间配置
    evaluation_period: int = 252                # 评估期间（交易日）
    rolling_window: int = 30                   # 滚动窗口大小
    min_periods: int = 20                      # 最小期间数

    # 性能配置
    parallel_workers: int = 8                   # 并行工作线程数
    chunk_size: int = 100                      # 数据处理块大小
    cache_enabled: bool = True                 # 是否启用缓存

    # 报告配置
    output_format: str = "json"                # 输出格式
    include_charts: bool = True                # 是否包含图表
    chart_resolution: Tuple[int, int] = (12, 8) # 图表分辨率

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99]


class StrategyPerformanceEvaluator:
    """
    策略性能评估器

    核心功能：
    1. 多维度性能指标计算
    2. 基准比较分析
    3. 风险分析和压力测试
    4. 时间序列分析
    5. 高性能并行计算
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        初始化性能评估器

        Args:
            config: 评估配置，如果为None则使用默认配置
        """
        self.config = config or EvaluationConfig()
        self.logger = get_logger(__name__)

        # 获取数据访问接口
        try:
            self.data_access = get_service(DataAccessInterface)
        except Exception as e:
            self.logger.warning(f"无法获取数据访问服务: {e}")
            self.data_access = None

        # 性能统计
        self.performance_stats = {
            'total_evaluations': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_efficiency': 0.0
        }

        # 缓存
        self.cache = {} if self.config.cache_enabled else None

        self.logger.info(f"策略性能评估器初始化完成，配置: {asdict(self.config)}")

    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def evaluate_strategy_performance(self,
                                    strategy_results: pd.DataFrame,
                                    strategy_name: str,
                                    benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        评估策略性能

        Args:
            strategy_results: 策略结果数据，包含日期、收益率等
            strategy_name: 策略名称
            benchmark_data: 基准数据，如果为None则自动获取

        Returns:
            Dict[str, Any]: 完整的性能评估结果
        """
        start_time = time.time()
        self.logger.info(f"开始评估策略 '{strategy_name}' 的性能")

        # 数据预处理和验证
        processed_data = self._preprocess_strategy_data(strategy_results)
        if processed_data.empty:
            raise ValueError("策略结果数据为空或格式不正确")

        # 获取基准数据
        if benchmark_data is None:
            benchmark_data = self._get_benchmark_data(
                processed_data.index[0], processed_data.index[-1]
            )

        # 计算基础性能指标
        performance_metrics = self._calculate_performance_metrics(
            processed_data, benchmark_data
        )

        # 计算风险指标
        risk_metrics = self._calculate_risk_metrics(processed_data)

        # 时间序列分析
        time_series_analysis = self._perform_time_series_analysis(processed_data)

        # 基准比较分析
        benchmark_comparison = self._perform_benchmark_analysis(
            processed_data, benchmark_data
        )

        # 压力测试
        stress_test_results = self._perform_stress_testing(processed_data)

        # 计算执行时间
        execution_time = time.time() - start_time

        # 构建完整结果
        evaluation_result = {
            'strategy_name': strategy_name,
            'evaluation_date': datetime.now().isoformat(),
            'evaluation_period': {
                'start_date': processed_data.index[0].strftime('%Y-%m-%d'),
                'end_date': processed_data.index[-1].strftime('%Y-%m-%d'),
                'total_days': len(processed_data),
                'trading_days': len(processed_data[processed_data['returns'].notna()])
            },
            'performance_metrics': performance_metrics.to_dict(),
            'risk_metrics': risk_metrics.to_dict(),
            'time_series_analysis': time_series_analysis.to_dict(),
            'benchmark_comparison': benchmark_comparison,
            'stress_test_results': stress_test_results,
            'execution_metrics': {
                'execution_time_seconds': execution_time,
                'data_points_processed': len(processed_data),
                'processing_rate': len(processed_data) / execution_time if execution_time > 0 else 0
            }
        }

        # 更新性能统计
        self._update_performance_stats(execution_time)

        self.logger.info(f"策略 '{strategy_name}' 性能评估完成，耗时: {execution_time:.2f}秒")

        return evaluation_result

    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def evaluate_multiple_strategies(self,
                                   strategies_data: Dict[str, pd.DataFrame],
                                   benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        批量评估多个策略性能

        Args:
            strategies_data: 多个策略的结果数据 {策略名称: 策略数据}
            benchmark_data: 基准数据

        Returns:
            Dict[str, Any]: 多策略性能评估结果
        """
        start_time = time.time()
        self.logger.info(f"开始批量评估 {len(strategies_data)} 个策略的性能")

        # 验证输入
        if not strategies_data:
            raise ValueError("策略数据不能为空")

        # 获取基准数据（如果没有提供）
        if benchmark_data is None:
            # 从所有策略中确定时间范围
            all_start_dates = []
            all_end_dates = []

            for strategy_data in strategies_data.values():
                if not strategy_data.empty:
                    all_start_dates.append(strategy_data.index[0])
                    all_end_dates.append(strategy_data.index[-1])

            if all_start_dates and all_end_dates:
                benchmark_data = self._get_benchmark_data(
                    min(all_start_dates), max(all_end_dates)
                )

        # 并行评估策略
        strategy_results = {}

        if self.config.parallel_workers > 1:
            # 多线程并行处理
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                future_to_strategy = {
                    executor.submit(
                        self.evaluate_strategy_performance,
                        strategy_data,
                        strategy_name,
                        benchmark_data
                    ): strategy_name
                    for strategy_name, strategy_data in strategies_data.items()
                }

                for future in as_completed(future_to_strategy):
                    strategy_name = future_to_strategy[future]
                    try:
                        result = future.result()
                        strategy_results[strategy_name] = result
                        self.logger.debug(f"✅ 策略 '{strategy_name}' 评估完成")
                    except Exception as e:
                        self.logger.error(f"❌ 策略 '{strategy_name}' 评估失败: {e}")
                        strategy_results[strategy_name] = {'error': str(e)}
        else:
            # 单线程顺序处理
            for strategy_name, strategy_data in strategies_data.items():
                try:
                    result = self.evaluate_strategy_performance(
                        strategy_data, strategy_name, benchmark_data
                    )
                    strategy_results[strategy_name] = result
                    self.logger.debug(f"✅ 策略 '{strategy_name}' 评估完成")
                except Exception as e:
                    self.logger.error(f"❌ 策略 '{strategy_name}' 评估失败: {e}")
                    strategy_results[strategy_name] = {'error': str(e)}

        # 计算综合统计
        comprehensive_stats = self._calculate_comprehensive_statistics(strategy_results)

        # 策略排名
        strategy_rankings = self._rank_strategies(strategy_results)

        # 相关性分析
        correlation_analysis = self._analyze_strategy_correlations(strategies_data)

        execution_time = time.time() - start_time

        # 构建最终结果
        final_result = {
            'evaluation_summary': {
                'total_strategies': len(strategies_data),
                'successful_evaluations': len([r for r in strategy_results.values() if 'error' not in r]),
                'evaluation_date': datetime.now().isoformat(),
                'total_execution_time': execution_time,
                'parallel_efficiency': self._calculate_parallel_efficiency(execution_time, len(strategies_data))
            },
            'individual_results': strategy_results,
            'comprehensive_statistics': comprehensive_stats,
            'strategy_rankings': strategy_rankings,
            'correlation_analysis': correlation_analysis,
            'benchmark_info': self._get_benchmark_info(benchmark_data)
        }

        self.logger.info(f"批量策略评估完成，总耗时: {execution_time:.2f}秒")

        return final_result

    def _preprocess_strategy_data(self, strategy_data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理策略数据

        Args:
            strategy_data: 原始策略数据

        Returns:
            pd.DataFrame: 处理后的数据
        """
        try:
            # 复制数据避免修改原始数据
            data = strategy_data.copy()

            # 确保索引是日期类型
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'date' in data.columns:
                    data.set_index('date', inplace=True)
                    data.index = pd.to_datetime(data.index)
                else:
                    raise ValueError("数据必须包含日期索引或日期列")

            # 确保包含必要的列
            required_columns = ['returns']
            if not all(col in data.columns for col in required_columns):
                # 尝试从价格数据计算收益率
                if 'close' in data.columns or 'price' in data.columns:
                    price_col = 'close' if 'close' in data.columns else 'price'
                    data['returns'] = data[price_col].pct_change()
                else:
                    raise ValueError("数据必须包含收益率列或价格列")

            # 处理缺失值
            data['returns'].fillna(0, inplace=True)

            # 计算累计收益
            if 'cumulative_returns' not in data.columns:
                data['cumulative_returns'] = (1 + data['returns']).cumprod()

            # 按日期排序
            data.sort_index(inplace=True)

            # 移除异常值（收益率超过100%或小于-100%的数据点）
            extreme_returns = (data['returns'] > 1.0) | (data['returns'] < -1.0)
            if extreme_returns.any():
                self.logger.warning(f"发现 {extreme_returns.sum()} 个异常收益率数据点，已处理")
                data.loc[extreme_returns, 'returns'] = np.nan
                data['returns'].fillna(method='ffill', inplace=True)
                data['returns'].fillna(0, inplace=True)
                # 重新计算累计收益
                data['cumulative_returns'] = (1 + data['returns']).cumprod()

            return data

        except Exception as e:
            self.logger.error(f"数据预处理失败: {e}")
            return pd.DataFrame()

    def _get_benchmark_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        获取基准数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.DataFrame: 基准数据
        """
        try:
            if self.data_access is None:
                # 数据纯净化：必须使用真实基准数据
                error_msg = "数据纯净化要求：无法获取真实基准数据，不允许使用模拟数据"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # 从ClickHouse获取基准数据
            query = f"""
            SELECT date, close
            FROM stock_info
            WHERE code = '{self.config.benchmark_code}'
            AND level = '日线'
            AND date >= '{start_date.strftime('%Y-%m-%d')}'
            AND date <= '{end_date.strftime('%Y-%m-%d')}'
            ORDER BY date ASC
            """

            benchmark_data = self.data_access.query_dataframe(query)

            if benchmark_data.empty:
                error_msg = f"数据纯净化要求：未获取到基准代码 {self.config.benchmark_code} 的真实数据，不允许使用模拟数据"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # 处理基准数据
            benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
            benchmark_data.set_index('date', inplace=True)
            benchmark_data['returns'] = benchmark_data['close'].pct_change()
            benchmark_data['returns'].fillna(0, inplace=True)
            benchmark_data['cumulative_returns'] = (1 + benchmark_data['returns']).cumprod()

            return benchmark_data

        except Exception as e:
            self.logger.error(f"获取基准数据失败: {e}")
            # 数据纯净化：不允许生成模拟基准数据，必须使用真实数据
            error_msg = f"数据纯净化要求：获取基准数据失败，不允许使用模拟数据。原因: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _calculate_performance_metrics(self,
                                     strategy_data: pd.DataFrame,
                                     benchmark_data: pd.DataFrame) -> PerformanceMetrics:
        """
        计算策略性能指标

        Args:
            strategy_data: 策略数据
            benchmark_data: 基准数据

        Returns:
            PerformanceMetrics: 性能指标
        """
        from analysis.performance_metrics_calculator import performance_calculator

        # 提取收益率数据
        strategy_returns = strategy_data['returns']
        benchmark_returns = benchmark_data['returns'] if not benchmark_data.empty else None

        # 使用性能计算器计算指标
        return performance_calculator.calculate_performance_metrics(
            returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            prices=strategy_data.get('cumulative_returns')
        )

    def _calculate_risk_metrics(self, strategy_data: pd.DataFrame) -> RiskMetrics:
        """
        计算风险指标

        Args:
            strategy_data: 策略数据

        Returns:
            RiskMetrics: 风险指标
        """
        from analysis.performance_metrics_calculator import performance_calculator

        return performance_calculator.calculate_risk_metrics(
            returns=strategy_data['returns'],
            prices=strategy_data.get('cumulative_returns')
        )

    def _perform_time_series_analysis(self, strategy_data: pd.DataFrame) -> TimeSeriesAnalysis:
        """
        执行时间序列分析

        Args:
            strategy_data: 策略数据

        Returns:
            TimeSeriesAnalysis: 时间序列分析结果
        """
        from analysis.performance_metrics_calculator import performance_calculator

        return performance_calculator.perform_time_series_analysis(
            returns=strategy_data['returns'],
            rolling_window=self.config.rolling_window
        )

    def _perform_benchmark_analysis(self,
                                  strategy_data: pd.DataFrame,
                                  benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """
        执行基准比较分析

        Args:
            strategy_data: 策略数据
            benchmark_data: 基准数据

        Returns:
            Dict[str, Any]: 基准比较分析结果
        """
        if benchmark_data.empty:
            return {'error': '无基准数据'}

        try:
            # 对齐数据
            aligned_data = self._align_data_for_comparison(strategy_data, benchmark_data)

            if aligned_data.empty:
                return {'error': '无法对齐策略和基准数据'}

            strategy_returns = aligned_data['strategy_returns']
            benchmark_returns = aligned_data['benchmark_returns']

            # 计算比较指标
            comparison_metrics = {
                'correlation': strategy_returns.corr(benchmark_returns),
                'beta': self._calculate_beta(strategy_returns, benchmark_returns),
                'alpha': self._calculate_alpha(strategy_returns, benchmark_returns),
                'tracking_error': (strategy_returns - benchmark_returns).std() * np.sqrt(self.config.evaluation_period),
                'information_ratio': self._calculate_information_ratio(strategy_returns, benchmark_returns),
                'excess_return_volatility': (strategy_returns - benchmark_returns).std(),
                'up_capture': self._calculate_up_capture_ratio(strategy_returns, benchmark_returns),
                'down_capture': self._calculate_down_capture_ratio(strategy_returns, benchmark_returns)
            }

            # 相对性能分析
            relative_performance = self._analyze_relative_performance(
                aligned_data['strategy_cumulative'],
                aligned_data['benchmark_cumulative']
            )

            return {
                'comparison_metrics': comparison_metrics,
                'relative_performance': relative_performance,
                'period_analysis': self._analyze_period_performance(strategy_returns, benchmark_returns)
            }

        except Exception as e:
            self.logger.error(f"基准比较分析失败: {e}")
            return {'error': str(e)}

    def _perform_stress_testing(self, strategy_data: pd.DataFrame) -> Dict[str, Any]:
        """
        执行压力测试

        Args:
            strategy_data: 策略数据

        Returns:
            Dict[str, Any]: 压力测试结果
        """
        returns = strategy_data['returns']
        stress_results = {}

        try:
            # 历史模拟法压力测试
            stress_results['historical_simulation'] = self._historical_simulation_stress_test(returns)

            # 蒙特卡罗模拟压力测试
            stress_results['monte_carlo_simulation'] = self._monte_carlo_stress_test(returns)

            # 场景分析
            stress_results['scenario_analysis'] = self._scenario_analysis(returns)

            # 极端事件分析
            stress_results['extreme_events'] = self._analyze_extreme_events(returns)

        except Exception as e:
            self.logger.error(f"压力测试失败: {e}")
            stress_results['error'] = str(e)

        return stress_results

    def _calculate_comprehensive_statistics(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算综合统计信息

        Args:
            strategy_results: 多个策略的评估结果

        Returns:
            Dict[str, Any]: 综合统计信息
        """
        successful_results = {k: v for k, v in strategy_results.items() if 'error' not in v}

        if not successful_results:
            return {'error': '没有成功的策略评估结果'}

        # 提取性能指标
        performance_metrics = {}
        for strategy_name, result in successful_results.items():
            if 'performance_metrics' in result:
                performance_metrics[strategy_name] = result['performance_metrics']

        if not performance_metrics:
            return {'error': '无法提取性能指标'}

        # 计算统计信息
        stats = {}

        # 收益率统计
        total_returns = [metrics['total_return'] for metrics in performance_metrics.values()]
        annualized_returns = [metrics['annualized_return'] for metrics in performance_metrics.values()]

        stats['return_statistics'] = {
            'mean_total_return': np.mean(total_returns),
            'median_total_return': np.median(total_returns),
            'std_total_return': np.std(total_returns),
            'min_total_return': np.min(total_returns),
            'max_total_return': np.max(total_returns),
            'mean_annualized_return': np.mean(annualized_returns),
            'median_annualized_return': np.median(annualized_returns)
        }

        # 风险统计
        volatilities = [metrics['volatility'] for metrics in performance_metrics.values()]
        max_drawdowns = [metrics['max_drawdown'] for metrics in performance_metrics.values()]

        stats['risk_statistics'] = {
            'mean_volatility': np.mean(volatilities),
            'median_volatility': np.median(volatilities),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'median_max_drawdown': np.median(max_drawdowns)
        }

        # 风险调整收益统计
        sharpe_ratios = [metrics['sharpe_ratio'] for metrics in performance_metrics.values()]
        sortino_ratios = [metrics['sortino_ratio'] for metrics in performance_metrics.values()]

        stats['risk_adjusted_statistics'] = {
            'mean_sharpe_ratio': np.mean(sharpe_ratios),
            'median_sharpe_ratio': np.median(sharpe_ratios),
            'mean_sortino_ratio': np.mean(sortino_ratios),
            'median_sortino_ratio': np.median(sortino_ratios)
        }

        return stats

    def _rank_strategies(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        对策略进行排名

        Args:
            strategy_results: 多个策略的评估结果

        Returns:
            Dict[str, Any]: 策略排名结果
        """
        successful_results = {k: v for k, v in strategy_results.items() if 'error' not in v}

        if not successful_results:
            return {'error': '没有成功的策略评估结果'}

        rankings = {}

        # 按不同指标排名
        ranking_criteria = [
            ('total_return', '总收益率', False),  # False表示降序
            ('annualized_return', '年化收益率', False),
            ('sharpe_ratio', '夏普比率', False),
            ('sortino_ratio', '索提诺比率', False),
            ('max_drawdown', '最大回撤', True),   # True表示升序（越小越好）
            ('volatility', '波动率', True),
            ('calmar_ratio', '卡玛比率', False)
        ]

        for metric, name, ascending in ranking_criteria:
            strategy_values = []
            for strategy_name, result in successful_results.items():
                if 'performance_metrics' in result and metric in result['performance_metrics']:
                    value = result['performance_metrics'][metric]
                    strategy_values.append((strategy_name, value))

            if strategy_values:
                # 按指标值排序
                sorted_strategies = sorted(strategy_values, key=lambda x: x[1], reverse=not ascending)
                rankings[f'{metric}_ranking'] = {
                    'metric_name': name,
                    'ranking': [{'rank': i+1, 'strategy': name, 'value': value}
                               for i, (name, value) in enumerate(sorted_strategies)]
                }

        # 综合排名（基于多个指标的加权平均）
        composite_scores = self._calculate_composite_scores(successful_results)
        if composite_scores:
            sorted_composite = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
            rankings['composite_ranking'] = {
                'metric_name': '综合评分',
                'ranking': [{'rank': i+1, 'strategy': name, 'value': score}
                           for i, (name, score) in enumerate(sorted_composite)]
            }

        return rankings

    def _analyze_strategy_correlations(self, strategies_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        分析策略间的相关性

        Args:
            strategies_data: 多个策略的数据

        Returns:
            Dict[str, Any]: 相关性分析结果
        """
        try:
            # 提取所有策略的收益率
            returns_data = {}
            for strategy_name, strategy_df in strategies_data.items():
                if not strategy_df.empty and 'returns' in strategy_df.columns:
                    returns_data[strategy_name] = strategy_df['returns']

            if len(returns_data) < 2:
                return {'error': '至少需要2个策略才能进行相关性分析'}

            # 创建收益率矩阵
            returns_matrix = pd.DataFrame(returns_data).dropna()

            if returns_matrix.empty:
                return {'error': '没有足够的重叠数据进行相关性分析'}

            # 计算相关性矩阵
            correlation_matrix = returns_matrix.corr()

            # 寻找高相关性策略对
            high_correlation_pairs = []
            for i, strategy1 in enumerate(correlation_matrix.columns):
                for j, strategy2 in enumerate(correlation_matrix.columns):
                    if i < j:  # 避免重复
                        corr_value = correlation_matrix.loc[strategy1, strategy2]
                        if abs(corr_value) > 0.7:  # 高相关性阈值
                            high_correlation_pairs.append({
                                'strategy1': strategy1,
                                'strategy2': strategy2,
                                'correlation': corr_value
                            })

            # 计算平均相关性
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            mean_correlation = upper_triangle.stack().mean()

            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'high_correlation_pairs': high_correlation_pairs,
                'mean_correlation': mean_correlation,
                'diversification_ratio': self._calculate_diversification_ratio(returns_matrix)
            }

        except Exception as e:
            self.logger.error(f"相关性分析失败: {e}")
            return {'error': str(e)}

    def _get_benchmark_info(self, benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取基准信息

        Args:
            benchmark_data: 基准数据

        Returns:
            Dict[str, Any]: 基准信息
        """
        if benchmark_data.empty:
            return {'error': '无基准数据'}

        try:
            return {
                'benchmark_code': self.config.benchmark_code,
                'period': {
                    'start_date': benchmark_data.index[0].strftime('%Y-%m-%d'),
                    'end_date': benchmark_data.index[-1].strftime('%Y-%m-%d'),
                    'total_days': len(benchmark_data)
                },
                'performance': {
                    'total_return': (benchmark_data['cumulative_returns'].iloc[-1] - 1) if 'cumulative_returns' in benchmark_data.columns else 0,
                    'volatility': benchmark_data['returns'].std() * np.sqrt(252) if 'returns' in benchmark_data.columns else 0,
                    'max_drawdown': self._calculate_max_drawdown_from_returns(benchmark_data['returns']) if 'returns' in benchmark_data.columns else 0
                }
            }

        except Exception as e:
            return {'error': str(e)}

    # Helper methods for calculations

    def _align_data_for_comparison(self,
                                 strategy_data: pd.DataFrame,
                                 benchmark_data: pd.DataFrame) -> pd.DataFrame:
        """对齐策略和基准数据用于比较"""
        try:
            aligned = pd.DataFrame({
                'strategy_returns': strategy_data['returns'],
                'benchmark_returns': benchmark_data['returns'],
                'strategy_cumulative': strategy_data['cumulative_returns'],
                'benchmark_cumulative': benchmark_data['cumulative_returns']
            }).dropna()

            return aligned
        except Exception:
            return pd.DataFrame()

    def _calculate_beta(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算Beta值"""
        if benchmark_returns.var() == 0:
            return 0.0
        return strategy_returns.cov(benchmark_returns) / benchmark_returns.var()

    def _calculate_alpha(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算Alpha值"""
        beta = self._calculate_beta(strategy_returns, benchmark_returns)
        strategy_mean = strategy_returns.mean() * 252
        benchmark_mean = benchmark_returns.mean() * 252
        return strategy_mean - (self.config.risk_free_rate + beta * (benchmark_mean - self.config.risk_free_rate))

    def _calculate_information_ratio(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算信息比率"""
        excess_returns = strategy_returns - benchmark_returns
        if excess_returns.std() == 0:
            return 0.0
        return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))

    def _calculate_up_capture_ratio(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算上涨捕获比率"""
        up_benchmark = benchmark_returns[benchmark_returns > 0]
        up_strategy = strategy_returns[benchmark_returns > 0]

        if up_benchmark.empty or up_benchmark.mean() == 0:
            return 0.0

        return up_strategy.mean() / up_benchmark.mean()

    def _calculate_down_capture_ratio(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算下跌捕获比率"""
        down_benchmark = benchmark_returns[benchmark_returns < 0]
        down_strategy = strategy_returns[benchmark_returns < 0]

        if down_benchmark.empty or down_benchmark.mean() == 0:
            return 0.0

        return down_strategy.mean() / down_benchmark.mean()

    def _analyze_relative_performance(self,
                                    strategy_cumulative: pd.Series,
                                    benchmark_cumulative: pd.Series) -> Dict[str, Any]:
        """分析相对性能"""
        relative_return = strategy_cumulative / benchmark_cumulative - 1

        return {
            'relative_total_return': relative_return.iloc[-1],
            'relative_volatility': relative_return.diff().std() * np.sqrt(252),
            'periods_outperformed': (strategy_cumulative.pct_change() > benchmark_cumulative.pct_change()).sum(),
            'total_periods': len(strategy_cumulative),
            'outperformance_ratio': (strategy_cumulative.pct_change() > benchmark_cumulative.pct_change()).mean()
        }

    def _analyze_period_performance(self,
                                  strategy_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> Dict[str, Any]:
        """分析分期间性能"""
        if not isinstance(strategy_returns.index, pd.DatetimeIndex):
            return {}

        period_analysis = {}

        # 年度性能
        yearly_strategy = strategy_returns.groupby(strategy_returns.index.year).apply(lambda x: (1 + x).prod() - 1)
        yearly_benchmark = benchmark_returns.groupby(benchmark_returns.index.year).apply(lambda x: (1 + x).prod() - 1)

        period_analysis['yearly'] = {
            'strategy': yearly_strategy.to_dict(),
            'benchmark': yearly_benchmark.to_dict(),
            'outperformance': (yearly_strategy - yearly_benchmark).to_dict()
        }

        # 季度性能
        quarterly_strategy = strategy_returns.groupby([strategy_returns.index.year, strategy_returns.index.quarter]).apply(
            lambda x: (1 + x).prod() - 1
        )
        quarterly_benchmark = benchmark_returns.groupby([benchmark_returns.index.year, benchmark_returns.index.quarter]).apply(
            lambda x: (1 + x).prod() - 1
        )

        period_analysis['quarterly'] = {
            'periods_count': len(quarterly_strategy),
            'avg_outperformance': (quarterly_strategy - quarterly_benchmark).mean()
        }

        return period_analysis

    def _historical_simulation_stress_test(self, returns: pd.Series) -> Dict[str, float]:
        """历史模拟法压力测试"""
        if returns.empty:
            return {}

        # 使用历史数据中的最差情况
        worst_day = returns.min()
        worst_week = returns.rolling(5).sum().min()
        worst_month = returns.rolling(21).sum().min()

        return {
            'worst_1_day': worst_day,
            'worst_1_week': worst_week,
            'worst_1_month': worst_month,
            'worst_1_day_percentile': stats.percentileofscore(returns, worst_day),
            'worst_week_percentile': stats.percentileofscore(returns.rolling(5).sum().dropna(), worst_week)
        }

    def _monte_carlo_stress_test(self, returns: pd.Series, num_simulations: int = 1000) -> Dict[str, float]:
        """蒙特卡罗压力测试"""
        if returns.empty:
            return {}

        # 估算收益率分布参数
        mean_return = returns.mean()
        std_return = returns.std()

        # 蒙特卡罗模拟
        np.random.seed(42)  # 确保可重复性
        simulated_returns = np.random.normal(mean_return, std_return, (num_simulations, 21))  # 21天的模拟

        # 计算模拟的月度收益
        monthly_returns = np.sum(simulated_returns, axis=1)

        return {
            'monte_carlo_var_95': np.percentile(monthly_returns, 5),
            'monte_carlo_var_99': np.percentile(monthly_returns, 1),
            'monte_carlo_expected_return': np.mean(monthly_returns),
            'monte_carlo_worst_case': np.min(monthly_returns),
            'monte_carlo_best_case': np.max(monthly_returns)
        }

    def _scenario_analysis(self, returns: pd.Series) -> Dict[str, float]:
        """场景分析"""
        if returns.empty:
            return {}

        current_volatility = returns.std()

        scenarios = {
            'bull_market': returns.mean() + 0.5 * current_volatility,
            'bear_market': returns.mean() - 1.5 * current_volatility,
            'high_volatility': returns.mean() * (1 + 2 * current_volatility),
            'market_crash': -0.20,  # 假设20%的市场崩盘
            'normal_market': returns.mean()
        }

        return scenarios

    def _analyze_extreme_events(self, returns: pd.Series) -> Dict[str, Any]:
        """分析极端事件"""
        if returns.empty:
            return {}

        # 定义极端事件（超过2个标准差）
        std_threshold = 2.0 * returns.std()
        extreme_positive = returns[returns > std_threshold]
        extreme_negative = returns[returns < -std_threshold]

        return {
            'extreme_positive_events': len(extreme_positive),
            'extreme_negative_events': len(extreme_negative),
            'extreme_positive_avg': extreme_positive.mean() if not extreme_positive.empty else 0,
            'extreme_negative_avg': extreme_negative.mean() if not extreme_negative.empty else 0,
            'total_extreme_events': len(extreme_positive) + len(extreme_negative),
            'extreme_event_frequency': (len(extreme_positive) + len(extreme_negative)) / len(returns)
        }

    def _calculate_composite_scores(self, strategy_results: Dict[str, Any]) -> Dict[str, float]:
        """计算综合评分"""
        composite_scores = {}

        # 定义权重（可以根据需要调整）
        weights = {
            'total_return': 0.25,
            'sharpe_ratio': 0.30,
            'max_drawdown': -0.20,  # 负权重，因为回撤越小越好
            'sortino_ratio': 0.15,
            'volatility': -0.10  # 负权重，因为波动率越小越好
        }

        for strategy_name, result in strategy_results.items():
            if 'performance_metrics' in result:
                metrics = result['performance_metrics']
                score = 0.0

                for metric, weight in weights.items():
                    if metric in metrics:
                        # 标准化指标值（简单的min-max标准化）
                        value = metrics[metric]
                        if weight > 0:
                            score += weight * max(0, value)  # 确保正值
                        else:
                            score += weight * min(0, -abs(value))  # 负值处理

                composite_scores[strategy_name] = score

        return composite_scores

    def _calculate_diversification_ratio(self, returns_matrix: pd.DataFrame) -> float:
        """计算分散化比率"""
        if returns_matrix.empty or returns_matrix.shape[1] < 2:
            return 0.0

        # 等权重组合的波动率
        equal_weight_portfolio_vol = (returns_matrix.mean(axis=1)).std()

        # 各策略波动率的加权平均
        individual_vols = returns_matrix.std()
        avg_individual_vol = individual_vols.mean()

        if avg_individual_vol == 0:
            return 0.0

        return equal_weight_portfolio_vol / avg_individual_vol

    def _calculate_max_drawdown_from_returns(self, returns: pd.Series) -> float:
        """从收益率计算最大回撤"""
        if returns.empty:
            return 0.0

        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max

        return abs(drawdown.min())

    def _update_performance_stats(self, execution_time: float):
        """更新性能统计"""
        self.performance_stats['total_evaluations'] += 1
        self.performance_stats['total_time'] += execution_time

    def _calculate_parallel_efficiency(self, total_time: float, num_strategies: int) -> float:
        """计算并行处理效率"""
        if num_strategies <= 1 or total_time <= 0:
            return 1.0

        # 估算单线程时间（基于单个策略的平均时间）
        avg_single_strategy_time = self.performance_stats['total_time'] / max(1, self.performance_stats['total_evaluations'])
        estimated_sequential_time = avg_single_strategy_time * num_strategies

        return min(estimated_sequential_time / total_time, self.config.parallel_workers)

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'evaluator_performance': self.performance_stats,
            'configuration': asdict(self.config),
            'cache_status': {
                'enabled': self.config.cache_enabled,
                'size': len(self.cache) if self.cache else 0,
                'hit_rate': (self.performance_stats['cache_hits'] /
                           max(1, self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']))
            },
            'memory_usage': self._get_memory_usage(),
            'system_info': self._get_system_info()
        }

    def _get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}

    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            import psutil

            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(),
                'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'total_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024
            }
        except ImportError:
            return {'error': 'psutil not available'}