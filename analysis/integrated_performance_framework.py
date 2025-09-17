#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略性能评估框架集成模块

提供统一的策略性能评估接口，集成以下功能：
- 高性能向量化计算
- 并行处理优化
- ClickHouse数据集成
- 缓存机制
- 完整的评估工作流

这是策略性能评估框架的主要入口点，设计目标：
- 支持1000+策略并行评估
- 评估报告生成时间<30秒
- 内存使用<2GB
- 指标计算准确度>99.95%
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger, get_service
from utils.decorators import performance_monitor, exception_handler
from db.interfaces.data_access_interface import DataAccessInterface
from analysis.strategy_performance_evaluator import (
    StrategyPerformanceEvaluator,
    EvaluationConfig,
    PerformanceMetrics,
    RiskMetrics,
    TimeSeriesAnalysis
)
from analysis.performance_metrics_calculator import performance_calculator
from analysis.performance_report_generator import PerformanceReportGenerator
from strategy.unified_base_strategy import UnifiedBaseStrategy
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


@dataclass
class PerformanceEvaluationRequest:
    """性能评估请求"""
    strategy_name: str
    strategy_data: pd.DataFrame
    benchmark_code: Optional[str] = None
    evaluation_period_days: int = 252
    include_stress_testing: bool = True
    include_time_series_analysis: bool = True
    output_formats: List[str] = None

    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['json', 'html']


@dataclass
class BatchEvaluationRequest:
    """批量评估请求"""
    strategies_data: Dict[str, pd.DataFrame]
    benchmark_code: Optional[str] = None
    evaluation_period_days: int = 252
    parallel_workers: int = 8
    chunk_size: int = 100
    output_formats: List[str] = None

    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['json', 'html']


class PerformanceEvaluationFramework:
    """
    策略性能评估框架主类

    提供完整的策略性能评估工作流：
    1. 数据预处理和验证
    2. 性能指标计算
    3. 风险分析和压力测试
    4. 基准比较分析
    5. 报告生成和可视化
    6. 缓存和优化
    """

    def __init__(self,
                 config: Optional[EvaluationConfig] = None,
                 cache_dir: str = "./cache/performance_evaluation",
                 output_dir: str = "./reports/performance_evaluation"):
        """
        初始化性能评估框架

        Args:
            config: 评估配置
            cache_dir: 缓存目录
            output_dir: 输出目录
        """
        self.config = config or EvaluationConfig()
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)

        # 创建目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.evaluator = StrategyPerformanceEvaluator(self.config)
        self.report_generator = PerformanceReportGenerator(str(self.output_dir))

        # 获取数据访问服务
        try:
            self.data_access = get_service(DataAccessInterface)
        except Exception as e:
            logger.warning(f"无法获取数据访问服务: {e}")
            self.data_access = None

        # 缓存管理
        self.cache = {}
        self.cache_enabled = True

        # 性能统计
        self.performance_stats = {
            'total_evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0,
            'parallel_efficiency': 0.0
        }

        logger.info(f"策略性能评估框架初始化完成")
        logger.info(f"缓存目录: {self.cache_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"配置: {asdict(self.config)}")

    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def evaluate_single_strategy(self, request: PerformanceEvaluationRequest) -> Dict[str, Any]:
        """
        评估单个策略性能

        Args:
            request: 评估请求

        Returns:
            Dict[str, Any]: 完整的评估结果，包含文件路径
        """
        start_time = time.time()

        logger.info(f"开始评估策略: {request.strategy_name}")

        try:
            # 1. 数据预处理和验证
            processed_data = self._preprocess_strategy_data(
                request.strategy_data,
                request.strategy_name
            )

            # 2. 检查缓存
            cache_key = self._generate_cache_key(request, processed_data)
            if self.cache_enabled:
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    logger.info(f"从缓存获取评估结果: {request.strategy_name}")
                    self.performance_stats['cache_hits'] += 1
                    return cached_result

            self.performance_stats['cache_misses'] += 1

            # 3. 获取基准数据
            benchmark_data = self._get_benchmark_data(
                processed_data.index[0],
                processed_data.index[-1],
                request.benchmark_code
            )

            # 4. 执行性能评估
            evaluation_result = self.evaluator.evaluate_strategy_performance(
                strategy_results=processed_data,
                strategy_name=request.strategy_name,
                benchmark_data=benchmark_data
            )

            # 5. 生成报告
            report_files = {}
            for output_format in request.output_formats:
                try:
                    file_path = self.report_generator.generate_single_strategy_report(
                        evaluation_result=evaluation_result,
                        output_format=output_format,
                        report_name=f"{request.strategy_name}_performance_report"
                    )
                    report_files[output_format] = file_path
                    logger.info(f"生成 {output_format} 格式报告: {file_path}")
                except Exception as e:
                    logger.error(f"生成 {output_format} 格式报告失败: {e}")
                    report_files[output_format] = f"Error: {str(e)}"

            # 6. 构建完整结果
            complete_result = {
                **evaluation_result,
                'report_files': report_files,
                'cache_key': cache_key,
                'framework_metadata': {
                    'framework_version': '1.0.0',
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'execution_time_seconds': time.time() - start_time,
                    'cache_status': 'miss',
                    'data_quality_score': self._calculate_data_quality_score(processed_data)
                }
            }

            # 7. 缓存结果
            if self.cache_enabled:
                self._save_to_cache(cache_key, complete_result)

            # 8. 更新性能统计
            self._update_performance_stats(time.time() - start_time)

            logger.info(f"策略 {request.strategy_name} 评估完成，耗时: {time.time() - start_time:.2f}秒")

            return complete_result

        except Exception as e:
            logger.error(f"策略 {request.strategy_name} 评估失败: {e}")
            raise

    @exception_handler(reraise=True)
    @performance_monitor(threshold=300.0)
    def evaluate_multiple_strategies(self, request: BatchEvaluationRequest) -> Dict[str, Any]:
        """
        批量评估多个策略性能

        Args:
            request: 批量评估请求

        Returns:
            Dict[str, Any]: 批量评估结果
        """
        start_time = time.time()

        logger.info(f"开始批量评估 {len(request.strategies_data)} 个策略")

        try:
            # 1. 验证输入数据
            if not request.strategies_data:
                raise ValueError("策略数据不能为空")

            # 2. 创建单个策略评估请求
            individual_requests = []
            for strategy_name, strategy_data in request.strategies_data.items():
                individual_request = PerformanceEvaluationRequest(
                    strategy_name=strategy_name,
                    strategy_data=strategy_data,
                    benchmark_code=request.benchmark_code,
                    evaluation_period_days=request.evaluation_period_days,
                    output_formats=['json']  # 批量评估只生成JSON格式
                )
                individual_requests.append(individual_request)

            # 3. 并行处理
            individual_results = {}
            failed_strategies = {}

            if request.parallel_workers > 1 and len(individual_requests) > 1:
                # 多线程并行处理
                logger.info(f"使用 {request.parallel_workers} 个线程并行处理")

                with ThreadPoolExecutor(max_workers=request.parallel_workers) as executor:
                    # 提交任务
                    future_to_request = {
                        executor.submit(self._evaluate_single_strategy_internal, req): req
                        for req in individual_requests
                    }

                    # 收集结果
                    for future in as_completed(future_to_request):
                        req = future_to_request[future]
                        try:
                            result = future.result()
                            individual_results[req.strategy_name] = result
                            logger.debug(f"✅ 策略 {req.strategy_name} 评估完成")
                        except Exception as e:
                            logger.error(f"❌ 策略 {req.strategy_name} 评估失败: {e}")
                            failed_strategies[req.strategy_name] = str(e)
            else:
                # 单线程顺序处理
                logger.info("使用单线程顺序处理")
                for req in individual_requests:
                    try:
                        result = self._evaluate_single_strategy_internal(req)
                        individual_results[req.strategy_name] = result
                        logger.debug(f"✅ 策略 {req.strategy_name} 评估完成")
                    except Exception as e:
                        logger.error(f"❌ 策略 {req.strategy_name} 评估失败: {e}")
                        failed_strategies[req.strategy_name] = str(e)

            # 4. 执行多策略比较分析
            comparison_result = self.evaluator.evaluate_multiple_strategies(
                request.strategies_data,
                benchmark_data=None  # 将在内部获取
            )

            # 5. 生成综合报告
            comprehensive_result = {
                **comparison_result,
                'individual_results': individual_results,
                'failed_strategies': failed_strategies,
                'batch_metadata': {
                    'total_strategies': len(request.strategies_data),
                    'successful_evaluations': len(individual_results),
                    'failed_evaluations': len(failed_strategies),
                    'success_rate': len(individual_results) / len(request.strategies_data) if request.strategies_data else 0,
                    'parallel_workers_used': request.parallel_workers,
                    'total_execution_time': time.time() - start_time,
                    'avg_time_per_strategy': (time.time() - start_time) / len(request.strategies_data) if request.strategies_data else 0
                }
            }

            # 6. 生成批量报告文件
            report_files = {}
            for output_format in request.output_formats:
                try:
                    file_path = self.report_generator.generate_multi_strategy_report(
                        evaluation_results=comprehensive_result,
                        output_format=output_format,
                        report_name=f"batch_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    report_files[output_format] = file_path
                    logger.info(f"生成批量 {output_format} 格式报告: {file_path}")
                except Exception as e:
                    logger.error(f"生成批量 {output_format} 格式报告失败: {e}")
                    report_files[output_format] = f"Error: {str(e)}"

            comprehensive_result['batch_report_files'] = report_files

            execution_time = time.time() - start_time
            logger.info(f"批量策略评估完成: {len(individual_results)}/{len(request.strategies_data)} 成功，"
                       f"耗时: {execution_time:.2f}秒，"
                       f"平均每策略: {execution_time/len(request.strategies_data):.2f}秒")

            return comprehensive_result

        except Exception as e:
            logger.error(f"批量策略评估失败: {e}")
            raise

    def _evaluate_single_strategy_internal(self, request: PerformanceEvaluationRequest) -> Dict[str, Any]:
        """
        内部单策略评估方法（用于并行处理）

        Args:
            request: 评估请求

        Returns:
            Dict[str, Any]: 评估结果
        """
        # 使用相同的逻辑，但不生成报告文件
        processed_data = self._preprocess_strategy_data(request.strategy_data, request.strategy_name)

        # 检查缓存
        cache_key = self._generate_cache_key(request, processed_data)
        if self.cache_enabled:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

        # 获取基准数据
        benchmark_data = self._get_benchmark_data(
            processed_data.index[0],
            processed_data.index[-1],
            request.benchmark_code
        )

        # 执行评估
        evaluation_result = self.evaluator.evaluate_strategy_performance(
            strategy_results=processed_data,
            strategy_name=request.strategy_name,
            benchmark_data=benchmark_data
        )

        # 缓存结果
        if self.cache_enabled:
            self._save_to_cache(cache_key, evaluation_result)

        return evaluation_result

    @performance_monitor(threshold=5.0)
    def _preprocess_strategy_data(self, strategy_data: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
        """
        预处理策略数据

        使用向量化计算优化性能

        Args:
            strategy_data: 原始策略数据
            strategy_name: 策略名称

        Returns:
            pd.DataFrame: 处理后的数据
        """
        try:
            logger.debug(f"开始预处理策略数据: {strategy_name}")

            # 复制数据避免修改原始数据
            data = strategy_data.copy()

            # 1. 索引处理（向量化）
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'date' in data.columns:
                    data.set_index('date', inplace=True)
                    data.index = pd.to_datetime(data.index)
                else:
                    raise ValueError("数据必须包含日期索引或日期列")

            # 2. 收益率计算（向量化）
            if 'returns' not in data.columns:
                if 'close' in data.columns:
                    data['returns'] = data['close'].pct_change()
                elif 'price' in data.columns:
                    data['returns'] = data['price'].pct_change()
                else:
                    raise ValueError("数据必须包含收益率列或价格列")

            # 3. 异常值处理（向量化）
            returns_series = data['returns']

            # 移除无穷大值
            returns_series = returns_series.replace([np.inf, -np.inf], np.nan)

            # 异常值检测（使用IQR方法，比逐个判断更高效）
            Q1 = returns_series.quantile(0.25)
            Q3 = returns_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            # 标记异常值
            extreme_mask = (returns_series < lower_bound) | (returns_series > upper_bound)
            if extreme_mask.any():
                num_extreme = extreme_mask.sum()
                logger.warning(f"策略 {strategy_name} 发现 {num_extreme} 个异常收益率数据点")
                # 使用前向填充处理异常值
                returns_series.loc[extreme_mask] = np.nan
                returns_series = returns_series.fillna(method='ffill').fillna(0)

            data['returns'] = returns_series

            # 4. 累计收益计算（向量化）
            data['cumulative_returns'] = (1 + data['returns']).cumprod()

            # 5. 其他衍生指标（向量化计算）
            # 滚动波动率
            data['rolling_volatility'] = data['returns'].rolling(window=20).std() * np.sqrt(252)

            # 滚动夏普比率
            rolling_excess_returns = data['returns'] - self.config.risk_free_rate / 252
            data['rolling_sharpe'] = (rolling_excess_returns.rolling(window=20).mean() * 252) / data['rolling_volatility']

            # 回撤序列
            running_max = data['cumulative_returns'].cummax()
            data['drawdown'] = (data['cumulative_returns'] - running_max) / running_max

            # 6. 数据质量检查
            data_quality_issues = []

            if data['returns'].isna().sum() > len(data) * 0.1:  # 超过10%的缺失值
                data_quality_issues.append("收益率缺失值过多")

            if data['returns'].std() == 0:
                data_quality_issues.append("收益率无变化")

            if len(data) < self.config.min_periods:
                data_quality_issues.append(f"数据长度不足，需要至少{self.config.min_periods}个数据点")

            if data_quality_issues:
                logger.warning(f"策略 {strategy_name} 数据质量问题: {', '.join(data_quality_issues)}")

            # 7. 排序（确保时间序列正确）
            data.sort_index(inplace=True)

            logger.debug(f"策略数据预处理完成: {strategy_name}, 数据点数: {len(data)}")

            return data

        except Exception as e:
            logger.error(f"预处理策略 {strategy_name} 数据失败: {e}")
            raise

    def _get_benchmark_data(self,
                          start_date: pd.Timestamp,
                          end_date: pd.Timestamp,
                          benchmark_code: Optional[str] = None) -> pd.DataFrame:
        """
        获取基准数据（优化版本）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            benchmark_code: 基准代码

        Returns:
            pd.DataFrame: 基准数据
        """
        # 使用指定的基准代码或配置中的默认基准
        benchmark_code = benchmark_code or self.config.benchmark_code

        # 生成基准数据缓存键
        cache_key = f"benchmark_{benchmark_code}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

        # 检查缓存
        if self.cache_enabled and cache_key in self.cache:
            logger.debug(f"从缓存获取基准数据: {benchmark_code}")
            return self.cache[cache_key]

        try:
            if self.data_access is None:
                logger.warning("无法获取真实基准数据，使用模拟数据")
                benchmark_data = self._generate_optimized_mock_benchmark(start_date, end_date)
            else:
                # 从ClickHouse获取基准数据（优化查询）
                query = f"""
                SELECT date, close, volume
                FROM stock_info WHERE level = %(level)s AND code = '{benchmark_code}'
                AND level = '日线'
                AND date >= '{start_date.strftime('%Y-%m-%d')}'
                AND date <= '{end_date.strftime('%Y-%m-%d')}'
                ORDER BY date ASC
                """

                benchmark_data = self.data_access.query_dataframe(query)

                if benchmark_data.empty:
                    logger.warning(f"未获取到基准 {benchmark_code} 数据，使用模拟数据")
                    benchmark_data = self._generate_optimized_mock_benchmark(start_date, end_date)
                else:
                    # 处理基准数据（向量化）
                    benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
                    benchmark_data.set_index('date', inplace=True)
                    benchmark_data['returns'] = benchmark_data['close'].pct_change().fillna(0)
                    benchmark_data['cumulative_returns'] = (1 + benchmark_data['returns']).cumprod()

            # 缓存结果
            if self.cache_enabled:
                self.cache[cache_key] = benchmark_data

            logger.debug(f"基准数据获取完成: {benchmark_code}, 数据点数: {len(benchmark_data)}")

            return benchmark_data

        except Exception as e:
            logger.error(f"获取基准数据失败: {e}")
            return self._generate_optimized_mock_benchmark(start_date, end_date)

    def _generate_optimized_mock_benchmark(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        生成优化的模拟基准数据（向量化实现）

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.DataFrame: 模拟基准数据
        """
        # 生成日期范围
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # 设置随机种子确保可重复性
        np.random.seed(42)

        # 向量化生成收益率（符合股市特征）
        # 使用GBM模型参数
        mu = 0.08 / 252  # 年化收益率8%
        sigma = 0.2 / np.sqrt(252)  # 年化波动率20%

        # 生成随机收益率
        random_shocks = np.random.normal(0, 1, len(dates))
        returns = mu + sigma * random_shocks

        # 添加一些现实特征
        # 1. 周末效应（周一收益率稍低）
        weekdays = pd.Series(dates).dt.dayofweek
        returns[weekdays == 0] *= 0.8  # 周一效应

        # 2. 波动聚集效应
        for i in range(1, len(returns)):
            if abs(returns[i-1]) > 2 * sigma:
                returns[i] *= 1.5  # 高波动后继续高波动

        # 创建DataFrame
        benchmark_data = pd.DataFrame({
            'returns': returns,
            'cumulative_returns': np.cumprod(1 + returns)
        }, index=dates)

        # 生成价格
        benchmark_data['close'] = 100 * benchmark_data['cumulative_returns']

        # 生成成交量
        base_volume = 1000000
        volume_multiplier = 1 + 0.5 * np.abs(returns) / sigma  # 波动大时成交量大
        benchmark_data['volume'] = (base_volume * volume_multiplier).astype(int)

        return benchmark_data

    def _generate_cache_key(self, request: PerformanceEvaluationRequest, processed_data: pd.DataFrame) -> str:
        """
        生成缓存键

        Args:
            request: 评估请求
            processed_data: 处理后的数据

        Returns:
            str: 缓存键
        """
        # 包含关键参数和数据特征
        key_components = [
            request.strategy_name,
            request.benchmark_code or 'default',
            str(request.evaluation_period_days),
            processed_data.index[0].strftime('%Y%m%d'),
            processed_data.index[-1].strftime('%Y%m%d'),
            str(len(processed_data)),
            # 数据内容哈希（确保数据变化时缓存失效）
            hashlib.md5(str(processed_data['returns'].sum()).encode()).hexdigest()[:8]
        ]

        cache_key = "_".join(key_components)
        return hashlib.md5(cache_key.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从缓存获取结果"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)

                # 检查缓存是否过期（24小时）
                cache_time = datetime.fromisoformat(cached_data.get('cache_timestamp', '2000-01-01'))
                if datetime.now() - cache_time > timedelta(hours=24):
                    logger.debug(f"缓存已过期: {cache_key}")
                    return None

                logger.debug(f"从缓存加载结果: {cache_key}")
                return cached_data.get('result')

            except Exception as e:
                logger.warning(f"读取缓存文件失败: {e}")
                return None

        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """保存结果到缓存"""
        if not self.cache_enabled:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            cache_data = {
                'cache_timestamp': datetime.now().isoformat(),
                'cache_key': cache_key,
                'result': self._make_json_serializable(result)
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2, default=str)

            logger.debug(f"结果已缓存: {cache_key}")

        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """将对象转换为JSON可序列化格式（增强版本）"""
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.Series):
            # 处理Series的索引和值
            if isinstance(obj.index, pd.DatetimeIndex):
                return {dt.isoformat(): self._make_json_serializable(val)
                       for dt, val in zip(obj.index, obj.values)}
            else:
                return {str(idx): self._make_json_serializable(val)
                       for idx, val in zip(obj.index, obj.values)}
        elif isinstance(obj, pd.DataFrame):
            # 处理DataFrame
            result = {}
            for col in obj.columns:
                if isinstance(obj.index, pd.DatetimeIndex):
                    result[str(col)] = {dt.isoformat(): self._make_json_serializable(val)
                                      for dt, val in zip(obj.index, obj[col].values)}
                else:
                    result[str(col)] = {str(idx): self._make_json_serializable(val)
                                      for idx, val in zip(obj.index, obj[col].values)}
            return result
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj) or obj is pd.NaT:
            return None
        elif hasattr(obj, '__dict__') and not callable(obj):
            # 处理自定义对象
            return {str(k): self._make_json_serializable(v) for k, v in obj.__dict__.items()}
        else:
            # 尝试转换为字符串
            try:
                return str(obj)
            except Exception:
                return None

    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """
        计算数据质量评分

        Args:
            data: 数据

        Returns:
            float: 质量评分 (0-100)
        """
        score = 100.0

        # 缺失值惩罚
        missing_ratio = data['returns'].isna().sum() / len(data)
        score -= missing_ratio * 30  # 最多扣30分

        # 数据长度评分
        if len(data) < self.config.min_periods:
            score -= 20
        elif len(data) < self.config.evaluation_period:
            score -= 10

        # 波动率评分（过低或过高都不好）
        volatility = data['returns'].std() * np.sqrt(252)
        if volatility < 0.05 or volatility > 1.0:
            score -= 15

        # 异常值评分
        Q1 = data['returns'].quantile(0.25)
        Q3 = data['returns'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((data['returns'] < (Q1 - 3 * IQR)) | (data['returns'] > (Q3 + 3 * IQR))).sum()
        outlier_ratio = outliers / len(data)
        score -= outlier_ratio * 25  # 最多扣25分

        return max(0, score)

    def _update_performance_stats(self, execution_time: float):
        """更新性能统计"""
        self.performance_stats['total_evaluations'] += 1
        self.performance_stats['total_execution_time'] += execution_time
        self.performance_stats['avg_execution_time'] = (
            self.performance_stats['total_execution_time'] /
            self.performance_stats['total_evaluations']
        )

    def get_framework_performance_report(self) -> Dict[str, Any]:
        """
        获取框架性能报告

        Returns:
            Dict[str, Any]: 性能报告
        """
        return {
            'framework_stats': self.performance_stats,
            'evaluator_stats': self.evaluator.get_performance_report(),
            'cache_stats': {
                'enabled': self.cache_enabled,
                'cache_dir': str(self.cache_dir),
                'cache_files_count': len(list(self.cache_dir.glob('*.json'))),
                'hit_rate': (self.performance_stats['cache_hits'] /
                           max(1, self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']))
            },
            'configuration': asdict(self.config),
            'system_info': self._get_system_info()
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            import psutil
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'memory_percent': psutil.virtual_memory().percent
            }
        except ImportError:
            return {'error': 'psutil not available'}

    def clear_cache(self):
        """清理缓存"""
        try:
            for cache_file in self.cache_dir.glob('*.json'):
                cache_file.unlink()

            self.cache.clear()
            logger.info("缓存已清理")

        except Exception as e:
            logger.error(f"清理缓存失败: {e}")


# 创建全局框架实例
performance_framework = PerformanceEvaluationFramework()


# 便捷函数
def evaluate_strategy_performance(strategy_name: str,
                                strategy_data: pd.DataFrame,
                                benchmark_code: Optional[str] = None,
                                output_formats: List[str] = None) -> Dict[str, Any]:
    """
    便捷的策略性能评估函数

    Args:
        strategy_name: 策略名称
        strategy_data: 策略数据
        benchmark_code: 基准代码
        output_formats: 输出格式

    Returns:
        Dict[str, Any]: 评估结果
    """
    request = PerformanceEvaluationRequest(
        strategy_name=strategy_name,
        strategy_data=strategy_data,
        benchmark_code=benchmark_code,
        output_formats=output_formats or ['json', 'html']
    )

    return performance_framework.evaluate_single_strategy(request)


def batch_evaluate_strategies(strategies_data: Dict[str, pd.DataFrame],
                            benchmark_code: Optional[str] = None,
                            parallel_workers: int = 8,
                            output_formats: List[str] = None) -> Dict[str, Any]:
    """
    便捷的批量策略性能评估函数

    Args:
        strategies_data: 策略数据字典
        benchmark_code: 基准代码
        parallel_workers: 并行工作线程数
        output_formats: 输出格式

    Returns:
        Dict[str, Any]: 批量评估结果
    """
    request = BatchEvaluationRequest(
        strategies_data=strategies_data,
        benchmark_code=benchmark_code,
        parallel_workers=parallel_workers,
        output_formats=output_formats or ['json', 'html']
    )

    return performance_framework.evaluate_multiple_strategies(request)