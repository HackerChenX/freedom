#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
参数组合测试器

验证不同参数设置下的技术指标性能和稳定性。
严格遵循六层架构原则，提供全面的参数优化测试功能。

L6: 测试应用层 - 本文件提供参数测试功能
L5: 测试业务层 - 具体参数测试逻辑
L4: 测试服务层 - 指标计算服务
L3: 测试数据层 - 测试数据管理
L2: 测试基础设施层 - 测试工具和配置
L1: 测试数据存储层 - 测试数据和结果存储
"""

import os
import sys
import time
import json
import itertools
import numpy as np
import pandas as pd
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from indicators.factory import IndicatorFactory
from indicators.base_indicator import BaseIndicator
from db.managers.query_executor import UnifiedQueryExecutor

logger = get_logger('parameter_combination_tester')


class ParameterTestType(Enum):
    """参数测试类型枚举"""
    GRID_SEARCH = "grid_search"                    # 网格搜索
    RANDOM_SEARCH = "random_search"                # 随机搜索
    BOUNDARY_TEST = "boundary_test"                # 边界测试
    PERFORMANCE_TEST = "performance_test"          # 性能测试
    STABILITY_TEST = "stability_test"              # 稳定性测试
    SENSITIVITY_ANALYSIS = "sensitivity_analysis" # 敏感性分析


class OptimizationMetric(Enum):
    """优化指标枚举"""
    ACCURACY = "accuracy"                      # 准确性
    STABILITY = "stability"                    # 稳定性
    SENSITIVITY = "sensitivity"                # 敏感性
    COMPUTATION_SPEED = "computation_speed"    # 计算速度
    SIGNAL_QUALITY = "signal_quality"          # 信号质量
    RISK_ADJUSTED_RETURN = "risk_adjusted_return" # 风险调整收益


@dataclass
class ParameterCombination:
    """参数组合"""
    indicator_name: str
    parameters: Dict[str, Any]
    combination_id: str
    description: str = ""


@dataclass
class ParameterTestResult:
    """参数测试结果"""
    combination: ParameterCombination
    test_type: ParameterTestType
    optimization_score: float
    stability_score: float
    performance_score: float
    execution_time: float
    memory_usage: float
    signal_count: int
    signal_quality: float
    error_rate: float
    detailed_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class ParameterOptimizationResult:
    """参数优化结果"""
    indicator_name: str
    best_combination: ParameterCombination
    best_score: float
    tested_combinations: int
    optimization_metric: OptimizationMetric
    parameter_rankings: List[ParameterTestResult] = field(default_factory=list)
    parameter_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterTestSuite:
    """参数测试套件结果"""
    suite_name: str
    total_indicators: int
    tested_indicators: int
    total_combinations_tested: int
    average_optimization_score: float
    best_performing_indicators: List[str] = field(default_factory=list)
    optimization_results: List[ParameterOptimizationResult] = field(default_factory=list)
    execution_summary: Dict[str, Any] = field(default_factory=dict)


class ParameterCombinationTester:
    """
    参数组合测试器
    
    负责对技术指标的不同参数组合进行全面测试和优化
    """
    
    def __init__(self):
        """初始化参数组合测试器"""
        self.indicator_factory = IndicatorFactory()
        self.query_executor = UnifiedQueryExecutor()
        
        # 测试配置
        self.test_config = {
            "max_combinations_per_indicator": 50,    # 每个指标最大测试组合数
            "optimization_timeout": 300,             # 优化超时时间（秒）
            "performance_threshold": 2.0,            # 性能阈值（秒）
            "memory_threshold": 512,                 # 内存阈值（MB）
            "min_signal_quality": 0.6,               # 最小信号质量阈值
            "stability_window": 10,                  # 稳定性测试窗口
            "concurrent_tests": 3                    # 并发测试数量
        }
        
        # 预定义的参数空间
        self.parameter_spaces = self._initialize_parameter_spaces()
        
        # 测试数据缓存
        self.test_data_cache = {}
        
        # 测试结果存储
        self.test_results: List[ParameterTestResult] = []
    
    def _initialize_parameter_spaces(self) -> Dict[str, Dict[str, List[Any]]]:
        """初始化指标参数空间"""
        return {
            "MA": {
                "period": [5, 10, 15, 20, 30, 60],
                "price_field": ["close", "high", "low", "open"]
            },
            "EMA": {
                "period": [5, 10, 12, 15, 20, 26, 30],
                "alpha": [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            "MACD": {
                "fast_period": [8, 10, 12, 15],
                "slow_period": [20, 24, 26, 30],
                "signal_period": [7, 9, 12, 15]
            },
            "RSI": {
                "period": [9, 14, 21, 28],
                "overbought": [65, 70, 75, 80],
                "oversold": [20, 25, 30, 35]
            },
            "KDJ": {
                "n": [5, 9, 14, 19],
                "m1": [2, 3, 5],
                "m2": [2, 3, 5]
            },
            "BOLL": {
                "period": [15, 20, 25, 30],
                "std_dev": [1.5, 2.0, 2.5, 3.0]
            },
            "ATR": {
                "period": [10, 14, 20, 28]
            },
            "CCI": {
                "period": [14, 20, 28],
                "constant": [0.015, 0.02, 0.025]
            },
            "WR": {
                "period": [9, 14, 21, 28]
            },
            "OBV": {
                "signal_period": [5, 10, 15, 20]
            }
        }
    
    @performance_monitor(threshold_seconds=180.0)
    @exception_handler(reraise=True)
    def run_comprehensive_parameter_tests(self, 
                                         indicators: Optional[List[str]] = None,
                                         test_types: Optional[List[ParameterTestType]] = None,
                                         optimization_metric: OptimizationMetric = OptimizationMetric.ACCURACY) -> ParameterTestSuite:
        """
        运行全面的参数组合测试
        
        Args:
            indicators: 要测试的指标列表，None表示测试所有指标
            test_types: 要运行的测试类型列表，None表示运行所有测试
            optimization_metric: 优化指标
            
        Returns:
            ParameterTestSuite: 测试套件结果
        """
        logger.info("开始运行全面的参数组合测试")
        start_time = time.time()
        
        # 确定要测试的指标
        if indicators is None:
            indicators = list(self.parameter_spaces.keys())
        
        # 确定要运行的测试类型
        if test_types is None:
            test_types = [ParameterTestType.GRID_SEARCH, ParameterTestType.PERFORMANCE_TEST]
        
        optimization_results = []
        
        # 获取测试数据
        test_data = self._get_parameter_test_data()
        if test_data is None or test_data.empty:
            logger.error("无法获取参数测试数据")
            return self._create_empty_test_suite()
        
        # 对每个指标进行参数优化
        for indicator_name in indicators:
            logger.info(f"测试指标参数: {indicator_name}")
            
            try:
                optimization_result = self._optimize_indicator_parameters(
                    indicator_name, test_data, test_types, optimization_metric
                )
                if optimization_result:
                    optimization_results.append(optimization_result)
            except Exception as e:
                logger.error(f"指标 {indicator_name} 参数优化失败: {e}")
        
        # 生成测试套件结果
        suite_result = self._generate_parameter_test_suite_result(
            optimization_results, optimization_metric
        )
        
        execution_time = time.time() - start_time
        suite_result.execution_summary = {
            "total_execution_time": execution_time,
            "optimization_metric": optimization_metric.value,
            "test_types": [t.value for t in test_types]
        }
        
        logger.info(f"参数组合测试完成，总耗时: {execution_time:.2f}秒")
        logger.info(f"平均优化评分: {suite_result.average_optimization_score:.4f}")
        
        return suite_result
    
    def _get_parameter_test_data(self) -> Optional[pd.DataFrame]:
        """获取参数测试数据"""
        try:
            # 从缓存获取
            if 'parameter_test_data' in self.test_data_cache:
                return self.test_data_cache['parameter_test_data']
            
            # 从数据库获取测试数据
            query = """
            SELECT code, name, date, open, high, low, close, volume, turnover_rate
            FROM stock_info 
            WHERE code IN ('000001', '000002', '000858', '002415', '600036')
            AND level = '日线'
            AND date >= '2024-01-01' AND date <= '2024-12-31'
            ORDER BY code, date ASC
            """
            
            data = self.query_executor.execute_query(query)
            if data is not None and not data.empty:
                self.test_data_cache['parameter_test_data'] = data
                return data
            else:
                # 生成模拟数据
                mock_data = self._generate_mock_parameter_test_data()
                self.test_data_cache['parameter_test_data'] = mock_data
                return mock_data
                
        except Exception as e:
            logger.warning(f"获取参数测试数据失败，使用模拟数据: {e}")
            mock_data = self._generate_mock_parameter_test_data()
            self.test_data_cache['parameter_test_data'] = mock_data
            return mock_data
    
    def _generate_mock_parameter_test_data(self) -> pd.DataFrame:
        """生成模拟参数测试数据"""
        stocks = ['000001', '000002', '000858']
        days = 200
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        
        all_data = []
        
        for stock_code in stocks:
            np.random.seed(hash(stock_code) % 2**32)
            base_price = 10.0 + np.random.uniform(5, 50)
            
            for i, date in enumerate(dates):
                # 模拟不同的市场模式
                if i < days // 3:  # 趋势上涨
                    trend = 0.001
                elif i < 2 * days // 3:  # 震荡
                    trend = 0.0
                else:  # 趋势下跌
                    trend = -0.001
                
                # 价格变化
                daily_change = trend + np.random.normal(0, 0.02)
                base_price *= (1 + daily_change)
                
                # 生成OHLC
                open_price = base_price * (1 + np.random.normal(0, 0.005))
                close_price = base_price * (1 + np.random.normal(0, 0.005))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
                
                volume = abs(np.random.normal(1000000, 300000))
                
                all_data.append({
                    'code': stock_code,
                    'name': f'测试股票{stock_code}',
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': int(volume),
                    'turnover_rate': round(np.random.uniform(0.5, 8.0), 2)
                })
        
        return pd.DataFrame(all_data)
    
    @exception_handler(reraise=False, default_return=None)
    def _optimize_indicator_parameters(self, indicator_name: str, 
                                     test_data: pd.DataFrame,
                                     test_types: List[ParameterTestType],
                                     optimization_metric: OptimizationMetric) -> Optional[ParameterOptimizationResult]:
        """
        优化单个指标的参数
        
        Args:
            indicator_name: 指标名称
            test_data: 测试数据
            test_types: 测试类型列表
            optimization_metric: 优化指标
            
        Returns:
            Optional[ParameterOptimizationResult]: 优化结果
        """
        try:
            if indicator_name not in self.parameter_spaces:
                logger.warning(f"指标 {indicator_name} 没有定义参数空间")
                return None
            
            # 生成参数组合
            parameter_combinations = self._generate_parameter_combinations(indicator_name)
            
            if not parameter_combinations:
                logger.warning(f"指标 {indicator_name} 无法生成参数组合")
                return None
            
            # 测试所有参数组合
            test_results = []
            
            with ThreadPoolExecutor(max_workers=self.test_config["concurrent_tests"]) as executor:
                futures = []
                
                for combination in parameter_combinations:
                    for test_type in test_types:
                        future = executor.submit(
                            self._test_parameter_combination,
                            combination, test_data, test_type, optimization_metric
                        )
                        futures.append(future)
                
                # 收集结果
                for future in futures:
                    try:
                        result = future.result(timeout=60)
                        if result:
                            test_results.append(result)
                    except Exception as e:
                        logger.error(f"参数组合测试失败: {e}")
            
            if not test_results:
                return None
            
            # 找到最佳参数组合
            best_result = max(test_results, key=lambda r: r.optimization_score)
            
            # 参数分析
            parameter_analysis = self._analyze_parameter_effects(test_results, indicator_name)
            
            return ParameterOptimizationResult(
                indicator_name=indicator_name,
                best_combination=best_result.combination,
                best_score=best_result.optimization_score,
                tested_combinations=len(set(r.combination.combination_id for r in test_results)),
                optimization_metric=optimization_metric,
                parameter_rankings=sorted(test_results, key=lambda r: r.optimization_score, reverse=True),
                parameter_analysis=parameter_analysis
            )
            
        except Exception as e:
            logger.error(f"优化指标 {indicator_name} 参数失败: {e}")
            return None
    
    def _generate_parameter_combinations(self, indicator_name: str) -> List[ParameterCombination]:
        """生成参数组合"""
        if indicator_name not in self.parameter_spaces:
            return []
        
        param_space = self.parameter_spaces[indicator_name]
        combinations = []
        
        try:
            # 获取所有参数的笛卡尔积
            param_names = list(param_space.keys())
            param_values = list(param_space.values())
            
            # 限制组合数量
            max_combinations = self.test_config["max_combinations_per_indicator"]
            
            # 如果组合数量太多，使用采样
            total_combinations = 1
            for values in param_values:
                total_combinations *= len(values)
            
            if total_combinations <= max_combinations:
                # 使用完整网格搜索
                for i, combination in enumerate(itertools.product(*param_values)):
                    param_dict = dict(zip(param_names, combination))
                    
                    combinations.append(ParameterCombination(
                        indicator_name=indicator_name,
                        parameters=param_dict,
                        combination_id=f"{indicator_name}_{i:03d}",
                        description=f"{indicator_name} 参数组合 {i+1}"
                    ))
            else:
                # 使用随机采样
                np.random.seed(42)
                for i in range(max_combinations):
                    param_dict = {}
                    for param_name, param_range in param_space.items():
                        param_dict[param_name] = np.random.choice(param_range)
                    
                    combinations.append(ParameterCombination(
                        indicator_name=indicator_name,
                        parameters=param_dict,
                        combination_id=f"{indicator_name}_random_{i:03d}",
                        description=f"{indicator_name} 随机参数组合 {i+1}"
                    ))
            
            # 添加默认参数组合
            default_params = self._get_default_parameters(indicator_name)
            if default_params:
                combinations.insert(0, ParameterCombination(
                    indicator_name=indicator_name,
                    parameters=default_params,
                    combination_id=f"{indicator_name}_default",
                    description=f"{indicator_name} 默认参数组合"
                ))
            
        except Exception as e:
            logger.error(f"生成参数组合失败: {e}")
        
        return combinations
    
    def _get_default_parameters(self, indicator_name: str) -> Dict[str, Any]:
        """获取指标默认参数"""
        defaults = {
            "MA": {"period": 20},
            "EMA": {"period": 12},
            "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "RSI": {"period": 14, "overbought": 70, "oversold": 30},
            "KDJ": {"n": 9, "m1": 3, "m2": 3},
            "BOLL": {"period": 20, "std_dev": 2.0},
            "ATR": {"period": 14},
            "CCI": {"period": 14, "constant": 0.015},
            "WR": {"period": 14},
            "OBV": {"signal_period": 10}
        }
        
        return defaults.get(indicator_name, {})
    
    @exception_handler(reraise=False, default_return=None)
    def _test_parameter_combination(self, combination: ParameterCombination, 
                                  test_data: pd.DataFrame,
                                  test_type: ParameterTestType,
                                  optimization_metric: OptimizationMetric) -> Optional[ParameterTestResult]:
        """
        测试单个参数组合
        
        Args:
            combination: 参数组合
            test_data: 测试数据
            test_type: 测试类型
            optimization_metric: 优化指标
            
        Returns:
            Optional[ParameterTestResult]: 测试结果
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # 创建指标实例
            indicator = self.indicator_factory.create_indicator(
                combination.indicator_name, **combination.parameters
            )
            
            if indicator is None:
                return None
            
            # 按股票分组测试
            stock_codes = test_data['code'].unique()
            all_metrics = []
            errors = []
            warnings = []
            
            for stock_code in stock_codes:
                stock_data = test_data[test_data['code'] == stock_code].copy()
                
                if len(stock_data) < 30:  # 数据太少跳过
                    continue
                
                try:
                    # 计算指标
                    calc_start = time.time()
                    result = indicator.calculate(stock_data)
                    calc_time = time.time() - calc_start
                    
                    # 评估结果
                    metrics = self._evaluate_indicator_result(
                        result, stock_data, combination, test_type, optimization_metric
                    )
                    
                    metrics['calculation_time'] = calc_time
                    all_metrics.append(metrics)
                    
                except Exception as e:
                    errors.append(f"股票 {stock_code} 计算失败: {str(e)}")
            
            if not all_metrics:
                return ParameterTestResult(
                    combination=combination,
                    test_type=test_type,
                    optimization_score=0.0,
                    stability_score=0.0,
                    performance_score=0.0,
                    execution_time=time.time() - start_time,
                    memory_usage=self._get_memory_usage() - start_memory,
                    signal_count=0,
                    signal_quality=0.0,
                    error_rate=1.0,
                    errors=errors
                )
            
            # 汇总指标
            optimization_score = np.mean([m['optimization_score'] for m in all_metrics])
            stability_score = np.mean([m['stability_score'] for m in all_metrics])
            performance_score = np.mean([m['performance_score'] for m in all_metrics])
            signal_count = sum([m['signal_count'] for m in all_metrics])
            signal_quality = np.mean([m['signal_quality'] for m in all_metrics])
            error_rate = len(errors) / (len(stock_codes) + len(errors)) if (len(stock_codes) + len(errors)) > 0 else 0.0
            
            # 详细指标
            detailed_metrics = {}
            for key in ['volatility', 'trend_consistency', 'noise_ratio']:
                if all(key in m for m in all_metrics):
                    detailed_metrics[key] = np.mean([m[key] for m in all_metrics])
            
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            return ParameterTestResult(
                combination=combination,
                test_type=test_type,
                optimization_score=optimization_score,
                stability_score=stability_score,
                performance_score=performance_score,
                execution_time=execution_time,
                memory_usage=memory_usage,
                signal_count=signal_count,
                signal_quality=signal_quality,
                error_rate=error_rate,
                detailed_metrics=detailed_metrics,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return ParameterTestResult(
                combination=combination,
                test_type=test_type,
                optimization_score=0.0,
                stability_score=0.0,
                performance_score=0.0,
                execution_time=time.time() - start_time,
                memory_usage=self._get_memory_usage() - start_memory,
                signal_count=0,
                signal_quality=0.0,
                error_rate=1.0,
                errors=[str(e)]
            )
    
    def _evaluate_indicator_result(self, result: Union[pd.DataFrame, pd.Series], 
                                 stock_data: pd.DataFrame,
                                 combination: ParameterCombination,
                                 test_type: ParameterTestType,
                                 optimization_metric: OptimizationMetric) -> Dict[str, float]:
        """评估指标计算结果"""
        try:
            metrics = {}
            
            # 基本有效性检查
            if result is None or (hasattr(result, 'empty') and result.empty):
                return self._create_default_metrics()
            
            # 转换为DataFrame格式
            if isinstance(result, pd.Series):
                result_df = result.to_frame()
            else:
                result_df = result
            
            # 数据质量评估
            total_values = result_df.size
            nan_count = result_df.isnull().sum().sum()
            valid_ratio = (total_values - nan_count) / total_values if total_values > 0 else 0.0
            
            # 数值稳定性评估
            numeric_columns = result_df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                # 计算波动性
                volatility = 0.0
                for col in numeric_columns:
                    col_data = result_df[col].dropna()
                    if len(col_data) > 1:
                        volatility += col_data.std() / col_data.mean() if col_data.mean() != 0 else 1.0
                volatility /= len(numeric_columns)
                
                # 趋势一致性
                trend_consistency = self._calculate_trend_consistency(result_df)
                
                # 噪声比率
                noise_ratio = self._calculate_noise_ratio(result_df)
            else:
                volatility = 1.0
                trend_consistency = 0.0
                noise_ratio = 1.0
            
            # 信号质量评估
            signal_count, signal_quality = self._evaluate_signal_quality(result_df, stock_data)
            
            # 计算各项评分
            stability_score = max(0.0, 1.0 - volatility) * valid_ratio
            performance_score = self._calculate_performance_score(combination)
            
            # 根据优化指标计算优化评分
            if optimization_metric == OptimizationMetric.ACCURACY:
                optimization_score = valid_ratio * trend_consistency * signal_quality
            elif optimization_metric == OptimizationMetric.STABILITY:
                optimization_score = stability_score
            elif optimization_metric == OptimizationMetric.SIGNAL_QUALITY:
                optimization_score = signal_quality
            else:
                optimization_score = (valid_ratio + stability_score + signal_quality) / 3
            
            metrics.update({
                'optimization_score': optimization_score,
                'stability_score': stability_score,
                'performance_score': performance_score,
                'signal_count': signal_count,
                'signal_quality': signal_quality,
                'valid_ratio': valid_ratio,
                'volatility': volatility,
                'trend_consistency': trend_consistency,
                'noise_ratio': noise_ratio
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"评估指标结果失败: {e}")
            return self._create_default_metrics()
    
    def _create_default_metrics(self) -> Dict[str, float]:
        """创建默认指标"""
        return {
            'optimization_score': 0.0,
            'stability_score': 0.0,
            'performance_score': 0.0,
            'signal_count': 0,
            'signal_quality': 0.0,
            'valid_ratio': 0.0,
            'volatility': 1.0,
            'trend_consistency': 0.0,
            'noise_ratio': 1.0
        }
    
    def _calculate_trend_consistency(self, result_df: pd.DataFrame) -> float:
        """计算趋势一致性"""
        try:
            numeric_columns = result_df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return 0.0
            
            consistencies = []
            for col in numeric_columns:
                data = result_df[col].dropna()
                if len(data) < 10:
                    continue
                
                # 计算趋势方向的一致性
                changes = data.diff().dropna()
                if len(changes) == 0:
                    continue
                
                positive_changes = (changes > 0).sum()
                negative_changes = (changes < 0).sum()
                total_changes = len(changes)
                
                # 趋势一致性 = 主导方向的比例
                consistency = max(positive_changes, negative_changes) / total_changes if total_changes > 0 else 0.0
                consistencies.append(consistency)
            
            return np.mean(consistencies) if consistencies else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_noise_ratio(self, result_df: pd.DataFrame) -> float:
        """计算噪声比率"""
        try:
            numeric_columns = result_df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return 1.0
            
            noise_ratios = []
            for col in numeric_columns:
                data = result_df[col].dropna()
                if len(data) < 10:
                    continue
                
                # 使用移动平均来估计信号，计算噪声比率
                if len(data) >= 5:
                    signal = data.rolling(window=5, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                    noise = data - signal
                    
                    signal_power = signal.var()
                    noise_power = noise.var()
                    
                    if signal_power > 0:
                        noise_ratio = noise_power / signal_power
                    else:
                        noise_ratio = 1.0
                    
                    noise_ratios.append(min(1.0, noise_ratio))
            
            return np.mean(noise_ratios) if noise_ratios else 1.0
            
        except Exception:
            return 1.0
    
    def _evaluate_signal_quality(self, result_df: pd.DataFrame, 
                               stock_data: pd.DataFrame) -> Tuple[int, float]:
        """评估信号质量"""
        try:
            signal_count = 0
            signal_quality = 0.0
            
            # 简化的信号检测：寻找交叉点和极值点
            numeric_columns = result_df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                data = result_df[col].dropna()
                if len(data) < 5:
                    continue
                
                # 检测交叉信号（与移动平均线的交叉）
                if len(data) >= 10:
                    ma = data.rolling(window=10).mean()
                    crossovers = ((data > ma) & (data.shift(1) <= ma.shift(1))).sum()
                    crossunders = ((data < ma) & (data.shift(1) >= ma.shift(1))).sum()
                    
                    signal_count += crossovers + crossunders
                
                # 简单的信号质量评估
                if len(data) > 1:
                    # 基于数据的平滑度和可预测性
                    smoothness = 1.0 / (1.0 + data.diff().std()) if data.diff().std() > 0 else 1.0
                    signal_quality += smoothness
            
            # 归一化信号质量
            if len(numeric_columns) > 0:
                signal_quality /= len(numeric_columns)
            
            return int(signal_count), min(1.0, signal_quality)
            
        except Exception:
            return 0, 0.0
    
    def _calculate_performance_score(self, combination: ParameterCombination) -> float:
        """计算性能评分"""
        try:
            # 基于参数复杂度的性能评分
            param_count = len(combination.parameters)
            complexity_penalty = param_count * 0.1
            
            # 基于参数值的合理性
            reasonableness_score = 1.0
            
            # 检查常见参数的合理性
            params = combination.parameters
            
            if 'period' in params:
                period = params['period']
                if isinstance(period, (int, float)):
                    if period < 2 or period > 100:
                        reasonableness_score *= 0.5
            
            if 'std_dev' in params:
                std_dev = params['std_dev']
                if isinstance(std_dev, (int, float)):
                    if std_dev < 0.5 or std_dev > 5.0:
                        reasonableness_score *= 0.7
            
            performance_score = max(0.0, reasonableness_score - complexity_penalty)
            
            return performance_score
            
        except Exception:
            return 0.5
    
    def _analyze_parameter_effects(self, test_results: List[ParameterTestResult], 
                                 indicator_name: str) -> Dict[str, Any]:
        """分析参数效应"""
        try:
            analysis = {
                'parameter_importance': {},
                'optimal_ranges': {},
                'sensitivity_analysis': {},
                'correlation_matrix': {}
            }
            
            if not test_results:
                return analysis
            
            # 提取参数和分数
            param_names = list(test_results[0].combination.parameters.keys())
            param_data = []
            scores = []
            
            for result in test_results:
                param_values = []
                for param_name in param_names:
                    param_values.append(result.combination.parameters.get(param_name, 0))
                param_data.append(param_values)
                scores.append(result.optimization_score)
            
            param_df = pd.DataFrame(param_data, columns=param_names)
            scores_array = np.array(scores)
            
            # 参数重要性分析（基于相关性）
            for param_name in param_names:
                try:
                    param_values = param_df[param_name].astype(float)
                    correlation = np.corrcoef(param_values, scores_array)[0, 1]
                    analysis['parameter_importance'][param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
                except:
                    analysis['parameter_importance'][param_name] = 0.0
            
            # 最优参数范围
            top_10_percent = int(len(test_results) * 0.1) or 1
            top_results = sorted(test_results, key=lambda r: r.optimization_score, reverse=True)[:top_10_percent]
            
            for param_name in param_names:
                top_values = [r.combination.parameters.get(param_name) for r in top_results]
                top_values = [v for v in top_values if v is not None]
                
                if top_values:
                    try:
                        numeric_values = [float(v) for v in top_values]
                        analysis['optimal_ranges'][param_name] = {
                            'min': min(numeric_values),
                            'max': max(numeric_values),
                            'mean': np.mean(numeric_values),
                            'std': np.std(numeric_values)
                        }
                    except:
                        analysis['optimal_ranges'][param_name] = {
                            'values': list(set(top_values))
                        }
            
            return analysis
            
        except Exception as e:
            logger.error(f"参数效应分析失败: {e}")
            return {'error': str(e)}
    
    def _generate_parameter_test_suite_result(self, 
                                            optimization_results: List[ParameterOptimizationResult],
                                            optimization_metric: OptimizationMetric) -> ParameterTestSuite:
        """生成参数测试套件结果"""
        total_indicators = len(optimization_results)
        tested_indicators = len([r for r in optimization_results if r.tested_combinations > 0])
        
        if optimization_results:
            total_combinations_tested = sum(r.tested_combinations for r in optimization_results)
            average_optimization_score = sum(r.best_score for r in optimization_results) / len(optimization_results)
            
            # 最佳性能指标
            best_performing_indicators = sorted(
                optimization_results, 
                key=lambda r: r.best_score, 
                reverse=True
            )[:5]
            best_performing_names = [r.indicator_name for r in best_performing_indicators]
        else:
            total_combinations_tested = 0
            average_optimization_score = 0.0
            best_performing_names = []
        
        return ParameterTestSuite(
            suite_name="参数组合测试套件",
            total_indicators=total_indicators,
            tested_indicators=tested_indicators,
            total_combinations_tested=total_combinations_tested,
            average_optimization_score=average_optimization_score,
            best_performing_indicators=best_performing_names,
            optimization_results=optimization_results
        )
    
    def _create_empty_test_suite(self) -> ParameterTestSuite:
        """创建空的测试套件"""
        return ParameterTestSuite(
            suite_name="参数组合测试套件",
            total_indicators=0,
            tested_indicators=0,
            total_combinations_tested=0,
            average_optimization_score=0.0,
            best_performing_indicators=[],
            optimization_results=[]
        )
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0


if __name__ == "__main__":
    # 示例使用
    tester = ParameterCombinationTester()
    
    # 运行参数优化测试
    result = tester.run_comprehensive_parameter_tests(
        indicators=['MA', 'MACD', 'RSI'],
        test_types=[ParameterTestType.GRID_SEARCH, ParameterTestType.PERFORMANCE_TEST],
        optimization_metric=OptimizationMetric.ACCURACY
    )
    
    print(f"参数组合测试完成")
    print(f"平均优化评分: {result.average_optimization_score:.4f}")
    print(f"测试的指标数量: {result.tested_indicators}/{result.total_indicators}")
    print(f"总测试组合数: {result.total_combinations_tested}")
    print(f"最佳性能指标: {', '.join(result.best_performing_indicators)}") 