#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生产级金融系统全面验证测试套件

专为金融量化交易系统设计的综合测试框架，确保所有核心金融逻辑
满足生产环境部署的严格标准：

1. 技术指标计算精度验证（6位小数精度）
2. 统计显著性验证体系完整性测试
3. 数值稳定性管理器全面测试
4. 金融风险指标计算准确性验证
5. 系统性能和稳定性压力测试

测试覆盖率要求：95%以上
性能要求：≤0.05秒/股票
数值精度要求：6位小数
"""

import unittest
import numpy as np
import pandas as pd
import time
import warnings
from decimal import Decimal, getcontext
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入被测试模块
from utils.financial_statistical_validator import (
    FinancialStatisticalValidator,
    StatisticalTestType,
    EffectSizeType,
    StatisticalTestResult,
    SampleSizeAnalysis,
    get_statistical_validator
)

from utils.numerical_stability_manager import (
    NumericalStabilityManager,
    get_stability_manager,
    ensure_precision,
    safe_divide,
    validate_rsi,
    check_anomalies
)

from indicators.complete_indicator_registry import (
    CompleteIndicatorRegistry,
    get_indicator_registry,
    initialize_indicators
)

from strategy.strategy_optimizer import (
    StrategyOptimizer,
    OptimizationParameter,
    OptimizationObjective,
    RiskMetrics,
    OptimizationResult,
    StrategyOptimizerConfig
)

# 禁用警告以保持测试输出清洁
warnings.filterwarnings('ignore')


class TestFinancialStatisticalValidator(unittest.TestCase):
    """金融统计验证器全面测试"""

    def setUp(self):
        """测试初始化"""
        self.validator = FinancialStatisticalValidator(
            alpha_level=0.05,
            min_power=0.8,
            min_effect_size=0.2
        )

        # 生成测试数据
        np.random.seed(42)  # 确保测试结果可重现
        self.sample_returns = np.random.normal(0.08, 0.15, 100)
        self.benchmark_returns = np.random.normal(0.05, 0.12, 100)
        self.control_returns = np.random.normal(0.03, 0.10, 80)

    def test_validator_initialization(self):
        """测试验证器初始化"""
        validator = FinancialStatisticalValidator()
        self.assertEqual(validator.alpha_level, 0.05)
        self.assertEqual(validator.min_power, 0.8)
        self.assertEqual(validator.min_effect_size, 0.2)

    def test_statistical_significance_validation(self):
        """测试统计显著性验证功能"""
        results = self.validator.validate_strategy_statistical_significance(
            returns_data=self.sample_returns.tolist(),
            benchmark_returns=self.benchmark_returns.tolist(),
            control_group_returns=self.control_returns.tolist()
        )

        # 验证返回结果结构完整性
        required_keys = [
            'sample_size_analysis', 'normality_tests', 'significance_tests',
            'effect_size_analysis', 'multiple_testing_correction', 'overall_assessment'
        ]
        for key in required_keys:
            self.assertIn(key, results)

        # 验证样本量分析
        sample_analysis = results['sample_size_analysis']
        self.assertIsInstance(sample_analysis, SampleSizeAnalysis)
        self.assertGreater(sample_analysis.current_sample_size, 0)

        # 验证正态性检验结果
        normality_tests = results['normality_tests']
        self.assertIsInstance(normality_tests, dict)

        # 验证显著性检验结果
        significance_tests = results['significance_tests']
        self.assertIsInstance(significance_tests, dict)

    def test_pattern_statistical_robustness(self):
        """测试技术形态模式统计稳健性验证"""
        # 生成模拟技术形态特征数据
        pattern_features = [
            {
                'rsi': 30.5 + np.random.normal(0, 5),
                'macd': 0.1 + np.random.normal(0, 0.05),
                'volume_ratio': 1.2 + np.random.normal(0, 0.3),
                'price_change': 0.02 + np.random.normal(0, 0.01)
            }
            for _ in range(50)
        ]
        success_outcomes = [np.random.random() > 0.4 for _ in range(50)]

        results = self.validator.validate_pattern_statistical_robustness(
            pattern_features=pattern_features,
            success_outcomes=success_outcomes
        )

        # 验证结果结构
        expected_keys = [
            'feature_significance_tests', 'pattern_association_tests',
            'cross_validation_results', 'generalization_analysis'
        ]
        for key in expected_keys:
            self.assertIn(key, results)

    def test_sample_adequacy_analysis(self):
        """测试样本量充足性分析"""
        analysis = self.validator._analyze_sample_adequacy(self.sample_returns.tolist())

        self.assertIsInstance(analysis, SampleSizeAnalysis)
        self.assertEqual(analysis.current_sample_size, len(self.sample_returns))
        self.assertIn(analysis.adequacy_status, ['ADEQUATE', 'MARGINAL', 'INADEQUATE'])
        self.assertGreater(analysis.power_achieved, 0)

    def test_normality_testing(self):
        """测试正态性检验功能"""
        # 测试正态分布数据
        normal_data = np.random.normal(0, 1, 100)
        results = self.validator._test_normality(normal_data.tolist())

        self.assertIsInstance(results, dict)
        # 应该包含多种正态性检验结果

        # 测试非正态分布数据
        non_normal_data = np.random.exponential(2, 100)
        results_non_normal = self.validator._test_normality(non_normal_data.tolist())
        self.assertIsInstance(results_non_normal, dict)

    def test_effect_size_calculation(self):
        """测试效应量计算"""
        effect_sizes = self.validator._calculate_effect_sizes(
            strategy_returns=self.sample_returns.tolist(),
            benchmark_returns=self.benchmark_returns.tolist(),
            control_returns=self.control_returns.tolist()
        )

        self.assertIsInstance(effect_sizes, dict)
        self.assertIn('cohen_d_vs_benchmark', effect_sizes)
        self.assertIn('cohen_d_vs_control', effect_sizes)
        self.assertIn('cohen_d_vs_zero', effect_sizes)

    def test_multiple_testing_correction(self):
        """测试多重假设检验校正"""
        # 模拟显著性检验结果
        mock_test_result = StatisticalTestResult(
            test_type=StatisticalTestType.T_TEST,
            statistic=2.5,
            p_value=0.02
        )

        significance_tests = {
            'test_group_1': {'test_1': mock_test_result, 'test_2': mock_test_result},
            'test_group_2': {'test_3': mock_test_result}
        }

        corrections = self.validator._correct_multiple_testing(significance_tests)
        self.assertIsInstance(corrections, dict)

    def test_statistical_validator_singleton(self):
        """测试全局验证器单例模式"""
        validator1 = get_statistical_validator()
        validator2 = get_statistical_validator()
        self.assertIs(validator1, validator2)


class TestNumericalStabilityManager(unittest.TestCase):
    """数值稳定性管理器全面测试"""

    def setUp(self):
        """测试初始化"""
        self.stability_manager = NumericalStabilityManager(precision=6)
        self.test_series = pd.Series([1.123456789, 2.987654321, 3.555555555, 4.999999999])

    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = NumericalStabilityManager(precision=8, decimal_context_precision=30)
        self.assertEqual(manager.precision, 8)
        self.assertEqual(manager.decimal_context_precision, 30)

    def test_series_precision_control(self):
        """测试Series精度控制"""
        # 测试默认精度（6位小数）
        precise_series = self.stability_manager.ensure_series_precision(self.test_series)
        self.assertTrue(all(len(str(val).split('.')[-1]) <= 6 for val in precise_series if not pd.isna(val)))

        # 测试自定义精度
        custom_precise = self.stability_manager.ensure_series_precision(self.test_series, 2)
        self.assertTrue(all(len(str(val).split('.')[-1]) <= 2 for val in custom_precise if not pd.isna(val)))

    def test_extreme_values_handling(self):
        """测试极值处理"""
        # 创建包含极值的测试数据
        extreme_series = pd.Series([1.0, 1e15, -1e15, np.inf, -np.inf, np.nan, 100.5])

        fixed_series = self.stability_manager.check_and_fix_extreme_values(extreme_series, "test_series")

        # 验证无穷大值已被处理
        self.assertFalse(np.isinf(fixed_series).any())

        # 验证极大值已被处理
        finite_values = fixed_series[~pd.isna(fixed_series)]
        self.assertTrue(all(abs(val) <= self.stability_manager.EXTREME_VALUE_THRESHOLD for val in finite_values))

    def test_safe_division(self):
        """测试安全除法运算"""
        # 测试Series除法
        numerator = pd.Series([10.0, 20.0, 30.0])
        denominator = pd.Series([2.0, 0.0, 5.0])

        result = self.stability_manager.safe_division(numerator, denominator, default_value=0.0)

        expected = pd.Series([5.0, 0.0, 6.0])  # 分母为0时返回默认值0.0
        pd.testing.assert_series_equal(result, expected)

        # 测试标量除法
        scalar_result = self.stability_manager.safe_division(10.0, 2.5)
        self.assertEqual(scalar_result, 4.0)

        # 测试除零情况
        zero_div_result = self.stability_manager.safe_division(10.0, 0.0, default_value=-999.0)
        self.assertEqual(zero_div_result, -999.0)

    def test_safe_sqrt(self):
        """测试安全开方运算"""
        # 测试Series开方
        test_values = pd.Series([4.0, -1.0, 0.0, 9.0])
        result = self.stability_manager.safe_sqrt(test_values)

        self.assertEqual(result[0], 2.0)  # sqrt(4) = 2
        self.assertTrue(pd.isna(result[1]))  # sqrt(-1) = NaN
        self.assertEqual(result[2], 0.0)  # sqrt(0) = 0
        self.assertEqual(result[3], 3.0)  # sqrt(9) = 3

        # 测试标量开方
        scalar_result = self.stability_manager.safe_sqrt(16.0)
        self.assertEqual(scalar_result, 4.0)

        negative_sqrt = self.stability_manager.safe_sqrt(-4.0)
        self.assertTrue(pd.isna(negative_sqrt))

    def test_safe_log(self):
        """测试安全对数运算"""
        # 测试Series对数
        test_values = pd.Series([1.0, np.e, 0.0, -1.0, 10.0])
        result = self.stability_manager.safe_log(test_values)

        self.assertAlmostEqual(result[0], 0.0, places=5)  # ln(1) = 0
        self.assertAlmostEqual(result[1], 1.0, places=5)  # ln(e) = 1
        self.assertTrue(pd.isna(result[2]))  # ln(0) = NaN
        self.assertTrue(pd.isna(result[3]))  # ln(-1) = NaN

        # 测试指定底数的对数
        log10_result = self.stability_manager.safe_log(test_values, base=10.0)
        self.assertAlmostEqual(log10_result[4], 1.0, places=5)  # log10(10) = 1

    def test_indicator_range_validation(self):
        """测试指标范围验证"""
        test_series = pd.Series([50.0, 120.0, -10.0, 75.0])

        # 测试通用范围验证
        validated = self.stability_manager.validate_indicator_range(
            test_series, "test_indicator", min_value=0.0, max_value=100.0
        )

        self.assertEqual(validated[0], 50.0)   # 正常值保持不变
        self.assertEqual(validated[1], 100.0)  # 超出最大值被截断
        self.assertEqual(validated[2], 0.0)    # 低于最小值被截断
        self.assertEqual(validated[3], 75.0)   # 正常值保持不变

    def test_rsi_range_validation(self):
        """测试RSI范围验证"""
        rsi_series = pd.Series([30.0, 120.0, -5.0, 85.5])
        validated_rsi = self.stability_manager.validate_rsi_range(rsi_series)

        # RSI应该在0-100范围内
        self.assertTrue(all(0.0 <= val <= 100.0 for val in validated_rsi))

    def test_calculation_anomalies_detection(self):
        """测试计算异常检测"""
        anomalous_series = pd.Series([
            1.0, np.nan, np.inf, -np.inf, 1e12, 0.0, 1e-15, 50.0
        ])

        anomalies = self.stability_manager.detect_calculation_anomalies(anomalous_series, "test_series")

        self.assertIn('nan_count', anomalies)
        self.assertIn('inf_count', anomalies)
        self.assertIn('extreme_count', anomalies)
        self.assertIn('zero_count', anomalies)

        self.assertEqual(anomalies['nan_count'], 1)
        self.assertEqual(anomalies['inf_count'], 2)  # +inf and -inf

    def test_precision_controlled_operation(self):
        """测试精度控制运算"""
        def test_operation(x, y):
            return x / y * 1.123456789

        result = self.stability_manager.precision_controlled_operation(
            test_operation, 10.0, 3.0
        )

        # 结果应该被四舍五入到指定精度
        self.assertEqual(len(str(result).split('.')[-1]), 6)

    def test_global_functions(self):
        """测试全局便捷函数"""
        test_series = pd.Series([1.123456789, 2.987654321])

        # 测试ensure_precision
        precise = ensure_precision(test_series)
        self.assertTrue(all(len(str(val).split('.')[-1]) <= 6 for val in precise))

        # 测试safe_divide
        safe_div = safe_divide(10.0, 3.0)
        self.assertAlmostEqual(safe_div, 3.333333, places=6)

        # 测试validate_rsi
        rsi_series = pd.Series([30.0, 120.0])
        validated = validate_rsi(rsi_series)
        self.assertTrue(all(0.0 <= val <= 100.0 for val in validated))

        # 测试check_anomalies
        anomalous = pd.Series([1.0, np.nan, np.inf])
        anomalies = check_anomalies(anomalous, "test")
        self.assertIsInstance(anomalies, dict)


class TestCompleteIndicatorRegistry(unittest.TestCase):
    """完整指标注册表测试"""

    def setUp(self):
        """测试初始化"""
        self.registry = CompleteIndicatorRegistry()

    def test_registry_initialization(self):
        """测试注册表初始化"""
        self.assertIsInstance(self.registry._indicators, dict)
        self.assertIsInstance(self.registry._failed_indicators, list)
        self.assertIsInstance(self.registry._registration_log, list)

    def test_indicator_registration(self):
        """测试指标注册功能"""
        registered_count = self.registry.register_all_indicators()

        # 验证注册数量
        self.assertGreater(registered_count, 80)  # 应该注册至少80个指标

        # 验证注册统计
        stats = self.registry.get_registration_stats()
        self.assertIn('total_indicators', stats)
        self.assertIn('successful_indicators', stats)
        self.assertIn('failed_indicators', stats)
        self.assertIn('success_rate', stats)

        # 验证成功率
        self.assertGreater(stats['success_rate'], 0.8)  # 至少80%成功率

    def test_core_indicators_registration(self):
        """测试核心指标注册"""
        core_count = self.registry._register_core_indicators()
        self.assertGreaterEqual(core_count, 6)  # 至少6个核心指标

    def test_indicator_retrieval(self):
        """测试指标获取功能"""
        self.registry.register_all_indicators()

        # 测试获取存在的指标
        all_indicators = self.registry.get_all_indicators()
        self.assertIsInstance(all_indicators, dict)
        self.assertGreater(len(all_indicators), 0)

        # 测试获取指标数量
        count = self.registry.get_indicator_count()
        self.assertEqual(count, len(all_indicators))

    def test_failed_indicators_tracking(self):
        """测试失败指标追踪"""
        self.registry.register_all_indicators()

        failed_indicators = self.registry.get_failed_indicators()
        self.assertIsInstance(failed_indicators, list)

    def test_global_registry_instance(self):
        """测试全局注册表实例"""
        global_registry = get_indicator_registry()
        self.assertIsInstance(global_registry, CompleteIndicatorRegistry)

        # 测试指标初始化
        count = initialize_indicators()
        self.assertGreater(count, 0)


class TestStrategyOptimizer(unittest.TestCase):
    """策略优化器全面测试"""

    def setUp(self):
        """测试初始化"""
        self.config = StrategyOptimizerConfig(
            optimization_method='grid',
            max_iterations=20,
            enable_validation=False  # 简化测试
        )
        self.optimizer = StrategyOptimizer(self.config)

        # 创建模拟策略
        self.mock_strategy = self._create_mock_strategy()

        # 创建模拟买点数据
        self.mock_buypoints = self._create_mock_buypoints()

    def _create_mock_strategy(self):
        """创建模拟策略"""
        mock_strategy = Mock()
        mock_strategy.strategy_id = "test_strategy_001"
        mock_strategy.strategy_name = "Test Strategy"
        mock_strategy.confidence_level = 0.75
        mock_strategy.optimization_params = {
            'min_score_threshold': 60.0,
            'max_results': 50,
            'risk_adjustment_factor': 1.0
        }
        mock_strategy.performance_metrics = {
            'expected_success_rate': 0.65
        }
        return mock_strategy

    def _create_mock_buypoints(self):
        """创建模拟买点数据"""
        buypoints = []
        for i in range(50):
            buypoint = Mock()
            buypoint.buypoint_date = f"2023-01-{i+1:02d}"
            buypoint.stock_code = f"00000{i}"
            buypoint.score = 60 + np.random.random() * 30
            buypoints.append(buypoint)
        return buypoints

    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        self.assertIsInstance(self.optimizer.config, StrategyOptimizerConfig)
        self.assertIsInstance(self.optimizer.default_parameter_space, list)
        self.assertIsInstance(self.optimizer.default_objectives, list)

    def test_parameter_space_definition(self):
        """测试参数空间定义"""
        param_space = self.optimizer.default_parameter_space

        # 验证参数空间完整性
        param_names = [param.name for param in param_space]
        expected_params = [
            'min_score_threshold', 'max_results', 'risk_adjustment_factor',
            'pattern_weight_multiplier', 'confidence_threshold'
        ]

        for expected_param in expected_params:
            self.assertIn(expected_param, param_names)

    def test_optimization_objectives(self):
        """测试优化目标定义"""
        objectives = self.optimizer.default_objectives

        # 验证目标完整性
        objective_names = [obj.name for obj in objectives]
        expected_objectives = [
            'expected_return', 'sharpe_ratio', 'success_rate', 'max_drawdown'
        ]

        for expected_obj in expected_objectives:
            self.assertIn(expected_obj, objective_names)

    def test_risk_metrics_calculation(self):
        """测试风险指标计算"""
        params = {
            'min_score_threshold': 65.0,
            'risk_adjustment_factor': 1.2,
            'confidence_threshold': 0.6
        }

        risk_metrics = self.optimizer._calculate_risk_metrics(
            self.mock_strategy, self.mock_buypoints, params
        )

        self.assertIsInstance(risk_metrics, RiskMetrics)
        self.assertGreaterEqual(risk_metrics.max_drawdown, 0)
        self.assertGreaterEqual(risk_metrics.volatility, 0)
        self.assertGreaterEqual(risk_metrics.success_rate, 0)
        self.assertLessEqual(risk_metrics.success_rate, 1)

    def test_performance_evaluation(self):
        """测试策略性能评估"""
        params = {
            'min_score_threshold': 70.0,
            'risk_adjustment_factor': 1.1,
            'confidence_threshold': 0.7
        }

        performance = self.optimizer._evaluate_strategy_performance(
            self.mock_strategy, self.mock_buypoints, params
        )

        self.assertIn('expected_return', performance)
        self.assertIn('expected_volatility', performance)
        self.assertIn('adjusted_success_rate', performance)

    def test_parameter_generation(self):
        """测试参数生成"""
        param_space = self.optimizer.default_parameter_space

        # 测试随机参数生成
        random_params = self.optimizer._generate_random_parameters(param_space)

        for param in param_space:
            self.assertIn(param.name, random_params)
            value = random_params[param.name]
            self.assertGreaterEqual(value, param.min_value)
            self.assertLessEqual(value, param.max_value)

    def test_objective_score_calculation(self):
        """测试目标分数计算"""
        params = {'min_score_threshold': 60.0, 'risk_adjustment_factor': 1.0}
        objectives = self.optimizer.default_objectives

        score = self.optimizer._calculate_objective_score(
            self.mock_strategy, self.mock_buypoints, params, objectives
        )

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_optimization_statistics(self):
        """测试优化统计功能"""
        stats = self.optimizer.get_optimization_statistics()

        expected_keys = [
            'total_optimizations', 'successful_optimizations',
            'average_improvement', 'best_score_achieved',
            'average_optimization_time', 'success_rate',
            'recent_optimizations'
        ]

        for key in expected_keys:
            self.assertIn(key, stats)


class TestPerformanceRequirements(unittest.TestCase):
    """性能要求测试"""

    def setUp(self):
        """性能测试初始化"""
        self.stability_manager = get_stability_manager()
        self.validator = get_statistical_validator()

    def test_indicator_calculation_performance(self):
        """测试指标计算性能（≤0.05秒/股票）"""
        # 模拟单只股票的技术指标计算
        stock_data = pd.DataFrame({
            'close': 100 + np.random.randn(252) * 2,
            'high': 102 + np.random.randn(252) * 2,
            'low': 98 + np.random.randn(252) * 2,
            'volume': 1000000 + np.random.randint(0, 500000, 252)
        })

        start_time = time.time()

        # 执行多个技术指标计算
        for _ in range(10):  # 模拟10个指标计算
            # RSI计算模拟
            returns = stock_data['close'].pct_change()
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)

            # 使用数值稳定性管理器
            avg_gains = self.stability_manager.ensure_series_precision(gains.rolling(14).mean())
            avg_losses = self.stability_manager.ensure_series_precision(losses.rolling(14).mean())

            rs = self.stability_manager.safe_division(avg_gains, avg_losses)
            rsi = 100 - (100 / (1 + rs))
            rsi = self.stability_manager.validate_rsi_range(rsi)

        execution_time = time.time() - start_time

        # 验证性能要求：≤0.05秒/股票
        self.assertLessEqual(execution_time, 0.05,
                           f"指标计算耗时 {execution_time:.4f}秒，超过0.05秒限制")

    def test_numerical_precision_accuracy(self):
        """测试数值精度要求（6位小数）"""
        # 测试高精度计算
        test_values = [1.123456789, 2.987654321, 3.141592653589793]

        for value in test_values:
            series = pd.Series([value])
            precise_series = self.stability_manager.ensure_series_precision(series, 6)

            # 验证精度控制
            precision_check = str(precise_series.iloc[0]).split('.')
            if len(precision_check) > 1:
                decimal_places = len(precision_check[1])
                self.assertLessEqual(decimal_places, 6,
                                   f"精度超过6位小数：{precise_series.iloc[0]}")

    def test_statistical_validation_performance(self):
        """测试统计验证性能"""
        # 生成大量测试数据
        large_returns = np.random.normal(0.05, 0.12, 1000)

        start_time = time.time()

        # 执行统计验证
        results = self.validator.validate_strategy_statistical_significance(
            returns_data=large_returns.tolist()
        )

        execution_time = time.time() - start_time

        # 验证统计验证不应该过慢（合理时间限制）
        self.assertLessEqual(execution_time, 10.0,
                           f"统计验证耗时 {execution_time:.2f}秒，过于耗时")

        # 验证结果质量
        self.assertIn('overall_assessment', results)
        assessment = results['overall_assessment']
        self.assertIn('overall_recommendation', assessment)


class TestSystemIntegration(unittest.TestCase):
    """系统集成测试"""

    def test_component_integration(self):
        """测试组件集成"""
        # 测试数值稳定性管理器与统计验证器的集成
        stability_manager = get_stability_manager()
        validator = get_statistical_validator()

        # 生成测试数据
        test_data = np.random.normal(0.08, 0.15, 100)

        # 通过稳定性管理器处理数据
        processed_series = stability_manager.ensure_series_precision(pd.Series(test_data))

        # 使用统计验证器验证处理后的数据
        results = validator.validate_strategy_statistical_significance(
            returns_data=processed_series.tolist()
        )

        # 验证集成结果
        self.assertIn('overall_assessment', results)
        self.assertIsNotNone(results['sample_size_analysis'])

    def test_error_handling_robustness(self):
        """测试错误处理稳健性"""
        stability_manager = get_stability_manager()

        # 测试异常数据处理
        problematic_data = pd.Series([np.nan, np.inf, -np.inf, 1e20, -1e20, 0])

        # 应该不会抛出异常
        try:
            cleaned_data = stability_manager.check_and_fix_extreme_values(problematic_data)
            anomalies = stability_manager.detect_calculation_anomalies(cleaned_data, "test")
            self.assertIsInstance(anomalies, dict)
        except Exception as e:
            self.fail(f"错误处理失败，抛出异常: {e}")

    def test_memory_efficiency(self):
        """测试内存效率"""
        import psutil
        import gc

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 执行大量计算
        stability_manager = get_stability_manager()
        for i in range(100):
            large_series = pd.Series(np.random.randn(10000))
            processed = stability_manager.ensure_series_precision(large_series)
            del processed, large_series

            if i % 10 == 0:
                gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # 内存增长应该在合理范围内（<100MB）
        self.assertLess(memory_increase, 100,
                       f"内存增长过多: {memory_increase:.2f}MB")


def run_comprehensive_financial_tests():
    """运行全面的金融系统测试"""

    print("=" * 80)
    print("生产级金融系统全面验证测试")
    print("=" * 80)

    # 创建测试套件
    test_suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        TestFinancialStatisticalValidator,
        TestNumericalStabilityManager,
        TestCompleteIndicatorRegistry,
        TestStrategyOptimizer,
        TestPerformanceRequirements,
        TestSystemIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )

    start_time = time.time()
    result = runner.run(test_suite)
    execution_time = time.time() - start_time

    # 生成测试报告
    print("\n" + "=" * 80)
    print("测试执行总结")
    print("=" * 80)
    print(f"总测试数量: {result.testsRun}")
    print(f"成功测试: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败测试: {len(result.failures)}")
    print(f"错误测试: {len(result.errors)}")
    print(f"执行时间: {execution_time:.2f} 秒")

    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) /
                   result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"测试成功率: {success_rate:.1f}%")

    # 详细失败信息
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    print("=" * 80)

    return result


if __name__ == '__main__':
    run_comprehensive_financial_tests()