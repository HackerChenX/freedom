#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生产级金融逻辑验证测试套件

对第二周金融逻辑完善代码进行全面的生产级测试验证：
1. 技术指标计算精度验证（6位小数精度）
2. NumericalStabilityManager数值稳定性管理器测试
3. 金融风险指标计算验证（VaR等）
4. 性能基准测试（≤0.05秒/股票）
5. 集成测试与压力测试
"""

import unittest
import numpy as np
import pandas as pd
import time
from decimal import Decimal
from typing import Dict, List, Any
import tempfile
import os
import sys
import warnings

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from utils.numerical_stability_manager import NumericalStabilityManager
from indicators.zxm.risk_control_indicators import ZXMRiskControl

warnings.filterwarnings('ignore')


class TestNumericalStabilityManager(unittest.TestCase):
    """测试NumericalStabilityManager数值稳定性管理器"""

    def setUp(self):
        """设置测试环境"""
        self.stability_manager = NumericalStabilityManager(precision=6)

        # 创建测试数据
        self.test_series = pd.Series([
            1.123456789, 2.987654321, 3.141592653, 0.000001234,
            np.inf, -np.inf, np.nan, 1e10, -1e10
        ])

        self.normal_series = pd.Series(np.random.normal(50, 15, 100))

    def test_precision_control(self):
        """测试6位小数精度控制"""
        print("\n=== 测试6位小数精度控制 ===")

        # 测试Series精度确保
        precise_series = self.stability_manager.ensure_series_precision(self.test_series)

        for value in precise_series.dropna():
            if not np.isinf(value):
                # 验证精度不超过6位小数
                decimal_str = str(Decimal(str(value)))
                if '.' in decimal_str:
                    decimal_places = len(decimal_str.split('.')[1])
                    self.assertLessEqual(decimal_places, 6,
                                       f"Value {value} has more than 6 decimal places")

        print(f"✓ Series精度控制: 所有有效值精度≤6位小数")

        # 测试标量精度控制
        test_values = [1.123456789, 3.141592653589793, 2.718281828459045]
        for value in test_values:
            result = self.stability_manager.precision_controlled_operation(
                lambda x: x, value
            )
            if not pd.isna(result):
                decimal_str = str(Decimal(str(result)))
                if '.' in decimal_str:
                    decimal_places = len(decimal_str.split('.')[1])
                    self.assertLessEqual(decimal_places, 6)

        print(f"✓ 标量精度控制: 精度控制正常")

    def test_extreme_value_detection(self):
        """测试极值检测和修正"""
        print("\n=== 测试极值检测和修正 ===")

        corrected_series = self.stability_manager.check_and_fix_extreme_values(
            self.test_series, name="TestSeries"
        )

        # 验证无穷大值被修正为NaN
        inf_count_original = np.isinf(self.test_series).sum()
        inf_count_corrected = np.isinf(corrected_series).sum()

        self.assertEqual(inf_count_corrected, 0, "Infinite values not properly handled")
        print(f"✓ 无穷大值修正: {inf_count_original} → 0")

        # 验证极值被修正
        extreme_mask = np.abs(corrected_series) > self.stability_manager.EXTREME_VALUE_THRESHOLD
        extreme_count = extreme_mask.sum()
        self.assertEqual(extreme_count, 0, "Extreme values not properly handled")
        print(f"✓ 极值修正: 所有极值已修正")

    def test_safe_mathematical_operations(self):
        """测试安全数学运算"""
        print("\n=== 测试安全数学运算 ===")

        # 测试安全除法
        numerator = pd.Series([1, 2, 3, 4, 5])
        denominator = pd.Series([2, 0, 1, np.nan, 2])

        result = self.stability_manager.safe_division(numerator, denominator)

        # 验证除零情况被正确处理
        self.assertTrue(pd.isna(result.iloc[1]), "Division by zero not handled")
        self.assertTrue(pd.isna(result.iloc[3]), "Division by NaN not handled")
        self.assertAlmostEqual(result.iloc[0], 0.5, places=6)
        self.assertAlmostEqual(result.iloc[2], 3.0, places=6)

        print(f"✓ 安全除法: 除零和NaN情况正确处理")

        # 测试安全开方
        test_values = pd.Series([4, 9, -1, 0, np.nan])
        sqrt_result = self.stability_manager.safe_sqrt(test_values)

        self.assertAlmostEqual(sqrt_result.iloc[0], 2.0, places=6)
        self.assertAlmostEqual(sqrt_result.iloc[1], 3.0, places=6)
        self.assertTrue(pd.isna(sqrt_result.iloc[2]), "Negative square root not handled")

        print(f"✓ 安全开方: 负值情况正确处理")

        # 测试安全对数
        log_values = pd.Series([1, np.e, -1, 0, np.nan])
        log_result = self.stability_manager.safe_log(log_values)

        self.assertAlmostEqual(log_result.iloc[0], 0.0, places=6)
        self.assertAlmostEqual(log_result.iloc[1], 1.0, places=6)
        self.assertTrue(pd.isna(log_result.iloc[2]), "Negative log not handled")
        self.assertTrue(pd.isna(log_result.iloc[3]), "Zero log not handled")

        print(f"✓ 安全对数: 非正值情况正确处理")

    def test_indicator_range_validation(self):
        """测试指标范围验证"""
        print("\n=== 测试指标范围验证 ===")

        # 测试RSI范围验证
        rsi_values = pd.Series([-10, 50, 110, 30, 70, np.nan])
        validated_rsi = self.stability_manager.validate_rsi_range(rsi_values)

        # 验证RSI范围在0-100之间
        valid_values = validated_rsi.dropna()
        self.assertTrue(all(valid_values >= 0), "RSI values below 0 not corrected")
        self.assertTrue(all(valid_values <= 100), "RSI values above 100 not corrected")

        print(f"✓ RSI范围验证: 所有值在0-100范围内")

        # 测试自定义范围验证
        custom_series = pd.Series([5, 15, 25, -5, 35])
        validated_custom = self.stability_manager.validate_indicator_range(
            custom_series, "CustomIndicator", min_value=0, max_value=30
        )

        valid_custom = validated_custom.dropna()
        self.assertTrue(all(valid_custom >= 0), "Custom min value not enforced")
        self.assertTrue(all(valid_custom <= 30), "Custom max value not enforced")

        print(f"✓ 自定义范围验证: 正确执行")

    def test_anomaly_detection(self):
        """测试异常检测"""
        print("\n=== 测试异常检测 ===")

        anomalies = self.stability_manager.detect_calculation_anomalies(
            self.test_series, "TestSeries"
        )

        # 验证异常统计结果
        expected_keys = ['nan_count', 'inf_count', 'extreme_count', 'zero_count']
        for key in expected_keys:
            self.assertIn(key, anomalies, f"Missing anomaly key: {key}")
            self.assertIsInstance(anomalies[key], (int, np.integer))

        print(f"✓ 异常检测统计: {anomalies}")


class TestZXMRiskControlIndicators(unittest.TestCase):
    """测试ZXM风险控制指标计算"""

    def setUp(self):
        """设置测试环境"""
        self.risk_indicator = ZXMRiskControl()

        # 创建模拟股价数据
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # 生成具有趋势和波动的价格数据
        price_base = 100
        returns = np.random.normal(0.001, 0.02, 100)  # 日收益率
        prices = [price_base]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        self.test_data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * np.random.uniform(1.0, 1.02) for p in prices],
            'low': [p * np.random.uniform(0.98, 1.0) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        })

    def test_risk_indicator_calculation(self):
        """测试风险指标计算"""
        print("\n=== 测试ZXM风险控制指标计算 ===")

        start_time = time.time()
        result = self.risk_indicator.calculate(self.test_data)
        calculation_time = time.time() - start_time

        # 验证性能要求（应该远小于0.05秒/股票）
        self.assertLess(calculation_time, 0.05,
                       f"Calculation too slow: {calculation_time:.4f}s > 0.05s")

        print(f"✓ 计算性能: {calculation_time:.6f}s < 0.05s")

        # 验证必要的风险指标列
        required_columns = [
            'VolatilityRisk', 'DrawdownRisk', 'VaRRisk', 'CompositeRiskScore',
            'HighRiskSignal', 'ModerateRiskSignal', 'LowRiskSignal',
            'RiskWarningSignal', 'RiskControlSignal'
        ]

        for col in required_columns:
            self.assertIn(col, result.columns, f"Missing required column: {col}")

        # 验证风险评分范围（0-100）
        risk_columns = ['VolatilityRisk', 'DrawdownRisk', 'VaRRisk', 'CompositeRiskScore']
        for col in risk_columns:
            valid_values = result[col].dropna()
            if len(valid_values) > 0:
                self.assertTrue(all(valid_values >= 0), f"{col} has values below 0")
                self.assertTrue(all(valid_values <= 100), f"{col} has values above 100")

        print(f"✓ 风险指标列: {len(required_columns)}个必需列存在")
        print(f"✓ 风险评分范围: 所有评分在0-100范围内")

        # 验证数值精度
        for col in risk_columns:
            valid_values = result[col].dropna()
            for value in valid_values:
                decimal_str = str(Decimal(str(value)))
                if '.' in decimal_str:
                    decimal_places = len(decimal_str.split('.')[1])
                    self.assertLessEqual(decimal_places, 6,
                                       f"{col} value {value} exceeds 6 decimal places")

        print(f"✓ 精度验证: 所有风险指标精度≤6位小数")

    def test_financial_risk_metrics(self):
        """测试金融风险指标计算（VaR、CVaR、波动率等）"""
        print("\n=== 测试金融风险指标 ===")

        result = self.risk_indicator.calculate(self.test_data)

        # 测试VaR计算
        var_values = result['VaR'].dropna()
        self.assertGreater(len(var_values), 0, "No VaR values calculated")

        # VaR值应该为负（损失）
        negative_var_count = (var_values < 0).sum()
        self.assertGreater(negative_var_count, 0, "VaR values should be negative (losses)")

        print(f"✓ VaR计算: {len(var_values)}个有效值，{negative_var_count}个负值")

        # 测试波动率计算
        volatility_values = result['Volatility'].dropna()
        self.assertGreater(len(volatility_values), 0, "No volatility values calculated")

        # 波动率应该为正
        positive_vol_count = (volatility_values > 0).sum()
        self.assertGreater(positive_vol_count, 0, "Volatility values should be positive")

        print(f"✓ 波动率计算: {len(volatility_values)}个有效值，{positive_vol_count}个正值")

        # 测试回撤计算
        drawdown_values = result['Drawdown'].dropna()
        self.assertGreater(len(drawdown_values), 0, "No drawdown values calculated")

        # 回撤应该为负或零
        non_positive_dd_count = (drawdown_values <= 0).sum()
        self.assertEqual(non_positive_dd_count, len(drawdown_values),
                        "Drawdown values should be non-positive")

        print(f"✓ 回撤计算: {len(drawdown_values)}个有效值，全部≤0")


class TestPerformanceBenchmark(unittest.TestCase):
    """性能基准测试"""

    def setUp(self):
        """设置测试环境"""
        self.target_time_per_stock = 0.05  # 秒
        self.stability_manager = NumericalStabilityManager()
        self.risk_indicator = ZXMRiskControl()

    def test_numerical_stability_performance(self):
        """测试数值稳定性管理器性能"""
        print("\n=== 测试数值稳定性管理器性能 ===")

        # 创建大数据集
        large_series = pd.Series(np.random.normal(50, 15, 10000))

        start_time = time.time()

        # 执行多项数值稳定性操作
        precise_series = self.stability_manager.ensure_series_precision(large_series)
        corrected_series = self.stability_manager.check_and_fix_extreme_values(precise_series)
        anomalies = self.stability_manager.detect_calculation_anomalies(corrected_series, "PerfTest")

        elapsed_time = time.time() - start_time
        operations_per_second = 10000 / elapsed_time

        # 验证性能（应该能处理大量数据）
        self.assertGreater(operations_per_second, 1000,
                          f"Performance too slow: {operations_per_second:.0f} ops/sec")

        print(f"✓ 数值稳定性性能: {operations_per_second:.0f} 操作/秒")
        print(f"✓ 处理时间: {elapsed_time:.4f}秒 (10,000个数据点)")

    def test_integrated_performance_benchmark(self):
        """集成性能基准测试"""
        print("\n=== 集成性能基准测试 ===")

        # 模拟单只股票的完整分析流程
        stock_count = 20
        total_start_time = time.time()

        for i in range(stock_count):
            # 生成单只股票数据
            np.random.seed(42 + i)
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            prices = np.random.normal(100, 10, 100)
            prices = np.cumsum(np.random.normal(0, 1, 100)) + 100  # 随机游走价格

            stock_data = pd.DataFrame({
                'date': dates,
                'open': prices * np.random.uniform(0.99, 1.01, 100),
                'high': prices * np.random.uniform(1.00, 1.02, 100),
                'low': prices * np.random.uniform(0.98, 1.00, 100),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, 100)
            })

            stock_start_time = time.time()

            # 完整分析流程
            # 1. 风险指标计算
            risk_result = self.risk_indicator.calculate(stock_data)

            # 2. 数值稳定性验证
            composite_risk = risk_result['CompositeRiskScore']
            validated_risk = self.stability_manager.ensure_series_precision(composite_risk)
            anomalies = self.stability_manager.detect_calculation_anomalies(validated_risk, f"Stock_{i}")

            stock_elapsed = time.time() - stock_start_time

            # 验证单股处理时间
            self.assertLess(stock_elapsed, self.target_time_per_stock,
                          f"Stock {i} processing too slow: {stock_elapsed:.4f}s > {self.target_time_per_stock}s")

        total_elapsed = time.time() - total_start_time
        avg_time_per_stock = total_elapsed / stock_count
        stocks_per_hour = 3600 / avg_time_per_stock

        print(f"✓ 总处理时间: {total_elapsed:.4f}秒")
        print(f"✓ 平均每股时间: {avg_time_per_stock:.6f}秒")
        print(f"✓ 每小时处理能力: {stocks_per_hour:.0f}股")
        print(f"✓ 性能达标: {'YES' if avg_time_per_stock <= self.target_time_per_stock else 'NO'}")

        # 验证整体性能目标
        self.assertLessEqual(avg_time_per_stock, self.target_time_per_stock,
                           f"Overall performance target not met: {avg_time_per_stock:.6f}s > {self.target_time_per_stock}s")


class TestIntegrationAndStressTest(unittest.TestCase):
    """集成测试和压力测试"""

    def setUp(self):
        """设置测试环境"""
        self.components = {
            'stability_manager': NumericalStabilityManager(),
            'risk_indicator': ZXMRiskControl()
        }

    def test_component_integration(self):
        """测试组件集成"""
        print("\n=== 测试组件集成 ===")

        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = np.cumsum(np.random.normal(0.001, 0.02, 50)) + 100

        test_data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 50)
        })

        # 集成测试流程
        try:
            # 1. 风险指标计算
            risk_result = self.components['risk_indicator'].calculate(test_data)
            self.assertIsInstance(risk_result, pd.DataFrame)

            # 2. 数值稳定性处理
            stability_mgr = self.components['stability_manager']
            composite_risk = risk_result['CompositeRiskScore']

            validated_risk = stability_mgr.ensure_series_precision(composite_risk)
            corrected_risk = stability_mgr.check_and_fix_extreme_values(validated_risk, "IntegrationTest")

            # 验证集成结果
            self.assertIsInstance(validated_risk, pd.Series)

            print(f"✓ 组件集成: 所有组件正常协作")
            print(f"✓ 数据流: 风险计算 → 稳定性验证")
            print(f"✓ 结果验证: 所有输出格式正确")

        except Exception as e:
            self.fail(f"Integration test failed: {e}")

    def test_stress_test_extreme_conditions(self):
        """压力测试 - 极端条件"""
        print("\n=== 压力测试 - 极端条件 ===")

        # 测试极端数据条件
        extreme_conditions = [
            # 1. 全NaN数据
            pd.Series([np.nan] * 100),
            # 2. 全零数据
            pd.Series([0.0] * 100),
            # 3. 极值数据
            pd.Series([1e10, -1e10, np.inf, -np.inf] * 25),
            # 4. 微小变动数据
            pd.Series([1.0 + i * 1e-10 for i in range(100)]),
            # 5. 高波动数据
            pd.Series(np.random.normal(0, 100, 100))
        ]

        stability_mgr = self.components['stability_manager']

        for i, extreme_data in enumerate(extreme_conditions):
            try:
                # 测试数值稳定性处理
                validated = stability_mgr.ensure_series_precision(extreme_data)
                corrected = stability_mgr.check_and_fix_extreme_values(validated, f"Extreme_{i}")
                anomalies = stability_mgr.detect_calculation_anomalies(corrected, f"Extreme_{i}")

                # 验证系统稳定性
                self.assertIsInstance(validated, pd.Series)
                self.assertIsInstance(anomalies, dict)

                print(f"✓ 极端条件 {i+1}: 系统稳定处理")

            except Exception as e:
                self.fail(f"Stress test failed for extreme condition {i+1}: {e}")

    def test_concurrent_processing_stability(self):
        """测试并发处理稳定性"""
        print("\n=== 测试并发处理稳定性 ===")

        import threading
        import queue

        def worker_function(worker_id, data_queue, result_queue):
            """工作线程函数"""
            try:
                stability_mgr = NumericalStabilityManager()
                while True:
                    data = data_queue.get(timeout=1)
                    if data is None:
                        break

                    # 处理数据
                    validated = stability_mgr.ensure_series_precision(data)
                    corrected = stability_mgr.check_and_fix_extreme_values(validated, f"Worker_{worker_id}")

                    result_queue.put(f"Worker_{worker_id}_success")
                    data_queue.task_done()

            except queue.Empty:
                pass
            except Exception as e:
                result_queue.put(f"Worker_{worker_id}_error: {e}")

        # 创建并发测试
        data_queue = queue.Queue()
        result_queue = queue.Queue()

        # 生成测试数据
        for i in range(20):
            test_series = pd.Series(np.random.normal(50, 15, 100))
            data_queue.put(test_series)

        # 启动工作线程
        threads = []
        for i in range(4):
            thread = threading.Thread(target=worker_function, args=(i, data_queue, result_queue))
            thread.start()
            threads.append(thread)

        # 等待完成
        data_queue.join()

        # 停止工作线程
        for _ in threads:
            data_queue.put(None)

        for thread in threads:
            thread.join()

        # 收集结果
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # 验证并发处理结果
        success_count = sum(1 for r in results if 'success' in r)
        error_count = sum(1 for r in results if 'error' in r)

        self.assertEqual(error_count, 0, f"Concurrent processing errors: {error_count}")
        self.assertGreater(success_count, 0, "No successful concurrent operations")

        print(f"✓ 并发处理: {success_count}个成功操作，{error_count}个错误")
        print(f"✓ 线程安全: 4个并发线程稳定运行")


def run_production_validation_suite():
    """运行完整的生产级验证测试套件"""
    print("="*80)
    print("生产级金融逻辑验证测试套件")
    print("="*80)

    # 创建测试套件
    test_suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        TestNumericalStabilityManager,
        TestZXMRiskControlIndicators,
        TestPerformanceBenchmark,
        TestIntegrationAndStressTest
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)

    # 生成测试报告
    print("\n" + "="*80)
    print("测试结果总结")
    print("="*80)
    print(f"总测试数: {result.testsRun}")
    print(f"成功测试: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败测试: {len(result.failures)}")
    print(f"错误测试: {len(result.errors)}")

    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    # 生成生产就绪性评估
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun

    print("\n" + "="*80)
    print("生产就绪性评估")
    print("="*80)
    print(f"测试通过率: {success_rate:.1%}")

    if success_rate >= 0.95:
        print("✅ 生产就绪状态: READY - 系统达到生产部署标准")
    elif success_rate >= 0.90:
        print("⚠️  生产就绪状态: CONDITIONAL - 需要修复部分问题后部署")
    else:
        print("❌ 生产就绪状态: NOT READY - 需要解决重大问题才能部署")

    return result


if __name__ == "__main__":
    # 运行生产级验证测试套件
    test_result = run_production_validation_suite()

    # 退出码基于测试结果
    sys.exit(0 if test_result.wasSuccessful() else 1)