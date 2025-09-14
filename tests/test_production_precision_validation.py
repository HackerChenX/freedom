#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生产级精度控制验证测试
Senior Quality Assurance Engineer

验证P0/P1问题修复后的精度控制和性能表现
专注于金融级数值稳定性和生产环境性能标准
"""

import pytest
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple, Any
from decimal import Decimal, getcontext

from utils.numerical_stability_manager import NumericalStabilityManager, get_stability_manager
from indicators.zxm.risk_control_indicators import ZXMRiskControl
from utils.technical_utils import calculate_ema_Utils, calculate_macd_Utils, calculate_rsi_Utils


class TestProductionPrecisionValidation:
    """生产级精度控制验证测试套件"""

    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        # 设置高精度计算环境
        getcontext().prec = 28

        # 初始化数值稳定性管理器
        cls.stability_mgr = get_stability_manager()

        # 创建测试数据
        cls.test_data = cls._create_comprehensive_test_data()

        # 性能基准
        cls.performance_threshold = 0.05  # 单股处理时间≤0.05秒
        cls.precision_requirement = 6  # 6位小数精度

    @classmethod
    def _create_comprehensive_test_data(cls) -> pd.DataFrame:
        """创建综合测试数据集，包含各种边界情况"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')

        # 基础价格数据
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, 1000)
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # 构造包含极端情况的数据
        data = pd.DataFrame({
            'date': dates,
            'open': np.array(prices) * np.random.uniform(0.995, 1.005, 1000),
            'high': np.array(prices) * np.random.uniform(1.005, 1.020, 1000),
            'low': np.array(prices) * np.random.uniform(0.980, 0.995, 1000),
            'close': prices,
            'volume': np.random.uniform(1000000, 10000000, 1000)
        })

        # 添加边界测试用例
        # 极小值
        data.loc[100:110, 'close'] = 0.000001
        # 极大值
        data.loc[200:210, 'close'] = 999999.999999
        # 连续相同值
        data.loc[300:310, 'close'] = 150.123456
        # 快速变化
        data.loc[400:410, 'close'] = np.linspace(100, 200, 11)

        return data

    def test_numerical_stability_manager_precision(self):
        """测试数值稳定性管理器的精度控制功能"""
        print("\n=== 测试数值稳定性管理器精度控制 ===")

        # 测试数据
        test_series = pd.Series([
            123.1234567890,  # 超出6位小数
            999.9999999,     # 接近极值
            0.0000001234,    # 极小值
            -456.7890123456  # 负数
        ])

        # 确保精度控制
        result = self.stability_mgr.ensure_series_precision(test_series)

        # 验证精度
        for i, val in enumerate(result):
            if not pd.isna(val):
                decimal_places = len(str(val).split('.')[-1]) if '.' in str(val) else 0
                assert decimal_places <= self.precision_requirement, \
                    f"值 {val} 超出6位小数精度要求"

        print(f"✓ 精度控制验证通过：所有值都符合{self.precision_requirement}位小数要求")

    def test_numerical_stability_manager_extreme_values(self):
        """测试极值处理功能"""
        print("\n=== 测试极值处理功能 ===")

        # 构造包含极值的测试数据
        extreme_data = pd.Series([
            np.inf,           # 正无穷
            -np.inf,          # 负无穷
            1e12,             # 极大值
            -1e12,            # 极大负值
            1e-15,            # 极小值
            np.nan,           # NaN值
            123.456789        # 正常值
        ])

        # 处理极值
        result = self.stability_mgr.check_and_fix_extreme_values(extreme_data, "TestSeries")

        # 验证无穷大值已被处理
        assert not np.isinf(result).any(), "仍存在无穷大值"

        # 验证极值已被处理
        finite_values = result[result.notna()]
        if len(finite_values) > 0:
            max_abs_val = finite_values.abs().max()
            assert max_abs_val <= self.stability_mgr.EXTREME_VALUE_THRESHOLD, \
                f"仍存在超出阈值的极值: {max_abs_val}"

        print("✓ 极值处理验证通过：无穷大值和极值都已正确处理")

    def test_numerical_stability_manager_safe_operations(self):
        """测试安全运算功能"""
        print("\n=== 测试安全运算功能 ===")

        # 测试安全除法
        numerator = pd.Series([10.0, 20.0, 30.0])
        denominator = pd.Series([2.0, 0.0, -5.0])  # 包含零除法

        result = self.stability_mgr.safe_division(numerator, denominator)

        # 验证结果
        assert result.iloc[0] == 5.0, "正常除法计算错误"
        assert pd.isna(result.iloc[1]), "零除法应返回NaN"
        assert result.iloc[2] == -6.0, "负数除法计算错误"

        # 测试安全开方
        sqrt_test = pd.Series([4.0, -1.0, 0.0, 16.0])
        sqrt_result = self.stability_mgr.safe_sqrt(sqrt_test)

        assert sqrt_result.iloc[0] == 2.0, "开方计算错误"
        assert pd.isna(sqrt_result.iloc[1]), "负数开方应返回NaN"
        assert sqrt_result.iloc[2] == 0.0, "零的开方应为0"
        assert sqrt_result.iloc[3] == 4.0, "开方计算错误"

        print("✓ 安全运算验证通过：除法和开方运算都能正确处理边界情况")

    def test_zxm_risk_control_precision_integration(self):
        """测试ZXM风险控制指标的精度集成"""
        print("\n=== 测试ZXM风险控制指标精度集成 ===")

        # 创建ZXM风险控制指标实例
        risk_indicator = ZXMRiskControl()

        # 计算指标
        result = risk_indicator.calculate(self.test_data)

        # 验证所有风险指标的精度
        precision_columns = [
            'VolatilityRisk', 'DrawdownRisk', 'VaRRisk',
            'CompositeRiskScore', 'Returns', 'Volatility'
        ]

        for col in precision_columns:
            if col in result.columns:
                series = result[col].dropna()
                for val in series:
                    if not pd.isna(val):
                        decimal_places = len(str(val).split('.')[-1]) if '.' in str(val) else 0
                        assert decimal_places <= self.precision_requirement, \
                            f"列 {col} 中的值 {val} 超出{self.precision_requirement}位小数精度要求"

        print(f"✓ ZXM风险控制指标精度验证通过：所有风险指标都符合{self.precision_requirement}位小数要求")

    def test_volatility_risk_specific_precision(self):
        """专门测试VolatilityRisk精度控制（P0问题验证）"""
        print("\n=== 专门测试VolatilityRisk精度控制 ===")

        risk_indicator = ZXMRiskControl()
        result = risk_indicator.calculate(self.test_data)

        # 重点验证VolatilityRisk列
        volatility_risk = result['VolatilityRisk'].dropna()

        max_decimal_places = 0
        problematic_values = []

        for val in volatility_risk:
            if not pd.isna(val):
                decimal_str = str(val)
                if '.' in decimal_str:
                    decimal_places = len(decimal_str.split('.')[-1])
                    max_decimal_places = max(max_decimal_places, decimal_places)

                    if decimal_places > self.precision_requirement:
                        problematic_values.append((val, decimal_places))

        # 验证没有超出精度的值
        assert len(problematic_values) == 0, \
            f"VolatilityRisk存在{len(problematic_values)}个超出精度的值: {problematic_values[:5]}"

        # 验证值在合理范围内（0-100）
        assert (volatility_risk >= 0).all() and (volatility_risk <= 100).all(), \
            "VolatilityRisk值应在0-100范围内"

        print(f"✓ VolatilityRisk精度验证通过：最大小数位数={max_decimal_places}，符合≤{self.precision_requirement}位要求")

    def test_technical_utils_precision(self):
        """测试技术分析工具的精度控制"""
        print("\n=== 测试技术分析工具精度控制 ===")

        close_prices = self.test_data['close']

        # 测试EMA精度
        ema_result = calculate_ema_Utils(close_prices, 12)
        ema_valid = ema_result.dropna()

        for val in ema_valid:
            decimal_places = len(str(val).split('.')[-1]) if '.' in str(val) else 0
            assert decimal_places <= self.precision_requirement, \
                f"EMA值 {val} 超出{self.precision_requirement}位小数精度"

        # 测试MACD精度
        dif, dea, macd = calculate_macd_Utils(close_prices)

        for series, name in [(dif, 'DIF'), (dea, 'DEA'), (macd, 'MACD')]:
            valid_values = series.dropna()
            for val in valid_values:
                decimal_places = len(str(val).split('.')[-1]) if '.' in str(val) else 0
                assert decimal_places <= self.precision_requirement, \
                    f"{name}值 {val} 超出{self.precision_requirement}位小数精度"

        # 测试RSI精度
        rsi_result = calculate_rsi_Utils(close_prices)
        rsi_valid = rsi_result.dropna()

        for val in rsi_valid:
            decimal_places = len(str(val).split('.')[-1]) if '.' in str(val) else 0
            assert decimal_places <= self.precision_requirement, \
                f"RSI值 {val} 超出{self.precision_requirement}位小数精度"

            # RSI应在0-100范围内
            assert 0 <= val <= 100, f"RSI值 {val} 超出0-100范围"

        print("✓ 技术分析工具精度验证通过：EMA、MACD、RSI都符合精度要求")

    def test_single_stock_performance_regression(self):
        """测试单股处理性能回归（P1问题验证）"""
        print("\n=== 测试单股处理性能回归 ===")

        # 创建单股数据集
        single_stock_data = self.test_data.copy()

        # 测试ZXM风险控制指标性能
        risk_indicator = ZXMRiskControl()

        # 执行性能测试
        start_time = time.time()
        result = risk_indicator.calculate(single_stock_data)
        end_time = time.time()

        processing_time = end_time - start_time

        # 验证处理时间
        assert processing_time <= self.performance_threshold, \
            f"单股处理时间 {processing_time:.4f}秒 超出阈值 {self.performance_threshold}秒"

        # 验证结果完整性
        assert not result.empty, "计算结果不应为空"
        assert 'CompositeRiskScore' in result.columns, "缺少综合风险评分"

        print(f"✓ 单股处理性能验证通过：处理时间={processing_time:.4f}秒 ≤ {self.performance_threshold}秒")

        return processing_time

    def test_batch_processing_performance(self):
        """测试批量处理性能"""
        print("\n=== 测试批量处理性能 ===")

        # 模拟多只股票数据
        num_stocks = 10
        risk_indicator = ZXMRiskControl()

        total_start_time = time.time()
        processing_times = []

        for i in range(num_stocks):
            # 为每只股票添加一些随机性
            stock_data = self.test_data.copy()
            stock_data['close'] = stock_data['close'] * np.random.uniform(0.8, 1.2)

            start_time = time.time()
            result = risk_indicator.calculate(stock_data)
            end_time = time.time()

            processing_time = end_time - start_time
            processing_times.append(processing_time)

            # 验证每只股票的处理时间
            assert processing_time <= self.performance_threshold, \
                f"股票{i+1}处理时间 {processing_time:.4f}秒 超出阈值"

        total_time = time.time() - total_start_time
        avg_time = np.mean(processing_times)
        max_time = max(processing_times)

        # 计算理论吞吐量
        theoretical_throughput = 3600 / avg_time  # 每小时处理数量

        print(f"✓ 批量处理性能验证通过：")
        print(f"  - 平均处理时间: {avg_time:.4f}秒")
        print(f"  - 最大处理时间: {max_time:.4f}秒")
        print(f"  - 理论吞吐量: {theoretical_throughput:,.0f}股/小时")
        print(f"  - 目标吞吐量: 72,000股/小时")

        # 验证是否达到目标吞吐量
        assert theoretical_throughput >= 72000, \
            f"理论吞吐量 {theoretical_throughput:,.0f}股/小时 未达到目标 72,000股/小时"

    def test_memory_usage_stability(self):
        """测试内存使用稳定性"""
        print("\n=== 测试内存使用稳定性 ===")

        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        risk_indicator = ZXMRiskControl()

        # 连续处理多次以测试内存泄漏
        for i in range(50):
            result = risk_indicator.calculate(self.test_data)

            # 每10次检查一次内存使用
            if i % 10 == 9:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory

                # 内存增长不应超过100MB
                assert memory_increase < 100, \
                    f"内存使用增长过多: {memory_increase:.2f}MB"

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"✓ 内存稳定性验证通过：")
        print(f"  - 初始内存: {initial_memory:.2f}MB")
        print(f"  - 最终内存: {final_memory:.2f}MB")
        print(f"  - 内存增长: {final_memory - initial_memory:.2f}MB")

    def test_edge_cases_handling(self):
        """测试边界情况处理"""
        print("\n=== 测试边界情况处理 ===")

        risk_indicator = ZXMRiskControl()

        # 测试各种边界情况
        edge_cases = [
            # 全零数据
            pd.DataFrame({
                'open': [0.0] * 100,
                'high': [0.0] * 100,
                'low': [0.0] * 100,
                'close': [0.0] * 100,
                'volume': [0.0] * 100
            }),

            # 全相同数据
            pd.DataFrame({
                'open': [100.0] * 100,
                'high': [100.0] * 100,
                'low': [100.0] * 100,
                'close': [100.0] * 100,
                'volume': [1000000.0] * 100
            }),

            # 包含NaN的数据
            pd.DataFrame({
                'open': [100.0, np.nan, 102.0] + [100.0] * 97,
                'high': [101.0, np.nan, 103.0] + [101.0] * 97,
                'low': [99.0, np.nan, 101.0] + [99.0] * 97,
                'close': [100.5, np.nan, 102.5] + [100.5] * 97,
                'volume': [1000000.0, np.nan, 1100000.0] + [1000000.0] * 97
            })
        ]

        for i, edge_data in enumerate(edge_cases):
            try:
                result = risk_indicator.calculate(edge_data)

                # 验证结果不为空且包含必要列
                assert not result.empty, f"边界情况{i+1}：结果为空"
                assert 'CompositeRiskScore' in result.columns, f"边界情况{i+1}：缺少综合风险评分"

                # 验证精度
                risk_score = result['CompositeRiskScore'].dropna()
                if len(risk_score) > 0:
                    for val in risk_score:
                        decimal_places = len(str(val).split('.')[-1]) if '.' in str(val) else 0
                        assert decimal_places <= self.precision_requirement, \
                            f"边界情况{i+1}：精度超标"

                print(f"✓ 边界情况{i+1}处理正常")

            except Exception as e:
                pytest.fail(f"边界情况{i+1}处理失败: {e}")

    def test_numerical_stability_under_stress(self):
        """测试数值稳定性压力测试"""
        print("\n=== 测试数值稳定性压力测试 ===")

        # 构造压力测试数据
        stress_data = pd.DataFrame({
            'open': np.random.uniform(0.001, 99999.999, 1000),
            'high': np.random.uniform(0.001, 99999.999, 1000),
            'low': np.random.uniform(0.001, 99999.999, 1000),
            'close': np.random.uniform(0.001, 99999.999, 1000),
            'volume': np.random.uniform(1, 999999999, 1000)
        })

        # 确保high >= close >= low
        for i in range(len(stress_data)):
            values = [stress_data.loc[i, 'high'], stress_data.loc[i, 'close'], stress_data.loc[i, 'low']]
            values.sort(reverse=True)
            stress_data.loc[i, 'high'] = values[0]
            stress_data.loc[i, 'close'] = values[1]
            stress_data.loc[i, 'low'] = values[2]

        risk_indicator = ZXMRiskControl()
        result = risk_indicator.calculate(stress_data)

        # 验证计算结果的数值稳定性
        for col in ['VolatilityRisk', 'DrawdownRisk', 'VaRRisk', 'CompositeRiskScore']:
            if col in result.columns:
                series = result[col].dropna()

                # 检查是否存在无穷大或NaN
                assert not np.isinf(series).any(), f"{col}存在无穷大值"

                # 检查精度
                for val in series:
                    decimal_places = len(str(val).split('.')[-1]) if '.' in str(val) else 0
                    assert decimal_places <= self.precision_requirement, \
                        f"{col}压力测试中精度超标: {val}"

        print("✓ 数值稳定性压力测试通过：在极端数据条件下保持稳定")

    def generate_precision_test_report(self) -> Dict[str, Any]:
        """生成精度测试报告"""
        print("\n=== 生成精度测试报告 ===")

        # 执行单股性能测试
        processing_time = self.test_single_stock_performance_regression()

        # 收集测试结果
        report = {
            "test_timestamp": pd.Timestamp.now().isoformat(),
            "precision_requirement": self.precision_requirement,
            "performance_threshold": self.performance_threshold,
            "single_stock_processing_time": processing_time,
            "performance_compliance": processing_time <= self.performance_threshold,
            "precision_tests": {
                "numerical_stability_manager": "PASSED",
                "extreme_values_handling": "PASSED",
                "safe_operations": "PASSED",
                "zxm_risk_control_integration": "PASSED",
                "volatility_risk_precision": "PASSED",
                "technical_utils_precision": "PASSED"
            },
            "edge_cases_tests": {
                "zero_data": "PASSED",
                "constant_data": "PASSED",
                "nan_data": "PASSED"
            },
            "stress_tests": {
                "numerical_stability": "PASSED",
                "memory_usage": "PASSED"
            },
            "overall_status": "PRODUCTION_READY"
        }

        return report


# 运行测试的便捷函数
def run_precision_validation_tests():
    """运行完整的精度验证测试套件"""
    test_suite = TestProductionPrecisionValidation()
    test_suite.setup_class()

    try:
        # 执行所有测试
        test_suite.test_numerical_stability_manager_precision()
        test_suite.test_numerical_stability_manager_extreme_values()
        test_suite.test_numerical_stability_manager_safe_operations()
        test_suite.test_zxm_risk_control_precision_integration()
        test_suite.test_volatility_risk_specific_precision()
        test_suite.test_technical_utils_precision()
        test_suite.test_single_stock_performance_regression()
        test_suite.test_batch_processing_performance()
        test_suite.test_memory_usage_stability()
        test_suite.test_edge_cases_handling()
        test_suite.test_numerical_stability_under_stress()

        # 生成报告
        report = test_suite.generate_precision_test_report()

        print("\n" + "="*60)
        print("生产级精度控制验证测试 - 全部通过")
        print("="*60)
        print(f"单股处理时间: {report['single_stock_processing_time']:.4f}秒")
        print(f"性能合规性: {'✓' if report['performance_compliance'] else '✗'}")
        print(f"精度要求: {report['precision_requirement']}位小数")
        print(f"整体状态: {report['overall_status']}")

        return report

    except Exception as e:
        print(f"\n测试失败: {e}")
        raise


if __name__ == "__main__":
    run_precision_validation_tests()