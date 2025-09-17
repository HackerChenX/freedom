#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标精度验证器

用于验证技术指标计算的精度、数值稳定性和边界情况处理
"""

import numpy as np
import pandas as pd
import unittest
from typing import Dict, List, Tuple, Any
from utils.technical_utils import (
    calculate_ema_Utils,
    calculate_macd_Utils,
    calculate_rsi_Utils
)
from utils.numerical_stability_manager import get_stability_manager
from utils.logger import get_logger

logger = get_logger(__name__)

class TechnicalIndicatorPrecisionValidator:
    """技术指标精度验证器"""

    def __init__(self):
        self.stability_manager = get_stability_manager()
        self.test_results = {}

    def generate_test_data(self) -> Dict[str, pd.Series]:
        """生成测试数据集"""
        np.random.seed(42)  # 确保结果可重复

        test_datasets = {}

        # 1. 标准测试数据
        standard_prices = pd.Series([
            100.0, 101.5, 102.3, 101.8, 103.2, 104.1, 103.5, 105.0,
            106.2, 105.8, 107.1, 108.3, 107.9, 109.2, 110.5, 109.8,
            111.2, 112.4, 111.9, 113.1, 114.2, 113.8, 115.0, 116.1
        ])
        test_datasets['standard'] = standard_prices

        # 2. 高波动测试数据
        volatile_prices = pd.Series([
            100.0, 95.5, 108.2, 92.1, 115.3, 88.7, 120.5, 85.2,
            125.1, 82.3, 130.2, 78.9, 135.5, 75.1, 140.2, 71.8,
            145.0, 68.5, 150.2, 65.1, 155.5, 61.9, 160.2, 58.7
        ])
        test_datasets['volatile'] = volatile_prices

        # 3. 极值测试数据
        extreme_prices = pd.Series([
            1e-6, 1e6, 1e-3, 1e3, 0.001, 1000000.0, 0.0001, 10000.0,
            1e-5, 1e5, 1e-4, 1e4, 0.01, 100000.0, 0.1, 10000.0,
            1.0, 1000.0, 10.0, 100.0, 50.0, 200.0, 75.0, 150.0
        ])
        test_datasets['extreme'] = extreme_prices

        # 4. 边界情况测试数据
        boundary_prices = pd.Series([
            100.0, 100.0, 100.0, 100.0, 100.1, 99.9, 100.0, 100.0,
            100.0, 100.0, 100.0001, 99.9999, 100.0, 100.0, 100.0,
            100.0, 100.00001, 99.99999, 100.0, 100.0, 100.0, 100.0
        ])
        test_datasets['boundary'] = boundary_prices

        # 5. 含有NaN的测试数据
        nan_prices = pd.Series([
            100.0, np.nan, 102.0, 103.0, np.nan, 105.0, 106.0,
            np.nan, 108.0, 109.0, 110.0, np.nan, 112.0, 113.0,
            114.0, 115.0, np.nan, 117.0, 118.0, 119.0, 120.0
        ])
        test_datasets['with_nan'] = nan_prices

        return test_datasets

    def validate_ema_precision(self, data: pd.Series, name: str) -> Dict[str, Any]:
        """验证EMA计算精度"""
        results = {'name': name, 'passed': True, 'errors': []}

        try:
            # 测试不同方法
            methods = ['standard', 'sma_init', 'pandas']
            for method in methods:
                ema = calculate_ema_Utils(data, period=12, method=method)

                # 检查精度
                if not ema.empty:
                    decimal_places = self._check_decimal_precision(ema)
                    if decimal_places > 6:
                        results['errors'].append(f"EMA {method} 精度超过6位: {decimal_places}")

                # 检查数值稳定性
                anomalies = self.stability_manager.detect_calculation_anomalies(ema, f"EMA_{method}")
                if anomalies['inf_count'] > 0:
                    results['errors'].append(f"EMA {method} 存在无穷大值")

                if anomalies['extreme_count'] > 0:
                    results['errors'].append(f"EMA {method} 存在极值")

        except Exception as e:
            results['errors'].append(f"EMA计算异常: {e}")

        results['passed'] = len(results['errors']) == 0
        return results

    def validate_macd_precision(self, data: pd.Series, name: str) -> Dict[str, Any]:
        """验证MACD计算精度"""
        results = {'name': name, 'passed': True, 'errors': []}

        try:
            dif, dea, macd = calculate_macd_Utils(data, fast_period=12, slow_period=26, signal_period=9)

            # 检查每个输出的精度
            for series_name, series in [('DIF', dif), ('DEA', dea), ('MACD', macd)]:
                if not series.empty:
                    decimal_places = self._check_decimal_precision(series)
                    if decimal_places > 6:
                        results['errors'].append(f"MACD {series_name} 精度超过6位: {decimal_places}")

                    # 检查数值稳定性
                    anomalies = self.stability_manager.detect_calculation_anomalies(series, f"MACD_{series_name}")
                    if anomalies['inf_count'] > 0:
                        results['errors'].append(f"MACD {series_name} 存在无穷大值")

                    if anomalies['extreme_count'] > 0:
                        results['errors'].append(f"MACD {series_name} 存在极值")

            # 验证MACD关系：MACD = (DIF - DEA) * 2
            if not dif.empty and not dea.empty and not macd.empty:
                calculated_macd = (dif - dea) * 2
                diff = np.abs(calculated_macd - macd).max()
                if diff > 1e-6:  # 允许微小的浮点误差
                    results['errors'].append(f"MACD计算关系验证失败，最大差异: {diff}")

        except Exception as e:
            results['errors'].append(f"MACD计算异常: {e}")

        results['passed'] = len(results['errors']) == 0
        return results

    def validate_rsi_precision(self, data: pd.Series, name: str) -> Dict[str, Any]:
        """验证RSI计算精度"""
        results = {'name': name, 'passed': True, 'errors': []}

        try:
            rsi = calculate_rsi_Utils(data, period=14)

            if not rsi.empty:
                # 检查精度
                decimal_places = self._check_decimal_precision(rsi)
                if decimal_places > 6:
                    results['errors'].append(f"RSI 精度超过6位: {decimal_places}")

                # 检查RSI范围 (0-100)
                valid_rsi = rsi.dropna()
                if len(valid_rsi) > 0:
                    if valid_rsi.min() < 0:
                        results['errors'].append(f"RSI存在负值: {valid_rsi.min()}")
                    if valid_rsi.max() > 100:
                        results['errors'].append(f"RSI存在超过100的值: {valid_rsi.max()}")

                # 检查数值稳定性
                anomalies = self.stability_manager.detect_calculation_anomalies(rsi, "RSI")
                if anomalies['inf_count'] > 0:
                    results['errors'].append("RSI存在无穷大值")

        except Exception as e:
            results['errors'].append(f"RSI计算异常: {e}")

        results['passed'] = len(results['errors']) == 0
        return results

    def _check_decimal_precision(self, series: pd.Series) -> int:
        """检查Series的小数位精度"""
        valid_values = series.dropna()
        if len(valid_values) == 0:
            return 0

        max_decimals = 0
        for value in valid_values:
            if isinstance(value, float) and not np.isinf(value):
                # 转换为字符串并检查小数位数
                str_value = f"{value:.10f}"
                if '.' in str_value:
                    decimal_part = str_value.split('.')[1].rstrip('0')
                    max_decimals = max(max_decimals, len(decimal_part))

        return max_decimals

    def run_comprehensive_validation(self) -> Dict[str, List[Dict[str, Any]]]:
        """运行全面验证测试"""
        logger.info("开始技术指标精度验证测试")

        test_datasets = self.generate_test_data()
        results = {
            'ema_results': [],
            'macd_results': [],
            'rsi_results': []
        }

        for dataset_name, data in test_datasets.items():
            logger.info(f"测试数据集: {dataset_name}")

            # 验证EMA
            ema_result = self.validate_ema_precision(data, dataset_name)
            results['ema_results'].append(ema_result)

            # 验证MACD
            macd_result = self.validate_macd_precision(data, dataset_name)
            results['macd_results'].append(macd_result)

            # 验证RSI
            rsi_result = self.validate_rsi_precision(data, dataset_name)
            results['rsi_results'].append(rsi_result)

        # 统计测试结果
        self._summarize_results(results)
        return results

    def _summarize_results(self, results: Dict[str, List[Dict[str, Any]]]):
        """汇总测试结果"""
        logger.info("=== 技术指标精度验证结果汇总 ===")

        for indicator_name, indicator_results in results.items():
            passed_count = sum(1 for result in indicator_results if result['passed'])
            total_count = len(indicator_results)

            logger.info(f"{indicator_name}: {passed_count}/{total_count} 通过")

            # 记录失败的测试
            for result in indicator_results:
                if not result['passed']:
                    logger.warning(f"  {result['name']} 失败:")
                    for error in result['errors']:
                        logger.warning(f"    - {error}")

def run_precision_validation():
    """运行精度验证测试的便捷函数"""
    validator = TechnicalIndicatorPrecisionValidator()
    return validator.run_comprehensive_validation()

if __name__ == "__main__":
    run_precision_validation()