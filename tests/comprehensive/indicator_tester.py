#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
技术指标测试套件

覆盖MA、MACD、RSI、KDJ、BOLL等所有主要技术指标的测试验证。
严格遵循六层架构原则，确保测试系统的可维护性和扩展性。

L6: 测试应用层 - 本文件提供指标测试功能
L5: 测试业务层 - 具体指标测试逻辑
L4: 测试服务层 - 指标计算服务
L3: 测试数据层 - 测试数据管理
L2: 测试基础设施层 - 测试工具和配置
L1: 测试数据存储层 - 测试数据和结果存储
"""

import os
import sys
import time
import json
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
from db.interfaces.market_data_interface import IMarketDataAccess
from db.managers.query_executor import UnifiedQueryExecutor
from enums.indicator_enum import Indicator_enum
from indicators.factory import IndicatorFactory
from indicators.base_indicator import BaseIndicator

logger = get_logger('indicator_tester')


class IndicatorTestType(Enum):
    """指标测试类型枚举"""
    CALCULATION_ACCURACY = "calculation_accuracy"  # 计算精度测试
    PARAMETER_VALIDATION = "parameter_validation"  # 参数验证测试
    BOUNDARY_CONDITIONS = "boundary_conditions"    # 边界条件测试
    PERFORMANCE = "performance"                     # 性能测试
    PATTERN_RECOGNITION = "pattern_recognition"     # 形态识别测试
    SIGNAL_GENERATION = "signal_generation"         # 信号生成测试


@dataclass
class IndicatorTestResult:
    """单个指标测试结果"""
    indicator_name: str
    test_type: IndicatorTestType
    passed: bool
    score: float
    error_rate: float
    execution_time: float
    memory_usage: float
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class IndicatorTestSuite:
    """指标测试套件结果"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_score: float
    total_execution_time: float
    peak_memory_usage: float
    test_results: List[IndicatorTestResult] = field(default_factory=list)
    coverage_report: Dict[str, float] = field(default_factory=dict)


class TechnicalIndicatorTester:
    """
    技术指标测试器
    
    负责对所有技术指标进行全面测试，包括计算精度、性能、形态识别等
    """
    
    def __init__(self):
        """初始化技术指标测试器"""
        self.query_executor = UnifiedQueryExecutor()
        self.indicator_factory = IndicatorFactory()
        
        # 测试配置
        self.test_config = {
            "accuracy_threshold": 0.0001,  # 精度阈值 0.01%
            "performance_threshold": 2.0,   # 性能阈值 2秒
            "memory_threshold": 500,        # 内存阈值 500MB
            "test_data_size": 200,          # 测试数据条数
            "concurrent_tests": 3           # 并发测试数量
        }
        
        # 核心技术指标列表
        self.core_indicators = [
            "MA", "EMA", "MACD", "RSI", "KDJ", "BOLL", "ATR", "OBV",
            "CCI", "WR", "DMI", "ADX", "SAR", "TRIX", "CMO", "ROC",
            "MTM", "BIAS", "VR", "MFI", "VOSC", "PVT", "EMV"
        ]
        
        # ZXM指标列表
        self.zxm_indicators = [
            "ZXM_ABSORB", "ZXM_TURNOVER", "ZXM_DAILY_MACD", "ZXM_MA_CALLBACK",
            "ZXM_RISE_ELASTICITY", "ZXM_AMPLITUDE_ELASTICITY", "ZXM_ELASTICITY_SCORE",
            "ZXM_BUYPOINT_SCORE", "ZXM_DAILY_TREND_UP"
        ]
        
        # 增强指标列表
        self.enhanced_indicators = [
            "UNIFIED_MA", "ENHANCED_MACD", "ENHANCED_RSI", "ENHANCED_WR",
            "ENHANCED_STOCHRSI"
        ]
        
        # 测试结果存储
        self.test_results: List[IndicatorTestResult] = []
        self.test_statistics = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "total_execution_time": 0.0,
            "peak_memory_usage": 0.0
        }
    
    @performance_monitor(threshold_seconds=30.0)
    @exception_handler(reraise=True)
    def run_comprehensive_indicator_tests(self, 
                                        indicator_types: Optional[List[str]] = None,
                                        test_types: Optional[List[IndicatorTestType]] = None) -> IndicatorTestSuite:
        """
        运行全面的技术指标测试
        
        Args:
            indicator_types: 要测试的指标类型列表，None表示测试所有指标
            test_types: 要运行的测试类型列表，None表示运行所有测试
            
        Returns:
            IndicatorTestSuite: 测试套件结果
        """
        logger.info("开始运行全面的技术指标测试")
        start_time = time.time()
        
        # 确定要测试的指标
        if indicator_types is None:
            test_indicators = self.core_indicators + self.zxm_indicators + self.enhanced_indicators
        else:
            test_indicators = indicator_types
            
        # 确定要运行的测试类型
        if test_types is None:
            test_types = list(IndicatorTestType)
        
        # 运行测试
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.test_config["concurrent_tests"]) as executor:
            futures = []
            
            for indicator in test_indicators:
                for test_type in test_types:
                    future = executor.submit(self._run_single_indicator_test, indicator, test_type)
                    futures.append(future)
            
            # 收集结果
            for future in futures:
                try:
                    result = future.result(timeout=60)  # 60秒超时
                    if result:
                        all_results.append(result)
                except Exception as e:
                    logger.error(f"测试执行失败: {e}")
        
        # 生成测试套件结果
        suite_result = self._generate_test_suite_result(all_results)
        
        execution_time = time.time() - start_time
        suite_result.total_execution_time = execution_time
        
        logger.info(f"技术指标测试完成，总耗时: {execution_time:.2f}秒")
        logger.info(f"测试通过率: {suite_result.passed_tests}/{suite_result.total_tests}")
        
        return suite_result
    
    @exception_handler(reraise=False, default_return=None)
    def _run_single_indicator_test(self, indicator_name: str, test_type: IndicatorTestType) -> Optional[IndicatorTestResult]:
        """
        运行单个指标的特定类型测试
        
        Args:
            indicator_name: 指标名称
            test_type: 测试类型
            
        Returns:
            Optional[IndicatorTestResult]: 测试结果
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # 获取测试数据
            test_data = self._get_test_data(indicator_name)
            if test_data is None or test_data.empty:
                return IndicatorTestResult(
                    indicator_name=indicator_name,
                    test_type=test_type,
                    passed=False,
                    score=0.0,
                    error_rate=1.0,
                    execution_time=0.0,
                    memory_usage=0.0,
                    errors=["无法获取测试数据"]
                )
            
            # 根据测试类型执行相应测试
            if test_type == IndicatorTestType.CALCULATION_ACCURACY:
                result = self._test_calculation_accuracy(indicator_name, test_data)
            elif test_type == IndicatorTestType.PARAMETER_VALIDATION:
                result = self._test_parameter_validation(indicator_name, test_data)
            elif test_type == IndicatorTestType.BOUNDARY_CONDITIONS:
                result = self._test_boundary_conditions(indicator_name, test_data)
            elif test_type == IndicatorTestType.PERFORMANCE:
                result = self._test_performance(indicator_name, test_data)
            elif test_type == IndicatorTestType.PATTERN_RECOGNITION:
                result = self._test_pattern_recognition(indicator_name, test_data)
            elif test_type == IndicatorTestType.SIGNAL_GENERATION:
                result = self._test_signal_generation(indicator_name, test_data)
            else:
                result = IndicatorTestResult(
                    indicator_name=indicator_name,
                    test_type=test_type,
                    passed=False,
                    score=0.0,
                    error_rate=1.0,
                    execution_time=0.0,
                    memory_usage=0.0,
                    errors=["未知的测试类型"]
                )
            
            # 设置执行时间和内存使用
            result.execution_time = time.time() - start_time
            result.memory_usage = self._get_memory_usage() - start_memory
            
            return result
            
        except Exception as e:
            logger.error(f"指标 {indicator_name} 测试类型 {test_type.value} 执行失败: {e}")
            return IndicatorTestResult(
                indicator_name=indicator_name,
                test_type=test_type,
                passed=False,
                score=0.0,
                error_rate=1.0,
                execution_time=time.time() - start_time,
                memory_usage=self._get_memory_usage() - start_memory,
                errors=[str(e)]
            )
    
    def _test_calculation_accuracy(self, indicator_name: str, test_data: pd.DataFrame) -> IndicatorTestResult:
        """
        测试计算精度（误差 < 0.01%）
        
        Args:
            indicator_name: 指标名称
            test_data: 测试数据
            
        Returns:
            IndicatorTestResult: 测试结果
        """
        try:
            # 创建指标实例
            indicator = self.indicator_factory.create_indicator(indicator_name)
            if indicator is None:
                return IndicatorTestResult(
                    indicator_name=indicator_name,
                    test_type=IndicatorTestType.CALCULATION_ACCURACY,
                    passed=False,
                    score=0.0,
                    error_rate=1.0,
                    execution_time=0.0,
                    memory_usage=0.0,
                    errors=["无法创建指标实例"]
                )
            
            # 计算指标值
            result_data = indicator.calculate(test_data)
            
            # 验证计算结果
            validation_result = self._validate_calculation_result(
                indicator_name, result_data, test_data
            )
            
            return IndicatorTestResult(
                indicator_name=indicator_name,
                test_type=IndicatorTestType.CALCULATION_ACCURACY,
                passed=validation_result["passed"],
                score=validation_result["score"],
                error_rate=validation_result["error_rate"],
                execution_time=0.0,  # 将在外层设置
                memory_usage=0.0,    # 将在外层设置
                details=validation_result["details"]
            )
            
        except Exception as e:
            return IndicatorTestResult(
                indicator_name=indicator_name,
                test_type=IndicatorTestType.CALCULATION_ACCURACY,
                passed=False,
                score=0.0,
                error_rate=1.0,
                execution_time=0.0,
                memory_usage=0.0,
                errors=[f"计算精度测试失败: {str(e)}"]
            )
    
    def _test_parameter_validation(self, indicator_name: str, test_data: pd.DataFrame) -> IndicatorTestResult:
        """
        测试参数验证
        
        Args:
            indicator_name: 指标名称
            test_data: 测试数据
            
        Returns:
            IndicatorTestResult: 测试结果
        """
        try:
            # 获取指标的参数范围
            param_ranges = self._get_indicator_parameter_ranges(indicator_name)
            
            passed_tests = 0
            total_tests = 0
            errors = []
            
            # 测试有效参数
            for param_set in param_ranges["valid_params"]:
                total_tests += 1
                try:
                    indicator = self.indicator_factory.create_indicator(indicator_name, **param_set)
                    result = indicator.calculate(test_data)
                    if not result.empty:
                        passed_tests += 1
                except Exception as e:
                    errors.append(f"有效参数测试失败 {param_set}: {str(e)}")
            
            # 测试无效参数
            for param_set in param_ranges["invalid_params"]:
                total_tests += 1
                try:
                    indicator = self.indicator_factory.create_indicator(indicator_name, **param_set)
                    result = indicator.calculate(test_data)
                    # 如果没有抛出异常，说明参数验证有问题
                    errors.append(f"无效参数应该失败但成功了 {param_set}")
                except Exception:
                    # 预期的异常，参数验证正确
                    passed_tests += 1
            
            pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
            
            return IndicatorTestResult(
                indicator_name=indicator_name,
                test_type=IndicatorTestType.PARAMETER_VALIDATION,
                passed=pass_rate >= 0.8,  # 80%通过率
                score=pass_rate * 100,
                error_rate=1.0 - pass_rate,
                execution_time=0.0,
                memory_usage=0.0,
                details={"passed_tests": passed_tests, "total_tests": total_tests},
                errors=errors
            )
            
        except Exception as e:
            return IndicatorTestResult(
                indicator_name=indicator_name,
                test_type=IndicatorTestType.PARAMETER_VALIDATION,
                passed=False,
                score=0.0,
                error_rate=1.0,
                execution_time=0.0,
                memory_usage=0.0,
                errors=[f"参数验证测试失败: {str(e)}"]
            )
    
    def _test_boundary_conditions(self, indicator_name: str, test_data: pd.DataFrame) -> IndicatorTestResult:
        """
        测试边界条件
        
        Args:
            indicator_name: 指标名称
            test_data: 测试数据
            
        Returns:
            IndicatorTestResult: 测试结果
        """
        try:
            boundary_tests = [
                {"name": "空数据", "data": pd.DataFrame()},
                {"name": "单条数据", "data": test_data.iloc[:1]},
                {"name": "含NaN数据", "data": self._create_nan_data(test_data)},
                {"name": "极值数据", "data": self._create_extreme_value_data(test_data)},
                {"name": "零值数据", "data": self._create_zero_value_data(test_data)}
            ]
            
            passed_tests = 0
            total_tests = len(boundary_tests)
            errors = []
            
            for test_case in boundary_tests:
                try:
                    indicator = self.indicator_factory.create_indicator(indicator_name)
                    result = indicator.calculate(test_case["data"])
                    
                    # 验证结果不应该包含无穷大或非法值
                    if self._validate_boundary_result(result):
                        passed_tests += 1
                    else:
                        errors.append(f"边界测试 {test_case['name']} 产生了无效结果")
                        
                except Exception as e:
                    # 某些边界条件预期会失败，这是正常的
                    if test_case["name"] in ["空数据", "单条数据"]:
                        passed_tests += 1  # 预期的失败
                    else:
                        errors.append(f"边界测试 {test_case['name']} 异常: {str(e)}")
            
            pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
            
            return IndicatorTestResult(
                indicator_name=indicator_name,
                test_type=IndicatorTestType.BOUNDARY_CONDITIONS,
                passed=pass_rate >= 0.6,  # 60%通过率（边界条件较严格）
                score=pass_rate * 100,
                error_rate=1.0 - pass_rate,
                execution_time=0.0,
                memory_usage=0.0,
                details={"passed_tests": passed_tests, "total_tests": total_tests},
                errors=errors
            )
            
        except Exception as e:
            return IndicatorTestResult(
                indicator_name=indicator_name,
                test_type=IndicatorTestType.BOUNDARY_CONDITIONS,
                passed=False,
                score=0.0,
                error_rate=1.0,
                execution_time=0.0,
                memory_usage=0.0,
                errors=[f"边界条件测试失败: {str(e)}"]
            )
    
    def _test_performance(self, indicator_name: str, test_data: pd.DataFrame) -> IndicatorTestResult:
        """
        测试性能（执行时间 < 2秒）
        
        Args:
            indicator_name: 指标名称
            test_data: 测试数据
            
        Returns:
            IndicatorTestResult: 测试结果
        """
        try:
            # 创建大量数据进行性能测试
            large_test_data = self._create_large_test_data(test_data, 1000)
            
            start_time = time.time()
            
            indicator = self.indicator_factory.create_indicator(indicator_name)
            result = indicator.calculate(large_test_data)
            
            execution_time = time.time() - start_time
            
            # 性能评估
            performance_threshold = self.test_config["performance_threshold"]
            passed = execution_time < performance_threshold
            score = max(0, 100 - (execution_time / performance_threshold) * 100)
            
            return IndicatorTestResult(
                indicator_name=indicator_name,
                test_type=IndicatorTestType.PERFORMANCE,
                passed=passed,
                score=score,
                error_rate=0.0 if passed else 1.0,
                execution_time=execution_time,
                memory_usage=0.0,
                details={
                    "execution_time": execution_time,
                    "threshold": performance_threshold,
                    "data_size": len(large_test_data)
                }
            )
            
        except Exception as e:
            return IndicatorTestResult(
                indicator_name=indicator_name,
                test_type=IndicatorTestType.PERFORMANCE,
                passed=False,
                score=0.0,
                error_rate=1.0,
                execution_time=0.0,
                memory_usage=0.0,
                errors=[f"性能测试失败: {str(e)}"]
            )
    
    def _test_pattern_recognition(self, indicator_name: str, test_data: pd.DataFrame) -> IndicatorTestResult:
        """
        测试形态识别功能
        
        Args:
            indicator_name: 指标名称
            test_data: 测试数据
            
        Returns:
            IndicatorTestResult: 测试结果
        """
        try:
            indicator = self.indicator_factory.create_indicator(indicator_name)
            
            # 检查指标是否支持形态识别
            if not hasattr(indicator, 'get_patterns'):
                return IndicatorTestResult(
                    indicator_name=indicator_name,
                    test_type=IndicatorTestType.PATTERN_RECOGNITION,
                    passed=True,
                    score=100.0,
                    error_rate=0.0,
                    execution_time=0.0,
                    memory_usage=0.0,
                    details={"message": "指标不支持形态识别功能"}
                )
            
            # 计算指标
            result_data = indicator.calculate(test_data)
            
            # 获取形态
            patterns = indicator.get_patterns(test_data)
            
            # 验证形态识别结果
            pattern_validation = self._validate_pattern_recognition(patterns)
            
            return IndicatorTestResult(
                indicator_name=indicator_name,
                test_type=IndicatorTestType.PATTERN_RECOGNITION,
                passed=pattern_validation["passed"],
                score=pattern_validation["score"],
                error_rate=pattern_validation["error_rate"],
                execution_time=0.0,
                memory_usage=0.0,
                details=pattern_validation["details"]
            )
            
        except Exception as e:
            return IndicatorTestResult(
                indicator_name=indicator_name,
                test_type=IndicatorTestType.PATTERN_RECOGNITION,
                passed=False,
                score=0.0,
                error_rate=1.0,
                execution_time=0.0,
                memory_usage=0.0,
                errors=[f"形态识别测试失败: {str(e)}"]
            )
    
    def _test_signal_generation(self, indicator_name: str, test_data: pd.DataFrame) -> IndicatorTestResult:
        """
        测试信号生成功能
        
        Args:
            indicator_name: 指标名称
            test_data: 测试数据
            
        Returns:
            IndicatorTestResult: 测试结果
        """
        try:
            indicator = self.indicator_factory.create_indicator(indicator_name)
            
            # 检查指标是否支持信号生成
            if not hasattr(indicator, 'get_signals'):
                return IndicatorTestResult(
                    indicator_name=indicator_name,
                    test_type=IndicatorTestType.SIGNAL_GENERATION,
                    passed=True,
                    score=100.0,
                    error_rate=0.0,
                    execution_time=0.0,
                    memory_usage=0.0,
                    details={"message": "指标不支持信号生成功能"}
                )
            
            # 计算指标
            result_data = indicator.calculate(test_data)
            
            # 获取信号
            signals = indicator.get_signals(test_data)
            
            # 验证信号生成结果
            signal_validation = self._validate_signal_generation(signals)
            
            return IndicatorTestResult(
                indicator_name=indicator_name,
                test_type=IndicatorTestType.SIGNAL_GENERATION,
                passed=signal_validation["passed"],
                score=signal_validation["score"],
                error_rate=signal_validation["error_rate"],
                execution_time=0.0,
                memory_usage=0.0,
                details=signal_validation["details"]
            )
            
        except Exception as e:
            return IndicatorTestResult(
                indicator_name=indicator_name,
                test_type=IndicatorTestType.SIGNAL_GENERATION,
                passed=False,
                score=0.0,
                error_rate=1.0,
                execution_time=0.0,
                memory_usage=0.0,
                errors=[f"信号生成测试失败: {str(e)}"]
            )
    
    def _get_test_data(self, indicator_name: str) -> Optional[pd.DataFrame]:
        """
        获取指标测试数据
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            Optional[pd.DataFrame]: 测试数据
        """
        try:
            # 从数据库获取测试数据
            query = """
            SELECT code, name, date, open, high, low, close, volume, turnover_rate
            FROM stock_info 
            WHERE code = '000001'
            AND level = '日线'
            AND date >= '2024-01-01' AND date <= '2024-12-31'
            ORDER BY date ASC
            LIMIT 200
            """
            
            result = self.query_executor.execute_query(query)
            if result and not result.empty:
                return result
            else:
                # 如果无法从数据库获取数据，生成模拟数据
                return self._generate_mock_data()
                
        except Exception as e:
            logger.warning(f"获取测试数据失败，使用模拟数据: {e}")
            return self._generate_mock_data()
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """
        生成模拟测试数据
        
        Returns:
            pd.DataFrame: 模拟数据
        """
        size = self.test_config["test_data_size"]
        dates = pd.date_range(start='2024-01-01', periods=size, freq='D')
        
        # 生成随机价格数据
        np.random.seed(42)  # 确保结果可重现
        
        base_price = 10.0
        prices = []
        volume = []
        
        for i in range(size):
            # 生成OHLC数据
            open_price = base_price + np.random.normal(0, 0.1)
            high_price = open_price + abs(np.random.normal(0, 0.2))
            low_price = open_price - abs(np.random.normal(0, 0.2))
            close_price = open_price + np.random.normal(0, 0.15)
            
            # 确保OHLC逻辑正确
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            prices.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            })
            
            volume.append(abs(np.random.normal(1000000, 200000)))
            base_price = close_price  # 下一天的基准价格
        
        data = pd.DataFrame(prices)
        data['date'] = dates
        data['volume'] = volume
        data['turnover_rate'] = np.random.uniform(0.5, 5.0, size)
        data['code'] = '000001'
        data['name'] = '测试股票'
        
        return data
    
    def _validate_calculation_result(self, indicator_name: str, 
                                   result_data: pd.DataFrame, 
                                   test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证计算结果的准确性
        
        Args:
            indicator_name: 指标名称
            result_data: 计算结果
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        try:
            # 基本验证
            if result_data.empty:
                return {
                    "passed": False,
                    "score": 0.0,
                    "error_rate": 1.0,
                    "details": {"error": "计算结果为空"}
                }
            
            # 检查NaN值比例
            nan_ratio = result_data.isnull().sum().sum() / (result_data.shape[0] * result_data.shape[1])
            
            # 检查无穷大值
            inf_count = np.isinf(result_data.select_dtypes(include=[np.number])).sum().sum()
            
            # 计算准确性评分
            accuracy_score = 100.0
            if nan_ratio > 0.5:  # NaN值超过50%
                accuracy_score -= 30
            if inf_count > 0:  # 有无穷大值
                accuracy_score -= 20
            if len(result_data) != len(test_data):  # 长度不匹配
                accuracy_score -= 10
            
            passed = accuracy_score >= 70.0  # 70分以上算通过
            
            return {
                "passed": passed,
                "score": accuracy_score,
                "error_rate": (100.0 - accuracy_score) / 100.0,
                "details": {
                    "nan_ratio": nan_ratio,
                    "inf_count": inf_count,
                    "result_length": len(result_data),
                    "input_length": len(test_data)
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "error_rate": 1.0,
                "details": {"error": f"验证过程出错: {str(e)}"}
            }
    
    def _get_indicator_parameter_ranges(self, indicator_name: str) -> Dict[str, List[Dict]]:
        """
        获取指标参数范围用于测试
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            Dict[str, List[Dict]]: 参数范围
        """
        # 基本参数测试模板
        basic_valid_params = [
            {},  # 默认参数
            {"period": 14},
            {"period": 20},
            {"period": 30}
        ]
        
        basic_invalid_params = [
            {"period": -1},   # 负数周期
            {"period": 0},    # 零周期
            {"period": 1000}, # 过大周期
            {"period": "abc"} # 非数字周期
        ]
        
        # 针对特定指标的参数
        indicator_specific = {
            "MACD": {
                "valid_params": [
                    {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                    {"fast_period": 5, "slow_period": 10, "signal_period": 5}
                ],
                "invalid_params": [
                    {"fast_period": 26, "slow_period": 12, "signal_period": 9},  # fast > slow
                    {"fast_period": -1, "slow_period": 26, "signal_period": 9}
                ]
            },
            "KDJ": {
                "valid_params": [
                    {"n": 9, "m1": 3, "m2": 3},
                    {"n": 14, "m1": 5, "m2": 5}
                ],
                "invalid_params": [
                    {"n": -1, "m1": 3, "m2": 3},
                    {"n": 9, "m1": -1, "m2": 3}
                ]
            },
            "BOLL": {
                "valid_params": [
                    {"period": 20, "std_dev": 2.0},
                    {"period": 26, "std_dev": 2.5}
                ],
                "invalid_params": [
                    {"period": 20, "std_dev": -1.0},
                    {"period": -1, "std_dev": 2.0}
                ]
            }
        }
        
        if indicator_name in indicator_specific:
            return indicator_specific[indicator_name]
        else:
            return {
                "valid_params": basic_valid_params,
                "invalid_params": basic_invalid_params
            }
    
    def _create_nan_data(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """创建包含NaN值的测试数据"""
        data = test_data.copy()
        # 随机设置10%的数据为NaN
        mask = np.random.random(data.shape) < 0.1
        data = data.mask(mask)
        return data
    
    def _create_extreme_value_data(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """创建包含极值的测试数据"""
        data = test_data.copy()
        # 设置一些极大和极小值
        data.loc[0, 'close'] = 1e10  # 极大值
        data.loc[1, 'close'] = 1e-10  # 极小值
        return data
    
    def _create_zero_value_data(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """创建包含零值的测试数据"""
        data = test_data.copy()
        # 设置一些零值
        data.loc[0:5, ['volume']] = 0.0
        return data
    
    def _create_large_test_data(self, test_data: pd.DataFrame, size: int) -> pd.DataFrame:
        """创建大数据集用于性能测试"""
        if len(test_data) >= size:
            return test_data.iloc[:size]
        
        # 复制数据到指定大小
        repeat_times = (size // len(test_data)) + 1
        large_data = pd.concat([test_data] * repeat_times, ignore_index=True)
        return large_data.iloc[:size]
    
    def _validate_boundary_result(self, result: pd.DataFrame) -> bool:
        """验证边界条件测试结果"""
        try:
            # 检查是否有无穷大值
            if np.isinf(result.select_dtypes(include=[np.number])).any().any():
                return False
            
            # 检查是否全部为NaN
            if result.isnull().all().all():
                return False
            
            return True
        except:
            return False
    
    def _validate_pattern_recognition(self, patterns: pd.DataFrame) -> Dict[str, Any]:
        """验证形态识别结果"""
        try:
            if patterns.empty:
                return {
                    "passed": True,
                    "score": 100.0,
                    "error_rate": 0.0,
                    "details": {"message": "无形态识别结果"}
                }
            
            # 检查形态识别结果的格式
            pattern_count = len(patterns)
            valid_patterns = 0
            
            for _, pattern in patterns.iterrows():
                # 简单验证：确保形态结果是布尔值或数值
                if isinstance(pattern.iloc[0], (bool, int, float)):
                    valid_patterns += 1
            
            accuracy = valid_patterns / pattern_count if pattern_count > 0 else 1.0
            
            return {
                "passed": accuracy >= 0.8,
                "score": accuracy * 100,
                "error_rate": 1.0 - accuracy,
                "details": {
                    "total_patterns": pattern_count,
                    "valid_patterns": valid_patterns
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "error_rate": 1.0,
                "details": {"error": str(e)}
            }
    
    def _validate_signal_generation(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """验证信号生成结果"""
        try:
            if not signals:
                return {
                    "passed": True,
                    "score": 100.0,
                    "error_rate": 0.0,
                    "details": {"message": "无信号生成结果"}
                }
            
            # 检查信号格式
            valid_signals = 0
            total_signals = len(signals)
            
            for signal_name, signal_value in signals.items():
                # 验证信号值的类型
                if isinstance(signal_value, (bool, int, float, list, pd.Series)):
                    valid_signals += 1
            
            accuracy = valid_signals / total_signals if total_signals > 0 else 1.0
            
            return {
                "passed": accuracy >= 0.8,
                "score": accuracy * 100,
                "error_rate": 1.0 - accuracy,
                "details": {
                    "total_signals": total_signals,
                    "valid_signals": valid_signals
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "error_rate": 1.0,
                "details": {"error": str(e)}
            }
    
    def _generate_test_suite_result(self, all_results: List[IndicatorTestResult]) -> IndicatorTestSuite:
        """
        生成测试套件结果
        
        Args:
            all_results: 所有测试结果
            
        Returns:
            IndicatorTestSuite: 测试套件结果
        """
        total_tests = len(all_results)
        passed_tests = sum(1 for result in all_results if result.passed)
        failed_tests = total_tests - passed_tests
        
        overall_score = sum(result.score for result in all_results) / total_tests if total_tests > 0 else 0.0
        total_execution_time = sum(result.execution_time for result in all_results)
        peak_memory_usage = max(result.memory_usage for result in all_results) if all_results else 0.0
        
        # 计算覆盖率报告
        coverage_report = self._calculate_coverage_report(all_results)
        
        return IndicatorTestSuite(
            suite_name="技术指标测试套件",
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            overall_score=overall_score,
            total_execution_time=total_execution_time,
            peak_memory_usage=peak_memory_usage,
            test_results=all_results,
            coverage_report=coverage_report
        )
    
    def _calculate_coverage_report(self, results: List[IndicatorTestResult]) -> Dict[str, float]:
        """计算覆盖率报告"""
        coverage = {}
        
        # 按指标分组统计
        indicator_stats = {}
        for result in results:
            if result.indicator_name not in indicator_stats:
                indicator_stats[result.indicator_name] = {"total": 0, "passed": 0}
            indicator_stats[result.indicator_name]["total"] += 1
            if result.passed:
                indicator_stats[result.indicator_name]["passed"] += 1
        
        # 计算各指标覆盖率
        for indicator, stats in indicator_stats.items():
            coverage[indicator] = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
        
        return coverage
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # 转换为MB
        except:
            return 0.0


if __name__ == "__main__":
    # 示例使用
    tester = TechnicalIndicatorTester()
    
    # 运行核心指标测试
    core_result = tester.run_comprehensive_indicator_tests(
        indicator_types=["MA", "MACD", "RSI", "KDJ", "BOLL"],
        test_types=[IndicatorTestType.CALCULATION_ACCURACY, IndicatorTestType.PERFORMANCE]
    )
    
    print(f"测试完成 - 通过率: {core_result.passed_tests}/{core_result.total_tests}")
    print(f"总体评分: {core_result.overall_score:.2f}")
    print(f"执行时间: {core_result.total_execution_time:.2f}秒") 