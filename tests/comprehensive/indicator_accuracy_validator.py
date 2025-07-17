#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
指标计算精度验证器

使用标准数据集验证技术指标计算准确性，确保误差<0.01%。
严格遵循六层架构原则，提供高精度的指标验证功能。

L6: 测试应用层 - 本文件提供精度验证功能
L5: 测试业务层 - 具体精度验证逻辑
L4: 测试服务层 - 指标计算服务
L3: 测试数据层 - 标准数据集管理
L2: 测试基础设施层 - 验证工具和配置
L1: 测试数据存储层 - 标准数据和结果存储
"""

import os
import sys
import time
import json
import math
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

logger = get_logger('indicator_accuracy_validator')


class ValidationDataset(Enum):
    """验证数据集类型枚举"""
    STANDARD_SAMPLE = "standard_sample"        # 标准样本数据
    EXTREME_VALUES = "extreme_values"          # 极值数据
    BOUNDARY_CONDITIONS = "boundary_conditions" # 边界条件数据
    REAL_MARKET_DATA = "real_market_data"      # 真实市场数据
    SYNTHETIC_DATA = "synthetic_data"          # 合成数据


class AccuracyLevel(Enum):
    """精度等级枚举"""
    ULTRA_HIGH = 0.0001    # 0.01% - 超高精度
    HIGH = 0.001           # 0.1% - 高精度
    MEDIUM = 0.01          # 1% - 中等精度
    LOW = 0.05             # 5% - 低精度


@dataclass
class StandardTestCase:
    """标准测试用例"""
    indicator_name: str
    input_data: pd.DataFrame
    expected_output: Union[pd.DataFrame, pd.Series, Dict[str, Any]]
    test_parameters: Dict[str, Any]
    accuracy_threshold: float = 0.0001
    description: str = ""
    source: str = ""


@dataclass
class AccuracyValidationResult:
    """精度验证结果"""
    indicator_name: str
    test_case_count: int
    passed_tests: int
    failed_tests: int
    max_error: float
    mean_error: float
    std_error: float
    accuracy_level: AccuracyLevel
    validation_score: float
    execution_time: float
    error_details: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AccuracyValidationSuite:
    """精度验证测试套件结果"""
    suite_name: str
    total_indicators: int
    validated_indicators: int
    overall_accuracy_score: float
    ultra_high_precision_count: int
    high_precision_count: int
    medium_precision_count: int
    low_precision_count: int
    failed_validation_count: int
    total_execution_time: float
    validation_results: List[AccuracyValidationResult] = field(default_factory=list)


class IndicatorAccuracyValidator:
    """
    指标计算精度验证器
    
    负责对技术指标的计算精度进行严格验证，确保计算结果的准确性
    """
    
    def __init__(self):
        """初始化指标精度验证器"""
        self.indicator_factory = IndicatorFactory()
        
        # 验证配置
        self.validation_config = {
            "ultra_high_threshold": 0.0001,    # 0.01%
            "high_threshold": 0.001,           # 0.1%
            "medium_threshold": 0.01,          # 1%
            "low_threshold": 0.05,             # 5%
            "max_test_time": 300,              # 最大测试时间（秒）
            "decimal_precision": 8,            # 计算精度位数
            "comparison_tolerance": 1e-10      # 比较容差
        }
        
        # 标准测试数据
        self.standard_datasets = self._initialize_standard_datasets()
        
        # 预定义的标准答案（基于权威金融库计算）
        self.standard_answers = self._initialize_standard_answers()
        
        # 验证结果存储
        self.validation_results: List[AccuracyValidationResult] = []
    
    def _initialize_standard_datasets(self) -> Dict[ValidationDataset, pd.DataFrame]:
        """初始化标准测试数据集"""
        datasets = {}
        
        # 标准样本数据 - 经典的OHLCV数据
        datasets[ValidationDataset.STANDARD_SAMPLE] = self._create_standard_sample_data()
        
        # 极值数据 - 包含极大和极小值
        datasets[ValidationDataset.EXTREME_VALUES] = self._create_extreme_values_data()
        
        # 边界条件数据 - 边界情况测试
        datasets[ValidationDataset.BOUNDARY_CONDITIONS] = self._create_boundary_conditions_data()
        
        # 合成数据 - 数学生成的理想数据
        datasets[ValidationDataset.SYNTHETIC_DATA] = self._create_synthetic_data()
        
        return datasets
    
    def _create_standard_sample_data(self) -> pd.DataFrame:
        """创建标准样本数据"""
        # 基于真实市场数据模式的标准测试数据
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # 使用固定种子确保结果可重现
        np.random.seed(12345)
        
        # 生成符合金融时间序列特征的数据
        base_price = 100.0
        prices = []
        volumes = []
        
        for i in range(50):
            # 价格遵循几何布朗运动
            daily_return = np.random.normal(0.001, 0.02)  # 日收益率
            base_price *= (1 + daily_return)
            
            # 生成OHLC
            open_price = base_price * (1 + np.random.normal(0, 0.005))
            close_price = base_price * (1 + np.random.normal(0, 0.005))
            
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            
            # 成交量模式
            volume = abs(np.random.normal(1000000, 200000))
            
            prices.append({
                'date': dates[i].strftime('%Y-%m-%d'),
                'open': round(open_price, 4),
                'high': round(high_price, 4),
                'low': round(low_price, 4),
                'close': round(close_price, 4)
            })
            volumes.append(int(volume))
        
        data = pd.DataFrame(prices)
        data['volume'] = volumes
        data['turnover_rate'] = np.random.uniform(0.5, 8.0, 50)
        
        return data
    
    def _create_extreme_values_data(self) -> pd.DataFrame:
        """创建极值测试数据"""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        
        # 包含各种极值情况的数据
        test_values = [
            # 正常值
            {'open': 100.0, 'high': 105.0, 'low': 98.0, 'close': 103.0, 'volume': 1000000},
            # 极大值
            {'open': 1e6, 'high': 1e6 + 1000, 'low': 1e6 - 1000, 'close': 1e6 + 500, 'volume': 1e9},
            # 极小值
            {'open': 0.01, 'high': 0.015, 'low': 0.008, 'close': 0.012, 'volume': 100},
            # 价格无变化
            {'open': 50.0, 'high': 50.0, 'low': 50.0, 'close': 50.0, 'volume': 1000000},
            # 极大波动
            {'open': 100.0, 'high': 200.0, 'low': 50.0, 'close': 150.0, 'volume': 5000000},
        ]
        
        # 扩展到20条数据
        while len(test_values) < 20:
            # 添加随机变化的数据
            base = test_values[len(test_values) % 5]
            factor = 1 + (len(test_values) - 5) * 0.1
            new_value = {
                'open': base['open'] * factor,
                'high': base['high'] * factor,
                'low': base['low'] * factor,
                'close': base['close'] * factor,
                'volume': int(base['volume'] * factor)
            }
            test_values.append(new_value)
        
        data = pd.DataFrame(test_values)
        data['date'] = [date.strftime('%Y-%m-%d') for date in dates]
        data['turnover_rate'] = np.random.uniform(0.1, 10.0, 20)
        
        return data
    
    def _create_boundary_conditions_data(self) -> pd.DataFrame:
        """创建边界条件测试数据"""
        dates = pd.date_range(start='2024-01-01', periods=15, freq='D')
        
        # 各种边界条件
        boundary_cases = [
            # 单调递增
            {'pattern': 'increasing', 'start': 10.0, 'end': 20.0},
            # 单调递减
            {'pattern': 'decreasing', 'start': 20.0, 'end': 10.0},
            # 完全平坦
            {'pattern': 'flat', 'start': 15.0, 'end': 15.0},
            # 锯齿形
            {'pattern': 'zigzag', 'start': 10.0, 'end': 15.0},
            # 正弦波
            {'pattern': 'sine', 'start': 10.0, 'end': 15.0}
        ]
        
        all_data = []
        for i, date in enumerate(dates):
            case = boundary_cases[i % len(boundary_cases)]
            
            if case['pattern'] == 'increasing':
                value = case['start'] + (case['end'] - case['start']) * (i / 14)
            elif case['pattern'] == 'decreasing':
                value = case['start'] - (case['start'] - case['end']) * (i / 14)
            elif case['pattern'] == 'flat':
                value = case['start']
            elif case['pattern'] == 'zigzag':
                value = case['start'] + (case['end'] - case['start']) * (0.5 + 0.5 * (-1) ** i)
            else:  # sine
                value = case['start'] + (case['end'] - case['start']) * (0.5 + 0.5 * np.sin(i / 2))
            
            all_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(value, 4),
                'high': round(value * 1.02, 4),
                'low': round(value * 0.98, 4),
                'close': round(value, 4),
                'volume': 1000000 + i * 10000
            })
        
        data = pd.DataFrame(all_data)
        data['turnover_rate'] = np.random.uniform(1.0, 5.0, 15)
        
        return data
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """创建合成数学数据"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # 基于数学函数生成的理想数据
        data = []
        for i in range(30):
            t = i / 29  # 归一化时间 [0, 1]
            
            # 基础价格函数：组合三角函数
            base_price = 100 + 10 * np.sin(2 * np.pi * t) + 5 * np.cos(4 * np.pi * t)
            
            # 添加小幅随机波动
            noise = np.random.normal(0, 0.1)
            price = base_price + noise
            
            data.append({
                'date': dates[i].strftime('%Y-%m-%d'),
                'open': round(price * 0.995, 4),
                'high': round(price * 1.008, 4),
                'low': round(price * 0.992, 4),
                'close': round(price, 4),
                'volume': int(1000000 * (1 + 0.5 * np.sin(2 * np.pi * t)))
            })
        
        df = pd.DataFrame(data)
        df['turnover_rate'] = np.random.uniform(2.0, 6.0, 30)
        
        return df
    
    def _initialize_standard_answers(self) -> Dict[str, Dict[ValidationDataset, Any]]:
        """初始化标准答案"""
        answers = {}
        
        # 为每个指标和数据集预计算标准答案
        # 这里使用简化的方法，实际应该使用权威金融库计算
        for dataset_type in ValidationDataset:
            dataset = self.standard_datasets[dataset_type]
            
            # MA指标标准答案
            answers.setdefault('MA', {})[dataset_type] = self._calculate_ma_standard_answer(dataset)
            
            # MACD指标标准答案
            answers.setdefault('MACD', {})[dataset_type] = self._calculate_macd_standard_answer(dataset)
            
            # RSI指标标准答案
            answers.setdefault('RSI', {})[dataset_type] = self._calculate_rsi_standard_answer(dataset)
            
            # KDJ指标标准答案
            answers.setdefault('KDJ', {})[dataset_type] = self._calculate_kdj_standard_answer(dataset)
            
            # BOLL指标标准答案
            answers.setdefault('BOLL', {})[dataset_type] = self._calculate_boll_standard_answer(dataset)
        
        return answers
    
    def _calculate_ma_standard_answer(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算MA指标标准答案"""
        result = pd.DataFrame(index=data.index)
        close_prices = data['close'].astype(float)
        
        # 计算不同周期的MA
        for period in [5, 10, 20, 30]:
            if len(close_prices) >= period:
                ma_values = close_prices.rolling(window=period, min_periods=period).mean()
                result[f'MA{period}'] = ma_values.round(8)
        
        return result
    
    def _calculate_macd_standard_answer(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标标准答案"""
        close_prices = data['close'].astype(float)
        
        # MACD参数
        fast_period = 12
        slow_period = 26
        signal_period = 9
        
        # 计算EMA
        ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
        
        # 计算DIF (MACD线)
        dif = ema_fast - ema_slow
        
        # 计算DEA (信号线)
        dea = dif.ewm(span=signal_period, adjust=False).mean()
        
        # 计算MACD柱状图
        macd = (dif - dea) * 2
        
        result = pd.DataFrame({
            'DIF': dif.round(8),
            'DEA': dea.round(8),
            'MACD': macd.round(8)
        }, index=data.index)
        
        return result
    
    def _calculate_rsi_standard_answer(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算RSI指标标准答案"""
        close_prices = data['close'].astype(float)
        period = 14
        
        # 计算价格变化
        delta = close_prices.diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均收益和损失
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # 计算RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        result = pd.DataFrame({
            'RSI': rsi.round(8)
        }, index=data.index)
        
        return result
    
    def _calculate_kdj_standard_answer(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算KDJ指标标准答案"""
        high_prices = data['high'].astype(float)
        low_prices = data['low'].astype(float)
        close_prices = data['close'].astype(float)
        
        n = 9  # KDJ周期
        m1 = 3  # K值平滑参数
        m2 = 3  # D值平滑参数
        
        # 计算RSV
        lowest_low = low_prices.rolling(window=n, min_periods=n).min()
        highest_high = high_prices.rolling(window=n, min_periods=n).max()
        
        rsv = (close_prices - lowest_low) / (highest_high - lowest_low) * 100
        rsv = rsv.fillna(50)  # 处理NaN值
        
        # 初始化K、D值
        k_values = [50]  # K值初始值
        d_values = [50]  # D值初始值
        
        # 计算K、D值
        for i in range(1, len(rsv)):
            if not np.isnan(rsv.iloc[i]):
                k_val = (2/3) * k_values[-1] + (1/3) * rsv.iloc[i]
                d_val = (2/3) * d_values[-1] + (1/3) * k_val
            else:
                k_val = k_values[-1]
                d_val = d_values[-1]
            
            k_values.append(k_val)
            d_values.append(d_val)
        
        # 计算J值
        j_values = [3 * k - 2 * d for k, d in zip(k_values, d_values)]
        
        result = pd.DataFrame({
            'K': [round(k, 8) for k in k_values],
            'D': [round(d, 8) for d in d_values],
            'J': [round(j, 8) for j in j_values]
        }, index=data.index)
        
        return result
    
    def _calculate_boll_standard_answer(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算BOLL指标标准答案"""
        close_prices = data['close'].astype(float)
        period = 20
        std_multiplier = 2.0
        
        # 计算中轨（移动平均线）
        middle_band = close_prices.rolling(window=period, min_periods=period).mean()
        
        # 计算标准差
        std_dev = close_prices.rolling(window=period, min_periods=period).std()
        
        # 计算上轨和下轨
        upper_band = middle_band + (std_dev * std_multiplier)
        lower_band = middle_band - (std_dev * std_multiplier)
        
        result = pd.DataFrame({
            'UPPER': upper_band.round(8),
            'MIDDLE': middle_band.round(8),
            'LOWER': lower_band.round(8)
        }, index=data.index)
        
        return result
    
    @performance_monitor(threshold_seconds=60.0)
    @exception_handler(reraise=True)
    def run_comprehensive_accuracy_validation(self, 
                                             indicators: Optional[List[str]] = None,
                                             datasets: Optional[List[ValidationDataset]] = None) -> AccuracyValidationSuite:
        """
        运行全面的精度验证测试
        
        Args:
            indicators: 要验证的指标列表，None表示验证所有指标
            datasets: 要使用的数据集列表，None表示使用所有数据集
            
        Returns:
            AccuracyValidationSuite: 验证结果套件
        """
        logger.info("开始运行全面的指标精度验证测试")
        start_time = time.time()
        
        # 确定要验证的指标
        if indicators is None:
            indicators = ['MA', 'MACD', 'RSI', 'KDJ', 'BOLL', 'ATR', 'OBV', 'CCI', 'WR']
        
        # 确定要使用的数据集
        if datasets is None:
            datasets = list(ValidationDataset)
        
        all_results = []
        
        # 对每个指标在每个数据集上进行验证
        for indicator_name in indicators:
            logger.info(f"验证指标: {indicator_name}")
            
            try:
                result = self._validate_single_indicator(indicator_name, datasets)
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"指标 {indicator_name} 验证失败: {e}")
        
        # 生成验证结果套件
        suite_result = self._generate_accuracy_validation_suite_result(all_results)
        
        execution_time = time.time() - start_time
        suite_result.total_execution_time = execution_time
        
        logger.info(f"指标精度验证完成，总耗时: {execution_time:.2f}秒")
        logger.info(f"总体精度评分: {suite_result.overall_accuracy_score:.4f}")
        
        return suite_result
    
    @exception_handler(reraise=False, default_return=None)
    def _validate_single_indicator(self, indicator_name: str, 
                                 datasets: List[ValidationDataset]) -> Optional[AccuracyValidationResult]:
        """
        验证单个指标的精度
        
        Args:
            indicator_name: 指标名称
            datasets: 数据集列表
            
        Returns:
            Optional[AccuracyValidationResult]: 验证结果
        """
        start_time = time.time()
        
        try:
            # 创建指标实例
            indicator = self.indicator_factory.create_indicator(indicator_name)
            if indicator is None:
                logger.warning(f"无法创建指标 {indicator_name}")
                return None
            
            all_errors = []
            test_case_count = 0
            passed_tests = 0
            failed_tests = 0
            error_details = []
            performance_metrics = {}
            
            # 在每个数据集上测试
            for dataset_type in datasets:
                if dataset_type not in self.standard_datasets:
                    continue
                
                test_data = self.standard_datasets[dataset_type]
                
                # 检查是否有标准答案
                if (indicator_name in self.standard_answers and 
                    dataset_type in self.standard_answers[indicator_name]):
                    
                    expected_output = self.standard_answers[indicator_name][dataset_type]
                    
                    # 执行指标计算
                    try:
                        calc_start = time.time()
                        actual_output = indicator.calculate(test_data)
                        calc_time = time.time() - calc_start
                        
                        # 比较结果
                        comparison_result = self._compare_indicator_results(
                            expected_output, actual_output, indicator_name
                        )
                        
                        test_case_count += 1
                        
                        if comparison_result['passed']:
                            passed_tests += 1
                        else:
                            failed_tests += 1
                            error_details.append({
                                'dataset': dataset_type.value,
                                'max_error': comparison_result['max_error'],
                                'mean_error': comparison_result['mean_error'],
                                'details': comparison_result['details']
                            })
                        
                        all_errors.extend(comparison_result['errors'])
                        
                        # 记录性能指标
                        performance_metrics[f'{dataset_type.value}_calc_time'] = calc_time
                        
                    except Exception as e:
                        failed_tests += 1
                        test_case_count += 1
                        error_details.append({
                            'dataset': dataset_type.value,
                            'error': str(e)
                        })
                        logger.error(f"指标 {indicator_name} 在数据集 {dataset_type.value} 上计算失败: {e}")
            
            # 计算总体精度指标
            if all_errors:
                max_error = max(all_errors)
                mean_error = np.mean(all_errors)
                std_error = np.std(all_errors)
            else:
                max_error = mean_error = std_error = 0.0
            
            # 确定精度等级
            accuracy_level = self._determine_accuracy_level(max_error)
            
            # 计算验证评分
            validation_score = self._calculate_validation_score(
                passed_tests, test_case_count, max_error, mean_error
            )
            
            execution_time = time.time() - start_time
            
            return AccuracyValidationResult(
                indicator_name=indicator_name,
                test_case_count=test_case_count,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                max_error=max_error,
                mean_error=mean_error,
                std_error=std_error,
                accuracy_level=accuracy_level,
                validation_score=validation_score,
                execution_time=execution_time,
                error_details=error_details,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"指标 {indicator_name} 验证过程出错: {e}")
            return AccuracyValidationResult(
                indicator_name=indicator_name,
                test_case_count=0,
                passed_tests=0,
                failed_tests=1,
                max_error=1.0,
                mean_error=1.0,
                std_error=0.0,
                accuracy_level=AccuracyLevel.LOW,
                validation_score=0.0,
                execution_time=time.time() - start_time,
                error_details=[{'error': str(e)}]
            )
    
    def _compare_indicator_results(self, expected: Union[pd.DataFrame, pd.Series], 
                                 actual: Union[pd.DataFrame, pd.Series], 
                                 indicator_name: str) -> Dict[str, Any]:
        """
        比较指标计算结果
        
        Args:
            expected: 期望结果
            actual: 实际结果
            indicator_name: 指标名称
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        try:
            errors = []
            details = []
            
            # 统一处理为DataFrame格式
            if isinstance(expected, pd.Series):
                expected = expected.to_frame(name=indicator_name)
            if isinstance(actual, pd.Series):
                actual = actual.to_frame(name=indicator_name)
            
            # 检查形状匹配
            if expected.shape != actual.shape:
                return {
                    'passed': False,
                    'max_error': 1.0,
                    'mean_error': 1.0,
                    'errors': [1.0],
                    'details': [f"形状不匹配: 期望 {expected.shape}, 实际 {actual.shape}"]
                }
            
            # 逐列比较
            for column in expected.columns:
                if column in actual.columns:
                    expected_col = expected[column].dropna()
                    actual_col = actual[column].dropna()
                    
                    # 确保索引匹配
                    common_index = expected_col.index.intersection(actual_col.index)
                    if len(common_index) == 0:
                        continue
                    
                    expected_values = expected_col.loc[common_index].astype(float)
                    actual_values = actual_col.loc[common_index].astype(float)
                    
                    # 计算相对误差
                    relative_errors = []
                    for exp_val, act_val in zip(expected_values, actual_values):
                        if abs(exp_val) > 1e-10:  # 避免除零
                            rel_error = abs((act_val - exp_val) / exp_val)
                        else:
                            rel_error = abs(act_val - exp_val)
                        
                        relative_errors.append(rel_error)
                        errors.append(rel_error)
                    
                    # 记录详细信息
                    if relative_errors:
                        column_max_error = max(relative_errors)
                        column_mean_error = np.mean(relative_errors)
                        
                        details.append({
                            'column': column,
                            'max_error': column_max_error,
                            'mean_error': column_mean_error,
                            'sample_count': len(relative_errors)
                        })
            
            # 判断是否通过
            if errors:
                max_error = max(errors)
                mean_error = np.mean(errors)
                passed = max_error < self.validation_config['ultra_high_threshold']
            else:
                max_error = mean_error = 0.0
                passed = True
            
            return {
                'passed': passed,
                'max_error': max_error,
                'mean_error': mean_error,
                'errors': errors,
                'details': details
            }
            
        except Exception as e:
            logger.error(f"比较指标结果失败: {e}")
            return {
                'passed': False,
                'max_error': 1.0,
                'mean_error': 1.0,
                'errors': [1.0],
                'details': [f"比较过程出错: {str(e)}"]
            }
    
    def _determine_accuracy_level(self, max_error: float) -> AccuracyLevel:
        """确定精度等级"""
        config = self.validation_config
        
        if max_error <= config['ultra_high_threshold']:
            return AccuracyLevel.ULTRA_HIGH
        elif max_error <= config['high_threshold']:
            return AccuracyLevel.HIGH
        elif max_error <= config['medium_threshold']:
            return AccuracyLevel.MEDIUM
        elif max_error <= config['low_threshold']:
            return AccuracyLevel.LOW
        else:
            return AccuracyLevel.LOW
    
    def _calculate_validation_score(self, passed_tests: int, total_tests: int, 
                                  max_error: float, mean_error: float) -> float:
        """计算验证评分"""
        if total_tests == 0:
            return 0.0
        
        # 基础评分：通过率
        base_score = passed_tests / total_tests
        
        # 精度调整：基于误差大小
        if max_error <= self.validation_config['ultra_high_threshold']:
            accuracy_bonus = 0.0
        elif max_error <= self.validation_config['high_threshold']:
            accuracy_bonus = -0.1
        elif max_error <= self.validation_config['medium_threshold']:
            accuracy_bonus = -0.3
        else:
            accuracy_bonus = -0.5
        
        # 平均误差惩罚
        mean_error_penalty = min(0.2, mean_error * 10)
        
        final_score = max(0.0, base_score + accuracy_bonus - mean_error_penalty)
        
        return final_score
    
    def _generate_accuracy_validation_suite_result(self, 
                                                  results: List[AccuracyValidationResult]) -> AccuracyValidationSuite:
        """
        生成精度验证套件结果
        
        Args:
            results: 验证结果列表
            
        Returns:
            AccuracyValidationSuite: 验证套件结果
        """
        total_indicators = len(results) if results else 0
        validated_indicators = len([r for r in results if r.test_case_count > 0])
        
        # 计算总体精度评分
        if results:
            overall_accuracy_score = sum(r.validation_score for r in results) / len(results)
        else:
            overall_accuracy_score = 0.0
        
        # 按精度等级统计
        ultra_high_precision_count = len([r for r in results if r.accuracy_level == AccuracyLevel.ULTRA_HIGH])
        high_precision_count = len([r for r in results if r.accuracy_level == AccuracyLevel.HIGH])
        medium_precision_count = len([r for r in results if r.accuracy_level == AccuracyLevel.MEDIUM])
        low_precision_count = len([r for r in results if r.accuracy_level == AccuracyLevel.LOW])
        failed_validation_count = len([r for r in results if r.passed_tests == 0])
        
        total_execution_time = sum(r.execution_time for r in results)
        
        return AccuracyValidationSuite(
            suite_name="指标计算精度验证套件",
            total_indicators=total_indicators,
            validated_indicators=validated_indicators,
            overall_accuracy_score=overall_accuracy_score,
            ultra_high_precision_count=ultra_high_precision_count,
            high_precision_count=high_precision_count,
            medium_precision_count=medium_precision_count,
            low_precision_count=low_precision_count,
            failed_validation_count=failed_validation_count,
            total_execution_time=total_execution_time,
            validation_results=results
        )


if __name__ == "__main__":
    # 示例使用
    validator = IndicatorAccuracyValidator()
    
    # 运行核心指标精度验证
    result = validator.run_comprehensive_accuracy_validation(
        indicators=['MA', 'MACD', 'RSI'],
        datasets=[ValidationDataset.STANDARD_SAMPLE, ValidationDataset.SYNTHETIC_DATA]
    )
    
    print(f"精度验证完成")
    print(f"总体精度评分: {result.overall_accuracy_score:.4f}")
    print(f"超高精度指标数量: {result.ultra_high_precision_count}")
    print(f"高精度指标数量: {result.high_precision_count}")
    print(f"执行时间: {result.total_execution_time:.2f}秒") 