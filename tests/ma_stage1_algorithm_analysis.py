#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MA指标阶段1: 算法差异预分析

验证MA算法与标准公式的一致性，确保SMA、EMA、WMA等计算准确
"""

import sys
import os
import time
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class MAAlgorithmAnalysis:
    """MA指标算法差异预分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.analysis_name = "MA算法差异预分析"
        self.start_time = datetime.now()
        
        # 算法验证标准
        self.algorithm_standards = {
            'sma_accuracy': 0.99999,      # SMA准确性要求99.999%
            'ema_accuracy': 0.99999,      # EMA准确性要求99.999%
            'wma_accuracy': 0.99999,      # WMA准确性要求99.999%
            'correlation_threshold': 0.999, # 与标准实现的相关性阈值
            'max_deviation': 0.001,       # 最大偏差0.1%
            'target_score': 95.0
        }
        
        logger.info(f"✅ {self.analysis_name}初始化完成")
        logger.info(f"🎯 目标: 验证MA算法与标准公式的一致性")
    
    def run_algorithm_analysis(self) -> Dict[str, Any]:
        """运行算法差异预分析"""
        logger.info("🚀 开始MA算法差异预分析")
        
        analysis_results = {
            'analysis_session': {
                'name': self.analysis_name,
                'start_time': self.start_time.isoformat(),
                'standards': self.algorithm_standards
            },
            'ma_import_test': {},
            'sma_algorithm_test': {},
            'ema_algorithm_test': {},
            'wma_algorithm_test': {},
            'cross_validation_test': {},
            'final_assessment': {},
            'final_status': 'IN_PROGRESS'
        }
        
        try:
            # 测试1: MA指标导入测试
            logger.info("📦 测试1: MA指标导入测试")
            import_result = self._test_ma_import()
            analysis_results['ma_import_test'] = import_result
            
            if not import_result.get('success', False):
                analysis_results['final_status'] = 'IMPORT_FAILED'
                return analysis_results
            
            # 获取MA实例
            ma_instance = import_result.get('ma_instance')
            
            # 测试2: SMA算法验证
            logger.info("📊 测试2: SMA算法验证")
            sma_result = self._test_sma_algorithm(ma_instance)
            analysis_results['sma_algorithm_test'] = sma_result
            
            # 测试3: EMA算法验证
            logger.info("📈 测试3: EMA算法验证")
            ema_result = self._test_ema_algorithm(ma_instance)
            analysis_results['ema_algorithm_test'] = ema_result
            
            # 测试4: WMA算法验证
            logger.info("⚖️ 测试4: WMA算法验证")
            wma_result = self._test_wma_algorithm(ma_instance)
            analysis_results['wma_algorithm_test'] = wma_result
            
            # 测试5: 交叉验证测试
            logger.info("🔄 测试5: 交叉验证测试")
            cross_validation_result = self._test_cross_validation(ma_instance)
            analysis_results['cross_validation_test'] = cross_validation_result
            
            # 最终评估
            logger.info("📊 最终评估")
            final_assessment = self._generate_final_assessment(analysis_results)
            analysis_results['final_assessment'] = final_assessment
            
            # 确定最终状态
            final_status = self._determine_final_status(final_assessment)
            analysis_results['final_status'] = final_status
            
            logger.info("✅ MA算法差异预分析完成")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ 分析过程中发生异常: {e}")
            analysis_results['final_status'] = 'ERROR'
            analysis_results['error'] = str(e)
            analysis_results['traceback'] = traceback.format_exc()
            return analysis_results
    
    def _test_ma_import(self) -> Dict[str, Any]:
        """测试MA指标导入"""
        logger.info("📦 测试MA指标导入...")
        
        import_result = {
            'success': False,
            'ma_instance': None,
            'available_methods': [],
            'missing_methods': [],
            'import_time': 0.0
        }
        
        try:
            start_time = time.time()
            
            # 尝试导入MA指标
            from indicators.ma import MaMa
            ma = MaMa()
            
            end_time = time.time()
            import_time = end_time - start_time
            
            # 检查必需方法
            required_methods = [
                'calculate', 'set_parameters', '_get_default_parameters',
                'minimum_periods', 'get_patterns'
            ]
            
            available_methods = []
            missing_methods = []
            
            for method in required_methods:
                if hasattr(ma, method):
                    available_methods.append(method)
                else:
                    missing_methods.append(method)
            
            import_result.update({
                'success': True,
                'ma_instance': ma,
                'available_methods': available_methods,
                'missing_methods': missing_methods,
                'import_time': import_time,
                'class_name': ma.__class__.__name__
            })
            
            logger.info(f"✅ MA指标导入成功: {ma.__class__.__name__}")
            return import_result
            
        except Exception as e:
            logger.error(f"❌ MA指标导入失败: {e}")
            import_result['error'] = str(e)
            return import_result
    
    def _test_sma_algorithm(self, ma_instance) -> Dict[str, Any]:
        """测试SMA算法"""
        logger.info("📊 测试SMA算法准确性...")
        
        sma_result = {
            'algorithm_type': 'SMA',
            'test_scenarios': [],
            'accuracy_score': 0.0,
            'correlation_score': 0.0,
            'overall_score': 0.0
        }
        
        try:
            # 创建多种测试场景
            test_scenarios = self._create_sma_test_scenarios()
            
            for scenario_name, test_data, expected_sma in test_scenarios:
                scenario_result = self._validate_sma_scenario(
                    ma_instance, scenario_name, test_data, expected_sma
                )
                sma_result['test_scenarios'].append(scenario_result)
            
            # 计算总体准确性
            accuracy_scores = [s.get('accuracy', 0) for s in sma_result['test_scenarios']]
            correlation_scores = [s.get('correlation', 0) for s in sma_result['test_scenarios']]
            
            sma_result['accuracy_score'] = np.mean(accuracy_scores) if accuracy_scores else 0
            sma_result['correlation_score'] = np.mean(correlation_scores) if correlation_scores else 0
            sma_result['overall_score'] = (sma_result['accuracy_score'] + sma_result['correlation_score']) / 2
            
            logger.info(f"✅ SMA算法测试完成: {sma_result['overall_score']:.1f}分")
            return sma_result
            
        except Exception as e:
            logger.error(f"❌ SMA算法测试失败: {e}")
            sma_result['error'] = str(e)
            return sma_result
    
    def _create_sma_test_scenarios(self) -> List[tuple]:
        """创建SMA测试场景"""
        scenarios = []
        
        # 场景1: 简单递增序列
        simple_data = pd.DataFrame({
            'close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        # 5周期SMA: [NaN, NaN, NaN, NaN, 3, 4, 5, 6, 7, 8]
        expected_sma1 = [np.nan, np.nan, np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        scenarios.append(('simple_increasing', simple_data, expected_sma1))
        
        # 场景2: 随机数据
        np.random.seed(42)
        random_prices = np.random.uniform(90, 110, 20)
        random_data = pd.DataFrame({'close': random_prices})
        # 计算标准SMA
        expected_sma2 = []
        for i in range(len(random_prices)):
            if i < 4:  # 5周期，前4个为NaN
                expected_sma2.append(np.nan)
            else:
                sma_val = np.mean(random_prices[i-4:i+1])
                expected_sma2.append(sma_val)
        scenarios.append(('random_data', random_data, expected_sma2))
        
        # 场景3: 波动数据
        wave_prices = [100 + 10 * np.sin(i * 0.5) for i in range(15)]
        wave_data = pd.DataFrame({'close': wave_prices})
        expected_sma3 = []
        for i in range(len(wave_prices)):
            if i < 4:
                expected_sma3.append(np.nan)
            else:
                sma_val = np.mean(wave_prices[i-4:i+1])
                expected_sma3.append(sma_val)
        scenarios.append(('wave_data', wave_data, expected_sma3))
        
        return scenarios
    
    def _validate_sma_scenario(self, ma_instance, scenario_name: str, test_data: pd.DataFrame, expected_sma: List[float]) -> Dict[str, Any]:
        """验证SMA场景"""
        scenario_result = {
            'scenario_name': scenario_name,
            'data_points': len(test_data),
            'accuracy': 0.0,
            'correlation': 0.0,
            'max_deviation': 0.0
        }
        
        try:
            # 设置MA参数为SMA，5周期
            ma_instance.set_parameters(period=5, ma_type='SMA')
            
            # 计算MA
            result = ma_instance.calculate(test_data)
            
            if result is not None and not result.empty and 'ma' in result.columns:
                calculated_sma = result['ma'].values
                
                # 比较有效值（跳过NaN）
                valid_indices = ~np.isnan(expected_sma)
                if valid_indices.sum() > 0:
                    expected_valid = np.array(expected_sma)[valid_indices]
                    calculated_valid = calculated_sma[valid_indices]
                    
                    if len(calculated_valid) == len(expected_valid):
                        # 计算准确性
                        deviations = np.abs(calculated_valid - expected_valid)
                        max_deviation = np.max(deviations)
                        mean_deviation = np.mean(deviations)
                        
                        # 准确性评分
                        accuracy = max(0, 100 - mean_deviation * 1000)  # 偏差越小分数越高
                        
                        # 相关性评分
                        if len(expected_valid) > 1:
                            correlation = np.corrcoef(expected_valid, calculated_valid)[0, 1]
                            correlation_score = correlation * 100
                        else:
                            correlation_score = 100
                        
                        scenario_result.update({
                            'accuracy': accuracy,
                            'correlation': correlation_score,
                            'max_deviation': max_deviation,
                            'mean_deviation': mean_deviation
                        })
                    else:
                        scenario_result['error'] = f"长度不匹配: 期望{len(expected_valid)}, 实际{len(calculated_valid)}"
                else:
                    scenario_result['error'] = "没有有效的期望值"
            else:
                scenario_result['error'] = "计算结果无效或缺少ma列"
                
        except Exception as e:
            scenario_result['error'] = str(e)
        
        return scenario_result
    
    def _test_ema_algorithm(self, ma_instance) -> Dict[str, Any]:
        """测试EMA算法"""
        logger.info("📈 测试EMA算法准确性...")
        
        ema_result = {
            'algorithm_type': 'EMA',
            'test_scenarios': [],
            'accuracy_score': 0.0,
            'correlation_score': 0.0,
            'overall_score': 0.0
        }
        
        try:
            # 创建EMA测试场景
            test_scenarios = self._create_ema_test_scenarios()
            
            for scenario_name, test_data, expected_ema in test_scenarios:
                scenario_result = self._validate_ema_scenario(
                    ma_instance, scenario_name, test_data, expected_ema
                )
                ema_result['test_scenarios'].append(scenario_result)
            
            # 计算总体准确性
            accuracy_scores = [s.get('accuracy', 0) for s in ema_result['test_scenarios']]
            correlation_scores = [s.get('correlation', 0) for s in ema_result['test_scenarios']]
            
            ema_result['accuracy_score'] = np.mean(accuracy_scores) if accuracy_scores else 0
            ema_result['correlation_score'] = np.mean(correlation_scores) if correlation_scores else 0
            ema_result['overall_score'] = (ema_result['accuracy_score'] + ema_result['correlation_score']) / 2
            
            logger.info(f"✅ EMA算法测试完成: {ema_result['overall_score']:.1f}分")
            return ema_result
            
        except Exception as e:
            logger.error(f"❌ EMA算法测试失败: {e}")
            ema_result['error'] = str(e)
            return ema_result
    
    def _create_ema_test_scenarios(self) -> List[tuple]:
        """创建EMA测试场景"""
        scenarios = []
        
        # 场景1: 简单递增序列
        simple_data = pd.DataFrame({
            'close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        # 计算标准EMA (5周期, alpha = 2/(5+1) = 0.333)
        alpha = 2.0 / (5 + 1)
        expected_ema1 = []
        ema_val = None
        
        for i, price in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            if i == 0:
                ema_val = price  # 第一个值作为初始EMA
                expected_ema1.append(ema_val)
            else:
                ema_val = alpha * price + (1 - alpha) * ema_val
                expected_ema1.append(ema_val)
        
        scenarios.append(('simple_increasing', simple_data, expected_ema1))
        
        return scenarios
    
    def _validate_ema_scenario(self, ma_instance, scenario_name: str, test_data: pd.DataFrame, expected_ema: List[float]) -> Dict[str, Any]:
        """验证EMA场景"""
        scenario_result = {
            'scenario_name': scenario_name,
            'data_points': len(test_data),
            'accuracy': 0.0,
            'correlation': 0.0,
            'max_deviation': 0.0
        }
        
        try:
            # 设置MA参数为EMA，5周期
            ma_instance.set_parameters(period=5, ma_type='EMA')
            
            # 计算MA
            result = ma_instance.calculate(test_data)
            
            if result is not None and not result.empty and 'ma' in result.columns:
                calculated_ema = result['ma'].values
                
                # 比较值
                if len(calculated_ema) == len(expected_ema):
                    expected_array = np.array(expected_ema)
                    
                    # 计算偏差
                    deviations = np.abs(calculated_ema - expected_array)
                    max_deviation = np.max(deviations)
                    mean_deviation = np.mean(deviations)
                    
                    # 准确性评分
                    accuracy = max(0, 100 - mean_deviation * 100)
                    
                    # 相关性评分
                    correlation = np.corrcoef(expected_array, calculated_ema)[0, 1]
                    correlation_score = correlation * 100
                    
                    scenario_result.update({
                        'accuracy': accuracy,
                        'correlation': correlation_score,
                        'max_deviation': max_deviation,
                        'mean_deviation': mean_deviation
                    })
                else:
                    scenario_result['error'] = f"长度不匹配: 期望{len(expected_ema)}, 实际{len(calculated_ema)}"
            else:
                scenario_result['error'] = "计算结果无效或缺少ma列"
                
        except Exception as e:
            scenario_result['error'] = str(e)
        
        return scenario_result
    
    def _test_wma_algorithm(self, ma_instance) -> Dict[str, Any]:
        """测试WMA算法"""
        logger.info("⚖️ 测试WMA算法准确性...")
        
        wma_result = {
            'algorithm_type': 'WMA',
            'test_scenarios': [],
            'accuracy_score': 0.0,
            'correlation_score': 0.0,
            'overall_score': 0.0
        }
        
        try:
            # 创建WMA测试场景
            test_scenarios = self._create_wma_test_scenarios()
            
            for scenario_name, test_data, expected_wma in test_scenarios:
                scenario_result = self._validate_wma_scenario(
                    ma_instance, scenario_name, test_data, expected_wma
                )
                wma_result['test_scenarios'].append(scenario_result)
            
            # 计算总体准确性
            accuracy_scores = [s.get('accuracy', 0) for s in wma_result['test_scenarios']]
            correlation_scores = [s.get('correlation', 0) for s in wma_result['test_scenarios']]
            
            wma_result['accuracy_score'] = np.mean(accuracy_scores) if accuracy_scores else 0
            wma_result['correlation_score'] = np.mean(correlation_scores) if correlation_scores else 0
            wma_result['overall_score'] = (wma_result['accuracy_score'] + wma_result['correlation_score']) / 2
            
            logger.info(f"✅ WMA算法测试完成: {wma_result['overall_score']:.1f}分")
            return wma_result
            
        except Exception as e:
            logger.error(f"❌ WMA算法测试失败: {e}")
            wma_result['error'] = str(e)
            return wma_result
    
    def _create_wma_test_scenarios(self) -> List[tuple]:
        """创建WMA测试场景"""
        scenarios = []
        
        # 场景1: 简单递增序列
        simple_data = pd.DataFrame({
            'close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        # 计算标准WMA (5周期)
        expected_wma1 = []
        prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        for i in range(len(prices)):
            if i < 4:  # 5周期，前4个为NaN
                expected_wma1.append(np.nan)
            else:
                # WMA计算: (价格1*1 + 价格2*2 + ... + 价格5*5) / (1+2+3+4+5)
                window_prices = prices[i-4:i+1]
                weights = list(range(1, 6))  # [1, 2, 3, 4, 5]
                weighted_sum = sum(p * w for p, w in zip(window_prices, weights))
                weight_sum = sum(weights)
                wma_val = weighted_sum / weight_sum
                expected_wma1.append(wma_val)
        
        scenarios.append(('simple_increasing', simple_data, expected_wma1))
        
        return scenarios
    
    def _validate_wma_scenario(self, ma_instance, scenario_name: str, test_data: pd.DataFrame, expected_wma: List[float]) -> Dict[str, Any]:
        """验证WMA场景"""
        scenario_result = {
            'scenario_name': scenario_name,
            'data_points': len(test_data),
            'accuracy': 0.0,
            'correlation': 0.0,
            'max_deviation': 0.0
        }
        
        try:
            # 设置MA参数为WMA，5周期
            ma_instance.set_parameters(period=5, ma_type='WMA')
            
            # 计算MA
            result = ma_instance.calculate(test_data)
            
            if result is not None and not result.empty and 'ma' in result.columns:
                calculated_wma = result['ma'].values
                
                # 比较有效值（跳过NaN）
                valid_indices = ~np.isnan(expected_wma)
                if valid_indices.sum() > 0:
                    expected_valid = np.array(expected_wma)[valid_indices]
                    calculated_valid = calculated_wma[valid_indices]
                    
                    if len(calculated_valid) == len(expected_valid):
                        # 计算偏差
                        deviations = np.abs(calculated_valid - expected_valid)
                        max_deviation = np.max(deviations)
                        mean_deviation = np.mean(deviations)
                        
                        # 准确性评分
                        accuracy = max(0, 100 - mean_deviation * 1000)
                        
                        # 相关性评分
                        if len(expected_valid) > 1:
                            correlation = np.corrcoef(expected_valid, calculated_valid)[0, 1]
                            correlation_score = correlation * 100
                        else:
                            correlation_score = 100
                        
                        scenario_result.update({
                            'accuracy': accuracy,
                            'correlation': correlation_score,
                            'max_deviation': max_deviation,
                            'mean_deviation': mean_deviation
                        })
                    else:
                        scenario_result['error'] = f"长度不匹配: 期望{len(expected_valid)}, 实际{len(calculated_valid)}"
                else:
                    scenario_result['error'] = "没有有效的期望值"
            else:
                scenario_result['error'] = "计算结果无效或缺少ma列"
                
        except Exception as e:
            scenario_result['error'] = str(e)
        
        return scenario_result
    
    def _test_cross_validation(self, ma_instance) -> Dict[str, Any]:
        """测试交叉验证"""
        logger.info("🔄 测试交叉验证...")
        
        cross_validation_result = {
            'pandas_comparison': {},
            'numpy_comparison': {},
            'consistency_test': {},
            'overall_score': 0.0
        }
        
        try:
            # 创建测试数据
            test_data = self._create_cross_validation_data()
            
            # 与pandas rolling mean比较
            pandas_comparison = self._compare_with_pandas(ma_instance, test_data)
            cross_validation_result['pandas_comparison'] = pandas_comparison
            
            # 与numpy比较
            numpy_comparison = self._compare_with_numpy(ma_instance, test_data)
            cross_validation_result['numpy_comparison'] = numpy_comparison
            
            # 一致性测试
            consistency_test = self._test_consistency(ma_instance, test_data)
            cross_validation_result['consistency_test'] = consistency_test
            
            # 计算总体评分
            pandas_score = pandas_comparison.get('correlation_score', 0)
            numpy_score = numpy_comparison.get('correlation_score', 0)
            consistency_score = consistency_test.get('consistency_score', 0)
            
            cross_validation_result['overall_score'] = (pandas_score + numpy_score + consistency_score) / 3
            
            logger.info(f"✅ 交叉验证测试完成: {cross_validation_result['overall_score']:.1f}分")
            return cross_validation_result
            
        except Exception as e:
            logger.error(f"❌ 交叉验证测试失败: {e}")
            cross_validation_result['error'] = str(e)
            return cross_validation_result
    
    def _create_cross_validation_data(self) -> pd.DataFrame:
        """创建交叉验证数据"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = np.random.uniform(90, 110, 100)
        
        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'volume': np.random.randint(1000000, 5000000, 100)
        })
    
    def _compare_with_pandas(self, ma_instance, test_data: pd.DataFrame) -> Dict[str, Any]:
        """与pandas rolling mean比较"""
        try:
            # 设置SMA参数
            ma_instance.set_parameters(period=20, ma_type='SMA')
            
            # 计算MA
            ma_result = ma_instance.calculate(test_data)
            
            # 计算pandas rolling mean
            pandas_sma = test_data['close'].rolling(window=20).mean()
            
            if ma_result is not None and 'ma' in ma_result.columns:
                ma_values = ma_result['ma'].values
                pandas_values = pandas_sma.values
                
                # 比较有效值
                valid_mask = ~(np.isnan(ma_values) | np.isnan(pandas_values))
                if valid_mask.sum() > 1:
                    correlation = np.corrcoef(ma_values[valid_mask], pandas_values[valid_mask])[0, 1]
                    correlation_score = correlation * 100
                    
                    return {
                        'correlation': correlation,
                        'correlation_score': correlation_score,
                        'valid_points': valid_mask.sum()
                    }
            
            return {'correlation_score': 0, 'error': '无法比较'}
            
        except Exception as e:
            return {'correlation_score': 0, 'error': str(e)}
    
    def _compare_with_numpy(self, ma_instance, test_data: pd.DataFrame) -> Dict[str, Any]:
        """与numpy比较"""
        try:
            # 设置SMA参数
            ma_instance.set_parameters(period=10, ma_type='SMA')
            
            # 计算MA
            ma_result = ma_instance.calculate(test_data)
            
            # 手动计算numpy SMA
            prices = test_data['close'].values
            numpy_sma = []
            
            for i in range(len(prices)):
                if i < 9:  # 10周期，前9个为NaN
                    numpy_sma.append(np.nan)
                else:
                    sma_val = np.mean(prices[i-9:i+1])
                    numpy_sma.append(sma_val)
            
            if ma_result is not None and 'ma' in ma_result.columns:
                ma_values = ma_result['ma'].values
                numpy_values = np.array(numpy_sma)
                
                # 比较有效值
                valid_mask = ~(np.isnan(ma_values) | np.isnan(numpy_values))
                if valid_mask.sum() > 1:
                    correlation = np.corrcoef(ma_values[valid_mask], numpy_values[valid_mask])[0, 1]
                    correlation_score = correlation * 100
                    
                    return {
                        'correlation': correlation,
                        'correlation_score': correlation_score,
                        'valid_points': valid_mask.sum()
                    }
            
            return {'correlation_score': 0, 'error': '无法比较'}
            
        except Exception as e:
            return {'correlation_score': 0, 'error': str(e)}
    
    def _test_consistency(self, ma_instance, test_data: pd.DataFrame) -> Dict[str, Any]:
        """测试一致性"""
        try:
            # 多次计算相同数据，检查结果一致性
            results = []
            
            for _ in range(5):
                ma_instance.set_parameters(period=15, ma_type='SMA')
                result = ma_instance.calculate(test_data)
                if result is not None and 'ma' in result.columns:
                    results.append(result['ma'].values)
            
            if len(results) >= 2:
                # 检查所有结果是否一致
                first_result = results[0]
                consistency_scores = []
                
                for other_result in results[1:]:
                    if len(first_result) == len(other_result):
                        # 比较非NaN值
                        valid_mask = ~(np.isnan(first_result) | np.isnan(other_result))
                        if valid_mask.sum() > 0:
                            diff = np.abs(first_result[valid_mask] - other_result[valid_mask])
                            max_diff = np.max(diff)
                            consistency_score = max(0, 100 - max_diff * 10000)  # 差异越小分数越高
                            consistency_scores.append(consistency_score)
                
                avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
                
                return {
                    'consistency_score': avg_consistency,
                    'test_runs': len(results),
                    'max_difference': max_diff if 'max_diff' in locals() else 0
                }
            
            return {'consistency_score': 0, 'error': '测试结果不足'}
            
        except Exception as e:
            return {'consistency_score': 0, 'error': str(e)}
    
    def _generate_final_assessment(self, analysis_results: Dict) -> Dict[str, Any]:
        """生成最终评估"""
        sma_score = analysis_results.get('sma_algorithm_test', {}).get('overall_score', 0)
        ema_score = analysis_results.get('ema_algorithm_test', {}).get('overall_score', 0)
        wma_score = analysis_results.get('wma_algorithm_test', {}).get('overall_score', 0)
        cross_validation_score = analysis_results.get('cross_validation_test', {}).get('overall_score', 0)
        
        overall_score = (sma_score + ema_score + wma_score + cross_validation_score) / 4
        
        return {
            'sma_algorithm_score': sma_score,
            'ema_algorithm_score': ema_score,
            'wma_algorithm_score': wma_score,
            'cross_validation_score': cross_validation_score,
            'overall_score': overall_score,
            'algorithm_accurate': overall_score >= 95.0,
            'target_achieved': overall_score >= 95.0
        }
    
    def _determine_final_status(self, final_assessment: Dict) -> str:
        """确定最终状态"""
        overall_score = final_assessment.get('overall_score', 0)
        
        if overall_score >= 95.0:
            return 'ALGORITHM_ACCURATE'
        elif overall_score >= 85.0:
            return 'ALGORITHM_MOSTLY_ACCURATE'
        else:
            return 'ALGORITHM_NEEDS_IMPROVEMENT'


def main():
    """主函数"""
    print("🚀 启动MA指标阶段1: 算法差异预分析")
    print("验证MA算法与标准公式的一致性")
    print("=" * 80)
    
    try:
        # 创建分析器
        analyzer = MAAlgorithmAnalysis()
        
        # 运行算法分析
        results = analyzer.run_algorithm_analysis()
        
        # 输出分析摘要
        print(f"\n📊 分析摘要:")
        print(f"最终状态: {results['final_status']}")
        
        if 'final_assessment' in results:
            assessment = results['final_assessment']
            print(f"SMA算法评分: {assessment.get('sma_algorithm_score', 0):.1f}/100")
            print(f"EMA算法评分: {assessment.get('ema_algorithm_score', 0):.1f}/100")
            print(f"WMA算法评分: {assessment.get('wma_algorithm_score', 0):.1f}/100")
            print(f"交叉验证评分: {assessment.get('cross_validation_score', 0):.1f}/100")
            print(f"总体评分: {assessment.get('overall_score', 0):.1f}/100")
            print(f"算法准确: {'✅ 是' if assessment.get('algorithm_accurate', False) else '❌ 否'}")
            print(f"目标达成: {'✅ 是' if assessment.get('target_achieved', False) else '❌ 否'}")
        
        if results['final_status'] == 'ALGORITHM_ACCURATE':
            print("🎉 MA算法差异预分析通过!")
            return 0
        else:
            print("⚠️ MA算法需要进一步优化")
            return 1
            
    except Exception as e:
        logger.error(f"💥 分析执行失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
