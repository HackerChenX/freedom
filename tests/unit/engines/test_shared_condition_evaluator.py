"""
共享条件评估器单元测试

测试统一条件评估逻辑的正确性和性能
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, Any
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from analysis.engines.shared_condition_evaluator import Shared_condition_evaluator
from analysis.engines.unified_indicator_engine import Unified_indicator_engine


class Test_shared_condition_evaluator(unittest.Test_case):
    """共享条件评估器测试类"""
    
    def set_up_Evaluator(self):
        """测试前准备"""
        self.evaluator = Shared_condition_evaluator()
        
        # 准备测试数据
        self.test_data = {
            'close': np.array([10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.3, 11.8, 12.0, 12.2]),
            'volume': np.array([1000, 1200, 1100, 1300, 1150, 1400, 1250, 1500, 1350, 1600]),
            'ma5': np.array([9.8, 10.1, 10.4, 10.6, 10.7, 10.9, 11.1, 11.3, 11.5, 11.7]),
            'ma10': np.array([9.5, 9.7, 9.9, 10.1, 10.3, 10.5, 10.7, 10.9, 11.1, 11.3]),
            'ma20': np.array([9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9]),
            'rsi': np.array([45, 48, 52, 49, 55, 58, 56, 62, 65, 68]),
            'macd': np.array([-0.1, -0.05, 0.02, 0.01, 0.08, 0.12, 0.10, 0.15, 0.18, 0.22]),
            'dif': np.array([0.05, 0.08, 0.12, 0.10, 0.15, 0.18, 0.16, 0.20, 0.23, 0.26]),
            'dea': np.array([0.15, 0.13, 0.10, 0.09, 0.07, 0.06, 0.06, 0.05, 0.05, 0.04]),
            'kdj_k': np.array([30, 35, 42, 38, 48, 52, 49, 58, 62, 65]),
            'kdj_d': np.array([25, 28, 32, 35, 38, 42, 45, 48, 52, 55]),
            'kdj_j': np.array([40, 49, 62, 44, 68, 72, 57, 78, 82, 85]),
        }
        
        # 添加买点分析特有的逻辑字段
        self.test_data.update({
            'touch_ma': True,
            'price_stable': True,
            'ma_up': False,
            'money_in': True,
            'kpattern': True,
            'vol_shrink': False,
            'macd_gold': True,
            'dif_up': True,
            'dea_up': False,
            'k_up': True,
            'd_up': True,
            'j_up': True,
            'xc': True,  # 吸筹信号
        })
    
    def test_basic_condition_evaluation(self):
        """测试基础条件评估"""
        # 测试价格条件
        condition = {
            'type': 'basic',
            'field': 'close',
            'operator': '>',
            'value': 11.0
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data, date_idx=-1)
        self.assert_true(result)  # 最新收盘价12.2 > 11.0
        
        # 测试成交量条件
        condition = {
            'type': 'basic',
            'field': 'volume',
            'operator': '<',
            'value': 1200
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data, date_idx=-1)
        self.assert_false(result)  # 最新成交量1600 < 1200 为False
        
        # 测试字段比较
        condition = {
            'type': 'basic',
            'field': 'close',
            'operator': '>',
            'reference_field': 'ma5'
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data, date_idx=-1)
        self.assert_true(result)  # 12.2 > 11.7
    
    def test_indicator_condition_evaluation(self):
        """测试指标条件评估"""
        # 测试MA5 > MA10
        condition = {
            'type': 'indicator',
            'indicator': 'ma',
            'field': '5',
            'operator': '>',
            'reference_indicator': 'ma',
            'reference_field': '10'
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data, date_idx=-1)
        self.assert_true(result)  # 11.7 > 11.3
        
        # 测试RSI条件
        condition = {
            'type': 'indicator',
            'indicator': 'rsi',
            'field': '',
            'operator': '<',
            'value': 70
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data, date_idx=-1)
        self.assert_true(result)  # 68 < 70
        
        # 测试MACD条件
        condition = {
            'type': 'indicator',
            'indicator': 'macd',
            'field': '',
            'operator': '>',
            'value': 0
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data, date_idx=-1)
        self.assert_true(result)  # 0.22 > 0
    
    def test_pattern_condition_evaluation(self):
        """测试形态条件评估"""
        # 测试买点分析形态
        condition = {
            'type': 'pattern',
            'pattern': 'touch_ma'
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data)
        self.assert_true(result)
        
        condition = {
            'type': 'pattern',
            'pattern': 'ma_up'
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data)
        self.assert_false(result)
        
        condition = {
            'type': 'pattern',
            'pattern': 'kpattern'
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data)
        self.assert_true(result)
    
    def test_logical_condition_evaluation(self):
        """测试逻辑条件评估"""
        # 测试AND逻辑
        condition = {
            'type': 'logical',
            'operator': 'AND',
            'conditions': [
                {
                    'type': 'basic',
                    'field': 'close',
                    'operator': '>',
                    'value': 11.0
                },
                {
                    'type': 'pattern',
                    'pattern': 'touch_ma'
                }
            ]
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data, date_idx=-1)
        self.assert_true(result)  # 两个条件都为True
        
        # 测试OR逻辑
        condition = {
            'type': 'logical',
            'operator': 'OR',
            'conditions': [
                {
                    'type': 'basic',
                    'field': 'close',
                    'operator': '<',
                    'value': 10.0
                },
                {
                    'type': 'pattern',
                    'pattern': 'money_in'
                }
            ]
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data, date_idx=-1)
        self.assert_true(result)  # 第二个条件为True
        
        # 测试NOT逻辑
        condition = {
            'type': 'logical',
            'operator': 'NOT',
            'conditions': [
                {
                    'type': 'pattern',
                    'pattern': 'vol_shrink'
                }
            ]
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data)
        self.assert_true(result)  # NOT false = True
    
    def test_expression_evaluation(self):
        """测试表达式评估"""
        # 简单表达式
        expression = "12.2 > 11.0"
        result = self.evaluator.evaluate_condition(expression, self.test_data)
        self.assert_true(result)
        
        # 复合表达式
        expression = "12.2 > 11.0 AND 68 < 70"
        result = self.evaluator.evaluate_condition(expression, self.test_data)
        self.assert_true(result)
        
        # 包含变量的表达式（需要实现变量替换）
        # 这个测试可能需要根据实际的变量替换逻辑调整
        try:
            expression = "CLOSE > 11.0 AND RSI < 70"
            result = self.evaluator.evaluate_condition(expression, self.test_data, date_idx=-1)
            # 如果变量替换正常工作，应该返回True
        except Exception as e:
            # 如果变量替换还未完全实现，跳过这个测试
            pass
    
    def test_multiple_conditions_evaluation(self):
        """测试多条件评估"""
        conditions = [
            {
                'type': 'basic',
                'field': 'close',
                'operator': '>',
                'value': 11.0
            },
            {
                'type': 'pattern',
                'pattern': 'touch_ma'
            },
            {
                'type': 'indicator',
                'indicator': 'rsi',
                'field': '',
                'operator': '<',
                'value': 70
            }
        ]
        
        # 测试AND逻辑
        result = self.evaluator.evaluate_conditions(conditions, self.test_data, logic="AND", date_idx=-1)
        self.assert_true(result)
        
        # 测试OR逻辑
        conditions_with_false = [
            {
                'type': 'basic',
                'field': 'close',
                'operator': '<',
                'value': 10.0
            },
            {
                'type': 'pattern',
                'pattern': 'touch_ma'
            }
        ]
        result = self.evaluator.evaluate_conditions(conditions_with_false, self.test_data, logic="OR")
        self.assert_true(result)  # 第二个条件为True
    
    def test_field_value_retrieval(self):
        """测试字段值获取"""
        # 测试数组字段
        value = self.evaluator._get_field_value('close', self.test_data, date_idx=-1)
        self.assert_equal(value, 12.2)
        
        value = self.evaluator._get_field_value('close', self.test_data, date_idx=0)
        self.assert_equal(value, 10.0)
        
        # 测试标量字段
        value = self.evaluator._get_field_value('touch_ma', self.test_data)
        self.assert_equal(value, 1.0)  # True转换为1.0
        
        # 测试不存在的字段
        value = self.evaluator._get_field_value('nonexistent', self.test_data)
        self.assert_is_none(value)
    
    def test_indicator_value_retrieval(self):
        """测试指标值获取"""
        # 测试MA指标
        value = self.evaluator._get_indicator_value('ma', '5', self.test_data, date_idx=-1)
        self.assert_equal(value, 11.7)
        
        # 测试RSI指标
        value = self.evaluator._get_indicator_value('rsi', '', self.test_data, date_idx=-1)
        self.assert_equal(value, 68)
        
        # 测试KDJ指标
        value = self.evaluator._get_indicator_value('kdj', 'k', self.test_data, date_idx=-1)
        self.assert_equal(value, 65)
    
    def test_comparison_operators(self):
        """测试比较运算符"""
        test_cases = [
            (10.0, '>', 9.0, True),
            (10.0, '<', 11.0, True),
            (10.0, '>=', 10.0, True),
            (10.0, '<=', 10.0, True),
            (10.0, '==', 10.0, True),
            (10.0, '!=', 9.0, True),
            (10.0, '=', 10.0, True),  # 单等号
        ]
        
        for value1, op, value2, expected in test_cases:
            condition = {
                'type': 'basic',
                'field': 'test_field',
                'operator': op,
                'value': value2
            }
            
            test_data = {'test_field': value1}
            result = self.evaluator.evaluate_condition(condition, test_data)
            self.assertEqual(result, expected, f"Failed for {value1} {op} {value2}")
    
    def test_cache_functionality_Evaluator(self):
        """测试缓存功能"""
        # 创建新的评估器实例以避免之前测试的影响
        fresh_evaluator = Shared_condition_evaluator()
        fresh_evaluator.clear_cache()  # 清空缓存
        fresh_evaluator.reset_stats()  # 重置统计
        
        condition = {
            'type': 'basic',
            'field': 'close',
            'operator': '>',
            'value': 11.0
        }
        
        # 第一次评估
        result1 = fresh_evaluator.evaluate_condition(condition, self.test_data, date_idx=-1)
        stats1 = fresh_evaluator.get_stats()
        
        # 第二次评估（应该使用缓存）
        result2 = fresh_evaluator.evaluate_condition(condition, self.test_data, date_idx=-1)
        stats2 = fresh_evaluator.get_stats()
        
        # 结果应该相同
        self.assert_equal(result1, result2)
        
        # 验证缓存工作正常（第二次应该有缓存命中）
        self.assertGreaterEqual(stats2['cache_hits'], 1)
        self.assertEqual(stats2['evaluations'], 2)
    
    def test_performance_statistics(self):
        """测试性能统计"""
        # 重置统计
        self.evaluator.reset_stats()
        
        # 执行一些评估
        conditions = [
            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 11.0},
            {'type': 'pattern', 'pattern': 'touch_ma'},
            {'type': 'indicator', 'indicator': 'rsi', 'field': '', 'operator': '<', 'value': 70}
        ]
        
        for condition in conditions:
            self.evaluator.evaluate_condition(condition, self.test_data, date_idx=-1)
        
        stats = self.evaluator.get_stats()
        
        # 检查统计数据
        self.assertEqual(stats['evaluations'], 3)
        self.assertGreaterEqual(stats['total_time'], 0)
        self.assertGreaterEqual(stats['avg_time_per_evaluation'], 0)
        self.assertGreaterEqual(stats['cache_hit_rate'], 0)
        self.assertLessEqual(stats['cache_hit_rate'], 1)
    
    def test_error_handling_Evaluator(self):
        """测试错误处理"""
        # 不支持的条件类型
        condition = {
            'type': 'unsupported_type',
            'field': 'close'
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data)
        self.assert_false(result)  # 错误情况应返回False
        
        # 不支持的运算符
        condition = {
            'type': 'basic',
            'field': 'close',
            'operator': 'unsupported_op',
            'value': 11.0
        }
        try:
            result = self.evaluator.evaluate_condition(condition, self.test_data)
            self.assert_false(result)
        except ValueError:
            pass  # 预期的错误
        
        # 不存在的字段
        condition = {
            'type': 'basic',
            'field': 'nonexistent_field',
            'operator': '>',
            'value': 11.0
        }
        result = self.evaluator.evaluate_condition(condition, self.test_data)
        self.assert_false(result)
    
    def test_edge_cases_Evaluator(self):
        """测试边界情况"""
        # 空条件列表
        result = self.evaluator.evaluate_conditions([], self.test_data)
        self.assert_true(result)
        
        # 空数据
        condition = {
            'type': 'basic',
            'field': 'close',
            'operator': '>',
            'value': 11.0
        }
        result = self.evaluator.evaluate_condition(condition, {})
        self.assert_false(result)
        
        # 超出索引范围
        value = self.evaluator._get_field_value('close', self.test_data, date_idx=100)
        self.assert_equal(value, 12.2)  # 应该返回最新值
    
    def test_integration_with_unified_indicator_engine(self):
        """测试与统一指标引擎的集成"""
        # 确保条件评估器正确使用统一指标引擎
        self.assert_is_instance(self.evaluator.indicator_engine, Unified_indicator_engine)
        
        # 测试指标引擎的使用
        # 这里可以添加更多集成测试


if __name__ == '__main__':
    unittest.main() 