#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
ClosedLoopValidator的单元测试

验证闭环验证器的功能
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from tests.unified_indicator_testing.components.closed_loop_validator import ClosedLoopValidator


class TestClosedLoopValidator(unittest.TestCase):
    """闭环验证器的单元测试"""
    
    def setUp(self):
        """测试前准备"""
        self.validator = ClosedLoopValidator()
        
        # 创建测试数据
        self.test_data_pool = self._create_test_data_pool()
        self.test_selection_results = self._create_test_selection_results()
    
    def tearDown(self):
        """测试后清理"""
        self.validator.cleanup()
    
    def _create_test_data_pool(self) -> list:
        """创建测试数据池"""
        data_pool = []
        
        # 创建目标股票数据（应该有明显的买点形态）
        for i in range(3):
            target_data = pd.DataFrame({
                'date': ['20250721', '20250722', '20250723', '20250724', '20250725'],
                'code': [f'TARGET_{i:03d}'] * 5,
                'name': [f'目标股票_{i}'] * 5,
                'open': [10.0, 10.2, 10.5, 10.8, 11.0],
                'high': [10.3, 10.6, 10.9, 11.2, 11.5],
                'low': [9.8, 10.0, 10.3, 10.6, 10.8],
                'close': [10.1, 10.4, 10.7, 11.0, 11.3],
                'volume': [1000000, 1200000, 1500000, 1800000, 2000000],
                'industry': ['测试行业'] * 5
            })
            data_pool.append(target_data)
        
        # 创建干扰股票数据（没有明显买点形态）
        for i in range(2):
            noise_data = pd.DataFrame({
                'date': ['20250721', '20250722', '20250723', '20250724', '20250725'],
                'code': [f'NOISE_{i:03d}'] * 5,
                'name': [f'干扰股票_{i}'] * 5,
                'open': [15.0, 14.9, 14.8, 14.7, 14.6],
                'high': [15.2, 15.0, 14.9, 14.8, 14.7],
                'low': [14.8, 14.7, 14.6, 14.5, 14.4],
                'close': [14.9, 14.8, 14.7, 14.6, 14.5],
                'volume': [800000, 750000, 700000, 650000, 600000],
                'industry': ['其他行业'] * 5
            })
            data_pool.append(noise_data)
        
        return data_pool
    
    def _create_test_selection_results(self) -> dict:
        """创建测试选股结果"""
        return {
            'execution_success': True,
            'selected_stocks': [
                {'code': 'TARGET_001', 'score': 0.9, 'signal_strength': 0.8},
                {'code': 'TARGET_002', 'score': 0.85, 'signal_strength': 0.75},
                {'code': 'NOISE_001', 'score': 0.7, 'signal_strength': 0.6}
            ],
            'total_candidates': 5,
            'selection_count': 3
        }
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.validator)
        self.assertIsNotNone(self.validator.validation_config)
        self.assertIsNotNone(self.validator.buypoint_patterns)
        self.assertIsNotNone(self.validator.validation_stats)
        
        # 验证配置参数
        config = self.validator.validation_config
        self.assertIn('entry_point_tolerance', config)
        self.assertIn('pattern_match_threshold', config)
        self.assertIn('min_confidence_score', config)
        
        # 验证支持的形态
        supported_patterns = self.validator.get_supported_patterns()
        self.assertIn('MACD', supported_patterns)
        self.assertIn('RSI', supported_patterns)
        self.assertIn('KDJ', supported_patterns)
        self.assertIn('BOLL', supported_patterns)
        self.assertIn('VOL', supported_patterns)
    
    def test_pattern_definitions(self):
        """测试形态定义"""
        # 测试MACD形态
        macd_patterns = self.validator.get_supported_patterns()['MACD']
        self.assertIn('GOLDEN_CROSS', macd_patterns)
        self.assertIn('DEATH_CROSS', macd_patterns)
        
        # 测试形态定义结构
        golden_cross = self.validator._get_pattern_definition('MACD', 'GOLDEN_CROSS')
        self.assertIsNotNone(golden_cross)
        self.assertIn('entry_conditions', golden_cross)
        self.assertIn('confirmation_signals', golden_cross)
        self.assertIn('risk_controls', golden_cross)
        
        # 测试不存在的形态
        unknown_pattern = self.validator._get_pattern_definition('UNKNOWN', 'UNKNOWN')
        self.assertIsNone(unknown_pattern)
    
    def test_technical_indicators_calculation(self):
        """测试技术指标计算"""
        test_data = self.test_data_pool[0]  # 使用第一个测试数据
        
        # 测试MACD计算
        macd_data = self.validator._calculate_macd(test_data.copy())
        self.assertIn('macd_line', macd_data.columns)
        self.assertIn('signal_line', macd_data.columns)
        self.assertIn('histogram', macd_data.columns)
        
        # 测试RSI计算
        rsi_data = self.validator._calculate_rsi(test_data.copy())
        self.assertIn('rsi', rsi_data.columns)
        self.assertTrue((rsi_data['rsi'] >= 0).all())
        self.assertTrue((rsi_data['rsi'] <= 100).all())
        
        # 测试KDJ计算
        kdj_data = self.validator._calculate_kdj(test_data.copy())
        self.assertIn('k', kdj_data.columns)
        self.assertIn('d', kdj_data.columns)
        self.assertIn('j', kdj_data.columns)
        
        # 测试布林带计算
        boll_data = self.validator._calculate_bollinger(test_data.copy())
        self.assertIn('upper_band', boll_data.columns)
        self.assertIn('middle_band', boll_data.columns)
        self.assertIn('lower_band', boll_data.columns)
        self.assertIn('band_width', boll_data.columns)
    
    def test_condition_evaluation(self):
        """测试条件评估"""
        test_data = self.test_data_pool[0].copy()
        enhanced_data = self.validator._calculate_technical_indicators(test_data, 'MACD')
        
        # 测试基本比较条件
        condition1 = {'field': 'close', 'operator': '>', 'value': 10.0}
        result1 = self.validator._evaluate_condition(enhanced_data, 2, condition1)
        self.assertIsInstance(result1, bool)
        
        # 测试字段比较条件
        condition2 = {'field': 'high', 'operator': '>', 'reference': 'low'}
        result2 = self.validator._evaluate_condition(enhanced_data, 2, condition2)
        self.assertTrue(result2)  # high应该总是大于low
        
        # 测试范围条件
        condition3 = {'field': 'close', 'operator': 'between', 'value': [10.0, 12.0]}
        result3 = self.validator._evaluate_condition(enhanced_data, 2, condition3)
        self.assertIsInstance(result3, bool)
    
    def test_cross_detection(self):
        """测试穿越检测"""
        # 创建模拟穿越数据
        cross_data = pd.DataFrame({
            'field1': [1, 2, 3, 4, 5],
            'field2': [5, 4, 3, 2, 1]
        })
        
        # 测试向上穿越
        cross_up = self.validator._check_cross_up(cross_data, 3, 'field1', 'field2', 2)
        self.assertTrue(cross_up)  # field1在索引3处应该穿越field2向上
        
        # 测试向下穿越
        cross_down = self.validator._check_cross_down(cross_data, 3, 'field2', 'field1', 2)
        self.assertTrue(cross_down)  # field2在索引3处应该穿越field1向下
    
    def test_entry_points_analysis(self):
        """测试入口点分析"""
        test_data = self.test_data_pool[0]  # 使用目标股票数据
        
        # 获取MACD金叉形态定义
        pattern_def = self.validator._get_pattern_definition('MACD', 'GOLDEN_CROSS')
        
        # 分析入口点
        entry_points = self.validator._analyze_entry_points(
            test_data, pattern_def, 'MACD', 'GOLDEN_CROSS'
        )
        
        # 验证入口点结构
        self.assertIsInstance(entry_points, list)
        
        if entry_points:
            entry_point = entry_points[0]
            self.assertIn('date', entry_point)
            self.assertIn('entry_price', entry_point)
            self.assertIn('total_score', entry_point)
            self.assertIn('pattern_match', entry_point)
    
    def test_single_stock_validation(self):
        """测试单个股票验证"""
        stock = {'code': 'TARGET_001', 'score': 0.9}
        stock_data = self.test_data_pool[0]  # 目标股票数据
        
        validation_result = self.validator._validate_single_stock(
            stock, stock_data, 'MACD', 'GOLDEN_CROSS'
        )
        
        # 验证结果结构
        self.assertIn('stock_code', validation_result)
        self.assertIn('is_valid', validation_result)
        self.assertIn('confidence_score', validation_result)
        self.assertIn('validation_details', validation_result)
        self.assertIn('entry_points', validation_result)
        
        # 验证数据类型
        self.assertEqual(validation_result['stock_code'], 'TARGET_001')
        self.assertIsInstance(validation_result['is_valid'], bool)
        self.assertIsInstance(validation_result['confidence_score'], float)
        self.assertIsInstance(validation_result['entry_points'], list)
    
    def test_validation_score_calculation(self):
        """测试验证分数计算"""
        # 创建模拟入口点
        entry_points = [
            {'total_score': 0.8, 'entry_score': 0.7},
            {'total_score': 0.9, 'entry_score': 0.8}
        ]
        
        pattern_def = self.validator._get_pattern_definition('MACD', 'GOLDEN_CROSS')
        stock_data = self.test_data_pool[0]
        
        score = self.validator._calculate_validation_score(entry_points, pattern_def, stock_data)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # 测试空入口点
        empty_score = self.validator._calculate_validation_score([], pattern_def, stock_data)
        self.assertEqual(empty_score, 0.0)
    
    def test_complete_validation_process(self):
        """测试完整验证流程"""
        validation_result = self.validator.validate_selection_results(
            self.test_selection_results,
            self.test_data_pool,
            'MACD',
            'GOLDEN_CROSS'
        )
        
        # 验证结果结构
        self.assertIn('indicator_name', validation_result)
        self.assertIn('pattern_type', validation_result)
        self.assertIn('validation_rate', validation_result)
        self.assertIn('successful_validations', validation_result)
        self.assertIn('total_validations', validation_result)
        self.assertIn('validation_results', validation_result)
        self.assertIn('summary', validation_result)
        
        # 验证数据正确性
        self.assertEqual(validation_result['indicator_name'], 'MACD')
        self.assertEqual(validation_result['pattern_type'], 'GOLDEN_CROSS')
        self.assertIsInstance(validation_result['validation_rate'], float)
        self.assertGreaterEqual(validation_result['validation_rate'], 0.0)
        self.assertLessEqual(validation_result['validation_rate'], 1.0)
    
    def test_validation_statistics(self):
        """测试验证统计"""
        # 执行一次验证以生成统计数据
        self.validator.validate_selection_results(
            self.test_selection_results,
            self.test_data_pool,
            'RSI',
            'OVERSOLD'
        )
        
        # 获取统计信息
        stats = self.validator.get_validation_statistics()
        
        # 验证统计结构
        self.assertIn('total_validated', stats)
        self.assertIn('successful_validations', stats)
        self.assertIn('failed_validations', stats)
        self.assertIn('overall_success_rate', stats)
        self.assertIn('pattern_matches', stats)
        
        # 验证统计数据
        self.assertGreaterEqual(stats['total_validated'], 0)
        self.assertIsInstance(stats['overall_success_rate'], float)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试空选股结果
        empty_result = self.validator.validate_selection_results(
            {'selected_stocks': []},
            self.test_data_pool,
            'MACD',
            'GOLDEN_CROSS'
        )
        self.assertEqual(empty_result['total_validations'], 0)
        
        # 测试无效指标
        invalid_result = self.validator.validate_selection_results(
            self.test_selection_results,
            self.test_data_pool,
            'INVALID',
            'UNKNOWN'
        )
        self.assertIn('validation_rate', invalid_result)
        
        # 测试空数据池
        empty_pool_result = self.validator.validate_selection_results(
            self.test_selection_results,
            [],
            'MACD',
            'GOLDEN_CROSS'
        )
        self.assertIn('validation_rate', empty_pool_result)
    
    def test_data_quality_checks(self):
        """测试数据质量检查"""
        test_data = self.test_data_pool[0]
        
        # 测试必需字段检查
        has_required = self.validator._check_required_fields(test_data)
        self.assertTrue(has_required)
        
        # 测试数据完整性
        completeness = self.validator._calculate_data_completeness(test_data)
        self.assertIsInstance(completeness, float)
        self.assertGreaterEqual(completeness, 0.0)
        self.assertLessEqual(completeness, 1.0)
        
        # 测试空数据
        empty_data = pd.DataFrame()
        empty_completeness = self.validator._calculate_data_completeness(empty_data)
        self.assertEqual(empty_completeness, 0.0)
    
    def test_cleanup(self):
        """测试资源清理"""
        # 执行一些操作生成统计数据
        self.validator.validate_selection_results(
            self.test_selection_results,
            self.test_data_pool,
            'KDJ',
            'GOLDEN_CROSS'
        )
        
        # 验证有统计数据
        stats_before = self.validator.get_validation_statistics()
        self.assertGreater(stats_before['total_validated'], 0)
        
        # 执行清理
        self.validator.cleanup()
        
        # 验证统计数据已重置
        stats_after = self.validator.get_validation_statistics()
        self.assertEqual(stats_after['total_validated'], 0)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
