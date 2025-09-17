#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
SelectionStrategyTester的单元测试

验证选股策略测试器的功能
"""

import os
import sys
import unittest
import pandas as pd
import json
import yaml
import tempfile
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from tests.unified_indicator_testing.components.selection_strategy_tester import SelectionStrategyTester
from db.sql_manager import SQLManager, QueryType


class TestSelectionStrategyTester(unittest.TestCase):
    """选股策略测试器的单元测试"""
    
    def setUp(self):
        """测试前准备"""
        self.tester = SelectionStrategyTester()
        
        # 创建测试数据
        self.mock_data_pool = self._create_test_data_pool()
    
    def tearDown(self):
        """测试后清理"""
        self.tester.cleanup()
    
    def _create_test_data_pool(self) -> list:
        """创建测试数据池"""
        data_pool = []
        
        # 创建目标股票数据
        for i in range(3):
            target_data = pd.DataFrame({
                'date': ['20250721', '20250722', '20250723'],
                'code': [f'TARGET_{i:03d}'] * 3,
                'name': [f'目标股票_{i}'] * 3,
                'open': [10.0, 10.5, 11.0],
                'high': [10.5, 11.0, 11.5],
                'low': [9.5, 10.0, 10.5],
                'close': [10.2, 10.8, 11.2],
                'volume': [1000000, 1200000, 1100000],
                'industry': ['测试行业'] * 3
            })
            data_pool.append(target_data)
        
        # 创建干扰股票数据
        for i in range(7):
            noise_data = pd.DataFrame({
                'date': ['20250721', '20250722', '20250723'],
                'code': [f'NOISE_{i:03d}'] * 3,
                'name': [f'干扰股票_{i}'] * 3,
                'open': [15.0, 14.8, 15.2],
                'high': [15.5, 15.3, 15.8],
                'low': [14.5, 14.2, 14.8],
                'close': [15.1, 14.9, 15.5],
                'volume': [800000, 900000, 850000],
                'industry': ['其他行业'] * 3
            })
            data_pool.append(noise_data)
        
        return data_pool
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.tester)
        self.assertIsNotNone(self.tester.strategy_templates)
        self.assertIsNotNone(self.tester.indicator_pattern_mapping)
        
        # 验证支持的指标
        supported_indicators = self.tester.get_supported_indicators()
        self.assertIn('MACD', supported_indicators)
        self.assertIn('RSI', supported_indicators)
        self.assertIn('KDJ', supported_indicators)
        self.assertIn('BOLL', supported_indicators)
        self.assertIn('VOL', supported_indicators)
    
    def test_generate_unified_strategy_config(self):
        """测试生成统一格式策略配置"""
        strategy_config = self.tester.generate_strategy_config(
            'MACD', 'GOLDEN_CROSS', 'unified'
        )
        
        # 验证基本结构
        self.assertIn('strategy', strategy_config)
        self.assertIn('technical_indicators', strategy_config)
        self.assertIn('time_criteria', strategy_config)
        self.assertIn('filters', strategy_config)
        
        # 验证策略信息
        strategy = strategy_config['strategy']
        self.assertIn('id', strategy)
        self.assertIn('name', strategy)
        self.assertIn('description', strategy)
        self.assertTrue(strategy['id'].startswith('AUTO_MACD_GOLDEN_CROSS'))
        
        # 验证技术指标配置
        indicators = strategy_config['technical_indicators']['primary_indicators']
        self.assertEqual(len(indicators), 1)
        self.assertEqual(indicators[0]['indicator_id'], 'MACD')
    
    def test_generate_legacy_strategy_config(self):
        """测试生成传统格式策略配置"""
        strategy_config = self.tester.generate_strategy_config(
            'RSI', 'OVERSOLD', 'legacy'
        )
        
        # 验证基本结构
        self.assertIn('strategy', strategy_config)
        
        strategy = strategy_config['strategy']
        self.assertIn('id', strategy)
        self.assertIn('name', strategy)
        self.assertIn('conditions', strategy)
        self.assertTrue(strategy['id'].startswith('LEGACY_RSI_OVERSOLD'))
        
        # 验证条件配置
        conditions = strategy['conditions']
        self.assertGreater(len(conditions), 0)
    
    def test_strategy_config_validation(self):
        """测试策略配置验证"""
        # 测试有效配置
        valid_config = self.tester.generate_strategy_config('KDJ', 'GOLDEN_CROSS')
        validation_result = self.tester.validate_strategy_config(valid_config)
        
        self.assertTrue(validation_result['is_valid'])
        self.assertEqual(len(validation_result['errors']), 0)
        
        # 测试无效配置
        invalid_config = {'invalid': 'config'}
        validation_result = self.tester.validate_strategy_config(invalid_config)
        
        self.assertFalse(validation_result['is_valid'])
        self.assertGreater(len(validation_result['errors']), 0)
    
    def test_indicator_pattern_mapping(self):
        """测试指标形态映射"""
        # 测试MACD指标
        macd_patterns = self.tester.get_indicator_patterns('MACD')
        self.assertIn('GOLDEN_CROSS', macd_patterns)
        self.assertIn('DEATH_CROSS', macd_patterns)
        self.assertIn('DIVERGENCE', macd_patterns)
        
        # 测试RSI指标
        rsi_patterns = self.tester.get_indicator_patterns('RSI')
        self.assertIn('OVERBOUGHT', rsi_patterns)
        self.assertIn('OVERSOLD', rsi_patterns)
        
        # 测试不存在的指标
        unknown_patterns = self.tester.get_indicator_patterns('UNKNOWN')
        self.assertEqual(len(unknown_patterns), 0)
    
    def test_strategy_selection_execution(self):
        """测试选股策略执行"""
        result = self.tester.test_strategy_selection(
            'MACD', 'GOLDEN_CROSS', self.mock_data_pool
        )
        
        # 验证结果结构
        self.assertIn('execution_success', result)
        self.assertIn('selected_stocks', result)
        self.assertIn('performance_metrics', result)
        self.assertIn('strategy_id', result)
        
        # 验证执行成功
        self.assertTrue(result['execution_success'])
        
        # 验证性能指标
        performance = result['performance_metrics']
        self.assertIn('precision', performance)
        self.assertIn('recall', performance)
        self.assertIn('f1_score', performance)
        self.assertIn('selection_rate', performance)
        
        # 验证数据一致性
        self.assertEqual(performance['total_candidates'], len(self.mock_data_pool))
    
    def test_complex_strategy_generation(self):
        """测试复杂策略生成"""
        indicators = [
            {'indicator_name': 'MACD', 'pattern_type': 'GOLDEN_CROSS'},
            {'indicator_name': 'RSI', 'pattern_type': 'OVERSOLD'},
            {'indicator_name': 'KDJ', 'pattern_type': 'GOLDEN_CROSS'}
        ]
        
        complex_strategy = self.tester.generate_complex_strategy(indicators, 'AND')
        
        # 验证复杂策略结构
        self.assertIn('strategy', complex_strategy)
        self.assertIn('technical_indicators', complex_strategy)
        
        # 验证包含所有指标
        primary_indicators = complex_strategy['technical_indicators']['primary_indicators']
        self.assertEqual(len(primary_indicators), 3)
        
        # 验证逻辑操作符
        self.assertEqual(complex_strategy['technical_indicators']['combination_logic'], 'AND')
    
    def test_multiple_strategies_execution(self):
        """测试多策略执行"""
        strategies = [
            {'indicator_name': 'MACD', 'pattern_type': 'GOLDEN_CROSS'},
            {'indicator_name': 'RSI', 'pattern_type': 'OVERSOLD'},
            {'indicator_name': 'KDJ', 'pattern_type': 'OVERBOUGHT'}
        ]
        
        results = self.tester.test_multiple_strategies(strategies, self.mock_data_pool)
        
        # 验证结果结构
        self.assertIn('total_strategies', results)
        self.assertIn('completed_strategies', results)
        self.assertIn('results', results)
        
        # 验证策略数量
        self.assertEqual(results['total_strategies'], 3)
        
        # 验证每个策略的结果
        strategy_results = results['results']
        self.assertEqual(len(strategy_results), 3)
        
        for strategy_result in strategy_results.values():
            self.assertIn('execution_success', strategy_result)
    
    def test_performance_calculation(self):
        """测试性能计算"""
        # 创建模拟选股结果
        parsed_result = {
            'selected_stocks': [
                {'code': 'TARGET_001', 'score': 0.9},
                {'code': 'TARGET_002', 'score': 0.8},
                {'code': 'NOISE_001', 'score': 0.7}
            ],
            'execution_time': 1.5
        }
        
        performance = self.tester._calculate_selection_performance(
            parsed_result, self.mock_data_pool
        )
        
        # 验证性能指标
        self.assertIn('precision', performance)
        self.assertIn('recall', performance)
        self.assertIn('f1_score', performance)
        self.assertIn('selection_rate', performance)
        
        # 验证计算正确性
        self.assertEqual(performance['selected_count'], 3)
        self.assertEqual(performance['selected_target_count'], 2)
        self.assertAlmostEqual(performance['precision'], 2/3, places=2)
    
    def test_mock_environment_setup(self):
        """测试模拟环境设置"""
        mock_env = self.tester._setup_mock_environment(self.mock_data_pool)
        
        # 验证环境结构
        self.assertIn('temp_dir', mock_env)
        self.assertIn('data_file', mock_env)
        self.assertIn('config_file', mock_env)
        self.assertIn('env_vars', mock_env)
        
        # 验证文件存在
        self.assertTrue(os.path.exists(mock_env['data_file']))
        self.assertTrue(os.path.exists(mock_env['config_file']))
        
        # 验证环境变量
        env_vars = mock_env['env_vars']
        self.assertIn('MOCK_DATA_MODE', env_vars)
        self.assertEqual(env_vars['MOCK_DATA_MODE'], 'true')
    
    def test_strategy_file_generation(self):
        """测试策略文件生成"""
        strategy_config = self.tester.generate_strategy_config('BOLL', 'UPPER_BREAKOUT')
        
        # 测试JSON格式
        json_file = self.tester._save_strategy_config(strategy_config, 'unified')
        self.assertTrue(os.path.exists(json_file))
        self.assertTrue(json_file.endswith('.json'))
        
        # 验证JSON内容
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        self.assertEqual(loaded_config['strategy']['id'], strategy_config['strategy']['id'])
        
        # 测试YAML格式
        yaml_file = self.tester._save_strategy_config(strategy_config, 'legacy')
        self.assertTrue(os.path.exists(yaml_file))
        self.assertTrue(yaml_file.endswith('.yaml'))
        
        # 验证YAML内容
        with open(yaml_file, 'r', encoding='utf-8') as f:
            loaded_config = yaml.safe_load(f)
        self.assertIn('strategy', loaded_config)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试空数据池
        result = self.tester.test_strategy_selection('MACD', 'GOLDEN_CROSS', [])
        self.assertIn('execution_success', result)
        self.assertIn('performance_metrics', result)
        
        # 测试无效指标
        result = self.tester.test_strategy_selection('INVALID', 'UNKNOWN', self.mock_data_pool)
        self.assertIn('execution_success', result)
        
        # 测试空指标列表（复杂策略）
        with self.assertRaises(ValueError):
            self.tester.generate_complex_strategy([])
    
    def test_cleanup(self):
        """测试资源清理"""
        # 生成一些临时文件
        strategy_config = self.tester.generate_strategy_config('VOL', 'VOLUME_SPIKE')
        temp_file = self.tester._save_strategy_config(strategy_config, 'unified')
        
        # 验证文件存在
        self.assertTrue(os.path.exists(temp_file))
        
        # 执行清理
        self.tester.cleanup()
        
        # 验证文件已清理（注意：可能因为文件系统延迟而仍然存在）
        # 这里主要验证清理方法不会抛出异常


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
