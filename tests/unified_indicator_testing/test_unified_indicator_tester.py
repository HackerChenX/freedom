#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
统一指标测试器的单元测试

验证UnifiedIndicatorTester核心类的功能
"""

import os
import sys
import unittest
import tempfile
import shutil
import pandas as pd
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from unified_indicator_tester import (
    UnifiedIndicatorTester, 
    FrameworkError,
    ConfigurationError,
    DataGenerationError
)


class TestUnifiedIndicatorTester(unittest.TestCase):
    """统一指标测试器的单元测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, "test_config.yaml")
        
        # 创建测试配置文件
        test_config = """
test_framework:
  strict_mode: true
  timeout_seconds: 60

data_generation:
  pool_size: 10
  default_history_days: 60

indicators_test_matrix:
  completed_indicators:
    TEST_INDICATOR:
      patterns: ['TEST_PATTERN']
      history_requirement: 30
      expected_accuracy: 1.0

validation_criteria:
  buypoint_recognition_accuracy: 1.0
  selection_precision: 0.95
"""
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            f.write(test_config)
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization_success(self):
        """测试成功初始化"""
        tester = UnifiedIndicatorTester(config_path=self.config_file)
        
        self.assertIsNotNone(tester.config)
        self.assertIsNotNone(tester.test_session_id)
        self.assertIn('test_framework', tester.config)
        self.assertIn('indicators_test_matrix', tester.config)
        
        # 清理
        tester.cleanup()
    
    def test_initialization_with_invalid_config(self):
        """测试无效配置的初始化"""
        invalid_config_file = os.path.join(self.test_dir, "invalid_config.yaml")

        with open(invalid_config_file, 'w', encoding='utf-8') as f:
            f.write("invalid: yaml: content: [")

        with self.assertRaises(FrameworkError):
            UnifiedIndicatorTester(config_path=invalid_config_file)
    
    def test_initialization_with_missing_config(self):
        """测试缺失配置文件的初始化"""
        missing_config = os.path.join(self.test_dir, "missing_config.yaml")

        # 应该使用默认配置，不抛出异常
        tester = UnifiedIndicatorTester(config_path=missing_config)
        self.assertIsNotNone(tester.config)
        # 验证默认配置包含必需的部分
        self.assertIn('validation_criteria', tester.config)
        tester.cleanup()
    
    def test_config_validation(self):
        """测试配置验证"""
        tester = UnifiedIndicatorTester(config_path=self.config_file)
        
        # 验证配置加载正确
        self.assertTrue(tester.config['test_framework']['strict_mode'])
        self.assertEqual(tester.config['data_generation']['pool_size'], 10)
        
        tester.cleanup()
    
    def test_data_pool_generation(self):
        """测试数据池生成"""
        tester = UnifiedIndicatorTester(config_path=self.config_file)
        
        try:
            # 测试数据池生成
            indicator_config = tester.config['indicators_test_matrix']['completed_indicators']['TEST_INDICATOR']
            data_pool = tester._generate_comprehensive_data_pool(
                'TEST_INDICATOR', 'TEST_PATTERN', indicator_config, pool_size=5
            )
            
            self.assertIsInstance(data_pool, list)
            self.assertGreater(len(data_pool), 0)
            
            # 验证数据格式
            for data in data_pool:
                self.assertIn('code', data.columns)
                self.assertIn('close', data.columns)
                self.assertGreater(len(data), 0)
                
        finally:
            tester.cleanup()
    
    def test_pattern_score_calculation(self):
        """测试形态评分计算"""
        tester = UnifiedIndicatorTester(config_path=self.config_file)
        
        try:
            # 测试正常评分计算
            pattern_results = {
                'buypoint': {'score': 0.9},
                'selection': {'score': 0.8},
                'validation': {'score': 1.0}
            }
            
            score = tester._calculate_pattern_score(pattern_results)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            
            # 测试异常情况
            invalid_results = {
                'buypoint': {'score': 'invalid'},
                'selection': {},
                'validation': None
            }
            
            score = tester._calculate_pattern_score(invalid_results)
            self.assertEqual(score, 0.0)
            
        finally:
            tester.cleanup()
    
    def test_overall_score_calculation(self):
        """测试总体评分计算"""
        tester = UnifiedIndicatorTester(config_path=self.config_file)
        
        try:
            # 测试正常评分计算
            results = {
                'pattern_tests': {'TEST_PATTERN': {'pattern_score': 0.9}},
                'complex_condition_tests': {'status': 'COMPLETED'},
                'performance_tests': {'status': 'COMPLETED'}
            }
            pattern_scores = [0.9]
            
            overall_score = tester._calculate_overall_score(results, pattern_scores)
            self.assertGreaterEqual(overall_score, 0.9)
            self.assertLessEqual(overall_score, 1.0)
            
            # 测试空评分列表
            empty_score = tester._calculate_overall_score(results, [])
            self.assertEqual(empty_score, 0.0)
            
        finally:
            tester.cleanup()
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with UnifiedIndicatorTester(config_path=self.config_file) as tester:
            self.assertIsNotNone(tester.test_session_id)
            self.assertIsNotNone(tester.config)
        
        # 验证清理已执行（通过检查临时目录是否存在）
        # 注意：这个测试可能不够严格，因为清理是异步的
    
    def test_history_requirement_calculation(self):
        """测试历史数据需求计算"""
        tester = UnifiedIndicatorTester(config_path=self.config_file)
        
        try:
            # 测试从指标配置获取
            indicator_config = {'history_requirement': 50}
            requirement = tester._get_history_requirement('TEST_INDICATOR', indicator_config)
            self.assertEqual(requirement, 50)
            
            # 测试从全局配置获取
            requirement = tester._get_history_requirement('TEST_INDICATOR')
            self.assertEqual(requirement, 30)  # 来自测试配置
            
            # 测试默认值
            requirement = tester._get_history_requirement('UNKNOWN_INDICATOR')
            self.assertEqual(requirement, 60)  # 来自默认配置
            
        finally:
            tester.cleanup()
    
    def test_error_handling(self):
        """测试错误处理"""
        tester = UnifiedIndicatorTester(config_path=self.config_file)

        try:
            # 测试不存在的指标，应该返回错误结果而不是抛出异常
            result = tester.test_indicator_comprehensive('NONEXISTENT_INDICATOR')

            # 验证返回了错误结果
            self.assertIn('error', result)
            self.assertEqual(result['overall_score'], 0.0)
            self.assertEqual(result['status'], 'ERROR')

            # 验证错误信息包含指标名称
            self.assertIn('NONEXISTENT_INDICATOR', result['error'])

        finally:
            tester.cleanup()


class TestDataGenerator(unittest.TestCase):
    """数据生成器的单元测试"""
    
    def test_basic_data_generation(self):
        """测试基础数据生成"""
        from tests.unified_indicator_testing.unified_indicator_tester import StockInfoCompatibleDataGenerator

        generator = StockInfoCompatibleDataGenerator()

        # 测试基础数据生成（使用公共方法）
        data = generator.generate_random_stockinfo_data('TEST001', 30)

        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 25)  # 允许一些误差
        self.assertIn('code', data.columns)
        self.assertIn('close', data.columns)
        if not data.empty:
            self.assertEqual(data['code'].iloc[0], 'TEST001')
    
    def test_stockinfo_compatible_data(self):
        """测试StockInfo兼容数据生成"""
        from tests.unified_indicator_testing.unified_indicator_tester import StockInfoCompatibleDataGenerator
        
        generator = StockInfoCompatibleDataGenerator()
        
        # 测试兼容数据生成
        data = generator.generate_stockinfo_compatible_data(
            'TEST_INDICATOR', 'TEST_PATTERN', 'TEST001', 30
        )
        
        self.assertIsInstance(data, pd.DataFrame)
        if not data.empty:
            self.assertIn('code', data.columns)
            self.assertIn('close', data.columns)


if __name__ == '__main__':
    # 设置测试环境
    import pandas as pd
    
    # 运行测试
    unittest.main(verbosity=2)
