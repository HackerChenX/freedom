#!/usr/bin/env python3
from config import get_config
"""
指标验证框架测试

测试指标验证框架的各项功能
"""

import os
import sys
import unittest
import json
import tempfile
from unittest.mock import MagicMock
import pandas as pd

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines.indicator_validation_framework import (
    Indicator_validation_framework,
    Indicator_validation_config,
    Validation_mode,
    Validation_result
)


class Test_indicator_validation_framework(unittest.TestCase):
    """指标验证框架测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = Indicator_validation_config(
            mode=Validation_mode.QUICK,
            pool_size=stock_get_config('performance.pool_size'),
            max_selection_ratio=0.1,
            parallel_workers=1,
            save_details=False
        )
        
        # Mock数据管理器
        self.mock_data_manager = Mock()
        self.mock_strategy_executor = Mock()
        self.mock_strategy_manager = Mock()
        
        # 创建测试实例
        with patch('analysis.engines.indicator_validation_framework.get_unified_data_manager'), \
             patch('analysis.engines.indicator_validation_framework.StrategyExecutor'), \
             patch('analysis.engines.indicator_validation_framework.StrategyManager'):
            self.framework = Indicator_validation_framework(self.config)
            self.framework.data_manager = self.mock_data_manager
            self.framework.strategy_executor = self.mock_strategy_executor
            self.framework.strategy_manager = self.mock_strategy_manager
    
    def test_init(self):
        """测试初始化"""
        self.assert_equal(self.framework.config.mode, Validation_mode.QUICK)
        self.assert_equal(self.framework.config.stock_pool_size, 100)
        self.assert_is_not_none(self.framework.config.validation_date)
    
    def test_get_indicators_by_mode_quick(self):
        """测试快速模式指标获取"""
        with patch('analysis.engines.indicator_validation_framework.complete_registry') as mock_registry:
            mock_registry.get_indicator_names.return_value = [
                'MA', 'EMA', 'MACD', 'RSI', 'KDJ', 'BOLL', 'CCI', 'OTHER_INDICATOR'
            ]
            
            indicators = self.framework._get_indicators_by_mode()
            
            # 快速模式应该只返回核心指标
            expected_core = ['MA', 'EMA', 'MACD', 'RSI', 'KDJ', 'BOLL', 'CCI']
            self.assert_true(all(ind in expected_core for ind in indicators))
            self.assertNotIn('OTHER_INDICATOR', indicators)
    
    def test_get_indicators_by_mode_full(self):
        """测试完整模式指标获取"""
        self.framework.config.mode = Validation_mode.FULL
        
        with patch('analysis.engines.indicator_validation_framework.complete_registry') as mock_registry:
            all_indicators = ['MA', 'EMA', 'MACD', 'RSI', 'OTHER1', 'OTHER2']
            mock_registry.get_indicator_names.return_value = all_indicators
            
            indicators = self.framework._get_indicators_by_mode()
            
            # 完整模式应该返回所有指标
            self.assert_equal(set(indicators), set(all_indicators))
    
    def test_categorize_indicators(self):
        """测试指标分类"""
        test_indicators = [
            'MA', 'EMA', 'ENHANCED_RSI', 'ZXM_DAILY_MACD', 
            'COMPOSITE', 'CANDLESTICK_PATTERNS', 'OTHER_INDICATOR'
        ]
        
        categorized = self.framework._categorize_indicators(test_indicators)
        
        # 验证分类顺序：基础指标应该在前面
        basic_indices = [i for i, ind in enumerate(categorized) if ind in ['MA', 'EMA']]
        enhanced_indices = [i for i, ind in enumerate(categorized) if ind.startswith('ENHANCED_')]
        zxm_indices = [i for i, ind in enumerate(categorized) if ind.startswith('ZXM_')]
        
        self.assert_true(all(b < e for b in basic_indices for e in enhanced_indices))
        self.assert_true(all(e < z for e in enhanced_indices for z in zxm_indices))
    
    def test_prepare_stock_pool(self):
        """测试股票池准备"""
        # Mock数据库查询结果
        mock_result = pd.DataFrame({
            'code': ['000001', '000002', '600000', '600036', '000858']
        })
        self.mock_data_manager.execute_query.return_value = mock_result
        
        stock_pool = self.framework._prepare_stock_pool()
        
        self.assert_equal(len(stock_pool), 5)
        self.assertIn('000001', stock_pool)
        self.mock_data_manager.execute_query.assert_called_once()
    
    def test_prepare_stock_pool_fallback(self):
        """测试股票池准备失败时的回退机制"""
        # Mock数据库查询失败
        self.mock_data_manager.execute_query.side_effect = Exception("数据库连接失败")
        
        stock_pool = self.framework._prepare_stock_pool()
        
        # 应该返回默认股票池
        self.assert_greater(len(stock_pool), 0)
        self.assertIn('000001', stock_pool)
    
    def test_generate_indicator_strategy_ma(self):
        """测试MA指标策略生成"""
        strategy = self.framework._generate_indicator_strategy('MA')
        
        self.assert_is_instance(strategy, dict)
        self.assertEqual(strategy['strategy_id'], 'validate_ma')
        self.assertIn('conditions', strategy)
        self.assertGreater(len(strategy['conditions']), 0)
        
        # 验证MA特定条件
        ma_condition = strategy['conditions'][0]
        self.assertEqual(ma_condition['indicator'], 'MA')
    
    def test_generate_indicator_strategy_macd(self):
        """测试MACD指标策略生成"""
        strategy = self.framework._generate_indicator_strategy('MACD')
        
        self.assert_is_instance(strategy, dict)
        self.assertIn('conditions', strategy)
        
        # 验证MACD特定条件
        macd_condition = strategy['conditions'][0]
        self.assertEqual(macd_condition['indicator'], 'MACD')
        self.assertEqual(macd_condition['field'], 'dif')
    
    def test_generate_indicator_strategy_generic(self):
        """测试通用指标策略生成"""
        strategy = self.framework._generate_indicator_strategy('UNKNOWN_INDICATOR')
        
        self.assert_is_instance(strategy, dict)
        self.assertIn('conditions', strategy)
        
        # 验证通用条件
        condition = strategy['conditions'][0]
        self.assertEqual(condition['indicator'], 'UNKNOWN_INDICATOR')
        self.assertEqual(condition['operator'], '>')
        self.assertEqual(condition['value'], 0)
    
    def test_execute_strategy_selection_success(self):
        """测试策略选股执行成功"""
        # Mock策略执行结果
        mock_result = pd.DataFrame({
            'code': ['000001', '000002', '600000'],
            'score': [0.8, 0.7, 0.6]
        })
        self.mock_strategy_executor.execute_strategy.return_value = mock_result
        
        strategy_config = {'test': 'config'}
        stock_pool = ['000001', '000002', '600000', '600036']
        
        selected_stocks = self.framework._execute_strategy_selection(strategy_config, stock_pool)
        
        self.assert_equal(len(selected_stocks), 3)
        self.assertEqual(selected_stocks, ['000001', '000002', '600000'])
    
    def test_execute_strategy_selection_empty_result(self):
        """测试策略选股执行返回空结果"""
        self.mock_strategy_executor.execute_strategy.return_value = pd.DataFrame()
        
        strategy_config = {'test': 'config'}
        stock_pool = ['000001', '000002']
        
        selected_stocks = self.framework._execute_strategy_selection(strategy_config, stock_pool)
        
        self.assert_equal(len(selected_stocks), 0)
    
    def test_execute_strategy_selection_error(self):
        """测试策略选股执行出错"""
        self.mock_strategy_executor.execute_strategy.side_effect = Exception("执行失败")
        
        strategy_config = {'test': 'config'}
        stock_pool = ['000001', '000002']
        
        selected_stocks = self.framework._execute_strategy_selection(strategy_config, stock_pool)
        
        self.assert_equal(len(selected_stocks), 0)
    
    def test_validate_indicator_success(self):
        """测试指标验证成功"""
        stock_pool = ['000001', '000002', '600000'] * 20  # 60只股票
        
        # Mock策略生成
        with patch.object(self.framework, '_generate_indicator_strategy') as mock_gen_strategy:
            mock_gen_strategy.return_value = {'test': 'strategy'}
            
            # Mock策略执行，返回3只股票（5%选股率）
            with patch.object(self.framework, '_execute_strategy_selection') as mock_execute:
                mock_execute.return_value = ['000001', '000002', '600000']
                
                result = self.framework._validate_indicator('MA', stock_pool)
                
                self.assertEqual(result['indicator_name'], 'MA')
                self.assertEqual(result['status'], ValidationResult.SUCCESS.value)
                self.assertEqual(result['selected_count'], 3)
                self.assertAlmostEqual(result['selection_ratio'], 0.05, places=2)
    
    def test_validate_indicator_no_selection(self):
        """测试指标验证无选股"""
        stock_pool = ['000001', '000002', '600000']
        
        with patch.object(self.framework, '_generate_indicator_strategy') as mock_gen_strategy:
            mock_gen_strategy.return_value = {'test': 'strategy'}
            
            with patch.object(self.framework, '_execute_strategy_selection') as mock_execute:
                mock_execute.return_value = []  # 无选股
                
                result = self.framework._validate_indicator('MA', stock_pool)
                
                self.assertEqual(result['status'], ValidationResult.NO_SELECTION.value)
                self.assertEqual(result['selected_count'], 0)
    
    def test_validate_indicator_over_selection(self):
        """测试指标验证过度选择"""
        stock_pool = ['000001', '000002', '600000'] * 10  # 30只股票
        
        with patch.object(self.framework, '_generate_indicator_strategy') as mock_gen_strategy:
            mock_gen_strategy.return_value = {'test': 'strategy'}
            
            # Mock策略执行，返回15只股票（50%选股率，超过10%阈值）
            with patch.object(self.framework, '_execute_strategy_selection') as mock_execute:
                mock_execute.return_value = ['stock' + str(i) for i in range(15)]
                
                result = self.framework._validate_indicator('MA', stock_pool)
                
                self.assertEqual(result['status'], ValidationResult.OVER_SELECTION.value)
                self.assertEqual(result['selected_count'], 15)
                self.assertAlmostEqual(result['selection_ratio'], 0.5, places=2)
    
    def test_validate_indicator_error(self):
        """测试指标验证出错"""
        stock_pool = ['000001', '000002']
        
        with patch.object(self.framework, '_generate_indicator_strategy') as mock_gen_strategy:
            mock_gen_strategy.side_effect = Exception("策略生成失败")
            
            result = self.framework._validate_indicator('MA', stock_pool)
            
            self.assertEqual(result['status'], ValidationResult.ERROR.value)
            self.assertIn('error_message', result)
    
    def test_validate_single_indicator(self):
        """测试单个指标验证"""
        with patch.object(self.framework, '_prepare_stock_pool') as mock_prepare:
            mock_prepare.return_value = ['000001', '000002', '600000']
            
            with patch.object(self.framework, '_validate_indicator') as mock_validate:
                mock_validate.return_value = {
                    'indicator_name': 'MA',
                    'status': ValidationResult.SUCCESS.value,
                    'selected_count': 2
                }
                
                result = self.framework.validate_single_indicator('MA')
                
                self.assertEqual(result['indicator_name'], 'MA')
                self.assertEqual(result['status'], ValidationResult.SUCCESS.value)
                mock_prepare.assert_called_once()
                mock_validate.assert_called_once_with('MA', ['000001', '000002', '600000'])
    
    def test_generate_summary(self):
        """测试总结生成"""
        test_results = [
            {
                'indicator_name': 'MA',
                'status': ValidationResult.SUCCESS.value,
                'selected_count': 5,
                'selection_ratio': 0.05
            },
            {
                'indicator_name': 'EMA',
                'status': ValidationResult.SUCCESS.value,
                'selected_count': 3,
                'selection_ratio': 0.03
            },
            {
                'indicator_name': 'RSI',
                'status': ValidationResult.NO_SELECTION.value,
                'selected_count': 0,
                'selection_ratio': 0.0
            },
            {
                'indicator_name': 'ERROR_IND',
                'status': ValidationResult.ERROR.value,
                'error_message': '测试错误'
            }
        ]
        
        summary = self.framework._generate_summary(test_results)
        
        self.assertEqual(summary['total_indicators'], 4)
        self.assertEqual(summary['validation_results'][ValidationResult.SUCCESS.value], 2)
        self.assertEqual(summary['validation_results'][ValidationResult.NO_SELECTION.value], 1)
        self.assertEqual(summary['validation_results'][ValidationResult.ERROR.value], 1)
        self.assertEqual(summary['success_rate'], 0.5)
        self.assertEqual(summary['total_selected_stocks'], 8)
        
        # 验证性能最好的指标排序
        self.assertEqual(summary['top_performing_indicators'][0]['indicator'], 'MA')
        self.assertEqual(summary['top_performing_indicators'][1]['indicator'], 'EMA')
        
        # 验证失败指标记录
        self.assertEqual(len(summary['failed_indicators']), 1)
        self.assertEqual(summary['failed_indicators'][0]['indicator'], 'ERROR_IND')
    
    def test_save_results_json(self):
        """测试JSON格式结果保存"""
        self.framework.config.save_details = True
        self.framework.config.output_format = "json"
        
        test_results = [{'indicator_name': 'MA', 'status': 'success'}]
        test_summary = {'total_indicators': 1}
        
        with patch('analysis.engines.indicator_validation_framework.get_result_dir') as mock_get_dir:
            with tempfile.Temporary_directory() as temp_dir:
                mock_get_dir.return_value = temp_dir
                
                # 不应该抛出异常
                try:
                    self.framework._save_results(test_results, test_summary)
                except Exception as e:
                    self.fail(f"保存结果时出错: {e}")
    
    def test_validate_sequential(self):
        """测试顺序验证"""
        indicators = ['MA', 'EMA', 'RSI']
        stock_pool = ['000001', '000002']
        
        with patch.object(self.framework, '_validate_indicator') as mock_validate:
            mock_validate.side_effect = [
                {'indicator_name': 'MA', 'status': ValidationResult.SUCCESS.value},
                {'indicator_name': 'EMA', 'status': ValidationResult.NO_SELECTION.value},
                {'indicator_name': 'RSI', 'status': ValidationResult.ERROR.value}
            ]
            
            results = self.framework._validate_sequential(indicators, stock_pool)
            
            self.assert_equal(len(results), 3)
            self.assert_equal(mock_validate.call_count, 3)
            self.assertEqual(self.framework.validation_stats['validated_indicators'], 3)
            self.assertEqual(self.framework.validation_stats['successful_validations'], 1)
            self.assertEqual(self.framework.validation_stats['failed_validations'], 2)


class Test_validation_config(unittest.TestCase):
    """验证配置测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = Indicator_validation_config()
        
        self.assert_equal(config.mode, Validation_mode.FULL)
        self.assert_equal(config.stock_pool_size, 1000)
        self.assert_equal(config.max_selection_ratio, 0.1)
        self.assert_equal(config.min_selection_count, 1)
        self.assert_equal(config.timeout_seconds, 300)
        self.assert_equal(config.parallel_workers, 4)
        self.assertEqual(config.output_format, "json")
        self.assert_true(config.save_details)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = Indicator_validation_config(
            mode=Validation_mode.QUICK,
            pool_size=stock_get_config('performance.pool_size'),
            max_selection_ratio=0.05,
            parallel_workers=2,
            output_format="csv"
        )
        
        self.assert_equal(config.mode, Validation_mode.QUICK)
        self.assert_equal(config.stock_pool_size, 500)
        self.assert_equal(config.max_selection_ratio, 0.05)
        self.assert_equal(config.parallel_workers, 2)
        self.assertEqual(config.output_format, "csv")


class Test_validation_enums(unittest.TestCase):
    """验证枚举测试类"""
    
    def test_validation_mode_enum(self):
        """测试验证模式枚举"""
        self.assertEqual(ValidationMode.QUICK.value, "quick")
        self.assertEqual(ValidationMode.PRIORITY.value, "priority")
        self.assertEqual(ValidationMode.CATEGORY.value, "category")
        self.assertEqual(ValidationMode.FULL.value, "full")
    
    def test_validation_result_enum(self):
        """测试验证结果枚举"""
        self.assertEqual(ValidationResult.SUCCESS.value, "success")
        self.assertEqual(ValidationResult.NO_SELECTION.value, "no_selection")
        self.assertEqual(ValidationResult.OVER_SELECTION.value, "over_selection")
        self.assertEqual(ValidationResult.ERROR.value, "error")
        self.assertEqual(ValidationResult.TIMEOUT.value, "timeout")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2) 