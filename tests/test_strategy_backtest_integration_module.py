#!/usr/bin/env python3
"""
策略与回测集成模块测试脚本

全面测试策略选股与买点回测的集成优化功能，
验证双向验证机制、数据流优化和集成分析控制器。

测试覆盖：
1. 集成引擎测试
2. 数据流优化器测试
3. 双向验证系统测试
4. 集成分析控制器测试
5. 性能基准测试
6. 错误处理测试
7. 集成工作流测试
8. 配置灵活性测试
9. 资源管理测试
"""

import sys
import os
import time
import unittest
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from strategy.integrated_strategy_backtest_engine import (
    IntegratedStrategyBacktestEngine, IntegrationConfig, IntegrationType, ValidationMode
)
from strategy.integrated_data_flow_optimizer import IntegratedDataFlowOptimizer
from strategy.bidirectional_validation_system import (
    BidirectionalValidationSystem, ValidationLevel
)
from strategy.integrated_analysis_controller import IntegratedAnalysisController
from utils.logger import get_logger

logger = get_logger(__name__)


class TestStrategyBacktestIntegrationModule(unittest.TestCase):
    """策略与回测集成模块测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_stock_codes = ['000001', '000002', '600000', '600036', '000858']
        self.test_date = '2024-01-15'
        self.start_time = time.time()
        
        logger.info("=" * 60)
        logger.info(f"开始测试: {self._testMethodName}")
        logger.info("=" * 60)
    
    def tearDown(self):
        """测试后清理"""
        execution_time = time.time() - self.start_time
        logger.info(f"测试 {self._testMethodName} 完成，耗时: {execution_time:.2f}秒")
        logger.info("-" * 60)
    
    def test_01_integrated_strategy_backtest_engine(self):
        """测试集成策略回测引擎"""
        logger.info("🔧 测试集成策略回测引擎...")
        
        try:
            # 创建不同配置的引擎
            configs = [
                IntegrationConfig(integration_type=IntegrationType.PARALLEL),
                IntegrationConfig(integration_type=IntegrationType.STRATEGY_FIRST),
                IntegrationConfig(validation_mode=ValidationMode.WEIGHTED),
                IntegrationConfig(validation_mode=ValidationMode.STRICT)
            ]
            
            for i, config in enumerate(configs):
                logger.info(f"测试配置 {i+1}: {config.integration_type.value}, {config.validation_mode.value}")
                
                engine = IntegratedStrategyBacktestEngine(config)
                
                # 测试集成分析
                result = engine.run_integrated_analysis(
                    stock_codes=self.test_stock_codes[:2],  # 限制数量以提高测试速度
                    analysis_date=self.test_date
                )
                
                # 验证结果
                self.assertIsNotNone(result)
                self.assertGreater(result.total_stocks, 0)
                self.assertIsInstance(result.execution_time, float)
                self.assertIsInstance(result.performance_metrics, dict)
                
                # 获取性能报告
                performance_report = engine.get_performance_report()
                self.assertIn('performance_stats', performance_report)
                self.assertIn('config', performance_report)
                
                # 清理资源
                engine.cleanup()
                
                logger.info(f"✅ 配置 {i+1} 测试通过")
            
            logger.info("✅ 集成策略回测引擎测试通过")
            
        except Exception as e:
            logger.error(f"❌ 集成策略回测引擎测试失败: {e}")
            raise
    
    def test_02_integrated_data_flow_optimizer(self):
        """测试集成数据流优化器"""
        logger.info("🔧 测试集成数据流优化器...")
        
        try:
            optimizer = IntegratedDataFlowOptimizer(cache_size_mb=64, max_workers=4)
            
            # 测试数据获取
            for stock_code in self.test_stock_codes[:3]:
                data = optimizer.get_stock_data(
                    stock_code=stock_code,
                    start_date='2024-01-01',
                    end_date='2024-01-31',
                    data_types=['daily', 'indicators'],
                    requester='test'
                )
                
                self.assertIsInstance(data, dict)
                logger.info(f"✅ 股票 {stock_code} 数据获取成功")
            
            # 测试缓存功能
            cache_stats_before = optimizer.get_cache_stats()
            
            # 重复请求相同数据（应该命中缓存）
            data = optimizer.get_stock_data(
                stock_code=self.test_stock_codes[0],
                start_date='2024-01-01',
                end_date='2024-01-31',
                data_types=['daily', 'indicators'],
                requester='test_cache'
            )
            
            cache_stats_after = optimizer.get_cache_stats()
            
            # 验证缓存统计
            self.assertIsInstance(cache_stats_before, dict)
            self.assertIsInstance(cache_stats_after, dict)
            
            # 测试缓存清理
            optimizer.clear_cache()
            cache_stats_cleared = optimizer.get_cache_stats()
            self.assertEqual(cache_stats_cleared['cache_stats']['total_size_bytes'], 0)
            
            # 清理资源
            optimizer.cleanup()
            
            logger.info("✅ 集成数据流优化器测试通过")
            
        except Exception as e:
            logger.error(f"❌ 集成数据流优化器测试失败: {e}")
            raise
    
    def test_03_bidirectional_validation_system(self):
        """测试双向验证系统"""
        logger.info("🔧 测试双向验证系统...")
        
        try:
            # 测试不同验证级别
            validation_levels = [
                ValidationLevel.BASIC,
                ValidationLevel.STANDARD,
                ValidationLevel.STRICT
            ]
            
            for level in validation_levels:
                logger.info(f"测试验证级别: {level.value}")
                
                validator = BidirectionalValidationSystem(level)
                
                # 创建模拟数据
                mock_strategy_result = {
                    'score': 75.0,
                    'recommendation': '买入',
                    'match_details': {
                        'passing_indicators': ['MACD', 'RSI', 'KDJ'],
                        'failing_indicators': ['BOLL']
                    }
                }
                
                mock_backtest_result = {
                    'buypoint_signals': [
                        type('Signal', (), {
                            'confidence': 0.8,
                            'signal_type': type('SignalType', (), {'value': 'VOLUME_BREAKOUT'})()
                        })(),
                        type('Signal', (), {
                            'confidence': 0.7,
                            'signal_type': type('SignalType', (), {'value': 'TREND_REVERSAL'})()
                        })()
                    ],
                    'backtest_summary': {
                        'success_rate': 0.75,
                        'average_score': 72.0
                    }
                }
                
                # 执行验证
                validation_report = validator.validate_integrated_result(
                    stock_code='TEST001',
                    strategy_result=mock_strategy_result,
                    backtest_result=mock_backtest_result
                )
                
                # 验证报告结构
                self.assertIsNotNone(validation_report)
                self.assertEqual(validation_report.stock_code, 'TEST001')
                self.assertEqual(validation_report.validation_level, level)
                self.assertIsInstance(validation_report.confidence_score, float)
                self.assertIsInstance(validation_report.recommendations, list)
                self.assertIsInstance(validation_report.execution_time, float)
                
                # 获取验证统计
                stats = validator.get_validation_stats()
                self.assertIn('total_validations', stats)
                self.assertIn('average_confidence', stats)
                
                logger.info(f"✅ 验证级别 {level.value} 测试通过")
            
            logger.info("✅ 双向验证系统测试通过")
            
        except Exception as e:
            logger.error(f"❌ 双向验证系统测试失败: {e}")
            raise
    
    def test_04_integrated_analysis_controller(self):
        """测试集成分析控制器"""
        logger.info("🔧 测试集成分析控制器...")
        
        try:
            controller = IntegratedAnalysisController(
                validation_level=ValidationLevel.STANDARD,
                enable_data_optimization=True
            )
            
            # 测试快速分析
            quick_result = controller.run_quick_analysis(
                stock_codes=self.test_stock_codes[:2],
                analysis_type="parallel",
                analysis_date=self.test_date
            )
            
            # 验证快速分析结果
            self.assertIsInstance(quick_result, dict)
            self.assertIn('analysis_type', quick_result)
            self.assertIn('execution_time', quick_result)
            self.assertIn('integration_summary', quick_result)
            
            logger.info(f"✅ 快速分析完成，耗时: {quick_result['execution_time']:.2f}秒")
            
            # 测试综合分析（使用更少的股票以提高测试速度）
            comprehensive_result = controller.run_comprehensive_analysis(
                stock_codes=self.test_stock_codes[:2],
                analysis_date=self.test_date,
                export_results=False  # 测试时不导出文件
            )
            
            # 验证综合分析结果
            self.assertIsInstance(comprehensive_result, dict)
            self.assertIn('report_metadata', comprehensive_result)
            self.assertIn('integration_summary', comprehensive_result)
            self.assertIn('validation_analysis', comprehensive_result)
            self.assertIn('investment_recommendations', comprehensive_result)
            
            # 获取性能报告
            performance_report = controller.get_performance_report()
            self.assertIn('controller_stats', performance_report)
            
            # 清理资源
            controller.cleanup()
            
            logger.info("✅ 集成分析控制器测试通过")
            
        except Exception as e:
            logger.error(f"❌ 集成分析控制器测试失败: {e}")
            raise
    
    def test_05_performance_benchmark(self):
        """测试性能基准"""
        logger.info("🔧 测试性能基准...")
        
        try:
            # 测试不同规模的数据处理性能
            test_cases = [
                {'stocks': 2, 'expected_time': 10.0},
                {'stocks': 5, 'expected_time': 20.0}
            ]
            
            for case in test_cases:
                start_time = time.time()
                
                controller = IntegratedAnalysisController()
                result = controller.run_quick_analysis(
                    stock_codes=self.test_stock_codes[:case['stocks']],
                    analysis_date=self.test_date
                )
                
                execution_time = time.time() - start_time
                
                # 验证性能
                self.assertLess(execution_time, case['expected_time'])
                self.assertIsInstance(result, dict)
                
                controller.cleanup()
                
                logger.info(f"✅ {case['stocks']}只股票处理完成，耗时: {execution_time:.2f}秒")
            
            logger.info("✅ 性能基准测试通过")
            
        except Exception as e:
            logger.error(f"❌ 性能基准测试失败: {e}")
            raise
    
    def test_06_error_handling(self):
        """测试错误处理"""
        logger.info("🔧 测试错误处理...")
        
        try:
            controller = IntegratedAnalysisController()
            
            # 测试空股票列表
            result = controller.run_quick_analysis(
                stock_codes=[],
                analysis_date=self.test_date
            )
            self.assertIsInstance(result, dict)
            
            # 测试无效日期格式
            result = controller.run_quick_analysis(
                stock_codes=['000001'],
                analysis_date='invalid_date'
            )
            self.assertIsInstance(result, dict)
            
            # 测试无效股票代码
            result = controller.run_quick_analysis(
                stock_codes=['INVALID_CODE'],
                analysis_date=self.test_date
            )
            self.assertIsInstance(result, dict)
            
            controller.cleanup()
            
            logger.info("✅ 错误处理测试通过")
            
        except Exception as e:
            logger.error(f"❌ 错误处理测试失败: {e}")
            raise
    
    def test_07_integration_workflow(self):
        """测试集成工作流"""
        logger.info("🔧 测试集成工作流...")
        
        try:
            # 创建完整的集成工作流
            config = IntegrationConfig(
                integration_type=IntegrationType.PARALLEL,
                validation_mode=ValidationMode.WEIGHTED,
                enable_caching=True
            )
            
            controller = IntegratedAnalysisController(
                integration_config=config,
                validation_level=ValidationLevel.STANDARD,
                enable_data_optimization=True
            )
            
            # 执行完整工作流
            result = controller.run_comprehensive_analysis(
                stock_codes=self.test_stock_codes[:2],
                analysis_date=self.test_date,
                export_results=False
            )
            
            # 验证工作流结果
            self.assertIn('report_metadata', result)
            self.assertIn('integration_summary', result)
            self.assertIn('validation_analysis', result)
            self.assertIn('investment_recommendations', result)
            self.assertIn('risk_assessment', result)
            self.assertIn('next_steps', result)
            
            # 验证投资建议结构
            recommendations = result['investment_recommendations']
            self.assertIn('priority_investments', recommendations)
            self.assertIn('portfolio_allocation', recommendations)
            self.assertIn('risk_level', recommendations)
            
            controller.cleanup()
            
            logger.info("✅ 集成工作流测试通过")
            
        except Exception as e:
            logger.error(f"❌ 集成工作流测试失败: {e}")
            raise
    
    def test_08_configuration_flexibility(self):
        """测试配置灵活性"""
        logger.info("🔧 测试配置灵活性...")
        
        try:
            # 测试不同的集成配置组合
            test_configs = [
                {
                    'integration_type': IntegrationType.STRATEGY_FIRST,
                    'validation_mode': ValidationMode.STRICT,
                    'strategy_weight': 0.8,
                    'backtest_weight': 0.2
                },
                {
                    'integration_type': IntegrationType.BACKTEST_FIRST,
                    'validation_mode': ValidationMode.CONSENSUS,
                    'strategy_weight': 0.3,
                    'backtest_weight': 0.7
                },
                {
                    'integration_type': IntegrationType.PARALLEL,
                    'validation_mode': ValidationMode.ADAPTIVE,
                    'enable_caching': False,
                    'max_parallel_tasks': 5
                }
            ]
            
            for i, config_dict in enumerate(test_configs):
                logger.info(f"测试配置组合 {i+1}")
                
                config = IntegrationConfig(**config_dict)
                controller = IntegratedAnalysisController(
                    integration_config=config,
                    validation_level=ValidationLevel.BASIC
                )
                
                result = controller.run_quick_analysis(
                    stock_codes=self.test_stock_codes[:1],
                    analysis_date=self.test_date
                )
                
                self.assertIsInstance(result, dict)
                self.assertIn('integration_summary', result)
                
                controller.cleanup()
                
                logger.info(f"✅ 配置组合 {i+1} 测试通过")
            
            logger.info("✅ 配置灵活性测试通过")
            
        except Exception as e:
            logger.error(f"❌ 配置灵活性测试失败: {e}")
            raise
    
    def test_09_resource_management(self):
        """测试资源管理"""
        logger.info("🔧 测试资源管理...")
        
        try:
            # 创建多个组件实例
            components = []
            
            # 创建集成引擎
            engine = IntegratedStrategyBacktestEngine()
            components.append(engine)
            
            # 创建数据优化器
            optimizer = IntegratedDataFlowOptimizer()
            components.append(optimizer)
            
            # 创建验证系统
            validator = BidirectionalValidationSystem()
            components.append(validator)
            
            # 创建控制器
            controller = IntegratedAnalysisController()
            components.append(controller)
            
            # 执行一些操作
            result = controller.run_quick_analysis(
                stock_codes=self.test_stock_codes[:1],
                analysis_date=self.test_date
            )
            self.assertIsInstance(result, dict)
            
            # 清理所有资源
            for component in components:
                if hasattr(component, 'cleanup'):
                    component.cleanup()
            
            logger.info("✅ 资源管理测试通过")
            
        except Exception as e:
            logger.error(f"❌ 资源管理测试失败: {e}")
            raise


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始策略与回测集成模块测试...")
    print("=" * 80)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestStrategyBacktestIntegrationModule)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    
    result = runner.run(test_suite)
    
    execution_time = time.time() - start_time
    
    # 输出测试结果
    print("=" * 80)
    print(f"📊 测试完成！")
    print(f"⏱️  总耗时: {execution_time:.2f}秒")
    print(f"✅ 通过: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")
    print(f"❌ 失败: {len(result.failures)}")
    print(f"💥 错误: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 成功率: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
