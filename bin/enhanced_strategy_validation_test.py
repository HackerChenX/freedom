#!/usr/bin/env python3
"""
增强的策略选股验证测试
修正空结果判定问题，实现严格的测试标准
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from strategy.strategy_engine import StrategyEngine
from strategy.strategy_executor import StrategyExecutor
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

class EnhancedStrategyValidationTester:
    """增强的策略选股验证测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.strategy_engine = None
        self.strategy_executor = None
        self.test_results = {
            'test_date': datetime.now().strftime('%Y-%m-%d'),
            'test_phase': '增强策略选股验证测试',
            'test_cases': {},
            'validation_standards': {
                'min_selections_per_strategy': 1,
                'min_success_rate': 0.8,
                'max_empty_results_tolerance': 0.2
            },
            'overall_metrics': {},
            'issues': [],
            'recommendations': []
        }
        
    @exception_handler(reraise=True)
    def initialize_components(self):
        """初始化测试组件"""
        logger.info("初始化增强策略验证测试组件...")
        
        try:
            # 初始化策略引擎
            self.strategy_engine = StrategyEngine()
            logger.info("✅ 策略引擎初始化完成")
            
            # 初始化策略执行器
            self.strategy_executor = StrategyExecutor()
            logger.info("✅ 策略执行器初始化完成")
            
            return True
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            self.test_results['issues'].append(f"组件初始化失败: {e}")
            return False
    
    @performance_monitor(threshold_seconds=60.0)
    def test_strategy_effectiveness_validation(self) -> Dict[str, Any]:
        """测试策略有效性验证"""
        logger.info("=== 测试策略有效性验证 ===")
        
        test_result = {
            'test_name': '策略有效性验证',
            'status': 'UNKNOWN',
            'execution_time': 0,
            'strategies_tested': 0,
            'effective_strategies': 0,
            'empty_result_strategies': 0,
            'failed_strategies': 0,
            'effectiveness_rate': '0%',
            'details': []
        }
        
        start_time = time.time()
        
        try:
            # 获取策略列表
            strategy_list = self.strategy_engine.get_available_strategies()
            logger.info(f"发现 {len(strategy_list)} 个可用策略")
            
            # 准备测试数据
            test_stocks = ['000001', '600519', '300005', '002415', '000858']
            target_date = '2025-05-09'
            
            strategies_tested = 0
            effective_strategies = 0
            empty_result_strategies = 0
            failed_strategies = 0
            
            for strategy_id in strategy_list[:10]:  # 测试前10个策略
                try:
                    logger.info(f"验证策略: {strategy_id}")
                    strategies_tested += 1
                    
                    # 执行策略选股
                    selections = self.strategy_engine.execute_strategy(
                        strategy_id, test_stocks, target_date
                    )
                    
                    selections_count = len(selections) if selections else 0
                    
                    if selections_count > 0:
                        # 策略有效
                        effective_strategies += 1
                        detail = {
                            'strategy_id': strategy_id,
                            'status': 'EFFECTIVE',
                            'selections_count': selections_count,
                            'sample_selections': selections[:3] if selections else [],
                            'validation_result': 'PASSED',
                            'effectiveness_score': min(selections_count / len(test_stocks), 1.0)
                        }
                        logger.info(f"  ✅ 策略 {strategy_id} 有效，选出 {selections_count} 只股票")
                        
                    elif selections_count == 0:
                        # 策略返回空结果
                        empty_result_strategies += 1
                        detail = {
                            'strategy_id': strategy_id,
                            'status': 'EMPTY_RESULT',
                            'selections_count': 0,
                            'sample_selections': [],
                            'validation_result': 'FAILED',
                            'failure_reason': '策略未选出任何股票',
                            'possible_causes': [
                                '策略条件过于严格',
                                '测试数据不符合策略要求',
                                '策略算法实现错误',
                                '数据库连接或数据问题'
                            ],
                            'effectiveness_score': 0.0
                        }
                        logger.warning(f"  ⚠️ 策略 {strategy_id} 返回空结果")
                        
                    test_result['details'].append(detail)
                    
                except Exception as e:
                    # 策略执行失败
                    failed_strategies += 1
                    detail = {
                        'strategy_id': strategy_id,
                        'status': 'EXECUTION_FAILED',
                        'selections_count': 0,
                        'sample_selections': [],
                        'validation_result': 'ERROR',
                        'error': str(e),
                        'effectiveness_score': 0.0
                    }
                    test_result['details'].append(detail)
                    logger.error(f"  ❌ 策略 {strategy_id} 执行失败: {e}")
            
            # 计算指标
            test_result['strategies_tested'] = strategies_tested
            test_result['effective_strategies'] = effective_strategies
            test_result['empty_result_strategies'] = empty_result_strategies
            test_result['failed_strategies'] = failed_strategies
            
            if strategies_tested > 0:
                effectiveness_rate = (effective_strategies / strategies_tested) * 100
                test_result['effectiveness_rate'] = f"{effectiveness_rate:.1f}%"
                
                # 应用严格的判定标准
                min_success_rate = self.test_results['validation_standards']['min_success_rate']
                max_empty_tolerance = self.test_results['validation_standards']['max_empty_results_tolerance']
                
                empty_rate = empty_result_strategies / strategies_tested
                
                if effectiveness_rate >= min_success_rate * 100 and empty_rate <= max_empty_tolerance:
                    test_result['status'] = 'SUCCESS'
                    test_result['validation_summary'] = f'策略有效性达标：{effectiveness_rate:.1f}%有效率，{empty_rate*100:.1f}%空结果率'
                elif effectiveness_rate >= 50:
                    test_result['status'] = 'PARTIAL_SUCCESS'
                    test_result['validation_summary'] = f'策略有效性部分达标：{effectiveness_rate:.1f}%有效率，需要改进'
                else:
                    test_result['status'] = 'FAILED'
                    test_result['validation_summary'] = f'策略有效性不达标：仅{effectiveness_rate:.1f}%有效率，{empty_rate*100:.1f}%空结果率'
            else:
                test_result['status'] = 'FAILED'
                test_result['effectiveness_rate'] = '0%'
                test_result['validation_summary'] = '未能测试任何策略'
                
        except Exception as e:
            test_result['status'] = 'FAILED'
            test_result['error'] = str(e)
            logger.error(f"策略有效性验证测试失败: {e}")
        
        test_result['execution_time'] = time.time() - start_time
        return test_result
    
    @performance_monitor(threshold_seconds=90.0)
    def test_data_driven_strategy_validation(self) -> Dict[str, Any]:
        """测试数据驱动的策略验证"""
        logger.info("=== 测试数据驱动的策略验证 ===")
        
        test_result = {
            'test_name': '数据驱动策略验证',
            'status': 'UNKNOWN',
            'execution_time': 0,
            'data_scenarios': 0,
            'successful_scenarios': 0,
            'validation_details': []
        }
        
        start_time = time.time()
        
        try:
            # 定义不同的数据场景
            data_scenarios = [
                {
                    'name': '大盘蓝筹股测试',
                    'stocks': ['000001', '600519', '000858', '600036'],
                    'expected_min_selections': 1,
                    'description': '测试大盘蓝筹股的策略选股效果'
                },
                {
                    'name': '创业板股票测试',
                    'stocks': ['300005', '300015', '300033'],
                    'expected_min_selections': 1,
                    'description': '测试创业板股票的策略选股效果'
                },
                {
                    'name': '混合股票池测试',
                    'stocks': ['000001', '300005', '600519', '002415'],
                    'expected_min_selections': 1,
                    'description': '测试混合股票池的策略选股效果'
                }
            ]
            
            target_date = '2025-05-09'
            test_strategy_id = 'auto_generated_300005_20250509_strategy'  # 使用一个已知策略
            
            successful_scenarios = 0
            
            for scenario in data_scenarios:
                try:
                    logger.info(f"执行数据场景: {scenario['name']}")
                    
                    # 执行策略选股
                    selections = self.strategy_engine.execute_strategy(
                        test_strategy_id, scenario['stocks'], target_date
                    )
                    
                    selections_count = len(selections) if selections else 0
                    expected_min = scenario['expected_min_selections']
                    
                    scenario_detail = {
                        'scenario_name': scenario['name'],
                        'test_stocks': scenario['stocks'],
                        'selections_count': selections_count,
                        'expected_min_selections': expected_min,
                        'meets_expectation': selections_count >= expected_min,
                        'description': scenario['description']
                    }
                    
                    if selections_count >= expected_min:
                        successful_scenarios += 1
                        scenario_detail['status'] = 'SUCCESS'
                        scenario_detail['validation'] = f'达到预期：选出{selections_count}只股票（≥{expected_min}）'
                        logger.info(f"  ✅ {scenario['name']}: 选出 {selections_count} 只股票")
                    else:
                        scenario_detail['status'] = 'FAILED'
                        scenario_detail['validation'] = f'未达预期：仅选出{selections_count}只股票（<{expected_min}）'
                        scenario_detail['failure_analysis'] = '可能原因：数据不足、策略条件不匹配、算法问题'
                        logger.warning(f"  ❌ {scenario['name']}: 仅选出 {selections_count} 只股票")
                    
                    test_result['validation_details'].append(scenario_detail)
                    
                except Exception as e:
                    scenario_detail = {
                        'scenario_name': scenario['name'],
                        'status': 'ERROR',
                        'error': str(e),
                        'validation': f'执行失败：{str(e)}'
                    }
                    test_result['validation_details'].append(scenario_detail)
                    logger.error(f"  ❌ {scenario['name']} 执行失败: {e}")
            
            # 计算结果
            test_result['data_scenarios'] = len(data_scenarios)
            test_result['successful_scenarios'] = successful_scenarios
            
            success_rate = (successful_scenarios / len(data_scenarios)) * 100
            
            if success_rate >= 80:
                test_result['status'] = 'SUCCESS'
            elif success_rate >= 50:
                test_result['status'] = 'PARTIAL_SUCCESS'
            else:
                test_result['status'] = 'FAILED'
            
            test_result['success_rate'] = f"{success_rate:.1f}%"
            
        except Exception as e:
            test_result['status'] = 'FAILED'
            test_result['error'] = str(e)
            logger.error(f"数据驱动策略验证测试失败: {e}")
        
        test_result['execution_time'] = time.time() - start_time
        return test_result
    
    def run_enhanced_validation_tests(self) -> Dict[str, Any]:
        """运行增强验证测试"""
        logger.info("🚀 开始增强策略选股验证测试")
        
        # 初始化组件
        if not self.initialize_components():
            self.test_results['overall_status'] = 'INITIALIZATION_FAILED'
            return self.test_results
        
        # 执行各项测试
        test_cases = [
            ('strategy_effectiveness_validation', self.test_strategy_effectiveness_validation),
            ('data_driven_strategy_validation', self.test_data_driven_strategy_validation)
        ]
        
        successful_tests = 0
        total_tests = len(test_cases)
        
        for test_name, test_method in test_cases:
            try:
                logger.info(f"执行测试: {test_name}")
                result = test_method()
                self.test_results['test_cases'][test_name] = result
                
                if result['status'] in ['SUCCESS', 'PARTIAL_SUCCESS']:
                    successful_tests += 1
                    
            except Exception as e:
                logger.error(f"测试 {test_name} 执行异常: {e}")
                self.test_results['test_cases'][test_name] = {
                    'test_name': test_name,
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # 计算整体指标
        success_rate = (successful_tests / total_tests) * 100
        self.test_results['overall_metrics'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': f"{success_rate:.1f}%",
            'completion_rate': f"{(len(self.test_results['test_cases'])/total_tests*100):.1f}%"
        }
        
        # 判断整体状态
        if success_rate >= 90:
            self.test_results['overall_status'] = 'SUCCESS'
        elif success_rate >= 70:
            self.test_results['overall_status'] = 'PARTIAL_SUCCESS'
        else:
            self.test_results['overall_status'] = 'FAILED'
        
        # 生成建议
        self._generate_recommendations()
        
        logger.info(f"增强验证测试完成，成功率: {success_rate:.1f}%")
        return self.test_results
    
    def _generate_recommendations(self):
        """生成改进建议"""
        recommendations = []
        
        # 分析测试结果并生成建议
        for test_name, test_result in self.test_results['test_cases'].items():
            if test_result.get('status') == 'FAILED':
                if 'empty_result_strategies' in test_result:
                    empty_count = test_result['empty_result_strategies']
                    if empty_count > 0:
                        recommendations.append(f"发现{empty_count}个策略返回空结果，建议检查策略条件和数据质量")
                
                if 'failed_strategies' in test_result:
                    failed_count = test_result['failed_strategies']
                    if failed_count > 0:
                        recommendations.append(f"发现{failed_count}个策略执行失败，建议检查策略实现和系统配置")
        
        # 通用建议
        recommendations.extend([
            "建议使用已知有效的历史数据进行策略测试",
            "建议为每个策略设置合理的预期结果范围",
            "建议增加策略执行过程的详细日志记录",
            "建议实现策略有效性的预检查机制"
        ])
        
        self.test_results['recommendations'] = recommendations

def main():
    """主函数"""
    tester = EnhancedStrategyValidationTester()
    results = tester.run_enhanced_validation_tests()
    
    # 保存测试结果
    os.makedirs('results/enhanced', exist_ok=True)
    output_file = 'results/enhanced/enhanced_strategy_validation_test_results.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 输出测试摘要
    print(f"\n📊 增强策略选股验证测试结果:")
    print(f"   整体状态: {results['overall_status']}")
    print(f"   成功率: {results['overall_metrics']['success_rate']}")
    print(f"   完成率: {results['overall_metrics']['completion_rate']}")
    
    if results.get('recommendations'):
        print(f"\n💡 改进建议:")
        for rec in results['recommendations'][:3]:
            print(f"   • {rec}")
    
    print(f"\n📄 详细结果已保存到: {output_file}")
    
    return 0 if results['overall_status'] in ['SUCCESS', 'PARTIAL_SUCCESS'] else 1

if __name__ == "__main__":
    sys.exit(main())
