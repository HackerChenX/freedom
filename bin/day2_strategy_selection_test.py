#!/usr/bin/env python3
"""
Day 2 策略选股生产脚本验证测试
按照生产级测试计划执行多策略选股和市场扫描功能验证
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from strategy.configurable_strategy_engine import ConfigurableStrategyEngine
from strategy.strategy_executor import StrategyExecutor
from strategy.strategy_manager import StrategyManager
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

class Day2StrategySelectionTester:
    """Day 2 策略选股测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.strategy_engine = None
        self.strategy_executor = None
        self.strategy_manager = None
        self.test_results = {
            'test_date': datetime.now().strftime('%Y-%m-%d'),
            'test_phase': 'Day 2: 策略选股生产脚本验证',
            'test_cases': {},
            'overall_metrics': {},
            'issues': [],
            'recommendations': []
        }
        
    @exception_handler(reraise=True)
    def initialize_components(self):
        """初始化策略组件"""
        logger.info("初始化策略选股组件...")
        
        try:
            # 初始化策略引擎
            self.strategy_engine = ConfigurableStrategyEngine()
            logger.info("✅ 可配置策略引擎初始化完成")
            
            # 初始化策略执行器
            self.strategy_executor = StrategyExecutor()
            logger.info("✅ 策略执行器初始化完成")
            
            # 初始化策略管理器
            self.strategy_manager = StrategyManager()
            logger.info("✅ 策略管理器初始化完成")
            
            return True
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            self.test_results['issues'].append(f"组件初始化失败: {e}")
            return False
    
    @performance_monitor(threshold_seconds=30.0)
    def test_multi_strategy_selection(self) -> Dict[str, Any]:
        """测试多策略选股功能"""
        logger.info("=== 测试多策略选股功能 ===")
        
        test_result = {
            'test_name': '多策略选股测试',
            'status': 'UNKNOWN',
            'execution_time': 0,
            'strategies_tested': 0,
            'successful_strategies': 0,
            'total_selections': 0,
            'details': []
        }
        
        start_time = time.time()
        
        try:
            # 获取可用策略
            available_strategies = self.strategy_engine.get_available_strategies()
            logger.info(f"发现 {len(available_strategies)} 个可用策略")
            
            # 测试股票池
            test_stocks = ['000001', '000002', '600519', '300005', '603359']
            target_date = '2025-05-09'
            
            successful_count = 0
            total_selections = 0
            
            # 测试前5个策略
            strategy_list = list(available_strategies.keys())[:5]
            test_result['strategies_tested'] = len(strategy_list)
            
            for strategy_id in strategy_list:
                try:
                    logger.info(f"测试策略: {strategy_id}")
                    
                    # 执行策略选股
                    selections = self.strategy_engine.execute_strategy(
                        strategy_id, test_stocks, target_date
                    )

                    # 修正测试判定标准：空结果不应该判定为成功
                    selections_count = len(selections) if selections else 0

                    if selections_count > 0:
                        strategy_detail = {
                            'strategy_id': strategy_id,
                            'status': 'SUCCESS',
                            'selections_count': selections_count,
                            'selections': selections[:3] if selections else [],
                            'validation': 'PASSED - 策略成功选出股票'
                        }
                        successful_count += 1
                        logger.info(f"  ✅ 策略 {strategy_id} 选出 {selections_count} 只股票")
                    else:
                        strategy_detail = {
                            'strategy_id': strategy_id,
                            'status': 'FAILED',
                            'selections_count': 0,
                            'selections': [],
                            'validation': 'FAILED - 策略未选出任何股票',
                            'failure_reason': '策略执行结果为空，可能原因：数据不足、策略条件过严、算法错误'
                        }
                        logger.warning(f"  ❌ 策略 {strategy_id} 未选出任何股票")

                    total_selections += selections_count
                    
                except Exception as e:
                    strategy_detail = {
                        'strategy_id': strategy_id,
                        'status': 'FAILED',
                        'error': str(e),
                        'selections_count': 0
                    }
                    logger.warning(f"  ❌ 策略 {strategy_id} 执行失败: {e}")
                
                test_result['details'].append(strategy_detail)
            
            test_result['successful_strategies'] = successful_count
            test_result['total_selections'] = total_selections
            test_result['success_rate'] = f"{(successful_count/len(strategy_list)*100):.1f}%"
            
            # 判断测试状态
            if successful_count >= len(strategy_list) * 0.8:  # 80%成功率
                test_result['status'] = 'SUCCESS'
            elif successful_count > 0:
                test_result['status'] = 'PARTIAL_SUCCESS'
            else:
                test_result['status'] = 'FAILED'
                
        except Exception as e:
            test_result['status'] = 'FAILED'
            test_result['error'] = str(e)
            logger.error(f"多策略选股测试失败: {e}")
        
        test_result['execution_time'] = time.time() - start_time
        return test_result
    
    @performance_monitor(threshold_seconds=60.0)
    def test_market_scanning(self) -> Dict[str, Any]:
        """测试市场扫描功能"""
        logger.info("=== 测试市场扫描功能 ===")
        
        test_result = {
            'test_name': '市场扫描验证',
            'status': 'UNKNOWN',
            'execution_time': 0,
            'scanned_stocks': 0,
            'scan_success': False,
            'performance_metrics': {}
        }
        
        start_time = time.time()
        
        try:
            # 模拟市场扫描（使用更大的股票池）
            market_stocks = [
                '000001', '000002', '000858', '600519', '600036',
                '300005', '300003', '603359', '002415', '000858'
            ]
            
            # 创建扫描策略配置（修正格式以匹配策略执行器期望）
            scan_strategy_plan = {
                'strategy_id': 'MARKET_SCAN_TEST',
                'name': '市场扫描测试策略',
                'conditions': [
                    {
                        'type': 'price',
                        'field': 'close',
                        'operator': '>',
                        'value': 5
                    }
                ],
                'filters': {
                    'market': ['主板', '创业板']
                },
                'parameters': {
                    'target_date': '2025-05-09',
                    'max_results': 20
                }
            }

            # 执行市场扫描
            logger.info(f"开始扫描 {len(market_stocks)} 只股票...")

            scan_results = self.strategy_executor.execute_strategy(
                strategy_plan=scan_strategy_plan,
                end_date='2025-05-09'
            )
            
            # 修正测试判定标准：必须有实际扫描结果才能判定为成功
            scan_results_count = len(scan_results) if hasattr(scan_results, '__len__') else 0

            test_result['scanned_stocks'] = len(market_stocks)
            test_result['scan_results_count'] = scan_results_count

            if scan_results_count > 0:
                test_result['scan_success'] = True
                test_result['status'] = 'SUCCESS'
                test_result['validation'] = 'PASSED - 市场扫描成功发现股票'
                logger.info(f"✅ 市场扫描完成，扫描 {len(market_stocks)} 只股票，发现 {scan_results_count} 个结果")
            else:
                test_result['scan_success'] = False
                test_result['status'] = 'FAILED'
                test_result['validation'] = 'FAILED - 市场扫描未发现任何股票'
                test_result['failure_reason'] = '市场扫描结果为空，可能原因：数据库连接问题、扫描条件过严、数据不足'
                logger.warning(f"❌ 市场扫描完成，扫描 {len(market_stocks)} 只股票，但未发现任何结果")
            
        except Exception as e:
            test_result['status'] = 'FAILED'
            test_result['error'] = str(e)
            test_result['scan_success'] = False
            logger.error(f"市场扫描测试失败: {e}")
        
        test_result['execution_time'] = time.time() - start_time
        return test_result
    
    @performance_monitor(threshold_seconds=45.0)
    def test_strategy_combination(self) -> Dict[str, Any]:
        """测试策略组合效果"""
        logger.info("=== 测试策略组合效果 ===")
        
        test_result = {
            'test_name': '策略组合测试',
            'status': 'UNKNOWN',
            'execution_time': 0,
            'combinations_tested': 0,
            'successful_combinations': 0,
            'details': []
        }
        
        start_time = time.time()
        
        try:
            # 获取策略组合
            combinations = self.strategy_engine.get_strategy_combinations()
            logger.info(f"发现 {len(combinations)} 个策略组合")
            
            test_stocks = ['000001', '600519', '300005']
            target_date = '2025-05-09'
            
            successful_count = 0
            
            # 测试前3个组合
            combination_list = list(combinations.keys())[:3]
            test_result['combinations_tested'] = len(combination_list)
            
            for combination_id in combination_list:
                try:
                    logger.info(f"测试组合: {combination_id}")
                    
                    # 执行组合策略
                    combo_results = self.strategy_engine.execute_combination(
                        combination_id, test_stocks, target_date
                    )

                    # 修正测试判定标准：空结果不应该判定为成功
                    results_count = len(combo_results) if combo_results else 0

                    if results_count > 0:
                        combination_detail = {
                            'combination_id': combination_id,
                            'status': 'SUCCESS',
                            'results_count': results_count,
                            'sample_results': combo_results[:2] if combo_results else [],
                            'validation': 'PASSED - 组合策略成功产生结果'
                        }
                        successful_count += 1
                        logger.info(f"  ✅ 组合 {combination_id} 执行成功，产生 {results_count} 个结果")
                    else:
                        combination_detail = {
                            'combination_id': combination_id,
                            'status': 'FAILED',
                            'results_count': 0,
                            'sample_results': [],
                            'validation': 'FAILED - 组合策略未产生任何结果',
                            'failure_reason': '组合策略执行结果为空，可能原因：数据不足、策略配置错误、组合逻辑问题'
                        }
                        logger.warning(f"  ❌ 组合 {combination_id} 未产生任何结果")
                    
                except Exception as e:
                    combination_detail = {
                        'combination_id': combination_id,
                        'status': 'FAILED',
                        'error': str(e)
                    }
                    logger.warning(f"  ❌ 组合 {combination_id} 执行失败: {e}")
                
                test_result['details'].append(combination_detail)
            
            test_result['successful_combinations'] = successful_count
            test_result['success_rate'] = f"{(successful_count/len(combination_list)*100):.1f}%"
            
            # 判断测试状态
            if successful_count >= len(combination_list) * 0.8:
                test_result['status'] = 'SUCCESS'
            elif successful_count > 0:
                test_result['status'] = 'PARTIAL_SUCCESS'
            else:
                test_result['status'] = 'FAILED'
                
        except Exception as e:
            test_result['status'] = 'FAILED'
            test_result['error'] = str(e)
            logger.error(f"策略组合测试失败: {e}")
        
        test_result['execution_time'] = time.time() - start_time
        return test_result
    
    def run_day2_tests(self) -> Dict[str, Any]:
        """运行Day 2完整测试"""
        logger.info("🚀 开始Day 2策略选股生产脚本验证测试")
        
        # 初始化组件
        if not self.initialize_components():
            self.test_results['overall_status'] = 'INITIALIZATION_FAILED'
            return self.test_results
        
        # 执行各项测试
        test_cases = [
            ('multi_strategy_selection', self.test_multi_strategy_selection),
            ('market_scanning', self.test_market_scanning),
            ('strategy_combination', self.test_strategy_combination)
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
        
        logger.info(f"Day 2测试完成，成功率: {success_rate:.1f}%")
        return self.test_results

def main():
    """主函数"""
    tester = Day2StrategySelectionTester()
    results = tester.run_day2_tests()
    
    # 保存测试结果
    os.makedirs('results/day2', exist_ok=True)
    output_file = 'results/day2/day2_strategy_selection_test_results.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 输出测试摘要
    print(f"\n📊 Day 2 策略选股测试结果:")
    print(f"   整体状态: {results['overall_status']}")
    print(f"   成功率: {results['overall_metrics']['success_rate']}")
    print(f"   完成率: {results['overall_metrics']['completion_rate']}")
    print(f"\n📄 详细结果已保存到: {output_file}")
    
    return 0 if results['overall_status'] in ['SUCCESS', 'PARTIAL_SUCCESS'] else 1

if __name__ == "__main__":
    sys.exit(main())
