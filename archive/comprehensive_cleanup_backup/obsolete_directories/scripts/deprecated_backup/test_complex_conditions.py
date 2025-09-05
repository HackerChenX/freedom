#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
复杂条件组合测试脚本

测试策略条件逻辑增强功能
验证AND、OR、NOT操作符和嵌套表达式的正确性
"""

import os
import sys
import json
from typing import Dict, Any

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import getLogger
from strategy.condition_parser import ConditionParser
from strategy.strategy_executor import UnifiedStrategyExecutor

logger = getLogger(__name__)


class ComplexConditionTester:
    """复杂条件测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.condition_parser = ConditionParser()
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        logger.info("复杂条件测试器初始化完成")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("=" * 60)
        logger.info("复杂条件组合逻辑测试")
        logger.info("=" * 60)
        
        try:
            # 测试1：简单条件数组（向后兼容）
            self._test_simple_conditions()
            
            # 测试2：AND逻辑组合
            self._test_and_logic()
            
            # 测试3：OR逻辑组合
            self._test_or_logic()
            
            # 测试4：NOT逻辑组合
            self._test_not_logic()
            
            # 测试5：嵌套逻辑组合
            self._test_nested_logic()
            
            # 测试6：表达式语法
            self._test_expression_syntax()
            
            # 测试7：权重计算
            self._test_weighted_conditions()
            
            # 测试8：集成测试
            self._test_integration()
            
            # 生成测试报告
            self._generate_test_report()
            
            return self.test_results
            
        except Exception as e:
            logger.error(f"复杂条件测试失败: {e}")
            self.test_results['error'] = str(e)
            return self.test_results
    
    def _run_test(self, test_name: str, test_function):
        """运行单个测试"""
        logger.info(f"\n🧪 测试: {test_name}")
        
        self.test_results['total_tests'] += 1
        
        try:
            result = test_function()
            self.test_results['passed_tests'] += 1
            self.test_results['test_details'].append({
                'test_name': test_name,
                'status': 'passed',
                'result': result
            })
            logger.info(f"✅ {test_name} - 通过")
            
        except Exception as e:
            self.test_results['failed_tests'] += 1
            self.test_results['test_details'].append({
                'test_name': test_name,
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"❌ {test_name} - 失败: {e}")
    
    def _test_simple_conditions(self):
        """测试简单条件数组（向后兼容）"""
        self._run_test("简单条件数组", self._simple_conditions_test)
    
    def _simple_conditions_test(self) -> Dict[str, Any]:
        """简单条件数组测试"""
        conditions = [
            {
                "id": "TEST_1",
                "field": "macd",
                "operator": ">",
                "value": 0,
                "weight": 1.0
            },
            {
                "id": "TEST_2", 
                "field": "rsi",
                "operator": "between",
                "value": [30, 70],
                "weight": 1.0
            }
        ]
        
        indicator_data = {
            "macd": 0.15,
            "rsi": 45
        }
        
        result = self.condition_parser.evaluate_conditions(conditions, indicator_data)
        
        assert result['success'] == True
        assert result['result'] == True
        assert result['score'] == 1.0
        assert result['logic_type'] == 'simple_array'
        
        return {
            'conditions_count': len(conditions),
            'result': result['result'],
            'score': result['score']
        }
    
    def _test_and_logic(self):
        """测试AND逻辑组合"""
        self._run_test("AND逻辑组合", self._and_logic_test)
    
    def _and_logic_test(self) -> Dict[str, Any]:
        """AND逻辑组合测试"""
        conditions = {
            "logic": "AND",
            "conditions": [
                {
                    "id": "MACD_POSITIVE",
                    "field": "macd",
                    "operator": ">",
                    "value": 0,
                    "weight": 1.0
                },
                {
                    "id": "RSI_NORMAL",
                    "field": "rsi",
                    "operator": "between",
                    "value": [30, 70],
                    "weight": 1.0
                }
            ]
        }
        
        # 测试所有条件满足的情况
        indicator_data_pass = {"macd": 0.15, "rsi": 45}
        result_pass = self.condition_parser.evaluate_conditions(conditions, indicator_data_pass)
        
        # 测试部分条件不满足的情况
        indicator_data_fail = {"macd": -0.05, "rsi": 45}
        result_fail = self.condition_parser.evaluate_conditions(conditions, indicator_data_fail)
        
        assert result_pass['success'] == True
        assert result_pass['result'] == True
        assert result_pass['logic'] == 'AND'
        
        assert result_fail['success'] == True
        assert result_fail['result'] == False
        assert result_fail['logic'] == 'AND'
        
        return {
            'pass_case': result_pass['result'],
            'fail_case': result_fail['result'],
            'logic_type': result_pass.get('logic_type', 'unknown')
        }
    
    def _test_or_logic(self):
        """测试OR逻辑组合"""
        self._run_test("OR逻辑组合", self._or_logic_test)
    
    def _or_logic_test(self) -> Dict[str, Any]:
        """OR逻辑组合测试"""
        conditions = {
            "logic": "OR",
            "conditions": [
                {
                    "id": "VOLUME_HIGH",
                    "field": "volume_ratio",
                    "operator": ">",
                    "value": 1.5,
                    "weight": 1.0
                },
                {
                    "id": "PRICE_UP",
                    "field": "price_change",
                    "operator": ">",
                    "value": 3,
                    "weight": 1.0
                }
            ]
        }
        
        # 测试第一个条件满足
        indicator_data_1 = {"volume_ratio": 2.0, "price_change": 1.0}
        result_1 = self.condition_parser.evaluate_conditions(conditions, indicator_data_1)
        
        # 测试第二个条件满足
        indicator_data_2 = {"volume_ratio": 1.0, "price_change": 5.0}
        result_2 = self.condition_parser.evaluate_conditions(conditions, indicator_data_2)
        
        # 测试都不满足
        indicator_data_none = {"volume_ratio": 1.0, "price_change": 1.0}
        result_none = self.condition_parser.evaluate_conditions(conditions, indicator_data_none)
        
        assert result_1['result'] == True
        assert result_2['result'] == True
        assert result_none['result'] == False
        
        return {
            'first_condition_pass': result_1['result'],
            'second_condition_pass': result_2['result'],
            'none_pass': result_none['result']
        }
    
    def _test_not_logic(self):
        """测试NOT逻辑组合"""
        self._run_test("NOT逻辑组合", self._not_logic_test)
    
    def _not_logic_test(self) -> Dict[str, Any]:
        """NOT逻辑组合测试"""
        conditions = {
            "logic": "NOT",
            "conditions": [
                {
                    "id": "OVERBOUGHT",
                    "field": "rsi",
                    "operator": ">",
                    "value": 80,
                    "weight": 1.0
                }
            ]
        }
        
        # 测试条件为真时（NOT应该返回False）
        indicator_data_true = {"rsi": 85}
        result_true = self.condition_parser.evaluate_conditions(conditions, indicator_data_true)
        
        # 测试条件为假时（NOT应该返回True）
        indicator_data_false = {"rsi": 45}
        result_false = self.condition_parser.evaluate_conditions(conditions, indicator_data_false)
        
        assert result_true['result'] == False  # NOT True = False
        assert result_false['result'] == True  # NOT False = True
        
        return {
            'not_true_case': result_true['result'],
            'not_false_case': result_false['result']
        }
    
    def _test_nested_logic(self):
        """测试嵌套逻辑组合"""
        self._run_test("嵌套逻辑组合", self._nested_logic_test)
    
    def _nested_logic_test(self) -> Dict[str, Any]:
        """嵌套逻辑组合测试"""
        conditions = {
            "logic": "AND",
            "conditions": [
                {
                    "id": "MACD_POSITIVE",
                    "field": "macd",
                    "operator": ">",
                    "value": 0,
                    "weight": 1.0
                },
                {
                    "logic": "OR",
                    "conditions": [
                        {
                            "id": "K_LOW",
                            "field": "k",
                            "operator": "<",
                            "value": 30,
                            "weight": 1.0
                        },
                        {
                            "id": "RSI_LOW",
                            "field": "rsi",
                            "operator": "<",
                            "value": 35,
                            "weight": 1.0
                        }
                    ],
                    "weight": 1.0
                }
            ]
        }
        
        # 测试嵌套条件满足
        indicator_data = {
            "macd": 0.15,
            "k": 25,
            "rsi": 50
        }
        
        result = self.condition_parser.evaluate_conditions(conditions, indicator_data)
        
        assert result['success'] == True
        assert result['result'] == True
        
        return {
            'nested_result': result['result'],
            'nested_score': result.get('score', 0)
        }
    
    def _test_expression_syntax(self):
        """测试表达式语法"""
        self._run_test("表达式语法", self._expression_syntax_test)
    
    def _expression_syntax_test(self) -> Dict[str, Any]:
        """表达式语法测试"""
        conditions = {
            "expression": "(MACD_UP AND RSI_OK) OR VOLUME_HIGH",
            "conditions": [
                {
                    "id": "MACD_UP",
                    "field": "macd",
                    "operator": ">",
                    "value": 0,
                    "weight": 1.0
                },
                {
                    "id": "RSI_OK",
                    "field": "rsi",
                    "operator": "between",
                    "value": [30, 70],
                    "weight": 1.0
                },
                {
                    "id": "VOLUME_HIGH",
                    "field": "volume_ratio",
                    "operator": ">",
                    "value": 1.5,
                    "weight": 1.0
                }
            ]
        }
        
        # 测试第一部分表达式满足 (MACD_UP AND RSI_OK)
        indicator_data_1 = {
            "macd": 0.15,
            "rsi": 45,
            "volume_ratio": 1.0
        }
        
        # 测试第二部分表达式满足 VOLUME_HIGH
        indicator_data_2 = {
            "macd": -0.05,
            "rsi": 45,
            "volume_ratio": 2.0
        }
        
        result_1 = self.condition_parser.evaluate_conditions(conditions, indicator_data_1)
        result_2 = self.condition_parser.evaluate_conditions(conditions, indicator_data_2)
        
        assert result_1['result'] == True
        assert result_2['result'] == True
        assert result_1['logic_type'] == 'expression'
        
        return {
            'expression_1_result': result_1['result'],
            'expression_2_result': result_2['result'],
            'expression': conditions['expression']
        }
    
    def _test_weighted_conditions(self):
        """测试权重计算"""
        self._run_test("权重计算", self._weighted_conditions_test)
    
    def _weighted_conditions_test(self) -> Dict[str, Any]:
        """权重计算测试"""
        conditions = [
            {
                "id": "HIGH_WEIGHT",
                "field": "macd",
                "operator": ">",
                "value": 0,
                "weight": 0.8
            },
            {
                "id": "LOW_WEIGHT",
                "field": "rsi",
                "operator": ">",
                "value": 50,
                "weight": 0.2
            }
        ]
        
        # 只有高权重条件满足
        indicator_data = {
            "macd": 0.15,
            "rsi": 30
        }
        
        result = self.condition_parser.evaluate_conditions(conditions, indicator_data)
        
        # 应该有0.8的权重满足
        expected_score = 0.8 / 1.0  # 0.8权重满足 / 1.0总权重
        
        assert result['success'] == True
        assert abs(result['score'] - expected_score) < 0.01
        
        return {
            'weighted_score': result['score'],
            'expected_score': expected_score,
            'weight_calculation_correct': abs(result['score'] - expected_score) < 0.01
        }
    
    def _test_integration(self):
        """测试集成"""
        self._run_test("集成测试", self._integration_test)
    
    def _integration_test(self) -> Dict[str, Any]:
        """集成测试"""
        # 加载复杂条件示例配置
        config_path = os.path.join(
            root_dir, 'config', 'strategy_templates', 'complex_conditions_example.json'
        )
        
        if not os.path.exists(config_path):
            raise Exception("复杂条件示例配置文件不存在")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            strategy_config = json.load(f)
        
        # 创建统一执行器
        executor = UnifiedStrategyExecutor(
            max_workers=1,
            enable_unified_config=True
        )
        
        # 测试配置解析
        parsed_strategy = executor._parse_unified_strategy_config(strategy_config)
        
        assert 'indicators' in parsed_strategy
        assert len(parsed_strategy['indicators']) > 0
        
        return {
            'config_loaded': True,
            'strategy_parsed': True,
            'indicators_count': len(parsed_strategy['indicators']),
            'strategy_id': parsed_strategy.get('strategy_id')
        }
    
    def _generate_test_report(self):
        """生成测试报告"""
        logger.info("\n" + "=" * 60)
        logger.info("复杂条件组合测试报告")
        logger.info("=" * 60)
        
        total_tests = self.test_results['total_tests']
        passed_tests = self.test_results['passed_tests']
        failed_tests = self.test_results['failed_tests']
        
        logger.info(f"测试总数: {total_tests}")
        logger.info(f"通过测试: {passed_tests}")
        logger.info(f"失败测试: {failed_tests}")
        logger.info(f"成功率: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        
        if failed_tests > 0:
            logger.info("\n失败的测试:")
            for test in self.test_results['test_details']:
                if test['status'] == 'failed':
                    logger.error(f"  ❌ {test['test_name']}: {test.get('error', 'Unknown error')}")
        
        if passed_tests == total_tests:
            logger.info("\n🎉 所有复杂条件测试通过！")
        else:
            logger.warning(f"\n⚠️ {failed_tests} 个测试失败，需要进一步调试")


def main():
    """主函数"""
    logger.info("开始复杂条件组合逻辑测试...")
    
    tester = ComplexConditionTester()
    results = tester.run_all_tests()
    
    # 保存测试结果
    results_file = os.path.join(root_dir, 'test_results', 'complex_conditions_test_results.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n测试结果已保存到: {results_file}")
    
    # 返回状态码
    if results['failed_tests'] == 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
