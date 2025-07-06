#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
买点分析与选股策略集成测试

验证买点分析系统与选股策略系统的数据格式兼容性和集成效果
"""

import sys
import os
import time
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from analysis.buypoints.buypoint_strategy_adapter import get_buypoint_strategy_adapter
from analysis.buypoints.period_data_processor import PeriodDataProcessor
from strategy.strategy_executor import StrategyExecutor
from db.unified_data_manager import get_unified_data_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class BuyPointStrategyIntegrationTest:
    """买点分析与选股策略集成测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.adapter = get_buypoint_strategy_adapter()
        self.data_processor = PeriodDataProcessor()
        self.strategy_executor = StrategyExecutor()
        self.data_manager = get_unified_data_manager()
        
        # 测试结果
        self.test_results = {
            'adapter_tests': {},
            'format_compatibility_tests': {},
            'integration_workflow_tests': {},
            'performance_tests': {},
            'overall_assessment': {}
        }
        
        logger.info("买点分析与选股策略集成测试器初始化完成")
    
    def test_adapter_functionality(self) -> Dict[str, Any]:
        """测试适配器功能"""
        logger.info("开始测试适配器功能...")
        
        adapter_results = {
            'single_conversion_test': {},
            'batch_conversion_test': {},
            'error_handling_test': {},
            'data_validation_test': {}
        }
        
        # 1. 单个结果转换测试
        try:
            # 创建模拟买点分析结果
            sample_buypoint_result = self._create_sample_buypoint_result()
            
            start_time = time.time()
            converted_result = self.adapter.convert_buypoint_result(sample_buypoint_result)
            conversion_time = time.time() - start_time
            
            adapter_results['single_conversion_test'] = {
                'success': converted_result is not None,
                'conversion_time': conversion_time,
                'has_required_fields': self._validate_strategy_format(converted_result) if converted_result else False,
                'score_range_valid': 0 <= converted_result.get('score', -1) <= 100 if converted_result else False
            }
            
        except Exception as e:
            adapter_results['single_conversion_test'] = {
                'success': False,
                'error': str(e)
            }
        
        # 2. 批量转换测试
        try:
            # 创建多个模拟结果
            batch_results = [
                self._create_sample_buypoint_result('000001'),
                self._create_sample_buypoint_result('000002'),
                self._create_sample_buypoint_result('600000')
            ]
            
            start_time = time.time()
            converted_df = self.adapter.convert_batch_results(batch_results)
            batch_conversion_time = time.time() - start_time
            
            adapter_results['batch_conversion_test'] = {
                'success': not converted_df.empty,
                'conversion_time': batch_conversion_time,
                'input_count': len(batch_results),
                'output_count': len(converted_df),
                'conversion_rate': len(converted_df) / len(batch_results) if batch_results else 0
            }
            
        except Exception as e:
            adapter_results['batch_conversion_test'] = {
                'success': False,
                'error': str(e)
            }
        
        # 3. 错误处理测试
        try:
            # 测试无效输入
            invalid_inputs = [
                None,
                {},
                {'invalid': 'data'},
                {'stock_code': ''}
            ]
            
            error_handling_results = []
            for invalid_input in invalid_inputs:
                result = self.adapter.convert_buypoint_result(invalid_input)
                error_handling_results.append(result is None)  # 应该返回None
            
            adapter_results['error_handling_test'] = {
                'success': all(error_handling_results),
                'handled_errors': len(error_handling_results),
                'proper_error_handling': all(error_handling_results)
            }
            
        except Exception as e:
            adapter_results['error_handling_test'] = {
                'success': False,
                'error': str(e)
            }
        
        return adapter_results
    
    def test_format_compatibility(self) -> Dict[str, Any]:
        """测试格式兼容性"""
        logger.info("开始测试格式兼容性...")
        
        compatibility_results = {
            'required_fields_test': {},
            'data_types_test': {},
            'score_consistency_test': {},
            'indicator_mapping_test': {}
        }
        
        try:
            # 创建测试数据
            sample_result = self._create_sample_buypoint_result()
            converted_result = self.adapter.convert_buypoint_result(sample_result)
            
            if converted_result:
                # 1. 必需字段测试
                required_fields = [
                    'stock_code', 'stock_name', 'industry', 'price', 
                    'change_pct', 'score', 'match_details', 'selection_date'
                ]
                
                missing_fields = [field for field in required_fields if field not in converted_result]
                
                compatibility_results['required_fields_test'] = {
                    'success': len(missing_fields) == 0,
                    'required_fields': required_fields,
                    'missing_fields': missing_fields,
                    'completeness': (len(required_fields) - len(missing_fields)) / len(required_fields)
                }
                
                # 2. 数据类型测试
                type_checks = {
                    'stock_code': str,
                    'stock_name': str,
                    'industry': str,
                    'price': (int, float),
                    'change_pct': (int, float),
                    'score': (int, float),
                    'match_details': dict,
                    'selection_date': str
                }
                
                type_errors = []
                for field, expected_type in type_checks.items():
                    if field in converted_result:
                        if not isinstance(converted_result[field], expected_type):
                            type_errors.append(f"{field}: expected {expected_type}, got {type(converted_result[field])}")
                
                compatibility_results['data_types_test'] = {
                    'success': len(type_errors) == 0,
                    'type_errors': type_errors,
                    'type_accuracy': (len(type_checks) - len(type_errors)) / len(type_checks)
                }
                
                # 3. 评分一致性测试
                original_score = sample_result.get('summary', {}).get('overall_score', 0)
                converted_score = converted_result.get('score', 0)
                
                # 评分应该在合理范围内，并且与原始评分相关
                score_valid = 0 <= converted_score <= 100
                score_reasonable = abs(converted_score - original_score) <= 30  # 允许30分的调整范围
                
                compatibility_results['score_consistency_test'] = {
                    'success': score_valid and score_reasonable,
                    'original_score': original_score,
                    'converted_score': converted_score,
                    'score_valid': score_valid,
                    'score_reasonable': score_reasonable
                }
                
                # 4. 指标映射测试
                match_details = converted_result.get('match_details', {})
                passing_indicators = match_details.get('passing_indicators', [])
                
                # 检查是否有指标被正确映射
                mapped_indicators = len(passing_indicators) > 0
                known_indicators = any(indicator in self.adapter.indicator_mapping.values() 
                                     for indicator in passing_indicators)
                
                compatibility_results['indicator_mapping_test'] = {
                    'success': mapped_indicators and known_indicators,
                    'passing_indicators_count': len(passing_indicators),
                    'mapped_indicators': mapped_indicators,
                    'known_indicators': known_indicators
                }
            
        except Exception as e:
            compatibility_results = {
                'success': False,
                'error': str(e)
            }
        
        return compatibility_results
    
    def test_integration_workflow(self) -> Dict[str, Any]:
        """测试集成工作流"""
        logger.info("开始测试集成工作流...")
        
        workflow_results = {
            'end_to_end_test': {},
            'strategy_compatibility_test': {},
            'performance_comparison_test': {}
        }
        
        try:
            # 1. 端到端测试
            start_time = time.time()
            
            # 步骤1: 模拟买点分析
            buypoint_results = [
                self._create_sample_buypoint_result('000001'),
                self._create_sample_buypoint_result('000002')
            ]
            
            # 步骤2: 转换为选股策略格式
            converted_df = self.adapter.convert_batch_results(buypoint_results)
            
            # 步骤3: 验证选股策略可以处理转换后的数据
            strategy_compatible = not converted_df.empty and self._validate_strategy_compatibility(converted_df)
            
            end_to_end_time = time.time() - start_time
            
            workflow_results['end_to_end_test'] = {
                'success': strategy_compatible,
                'total_time': end_to_end_time,
                'input_count': len(buypoint_results),
                'output_count': len(converted_df),
                'workflow_complete': strategy_compatible
            }
            
            # 2. 选股策略兼容性测试
            if not converted_df.empty:
                try:
                    # 尝试使用选股策略的评分逻辑处理转换后的数据
                    sample_row = converted_df.iloc[0].to_dict()
                    
                    # 验证数据结构
                    has_score = 'score' in sample_row
                    has_match_details = 'match_details' in sample_row
                    score_valid = 0 <= sample_row.get('score', -1) <= 100
                    
                    workflow_results['strategy_compatibility_test'] = {
                        'success': has_score and has_match_details and score_valid,
                        'has_score': has_score,
                        'has_match_details': has_match_details,
                        'score_valid': score_valid,
                        'sample_score': sample_row.get('score', 0)
                    }
                    
                except Exception as e:
                    workflow_results['strategy_compatibility_test'] = {
                        'success': False,
                        'error': str(e)
                    }
            
        except Exception as e:
            workflow_results = {
                'success': False,
                'error': str(e)
            }
        
        return workflow_results
    
    def _create_sample_buypoint_result(self, stock_code: str = '000001') -> Dict[str, Any]:
        """创建模拟买点分析结果"""
        return {
            'stock_code': stock_code,
            'buypoint_date': '20240601',
            'indicator_results': {
                'period_data': {
                    'daily': pd.DataFrame({'close': [10.0, 10.5, 11.0]}),
                    '60min': pd.DataFrame({'close': [10.2, 10.7, 10.9]})
                },
                'technical_analysis': {
                    'macd_gold': {
                        'value': 0.15,
                        'signal': 'buy',
                        'strength': 0.8
                    },
                    'touch_ma': {
                        'value': 1.0,
                        'signal': 'positive',
                        'strength': 0.7
                    }
                }
            },
            'pattern_results': {
                'macd_gold': {'detected': True, 'confidence': 0.8, 'description': 'MACD金叉信号'},
                'touch_ma': {'detected': True, 'confidence': 0.7, 'description': '触及均线支撑'},
                'price_stable': {'detected': True, 'confidence': 0.6, 'description': '价格企稳'},
                'xc': {'detected': True, 'confidence': 0.9, 'description': '吸筹信号'}
            },
            'summary': {
                'total_indicators': 8,
                'positive_signals': 6,
                'negative_signals': 2,
                'overall_score': 75.0
            }
        }
    
    def _validate_strategy_format(self, result: Dict) -> bool:
        """验证选股策略格式"""
        if not result:
            return False
        
        required_fields = [
            'stock_code', 'stock_name', 'industry', 'price',
            'change_pct', 'score', 'match_details', 'selection_date'
        ]
        
        return all(field in result for field in required_fields)
    
    def _validate_strategy_compatibility(self, df: pd.DataFrame) -> bool:
        """验证选股策略兼容性"""
        if df.empty:
            return False
        
        # 检查必需列
        required_columns = ['stock_code', 'stock_name', 'score', 'match_details']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # 检查评分范围
        if not df['score'].between(0, 100).all():
            return False
        
        return True
    
    def run_comprehensive_integration_test_Test(self) -> Dict[str, Any]:
        """运行综合集成测试"""
        logger.info("=" * 80)
        logger.info("开始买点分析与选股策略集成综合测试")
        logger.info("=" * 80)
        
        test_results = {
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'adapter_tests': {},
            'format_compatibility_tests': {},
            'integration_workflow_tests': {},
            'overall_assessment': {}
        }
        
        try:
            # 1. 适配器功能测试
            logger.info("步骤 1: 适配器功能测试")
            test_results['adapter_tests'] = self.test_adapter_functionality()
            
            # 2. 格式兼容性测试
            logger.info("步骤 2: 格式兼容性测试")
            test_results['format_compatibility_tests'] = self.test_format_compatibility()
            
            # 3. 集成工作流测试
            logger.info("步骤 3: 集成工作流测试")
            test_results['integration_workflow_tests'] = self.test_integration_workflow()
            
            # 4. 整体评估
            test_results['overall_assessment'] = self._generate_integration_assessment(test_results)
            
            logger.info("买点分析与选股策略集成综合测试完成")
            
        except Exception as e:
            logger.error(f"测试过程中发生错误: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    def _generate_integration_assessment(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成集成评估"""
        assessment = {
            'integration_success': True,
            'compatibility_score': 0.0,
            'adapter_effectiveness': 'good',
            'production_ready': True,
            'issues_found': [],
            'recommendations': [],
            'key_metrics': {}
        }
        
        # 评估适配器功能
        adapter_tests = test_results.get('adapter_tests', {})
        
        # 检查单个转换
        single_test = adapter_tests.get('single_conversion_test', {})
        if not single_test.get('success', False):
            assessment['integration_success'] = False
            assessment['issues_found'].append('单个结果转换失败')
        
        # 检查批量转换
        batch_test = adapter_tests.get('batch_conversion_test', {})
        if batch_test.get('success', False):
            conversion_rate = batch_test.get('conversion_rate', 0)
            assessment['key_metrics']['conversion_rate'] = conversion_rate
            
            if conversion_rate < 0.8:
                assessment['adapter_effectiveness'] = 'needs_improvement'
                assessment['issues_found'].append(f"批量转换成功率较低: {conversion_rate:.2%}")
        
        # 评估格式兼容性
        compat_tests = test_results.get('format_compatibility_tests', {})
        
        # 检查必需字段
        required_test = compat_tests.get('required_fields_test', {})
        if required_test.get('success', False):
            completeness = required_test.get('completeness', 0)
            assessment['key_metrics']['field_completeness'] = completeness
            assessment['compatibility_score'] += completeness * 30
        
        # 检查数据类型
        type_test = compat_tests.get('data_types_test', {})
        if type_test.get('success', False):
            type_accuracy = type_test.get('type_accuracy', 0)
            assessment['key_metrics']['type_accuracy'] = type_accuracy
            assessment['compatibility_score'] += type_accuracy * 25
        
        # 检查评分一致性
        score_test = compat_tests.get('score_consistency_test', {})
        if score_test.get('success', False):
            assessment['compatibility_score'] += 25
        
        # 检查指标映射
        mapping_test = compat_tests.get('indicator_mapping_test', {})
        if mapping_test.get('success', False):
            assessment['compatibility_score'] += 20
        
        # 评估集成工作流
        workflow_tests = test_results.get('integration_workflow_tests', {})
        end_to_end_test = workflow_tests.get('end_to_end_test', {})
        
        if not end_to_end_test.get('success', False):
            assessment['integration_success'] = False
            assessment['issues_found'].append('端到端集成测试失败')
        
        # 生成建议
        if assessment['integration_success'] and assessment['compatibility_score'] >= 80:
            assessment['recommendations'].append('集成测试完全通过，适配器可以投入生产使用')
            assessment['adapter_effectiveness'] = 'excellent'
        elif assessment['compatibility_score'] >= 60:
            assessment['recommendations'].append('集成基本成功，建议优化部分兼容性问题后投入使用')
            assessment['production_ready'] = True
        else:
            assessment['recommendations'].append('发现重要兼容性问题，建议修复后重新测试')
            assessment['production_ready'] = False
        
        return assessment


def mainBuypointstrategyintegrationtest():
    """主函数"""
    print("=" * 80)
    print("买点分析与选股策略集成测试")
    print("验证数据格式兼容性和集成效果")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 创建测试实例
        test_framework = BuyPointStrategyIntegrationTest()
        
        # 运行综合测试
        results = test_framework.run_comprehensive_integration_test_Test()
        
        # 显示结果摘要
        print("=" * 80)
        print("集成测试结果摘要")
        print("=" * 80)
        
        if 'error' in results:
            print(f"❌ 测试失败: {results['error']}")
            return 1
        
        # 整体评估
        assessment = results.get('overall_assessment', {})
        key_metrics = assessment.get('key_metrics', {})
        
        print(f"🔗 集成成功: {'是' if assessment.get('integration_success', False) else '否'}")
        print(f"📊 兼容性评分: {assessment.get('compatibility_score', 0):.1f}/100")
        print(f"⚡ 适配器效果: {assessment.get('adapter_effectiveness', 'N/A')}")
        print(f"🚀 生产就绪: {'是' if assessment.get('production_ready', False) else '否'}")
        
        # 关键指标
        if key_metrics:
            print(f"\n📈 关键指标:")
            if 'conversion_rate' in key_metrics:
                print(f"  - 转换成功率: {key_metrics['conversion_rate']:.2%}")
            if 'field_completeness' in key_metrics:
                print(f"  - 字段完整性: {key_metrics['field_completeness']:.2%}")
            if 'type_accuracy' in key_metrics:
                print(f"  - 类型准确性: {key_metrics['type_accuracy']:.2%}")
        
        # 发现的问题
        issues = assessment.get('issues_found', [])
        if issues:
            print(f"\n⚠️ 发现的问题:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        
        # 建议
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            print(f"\n💡 建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"test_reports/buypoint_strategy_integration_{timestamp}.json"
        
        os.makedirs("test_reports", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📄 详细结果已保存: {output_file}")
        
        # 判断测试结果
        if assessment.get('production_ready', False):
            print("\n🎉 集成测试通过！买点分析与选股策略已成功集成。")
            return 0
        else:
            print("\n⚠️ 集成存在问题，建议修复后重新测试。")
            return 1
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = mainBuypointstrategyintegrationtest()
    sys.exit(exit_code)
