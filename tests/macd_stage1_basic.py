#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD指标阶段1基础功能验证

基于RSI项目成功经验，验证MACD指标基本计算功能
确保DIF、DEA、MACD三个组件计算正确
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from utils.technical_utils import calculate_macd_Utils
except ImportError as e:
    print(f"导入错误: {e}")

class MACDStage1Validator:
    """MACD阶段1验证器"""
    
    def __init__(self):
        """初始化阶段1验证器"""
        self.validator_name = "MACD阶段1验证器"
        self.macd_indicator = MacdMacd()
        
        # 阶段1验证配置
        self.stage1_config = {
            'basic_functionality_target': 1.0,  # 100%基础功能
            'calculation_accuracy_target': 0.99,  # 99%计算准确性
            'component_completeness_target': 1.0,  # 100%组件完整性
            'error_handling_target': 0.8  # 80%错误处理
        }
        
        print(f"✅ {self.validator_name}初始化完成")
        print(f"🎯 验证MACD指标基础功能")
    
    def create_test_data(self) -> pd.DataFrame:
        """创建测试数据"""
        
        # 创建标准测试数据集
        dates = pd.date_range('2025-01-01', periods=50, freq='D')
        
        # 生成有趋势的价格数据
        base_price = 100
        trend = np.linspace(0, 10, 50)  # 上升趋势
        noise = np.random.normal(0, 1, 50) * 0.5  # 小幅波动
        
        prices = base_price + trend + noise
        
        # 确保价格合理性
        prices = np.maximum(prices, 50)  # 最低价格50
        
        test_data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 50)
        })
        
        return test_data
    
    def test_basic_functionality(self) -> Dict[str, Any]:
        """测试基础功能"""
        
        print(f"\n🔧 测试MACD基础功能")
        print("=" * 60)
        
        functionality_result = {
            'test_type': 'BASIC_FUNCTIONALITY',
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'overall_score': 0.0,
            'status': 'IN_PROGRESS'
        }
        
        try:
            # 创建测试数据
            test_data = self.create_test_data()
            print(f"📊 创建测试数据: {len(test_data)}个数据点")
            
            # 测试1: MACD指标类基础计算
            print(f"\n🔧 测试1: MACD指标类基础计算")
            
            try:
                macd_result = self.macd_indicator._calculate_macd(test_data)
                
                if macd_result is not None and not macd_result.empty:
                    # 检查返回的组件
                    expected_components = ['macd_line', 'macd_signal', 'macd_histogram']
                    found_components = []
                    
                    for component in expected_components:
                        if component in macd_result.columns:
                            found_components.append(component)
                            component_data = macd_result[component].dropna()
                            print(f"    ✅ {component}: {len(component_data)}个有效值")
                        else:
                            print(f"    ❌ {component}: 缺失")
                    
                    completeness_score = len(found_components) / len(expected_components)
                    
                    functionality_result['tests'].append({
                        'test': 'MACD_INDICATOR_CALCULATION',
                        'result': 'PASSED' if completeness_score >= 0.8 else 'FAILED',
                        'score': completeness_score,
                        'found_components': found_components,
                        'expected_components': expected_components
                    })
                    
                    print(f"  📊 组件完整性: {completeness_score:.1%}")
                else:
                    functionality_result['tests'].append({
                        'test': 'MACD_INDICATOR_CALCULATION',
                        'result': 'FAILED',
                        'score': 0.0,
                        'reason': 'NO_RESULT'
                    })
                    print(f"  ❌ MACD指标计算失败")
            
            except Exception as e:
                functionality_result['tests'].append({
                    'test': 'MACD_INDICATOR_CALCULATION',
                    'result': 'ERROR',
                    'score': 0.0,
                    'error': str(e)
                })
                print(f"  ❌ MACD指标计算异常: {e}")
            
            # 测试2: 技术工具函数计算
            print(f"\n🔧 测试2: 技术工具函数计算")
            
            try:
                dif, dea, macd = calculate_macd_Utils(
                    test_data['close'], 
                    fast_period=12, 
                    slow_period=26, 
                    signal_period=9
                )
                
                if dif is not None and not dif.empty:
                    dif_clean = dif.dropna()
                    dea_clean = dea.dropna()
                    macd_clean = macd.dropna()
                    
                    # 检查数据质量
                    data_quality_score = 1.0
                    
                    if len(dif_clean) == 0:
                        data_quality_score -= 0.4
                        print(f"    ❌ DIF无有效数据")
                    else:
                        print(f"    ✅ DIF: {len(dif_clean)}个有效值")
                    
                    if len(dea_clean) == 0:
                        data_quality_score -= 0.4
                        print(f"    ❌ DEA无有效数据")
                    else:
                        print(f"    ✅ DEA: {len(dea_clean)}个有效值")
                    
                    if len(macd_clean) == 0:
                        data_quality_score -= 0.2
                        print(f"    ❌ MACD无有效数据")
                    else:
                        print(f"    ✅ MACD: {len(macd_clean)}个有效值")
                    
                    functionality_result['tests'].append({
                        'test': 'UTILS_FUNCTION_CALCULATION',
                        'result': 'PASSED' if data_quality_score >= 0.8 else 'FAILED',
                        'score': data_quality_score,
                        'dif_count': len(dif_clean),
                        'dea_count': len(dea_clean),
                        'macd_count': len(macd_clean)
                    })
                    
                    print(f"  📊 数据质量评分: {data_quality_score:.1%}")
                else:
                    functionality_result['tests'].append({
                        'test': 'UTILS_FUNCTION_CALCULATION',
                        'result': 'FAILED',
                        'score': 0.0,
                        'reason': 'NO_RESULT'
                    })
                    print(f"  ❌ 工具函数计算失败")
            
            except Exception as e:
                functionality_result['tests'].append({
                    'test': 'UTILS_FUNCTION_CALCULATION',
                    'result': 'ERROR',
                    'score': 0.0,
                    'error': str(e)
                })
                print(f"  ❌ 工具函数计算异常: {e}")
            
            # 测试3: 参数验证
            print(f"\n🔧 测试3: 参数验证")
            
            parameter_tests = [
                {'fast': 12, 'slow': 26, 'signal': 9, 'description': '标准参数'},
                {'fast': 5, 'slow': 10, 'signal': 5, 'description': '短期参数'},
                {'fast': 24, 'slow': 52, 'signal': 18, 'description': '长期参数'}
            ]
            
            parameter_scores = []
            
            for param_test in parameter_tests:
                try:
                    # 创建新的MACD指标实例，使用不同参数
                    test_macd = MacdMacd(
                        fast_period=param_test['fast'],
                        slow_period=param_test['slow'],
                        signal_period=param_test['signal']
                    )

                    # 计算MACD
                    param_result = test_macd._calculate_macd(test_data)
                    
                    if param_result is not None and not param_result.empty:
                        if 'macd_line' in param_result.columns:
                            macd_line = param_result['macd_line'].dropna()
                            if len(macd_line) > 0:
                                parameter_scores.append(1.0)
                                print(f"    ✅ {param_test['description']}: 成功")
                            else:
                                parameter_scores.append(0.5)
                                print(f"    ⚠️ {param_test['description']}: 无有效值")
                        else:
                            parameter_scores.append(0.0)
                            print(f"    ❌ {param_test['description']}: 缺少MACD线")
                    else:
                        parameter_scores.append(0.0)
                        print(f"    ❌ {param_test['description']}: 计算失败")
                
                except Exception as e:
                    parameter_scores.append(0.0)
                    print(f"    ❌ {param_test['description']}: 异常 - {e}")
            
            # 注意：不需要恢复默认参数，因为使用的是独立实例
            
            parameter_score = sum(parameter_scores) / len(parameter_scores) if parameter_scores else 0
            
            functionality_result['tests'].append({
                'test': 'PARAMETER_VALIDATION',
                'result': 'PASSED' if parameter_score >= 0.8 else 'FAILED',
                'score': parameter_score,
                'parameter_tests': parameter_tests,
                'parameter_scores': parameter_scores
            })
            
            print(f"  📊 参数验证评分: {parameter_score:.1%}")
            
            # 计算总体评分
            test_scores = [test['score'] for test in functionality_result['tests']]
            functionality_result['overall_score'] = sum(test_scores) / len(test_scores) if test_scores else 0
            
            if functionality_result['overall_score'] >= self.stage1_config['basic_functionality_target']:
                functionality_result['status'] = 'PASSED'
                print(f"\n✅ 基础功能测试通过: {functionality_result['overall_score']:.1%}")
            else:
                functionality_result['status'] = 'FAILED'
                print(f"\n❌ 基础功能测试失败: {functionality_result['overall_score']:.1%}")
        
        except Exception as e:
            functionality_result['status'] = 'ERROR'
            functionality_result['error'] = str(e)
            print(f"❌ 基础功能测试异常: {e}")
        
        return functionality_result
    
    def test_calculation_accuracy(self) -> Dict[str, Any]:
        """测试计算准确性"""
        
        print(f"\n📊 测试MACD计算准确性")
        print("=" * 60)
        
        accuracy_result = {
            'test_type': 'CALCULATION_ACCURACY',
            'timestamp': datetime.now().isoformat(),
            'accuracy_tests': [],
            'overall_accuracy': 0.0,
            'status': 'IN_PROGRESS'
        }
        
        try:
            # 创建测试数据
            test_data = self.create_test_data()
            
            # 使用系统MACD和工具函数进行对比
            print(f"📊 对比系统MACD与工具函数结果")
            
            # 系统MACD
            system_result = self.macd_indicator._calculate_macd(test_data)
            
            # 工具函数MACD
            utils_dif, utils_dea, utils_macd = calculate_macd_Utils(
                test_data['close'], 
                fast_period=12, 
                slow_period=26, 
                signal_period=9
            )
            
            if (system_result is not None and not system_result.empty and 
                utils_dif is not None and not utils_dif.empty):
                
                # 提取系统结果
                system_dif = system_result.get('macd_line', pd.Series()).dropna()
                system_dea = system_result.get('macd_signal', pd.Series()).dropna()
                system_hist = system_result.get('macd_histogram', pd.Series()).dropna()
                
                # 提取工具函数结果
                utils_dif_clean = utils_dif.dropna()
                utils_dea_clean = utils_dea.dropna()
                utils_macd_clean = utils_macd.dropna()
                
                # 对比DIF
                if len(system_dif) > 0 and len(utils_dif_clean) > 0:
                    min_len = min(len(system_dif), len(utils_dif_clean))
                    
                    system_dif_values = system_dif.tail(min_len).values
                    utils_dif_values = utils_dif_clean.tail(min_len).values
                    
                    dif_differences = np.abs(system_dif_values - utils_dif_values)
                    dif_max_diff = dif_differences.max()
                    dif_avg_diff = dif_differences.mean()
                    
                    # 计算DIF准确率
                    tolerance = 1e-6
                    dif_accurate_count = np.sum(dif_differences <= tolerance)
                    dif_accuracy = dif_accurate_count / len(dif_differences)
                    
                    accuracy_result['accuracy_tests'].append({
                        'component': 'DIF',
                        'compared_values': min_len,
                        'max_difference': float(dif_max_diff),
                        'avg_difference': float(dif_avg_diff),
                        'accuracy': dif_accuracy,
                        'status': 'PASSED' if dif_accuracy >= 0.99 else 'FAILED'
                    })
                    
                    print(f"  ✅ DIF对比: {min_len}个值, 最大差异{dif_max_diff:.6f}, 准确率{dif_accuracy:.1%}")
                
                # 对比DEA
                if len(system_dea) > 0 and len(utils_dea_clean) > 0:
                    min_len = min(len(system_dea), len(utils_dea_clean))
                    
                    system_dea_values = system_dea.tail(min_len).values
                    utils_dea_values = utils_dea_clean.tail(min_len).values
                    
                    dea_differences = np.abs(system_dea_values - utils_dea_values)
                    dea_max_diff = dea_differences.max()
                    dea_avg_diff = dea_differences.mean()
                    
                    # 计算DEA准确率
                    dea_accurate_count = np.sum(dea_differences <= tolerance)
                    dea_accuracy = dea_accurate_count / len(dea_differences)
                    
                    accuracy_result['accuracy_tests'].append({
                        'component': 'DEA',
                        'compared_values': min_len,
                        'max_difference': float(dea_max_diff),
                        'avg_difference': float(dea_avg_diff),
                        'accuracy': dea_accuracy,
                        'status': 'PASSED' if dea_accuracy >= 0.99 else 'FAILED'
                    })
                    
                    print(f"  ✅ DEA对比: {min_len}个值, 最大差异{dea_max_diff:.6f}, 准确率{dea_accuracy:.1%}")
                
                # 计算总体准确率
                accuracies = [test['accuracy'] for test in accuracy_result['accuracy_tests']]
                accuracy_result['overall_accuracy'] = sum(accuracies) / len(accuracies) if accuracies else 0
                
                if accuracy_result['overall_accuracy'] >= self.stage1_config['calculation_accuracy_target']:
                    accuracy_result['status'] = 'PASSED'
                    print(f"\n✅ 计算准确性测试通过: {accuracy_result['overall_accuracy']:.1%}")
                else:
                    accuracy_result['status'] = 'FAILED'
                    print(f"\n❌ 计算准确性测试失败: {accuracy_result['overall_accuracy']:.1%}")
            else:
                accuracy_result['status'] = 'NO_DATA'
                print(f"\n❌ 无法获取对比数据")
        
        except Exception as e:
            accuracy_result['status'] = 'ERROR'
            accuracy_result['error'] = str(e)
            print(f"❌ 计算准确性测试异常: {e}")
        
        return accuracy_result
    
    def test_error_handling(self) -> Dict[str, Any]:
        """测试错误处理"""
        
        print(f"\n🛡️ 测试MACD错误处理")
        print("=" * 60)
        
        error_result = {
            'test_type': 'ERROR_HANDLING',
            'timestamp': datetime.now().isoformat(),
            'error_tests': [],
            'handling_score': 0.0,
            'status': 'IN_PROGRESS'
        }
        
        try:
            error_scenarios = [
                {'name': '空数据', 'data': pd.DataFrame()},
                {'name': '单行数据', 'data': pd.DataFrame({'close': [100]})},
                {'name': '数据不足', 'data': pd.DataFrame({'close': [100, 101, 102]})},
                {'name': '包含NaN', 'data': pd.DataFrame({'close': [100, np.nan, 102, 103, 104]})},
                {'name': '包含无穷大', 'data': pd.DataFrame({'close': [100, 101, np.inf, 103, 104]})}
            ]
            
            handled_errors = 0
            
            for scenario in error_scenarios:
                print(f"  🔧 测试场景: {scenario['name']}")
                
                try:
                    # 测试系统MACD
                    result = self.macd_indicator._calculate_macd(scenario['data'])
                    
                    # 检查结果是否合理
                    if result is not None:
                        if isinstance(result, pd.DataFrame):
                            print(f"    ✅ 返回DataFrame，形状: {result.shape}")
                            handled_errors += 1
                        else:
                            print(f"    ⚠️ 返回非DataFrame类型")
                            handled_errors += 0.5
                    else:
                        print(f"    ✅ 返回None（合理处理）")
                        handled_errors += 1
                    
                    error_result['error_tests'].append({
                        'scenario': scenario['name'],
                        'status': 'HANDLED',
                        'result_type': type(result).__name__ if result is not None else 'None'
                    })
                
                except Exception as e:
                    print(f"    ❌ 抛出异常: {e}")
                    error_result['error_tests'].append({
                        'scenario': scenario['name'],
                        'status': 'EXCEPTION',
                        'error': str(e)
                    })
            
            # 计算错误处理评分
            error_result['handling_score'] = handled_errors / len(error_scenarios)
            
            if error_result['handling_score'] >= self.stage1_config['error_handling_target']:
                error_result['status'] = 'PASSED'
                print(f"\n✅ 错误处理测试通过: {error_result['handling_score']:.1%}")
            else:
                error_result['status'] = 'FAILED'
                print(f"\n❌ 错误处理测试失败: {error_result['handling_score']:.1%}")
        
        except Exception as e:
            error_result['status'] = 'ERROR'
            error_result['error'] = str(e)
            print(f"❌ 错误处理测试异常: {e}")
        
        return error_result
    
    def run_complete_stage1_validation(self) -> Dict[str, Any]:
        """运行完整的阶段1验证"""
        
        print(f"\n🎯 MACD指标阶段1基础功能验证")
        print("基于RSI项目成功经验，验证MACD指标基本计算功能")
        print("=" * 80)
        
        stage1_results = {
            'validation_type': 'MACD_STAGE1_VALIDATION',
            'validator': self.validator_name,
            'start_time': datetime.now().isoformat(),
            'stage1_config': self.stage1_config,
            'basic_functionality': {},
            'calculation_accuracy': {},
            'error_handling': {},
            'overall_assessment': {},
            'final_status': 'IN_PROGRESS'
        }
        
        try:
            # 1. 基础功能测试
            functionality_result = self.test_basic_functionality()
            stage1_results['basic_functionality'] = functionality_result
            
            # 2. 计算准确性测试
            accuracy_result = self.test_calculation_accuracy()
            stage1_results['calculation_accuracy'] = accuracy_result
            
            # 3. 错误处理测试
            error_result = self.test_error_handling()
            stage1_results['error_handling'] = error_result
            
            # 4. 总体评估
            overall_assessment = self._assess_stage1_results(
                functionality_result, accuracy_result, error_result
            )
            stage1_results['overall_assessment'] = overall_assessment
            stage1_results['final_status'] = overall_assessment['final_status']
            
            stage1_results['end_time'] = datetime.now().isoformat()
            
            print(f"\n🏆 MACD阶段1验证完成")
            print(f"最终状态: {stage1_results['final_status']}")
            print(f"总体评分: {overall_assessment.get('total_score', 0):.1f}/100")
            
        except Exception as e:
            stage1_results['final_status'] = 'ERROR'
            stage1_results['error'] = str(e)
            print(f"❌ 阶段1验证异常: {e}")
        
        # 保存结果
        self._save_stage1_results(stage1_results)
        
        return stage1_results
    
    def _assess_stage1_results(self, functionality_result: Dict, accuracy_result: Dict, 
                              error_result: Dict) -> Dict[str, Any]:
        """评估阶段1结果"""
        
        assessment = {
            'assessment_type': 'STAGE1_ASSESSMENT',
            'individual_scores': {},
            'total_score': 0.0,
            'final_status': 'UNKNOWN'
        }
        
        # 评估各项测试
        tests = {
            'basic_functionality': functionality_result,
            'calculation_accuracy': accuracy_result,
            'error_handling': error_result
        }
        
        total_score = 0.0
        
        for test_name, test_result in tests.items():
            test_status = test_result.get('status', 'UNKNOWN')
            
            if test_status == 'PASSED':
                score = 100
            elif test_status == 'FAILED':
                score = 60
            else:
                score = 30
            
            assessment['individual_scores'][test_name] = {
                'status': test_status,
                'score': score
            }
            
            total_score += score
        
        assessment['total_score'] = total_score / len(tests)
        
        # 确定最终状态
        if assessment['total_score'] >= 90:
            assessment['final_status'] = 'PASSED'
        elif assessment['total_score'] >= 70:
            assessment['final_status'] = 'CONDITIONAL_PASS'
        else:
            assessment['final_status'] = 'FAILED'
        
        return assessment
    
    def _save_stage1_results(self, results: Dict[str, Any]):
        """保存阶段1结果"""
        
        results_dir = Path("validation/macd_validation_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"MACD阶段1验证结果_{timestamp}.json"
        
        # 转换numpy类型
        def convert_types(obj):
            if isinstance(obj, (np.bool_, np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_types(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 阶段1验证结果已保存: {results_file}")

def main():
    """主函数"""
    print("🎯 MACD指标阶段1基础功能验证")
    print("基于RSI项目成功经验，验证MACD指标基本计算功能")
    
    # 创建阶段1验证器
    validator = MACDStage1Validator()
    
    # 运行完整的阶段1验证
    results = validator.run_complete_stage1_validation()
    
    # 显示结果摘要
    print(f"\n📊 MACD阶段1验证结果摘要")
    print("=" * 80)
    
    if 'overall_assessment' in results:
        assessment = results['overall_assessment']
        print(f"总体评分: {assessment.get('total_score', 0):.1f}/100")
        print(f"最终状态: {assessment.get('final_status', 'UNKNOWN')}")
        
        print(f"\n📋 各项测试结果:")
        for test_name, test_score in assessment.get('individual_scores', {}).items():
            status_icon = "✅" if test_score['status'] == 'PASSED' else "⚠️" if 'CONDITIONAL' in test_score['status'] else "❌"
            print(f"  {status_icon} {test_name}: {test_score['status']} ({test_score['score']}/100)")

if __name__ == "__main__":
    main()
