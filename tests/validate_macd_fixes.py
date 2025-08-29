#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证MACD指标修复效果

测试修复后的MACD指标是否达到PASSED状态
"""

import sys
import os
import time
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class MACDFixValidator:
    """MACD修复效果验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.validator_name = "MACD修复效果验证器"
        self.start_time = datetime.now()
        
        # 验证标准
        self.validation_standards = {
            'architecture_compliance': {
                'required_methods': [
                    'set_parameters_Indicator_Base_Indicator',
                    '_get_default_parameters',
                    'calculate_raw_score_Indicator_Base_Indicator',
                    'get_patterns_Indicator_Base_Indicator'
                ],
                'forbidden_methods': [
                    'set_parameters_Macd_Macd_Macd_macd_duplicate'
                ],
                'min_score': 90.0
            },
            'parameter_management': {
                'required_parameters': ['fast_period', 'slow_period', 'signal_period'],
                'parameter_validation': True,
                'min_score': 90.0
            },
            'empty_data_handling': {
                'handle_empty_data': True,
                'handle_insufficient_data': True,
                'handle_missing_columns': True,
                'min_score': 85.0
            },
            'overall_target': {
                'min_score': 95.0,
                'target_status': 'PASSED'
            }
        }
        
        logger.info(f"✅ {self.validator_name}初始化完成")
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """运行综合验证"""
        logger.info("🚀 开始MACD修复效果综合验证")
        
        validation_results = {
            'validation_session': {
                'name': self.validator_name,
                'start_time': self.start_time.isoformat(),
                'standards': self.validation_standards
            },
            'test_results': {},
            'overall_assessment': {},
            'final_status': 'IN_PROGRESS'
        }
        
        try:
            # 测试1: 架构合规性验证
            logger.info("🔍 测试1: 架构合规性验证")
            architecture_result = self._test_architecture_compliance()
            validation_results['test_results']['architecture_compliance'] = architecture_result
            
            # 测试2: 参数管理验证
            logger.info("⚙️ 测试2: 参数管理验证")
            parameter_result = self._test_parameter_management()
            validation_results['test_results']['parameter_management'] = parameter_result
            
            # 测试3: 空数据处理验证
            logger.info("🛡️ 测试3: 空数据处理验证")
            empty_data_result = self._test_empty_data_handling()
            validation_results['test_results']['empty_data_handling'] = empty_data_result
            
            # 测试4: 基本功能验证
            logger.info("🔧 测试4: 基本功能验证")
            functionality_result = self._test_basic_functionality()
            validation_results['test_results']['basic_functionality'] = functionality_result
            
            # 综合评估
            logger.info("📊 综合评估")
            overall_assessment = self._assess_overall_performance(validation_results['test_results'])
            validation_results['overall_assessment'] = overall_assessment
            
            # 确定最终状态
            final_status = self._determine_final_status(overall_assessment)
            validation_results['final_status'] = final_status
            
            logger.info("✅ MACD修复效果综合验证完成")
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ 验证过程中发生异常: {e}")
            validation_results['final_status'] = 'ERROR'
            validation_results['error'] = str(e)
            validation_results['traceback'] = traceback.format_exc()
            return validation_results
    
    def _test_architecture_compliance(self) -> Dict[str, Any]:
        """测试架构合规性"""
        logger.info("🔍 测试MACD架构合规性...")
        
        test_result = {
            'test_name': 'architecture_compliance',
            'required_methods_check': {},
            'forbidden_methods_check': {},
            'method_naming_check': {},
            'score': 0.0,
            'status': 'UNKNOWN'
        }
        
        try:
            # 导入MACD指标
            from indicators.macd import MacdMacd
            macd = MacdMacd()
            
            # 检查必需方法
            required_methods = self.validation_standards['architecture_compliance']['required_methods']
            missing_methods = []
            present_methods = []
            
            for method in required_methods:
                if hasattr(macd, method):
                    present_methods.append(method)
                else:
                    missing_methods.append(method)
            
            test_result['required_methods_check'] = {
                'required': required_methods,
                'present': present_methods,
                'missing': missing_methods,
                'compliance_rate': len(present_methods) / len(required_methods) * 100
            }
            
            # 检查禁止的方法（应该已被删除）
            forbidden_methods = self.validation_standards['architecture_compliance']['forbidden_methods']
            found_forbidden = []
            
            for method in forbidden_methods:
                if hasattr(macd, method):
                    found_forbidden.append(method)
            
            test_result['forbidden_methods_check'] = {
                'forbidden': forbidden_methods,
                'found': found_forbidden,
                'clean': len(found_forbidden) == 0
            }
            
            # 检查方法命名规范
            all_methods = [method for method in dir(macd) if not method.startswith('__')]
            irregular_methods = []
            
            for method in all_methods:
                if method.count('_') > 6 or 'duplicate' in method.lower():
                    irregular_methods.append(method)
            
            test_result['method_naming_check'] = {
                'total_methods': len(all_methods),
                'irregular_methods': irregular_methods[:5],  # 只显示前5个
                'irregular_count': len(irregular_methods),
                'naming_compliance_rate': (len(all_methods) - len(irregular_methods)) / len(all_methods) * 100
            }
            
            # 计算总体评分
            compliance_score = test_result['required_methods_check']['compliance_rate']
            forbidden_score = 100.0 if test_result['forbidden_methods_check']['clean'] else 0.0
            naming_score = test_result['method_naming_check']['naming_compliance_rate']
            
            test_result['score'] = (compliance_score + forbidden_score + naming_score) / 3
            
            if test_result['score'] >= 90.0:
                test_result['status'] = 'PASSED'
            elif test_result['score'] >= 75.0:
                test_result['status'] = 'CONDITIONAL_PASS'
            else:
                test_result['status'] = 'FAILED'
            
            logger.info(f"✅ 架构合规性测试完成: {test_result['score']:.1f}分")
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 架构合规性测试失败: {e}")
            test_result['error'] = str(e)
            test_result['status'] = 'ERROR'
            return test_result
    
    def _test_parameter_management(self) -> Dict[str, Any]:
        """测试参数管理"""
        logger.info("⚙️ 测试MACD参数管理...")
        
        test_result = {
            'test_name': 'parameter_management',
            'default_parameters_check': {},
            'parameter_setting_check': {},
            'parameter_validation_check': {},
            'score': 0.0,
            'status': 'UNKNOWN'
        }
        
        try:
            from indicators.macd import MacdMacd
            
            # 测试默认参数
            macd = MacdMacd()
            default_params = macd._get_default_parameters()
            
            required_params = self.validation_standards['parameter_management']['required_parameters']
            missing_params = []
            present_params = []
            
            for param in required_params:
                if param in default_params:
                    present_params.append(param)
                else:
                    missing_params.append(param)
            
            test_result['default_parameters_check'] = {
                'required': required_params,
                'present': present_params,
                'missing': missing_params,
                'default_params': default_params,
                'completeness': len(present_params) / len(required_params) * 100
            }
            
            # 测试参数设置
            try:
                macd.set_parameters(fast_period=10, slow_period=20, signal_period=5)
                parameter_setting_success = True
                setting_error = None
            except Exception as e:
                parameter_setting_success = False
                setting_error = str(e)
            
            test_result['parameter_setting_check'] = {
                'success': parameter_setting_success,
                'error': setting_error
            }
            
            # 计算评分
            default_score = test_result['default_parameters_check']['completeness']
            setting_score = 100.0 if parameter_setting_success else 0.0
            
            test_result['score'] = (default_score + setting_score) / 2
            
            if test_result['score'] >= 90.0:
                test_result['status'] = 'PASSED'
            elif test_result['score'] >= 75.0:
                test_result['status'] = 'CONDITIONAL_PASS'
            else:
                test_result['status'] = 'FAILED'
            
            logger.info(f"✅ 参数管理测试完成: {test_result['score']:.1f}分")
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 参数管理测试失败: {e}")
            test_result['error'] = str(e)
            test_result['status'] = 'ERROR'
            return test_result
    
    def _test_empty_data_handling(self) -> Dict[str, Any]:
        """测试空数据处理"""
        logger.info("🛡️ 测试MACD空数据处理...")
        
        test_result = {
            'test_name': 'empty_data_handling',
            'empty_data_test': {},
            'insufficient_data_test': {},
            'missing_columns_test': {},
            'score': 0.0,
            'status': 'UNKNOWN'
        }
        
        try:
            from indicators.macd import MacdMacd
            macd = MacdMacd()
            
            # 测试1: 完全空数据
            try:
                empty_df = pd.DataFrame()
                result = macd.calculate(empty_df)
                empty_data_handled = True
                empty_error = None
            except Exception as e:
                empty_data_handled = False
                empty_error = str(e)
            
            test_result['empty_data_test'] = {
                'handled': empty_data_handled,
                'error': empty_error
            }
            
            # 测试2: 数据长度不足
            try:
                short_df = pd.DataFrame({
                    'close': [100, 101, 102],
                    'open': [99, 100, 101],
                    'high': [101, 102, 103],
                    'low': [98, 99, 100],
                    'volume': [1000, 1100, 1200]
                })
                result = macd.calculate(short_df)
                insufficient_data_handled = True
                insufficient_error = None
            except Exception as e:
                insufficient_data_handled = False
                insufficient_error = str(e)
            
            test_result['insufficient_data_test'] = {
                'handled': insufficient_data_handled,
                'error': insufficient_error
            }
            
            # 测试3: 缺少必需列
            try:
                missing_col_df = pd.DataFrame({
                    'open': [99, 100, 101] * 20,
                    'high': [101, 102, 103] * 20,
                    'low': [98, 99, 100] * 20,
                    'volume': [1000, 1100, 1200] * 20
                    # 故意不包含 'close' 列
                })
                result = macd.calculate(missing_col_df)
                missing_columns_handled = True
                missing_error = None
            except Exception as e:
                missing_columns_handled = False
                missing_error = str(e)
            
            test_result['missing_columns_test'] = {
                'handled': missing_columns_handled,
                'error': missing_error
            }
            
            # 计算评分
            empty_score = 100.0 if empty_data_handled else 0.0
            insufficient_score = 100.0 if insufficient_data_handled else 0.0
            missing_score = 100.0 if missing_columns_handled else 0.0
            
            test_result['score'] = (empty_score + insufficient_score + missing_score) / 3
            
            if test_result['score'] >= 85.0:
                test_result['status'] = 'PASSED'
            elif test_result['score'] >= 70.0:
                test_result['status'] = 'CONDITIONAL_PASS'
            else:
                test_result['status'] = 'FAILED'
            
            logger.info(f"✅ 空数据处理测试完成: {test_result['score']:.1f}分")
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 空数据处理测试失败: {e}")
            test_result['error'] = str(e)
            test_result['status'] = 'ERROR'
            return test_result
    
    def _test_basic_functionality(self) -> Dict[str, Any]:
        """测试基本功能"""
        logger.info("🔧 测试MACD基本功能...")
        
        test_result = {
            'test_name': 'basic_functionality',
            'calculation_test': {},
            'patterns_test': {},
            'score': 0.0,
            'status': 'UNKNOWN'
        }
        
        try:
            from indicators.macd import MacdMacd
            macd = MacdMacd()
            
            # 创建测试数据
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            prices = 100 + np.cumsum(np.random.normal(0.1, 1, 100))
            
            test_data = pd.DataFrame({
                'date': dates,
                'open': prices * 0.99,
                'high': prices * 1.02,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, 100)
            })
            
            # 测试计算功能
            try:
                calc_result = macd.calculate(test_data)
                calculation_success = calc_result is not None and not calc_result.empty
                calc_error = None
            except Exception as e:
                calculation_success = False
                calc_error = str(e)
            
            test_result['calculation_test'] = {
                'success': calculation_success,
                'error': calc_error
            }
            
            # 测试形态识别
            try:
                patterns_result = macd.get_patterns(test_data)
                patterns_success = patterns_result is not None and not patterns_result.empty
                patterns_error = None
            except Exception as e:
                patterns_success = False
                patterns_error = str(e)
            
            test_result['patterns_test'] = {
                'success': patterns_success,
                'error': patterns_error
            }
            
            # 计算评分
            calc_score = 100.0 if calculation_success else 0.0
            patterns_score = 100.0 if patterns_success else 0.0
            
            test_result['score'] = (calc_score + patterns_score) / 2
            
            if test_result['score'] >= 90.0:
                test_result['status'] = 'PASSED'
            elif test_result['score'] >= 75.0:
                test_result['status'] = 'CONDITIONAL_PASS'
            else:
                test_result['status'] = 'FAILED'
            
            logger.info(f"✅ 基本功能测试完成: {test_result['score']:.1f}分")
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 基本功能测试失败: {e}")
            test_result['error'] = str(e)
            test_result['status'] = 'ERROR'
            return test_result
    
    def _assess_overall_performance(self, test_results: Dict) -> Dict[str, Any]:
        """评估总体性能"""
        scores = []
        statuses = []
        
        for test_name, result in test_results.items():
            if 'score' in result:
                scores.append(result['score'])
            if 'status' in result:
                statuses.append(result['status'])
        
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        # 确定总体状态
        if overall_score >= 95.0 and all(status in ['PASSED', 'CONDITIONAL_PASS'] for status in statuses):
            overall_status = 'PASSED'
        elif overall_score >= 85.0:
            overall_status = 'CONDITIONAL_PASS_IMPROVED'
        elif overall_score >= 75.0:
            overall_status = 'CONDITIONAL_PASS'
        else:
            overall_status = 'NEEDS_IMPROVEMENT'
        
        return {
            'overall_score': overall_score,
            'overall_status': overall_status,
            'individual_scores': {name: result.get('score', 0) for name, result in test_results.items()},
            'individual_statuses': {name: result.get('status', 'UNKNOWN') for name, result in test_results.items()}
        }
    
    def _determine_final_status(self, overall_assessment: Dict) -> str:
        """确定最终状态"""
        return overall_assessment.get('overall_status', 'UNKNOWN')


def main():
    """主函数"""
    print("🚀 启动MACD修复效果验证")
    print("验证MACD指标是否达到PASSED状态")
    print("=" * 80)
    
    try:
        # 创建验证器
        validator = MACDFixValidator()
        
        # 运行综合验证
        results = validator.run_comprehensive_validation()
        
        # 输出验证摘要
        print(f"\n📊 验证摘要:")
        print(f"最终状态: {results['final_status']}")
        
        if 'overall_assessment' in results:
            overall_score = results['overall_assessment'].get('overall_score', 0)
            print(f"总体评分: {overall_score:.1f}/100")
            
            # 显示各项测试结果
            print(f"\n📋 详细结果:")
            for test_name, score in results['overall_assessment'].get('individual_scores', {}).items():
                status = results['overall_assessment'].get('individual_statuses', {}).get(test_name, 'UNKNOWN')
                print(f"  {test_name}: {score:.1f}分 ({status})")
        
        if results['final_status'] == 'PASSED':
            print("🎉 MACD指标成功达到PASSED状态!")
            return 0
        else:
            print("⚠️ MACD指标仍需进一步改进")
            return 1
            
    except Exception as e:
        logger.error(f"💥 验证执行失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
