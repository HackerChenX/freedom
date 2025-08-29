#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MA指标阶段4: 架构合规性验证

严格遵循分层架构设计，不直接写SQL，使用依赖注入
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


class MAArchitectureCompliance:
    """MA指标架构合规性验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.verification_name = "MA架构合规性验证"
        self.start_time = datetime.now()
        
        # 架构合规性验证标准
        self.architecture_standards = {
            'layered_architecture': 95.0,     # 分层架构要求95%
            'dependency_injection': 95.0,     # 依赖注入要求95%
            'no_direct_sql': 95.0,           # 不直接写SQL要求95%
            'interface_compliance': 95.0,     # 接口合规性要求95%
            'separation_of_concerns': 95.0,   # 关注点分离要求95%
            'target_score': 95.0
        }
        
        logger.info(f"✅ {self.verification_name}初始化完成")
        logger.info(f"🎯 目标: 验证MA架构合规性")
    
    def run_architecture_compliance_verification(self) -> Dict[str, Any]:
        """运行架构合规性验证"""
        logger.info("🚀 开始MA架构合规性验证")
        
        verification_results = {
            'verification_session': {
                'name': self.verification_name,
                'start_time': self.start_time.isoformat(),
                'standards': self.architecture_standards
            },
            'layered_architecture_tests': {},
            'dependency_injection_tests': {},
            'no_direct_sql_tests': {},
            'interface_compliance_tests': {},
            'separation_of_concerns_tests': {},
            'final_assessment': {},
            'final_status': 'IN_PROGRESS'
        }
        
        try:
            # 导入MA指标
            from indicators.ma import MaMa
            ma = MaMa()
            
            # 测试1: 分层架构测试
            logger.info("🏗️ 测试1: 分层架构测试")
            layered_result = self._test_layered_architecture(ma)
            verification_results['layered_architecture_tests'] = layered_result
            
            # 测试2: 依赖注入测试
            logger.info("💉 测试2: 依赖注入测试")
            di_result = self._test_dependency_injection(ma)
            verification_results['dependency_injection_tests'] = di_result
            
            # 测试3: 无直接SQL测试
            logger.info("🚫 测试3: 无直接SQL测试")
            no_sql_result = self._test_no_direct_sql(ma)
            verification_results['no_direct_sql_tests'] = no_sql_result
            
            # 测试4: 接口合规性测试
            logger.info("📋 测试4: 接口合规性测试")
            interface_result = self._test_interface_compliance(ma)
            verification_results['interface_compliance_tests'] = interface_result
            
            # 测试5: 关注点分离测试
            logger.info("🎯 测试5: 关注点分离测试")
            separation_result = self._test_separation_of_concerns(ma)
            verification_results['separation_of_concerns_tests'] = separation_result
            
            # 最终评估
            logger.info("📊 最终评估")
            final_assessment = self._generate_final_assessment(verification_results)
            verification_results['final_assessment'] = final_assessment
            
            # 确定最终状态
            final_status = self._determine_final_status(final_assessment)
            verification_results['final_status'] = final_status
            
            logger.info("✅ MA架构合规性验证完成")
            return verification_results
            
        except Exception as e:
            logger.error(f"❌ 验证过程中发生异常: {e}")
            verification_results['final_status'] = 'ERROR'
            verification_results['error'] = str(e)
            verification_results['traceback'] = traceback.format_exc()
            return verification_results
    
    def _test_layered_architecture(self, ma) -> Dict[str, Any]:
        """测试分层架构"""
        logger.info("🏗️ 测试MA分层架构...")
        
        layered_result = {
            'inheritance_structure_test': {},
            'method_organization_test': {},
            'abstraction_compliance_test': {},
            'overall_score': 0.0
        }
        
        try:
            # 测试1: 继承结构测试
            inheritance_test = self._test_inheritance_structure(ma)
            layered_result['inheritance_structure_test'] = inheritance_test
            
            # 测试2: 方法组织测试
            method_org_test = self._test_method_organization(ma)
            layered_result['method_organization_test'] = method_org_test
            
            # 测试3: 抽象合规性测试
            abstraction_test = self._test_abstraction_compliance(ma)
            layered_result['abstraction_compliance_test'] = abstraction_test
            
            # 计算总体评分
            scores = [
                inheritance_test.get('score', 0),
                method_org_test.get('score', 0),
                abstraction_test.get('score', 0)
            ]
            layered_result['overall_score'] = sum(scores) / len(scores)
            
            logger.info(f"✅ 分层架构测试完成: {layered_result['overall_score']:.1f}分")
            return layered_result
            
        except Exception as e:
            logger.error(f"❌ 分层架构测试失败: {e}")
            layered_result['error'] = str(e)
            return layered_result
    
    def _test_inheritance_structure(self, ma) -> Dict[str, Any]:
        """测试继承结构"""
        try:
            # 检查继承结构
            ma_class = ma.__class__
            base_classes = ma_class.__bases__
            
            # 检查是否有适当的基类
            has_base_class = len(base_classes) > 0
            base_class_names = [cls.__name__ for cls in base_classes]
            
            # 检查是否继承自抽象基类或指标基类
            has_indicator_base = any('Indicator' in name or 'Base' in name or 'Mixin' in name for name in base_class_names)
            
            # 检查MRO（方法解析顺序）
            mro = ma_class.__mro__
            mro_length = len(mro)
            
            score = 0
            if has_base_class:
                score += 40
            if has_indicator_base:
                score += 40
            if mro_length >= 3:  # 至少有自己、基类、object
                score += 20
            
            return {
                'has_base_class': has_base_class,
                'base_class_names': base_class_names,
                'has_indicator_base': has_indicator_base,
                'mro_length': mro_length,
                'mro_classes': [cls.__name__ for cls in mro],
                'score': score
            }
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_method_organization(self, ma) -> Dict[str, Any]:
        """测试方法组织"""
        try:
            # 获取所有方法（安全地获取）
            all_methods = []
            for method_name in dir(ma):
                try:
                    attr = getattr(ma, method_name)
                    if callable(attr):
                        all_methods.append(method_name)
                except Exception:
                    continue
            
            # 分类方法
            public_methods = [m for m in all_methods if not m.startswith('_')]
            private_methods = [m for m in all_methods if m.startswith('_') and not m.startswith('__')]
            magic_methods = [m for m in all_methods if m.startswith('__') and m.endswith('__')]
            
            # 检查必需的公共方法
            required_public = ['calculate', 'set_parameters', 'get_patterns']
            has_required_public = [m for m in required_public if m in public_methods]
            
            # 检查是否有适当的私有方法（实现细节）
            has_private_implementation = len(private_methods) > 0
            
            # 评分
            score = 0
            if len(has_required_public) >= 2:
                score += 50
            if has_private_implementation:
                score += 30
            if len(public_methods) >= 3:
                score += 20
            
            return {
                'total_methods': len(all_methods),
                'public_methods': len(public_methods),
                'private_methods': len(private_methods),
                'magic_methods': len(magic_methods),
                'required_public_methods': has_required_public,
                'has_private_implementation': has_private_implementation,
                'score': score
            }
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_abstraction_compliance(self, ma) -> Dict[str, Any]:
        """测试抽象合规性"""
        try:
            # 检查是否实现了抽象方法
            required_abstract_methods = [
                'calculate', '_get_default_parameters', 'minimum_periods'
            ]
            
            implemented_methods = []
            missing_methods = []
            
            for method in required_abstract_methods:
                if hasattr(ma, method):
                    implemented_methods.append(method)
                else:
                    missing_methods.append(method)
            
            # 检查minimum_periods是否为属性
            has_minimum_periods_property = hasattr(ma, 'minimum_periods')
            
            # 评分
            implementation_rate = len(implemented_methods) / len(required_abstract_methods)
            score = implementation_rate * 100
            
            return {
                'required_abstract_methods': required_abstract_methods,
                'implemented_methods': implemented_methods,
                'missing_methods': missing_methods,
                'has_minimum_periods_property': has_minimum_periods_property,
                'implementation_rate': implementation_rate,
                'score': score
            }
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_dependency_injection(self, ma) -> Dict[str, Any]:
        """测试依赖注入"""
        logger.info("💉 测试MA依赖注入...")
        
        try:
            # 检查是否使用了依赖注入容器
            uses_di_container = False
            has_logger_injection = False
            has_service_injection = False
            
            # 检查源代码中是否使用了依赖注入
            import inspect
            source_lines = inspect.getsourcelines(ma.__class__)[0]
            source_code = ''.join(source_lines)
            
            # 检查依赖注入的使用
            di_patterns = [
                'get_logger',
                'dependency_injection',
                'inject',
                'container'
            ]

            found_di_patterns = []
            for pattern in di_patterns:
                if pattern in source_code:
                    uses_di_container = True
                    found_di_patterns.append(pattern)

            # 检查是否有logger注入
            if hasattr(ma, 'logger') or 'logger' in source_code or 'get_logger' in source_code:
                has_logger_injection = True

            # 检查模块级别的依赖注入使用
            import inspect
            module = inspect.getmodule(ma.__class__)
            if module:
                module_source = inspect.getsource(module)
                if 'get_logger' in module_source or 'dependency_injection' in module_source:
                    uses_di_container = True
                    has_logger_injection = True
            
            # 评分
            score = 0
            if uses_di_container:
                score += 60
            if has_logger_injection:
                score += 40
            
            return {
                'uses_di_container': uses_di_container,
                'has_logger_injection': has_logger_injection,
                'has_service_injection': has_service_injection,
                'di_patterns_found': found_di_patterns,
                'score': score
            }
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_no_direct_sql(self, ma) -> Dict[str, Any]:
        """测试无直接SQL"""
        logger.info("🚫 测试MA无直接SQL...")
        
        try:
            # 检查源代码中是否有直接的SQL语句
            import inspect
            source_lines = inspect.getsourcelines(ma.__class__)[0]
            source_code = ''.join(source_lines).lower()
            
            # SQL关键词检查（更精确的SQL语句检测）
            sql_patterns = [
                r'select\s+.*\s+from\s+',
                r'insert\s+into\s+',
                r'update\s+.*\s+set\s+',
                r'delete\s+from\s+',
                r'create\s+table\s+',
                r'drop\s+table\s+',
                r'alter\s+table\s+'
            ]
            
            import re
            found_sql_patterns = []
            for pattern in sql_patterns:
                if re.search(pattern, source_code, re.IGNORECASE):
                    found_sql_patterns.append(pattern)
            
            # 检查是否有数据库连接相关代码
            db_patterns = [
                'connection', 'cursor', 'execute',
                'fetchall', 'fetchone', 'commit'
            ]
            
            found_db_patterns = []
            for pattern in db_patterns:
                if pattern in source_code:
                    found_db_patterns.append(pattern)
            
            # 评分：没有直接SQL得满分
            has_direct_sql = len(found_sql_patterns) > 0
            has_db_operations = len(found_db_patterns) > 0

            if not has_direct_sql and not has_db_operations:
                score = 100
            elif not has_direct_sql:
                score = 80
            else:
                score = 0

            return {
                'has_direct_sql': has_direct_sql,
                'found_sql_patterns': found_sql_patterns,
                'has_db_operations': has_db_operations,
                'found_db_patterns': found_db_patterns,
                'complies_with_no_sql_rule': not has_direct_sql,
                'score': score
            }
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_interface_compliance(self, ma) -> Dict[str, Any]:
        """测试接口合规性"""
        logger.info("📋 测试MA接口合规性...")
        
        try:
            # 检查标准接口方法
            standard_interface = {
                'calculate': {'required': True, 'signature_check': True},
                'set_parameters': {'required': True, 'signature_check': True},
                '_get_default_parameters': {'required': True, 'signature_check': True},
                'get_patterns': {'required': True, 'signature_check': False},
                'minimum_periods': {'required': True, 'is_property': True}
            }
            
            interface_compliance = {}
            total_score = 0
            max_score = len(standard_interface) * 20
            
            for method_name, requirements in standard_interface.items():
                compliance = {'exists': False, 'signature_correct': False, 'is_property': False}
                
                if hasattr(ma, method_name):
                    compliance['exists'] = True
                    total_score += 10
                    
                    # 检查是否为属性
                    if requirements.get('is_property', False):
                        if isinstance(getattr(ma.__class__, method_name, None), property):
                            compliance['is_property'] = True
                            total_score += 10
                    else:
                        # 检查是否可调用
                        if callable(getattr(ma, method_name)):
                            compliance['signature_correct'] = True
                            total_score += 10
                
                interface_compliance[method_name] = compliance
            
            final_score = (total_score / max_score) * 100 if max_score > 0 else 0
            
            return {
                'interface_compliance': interface_compliance,
                'total_score': total_score,
                'max_score': max_score,
                'compliance_rate': total_score / max_score if max_score > 0 else 0,
                'score': final_score
            }
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_separation_of_concerns(self, ma) -> Dict[str, Any]:
        """测试关注点分离"""
        logger.info("🎯 测试MA关注点分离...")

        try:
            # 简化的关注点分离测试
            # 检查是否有不同职责的方法（安全地获取）
            all_methods = []
            for method_name in dir(ma):
                if not method_name.startswith('__'):
                    try:
                        attr = getattr(ma, method_name)
                        if callable(attr):
                            all_methods.append(method_name)
                    except Exception:
                        continue

            # 按职责分类方法
            calculation_methods = [m for m in all_methods if 'calculate' in m.lower()]
            parameter_methods = [m for m in all_methods if 'parameter' in m.lower() or 'set_' in m.lower()]
            pattern_methods = [m for m in all_methods if 'pattern' in m.lower() or 'signal' in m.lower()]
            utility_methods = [m for m in all_methods if m not in calculation_methods + parameter_methods + pattern_methods]

            # 检查职责分离情况
            has_calculation_separation = len(calculation_methods) > 0
            has_parameter_separation = len(parameter_methods) > 0
            has_pattern_separation = len(pattern_methods) > 0
            has_utility_separation = len(utility_methods) > 0

            # 检查方法数量分布是否合理
            total_methods = len(all_methods)
            method_distribution_reasonable = total_methods >= 5  # 至少有5个方法

            # 评分
            score = 0
            if has_calculation_separation:
                score += 25
            if has_parameter_separation:
                score += 25
            if has_pattern_separation:
                score += 25
            if method_distribution_reasonable:
                score += 25

            return {
                'total_methods': total_methods,
                'calculation_methods': calculation_methods,
                'parameter_methods': parameter_methods,
                'pattern_methods': pattern_methods,
                'utility_methods': utility_methods,
                'has_calculation_separation': has_calculation_separation,
                'has_parameter_separation': has_parameter_separation,
                'has_pattern_separation': has_pattern_separation,
                'method_distribution_reasonable': method_distribution_reasonable,
                'separation_quality': 'good' if score >= 75 else 'needs_improvement',
                'score': score
            }

        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _analyze_method_responsibility(self, method_name: str) -> str:
        """分析方法职责"""
        if 'calculate' in method_name.lower():
            return 'calculation'
        elif 'parameter' in method_name.lower():
            return 'parameter_management'
        elif 'pattern' in method_name.lower():
            return 'pattern_recognition'
        elif 'signal' in method_name.lower():
            return 'signal_generation'
        else:
            return 'utility'
    
    def _generate_final_assessment(self, verification_results: Dict) -> Dict[str, Any]:
        """生成最终评估"""
        layered_score = verification_results.get('layered_architecture_tests', {}).get('overall_score', 0)
        di_score = verification_results.get('dependency_injection_tests', {}).get('score', 0)
        no_sql_score = verification_results.get('no_direct_sql_tests', {}).get('score', 0)
        interface_score = verification_results.get('interface_compliance_tests', {}).get('score', 0)
        separation_score = verification_results.get('separation_of_concerns_tests', {}).get('score', 0)
        
        overall_score = (layered_score + di_score + no_sql_score + interface_score + separation_score) / 5
        
        return {
            'layered_architecture_score': layered_score,
            'dependency_injection_score': di_score,
            'no_direct_sql_score': no_sql_score,
            'interface_compliance_score': interface_score,
            'separation_of_concerns_score': separation_score,
            'overall_score': overall_score,
            'architecture_compliant': overall_score >= 95.0,
            'target_achieved': overall_score >= 95.0
        }
    
    def _determine_final_status(self, final_assessment: Dict) -> str:
        """确定最终状态"""
        overall_score = final_assessment.get('overall_score', 0)
        
        if overall_score >= 95.0:
            return 'ARCHITECTURE_COMPLIANT'
        elif overall_score >= 85.0:
            return 'MOSTLY_ARCHITECTURE_COMPLIANT'
        else:
            return 'ARCHITECTURE_NEEDS_IMPROVEMENT'


def main():
    """主函数"""
    print("🚀 启动MA指标阶段4: 架构合规性验证")
    print("严格遵循分层架构设计，不直接写SQL，使用依赖注入")
    print("=" * 80)
    
    try:
        # 创建验证器
        verifier = MAArchitectureCompliance()
        
        # 运行架构合规性验证
        results = verifier.run_architecture_compliance_verification()
        
        # 输出验证摘要
        print(f"\n📊 验证摘要:")
        print(f"最终状态: {results['final_status']}")
        
        if 'final_assessment' in results:
            assessment = results['final_assessment']
            print(f"分层架构评分: {assessment.get('layered_architecture_score', 0):.1f}/100")
            print(f"依赖注入评分: {assessment.get('dependency_injection_score', 0):.1f}/100")
            print(f"无直接SQL评分: {assessment.get('no_direct_sql_score', 0):.1f}/100")
            print(f"接口合规性评分: {assessment.get('interface_compliance_score', 0):.1f}/100")
            print(f"关注点分离评分: {assessment.get('separation_of_concerns_score', 0):.1f}/100")
            print(f"总体评分: {assessment.get('overall_score', 0):.1f}/100")
            print(f"架构合规: {'✅ 是' if assessment.get('architecture_compliant', False) else '❌ 否'}")
            print(f"目标达成: {'✅ 是' if assessment.get('target_achieved', False) else '❌ 否'}")
        
        if results['final_status'] == 'ARCHITECTURE_COMPLIANT':
            print("🎉 MA架构合规性验证通过!")
            return 0
        else:
            print("⚠️ MA架构合规性需要进一步优化")
            return 1
            
    except Exception as e:
        logger.error(f"💥 验证执行失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
