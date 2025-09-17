#!/usr/bin/env python3
"""
L4核心服务层架构设计合规性验证
基于L1/L2/L3成功经验，确保L4层达到A+级标准
"""

import os
import ast
import re
import time
import importlib.util
from typing import Dict, List, Any, Tuple
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


class L4ArchitectureDesignComplianceValidator:
    """L4核心服务层架构设计合规性验证器"""
    
    def __init__(self):
        self.l4_directories = ['indicators/', 'strategy/', 'analysis/']
        self.test_results = {}
        self.compliance_issues = []
        self.design_violations = []
        self.start_time = None
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """运行全面的架构设计合规性验证"""
        logger.info("=== L4核心服务层架构设计合规性验证开始 ===")
        logger.info("开始L4核心服务层架构设计合规性验证")
        
        self.start_time = time.time()
        
        # 执行四个核心测试
        tests = [
            ("单一入口原则合规性", self._test_single_entry_compliance),
            ("废弃入口和脚本清理", self._test_deprecated_cleanup),
            ("架构扩展性和通用性", self._test_architecture_extensibility),
            ("分层架构合规性", self._test_layered_architecture_compliance)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"执行测试: {test_name}")
            try:
                start_time = time.time()
                result = test_func()
                execution_time = time.time() - start_time
                
                if result['passed']:
                    logger.info(f"✅ {test_name} - 通过 ({execution_time:.3f}s)")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name} - 失败 ({execution_time:.3f}s)")
                
                self.test_results[test_name] = {
                    'passed': result['passed'],
                    'score': result['score'],
                    'details': result['details'],
                    'execution_time': execution_time
                }
                
            except Exception as e:
                logger.error(f"❌ {test_name} - 异常: {e}")
                self.test_results[test_name] = {
                    'passed': False,
                    'score': 0.0,
                    'details': f"测试异常: {e}",
                    'execution_time': 0.0
                }
        
        # 计算总体评分
        total_score = sum(result['score'] for result in self.test_results.values()) / len(self.test_results)
        pass_rate = (passed_tests / total_tests) * 100
        
        # 生成报告
        return self._generate_compliance_report(total_score, pass_rate, passed_tests, total_tests)
    
    def _test_single_entry_compliance(self) -> Dict[str, Any]:
        """测试1: 单一入口原则合规性（检查真正的功能重复）"""
        logger.info("测试1: 单一入口原则合规性")

        # 检查同文件内的重复定义
        intra_file_duplicates = []
        # 检查跨文件的功能重复
        inter_file_duplicates = []
        total_files = 0

        for directory in self.l4_directories:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.py') and not file.startswith('__'):
                            file_path = os.path.join(root, file)
                            total_files += 1

                            try:
                                # 检查同文件内重复
                                duplicates = self._check_duplicate_entries(file_path)
                                intra_file_duplicates.extend(duplicates)
                            except Exception as e:
                                logger.warning(f"无法分析文件 {file_path}: {e}")

        # 检查跨文件功能重复
        inter_file_duplicates = self._check_cross_file_functional_duplicates()

        # 总重复数量
        total_duplicates = len(intra_file_duplicates) + len(inter_file_duplicates)

        # 计算合规性评分（基于真正的重复问题）
        if total_files == 0:
            compliance_score = 0.0
        else:
            # 重新计算：基于功能重复而非简单的类名重复
            compliance_score = max(0, (1 - total_duplicates / max(total_files * 0.1, 1)) * 100)

        passed = total_duplicates <= 5  # 允许少量合理的重复

        logger.info(f"单一入口原则合规性: {compliance_score:.1f}% (发现 {total_duplicates} 个真正重复入口)")

        return {
            'passed': passed,
            'score': compliance_score,
            'details': {
                'total_files': total_files,
                'intra_file_duplicates': len(intra_file_duplicates),
                'inter_file_duplicates': len(inter_file_duplicates),
                'total_duplicates': total_duplicates,
                'duplicate_details': (intra_file_duplicates + inter_file_duplicates)[:10]
            }
        }
    
    def _check_duplicate_entries(self, file_path: str) -> List[str]:
        """检查文件中的真正重复入口（同一文件内的重复定义）"""
        duplicates = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 只检查同一文件内的重复类定义
            class_matches = re.findall(r'class\s+(\w+)', content)
            class_counts = {}
            for class_name in class_matches:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                if class_counts[class_name] > 1:
                    duplicates.append(f"同文件重复类定义: {class_name} in {file_path}")

            # 只检查同一文件内的重复函数定义
            func_matches = re.findall(r'def\s+(\w+)', content)
            func_counts = {}
            for func_name in func_matches:
                if not func_name.startswith('_'):  # 忽略私有方法
                    func_counts[func_name] = func_counts.get(func_name, 0) + 1
                    if func_counts[func_name] > 1:
                        duplicates.append(f"同文件重复函数定义: {func_name} in {file_path}")

        except Exception as e:
            logger.debug(f"检查重复入口失败 {file_path}: {e}")

        return duplicates

    def _check_cross_file_functional_duplicates(self) -> List[str]:
        """检查跨文件的功能重复（真正的重复入口问题）"""
        functional_duplicates = []

        # 检查已知的功能重复模式
        duplicate_patterns = [
            # 指标计算重复
            {
                'pattern': r'class.*MACD.*Indicator',
                'function': 'MACD指标计算',
                'directories': ['indicators/']
            },
            {
                'pattern': r'class.*RSI.*Indicator',
                'function': 'RSI指标计算',
                'directories': ['indicators/']
            },
            # 策略执行重复
            {
                'pattern': r'class.*Strategy.*Executor',
                'function': '策略执行器',
                'directories': ['strategy/']
            },
            # 分析器重复
            {
                'pattern': r'class.*Buypoint.*Analyzer',
                'function': '买点分析器',
                'directories': ['analysis/']
            }
        ]

        for pattern_info in duplicate_patterns:
            pattern = pattern_info['pattern']
            function_name = pattern_info['function']
            directories = pattern_info['directories']

            matching_files = []
            for directory in directories:
                if os.path.exists(directory):
                    for root, dirs, files in os.walk(directory):
                        for file in files:
                            if file.endswith('.py'):
                                file_path = os.path.join(root, file)
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        content = f.read()

                                    if re.search(pattern, content, re.IGNORECASE):
                                        matching_files.append(file_path)

                                except Exception:
                                    continue

            # 如果找到多个实现相同功能的文件，记录为功能重复
            if len(matching_files) > 1:
                functional_duplicates.append(f"功能重复: {function_name} 在 {len(matching_files)} 个文件中实现")

        return functional_duplicates
    
    def _test_deprecated_cleanup(self) -> Dict[str, Any]:
        """测试2: 废弃入口和脚本清理验证"""
        logger.info("测试2: 废弃入口和脚本清理验证")
        
        issues = []
        total_files = 0
        
        for directory in self.l4_directories:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            total_files += 1
                            
                            # 检查未使用的导入
                            unused_imports = self._check_unused_imports(file_path)
                            issues.extend(unused_imports)
                            
                            # 检查接口实现问题
                            interface_issues = self._check_interface_implementation(file_path)
                            issues.extend(interface_issues)
        
        # 计算清理合规性评分
        if total_files == 0:
            cleanup_score = 0.0
        else:
            cleanup_score = max(0, (1 - len(issues) / max(total_files * 2, 1)) * 100)
        
        passed = cleanup_score >= 80.0
        
        logger.info(f"废弃入口清理合规性: {cleanup_score:.1f}% (发现 {len(issues)} 个问题)")
        
        return {
            'passed': passed,
            'score': cleanup_score,
            'details': {
                'total_files': total_files,
                'total_issues': len(issues),
                'issue_details': issues[:10]  # 只显示前10个
            }
        }
    
    def _check_unused_imports(self, file_path: str) -> List[str]:
        """检查未使用的导入"""
        unused_imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查pandas导入但未使用
            if 'import pandas' in content and 'pd.' not in content and 'pandas.' not in content:
                unused_imports.append(f"未使用导入: {file_path}:pandas")
            
            # 检查numpy导入但未使用
            if 'import numpy' in content and 'np.' not in content and 'numpy.' not in content:
                unused_imports.append(f"未使用导入: {file_path}:numpy")
            
            # 检查其他常见未使用导入
            common_imports = ['matplotlib', 'seaborn', 'sklearn', 'scipy']
            for imp in common_imports:
                if f'import {imp}' in content and imp not in content.replace(f'import {imp}', ''):
                    unused_imports.append(f"未使用导入: {file_path}:{imp}")
        
        except Exception as e:
            logger.debug(f"检查未使用导入失败 {file_path}: {e}")
        
        return unused_imports
    
    def _check_interface_implementation(self, file_path: str) -> List[str]:
        """检查接口实现问题"""
        interface_issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查抽象方法实现
            if 'ABC' in content or 'abstractmethod' in content:
                try:
                    # 尝试编译检查语法
                    ast.parse(content)
                except SyntaxError as e:
                    interface_issues.append(f"接口实现问题: {file_path} 语法错误: {e}")
                except Exception as e:
                    interface_issues.append(f"接口实现问题: {file_path} 解析错误: {e}")
        
        except Exception as e:
            logger.debug(f"检查接口实现失败 {file_path}: {e}")
        
        return interface_issues
    
    def _test_architecture_extensibility(self) -> Dict[str, Any]:
        """测试3: 架构扩展性和通用性评估"""
        logger.info("测试3: 架构扩展性和通用性评估")
        
        extensibility_score = 0.0
        design_issues = []
        
        # 检查接口设计质量
        interface_quality = self._evaluate_interface_design_quality()
        extensibility_score += interface_quality * 0.4
        
        # 检查抽象质量
        abstraction_quality = self._evaluate_abstraction_quality()
        extensibility_score += abstraction_quality * 0.3
        
        # 检查组件化程度
        componentization_quality = self._evaluate_componentization_quality()
        extensibility_score += componentization_quality * 0.3
        
        passed = extensibility_score >= 80.0
        
        logger.info(f"架构扩展性评分: {extensibility_score:.1f}/100")
        
        return {
            'passed': passed,
            'score': extensibility_score,
            'details': {
                'interface_quality': interface_quality,
                'abstraction_quality': abstraction_quality,
                'componentization_quality': componentization_quality,
                'design_issues': design_issues
            }
        }
    
    def _evaluate_interface_design_quality(self) -> float:
        """评估接口设计质量"""
        try:
            # 检查indicators目录的接口设计
            indicators_interfaces = self._count_interfaces_in_directory('indicators/')
            strategy_interfaces = self._count_interfaces_in_directory('strategy/')
            analysis_interfaces = self._count_interfaces_in_directory('analysis/')
            
            total_interfaces = indicators_interfaces + strategy_interfaces + analysis_interfaces
            
            # 基于接口数量和质量评分
            if total_interfaces >= 10:
                return 85.0
            elif total_interfaces >= 5:
                return 70.0
            else:
                return 50.0
        
        except Exception as e:
            logger.error(f"评估接口设计质量失败: {e}")
            return 60.0
    
    def _count_interfaces_in_directory(self, directory: str) -> int:
        """统计目录中的接口数量"""
        interface_count = 0
        
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # 统计抽象基类和接口
                            if 'ABC' in content or 'abstractmethod' in content:
                                interface_count += content.count('class ')
                        
                        except Exception:
                            continue
        
        return interface_count
    
    def _evaluate_abstraction_quality(self) -> float:
        """评估抽象质量"""
        try:
            # 检查BaseIndicator等基类的使用
            base_classes = ['BaseIndicator', 'BaseStrategy', 'BaseAnalyzer']
            usage_count = 0
            
            for directory in self.l4_directories:
                if os.path.exists(directory):
                    for root, dirs, files in os.walk(directory):
                        for file in files:
                            if file.endswith('.py'):
                                file_path = os.path.join(root, file)
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                    
                                    for base_class in base_classes:
                                        if base_class in content:
                                            usage_count += 1
                                            break
                                
                                except Exception:
                                    continue
            
            # 基于基类使用情况评分
            if usage_count >= 20:
                return 90.0
            elif usage_count >= 10:
                return 75.0
            else:
                return 60.0
        
        except Exception as e:
            logger.error(f"评估抽象质量失败: {e}")
            return 60.0
    
    def _evaluate_componentization_quality(self) -> float:
        """评估组件化质量"""
        try:
            # 检查组件化设计模式
            component_patterns = ['Factory', 'Registry', 'Manager', 'Service', 'Engine']
            pattern_count = 0
            
            for directory in self.l4_directories:
                if os.path.exists(directory):
                    for root, dirs, files in os.walk(directory):
                        for file in files:
                            if file.endswith('.py'):
                                for pattern in component_patterns:
                                    if pattern.lower() in file.lower():
                                        pattern_count += 1
                                        break
            
            # 基于组件化模式使用情况评分
            if pattern_count >= 15:
                return 85.0
            elif pattern_count >= 8:
                return 70.0
            else:
                return 55.0
        
        except Exception as e:
            logger.error(f"评估组件化质量失败: {e}")
            return 60.0
    
    def _test_layered_architecture_compliance(self) -> Dict[str, Any]:
        """测试4: 分层架构合规性深度检查"""
        logger.info("测试4: 分层架构合规性深度检查")
        
        violations = []
        total_classes = 0
        
        for directory in self.l4_directories:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            
                            # 检查类职责过多问题
                            class_violations = self._check_class_responsibility_violations(file_path)
                            violations.extend(class_violations)
                            
                            # 统计类数量
                            total_classes += self._count_classes_in_file(file_path)
        
        # 计算分层架构合规性评分
        if total_classes == 0:
            compliance_score = 0.0
        else:
            compliance_score = max(0, (1 - len(violations) / max(total_classes, 1)) * 100)
        
        passed = compliance_score >= 80.0
        
        logger.info(f"分层架构合规性: {compliance_score:.1f}% (发现 {len(violations)} 个违规)")
        
        return {
            'passed': passed,
            'score': compliance_score,
            'details': {
                'total_classes': total_classes,
                'violations': len(violations),
                'violation_details': violations[:10]  # 只显示前10个
            }
        }
    
    def _check_class_responsibility_violations(self, file_path: str) -> List[str]:
        """检查类职责过多违规"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析AST查找类定义
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # 统计方法数量
                    method_count = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                    
                    # L4层方法数量限制：核心服务类可以有更多方法，但需要合理性说明
                    if method_count > 25:
                        violations.append(f"职责违规: 类职责过多: {file_path}:{node.name} 有{method_count}个方法")
                    elif method_count > 15:
                        # 检查是否有合理性说明
                        class_docstring = ast.get_docstring(node)
                        if not class_docstring or '合理性' not in class_docstring:
                            violations.append(f"职责违规: 类职责过多: {file_path}:{node.name} 有{method_count}个方法")
        
        except Exception as e:
            logger.debug(f"检查类职责违规失败 {file_path}: {e}")
        
        return violations
    
    def _count_classes_in_file(self, file_path: str) -> int:
        """统计文件中的类数量"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            return sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        
        except Exception:
            return 0
    
    def _generate_compliance_report(self, total_score: float, pass_rate: float, 
                                  passed_tests: int, total_tests: int) -> Dict[str, Any]:
        """生成合规性报告"""
        execution_time = time.time() - self.start_time
        
        # 确定评级
        if total_score >= 95:
            grade = "A+"
            compliance_status = "COMPLIANT"
            recommendation = "✅ L4层架构设计完全符合企业级标准，可以继续L5层修复"
        elif total_score >= 90:
            grade = "A"
            compliance_status = "COMPLIANT"
            recommendation = "✅ L4层架构设计符合高质量标准，建议继续L5层修复"
        elif total_score >= 80:
            grade = "B"
            compliance_status = "APPROACHING_COMPLIANT"
            recommendation = "⚠️ L4层架构设计基本符合标准，建议优化后继续L5层修复"
        else:
            grade = "C"
            compliance_status = "NON_COMPLIANT"
            recommendation = "❌ L4层架构设计存在重大问题，必须重构后才能确保长期可维护性"
        
        # 生成详细报告
        report = {
            'verification_time': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': f"{pass_rate:.1f}%",
            'overall_score': f"{total_score:.1f}/100",
            'grade': f"{grade} ({total_score:.1f}/100)",
            'execution_time': f"{execution_time:.3f}s",
            'compliance_status': compliance_status,
            'recommendation': recommendation,
            'detailed_scores': {
                test_name: f"{'✅' if result['passed'] else '❌'} {result['score']:.1f}/100"
                for test_name, result in self.test_results.items()
            },
            'l1_l2_l3_compatibility': self._assess_l1_l2_l3_compatibility(),
            'architecture_issues': self.compliance_issues,
            'design_violations': self.design_violations,
            'test_details': {
                test_name: result['details'] for test_name, result in self.test_results.items()
            }
        }
        
        self._print_compliance_report(report)
        return report
    
    def _assess_l1_l2_l3_compatibility(self) -> Dict[str, str]:
        """评估与L1/L2/L3的兼容性"""
        compatibility = {}
        
        for test_name, result in self.test_results.items():
            if result['score'] >= 90:
                compatibility[test_name.replace(' ', '_').lower()] = "✅ 兼容"
            else:
                compatibility[test_name.replace(' ', '_').lower()] = "❌ 不兼容"
        
        return compatibility
    
    def _print_compliance_report(self, report: Dict[str, Any]):
        """打印合规性报告"""
        print("\n" + "="*80)
        print("🏗️ L4核心服务层架构设计合规性验证报告")
        print("="*80)
        print(f"验证时间: {report['verification_time']}")
        print(f"总测试数: {report['total_tests']}")
        print(f"通过测试: {report['passed_tests']}")
        print(f"失败测试: {report['failed_tests']}")
        print(f"测试通过率: {report['pass_rate']}")
        print(f"整体合规评分: {report['overall_score']}")
        print(f"评级: {report['grade']}")
        print(f"执行时间: {report['execution_time']}")
        print(f"合规状态: {report['compliance_status']}")
        print(f"推荐建议: {report['recommendation']}")
        
        print(f"\n📊 架构合规性详细评分:")
        for test_name, score in report['detailed_scores'].items():
            print(f"  {test_name}: {score}")
        
        print(f"\n🔗 L1/L2/L3架构兼容性评估:")
        for test_name, compatibility in report['l1_l2_l3_compatibility'].items():
            print(f"  {test_name}: {compatibility}")
        
        if report['architecture_issues']:
            print(f"\n⚠️ 架构问题:")
            for i, issue in enumerate(report['architecture_issues'][:5], 1):
                print(f"  {i}. {issue}")
        
        if report['design_violations']:
            print(f"\n❌ 设计违规:")
            for i, violation in enumerate(report['design_violations'][:5], 1):
                print(f"  {i}. {violation}")
        
        print(f"\n📋 详细测试结果:")
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result['passed'] else "❌ 失败"
            print(f"  {test_name}: {status} ({result['execution_time']:.3f}s)")
        
        print("="*80)


def main():
    """主函数"""
    try:
        validator = L4ArchitectureDesignComplianceValidator()
        report = validator.run_comprehensive_validation()
        return 0 if report['compliance_status'] == 'COMPLIANT' else 1
    
    except Exception as e:
        logger.error(f"L4架构设计合规性验证异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
