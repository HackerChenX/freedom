#!/usr/bin/env python3
"""
L3数据服务层架构设计合规性验证脚本

基于L1/L2架构合规审计标准（第2.6节配置管理入口统一标准），
对L3层进行深度架构设计合规性验证
"""

import sys
import os
import ast
import time
import importlib.util
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
import re

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger
from utils.enhanced_exception_handler import exception_handler
from utils.enhanced_performance_monitor import performance_monitor

logger = get_logger(__name__)


class L3ArchitectureDesignComplianceValidator:
    """L3数据服务层架构设计合规性验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.test_results = {}
        self.architecture_issues = []
        self.redundant_entries = []
        self.deprecated_files = []
        self.design_violations = []
        self.compliance_scores = {}
        self.start_time = time.time()
        
        # L3层标准入口定义（基于L1/L2单一入口原则）
        self.l3_standard_entries = {
            'cache_service': 'db.services.cache_service.CacheService',
            'data_access_manager': 'db.managers.data_access_manager.DataAccessManager',
            'service_registry': 'db.service_registry.ServiceRegistry',
            'parallel_processor': 'db.parallel_processor.ParallelProcessor'
        }
        
        # L3层接口定义
        self.l3_interfaces = {
            'cache_interface': 'db.interfaces.cache_interface.ICacheService',
            'data_access_interface': 'db.interfaces.data_access_interface.IDataAccess',
            'connection_interface': 'db.interfaces.connection_interface.IConnectionManager'
        }
        
        logger.info("=== L3数据服务层架构设计合规性验证开始 ===")
    
    @exception_handler(reraise=False, default_return=False)
    def test_single_entry_point_compliance(self) -> bool:
        """测试单一入口原则合规性"""
        logger.info("测试1: 单一入口原则合规性")
        
        try:
            # 获取L3层所有Python文件
            l3_files = self._get_l3_python_files()
            
            # 分析功能入口点
            entry_points = defaultdict(list)
            duplicate_entries = []
            
            for file_path in l3_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 分析类定义
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            class_name = node.name
                            
                            # 检查是否是服务类
                            if any(suffix in class_name for suffix in ['Service', 'Manager', 'Registry', 'Processor']):
                                full_path = f"{file_path}:{class_name}"
                                entry_points[class_name].append(full_path)
                    
                except Exception as e:
                    logger.warning(f"无法分析文件 {file_path}: {e}")
            
            # 检查重复入口
            for class_name, paths in entry_points.items():
                if len(paths) > 1:
                    duplicate_entries.append({
                        'class_name': class_name,
                        'paths': paths,
                        'count': len(paths)
                    })
            
            # 检查是否存在功能重复的服务
            functional_duplicates = self._detect_functional_duplicates(entry_points)
            
            # 计算合规性评分
            total_services = len(entry_points)
            duplicate_count = len(duplicate_entries) + len(functional_duplicates)
            compliance_rate = ((total_services - duplicate_count) / total_services * 100) if total_services > 0 else 100
            
            self.compliance_scores['single_entry'] = compliance_rate
            
            if duplicate_entries:
                for dup in duplicate_entries:
                    self.architecture_issues.append(f"重复入口: {dup['class_name']} 存在 {dup['count']} 个定义")
                    self.redundant_entries.extend(dup['paths'][1:])  # 除第一个外都是冗余的
            
            if functional_duplicates:
                for dup in functional_duplicates:
                    self.architecture_issues.append(f"功能重复: {dup}")
            
            logger.info(f"单一入口原则合规性: {compliance_rate:.1f}% (发现 {duplicate_count} 个重复入口)")
            
            return compliance_rate >= 95  # A+级标准要求95%以上合规
            
        except Exception as e:
            self.architecture_issues.append(f"单一入口原则检查异常: {e}")
            return False
    
    @exception_handler(reraise=False, default_return=False)
    def test_deprecated_entries_cleanup(self) -> bool:
        """测试废弃入口和脚本清理"""
        logger.info("测试2: 废弃入口和脚本清理验证")
        
        try:
            l3_files = self._get_l3_python_files()
            
            # 检查废弃文件模式（基于L1/L2修复经验）
            deprecated_patterns = [
                r'.*_old\.py$',
                r'.*_backup\.py$',
                r'.*_deprecated\.py$',
                r'.*_legacy\.py$',
                r'.*_temp\.py$',
                r'.*_test\.py$',  # 测试文件不应在生产代码中
                r'.*_broken\.py$'
            ]
            
            deprecated_files = []
            unused_imports = []
            dead_code_files = []
            
            for file_path in l3_files:
                # 检查文件名模式
                for pattern in deprecated_patterns:
                    if re.match(pattern, os.path.basename(file_path)):
                        deprecated_files.append(file_path)
                        break
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查未使用的导入
                    unused = self._detect_unused_imports(file_path, content)
                    if unused:
                        unused_imports.extend(unused)
                    
                    # 检查死代码
                    if self._is_dead_code_file(file_path, content):
                        dead_code_files.append(file_path)
                        
                except Exception as e:
                    logger.warning(f"无法检查废弃入口 {file_path}: {e}")
            
            # 检查接口实现完整性
            interface_issues = self._check_interface_implementation_completeness()
            
            total_issues = len(deprecated_files) + len(unused_imports) + len(dead_code_files) + len(interface_issues)
            total_files = len(l3_files)
            cleanup_rate = ((total_files - total_issues) / total_files * 100) if total_files > 0 else 100
            
            self.compliance_scores['cleanup'] = cleanup_rate
            self.deprecated_files = deprecated_files
            
            if deprecated_files:
                for file in deprecated_files:
                    self.architecture_issues.append(f"废弃文件: {file}")
            
            if unused_imports:
                for imp in unused_imports:
                    self.architecture_issues.append(f"未使用导入: {imp}")
            
            if dead_code_files:
                for file in dead_code_files:
                    self.architecture_issues.append(f"死代码文件: {file}")
            
            if interface_issues:
                for issue in interface_issues:
                    self.architecture_issues.append(f"接口实现问题: {issue}")
            
            logger.info(f"废弃入口清理合规性: {cleanup_rate:.1f}% (发现 {total_issues} 个问题)")
            
            return cleanup_rate >= 95
            
        except Exception as e:
            self.architecture_issues.append(f"废弃入口清理检查异常: {e}")
            return False
    
    @exception_handler(reraise=False, default_return=False)
    def test_architecture_extensibility(self) -> bool:
        """测试架构扩展性和通用性"""
        logger.info("测试3: 架构扩展性和通用性评估")
        
        try:
            # 检查接口设计质量
            interface_quality = self._evaluate_interface_design_quality()
            
            # 检查组件耦合度
            coupling_analysis = self._analyze_component_coupling()
            
            # 检查抽象层次
            abstraction_quality = self._evaluate_abstraction_quality()
            
            # 检查扩展点设计
            extension_points = self._analyze_extension_points()
            
            # 综合评分
            extensibility_score = (
                interface_quality['score'] * 0.3 +
                coupling_analysis['score'] * 0.3 +
                abstraction_quality['score'] * 0.2 +
                extension_points['score'] * 0.2
            )
            
            self.compliance_scores['extensibility'] = extensibility_score
            
            # 记录发现的问题
            for category, analysis in [
                ('接口设计', interface_quality),
                ('组件耦合', coupling_analysis),
                ('抽象质量', abstraction_quality),
                ('扩展点', extension_points)
            ]:
                for issue in analysis.get('issues', []):
                    self.design_violations.append(f"{category}问题: {issue}")
            
            logger.info(f"架构扩展性评分: {extensibility_score:.1f}/100")
            
            return extensibility_score >= 85  # 扩展性要求相对较高
            
        except Exception as e:
            self.architecture_issues.append(f"架构扩展性检查异常: {e}")
            return False
    
    @exception_handler(reraise=False, default_return=False)
    def test_layered_architecture_compliance(self) -> bool:
        """测试分层架构合规性"""
        logger.info("测试4: 分层架构合规性深度检查")
        
        try:
            l3_files = self._get_l3_python_files()
            
            # 检查分层违规
            layering_violations = []
            dependency_cycles = []
            interface_violations = []
            responsibility_violations = []
            
            for file_path in l3_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查跨层导入
                    cross_layer_imports = self._detect_cross_layer_imports(file_path, content)
                    layering_violations.extend(cross_layer_imports)
                    
                    # 检查循环依赖
                    cycles = self._detect_dependency_cycles(file_path, content)
                    dependency_cycles.extend(cycles)
                    
                    # 检查接口违规
                    interface_issues = self._detect_interface_violations(file_path, content)
                    interface_violations.extend(interface_issues)
                    
                    # 检查职责违规
                    responsibility_issues = self._detect_responsibility_violations(file_path, content)
                    responsibility_violations.extend(responsibility_issues)
                    
                except Exception as e:
                    logger.warning(f"无法检查分层架构 {file_path}: {e}")
            
            total_violations = (
                len(layering_violations) + 
                len(dependency_cycles) + 
                len(interface_violations) + 
                len(responsibility_violations)
            )
            
            total_files = len(l3_files)
            compliance_rate = ((total_files - total_violations) / total_files * 100) if total_files > 0 else 100
            
            self.compliance_scores['layered_architecture'] = compliance_rate
            
            # 记录违规
            for violation in layering_violations:
                self.design_violations.append(f"分层违规: {violation}")
            
            for cycle in dependency_cycles:
                self.design_violations.append(f"循环依赖: {cycle}")
            
            for violation in interface_violations:
                self.design_violations.append(f"接口违规: {violation}")
            
            for violation in responsibility_violations:
                self.design_violations.append(f"职责违规: {violation}")
            
            logger.info(f"分层架构合规性: {compliance_rate:.1f}% (发现 {total_violations} 个违规)")
            
            return compliance_rate >= 95
            
        except Exception as e:
            self.architecture_issues.append(f"分层架构合规性检查异常: {e}")
            return False
    
    def _get_l3_python_files(self) -> List[str]:
        """获取L3层所有Python文件"""
        l3_files = []
        l3_directories = ['db/services', 'db/managers', 'db/interfaces']
        
        for directory in l3_directories:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.py') and not file.startswith('__'):
                            l3_files.append(os.path.join(root, file))
        
        # 添加其他L3相关文件
        additional_files = [
            'db/service_registry.py',
            'db/parallel_processor.py'
        ]
        
        for file_path in additional_files:
            if os.path.exists(file_path):
                l3_files.append(file_path)
        
        return l3_files
    
    def _detect_functional_duplicates(self, entry_points: Dict[str, List[str]]) -> List[str]:
        """检测功能重复的服务"""
        functional_duplicates = []
        
        # 检查功能相似的类名
        similar_patterns = [
            (['CacheService', 'CacheManager'], '缓存服务'),
            (['DataAccessManager', 'DataManager', 'DatabaseManager'], '数据访问'),
            (['ServiceRegistry', 'ServiceManager'], '服务注册'),
        ]
        
        for patterns, function_name in similar_patterns:
            found_classes = []
            for pattern in patterns:
                if pattern in entry_points:
                    found_classes.extend(entry_points[pattern])
            
            if len(found_classes) > 1:
                functional_duplicates.append(f"{function_name}功能重复: {found_classes}")
        
        return functional_duplicates
    
    def _detect_unused_imports(self, file_path: str, content: str) -> List[str]:
        """检测未使用的导入"""
        unused_imports = []
        
        try:
            tree = ast.parse(content)
            
            # 收集所有导入
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.append(alias.name)
            
            # 检查是否在代码中使用
            for imp in imports:
                if imp not in content.replace(f'import {imp}', '').replace(f'from {imp}', ''):
                    unused_imports.append(f"{file_path}:{imp}")
        
        except Exception:
            pass  # 语法错误等情况忽略
        
        return unused_imports
    
    def _is_dead_code_file(self, file_path: str, content: str) -> bool:
        """检查是否是死代码文件"""
        # 检查文件是否只有导入和注释
        try:
            tree = ast.parse(content)
            
            # 统计有效代码行
            effective_nodes = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)):
                    effective_nodes += 1
            
            return effective_nodes == 0
            
        except Exception:
            return False
    
    def _check_interface_implementation_completeness(self) -> List[str]:
        """检查接口实现完整性"""
        issues = []
        
        # 检查ICacheService实现
        try:
            from db.interfaces.cache_interface import ICacheService
            from db.services.cache_service import CacheService
            
            cache_service = CacheService()
            if not isinstance(cache_service, ICacheService):
                issues.append("CacheService未正确实现ICacheService接口")
        except Exception as e:
            issues.append(f"缓存服务接口检查失败: {e}")
        
        # 检查IDataAccess实现
        try:
            from db.interfaces.data_access_interface import IDataAccess
            from db.managers.data_access_manager import DataAccessManager
            
            data_manager = DataAccessManager()
            if not isinstance(data_manager, IDataAccess):
                issues.append("DataAccessManager未正确实现IDataAccess接口")
        except Exception as e:
            issues.append(f"数据访问接口检查失败: {e}")
        
        return issues
    
    def _evaluate_interface_design_quality(self) -> Dict[str, Any]:
        """评估接口设计质量"""
        quality_score = 85  # 基础分
        issues = []
        
        try:
            # 检查ICacheService接口
            from db.interfaces.cache_interface import ICacheService
            
            # 检查方法数量（过多表示接口过于复杂）
            cache_methods = [method for method in dir(ICacheService) if not method.startswith('_')]
            if len(cache_methods) > 20:
                issues.append(f"ICacheService接口过于复杂，包含{len(cache_methods)}个方法")
                quality_score -= 10
            
            # 检查方法命名一致性
            naming_inconsistencies = self._check_method_naming_consistency(cache_methods)
            if naming_inconsistencies:
                issues.extend(naming_inconsistencies)
                quality_score -= 5
                
        except Exception as e:
            issues.append(f"接口设计质量检查失败: {e}")
            quality_score -= 20
        
        return {'score': quality_score, 'issues': issues}
    
    def _analyze_component_coupling(self) -> Dict[str, Any]:
        """分析组件耦合度"""
        coupling_score = 90  # 基础分
        issues = []
        
        l3_files = self._get_l3_python_files()
        
        # 分析导入关系
        import_graph = defaultdict(set)
        
        for file_path in l3_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取导入的L3层模块
                for line in content.split('\n'):
                    if line.strip().startswith('from db.') and 'import' in line:
                        imported_module = line.split('from')[1].split('import')[0].strip()
                        if imported_module.startswith('db.'):
                            import_graph[file_path].add(imported_module)
                            
            except Exception:
                continue
        
        # 检查高耦合
        for file_path, imports in import_graph.items():
            if len(imports) > 5:  # 导入超过5个L3模块认为高耦合
                issues.append(f"高耦合文件: {file_path} 导入了{len(imports)}个L3模块")
                coupling_score -= 5
        
        return {'score': max(coupling_score, 0), 'issues': issues}
    
    def _evaluate_abstraction_quality(self) -> Dict[str, Any]:
        """评估抽象质量"""
        abstraction_score = 88  # 基础分
        issues = []
        
        # 检查抽象基类使用
        try:
            from db.interfaces.cache_interface import ICacheService
            from db.interfaces.data_access_interface import IDataAccess
            
            # 检查是否正确使用ABC
            import inspect
            if not inspect.isabstract(ICacheService):
                issues.append("ICacheService应该是抽象基类")
                abstraction_score -= 10
            
            if not inspect.isabstract(IDataAccess):
                issues.append("IDataAccess应该是抽象基类")
                abstraction_score -= 10
                
        except Exception as e:
            issues.append(f"抽象质量检查失败: {e}")
            abstraction_score -= 15
        
        return {'score': max(abstraction_score, 0), 'issues': issues}
    
    def _analyze_extension_points(self) -> Dict[str, Any]:
        """分析扩展点设计"""
        extension_score = 85  # 基础分
        issues = []
        
        # 检查是否有良好的扩展机制
        l3_files = self._get_l3_python_files()
        
        extension_patterns = 0
        for file_path in l3_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查扩展模式
                if 'register' in content.lower():
                    extension_patterns += 1
                if 'plugin' in content.lower():
                    extension_patterns += 1
                if 'factory' in content.lower():
                    extension_patterns += 1
                    
            except Exception:
                continue
        
        if extension_patterns < 2:
            issues.append("缺少足够的扩展点设计模式")
            extension_score -= 15
        
        return {'score': extension_score, 'issues': issues}
    
    def _detect_cross_layer_imports(self, file_path: str, content: str) -> List[str]:
        """检测跨层导入"""
        violations = []
        
        # L3层不应导入L4/L5/L6层
        forbidden_imports = ['from analysis', 'from strategy', 'from indicators', 'from api', 'from bin']
        
        for line in content.split('\n'):
            for forbidden in forbidden_imports:
                if line.strip().startswith(forbidden):
                    violations.append(f"{file_path}: {line.strip()}")
        
        return violations
    
    def _detect_dependency_cycles(self, file_path: str, content: str) -> List[str]:
        """检测循环依赖"""
        cycles = []
        
        # 简化的循环依赖检测
        # 实际实现需要构建完整的依赖图
        if 'from db.services' in content and 'db/services/' in file_path:
            # 服务间相互导入可能形成循环
            for line in content.split('\n'):
                if line.strip().startswith('from db.services') and 'import' in line:
                    cycles.append(f"可能的循环依赖: {file_path} -> {line.strip()}")
        
        return cycles
    
    def _detect_interface_violations(self, file_path: str, content: str) -> List[str]:
        """检测接口违规"""
        violations = []
        
        # 检查是否绕过接口直接访问实现
        if 'from db.services' in content and 'interfaces' not in file_path:
            violations.append(f"可能绕过接口: {file_path}")
        
        return violations
    
    def _detect_responsibility_violations(self, file_path: str, content: str) -> List[str]:
        """检测职责违规"""
        violations = []
        
        # 检查单一职责原则
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # 检查类是否承担过多职责
                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    if len(methods) > 15:  # 方法过多可能违反单一职责
                        violations.append(f"类职责过多: {file_path}:{node.name} 有{len(methods)}个方法")
        except Exception:
            pass
        
        return violations
    
    def _check_method_naming_consistency(self, methods: List[str]) -> List[str]:
        """检查方法命名一致性"""
        inconsistencies = []
        
        # 检查命名模式
        get_methods = [m for m in methods if m.startswith('get_')]
        set_methods = [m for m in methods if m.startswith('set_')]
        
        # 检查get/set配对
        for get_method in get_methods:
            expected_set = get_method.replace('get_', 'set_')
            if expected_set not in set_methods:
                inconsistencies.append(f"缺少配对的set方法: {expected_set}")
        
        return inconsistencies
    
    @performance_monitor(threshold_seconds=30.0)
    def run_comprehensive_architecture_compliance_validation(self) -> Dict[str, Any]:
        """运行全面架构设计合规性验证"""
        logger.info("开始L3数据服务层架构设计合规性验证")
        
        # 定义测试用例
        test_cases = [
            ("单一入口原则合规性", self.test_single_entry_point_compliance),
            ("废弃入口和脚本清理", self.test_deprecated_entries_cleanup),
            ("架构扩展性和通用性", self.test_architecture_extensibility),
            ("分层架构合规性", self.test_layered_architecture_compliance),
        ]
        
        # 执行测试
        passed_tests = 0
        total_tests = len(test_cases)
        
        for test_name, test_func in test_cases:
            logger.info(f"执行测试: {test_name}")
            start_time = time.time()
            
            try:
                result = test_func()
                execution_time = time.time() - start_time
                
                self.test_results[test_name] = {
                    'passed': result,
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                if result:
                    passed_tests += 1
                    logger.info(f"✅ {test_name} - 通过 ({execution_time:.3f}s)")
                else:
                    logger.error(f"❌ {test_name} - 失败 ({execution_time:.3f}s)")
                    
            except Exception as e:
                execution_time = time.time() - start_time
                self.test_results[test_name] = {
                    'passed': False,
                    'execution_time': execution_time,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                logger.error(f"❌ {test_name} - 异常: {e}")
        
        # 计算总体架构合规性评分
        overall_compliance = sum(self.compliance_scores.values()) / len(self.compliance_scores) if self.compliance_scores else 0
        pass_rate = (passed_tests / total_tests) * 100
        total_time = time.time() - self.start_time
        
        # 评级计算（基于L1/L2 A+级标准）
        if overall_compliance >= 95 and pass_rate == 100:
            grade = "A+"
            score = 95 + (overall_compliance - 95) * 5
        elif overall_compliance >= 90 and pass_rate >= 75:
            grade = "A"
            score = 90 + (overall_compliance - 90) * 5 / 5
        elif overall_compliance >= 80:
            grade = "B"
            score = 80 + (overall_compliance - 80) * 10 / 10
        else:
            grade = "C"
            score = overall_compliance
        
        # 生成架构设计合规性报告
        compliance_report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': pass_rate,
            'overall_compliance_score': round(overall_compliance, 1),
            'grade': grade,
            'score': round(score, 1),
            'total_execution_time': round(total_time, 3),
            'compliance_scores': self.compliance_scores,
            'test_results': self.test_results,
            'architecture_issues': self.architecture_issues,
            'redundant_entries': self.redundant_entries,
            'deprecated_files': self.deprecated_files,
            'design_violations': self.design_violations,
            'validation_status': 'COMPLIANT' if pass_rate == 100 and overall_compliance >= 90 else 'NON_COMPLIANT',
            'l1_l2_compatibility': self._assess_l1_l2_architecture_compatibility(),
            'recommendation': self._get_architecture_recommendation(overall_compliance, pass_rate)
        }
        
        return compliance_report
    
    def _assess_l1_l2_architecture_compatibility(self) -> Dict[str, Any]:
        """评估与L1/L2架构的兼容性"""
        return {
            'single_entry_compatibility': self.compliance_scores.get('single_entry', 0) >= 95,
            'cleanup_compatibility': self.compliance_scores.get('cleanup', 0) >= 95,
            'extensibility_compatibility': self.compliance_scores.get('extensibility', 0) >= 85,
            'layered_architecture_compatibility': self.compliance_scores.get('layered_architecture', 0) >= 95
        }
    
    def _get_architecture_recommendation(self, compliance_score: float, pass_rate: float) -> str:
        """获取架构推荐建议"""
        if compliance_score >= 95 and pass_rate == 100:
            return "✅ L3层架构设计完全符合L1/L2标准，达到A+级质量水平"
        elif compliance_score >= 90 and pass_rate >= 75:
            return "⚠️ L3层架构设计基本符合标准，建议修复剩余设计问题"
        else:
            return "❌ L3层架构设计存在重大问题，必须重构后才能确保长期可维护性"


def main():
    """主函数"""
    try:
        validator = L3ArchitectureDesignComplianceValidator()
        report = validator.run_comprehensive_architecture_compliance_validation()
        
        # 输出架构设计合规性报告
        print("\n" + "="*80)
        print("🏗️ L3数据服务层架构设计合规性验证报告")
        print("="*80)
        print(f"验证时间: {report['timestamp']}")
        print(f"总测试数: {report['total_tests']}")
        print(f"通过测试: {report['passed_tests']}")
        print(f"失败测试: {report['failed_tests']}")
        print(f"测试通过率: {report['pass_rate']:.1f}%")
        print(f"整体合规评分: {report['overall_compliance_score']}/100")
        print(f"评级: {report['grade']} ({report['score']}/100)")
        print(f"执行时间: {report['total_execution_time']}s")
        print(f"合规状态: {report['validation_status']}")
        print(f"推荐建议: {report['recommendation']}")
        
        print("\n📊 架构合规性详细评分:")
        for dimension, score in report['compliance_scores'].items():
            status = "✅" if score >= 90 else "⚠️" if score >= 80 else "❌"
            print(f"  {dimension}: {status} {score:.1f}/100")
        
        print("\n🔗 L1/L2架构兼容性评估:")
        compatibility = report['l1_l2_compatibility']
        for component, is_compatible in compatibility.items():
            status = "✅ 兼容" if is_compatible else "❌ 不兼容"
            print(f"  {component}: {status}")
        
        if report['architecture_issues']:
            print("\n⚠️ 架构问题:")
            for i, issue in enumerate(report['architecture_issues'], 1):
                print(f"  {i}. {issue}")
        
        if report['redundant_entries']:
            print("\n🔄 冗余入口:")
            for i, entry in enumerate(report['redundant_entries'], 1):
                print(f"  {i}. {entry}")
        
        if report['deprecated_files']:
            print("\n🗑️ 废弃文件:")
            for i, file in enumerate(report['deprecated_files'], 1):
                print(f"  {i}. {file}")
        
        if report['design_violations']:
            print("\n❌ 设计违规:")
            for i, violation in enumerate(report['design_violations'], 1):
                print(f"  {i}. {violation}")
        
        print("\n📋 详细测试结果:")
        for test_name, result in report['test_results'].items():
            status = "✅ 通过" if result['passed'] else "❌ 失败"
            print(f"  {test_name}: {status} ({result['execution_time']:.3f}s)")
        
        print("="*80)
        
        # 返回适当的退出码
        return 0 if report['validation_status'] == 'COMPLIANT' else 1
        
    except Exception as e:
        logger.error(f"架构设计合规性验证过程发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
