#!/usr/bin/env python3
"""
命名规范验证工具

检查代码命名规范，包括：
1. 类名大驼峰命名
2. 方法名小写+下划线
3. 变量名小写+下划线
4. 常量名大写+下划线
5. 模块名小写+下划线
6. 检测命名冲突和重复
"""

import os
import sys
import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class NamingViolation:
    """命名规范违规"""
    file_path: str
    line_number: int
    element_type: str  # 'class', 'method', 'variable', 'constant', 'module'
    element_name: str
    violation_type: str
    severity: str
    suggestion: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file': self.file_path,
            'line': self.line_number,
            'element_type': self.element_type,
            'element_name': self.element_name,
            'violation_type': self.violation_type,
            'severity': self.severity,
            'suggestion': self.suggestion
        }


class NamingConventionChecker:
    """命名规范检查器"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.violations = []
        
        # 命名规范模式
        self.patterns = {
            'class_name': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),  # 大驼峰
            'method_name': re.compile(r'^[a-z][a-z0-9_]*$'),   # 小写+下划线
            'variable_name': re.compile(r'^[a-z][a-z0-9_]*$'), # 小写+下划线
            'constant_name': re.compile(r'^[A-Z][A-Z0-9_]*$'), # 大写+下划线
            'module_name': re.compile(r'^[a-z][a-z0-9_]*$'),   # 小写+下划线
            'private_name': re.compile(r'^_[a-z][a-z0-9_]*$'), # 私有成员
            'protected_name': re.compile(r'^_[a-z][a-z0-9_]*$') # 保护成员
        }
        
        # 常见的违规模式
        self.violation_patterns = {
            'camelCase': re.compile(r'^[a-z][a-zA-Z0-9]*[A-Z]'),
            'PascalCase_wrong': re.compile(r'^[A-Z][a-zA-Z0-9]*[a-z]'),
            'SCREAMING_SNAKE_CASE': re.compile(r'^[A-Z][A-Z0-9_]*$'),
            'mixed_case': re.compile(r'^[a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*[a-z]'),
            'double_underscore': re.compile(r'.*__.*'),
            'single_char': re.compile(r'^[a-zA-Z]$'),
            'numbers_only': re.compile(r'^[0-9]+$'),
            'special_chars': re.compile(r'[^a-zA-Z0-9_]')
        }
        
        # 保留字和内建名称
        self.reserved_names = {
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del',
            'elif', 'else', 'except', 'exec', 'finally', 'for', 'from', 'global',
            'if', 'import', 'in', 'is', 'lambda', 'not', 'or', 'pass', 'print',
            'raise', 'return', 'try', 'while', 'with', 'yield', 'None', 'True',
            'False', '__name__', '__main__', '__file__', '__doc__', '__init__',
            'str', 'int', 'float', 'list', 'dict', 'tuple', 'set', 'bool'
        }
        
        # 常见缩写词典
        self.abbreviations = {
            'mgr': 'manager',
            'cfg': 'config',
            'db': 'database',
            'url': 'URL',
            'http': 'HTTP',
            'api': 'API',
            'json': 'JSON',
            'xml': 'XML',
            'sql': 'SQL',
            'csv': 'CSV',
            'id': 'identifier',
            'num': 'number',
            'str': 'string',
            'obj': 'object',
            'req': 'request',
            'resp': 'response',
            'auth': 'authentication',
            'calc': 'calculator',
            'util': 'utility',
            'impl': 'implementation'
        }
    
    def check_all_naming_conventions(self) -> List[NamingViolation]:
        """检查所有命名规范"""
        print("🔍 开始命名规范检查...")
        
        violations = []
        
        # 1. 检查模块命名
        module_violations = self._check_module_names()
        violations.extend(module_violations)
        
        # 2. 检查代码元素命名
        code_violations = self._check_code_elements()
        violations.extend(code_violations)
        
        # 3. 检查命名冲突
        conflict_violations = self._check_naming_conflicts()
        violations.extend(conflict_violations)
        
        # 4. 检查命名一致性
        consistency_violations = self._check_naming_consistency()
        violations.extend(consistency_violations)
        
        print(f"    发现 {len(violations)} 个命名问题")
        return violations
    
    def _check_module_names(self) -> List[NamingViolation]:
        """检查模块命名"""
        print("  📋 检查模块命名...")
        violations = []
        
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            module_name = py_file.stem
            relative_path = str(py_file.relative_to(self.root_dir))
            
            # 检查模块名规范
            if not self.patterns['module_name'].match(module_name):
                suggestion = self._suggest_module_name(module_name)
                violations.append(NamingViolation(
                    file_path=relative_path,
                    line_number=1,
                    element_type="module",
                    element_name=module_name,
                    violation_type="module_naming",
                    severity="medium",
                    suggestion=suggestion
                ))
        
        print(f"    发现 {len(violations)} 个模块命名问题")
        return violations
    
    def _check_code_elements(self) -> List[NamingViolation]:
        """检查代码元素命名"""
        print("  📋 检查代码元素命名...")
        violations = []
        
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                relative_path = str(py_file.relative_to(self.root_dir))
                
                # 遍历AST节点
                for node in ast.walk(tree):
                    node_violations = self._check_ast_node(node, relative_path)
                    violations.extend(node_violations)
                
            except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
                continue
        
        print(f"    发现 {len(violations)} 个代码元素命名问题")
        return violations
    
    def _check_ast_node(self, node: ast.AST, file_path: str) -> List[NamingViolation]:
        """检查AST节点命名"""
        violations = []
        
        if isinstance(node, ast.ClassDef):
            # 检查类名
            if not self.patterns['class_name'].match(node.name):
                suggestion = self._suggest_class_name(node.name)
                violations.append(NamingViolation(
                    file_path=file_path,
                    line_number=node.lineno,
                    element_type="class",
                    element_name=node.name,
                    violation_type="class_naming",
                    severity="high",
                    suggestion=suggestion
                ))
        
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # 检查方法/函数名
            if not node.name.startswith('_') and not self.patterns['method_name'].match(node.name):
                suggestion = self._suggest_method_name(node.name)
                violations.append(NamingViolation(
                    file_path=file_path,
                    line_number=node.lineno,
                    element_type="method",
                    element_name=node.name,
                    violation_type="method_naming",
                    severity="medium",
                    suggestion=suggestion
                ))
            
            # 检查参数名
            for arg in node.args.args:
                if not self.patterns['variable_name'].match(arg.arg) and arg.arg not in ['self', 'cls']:
                    suggestion = self._suggest_variable_name(arg.arg)
                    violations.append(NamingViolation(
                        file_path=file_path,
                        line_number=node.lineno,
                        element_type="parameter",
                        element_name=arg.arg,
                        violation_type="parameter_naming",
                        severity="low",
                        suggestion=suggestion
                    ))
        
        elif isinstance(node, ast.Assign):
            # 检查变量名
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if self._is_constant_assignment(node):
                        # 常量检查
                        if not self.patterns['constant_name'].match(name):
                            suggestion = self._suggest_constant_name(name)
                            violations.append(NamingViolation(
                                file_path=file_path,
                                line_number=node.lineno,
                                element_type="constant",
                                element_name=name,
                                violation_type="constant_naming",
                                severity="medium",
                                suggestion=suggestion
                            ))
                    else:
                        # 变量检查
                        if not self.patterns['variable_name'].match(name):
                            suggestion = self._suggest_variable_name(name)
                            violations.append(NamingViolation(
                                file_path=file_path,
                                line_number=node.lineno,
                                element_type="variable",
                                element_name=name,
                                violation_type="variable_naming",
                                severity="low",
                                suggestion=suggestion
                            ))
        
        return violations
    
    def _check_naming_conflicts(self) -> List[NamingViolation]:
        """检查命名冲突"""
        print("  📋 检查命名冲突...")
        violations = []
        
        name_registry = defaultdict(list)
        
        # 收集所有名称
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                relative_path = str(py_file.relative_to(self.root_dir))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        name_registry[node.name].append({
                            'type': 'class',
                            'file': relative_path,
                            'line': node.lineno
                        })
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        name_registry[node.name].append({
                            'type': 'function',
                            'file': relative_path,
                            'line': node.lineno
                        })
                        
            except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
                continue
        
        # 检查冲突
        for name, occurrences in name_registry.items():
            if len(occurrences) > 1:
                # 检查是否是真正的冲突
                unique_files = set(occ['file'] for occ in occurrences)
                if len(unique_files) > 1:
                    for occ in occurrences:
                        violations.append(NamingViolation(
                            file_path=occ['file'],
                            line_number=occ['line'],
                            element_type=occ['type'],
                            element_name=name,
                            violation_type="naming_conflict",
                            severity="high",
                            suggestion=f"名称 '{name}' 与其他文件中的{occ['type']}冲突，建议使用更具体的名称"
                        ))
        
        print(f"    发现 {len(violations)} 个命名冲突")
        return violations
    
    def _check_naming_consistency(self) -> List[NamingViolation]:
        """检查命名一致性"""
        print("  📋 检查命名一致性...")
        violations = []
        
        # 检查常见的不一致模式
        inconsistent_patterns = [
            # 相同概念的不同表达
            (r'.*[Mm]anager.*', r'.*[Mm]gr.*', 'manager命名不一致'),
            (r'.*[Cc]onfig.*', r'.*[Cc]fg.*', 'config命名不一致'),
            (r'.*[Dd]atabase.*', r'.*[Dd]b.*', 'database命名不一致'),
            (r'.*[Cc]alculator.*', r'.*[Cc]alc.*', 'calculator命名不一致'),
            (r'.*[Uu]tility.*', r'.*[Uu]til.*', 'utility命名不一致'),
        ]
        
        all_names = []
        
        # 收集所有名称
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                relative_path = str(py_file.relative_to(self.root_dir))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        all_names.append({
                            'name': node.name,
                            'type': 'class',
                            'file': relative_path,
                            'line': node.lineno
                        })
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        all_names.append({
                            'name': node.name,
                            'type': 'function',
                            'file': relative_path,
                            'line': node.lineno
                        })
                        
            except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
                continue
        
        # 检查不一致性
        for pattern1, pattern2, message in inconsistent_patterns:
            group1 = [n for n in all_names if re.match(pattern1, n['name'], re.IGNORECASE)]
            group2 = [n for n in all_names if re.match(pattern2, n['name'], re.IGNORECASE)]
            
            if group1 and group2:
                # 发现不一致
                for item in group1 + group2:
                    violations.append(NamingViolation(
                        file_path=item['file'],
                        line_number=item['line'],
                        element_type=item['type'],
                        element_name=item['name'],
                        violation_type="naming_inconsistency",
                        severity="medium",
                        suggestion=f"{message}，建议统一命名风格"
                    ))
        
        print(f"    发现 {len(violations)} 个命名一致性问题")
        return violations
    
    def _is_constant_assignment(self, node: ast.Assign) -> bool:
        """判断是否是常量赋值"""
        if isinstance(node.value, (ast.Constant, ast.Str, ast.Num)):
            return True
        if isinstance(node.value, ast.Name) and node.value.id.isupper():
            return True
        return False
    
    def _suggest_class_name(self, name: str) -> str:
        """建议类名"""
        # 转换为大驼峰
        if '_' in name:
            parts = name.split('_')
            suggested = ''.join(part.capitalize() for part in parts)
        else:
            suggested = name.capitalize()
        
        return f"建议使用大驼峰命名: '{suggested}'"
    
    def _suggest_method_name(self, name: str) -> str:
        """建议方法名"""
        # 转换为小写+下划线
        suggested = re.sub(r'([A-Z])', r'_\1', name).lower().lstrip('_')
        return f"建议使用小写+下划线命名: '{suggested}'"
    
    def _suggest_variable_name(self, name: str) -> str:
        """建议变量名"""
        # 转换为小写+下划线
        suggested = re.sub(r'([A-Z])', r'_\1', name).lower().lstrip('_')
        return f"建议使用小写+下划线命名: '{suggested}'"
    
    def _suggest_constant_name(self, name: str) -> str:
        """建议常量名"""
        # 转换为大写+下划线
        suggested = re.sub(r'([a-z])([A-Z])', r'\1_\2', name).upper()
        return f"建议使用大写+下划线命名: '{suggested}'"
    
    def _suggest_module_name(self, name: str) -> str:
        """建议模块名"""
        # 转换为小写+下划线
        suggested = re.sub(r'([A-Z])', r'_\1', name).lower().lstrip('_')
        return f"建议使用小写+下划线命名: '{suggested}'"
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否应跳过文件"""
        skip_patterns = [
            '__pycache__',
            '.pyc',
            'test_',
            '_test.py',
            'tests/',
            'examples/',
            '.git/',
            'node_modules/',
            'venv/',
            '.env',
            'migration'
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def generate_report(self, violations: List[NamingViolation]) -> Dict[str, Any]:
        """生成检查报告"""
        total_violations = len(violations)
        high_severity = sum(1 for v in violations if v.severity == "high")
        medium_severity = sum(1 for v in violations if v.severity == "medium")
        low_severity = sum(1 for v in violations if v.severity == "low")
        
        # 按类型分组
        type_counts = defaultdict(int)
        for v in violations:
            type_counts[v.violation_type] += 1
        
        # 按元素类型分组
        element_counts = defaultdict(int)
        for v in violations:
            element_counts[v.element_type] += 1
        
        # 计算命名规范分数
        if total_violations == 0:
            naming_score = 100
        else:
            # 严重程度权重计算
            weighted_score = high_severity * 8 + medium_severity * 4 + low_severity * 1
            naming_score = max(0, 100 - weighted_score)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_violations': total_violations,
            'severity_breakdown': {
                'high': high_severity,
                'medium': medium_severity,
                'low': low_severity
            },
            'violation_type_breakdown': dict(type_counts),
            'element_type_breakdown': dict(element_counts),
            'naming_score': naming_score,
            'assessment': self._get_naming_assessment(naming_score),
            'violations': [v.to_dict() for v in violations]
        }
    
    def _get_naming_assessment(self, score: int) -> str:
        """获取命名规范评估"""
        if score >= 95:
            return "优秀"
        elif score >= 85:
            return "良好"
        elif score >= 75:
            return "中等"
        elif score >= 65:
            return "需要改进"
        else:
            return "命名规范差"


def main_naming_convention_checker():
    """主函数"""
    root_dir = os.getcwd()
    checker = NamingConventionChecker(root_dir)
    
    print("🚀 命名规范检查工具")
    print("=" * 50)
    
    # 执行检查
    violations = checker.check_all_naming_conventions()
    
    # 生成报告
    report = checker.generate_report(violations)
    
    # 输出结果
    print(f"\n📊 检查结果:")
    print(f"  总违规数: {report['total_violations']}")
    print(f"  严重违规: {report['severity_breakdown']['high']}")
    print(f"  中等违规: {report['severity_breakdown']['medium']}")
    print(f"  轻微违规: {report['severity_breakdown']['low']}")
    print(f"  命名规范分数: {report['naming_score']}/100")
    print(f"  评估等级: {report['assessment']}")
    
    # 按类型显示统计
    print(f"\n📋 违规类型统计:")
    for violation_type, count in report['violation_type_breakdown'].items():
        print(f"  {violation_type}: {count}")
    
    # 保存详细报告
    report_file = 'naming_convention_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 详细报告已保存到: {report_file}")
    
    # 输出一些具体的违规示例
    if violations:
        print(f"\n🔍 违规示例:")
        for i, violation in enumerate(violations[:5]):  # 只显示前5个
            print(f"\n  {i+1}. {violation.violation_type} - {violation.element_type}")
            print(f"     文件: {violation.file_path}:{violation.line_number}")
            print(f"     名称: {violation.element_name}")
            print(f"     建议: {violation.suggestion}")
    
    # 返回状态码
    if report['severity_breakdown']['high'] > 0:
        print(f"\n❌ 发现严重命名问题，建议立即修复")
        return 1
    elif report['naming_score'] < 85:
        print(f"\n⚠️  命名规范较差，建议改进")
        return 1
    else:
        print(f"\n✅ 命名规范检查通过")
        return 0


if __name__ == "__main__":
    sys.exit(main())