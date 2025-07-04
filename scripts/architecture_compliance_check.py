#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
架构合规性检查脚本
自动检测代码是否违反架构规则标准
"""

import os
import sys
import re
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

class ArchitectureComplianceChecker:
    """架构合规性检查器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.violations = defaultdict(list)
        self.layer_mapping = {
            'L6': ['bin', 'api'],
            'L5': ['strategy', 'analysis'],
            'L4': ['indicators', 'formula'],
            'L3': ['db/interfaces', 'db/managers'],
            'L2': ['db/clickhouse_db.py'],
            'L1': ['utils', 'config', 'enums']
        }
        
    def check_all_violations(self) -> Dict[str, List]:
        """检查所有架构违规"""
        print("开始架构合规性检查...")
        
        # 1. 检查分层架构违规
        self._check_layer_violations()
        
        # 2. 检查直接数据库依赖
        self._check_direct_db_dependencies()
        
        # 3. 检查代码重复
        self._check_code_duplication()
        
        # 4. 检查命名规范
        self._check_naming_conventions()
        
        # 5. 检查导入规范
        self._check_import_violations()
        
        # 6. 检查数据库查询规范
        self._check_database_query_violations()
        
        return dict(self.violations)
    
    def _check_layer_violations(self):
        """检查分层架构违规"""
        print("检查分层架构违规...")
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 获取文件所在层级
                file_layer = self._get_file_layer(py_file)
                if not file_layer:
                    continue
                
                # 检查导入的模块层级
                imports = self._extract_imports(content)
                for import_module in imports:
                    import_layer = self._get_module_layer(import_module)
                    if import_layer and self._is_layer_violation(file_layer, import_layer):
                        self.violations['layer_violations'].append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'file_layer': file_layer,
                            'import_module': import_module,
                            'import_layer': import_layer,
                            'violation': f'L{file_layer}层文件不能直接导入L{import_layer}层模块'
                        })
                        
            except Exception as e:
                print(f"检查文件 {py_file} 时出错: {e}")
    
    def _check_direct_db_dependencies(self):
        """检查直接数据库依赖违规"""
        print("检查直接数据库依赖...")
        
        prohibited_imports = [
            'from db.clickhouse_db import',
            'import db.clickhouse_db',
            'get_clickhouse_db'
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file) or 'db/' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for line_num, line in enumerate(content.split('\n'), 1):
                    for prohibited in prohibited_imports:
                        if prohibited in line:
                            self.violations['direct_db_dependencies'].append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': line_num,
                                'content': line.strip(),
                                'violation': f'禁止直接依赖数据库实现: {prohibited}'
                            })
                            
            except Exception as e:
                print(f"检查文件 {py_file} 时出错: {e}")
    
    def _check_code_duplication(self):
        """检查代码重复"""
        print("检查代码重复...")
        
        # 收集所有类名和方法名
        class_methods = defaultdict(list)
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析AST
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_methods[node.name].append(str(py_file.relative_to(self.project_root)))
                    elif isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_'):  # 忽略私有方法
                            class_methods[node.name].append(str(py_file.relative_to(self.project_root)))
                            
            except Exception as e:
                print(f"解析文件 {py_file} 时出错: {e}")
        
        # 检查重复
        for name, files in class_methods.items():
            if len(files) > 1:
                self.violations['code_duplication'].append({
                    'name': name,
                    'files': files,
                    'violation': f'发现重复的类/方法名: {name}'
                })
    
    def _check_naming_conventions(self):
        """检查命名规范"""
        print("检查命名规范...")
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # 检查类名（应该是大驼峰）
                        if not self._is_pascal_case(node.name):
                            self.violations['naming_violations'].append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'type': 'class',
                                'name': node.name,
                                'violation': '类名应使用大驼峰命名法'
                            })
                    
                    elif isinstance(node, ast.FunctionDef):
                        # 检查方法名（应该是小写+下划线）
                        if not self._is_snake_case(node.name) and not node.name.startswith('__'):
                            self.violations['naming_violations'].append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'type': 'function',
                                'name': node.name,
                                'violation': '方法名应使用小写+下划线命名法'
                            })
                            
            except Exception as e:
                print(f"检查文件 {py_file} 时出错: {e}")
    
    def _check_import_violations(self):
        """检查导入规范违规"""
        print("检查导入规范...")
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for line_num, line in enumerate(content.split('\n'), 1):
                    line = line.strip()
                    
                    # 检查通配符导入
                    if re.match(r'from .* import \*', line):
                        self.violations['import_violations'].append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': line_num,
                            'content': line,
                            'violation': '禁止使用通配符导入'
                        })
                    
                    # 检查相对导入
                    if re.match(r'from \.\.', line):
                        self.violations['import_violations'].append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': line_num,
                            'content': line,
                            'violation': '禁止使用相对导入'
                        })
                        
            except Exception as e:
                print(f"检查文件 {py_file} 时出错: {e}")
    
    def _check_database_query_violations(self):
        """检查数据库查询规范违规"""
        print("检查数据库查询规范...")
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for line_num, line in enumerate(content.split('\n'), 1):
                    line_lower = line.lower().strip()
                    
                    # 检查SELECT *
                    if 'select *' in line_lower and 'from' in line_lower:
                        self.violations['database_query_violations'].append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': line_num,
                            'content': line.strip(),
                            'violation': '禁止使用SELECT *'
                        })
                    
                    # 检查缺少WHERE条件的查询
                    if ('select' in line_lower and 'from stock_info' in line_lower and 
                        'where' not in line_lower and 'limit' not in line_lower):
                        self.violations['database_query_violations'].append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': line_num,
                            'content': line.strip(),
                            'violation': '查询stock_info表必须包含WHERE条件'
                        })
                        
            except Exception as e:
                print(f"检查文件 {py_file} 时出错: {e}")
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否应该跳过文件"""
        skip_patterns = [
            '__pycache__',
            '.git',
            '.idea',
            'venv',
            '.pytest_cache',
            'test_',
            '_test.py'
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def _get_file_layer(self, file_path: Path) -> str:
        """获取文件所在的架构层级"""
        relative_path = file_path.relative_to(self.project_root)
        path_parts = relative_path.parts
        
        for layer, directories in self.layer_mapping.items():
            for directory in directories:
                if directory in str(relative_path):
                    return layer.replace('L', '')
        
        return None
    
    def _get_module_layer(self, module_name: str) -> str:
        """获取模块所在的架构层级"""
        for layer, directories in self.layer_mapping.items():
            for directory in directories:
                if module_name.startswith(directory.replace('/', '.')):
                    return layer.replace('L', '')
        
        return None
    
    def _is_layer_violation(self, from_layer: str, to_layer: str) -> bool:
        """判断是否为分层违规"""
        layer_order = ['6', '5', '4', '3', '2', '1']
        
        try:
            from_index = layer_order.index(from_layer)
            to_index = layer_order.index(to_layer)
            
            # 只能调用相邻下层或同层
            return to_index < from_index - 1
        except ValueError:
            return False
    
    def _extract_imports(self, content: str) -> List[str]:
        """提取文件中的导入模块"""
        imports = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        
        except Exception:
            # 如果AST解析失败，使用正则表达式
            import_patterns = [
                r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
                r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                imports.extend(matches)
        
        return imports
    
    def _is_pascal_case(self, name: str) -> bool:
        """检查是否为大驼峰命名"""
        return bool(re.match(r'^[A-Z][a-zA-Z0-9]*$', name))
    
    def _is_snake_case(self, name: str) -> bool:
        """检查是否为小写+下划线命名"""
        return bool(re.match(r'^[a-z_][a-z0-9_]*$', name))
    
    def generate_report(self) -> str:
        """生成检查报告"""
        report = ["# 架构合规性检查报告\n"]
        
        total_violations = sum(len(violations) for violations in self.violations.values())
        
        if total_violations == 0:
            report.append("✅ 恭喜！未发现架构违规问题。\n")
            return "\n".join(report)
        
        report.append(f"❌ 发现 {total_violations} 个架构违规问题：\n")
        
        # 按类型统计违规
        for violation_type, violations in self.violations.items():
            if not violations:
                continue
                
            report.append(f"## {violation_type.replace('_', ' ').title()} ({len(violations)}个)\n")
            
            for i, violation in enumerate(violations[:10], 1):  # 只显示前10个
                report.append(f"{i}. **文件**: {violation.get('file', 'N/A')}")
                if 'line' in violation:
                    report.append(f"   **行号**: {violation['line']}")
                if 'content' in violation:
                    report.append(f"   **内容**: `{violation['content']}`")
                report.append(f"   **违规**: {violation['violation']}\n")
            
            if len(violations) > 10:
                report.append(f"   ... 还有 {len(violations) - 10} 个类似违规\n")
        
        return "\n".join(report)


def main():
    """主函数"""
    checker = ArchitectureComplianceChecker(root_dir)
    violations = checker.check_all_violations()
    
    # 生成报告
    report = checker.generate_report()
    print(report)
    
    # 保存报告到文件
    report_file = os.path.join(root_dir, 'reports', 'architecture_compliance_report.md')
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"详细报告已保存到: {report_file}")
    
    # 返回退出码
    total_violations = sum(len(v) for v in violations.values())
    if total_violations > 0:
        print(f"\n❌ 发现 {total_violations} 个违规问题，请修复后再提交代码！")
        sys.exit(1)
    else:
        print("\n✅ 架构合规性检查通过！")
        sys.exit(0)


if __name__ == "__main__":
    main() 