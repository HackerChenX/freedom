#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
架构合规性检查脚本
检查项目代码是否符合架构规范
"""

import os
import sys
import re
import ast
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.path_utils import get_project_root, get_reports_dir

@dataclass
class ViolationInfo:
    """违规信息"""
    file_path: str
    line_number: Optional[int] = None
    content: Optional[str] = None
    violation_type: str = ""
    description: str = ""

class ArchitectureComplianceChecker:
    """架构合规性检查器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.project_root = get_project_root()
        self.reports_dir = get_reports_dir()
        
        # 确保报告目录存在
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # 违规模式定义 - 用于检查其他文件的直接数据库依赖
        self.direct_db_patterns = [
            r'from\s+db\.clickhouse_db\s+import',
            r'import\s+db\.clickhouse_db',
            r'get_clickhouse_db'
        ]
        
        # 分层架构定义
        self.layer_mapping = {
            'L1': ['utils', 'enums', 'config'],
            'L2': ['db', 'indicators', 'formula'],
            'L3': ['analysis', 'strategy'],
            'L4': ['bin', 'scripts'],
            'L5': ['tests']
        }
        
        # 创建文件到层级的映射
        self.file_to_layer = {}
        for layer, dirs in self.layer_mapping.items():
            for dir_name in dirs:
                self.file_to_layer[dir_name] = layer
        
        # 违规统计
        self.violations = {
            'layer_violations': [],
            'direct_db_dependencies': [],
            'code_duplication': [],
            'naming_violations': [],
            'import_violations': [],
            'database_query_violations': []
        }
    
    def _get_files_to_check(self) -> List[str]:
        """获取需要检查的文件列表"""
        files_to_check = []
        
        # 检查的目录
        check_dirs = [
            'analysis', 'api', 'bin', 'config', 'db', 'enums', 
            'formula', 'indicators', 'scripts', 'strategy', 
            'tests', 'utils', 'examples'
        ]
        
        for dir_name in check_dirs:
            dir_path = os.path.join(self.project_root, dir_name)
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        if file.endswith('.py') and not file.startswith('__'):
                            file_path = os.path.join(root, file)
                            # 跳过当前检查脚本本身
                            if not file_path.endswith('architecture_compliance_check.py'):
                                files_to_check.append(file_path)
        
        return files_to_check
    
    def _check_layer_violations(self, files: List[str]) -> List[Violation_info]:
        """检查分层架构违规"""
        violations = []
        
        for file_path in files:
            try:
                if not os.path.exists(file_path):
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 获取文件所属层级
                rel_path = os.path.relpath(file_path, self.project_root)
                path_parts = rel_path.split(os.sep)
                
                if len(path_parts) < 2:
                    continue
                
                file_layer = self.file_to_layer.get(path_parts[0])
                if not file_layer:
                    continue
                
                # 检查导入语句
                import_pattern = r'from\s+(\w+)(?:\.\w+)*\s+import|import\s+(\w+)(?:\.\w+)*'
                matches = re.finditer(import_pattern, content)
                
                for match in matches:
                    imported_module = match.group(1) or match.group(2)
                    if imported_module in self.file_to_layer:
                        imported_layer = self.file_to_layer[imported_module]
                        
                        # 检查是否违反分层原则
                        if self._is_layer_violation(file_layer, imported_layer):
                            violations.append(Violation_info(
                                file_path=rel_path,
                                violation_type="Layer Violation",
                                description=f"L{file_layer[-1]}层文件不能直接导入L{imported_layer[-1]}层模块"
                            ))
                            
            except Exception as e:
                self.logger.error(f"检查文件 {file_path} 的分层违规时出错: {e}")
                continue
        
        return violations
    
    def _is_layer_violation(self, from_layer: str, to_layer: str) -> bool:
        """判断是否违反分层原则"""
        layer_order = ['L1', 'L2', 'L3', 'L4', 'L5']
        
        try:
            from_index = layer_order.index(from_layer)
            to_index = layer_order.index(to_layer)
            
            # 低层不能依赖高层
            return from_index < to_index
        except ValueError:
            return False
    
    def _check_direct_db_dependencies(self, files: List[str]) -> List[Violation_info]:
        """检查直接数据库依赖"""
        violations = []
        
        for file_path in files:
            try:
                if not os.path.exists(file_path):
                    continue
                
                # 跳过架构合规性检查脚本本身
                if file_path.endswith('architecture_compliance_check.py'):
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    # 跳过注释行
                    stripped_line = line.strip()
                    if stripped_line.startswith('#'):
                        continue
                    
                    for pattern in self.direct_db_patterns:
                        if re.search(pattern, line):
                            # 修复f-string中的反斜杠问题
                            clean_pattern = pattern.replace('\\\\', '\\')
                            violations.append(Violation_info(
                                file_path=os.path.relpath(file_path, self.project_root),
                                line_number=line_num,
                                content=line.strip(),
                                violation_type="Direct DB Dependency",
                                description=f"禁止直接依赖数据库实现: {clean_pattern}"
                            ))
                            
            except Exception as e:
                self.logger.error(f"检查文件 {file_path} 的直接数据库依赖时出错: {e}")
                continue
        
        return violations
    
    def _check_code_duplication(self, files: List[str]) -> List[Violation_info]:
        """检查代码重复"""
        violations = []
        
        # 收集所有类名和方法名
        class_method_names = {}
        
        for file_path in files:
            try:
                if not os.path.exists(file_path):
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 使用AST解析Python代码
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Class_def):
                            name = node.name
                            if name in class_method_names:
                                class_method_names[name].append(file_path)
                            else:
                                class_method_names[name] = [file_path]
                        elif isinstance(node, ast.Function_def):
                            name = node.name
                            if name in class_method_names:
                                class_method_names[name].append(file_path)
                            else:
                                class_method_names[name] = [file_path]
                                
                except Syntax_error:
                    # 如果AST解析失败，跳过该文件
                    continue
                    
            except Exception as e:
                self.logger.error(f"检查文件 {file_path} 的代码重复时出错: {e}")
                continue
        
        # 找出重复的名称
        for name, files_list in class_method_names.items():
            if len(files_list) > 1:
                violations.append(Violation_info(
                    file_path="N/A",
                    violation_type="Code Duplication",
                    description=f"发现重复的类/方法名: {name}"
                ))
        
        return violations
    
    def _check_naming_violations(self, files: List[str]) -> List[Violation_info]:
        """检查命名规范违规"""
        violations = []
        
        for file_path in files:
            try:
                if not os.path.exists(file_path):
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 使用AST解析Python代码
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Class_def):
                            # 检查类名是否使用大驼峰命名法
                            if not self._is_camel_case(node.name):
                                violations.append(Violation_info(
                                    file_path=os.path.relpath(file_path, self.project_root),
                                    violation_type="Naming Violation",
                                    description="类名应使用大驼峰命名法"
                                ))
                                
                except Syntax_error:
                    # 如果AST解析失败，跳过该文件
                    continue
                    
            except Exception as e:
                self.logger.error(f"检查文件 {file_path} 的命名规范时出错: {e}")
                continue
        
        return violations
    
    def _is_camel_case(self, name: str) -> bool:
        """检查是否是驼峰命名法"""
        # 简单的驼峰命名法检查
        return name[0].isupper() and '_' not in name
    
    def _check_import_violations(self, files: List[str]) -> List[Violation_info]:
        """检查导入规范违规"""
        violations = []
        
        for file_path in files:
            try:
                if not os.path.exists(file_path):
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    # 检查通配符导入
                    if re.search(r'from\s+\w+\s+import\s+\*', line):
                        violations.append(Violation_info(
                            file_path=os.path.relpath(file_path, self.project_root),
                            line_number=line_num,
                            content=line.strip(),
                            violation_type="Import Violation",
                            description="禁止使用通配符导入"
                        ))
                        
            except Exception as e:
                self.logger.error(f"检查文件 {file_path} 的导入规范时出错: {e}")
                continue
        
        return violations
    
    def _check_database_query_violations(self, files: List[str]) -> List[Violation_info]:
        """检查数据库查询规范违规"""
        violations = []
        
        for file_path in files:
            try:
                if not os.path.exists(file_path):
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    # 检查SELECT code, name, date, level, open, close, high, low, volume
                    if re.search(r'SELECT\s+\*', line, re.IGNORECASE):
                        violations.append(Violation_info(
                            file_path=os.path.relpath(file_path, self.project_root),
                            line_number=line_num,
                            content=line.strip(),
                            violation_type="Database Query Violation",
                            description="禁止使用SELECT code, name, date, level, open, close, high, low, volume"
                        ))
                    
                    # 检查stock_info表查询是否有WHERE条件
                    if re.search(r'FROM\s+stock_info(?!\s+WHERE)', line, re.IGNORECASE):
                        violations.append(Violation_info(
                            file_path=os.path.relpath(file_path, self.project_root),
                            line_number=line_num,
                            content=line.strip(),
                            violation_type="Database Query Violation",
                            description="查询stock_info表必须包含WHERE条件"
                        ))
                        
            except Exception as e:
                self.logger.error(f"检查文件 {file_path} 的数据库查询规范时出错: {e}")
                continue
        
        return violations
    
    def run_compliance_check(self) -> Dict[str, List[Violation_info]]:
        """运行合规性检查"""
        self.logger.info("开始架构合规性检查...")
        
        # 获取需要检查的文件
        files_to_check = self._get_files_to_check()
        self.logger.info(f"共需检查 {len(files_to_check)} 个文件")
        
        # 执行各项检查
        self.logger.info("检查分层架构违规...")
        self.violations['layer_violations'] = self._check_layer_violations(files_to_check)
        
        self.logger.info("检查直接数据库依赖...")
        self.violations['direct_db_dependencies'] = self._check_direct_db_dependencies(files_to_check)
        
        self.logger.info("检查代码重复...")
        self.violations['code_duplication'] = self._check_code_duplication(files_to_check)
        
        self.logger.info("检查命名规范...")
        self.violations['naming_violations'] = self._check_naming_violations(files_to_check)
        
        self.logger.info("检查导入规范...")
        self.violations['import_violations'] = self._check_import_violations(files_to_check)
        
        self.logger.info("检查数据库查询规范...")
        self.violations['database_query_violations'] = self._check_database_query_violations(files_to_check)
        
        return self.violations
    
    def generate_report_Check(self, violations: Dict[str, List[Violation_info]]) -> str:
        """生成合规性报告"""
        report_lines = []
        
        # 统计总违规数
        total_violations = sum(len(v) for v in violations.values())
        
        report_lines.append("# 架构合规性检查报告\n")
        
        if total_violations == 0:
            report_lines.append("✅ 未发现架构违规问题！\n")
        else:
            report_lines.append(f"❌ 发现 {total_violations} 个架构违规问题：\n")
        
        # 各类违规详情
        for violation_type, violation_list in violations.items():
            if not violation_list:
                continue
                
            type_name = violation_type.replace('_', ' ').title()
            report_lines.append(f"## {type_name} ({len(violation_list)}个)\n")
            
            for i, violation in enumerate(violation_list[:10], 1):  # 只显示前10个
                report_lines.append(f"{i}. **文件**: {violation.file_path}")
                if violation.line_number:
                    report_lines.append(f"   **行号**: {violation.line_number}")
                if violation.content:
                    report_lines.append(f"   **内容**: `{violation.content}`")
                report_lines.append(f"   **违规**: {violation.description}\n")
            
            if len(violation_list) > 10:
                report_lines.append(f"   ... 还有 {len(violation_list) - 10} 个类似违规\n")
        
        # 保存报告
        report_content = '\n'.join(report_lines)
        report_file = os.path.join(self.reports_dir, 'architecture_compliance_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"详细报告已保存到: {report_file}")
        
        return report_content

def main_architecturecompliancecheck():
    """主函数"""
    checker = Architecture_compliance_checker()
    
    # 运行检查
    violations = checker.run_compliance_check()
    
    # 生成报告
    report = checker.generate_report_Check(violations)
    
    # 打印摘要
    total_violations = sum(len(v) for v in violations.values())
    
    if total_violations == 0:
        print("✅ 架构合规性检查通过！")
        return 0
    else:
        print(f"❌ 发现 {total_violations} 个违规问题，请修复后再提交代码！")
        return 1

if __name__ == "__main__":
    sys.exit(main_architecturecompliancecheck()) 