#!/usr/bin/env python3
"""
测试框架问题修复工具
自动检测和修复测试框架中的常见问题，按优先级执行修复
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

class Priority(Enum):
    P0 = "P0-Critical"
    P1 = "P1-Important" 
    P2 = "P2-Optimization"

@dataclass
class Issue:
    file_path: str
    line_number: int
    issue_type: str
    description: str
    priority: Priority
    fix_suggestion: str
    
class TestFrameworkFixer:
    """测试框架修复工具"""
    
    def __init__(self, root_dir: str = "tests"):
        self.root_dir = root_dir
        self.issues: List[Issue] = []
        self.fixes_applied = 0
        
    def scan_all_issues(self) -> List[Issue]:
        """扫描所有问题"""
        print("🔍 扫描测试框架问题...")
        
        # P0: 关键问题
        self._scan_test_case_naming()
        self._scan_missing_imports()
        self._scan_syntax_errors()
        
        # P1: 重要问题
        self._scan_setup_method_issues()
        self._scan_class_initialization()
        
        # P2: 优化问题
        self._scan_import_style()
        self._scan_performance_issues()
        
        return self.issues
    
    def _scan_test_case_naming(self):
        """扫描Test_case命名问题 (P0)"""
        pattern = re.compile(r'unittest\.Test_case|Test_case')
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for i, line in enumerate(lines, 1):
                    if pattern.search(line):
                        self.issues.append(Issue(
                            file_path=file_path,
                            line_number=i,
                            issue_type="Test_case_naming",
                            description=f"使用了Test_case而非TestCase: {line.strip()}",
                            priority=Priority.P0,
                            fix_suggestion="替换为unittest.TestCase"
                        ))
            except Exception as e:
                print(f"❌ 无法读取文件 {file_path}: {e}")
    
    def _scan_missing_imports(self):
        """扫描缺失导入 (P0)"""
        common_missing = [
            'LogCaptureMixin',
            'TestDataGenerator', 
            'IndicatorTestMixin',
            'MagicMock',
            'get_config'
        ]
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for missing_import in common_missing:
                    if missing_import in content and f'import {missing_import}' not in content:
                        self.issues.append(Issue(
                            file_path=file_path,
                            line_number=1,
                            issue_type="missing_import",
                            description=f"使用了{missing_import}但未导入",
                            priority=Priority.P0,
                            fix_suggestion=f"添加正确的导入语句"
                        ))
            except Exception as e:
                print(f"❌ 无法分析导入 {file_path}: {e}")
    
    def _scan_syntax_errors(self):
        """扫描语法错误 (P0)"""
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    self.issues.append(Issue(
                        file_path=file_path,
                        line_number=e.lineno or 1,
                        issue_type="syntax_error",
                        description=f"语法错误: {e.msg}",
                        priority=Priority.P0,
                        fix_suggestion="修复语法错误"
                    ))
            except Exception as e:
                print(f"❌ 无法检查语法 {file_path}: {e}")
    
    def _scan_setup_method_issues(self):
        """扫描setUp方法问题 (P1)"""
        setup_pattern = re.compile(r'def setUp\w*\(|def setup\(')
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # 检查是否有测试类但缺少setUp方法
                has_test_class = any('class Test' in line for line in lines)
                has_setup = any(setup_pattern.search(line) for line in lines)
                
                if has_test_class and not has_setup:
                    self.issues.append(Issue(
                        file_path=file_path,
                        line_number=1,
                        issue_type="missing_setup",
                        description="测试类缺少setUp方法",
                        priority=Priority.P1,
                        fix_suggestion="添加setUp方法进行测试初始化"
                    ))
                    
            except Exception as e:
                print(f"❌ 无法检查setUp方法 {file_path}: {e}")
    
    def _scan_class_initialization(self):
        """扫描类初始化问题 (P1)"""
        # 这里可以添加更复杂的AST分析逻辑
        pass
    
    def _scan_import_style(self):
        """扫描导入风格问题 (P2)"""
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for i, line in enumerate(lines, 1):
                    # 检查通配符导入
                    if 'from * import' in line or 'import *' in line:
                        self.issues.append(Issue(
                            file_path=file_path,
                            line_number=i,
                            issue_type="wildcard_import",
                            description=f"使用了通配符导入: {line.strip()}",
                            priority=Priority.P2,
                            fix_suggestion="使用明确的导入语句"
                        ))
                        
                    # 检查相对导入
                    if 'from ..' in line:
                        self.issues.append(Issue(
                            file_path=file_path,
                            line_number=i,
                            issue_type="relative_import",
                            description=f"使用了相对导入: {line.strip()}",
                            priority=Priority.P2,
                            fix_suggestion="使用绝对导入路径"
                        ))
                        
            except Exception as e:
                print(f"❌ 无法检查导入风格 {file_path}: {e}")
    
    def _scan_performance_issues(self):
        """扫描性能问题 (P2)"""
        # 可以添加检查长时间运行的测试等
        pass
    
    def _get_python_files(self) -> List[str]:
        """获取所有Python文件"""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files
    
    def auto_fix_p0_issues(self) -> int:
        """自动修复P0级别问题"""
        print("🔧 自动修复P0级别问题...")
        fixes_count = 0
        
        p0_issues = [issue for issue in self.issues if issue.priority == Priority.P0]
        
        for issue in p0_issues:
            if issue.issue_type == "Test_case_naming":
                if self._fix_test_case_naming(issue.file_path):
                    fixes_count += 1
                    print(f"✅ 修复 {issue.file_path} 中的Test_case命名")
        
        return fixes_count
    
    def _fix_test_case_naming(self, file_path: str) -> bool:
        """修复Test_case命名问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 修复Test_case命名
            content = re.sub(r'unittest\.Test_case', 'unittest.TestCase', content)
            content = re.sub(r'(?<!unittest\.)Test_case(?=\s*[:\(])', 'TestCase', content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            return True
        except Exception as e:
            print(f"❌ 修复失败 {file_path}: {e}")
            return False
    
    def generate_report(self) -> str:
        """生成问题报告"""
        report = []
        report.append("# 测试框架问题分析报告\n")
        
        # 按优先级分组
        by_priority = {}
        for issue in self.issues:
            if issue.priority not in by_priority:
                by_priority[issue.priority] = []
            by_priority[issue.priority].append(issue)
        
        # 生成统计
        report.append("## 问题统计\n")
        for priority in Priority:
            count = len(by_priority.get(priority, []))
            report.append(f"- **{priority.value}**: {count} 个问题")
        report.append(f"\n**总计**: {len(self.issues)} 个问题\n")
        
        # 详细问题列表
        for priority in Priority:
            if priority not in by_priority:
                continue
                
            report.append(f"## {priority.value} 问题\n")
            issues = by_priority[priority]
            
            # 按问题类型分组
            by_type = {}
            for issue in issues:
                if issue.issue_type not in by_type:
                    by_type[issue.issue_type] = []
                by_type[issue.issue_type].append(issue)
            
            for issue_type, type_issues in by_type.items():
                report.append(f"### {issue_type} ({len(type_issues)} 个)\n")
                for issue in type_issues[:10]:  # 只显示前10个
                    report.append(f"- `{issue.file_path}:{issue.line_number}` - {issue.description}")
                
                if len(type_issues) > 10:
                    report.append(f"- ... 还有 {len(type_issues) - 10} 个类似问题")
                report.append("")
        
        return "\n".join(report)

def main():
    """主函数"""
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "tests"
    
    print(f"🚀 启动测试框架修复工具 (目录: {root_dir})")
    
    fixer = TestFrameworkFixer(root_dir)
    
    # 扫描问题
    issues = fixer.scan_all_issues()
    print(f"📊 发现 {len(issues)} 个问题")
    
    # 按优先级统计
    by_priority = {}
    for issue in issues:
        by_priority[issue.priority] = by_priority.get(issue.priority, 0) + 1
    
    for priority in Priority:
        count = by_priority.get(priority, 0)
        print(f"   {priority.value}: {count} 个")
    
    # 自动修复P0问题
    if by_priority.get(Priority.P0, 0) > 0:
        print("\n🔧 开始自动修复P0问题...")
        fixes = fixer.auto_fix_p0_issues()
        print(f"✅ 自动修复了 {fixes} 个P0问题")
    
    # 生成报告
    report = fixer.generate_report()
    report_file = "test_framework_issues_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📋 问题报告已生成: {report_file}")
    print("\n📝 修复建议:")
    print("1. 优先修复P0问题（已自动修复部分）")
    print("2. 手动检查和修复剩余的P0问题")
    print("3. 按需修复P1和P2问题")
    print("4. 运行测试验证修复效果")

if __name__ == "__main__":
    main() 