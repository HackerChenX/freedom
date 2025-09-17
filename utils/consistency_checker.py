"""
自动化一致性检查工具
检测和修复系统中的重复修复问题
"""

import os
import re
import ast
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from db.sql_manager import SQLManager, QueryType


class InconsistencyType(Enum):
    """不一致类型枚举"""
    FIELD_MAPPING = "field_mapping"
    METHOD_NAMING = "method_naming"
    INTERFACE_MISMATCH = "interface_mismatch"
    HARDCODED_VALUES = "hardcoded_values"
    DUPLICATE_LOGIC = "duplicate_logic"


@dataclass
class InconsistencyIssue:
    """不一致问题"""
    issue_type: InconsistencyType
    file_path: str
    line_number: int
    description: str
    current_value: str
    suggested_fix: str
    severity: str  # HIGH, MEDIUM, LOW


class ConsistencyChecker:
    """一致性检查器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.issues: List[InconsistencyIssue] = []
        
        # 定义检查规则
        self.field_mapping_rules = {
            'turnover_rate': 'turnover_rate',  # 数据库中的实际字段名
            'stock_code': 'code',
            'stock_name': 'name',
            'trade_date': 'date'
        }
        
        self.method_naming_rules = {
            'analyze_buypoint': ['analyze_buypoint', 'analyze_multi_period_buypoint'],
            'get_stock_data': ['get_stock_data', 'query_stock_data', 'fetch_stock_data'],
            'calculate_indicator': ['calculate', 'compute', 'calculate_indicator']
        }
        
        self.hardcoded_patterns = [
            r'turnover_rate',
            r'SELECT\s+\*\s+FROM',  # 禁止SELECT *
            r'localhost:9000',      # 硬编码数据库地址
            r'password\s*=\s*["\'][^"\']+["\']'  # 硬编码密码
        ]
    
    def check_all(self) -> List[InconsistencyIssue]:
        """执行所有一致性检查"""
        self.issues.clear()
        
        print("🔍 开始执行一致性检查...")
        
        # 1. 检查字段映射一致性
        self._check_field_mapping_consistency()
        
        # 2. 检查方法命名一致性
        self._check_method_naming_consistency()
        
        # 3. 检查硬编码值
        self._check_hardcoded_values()
        
        # 4. 检查重复逻辑
        self._check_duplicate_logic()
        
        # 5. 检查接口一致性
        self._check_interface_consistency()
        
        print(f"✅ 一致性检查完成，发现 {len(self.issues)} 个问题")
        
        return self.issues
    
    def _check_field_mapping_consistency(self):
        """检查字段映射一致性"""
        print("  检查字段映射一致性...")
        
        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    # 检查turnover_rate字段使用
                    if 'turnover_rate' in line and 'SELECT' in line.upper():
                        self.issues.append(InconsistencyIssue(
                            issue_type=InconsistencyType.FIELD_MAPPING,
                            file_path=str(py_file),
                            line_number=i,
                            description="使用了不存在的turnover_rate字段",
                            current_value=line.strip(),
                            suggested_fix="使用 'turnover_rate' 字段或通过unified_field_mapper获取正确字段名",
                            severity="HIGH"
                        ))
                    
                    # 检查SELECT * 使用
                    if re.search(r'SELECT\s+\*\s+FROM', line, re.IGNORECASE):
                        self.issues.append(InconsistencyIssue(
                            issue_type=InconsistencyType.HARDCODED_VALUES,
                            file_path=str(py_file),
                            line_number=i,
                            description="使用了SELECT *，应该明确指定字段",
                            current_value=line.strip(),
                            suggested_fix="使用unified_query_builder构建查询",
                            severity="MEDIUM"
                        ))
                        
            except Exception as e:
                continue
    
    def _check_method_naming_consistency(self):
        """检查方法命名一致性"""
        print("  检查方法命名一致性...")
        
        method_definitions = {}
        method_calls = {}
        
        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 查找方法定义
                for match in re.finditer(r'def\s+(analyze_\w+)', content):
                    method_name = match.group(1)
                    line_num = content[:match.start()].count('\n') + 1
                    
                    if method_name not in method_definitions:
                        method_definitions[method_name] = []
                    method_definitions[method_name].append((str(py_file), line_num))
                
                # 查找方法调用
                for match in re.finditer(r'\.?(analyze_\w+)\s*\(', content):
                    method_name = match.group(1)
                    line_num = content[:match.start()].count('\n') + 1
                    
                    if method_name not in method_calls:
                        method_calls[method_name] = []
                    method_calls[method_name].append((str(py_file), line_num))
                        
            except Exception as e:
                continue
        
        # 检查方法定义不一致
        analyze_methods = [name for name in method_definitions.keys() if name.startswith('analyze_')]
        if len(analyze_methods) > 1:
            for method_name in analyze_methods:
                for file_path, line_num in method_definitions[method_name]:
                    if method_name != 'analyze_buypoint':  # 标准方法名
                        self.issues.append(InconsistencyIssue(
                            issue_type=InconsistencyType.METHOD_NAMING,
                            file_path=file_path,
                            line_number=line_num,
                            description=f"方法名不一致: {method_name}",
                            current_value=method_name,
                            suggested_fix="使用统一的analyze_buypoint方法名或实现IUnifiedAnalyzer接口",
                            severity="MEDIUM"
                        ))
    
    def _check_hardcoded_values(self):
        """检查硬编码值"""
        print("  检查硬编码值...")
        
        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    for pattern in self.hardcoded_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            self.issues.append(InconsistencyIssue(
                                issue_type=InconsistencyType.HARDCODED_VALUES,
                                file_path=str(py_file),
                                line_number=i,
                                description=f"发现硬编码值: {pattern}",
                                current_value=line.strip(),
                                suggested_fix="使用配置文件或统一的常量定义",
                                severity="MEDIUM"
                            ))
                            
            except Exception as e:
                continue
    
    def _check_duplicate_logic(self):
        """检查重复逻辑"""
        print("  检查重复逻辑...")
        
        # 查找相似的查询逻辑
        query_patterns = {}
        
        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 查找SQL查询模式
                sql_matches = re.finditer(r'SELECT.*?FROM.*?WHERE.*?ORDER BY', content, re.IGNORECASE | re.DOTALL)
                for match in sql_matches:
                    sql_text = match.group(0)
                    # 标准化SQL文本
                    normalized_sql = re.sub(r'\s+', ' ', sql_text).strip().upper()
                    
                    if normalized_sql not in query_patterns:
                        query_patterns[normalized_sql] = []
                    
                    line_num = content[:match.start()].count('\n') + 1
                    query_patterns[normalized_sql].append((str(py_file), line_num))
                        
            except Exception as e:
                continue
        
        # 报告重复的查询逻辑
        for sql_pattern, locations in query_patterns.items():
            if len(locations) > 1:
                for file_path, line_num in locations:
                    self.issues.append(InconsistencyIssue(
                        issue_type=InconsistencyType.DUPLICATE_LOGIC,
                        file_path=file_path,
                        line_number=line_num,
                        description=f"发现重复的查询逻辑，共{len(locations)}处",
                        current_value=sql_pattern[:100] + "...",
                        suggested_fix="使用unified_query_builder统一查询构建",
                        severity="LOW"
                    ))
    
    def _check_interface_consistency(self):
        """检查接口一致性"""
        print("  检查接口一致性...")
        
        # 这里可以添加更多接口一致性检查
        # 例如检查方法签名、返回值格式等
        pass
    
    def _get_python_files(self) -> List[Path]:
        """获取所有Python文件"""
        python_files = []
        
        # 排除的目录
        exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 
                       'venv', 'env', '.venv', 'archive', 'backup'}
        
        for root, dirs, files in os.walk(self.project_root):
            # 过滤排除的目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def generate_report(self) -> str:
        """生成检查报告"""
        if not self.issues:
            return "✅ 未发现一致性问题"
        
        report = ["🔍 **一致性检查报告**", "=" * 50, ""]
        
        # 按严重程度分组
        high_issues = [issue for issue in self.issues if issue.severity == "HIGH"]
        medium_issues = [issue for issue in self.issues if issue.severity == "MEDIUM"]
        low_issues = [issue for issue in self.issues if issue.severity == "LOW"]
        
        # 统计信息
        report.extend([
            f"📊 **问题统计:**",
            f"  - 高严重性: {len(high_issues)}",
            f"  - 中严重性: {len(medium_issues)}",
            f"  - 低严重性: {len(low_issues)}",
            f"  - 总计: {len(self.issues)}",
            ""
        ])
        
        # 详细问题列表
        for severity, issues in [("HIGH", high_issues), ("MEDIUM", medium_issues), ("LOW", low_issues)]:
            if issues:
                report.extend([f"## {severity} 严重性问题", ""])
                
                for i, issue in enumerate(issues, 1):
                    report.extend([
                        f"### {i}. {issue.description}",
                        f"**文件:** {issue.file_path}:{issue.line_number}",
                        f"**类型:** {issue.issue_type.value}",
                        f"**当前值:** `{issue.current_value}`",
                        f"**建议修复:** {issue.suggested_fix}",
                        ""
                    ])
        
        return "\n".join(report)
    
    def auto_fix_issues(self, issue_types: Optional[List[InconsistencyType]] = None) -> int:
        """自动修复问题"""
        if issue_types is None:
            issue_types = [InconsistencyType.FIELD_MAPPING, InconsistencyType.HARDCODED_VALUES]
        
        fixed_count = 0
        
        for issue in self.issues:
            if issue.issue_type in issue_types and issue.severity == "HIGH":
                try:
                    if self._apply_fix(issue):
                        fixed_count += 1
                except Exception as e:
                    print(f"修复失败 {issue.file_path}:{issue.line_number} - {e}")
        
        return fixed_count
    
    def _apply_fix(self, issue: InconsistencyIssue) -> bool:
        """应用修复"""
        # 这里可以实现自动修复逻辑
        # 例如替换turnover_rate为turnover_rate等
        return False


def run_consistency_check(project_root: str = ".") -> str:
    """运行一致性检查"""
    checker = ConsistencyChecker(project_root)
    issues = checker.check_all()
    return checker.generate_report()


if __name__ == "__main__":
    report = run_consistency_check()
    print(report)
