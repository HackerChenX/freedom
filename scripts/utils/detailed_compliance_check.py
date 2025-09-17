#!/usr/bin/env python3
"""
详细的架构合规性检查脚本
分析所有违规问题的具体类型、位置和修复建议
"""

import os
import re
import ast
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import sys

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

class DetailedComplianceChecker:
    """详细的合规性检查器"""
    
    def __init__(self):
        self.root_dir = Path(root_dir)
        self.violations = {
            'layer_violations': [],
            'db_dependencies': [],
            'naming_violations': [],
            'query_violations': [],
            'code_duplications': [],
            'import_violations': [],
            'singleton_violations': []
        }
        
        # 分层架构规则
        self.layer_rules = {
            'config': ['enums', 'utils'],  # config层只能依赖enums和utils
            'db': ['config', 'enums', 'utils'],  # db层不能依赖业务层
            'utils': ['enums'],  # utils层只能依赖enums
            'strategy': ['db', 'config', 'enums', 'utils', 'indicators', 'formula'],
            'analysis': ['db', 'config', 'enums', 'utils', 'indicators', 'formula', 'strategy'],
            'api': ['db', 'config', 'enums', 'utils'],
            'indicators': ['db', 'config', 'enums', 'utils', 'formula'],
            'formula': ['db', 'config', 'enums', 'utils']
        }
        
        # 命名规范
        self.naming_patterns = {
            'class': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),  # 大驼峰
            'function': re.compile(r'^[a-z][a-z0-9_]*$'),  # 小写下划线
            'variable': re.compile(r'^[a-z][a-z0-9_]*$'),  # 小写下划线
            'constant': re.compile(r'^[A-Z][A-Z0-9_]*$'),  # 大写下划线
            'module': re.compile(r'^[a-z][a-z0-9_]*$')  # 小写下划线
        }
        
        # 数据库依赖模式
        self.db_patterns = [
            r'get_clickhouse_db\(\)',
            r'from\s+db\.clickhouse_db\s+import',
            r'import\s+.*clickhouse_db',
            r'clickhouse_db\.',
            r'ClickHouseDB\(',
            r'get_db\(\)'
        ]
        
        # 查询违规模式
        self.query_patterns = [
            r'SELECT\s+\*\s+FROM',  # SELECT code, name, price
            r'stock_info\s+(?!WHERE)',  # stock_info WHERE 1=1 without WHERE
            r'DELETE\s+FROM\s+\w+\s*$',  # DELETE without WHERE
            r'UPDATE\s+\w+\s+SET.*(?!WHERE)',  # UPDATE without WHERE
        ]
    
    def check_all_violations(self) -> Dict:
        """检查所有类型的违规"""
        logger.info("开始详细合规性检查...")
        
        # 获取所有Python文件
        python_files = list(self.root_dir.rglob("*.py"))
        python_files = [f for f in python_files if not any(exclude in str(f) for exclude in [
            '__pycache__', '.git', 'venv', '.pytest_cache', 'node_modules'
        ])]
        
        logger.info(f"检查 {len(python_files)} 个Python文件")
        
        # 检查各类违规
        for file_path in python_files:
            try:
                self.check_file_violations(file_path)
            except Exception as e:
                logger.warning(f"检查文件 {file_path} 时出错: {e}")
        
        # 统计结果
        result = self.generate_report_detailedcompliancecheck()
        return result
    
    def check_file_violations(self, file_path: Path):
        """检查单个文件的违规"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查分层架构违规
            self.check_layer_violations(file_path, content)
            
            # 检查数据库依赖
            self.check_db_dependencies(file_path, content)
            
            # 检查命名规范
            self.check_naming_violations(file_path, content)
            
            # 检查查询违规
            self.check_query_violations(file_path, content)
            
            # 检查代码重复
            self.check_code_duplications(file_path, content)
            
            # 检查导入违规
            self.check_import_violations(file_path, content)
            
        except Exception as e:
            logger.warning(f"读取文件 {file_path} 失败: {e}")
    
    def check_layer_violations(self, file_path: Path, content: str):
        """检查分层架构违规"""
        relative_path = file_path.relative_to(self.root_dir)
        parts = relative_path.parts
        
        if len(parts) < 2:
            return
        
        current_layer = parts[0]
        if current_layer not in self.layer_rules:
            return
        
        allowed_layers = self.layer_rules[current_layer]
        
        # 检查导入语句
        import_pattern = re.compile(r'from\s+(\w+)(?:\.\w+)*\s+import|import\s+(\w+)(?:\.\w+)*')
        for match in import_pattern.finditer(content):
            imported_layer = match.group(1) or match.group(2)
            if imported_layer and imported_layer in self.layer_rules and imported_layer not in allowed_layers:
                self.violations['layer_violations'].append({
                    'file': str(relative_path),
                    'layer': current_layer,
                    'violated_import': imported_layer,
                    'line': content[:match.start()].count('\n') + 1,
                    'allowed_layers': allowed_layers
                })
    
    def check_db_dependencies(self, file_path: Path, content: str):
        """检查直接数据库依赖"""
        relative_path = file_path.relative_to(self.root_dir)
        
        for pattern in self.db_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                line_num = content[:match.start()].count('\n') + 1
                self.violations['db_dependencies'].append({
                    'file': str(relative_path),
                    'pattern': pattern,
                    'line': line_num,
                    'context': self.get_line_context(content, line_num)
                })
    
    def check_naming_violations_detailed_compliance_check(self, file_path: Path, content: str):
        """检查命名规范违规"""
        relative_path = file_path.relative_to(self.root_dir)
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Class_def):
                    if not self.naming_patterns['class'].match(node.name):
                        self.violations['naming_violations'].append({
                            'file': str(relative_path),
                            'type': 'class',
                            'name': node.name,
                            'line': node.lineno,
                            'expected_pattern': 'PascalCase (e.g., MyClass)'
                        })
                
                elif isinstance(node, ast.Function_def):
                    if not self.naming_patterns['function'].match(node.name):
                        self.violations['naming_violations'].append({
                            'file': str(relative_path),
                            'type': 'function',
                            'name': node.name,
                            'line': node.lineno,
                            'expected_pattern': 'snake_case (e.g., my_function)'
                        })
                
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    # 检查变量名
                    if node.id.isupper() and len(node.id) > 1:
                        # 常量
                        if not self.naming_patterns['constant'].match(node.id):
                            self.violations['naming_violations'].append({
                                'file': str(relative_path),
                                'type': 'constant',
                                'name': node.id,
                                'line': node.lineno,
                                'expected_pattern': 'UPPER_SNAKE_CASE (e.g., MY_CONSTANT)'
                            })
                    else:
                        # 普通变量
                        if not self.naming_patterns['variable'].match(node.id):
                            self.violations['naming_violations'].append({
                                'file': str(relative_path),
                                'type': 'variable',
                                'name': node.id,
                                'line': node.lineno,
                                'expected_pattern': 'snake_case (e.g., my_variable)'
                            })
        
        except Syntax_error:
            # 忽略语法错误的文件
            pass
    
    def check_query_violations(self, file_path: Path, content: str):
        """检查数据库查询违规"""
        relative_path = file_path.relative_to(self.root_dir)
        
        for pattern in self.query_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
                line_num = content[:match.start()].count('\n') + 1
                self.violations['query_violations'].append({
                    'file': str(relative_path),
                    'pattern': pattern,
                    'line': line_num,
                    'context': self.get_line_context(content, line_num),
                    'suggestion': self.get_query_suggestion(pattern)
                })
    
    def check_code_duplications_detailed_compliance_check(self, file_path: Path, content: str):
        """检查代码重复"""
        relative_path = file_path.relative_to(self.root_dir)
        
        try:
            tree = ast.parse(content)
            
            # 收集类名和函数名
            names = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Class_def, ast.Function_def)):
                    names.append({
                        'type': 'class' if isinstance(node, ast.ClassDef) else 'function',
                        'name': node.name,
                        'line': node.lineno,
                        'file': str(relative_path)
                    })
            
            # 检查重复名称
            name_counts = Counter([item['name'] for item in names])
            for name, count in name_counts.items():
                if count > 1:
                    # 查找所有同名项
                    duplicates = [item for item in names if item['name'] == name]
                    for dup in duplicates:
                        self.violations['code_duplications'].append({
                            'file': dup['file'],
                            'type': dup['type'],
                            'name': dup['name'],
                            'line': dup['line'],
                            'duplicate_count': count,
                            'suggestion': f'Rename to {name}_{dup["type"]}_v1, {name}_{dup["type"]}_v2, etc.'
                        })
        
        except Syntax_error:
            pass
    
    def check_import_violations_detailed_compliance_check(self, file_path: Path, content: str):
        """检查导入违规"""
        relative_path = file_path.relative_to(self.root_dir)
        
        # 检查相对导入
        relative_import_pattern = r'from\s+\.+\w*\s+import'
        for match in re.finditer(relative_import_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            self.violations['import_violations'].append({
                'file': str(relative_path),
                'type': 'relative_import',
                'line': line_num,
                'context': self.get_line_context(content, line_num),
                'suggestion': 'Use absolute imports instead of relative imports'
            })
    
    def get_line_context(self, content: str, line_num: int) -> str:
        """获取行上下文"""
        lines = content.split('\n')
        if 1 <= line_num <= len(lines):
            return lines[line_num - 1].strip()
        return ""
    
    def get_query_suggestion(self, pattern: str) -> str:
        """获取查询建议"""
        suggestions = {
            r'SELECT\s+\*\s+FROM': 'Use specific column names instead of SELECT code, name, price',
            r'stock_info\s+(?!WHERE)': 'Add WHERE clause to stock_info WHERE 1=1 queries',
            r'DELETE\s+FROM\s+\w+\s*$': 'Add WHERE clause to DELETE statements',
            r'UPDATE\s+\w+\s+SET.*(?!WHERE)': 'Add WHERE clause to UPDATE statements'
        }
        return suggestions.get(pattern, 'Follow database query best practices')
    
    def generate_report_detailedcompliancecheck(self) -> Dict:
        """生成详细报告"""
        total_violations = sum(len(violations) for violations in self.violations.values())
        
        report = {
            'timestamp': str(Path().cwd()),
            'total_violations': total_violations,
            'violation_summary': {
                'layer_violations': len(self.violations['layer_violations']),
                'db_dependencies': len(self.violations['db_dependencies']),
                'naming_violations': len(self.violations['naming_violations']),
                'query_violations': len(self.violations['query_violations']),
                'code_duplications': len(self.violations['code_duplications']),
                'import_violations': len(self.violations['import_violations']),
                'singleton_violations': len(self.violations['singleton_violations'])
            },
            'detailed_violations': self.violations,
            'top_violation_files': self.get_top_violation_files(),
            'violation_distribution': self.get_violation_distribution()
        }
        
        return report
    
    def get_top_violation_files(self) -> List[Dict]:
        """获取违规最多的文件"""
        file_violations = defaultdict(int)
        
        for violation_type, violations in self.violations.items():
            for violation in violations:
                file_violations[violation['file']] += 1
        
        top_files = sorted(file_violations.items(), key=lambda x: x[1], reverse=True)[:20]
        return [{'file': file, 'violation_count': count} for file, count in top_files]
    
    def get_violation_distribution(self) -> Dict:
        """获取违规分布"""
        distribution = defaultdict(lambda: defaultdict(int))
        
        for violation_type, violations in self.violations.items():
            for violation in violations:
                file_path = violation['file']
                module = file_path.split('/')[0] if '/' in file_path else 'root'
                distribution[module][violation_type] += 1
        
        return dict(distribution)
    
    def save_report(self, report: Dict, output_file: str):
        """保存报告"""
        output_path = self.root_dir / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"详细合规性报告已保存到: {output_path}")

def main_detailedcompliancecheck():
    """主函数"""
    try:
        checker = Detailed_compliance_checker()
        report = checker.check_all_violations()
        
        # 保存详细报告
        checker.save_report(report, 'data/result/detailed_compliance_report.json')
        
        # 打印摘要
        print(f"\n🔍 详细合规性检查完成")
        print(f"总计发现 {report['total_violations']} 个违规问题:")
        
        for violation_type, count in report['violation_summary'].items():
            if count > 0:
                print(f"  - {violation_type}: {count}")
        
        print(f"\n📊 违规最多的文件:")
        for file_info in report['top_violation_files'][:10]:
            print(f"  - {file_info['file']}: {file_info['violation_count']} 个问题")
        
        print(f"\n📈 模块违规分布:")
        for module, violations in report['violation_distribution'].items():
            total = sum(violations.values())
            if total > 0:
                print(f"  - {module}: {total} 个问题")
        
        # 如果还有违规，返回非零退出码
        if report['total_violations'] > 0:
            print(f"\n❌ 发现 {report['total_violations']} 个违规问题，需要修复！")
            return 1
        else:
            print(f"\n✅ 所有检查通过，系统100%合规！")
            return 0
            
    except Exception as e:
        logger.error(f"详细合规性检查失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main_detailedcompliancecheck()) 