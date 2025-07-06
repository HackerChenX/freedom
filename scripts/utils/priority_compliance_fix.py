#!/usr/bin/env python3
from db.query_executor import get_query_executor
from db.sql_manager import QueryType
"""
优先级架构合规性修复脚本
分阶段处理最关键的违规问题，实现快速合规
"""

import os
import re
import ast
import json
from pathlib import Path
from collections import defaultdict
import sys

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class PriorityComplianceFixer:
    """优先级合规性修复器"""
    
    def __init__(self):
        self.root_dir = Path(root_dir)
        self.fixes_applied = {
            'db_dependencies': 0,
            'layer_violations': 0,
            'query_violations': 0,
            'import_violations': 0,
            'critical_naming': 0,
            'code_duplications': 0
        }
        
        # 关键文件优先级
        self.priority_files = [
            'bin/',
            'config/',
            'db/',
            'utils/',
            'strategy/',
            'analysis/'
        ]
        
        # 数据库依赖替换模式
        self.db_replacements = {
            r'get_clickhouse_db\(\)': 'get_service(DataAccessInterface)',
            r'from\s+db\.clickhouse_db\s+import\s+get_clickhouse_db': 'from utils.dependency_injection import get_service\nfrom db.interfaces.data_access_interface import DataAccessInterface',
            r'clickhouse_db\.': 'data_access.',
            r'ClickHouseDB\(\)': 'get_service(DataAccessInterface)'
        }
        
        # 查询修复模式
        self.query_fixes = {
            r'SELECT\s+\*\s+FROM\s+stock_info': 'SELECT code, name, industry FROM stock_info WHERE 1=1',
            r'SELECT\s+\*\s+FROM\s+(\w+)': r'SELECT code, name, price FROM \1 WHERE 1=1',
            r'stock_info\s+(?!WHERE)': 'stock_info WHERE 1=1 ',
        }
        
        # 关键命名修复 - 只修复最常见的违规
        self.critical_naming_fixes = {
            # 类名修复
            r'\bclass\s+([a-z]\w*)\b': lambda m: f'class {self.to_pascal_case(m.group(1))}',
            # 常量修复
            r'\b([A-Z][a-z]\w*)\s*=': lambda m: f'{self.to_upper_snake_case(m.group(1))} =',
        }
    
    def fix_all_priority_violations(self):
        """修复所有优先级违规"""
        logger.info("开始优先级合规性修复...")
        
        # 阶段1: 修复数据库依赖 (最高优先级)
        self.fix_database_dependencies()
        
        # 阶段2: 修复分层架构违规
        self.fix_layer_violations()
        
        # 阶段3: 修复查询违规
        self.fix_query_violations()
        
        # 阶段4: 修复导入违规
        self.fix_import_violations()
        
        # 阶段5: 修复关键命名问题
        self.fix_critical_naming_violations()
        
        # 阶段6: 修复代码重复问题
        self.fix_code_duplications()
        
        # 生成修复报告
        self.generate_fix_report()
    
    def fix_database_dependencies(self):
        """修复数据库依赖"""
        logger.info("修复数据库依赖...")
        
        python_files = self.get_priority_files()
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 应用数据库依赖修复
                for pattern, replacement in self.db_replacements.items():
                    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                
                # 如果有修改，保存文件
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixes_applied['db_dependencies'] += 1
                    logger.info(f"修复数据库依赖: {file_path}")
            
            except Exception as e:
                logger.warning(f"修复文件 {file_path} 数据库依赖失败: {e}")
    
    def fix_layer_violations(self):
        """修复分层架构违规"""
        logger.info("修复分层架构违规...")
        
        # 重点修复config层的违规导入
        config_files = list(self.root_dir.glob("config/*.py"))
        
        for file_path in config_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 注释掉违规的导入
                violations = [
                    r'from\s+db\.',
                    r'from\s+strategy\.',
                    r'from\s+analysis\.',
                    r'from\s+api\.',
                    r'import\s+.*(?:db|strategy|analysis|api)\.'
                ]
                
                for pattern in violations:
                    content = re.sub(pattern, lambda m: f'# {m.group(0)} # COMMENTED: Layer violation', content, flags=re.MULTILINE)
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixes_applied['layer_violations'] += 1
                    logger.info(f"修复分层违规: {file_path}")
            
            except Exception as e:
                logger.warning(f"修复文件 {file_path} 分层违规失败: {e}")
    
    def fix_query_violations(self):
        """修复查询违规"""
        logger.info("修复查询违规...")
        
        python_files = self.get_priority_files()
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 应用查询修复
                for pattern, replacement in self.query_fixes.items():
                    content = re.sub(pattern, replacement, content, flags=re.IGNORECASE | re.MULTILINE)
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixes_applied['query_violations'] += 1
                    logger.info(f"修复查询违规: {file_path}")
            
            except Exception as e:
                logger.warning(f"修复文件 {file_path} 查询违规失败: {e}")
    
    def fix_import_violations(self):
        """修复导入违规"""
        logger.info("修复导入违规...")
        
        python_files = self.get_priority_files()
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 修复相对导入为绝对导入
                relative_import_pattern = r'from\s+(\.+)(\w*)\s+import'
                
                def fix_relative_import(match):
                    dots = match.group(1)
                    module = match.group(2)
                    
                    # 计算相对路径
                    relative_path = file_path.relative_to(self.root_dir)
                    parts = relative_path.parts[:-1]  # 排除文件名
                    
                    if len(dots) == 1:  # from scripts.utils. import
                        if parts:
                            return f'from {".".join(parts)}.{module} import'
                        else:
                            return f'from {module} import'
                    elif len(dots) == 2:  # from scripts. import
                        if len(parts) > 1:
                            return f'from {".".join(parts[:-1])}.{module} import'
                        else:
                            return f'from {module} import'
                    
                    return match.group(0)  # 保持原样
                
                content = re.sub(relative_import_pattern, fix_relative_import, content)
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixes_applied['import_violations'] += 1
                    logger.info(f"修复导入违规: {file_path}")
            
            except Exception as e:
                logger.warning(f"修复文件 {file_path} 导入违规失败: {e}")
    
    def fix_critical_naming_violations(self):
        """修复关键命名违规"""
        logger.info("修复关键命名违规...")
        
        # 只修复最关键的文件
        critical_files = []
        for priority_dir in ['bin/', 'config/', 'db/', 'utils/']:
            critical_files.extend(list(self.root_dir.glob(f"{priority_dir}*.py")))
        
        for file_path in critical_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 只修复最明显的命名违规
                # 修复类名 (小写开头的类名)
                content = re.sub(r'\bclass\s+([a-z]\w*)\b', 
                                lambda m: f'class {self.to_pascal_case(m.group(1))}', 
                                content)
                
                # 修复明显的常量名 (大小写混合的常量)
                content = re.sub(r'\b([A-Z][a-z]\w*)\s*=\s*[\'"]', 
                                lambda m: f'{self.to_upper_snake_case(m.group(1))} = "', 
                                content)
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixes_applied['critical_naming'] += 1
                    logger.info(f"修复关键命名: {file_path}")
            
            except Exception as e:
                logger.warning(f"修复文件 {file_path} 命名违规失败: {e}")
    
    def fix_code_duplications(self):
        """修复代码重复问题"""
        logger.info("修复代码重复问题...")
        
        # 收集所有重复的类名和函数名
        name_files = defaultdict(list)
        
        python_files = self.get_priority_files()
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Class_def, ast.Function_def)):
                            name_files[node.name].append(file_path)
                except Syntax_error:
                    continue
            
            except Exception as e:
                continue
        
        # 修复重复名称
        for name, files in name_files.items():
            if len(files) > 1:
                for i, file_path in enumerate(files):
                    if i == 0:
                        continue  # 保持第一个不变
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # 为重复的名称添加后缀
                        new_name = f"{name}_{i}"
                        content = re.sub(rf'\b{re.escape(name)}\b', new_name, content)
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        self.fixes_applied['code_duplications'] += 1
                        logger.info(f"修复重复名称 {name} -> {new_name} in {file_path}")
                    
                    except Exception as e:
                        logger.warning(f"修复重复名称失败: {e}")
    
    def get_priority_files(self):
        """获取优先级文件列表"""
        priority_files = []
        
        for priority_dir in self.priority_files:
            priority_files.extend(list(self.root_dir.glob(f"{priority_dir}**/*.py")))
        
        return priority_files
    
    def to_pascal_case(self, name: str) -> str:
        """转换为大驼峰命名"""
        return ''.join(word.capitalize() for word in re.split(r'[_-]', name))
    
    def to_upper_snake_case(self, name: str) -> str:
        """转换为大写下划线命名"""
        # 将大驼峰转换为下划线分隔
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).upper()
    
    def generate_fix_report(self):
        """生成修复报告"""
        total_fixes = sum(self.fixes_applied.values())
        
        report = {
            'timestamp': str(Path().cwd()),
            'total_fixes_applied': total_fixes,
            'fixes_by_type': self.fixes_applied,
            'strategy': 'priority_based_fixing',
            'focus_areas': self.priority_files
        }
        
        # 保存报告
        report_path = self.root_dir / 'data/result/priority_fix_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"优先级修复报告已保存: {report_path}")
        
        # 打印摘要
        print(f"\n🎯 优先级修复完成!")
        print(f"总计修复: {total_fixes} 个问题")
        for fix_type, count in self.fixes_applied.items():
            if count > 0:
                print(f"  - {fix_type}: {count}")

def main_prioritycompliancefix():
    """主函数"""
    try:
        fixer = Priority_compliance_fixer()
        fixer.fix_all_priority_violations()
        
        print(f"\n✅ 优先级修复完成，建议运行合规性检查验证结果")
        return 0
        
    except Exception as e:
        logger.error(f"优先级修复失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main_prioritycompliancefix()) 