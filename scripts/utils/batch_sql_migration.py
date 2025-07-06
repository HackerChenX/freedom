#!/usr/bin/env python3
"""
批量SQL迁移工具

将分散在各个文件中的SQL查询迁移到统一的SQL管理系统。
遵循架构规范，提供自动化的迁移和验证功能。
"""

import os
import sys
import re
import ast
import logging
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from db.sql_manager import QueryType, get_sql_manager

logger = get_logger(__name__)

class SQLMigrationTool:
    """SQL迁移工具"""
    
    def __init__(self):
        self.sql_manager = get_sql_manager()
        self.sql_patterns = [
            # 基本SQL模式
            r'SELECT\s+.*?\s+FROM\s+\w+',
            r'INSERT\s+INTO\s+\w+',
            r'UPDATE\s+\w+\s+SET',
            r'DELETE\s+FROM\s+\w+',
            r'CREATE\s+TABLE\s+\w+',
            r'ALTER\s+TABLE\s+\w+',
            r'DROP\s+TABLE\s+\w+',
            # 字符串中的SQL
            r'["\']SELECT\s+.*?FROM\s+.*?["\']',
            r'["\']INSERT\s+INTO\s+.*?["\']',
            r'["\']UPDATE\s+.*?SET\s+.*?["\']',
            r'["\']DELETE\s+FROM\s+.*?["\']',
        ]
        self.excluded_dirs = {
            'venv', '__pycache__', '.git', 'node_modules', 
            'logs', 'cache', 'tmp', 'metadata', 'store'
        }
        self.target_files = []
    
    def scan_sql_files(self, directory: str = ".") -> List[str]:
        """扫描包含SQL的文件
        
        Args:
            directory: 扫描目录
            
        Returns:
            List[str]: 包含SQL的文件列表
        """
        sql_files = []
        
        for root, dirs, files in os.walk(directory):
            # 排除不需要的目录
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if self._contains_sql(file_path):
                        sql_files.append(file_path)
        
        return sql_files
    
    def _contains_sql(self, file_path: str) -> bool:
        """检查文件是否包含SQL语句
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否包含SQL
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 检查SQL关键词模式
            for pattern in self.sql_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                    return True
                    
            return False
        except Exception as e:
            logger.warning(f"无法读取文件 {file_path}: {e}")
            return False
    
    def extract_sql_queries(self, file_path: str) -> List[Dict[str, str]]:
        """从文件中提取SQL查询
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Dict[str, str]]: SQL查询信息列表
        """
        queries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找字符串中的SQL
            string_queries = self._extract_string_sql(content)
            for query in string_queries:
                queries.append({
                    'file': file_path,
                    'type': 'string',
                    'sql': query,
                    'context': self._get_sql_context(content, query)
                })
            
            # 查找变量赋值中的SQL
            variable_queries = self._extract_variable_sql(content)
            for query in variable_queries:
                queries.append({
                    'file': file_path,
                    'type': 'variable',
                    'sql': query,
                    'context': self._get_sql_context(content, query)
                })
                
        except Exception as e:
            logger.error(f"提取SQL失败 {file_path}: {e}")
        
        return queries
    
    def _extract_string_sql(self, content: str) -> List[str]:
        """提取字符串中的SQL"""
        queries = []
        
        # 匹配三引号字符串中的SQL
        triple_quote_pattern = r'"""(.*?)"""'
        for match in re.finditer(triple_quote_pattern, content, re.DOTALL):
            text = match.group(1)
            if self._is_sql_query(text):
                queries.append(text.strip())
        
        # 匹配单引号字符串中的SQL
        single_quote_pattern = r"'([^']*(?:SELECT|INSERT|UPDATE|DELETE)[^']*?)'"
        for match in re.finditer(single_quote_pattern, content, re.IGNORECASE):
            queries.append(match.group(1).strip())
        
        # 匹配双引号字符串中的SQL
        double_quote_pattern = r'"([^"]*(?:SELECT|INSERT|UPDATE|DELETE)[^"]*?)"'
        for match in re.finditer(double_quote_pattern, content, re.IGNORECASE):
            queries.append(match.group(1).strip())
            
        return queries
    
    def _extract_variable_sql(self, content: str) -> List[str]:
        """提取变量赋值中的SQL"""
        queries = []
        
        # 匹配变量赋值
        assignment_pattern = r'(\w+)\s*=\s*(["\'])([^"\']*(?:SELECT|INSERT|UPDATE|DELETE)[^"\']*?)\2'
        for match in re.finditer(assignment_pattern, content, re.IGNORECASE):
            queries.append(match.group(3).strip())
            
        return queries
    
    def _is_sql_query(self, text: str) -> bool:
        """判断文本是否为SQL查询"""
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
        text_upper = text.upper()
        
        for keyword in sql_keywords:
            if keyword in text_upper and 'FROM' in text_upper:
                return True
        
        return False
    
    def _get_sql_context(self, content: str, sql: str) -> str:
        """获取SQL的上下文信息"""
        try:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if sql[:50] in line:  # 匹配前50个字符
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context_lines = lines[start:end]
                    return '\n'.join(f"{start + j + 1}: {l}" for j, l in enumerate(context_lines))
        except:
            pass
        return "无法获取上下文"
    
    def suggest_query_type(self, sql: str) -> QueryType:
        """建议查询类型
        
        Args:
            sql: SQL查询语句
            
        Returns:
            QueryType: 建议的查询类型
        """
        sql_lower = sql.lower()
        
        if 'stock_info' in sql_lower:
            if 'count(' in sql_lower:
                return QueryType.STOCK_COUNT
            elif 'distinct code' in sql_lower:
                return QueryType.STOCK_LIST
            elif 'code in' in sql_lower or 'code = any' in sql_lower:
                return QueryType.BATCH_STOCK_DATA
            else:
                return QueryType.STOCK_DATA
        
        if 'indicator' in sql_lower:
            return QueryType.INDICATOR_DATA
        
        if 'strategy' in sql_lower:
            return QueryType.STRATEGY_CONFIG
        
        if 'industry' in sql_lower:
            return QueryType.INDUSTRY_LIST
        
        # 默认返回通用查询类型
        return QueryType.STOCK_DATA
    
    def generate_migration_plan(self, directory: str = ".") -> Dict[str, List[Dict]]:
        """生成迁移计划
        
        Args:
            directory: 扫描目录
            
        Returns:
            Dict[str, List[Dict]]: 迁移计划
        """
        logger.info("开始扫描SQL文件...")
        sql_files = self.scan_sql_files(directory)
        
        migration_plan = {}
        total_queries = 0
        
        for file_path in sql_files:
            logger.info(f"分析文件: {file_path}")
            queries = self.extract_sql_queries(file_path)
            
            if queries:
                migration_plan[file_path] = []
                for query_info in queries:
                    suggested_type = self.suggest_query_type(query_info['sql'])
                    migration_plan[file_path].append({
                        'sql': query_info['sql'],
                        'type': query_info['type'],
                        'suggested_query_type': suggested_type,
                        'context': query_info['context']
                    })
                    total_queries += 1
        
        logger.info(f"扫描完成: 发现 {len(sql_files)} 个文件，{total_queries} 个SQL查询")
        return migration_plan
    
    def create_migration_script(self, migration_plan: Dict[str, List[Dict]], 
                              output_file: str = "sql_migration_script.py") -> str:
        """创建迁移脚本
        
        Args:
            migration_plan: 迁移计划
            output_file: 输出文件名
            
        Returns:
            str: 生成的脚本路径
        """
        script_content = '''#!/usr/bin/env python3
"""
自动生成的SQL迁移脚本

此脚本用于将分散的SQL查询迁移到统一的SQL管理系统。
请仔细审查每个迁移项，确保迁移的正确性。
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.sql_manager import get_sql_manager, QueryType
from db.query_executor import get_query_executor

def migrate_sql_queries():
    """执行SQL查询迁移"""
    sql_manager = get_sql_manager()
    query_executor = get_query_executor()
    
    print("开始SQL查询迁移...")
    
    # 迁移计划
    migrations = [
'''
        
        for file_path, queries in migration_plan.items():
            script_content += f'\n        # 文件: {file_path}\n'
            for i, query_info in enumerate(queries):
                script_content += f'''        {{
            'file': '{file_path}',
            'query_id': '{os.path.basename(file_path)}_{i}',
            'sql': """{query_info['sql']}""",
            'suggested_type': QueryType.{query_info['suggested_query_type'].name},
            'context': """{query_info['context']}"""
        }},
'''
        
        script_content += '''
    ]
    
    for migration in migrations:
        print(f"处理文件: {migration['file']}")
        print(f"查询ID: {migration['query_id']}")
        print(f"建议类型: {migration['suggested_type']}")
        print(f"SQL: {migration['sql'][:100]}...")
        print("=" * 50)
        
        # 这里可以添加实际的迁移逻辑
        # 例如：替换文件中的SQL为统一接口调用
        
    print(f"迁移完成，共处理 {len(migrations)} 个SQL查询")

if __name__ == "__main__":
    migrate_sql_queries()
'''
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        logger.info(f"迁移脚本已生成: {output_file}")
        return output_file

def main():
    """主函数"""
    print("=== SQL迁移工具 ===")
    
    migration_tool = SQLMigrationTool()
    
    # 生成迁移计划
    migration_plan = migration_tool.generate_migration_plan()
    
    if not migration_plan:
        print("未发现需要迁移的SQL文件")
        return
    
    # 打印摘要
    print(f"\n发现 {len(migration_plan)} 个文件需要迁移:")
    total_queries = 0
    for file_path, queries in migration_plan.items():
        print(f"  {file_path}: {len(queries)} 个查询")
        total_queries += len(queries)
    
    print(f"\n总计: {total_queries} 个SQL查询需要迁移")
    
    # 生成迁移脚本
    script_path = migration_tool.create_migration_script(migration_plan)
    print(f"\n迁移脚本已生成: {script_path}")
    print("请审查脚本内容，然后执行迁移")

if __name__ == "__main__":
    main() 