#!/usr/bin/env python3
"""
简化SQL迁移工具

扫描系统中分散的SQL查询，生成迁移报告和建议。
不依赖复杂的系统模块，可独立运行。
"""

import os
import re
from typing import Dict, List, Set, Tuple

class SimpleSQLScanner:
    """简化SQL扫描器"""
    
    def __init__(self):
        self.sql_patterns = [
            r'SELECT\s+.*?\s+FROM\s+\w+',
            r'INSERT\s+INTO\s+\w+',
            r'UPDATE\s+\w+\s+SET',
            r'DELETE\s+FROM\s+\w+',
        ]
        self.excluded_dirs = {
            'venv', '__pycache__', '.git', 'node_modules', 
            'logs', 'cache', 'tmp', 'metadata', 'store', '.venv'
        }
    
    def scan_directory(self, directory: str = ".") -> Dict[str, List[str]]:
        """扫描目录中的SQL查询
        
        Args:
            directory: 扫描目录
            
        Returns:
            Dict[str, List[str]]: 文件路径到SQL查询列表的映射
        """
        sql_files = {}
        
        for root, dirs, files in os.walk(directory):
            # 排除不需要的目录
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    queries = self._extract_sql_from_file(file_path)
                    if queries:
                        sql_files[file_path] = queries
        
        return sql_files
    
    def _extract_sql_from_file(self, file_path: str) -> List[str]:
        """从文件中提取SQL查询"""
        queries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 查找SQL模式
            for pattern in self.sql_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    sql = match.group(0).strip()
                    if len(sql) > 20:  # 过滤太短的匹配
                        queries.append(sql)
            
            # 查找字符串中的SQL
            string_queries = self._extract_string_sql(content)
            queries.extend(string_queries)
            
        except Exception as e:
            print(f"警告: 无法读取文件 {file_path}: {e}")
        
        return list(set(queries))  # 去重
    
    def _extract_string_sql(self, content: str) -> List[str]:
        """提取字符串中的SQL"""
        queries = []
        
        # 三引号字符串
        triple_quote_pattern = r'"""(.*?)"""'
        for match in re.finditer(triple_quote_pattern, content, re.DOTALL):
            text = match.group(1)
            if self._is_sql_query(text):
                queries.append(text.strip())
        
        # 单行字符串
        string_patterns = [
            r'"([^"]*(?:SELECT|INSERT|UPDATE|DELETE)[^"]*?)"',
            r"'([^']*(?:SELECT|INSERT|UPDATE|DELETE)[^']*?)'"
        ]
        
        for pattern in string_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                sql = match.group(1).strip()
                if len(sql) > 20:
                    queries.append(sql)
        
        return queries
    
    def _is_sql_query(self, text: str) -> bool:
        """判断文本是否为SQL查询"""
        text_upper = text.upper()
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
        
        for keyword in sql_keywords:
            if keyword in text_upper and ('FROM' in text_upper or 'INTO' in text_upper):
                return True
        
        return False
    
    def categorize_queries(self, sql_files: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
        """分类SQL查询
        
        Args:
            sql_files: 文件到SQL查询的映射
            
        Returns:
            Dict[str, List[Tuple[str, str]]]: 分类结果
        """
        categories = {
            'stock_data': [],
            'batch_queries': [],
            'count_queries': [],
            'list_queries': [],
            'other': []
        }
        
        for file_path, queries in sql_files.items():
            for query in queries:
                query_lower = query.lower()
                category = 'other'
                
                if 'stock_info' in query_lower:
                    if 'count(' in query_lower:
                        category = 'count_queries'
                    elif 'distinct' in query_lower:
                        category = 'list_queries'
                    elif 'in (' in query_lower or 'any(' in query_lower:
                        category = 'batch_queries'
                    else:
                        category = 'stock_data'
                
                categories[category].append((file_path, query))
        
        return categories
    
    def generate_report(self, sql_files: Dict[str, List[str]], output_file: str = "sql_migration_report.md"):
        """生成迁移报告"""
        categories = self.categorize_queries(sql_files)
        
        total_files = len(sql_files)
        total_queries = sum(len(queries) for queries in sql_files.values())
        
        report = f"""# SQL查询迁移报告

## 概述

- **扫描文件数**: {total_files}
- **发现SQL查询数**: {total_queries}
- **生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 文件统计

"""
        
        for file_path, queries in sorted(sql_files.items()):
            report += f"### {file_path}\n"
            report += f"- 查询数量: {len(queries)}\n\n"
            for i, query in enumerate(queries, 1):
                report += f"{i}. ```sql\n{query[:200]}{'...' if len(query) > 200 else ''}\n```\n\n"
        
        report += """## 分类统计

"""
        
        for category, items in categories.items():
            if items:
                report += f"### {category.replace('_', ' ').title()}\n"
                report += f"- 数量: {len(items)}\n\n"
        
        report += """## 迁移建议

1. **优先级1**: stock_data 和 batch_queries - 核心业务查询
2. **优先级2**: count_queries 和 list_queries - 统计和列表查询
3. **优先级3**: other - 其他查询

## 迁移步骤

1. 使用 `db/sql_manager.py` 定义标准查询模板
2. 使用 `db/query_executor.py` 替换直接SQL调用
3. 测试验证迁移效果
4. 更新相关文档

"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"迁移报告已生成: {output_file}")
        return output_file

def main_simple_sql_migration():
    """主函数"""
    print("=== 简化SQL迁移工具 ===")
    
    scanner = SimpleSQLScanner()
    
    # 扫描SQL文件
    print("正在扫描SQL查询...")
    sql_files = scanner.scan_directory()
    
    if not sql_files:
        print("未发现包含SQL的文件")
        return
    
    # 统计信息
    total_files = len(sql_files)
    total_queries = sum(len(queries) for queries in sql_files.values())
    
    print(f"\n扫描结果:")
    print(f"- 包含SQL的文件: {total_files} 个")
    print(f"- 发现SQL查询: {total_queries} 个")
    
    # 显示文件列表
    print(f"\n文件列表:")
    for file_path, queries in sorted(sql_files.items()):
        print(f"  {file_path}: {len(queries)} 个查询")
    
    # 生成报告
    report_file = scanner.generate_report(sql_files)
    print(f"\n详细报告: {report_file}")
    
    # 分类统计
    categories = scanner.categorize_queries(sql_files)
    print(f"\n分类统计:")
    for category, items in categories.items():
        if items:
            print(f"  {category}: {len(items)} 个")

if __name__ == "__main__":
    main_simple_sql_migration() 