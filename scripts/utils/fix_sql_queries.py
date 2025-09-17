#!/usr/bin/env python3
"""
自动修复SQL分散问题

扫描代码中的SQL语句，并替换为统一SQL管理系统的调用。
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class SQLQueryFixer:
    """SQL查询修复器"""
    
    def __init__(self):
        self.fixed_files = []
        self.sql_patterns = self._init_sql_patterns()
        self.common_queries = self._init_common_queries()
    
    def _init_sql_patterns(self) -> List[re.Pattern]:
        """初始化SQL匹配模式"""
        return [
            # SELECT语句
            re.compile(r'["\']SELECT\s+.*?FROM\s+\w+.*?["\']', re.IGNORECASE | re.DOTALL),
            # INSERT语句
            re.compile(r'["\']INSERT\s+INTO\s+\w+.*?["\']', re.IGNORECASE | re.DOTALL),
            # UPDATE语句
            re.compile(r'["\']UPDATE\s+\w+\s+SET.*?["\']', re.IGNORECASE | re.DOTALL),
            # DELETE语句
            re.compile(r'["\']DELETE\s+FROM\s+\w+.*?["\']', re.IGNORECASE | re.DOTALL),
            # CREATE语句
            re.compile(r'["\']CREATE\s+TABLE.*?["\']', re.IGNORECASE | re.DOTALL),
            # 多行SQL（三引号）
            re.compile(r'"""[\s\S]*?SELECT[\s\S]*?FROM[\s\S]*?"""', re.IGNORECASE),
            re.compile(r"'''[\s\S]*?SELECT[\s\S]*?FROM[\s\S]*?'''", re.IGNORECASE),
        ]
    
    def _init_common_queries(self) -> Dict[str, str]:
        """初始化常见查询的替换模式"""
        return {
            # 股票数据查询
            r'SELECT\s+.*?FROM\s+stock_info\s+WHERE\s+code\s*=.*?AND\s+date\s+BETWEEN.*?ORDER\s+BY\s+date': 
                'executor.get_stock_data(code, start_date, end_date, level)',
            
            # 股票列表查询
            r'SELECT\s+DISTINCT\s+code.*?FROM\s+stock_info\s+WHERE\s+date\s*=.*?ORDER\s+BY\s+code':
                'executor.get_stock_list(level)',
            
            # 股票信息查询
            r'SELECT\s+code,\s*name,\s*industry.*?FROM\s+stock_info\s+WHERE\s+code\s*=.*?GROUP\s+BY':
                'executor.get_stock_info(code, level)',
            
            # 行业列表查询
            r'SELECT\s+DISTINCT\s+industry.*?FROM\s+stock_info.*?GROUP\s+BY\s+industry':
                'executor.get_industry_list(level)',
            
            # 股票数量查询
            r'SELECT\s+COUNT\(DISTINCT\s+code\).*?FROM\s+stock_info':
                'executor.get_stock_count(level)',
            
            # 最新数据查询
            r'SELECT\s+.*?FROM\s+stock_info\s+WHERE\s+code\s*=.*?ORDER\s+BY\s+date\s+DESC\s+LIMIT':
                'executor.get_latest_data(code, limit, level)',
        }
    
    def scan_file_fix_sql_queries(self, file_path: str) -> List[Tuple[int, str]]:
        """扫描文件中的SQL语句
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Tuple[int, str]]: 发现的SQL语句列表 (行号, SQL内容)
        """
        sql_queries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # 使用所有模式匹配SQL语句
            for pattern in self.sql_patterns:
                for match in pattern.finditer(content):
                    sql_text = match.group(0)
                    
                    # 找到SQL语句所在的行号
                    line_num = content[:match.start()].count('\n') + 1
                    
                    # 清理SQL文本
                    cleaned_sql = self._clean_sql(sql_text)
                    if cleaned_sql:
                        sql_queries.append((line_num, cleaned_sql))
        
        except Exception as e:
            logger.error(f"扫描文件失败 {file_path}: {e}")
        
        return sql_queries
    
    def _clean_sql(self, sql_text: str) -> str:
        """清理SQL文本
        
        Args:
            sql_text: 原始SQL文本
            
        Returns:
            str: 清理后的SQL文本
        """
        # 移除引号
        sql_text = sql_text.strip('"\'')
        
        # 移除三引号
        sql_text = re.sub(r'^"""', '', sql_text)
        sql_text = re.sub(r'"""$', '', sql_text)
        sql_text = re.sub(r"^'''", '', sql_text)
        sql_text = re.sub(r"'''$", '', sql_text)
        
        # 移除多余的空白字符
        sql_text = re.sub(r'\s+', ' ', sql_text).strip()
        
        # 过滤掉太短的SQL（可能是误匹配）
        if len(sql_text) < 20:
            return ""
        
        return sql_text
    
    def fix_file_fix_sql_queries(self, file_path: str) -> bool:
        """修复文件中的SQL语句
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否有修改
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            modified = False
            
            # 检查是否需要添加导入
            sql_queries = self.scan_file(file_path)
            if sql_queries:
                # 添加导入语句
                if 'from db.query_executor import get_query_executor' not in content:
                    import_pos = self._find_import_position(content)
                    if import_pos is not None:
                        import_lines = [
                            "from db.query_executor import get_query_executor",
                            "from db.sql_manager import QueryType",
from db.sql_manager import SQLManager, QueryType
                            ""
                        ]
                        import_text = '\n'.join(import_lines)
                        content = content[:import_pos] + import_text + content[import_pos:]
                        modified = True
                
                # 添加执行器实例化
                if 'executor = get_query_executor()' not in content:
                    # 在类定义或函数定义前添加
                    class_match = re.search(r'^class\s+\w+.*?:', content, re.MULTILINE)
                    if class_match:
                        pos = class_match.start()
                        content = content[:pos] + "executor = get_query_executor()\n\n" + content[pos:]
                        modified = True
            
            # 应用简单的替换（注释掉复杂的SQL，添加TODO）
            for pattern in self.sql_patterns:
                matches = list(pattern.finditer(content))
                for match in reversed(matches):  # 从后往前替换，避免位置偏移
                    sql_text = match.group(0)
                    
                    # 生成替换文本
                    replacement = self._generate_replacement(sql_text)
                    if replacement:
                        content = content[:match.start()] + replacement + content[match.end():]
                        modified = True
            
            # 如果有修改，写回文件
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"修复文件: {file_path}")
                self.fixed_files.append(file_path)
                return True
        
        except Exception as e:
            logger.error(f"修复文件失败 {file_path}: {e}")
        
        return False
    
    def _generate_replacement(self, sql_text: str) -> str:
        """生成替换文本
        
        Args:
            sql_text: SQL文本
            
        Returns:
            str: 替换文本
        """
        cleaned_sql = self._clean_sql(sql_text)
        
        # 检查是否是常见查询模式
        for pattern, replacement in self.common_queries.items():
            if re.search(pattern, cleaned_sql, re.IGNORECASE):
                return f"# TODO: 使用统一查询接口替换\n        # {replacement}\n        # 原SQL: {cleaned_sql[:100]}..."
        
        # 默认替换
        return f"# TODO: 迁移到SQL管理系统\n        # 原SQL: {cleaned_sql[:100]}...\n        # 使用 executor.execute_custom_query() 或添加到 sql_manager.py"
    
    def _find_import_position(self, content: str) -> int:
        """找到合适的导入位置
        
        Args:
            content: 文件内容
            
        Returns:
            int: 插入位置，如果找不到则返回None
        """
        lines = content.split('\n')
        
        # 找到最后一个import语句的位置
        last_import_line = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                last_import_line = i
        
        if last_import_line >= 0:
            # 在最后一个import语句后插入
            pos = 0
            for i in range(last_import_line + 1):
                pos += len(lines[i]) + 1  # +1 for newline
            return pos
        
        # 如果没有找到import语句，在文件开头插入
        return 0
    
    def scan_directory_fix_sql_queries(self, directory: str, extensions: Set[str] = None) -> Dict[str, List[Tuple[int, str]]]:
        """扫描目录中的SQL语句
        
        Args:
            directory: 目录路径
            extensions: 文件扩展名集合
            
        Returns:
            Dict[str, List[Tuple[int, str]]]: 文件路径到SQL语句列表的映射
        """
        if extensions is None:
            extensions = {'.py'}
        
        sql_files = {}
        
        for root, dirs, files in os.walk(directory):
            # 跳过特定目录
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'venv', 'env'}]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    sql_queries = self.scan_file(file_path)
                    if sql_queries:
                        sql_files[file_path] = sql_queries
        
        return sql_files
    
    def generate_report_fix_sql_queries(self, sql_files: Dict[str, List[Tuple[int, str]]]) -> str:
        """生成扫描报告
        
        Args:
            sql_files: SQL文件字典
            
        Returns:
            str: 报告内容
        """
        report = ["SQL语句分散扫描报告", "=" * 50, ""]
        
        total_queries = sum(len(queries) for queries in sql_files.values())
        report.append(f"总计发现 {total_queries} 个SQL语句")
        report.append(f"涉及 {len(sql_files)} 个文件")
        report.append("")
        
        for file_path, queries in sql_files.items():
            report.append(f"文件: {file_path}")
            report.append("-" * 40)
            
            for line_num, sql_text in queries:
                report.append(f"  行 {line_num}: {sql_text[:80]}...")
            
            report.append("")
        
        return "\n".join(report)

def main_fix_sql_queries():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python fix_sql_queries.py <scan|fix> [directory]")
        sys.exit(1)
    
    action = sys.argv[1]
    directory = sys.argv[2] if len(sys.argv) > 2 else root_dir
    
    fixer = SQLQueryFixer()
    
    if action == 'scan':
        print(f"扫描目录: {directory}")
        sql_files = fixer.scan_directory(directory)
        
        if sql_files:
            report = fixer.generate_report(sql_files)
            print(report)
            
            # 保存报告
            report_file = os.path.join(root_dir, 'sql_queries_report.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n报告已保存到: {report_file}")
        else:
            print("未发现SQL语句")
    
    elif action == 'fix':
        print(f"修复目录: {directory}")
        print("注意: 这将在SQL语句位置添加TODO注释，需要手动完成迁移")
        
        # 确认操作
        response = input("确定要继续吗？(y/N): ")
        if response.lower() != 'y':
            print("操作已取消")
            return
        
        # 只修复特定目录，避免破坏关键文件
        safe_dirs = ['analysis', 'strategy', 'examples', 'tests']
        target_dir = os.path.basename(directory)
        
        if target_dir not in safe_dirs and directory != root_dir:
            print(f"为安全起见，只能修复以下目录: {safe_dirs}")
            return
        
        fixed_count = fixer.fix_directory(directory)
        
        if fixed_count > 0:
            print(f"成功修复 {fixed_count} 个文件")
            print("修复的文件:")
            for file_path in fixer.fixed_files:
                print(f"  - {file_path}")
            print("\n请手动检查修复结果，并完成SQL迁移")
        else:
            print("未发现需要修复的文件")
    
    else:
        print("无效的操作，请使用 'scan' 或 'fix'")
        sys.exit(1)

if __name__ == "__main__":
    main_fix_sql_queries() 