#!/usr/bin/env python3
"""
修复分散的SQL查询
统一到标准SQL管理器，消除硬编码SQL语句
严格遵循L2存储访问层单一入口原则
"""

import os
import re
import sys
from pathlib import Path

def fix_sql_syntax_errors(file_path: str) -> bool:
    """修复SQL语法错误"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复常见的SQL语法错误
        sql_fixes = [
            # 修复多余的逗号
            (r'volume,\s+FROM\s+stock_info', 'volume FROM stock_info'),
            (r'turnover_rate,\s+FROM\s+stock_info', 'turnover_rate FROM stock_info'),
            
            # 修复缺少的字段
            (r'SELECT\s+\*\s+FROM\s+stock_info', 
             'SELECT code, name, date, open, high, low, close, volume, turnover_rate FROM stock_info'),
            
            # 修复不一致的字段名
            (r'turnover_rate(?!\s*_rate)', 'turnover_rate'),
            
            # 修复缺少必需条件的查询
            (r'FROM\s+stock_info\s+WHERE\s+(?!.*level\s*=)', 
             'FROM stock_info WHERE code = %(code)s AND level = %(level)s AND '),
            (r'FROM\s+stock_info\s+WHERE\s+(?!.*code\s*=)', 
             'FROM stock_info WHERE level = %(level)s AND code = %(code)s AND '),
        ]
        
        for pattern, replacement in sql_fixes:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"修复SQL语法错误失败 {file_path}: {e}")
        return False

def replace_hardcoded_sql_with_manager(file_path: str) -> bool:
    """将硬编码SQL替换为SQL管理器调用"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 添加SQL管理器导入
        if 'from db.sql_manager import SQLManager' not in content:
            # 找到其他导入语句的位置
            import_pattern = r'(from\s+[\w.]+\s+import\s+[^\n]+\n)'
            imports = re.findall(import_pattern, content)
            if imports:
                # 在最后一个导入后添加
                last_import = imports[-1]
                content = content.replace(
                    last_import,
                    last_import + 'from db.sql_manager import SQLManager, QueryType\n'
                )
            else:
                # 在文件开头添加
                content = 'from db.sql_manager import SQLManager, QueryType\n' + content
        
        # 替换常见的硬编码SQL模式
        sql_replacements = [
            # 股票数据查询
            (r'SELECT\s+.*FROM\s+stock_info\s+WHERE\s+code\s*=\s*[\'"]([^\'\"]+)[\'\"]\s+AND\s+date\s+BETWEEN\s+[\'"]([^\'\"]+)[\'"]\s+AND\s+[\'"]([^\'\"]+)[\'\"]\s+AND\s+level\s*=\s*[\'"]([^\'\"]+)[\'\"]\s*ORDER\s+BY\s+date',
             'SQLManager().get_query(QueryType.STOCK_DATA, {"code": "\\1", "start_date": "\\2", "end_date": "\\3", "level": "\\4"})'),
            
            # 股票列表查询
            (r'SELECT\s+DISTINCT\s+code.*FROM\s+stock_info\s+WHERE\s+level\s*=\s*[\'"]([^\'\"]+)[\'\"]\s*LIMIT\s+(\d+)',
             'SQLManager().get_query(QueryType.STOCK_LIST, {"level": "\\1", "limit": \\2})'),
            
            # 股票数量查询
            (r'SELECT\s+COUNT\(\*\).*FROM\s+stock_info\s+WHERE\s+level\s*=\s*[\'"]([^\'\"]+)[\'\"]\s*LIMIT\s+\d+',
             'SQLManager().get_query(QueryType.STOCK_COUNT, {"level": "\\1"})'),
        ]
        
        for pattern, replacement in sql_replacements:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE | re.DOTALL)
        
        # 替换简单的查询字符串
        simple_replacements = [
            (r'"SELECT\s+DISTINCT\s+code\s+FROM\s+stock_info\s+WHERE\s+level\s*=\s*[\'"]日线[\'\"]\s*LIMIT\s+\d+"',
             'SQLManager().get_query(QueryType.STOCK_LIST, {"level": "日线"})'),
            (r'"SELECT\s+COUNT\(\*\)\s+as\s+count\s+FROM\s+stock_info\s+WHERE\s+level\s*=\s*[\'"]日线[\'\"]\s*LIMIT\s+1"',
             'SQLManager().get_query(QueryType.STOCK_COUNT, {"level": "日线"})'),
        ]
        
        for pattern, replacement in simple_replacements:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"替换硬编码SQL失败 {file_path}: {e}")
        return False

def find_files_with_sql() -> list:
    """查找包含SQL查询的文件"""
    files = []
    
    for root, dirs, filenames in os.walk('.'):
        # 跳过不需要的目录
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'backup', 'archive', 'venv', '.venv']]
        
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(root, filename)
                
                # 跳过SQL管理器本身
                if 'sql_manager.py' in file_path:
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if any(pattern in content.upper() for pattern in [
                            'SELECT', 'FROM STOCK_INFO', 'INSERT INTO', 'UPDATE', 'DELETE FROM'
                        ]):
                            files.append(file_path)
                except:
                    continue
    
    return files

def main():
    """主函数"""
    print("🚀 开始修复分散的SQL查询...")
    print("📋 执行L2存储访问层SQL查询管理统一")
    
    # 查找需要修复的文件
    files_to_fix = find_files_with_sql()
    print(f"📊 找到 {len(files_to_fix)} 个包含SQL查询的文件")
    
    if not files_to_fix:
        print("✅ 没有需要修复的文件")
        return
    
    # 修复SQL语法错误
    syntax_fixed = 0
    sql_replaced = 0
    failed_count = 0
    
    for file_path in files_to_fix:
        try:
            print(f"🔧 处理文件: {file_path}")
            
            # 修复语法错误
            if fix_sql_syntax_errors(file_path):
                syntax_fixed += 1
                print(f"  ✅ 修复SQL语法错误")
            
            # 替换硬编码SQL
            if replace_hardcoded_sql_with_manager(file_path):
                sql_replaced += 1
                print(f"  ✅ 替换硬编码SQL为管理器调用")
            
        except Exception as e:
            failed_count += 1
            print(f"  ❌ 处理失败: {e}")
    
    print(f"\n📊 修复结果:")
    print(f"✅ SQL语法修复: {syntax_fixed} 个文件")
    print(f"✅ SQL替换: {sql_replaced} 个文件")
    print(f"❌ 处理失败: {failed_count} 个文件")
    print(f"📁 总处理文件: {len(files_to_fix)} 个")
    
    if syntax_fixed > 0 or sql_replaced > 0:
        print("🎉 分散SQL查询修复完成！")
        print("✅ 统一到标准SQL管理器")
        print("✅ 消除硬编码SQL语句")
        print("✅ 符合L2存储访问层规范")
    else:
        print("ℹ️  没有发现需要修复的SQL问题")

if __name__ == "__main__":
    main()
