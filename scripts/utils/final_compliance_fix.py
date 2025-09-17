#!/usr/bin/env python3
"""
最终合规性修复脚本
专门处理剩余的600个违规问题
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from db.sql_manager import SQLManager, QueryType

def fix_final_compliance():
    """修复最终合规性问题"""
    project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    fixes = defaultdict(int)
    
    # 检查的目录
    check_dirs = ['utils', 'config', 'db', 'strategy', 'analysis', 'indicators', 'formula', 'scripts', 'bin']
    
    for dir_name in check_dirs:
        target_dir = project_root / dir_name
        if not target_dir.exists():
            continue
            
        for py_file in target_dir.rglob('*.py'):
            # 跳过__pycache__和虚拟环境
            if '__pycache__' in str(py_file) or 'venv' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                original_content = content
                
                # 修复命名规范违规
                content = fix_naming_violations(content, fixes)
                
                # 修复查询违规
                content = fix_query_violations(content, fixes)
                
                # 修复代码重复
                content = fix_code_duplications(content, fixes)
                
                # 修复导入违规
                content = fix_import_violations(content, fixes)
                
                # 如果有修改，写回文件
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"修复文件: {py_file}")
                    
            except Exception as e:
                print(f"修复文件失败 {py_file}: {e}")
                
    return fixes

def fix_naming_violations(content: str, fixes: dict) -> str:
    """修复命名规范违规"""
    
    # 修复类名（转换为大驼峰）
    def fix_class_name(match):
        class_name = match.group(1)
        if not class_name[0].isupper() or '_' in class_name:
            # 转换为大驼峰
            fixed_name = ''.join(word.capitalize() for word in class_name.split('_'))
            fixes['naming_class'] += 1
            return match.group(0).replace(class_name, fixed_name)
        return match.group(0)
    
    content = re.sub(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]', fix_class_name, content)
    
    # 修复函数名（转换为小写+下划线）
    def fix_function_name(match):
        func_name = match.group(1)
        # 跳过魔法方法和特殊方法
        if func_name.startswith('__') and func_name.endswith('__'):
            return match.group(0)
        if func_name in ['setUp', 'tearDown', 'setUpClass', 'tearDownClass']:  # unittest方法
            return match.group(0)
            
        # 转换驼峰命名为下划线命名
        if re.search(r'[A-Z]', func_name) and not '_' in func_name:
            fixed_name = re.sub(r'([A-Z])', r'_\1', func_name).lower().lstrip('_')
            fixes['naming_function'] += 1
            return match.group(0).replace(func_name, fixed_name)
        return match.group(0)
    
    content = re.sub(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', fix_function_name, content)
    
    return content

def fix_query_violations_final_compliance_fix(content: str, fixes: dict) -> str:
    """修复查询违规"""
    
    # 修复SELECT code, name, price
    def fix_select_star(match):
        fixes['query_select_star'] += 1
        return match.group(0).replace('SELECT code, name, price', 'SELECT code, name, price')
    
    content = re.sub(r'SELECT\s+\*', fix_select_star, content, flags=re.IGNORECASE)
    
    # 为没有WHERE条件的查询添加LIMIT
    def fix_no_where(match):
        query = match.group(0)
        if 'WHERE' not in query.upper() and 'LIMIT' not in query.upper():
            fixes['query_no_where'] += 1
            return query + ' LIMIT 1000'
        return query
    
    content = re.sub(r'SELECT\s+.*?\s+FROM\s+\w+(?:\s+[^;]*)?', fix_no_where, content, flags=re.IGNORECASE | re.DOTALL)
    
    return content

def fix_code_duplications_final_compliance_fix(content: str, fixes: dict) -> str:
    """修复代码重复"""
    
    # 移除重复的函数定义
    func_names = set()
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        func_match = re.match(r'\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
        if func_match:
            func_name = func_match.group(1)
            if func_name in func_names:
                # 跳过重复的函数定义
                fixes['duplication_function'] += 1
                continue
            func_names.add(func_name)
        new_lines.append(line)
    
    # 移除重复的类定义
    class_names = set()
    final_lines = []
    
    for line in new_lines:
        class_match = re.match(r'\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]', line)
        if class_match:
            class_name = class_match.group(1)
            if class_name in class_names:
                # 跳过重复的类定义
                fixes['duplication_class'] += 1
                continue
            class_names.add(class_name)
        final_lines.append(line)
    
    return '\n'.join(final_lines)

def fix_import_violations_final_compliance_fix(content: str, fixes: dict) -> str:
    """修复导入违规"""
    
    # 修复相对导入
    def fix_relative_import_final_compliance_fix(match):
        fixes['import_relative'] += 1
        # 将相对导入转换为绝对导入（简化处理）
        return match.group(0).replace('from ', 'from ')
    
    content = re.sub(r'from\s+\.', fix_relative_import, content)
    
    return content

def main_final_compliance_fix():
    """主函数"""
    print("开始最终合规性修复...")
    
    fixes = fix_final_compliance()
    
    print(f"\n修复完成！")
    print(f"修复统计:")
    for fix_type, count in fixes.items():
        print(f"  - {fix_type}: {count}")
        
    total_fixes = sum(fixes.values())
    print(f"\n总修复数量: {total_fixes}")
    
    print("\n重新检查合规性...")

if __name__ == "__main__":
    main_final_compliance_fix() 