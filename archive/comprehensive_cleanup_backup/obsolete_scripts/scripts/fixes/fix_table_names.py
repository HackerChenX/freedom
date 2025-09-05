#!/usr/bin/env python3
"""
修复所有Python文件中错误的表名
将 stock_data, stock_daily_data 等错误表名修复为 stock_info WHERE 1=1 """

import os
import re
import glob

def fix_table_names():
    """修复表名"""
    
    # 需要修复的文件模式
    patterns = [
        "**/*.py",
        "scripts/**/*.py", 
        "analysis/**/*.py",
        "strategy/**/*.py"
    ]
    
    # 错误的表名到正确表名的映射
    table_fixes = [
        (r'\bstock_data\b', 'stock_info'),
        (r'\bstock_daily_data\b', 'stock_info'),
        (r'FROM\s+stock_info\s+WHERE\s+(?!.*level\s*=)', 'FROM stock_info WHERE level = \'日线\' AND '),
        (r'FROM\s+stock_info\s+WHERE\s+(.*?)(?=\s+ORDER|\s+GROUP|\s+LIMIT|$)', 
         lambda m: f"FROM stock_info WHERE level = '日线' AND {m.group(1).strip()}")
    ]
    
    files_fixed = 0
    total_fixes = 0
    
    # 收集所有Python文件
    all_files = set()
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.update(files)
    
    print(f"检查 {len(all_files)} 个Python文件...")
    
    for file_path in all_files:
        if not file_path.endswith('.py'):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            file_fixes = 0
            
            # 应用修复
            for pattern, replacement in table_fixes:
                if callable(replacement):
                    matches = list(re.finditer(pattern, content))
                    for match in reversed(matches):  # 从后往前替换避免位置偏移
                        new_text = replacement(match)
                        content = content[:match.start()] + new_text + content[match.end():]
                        file_fixes += 1
                else:
                    new_content = re.sub(pattern, replacement, content)
                    if new_content != content:
                        file_fixes += re.subn(pattern, replacement, content)[1]
                        content = new_content
            
            # 如果有修改，写回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                files_fixed += 1
                total_fixes += file_fixes
                print(f"✅ 修复 {file_path}: {file_fixes} 处")
                
        except Exception as e:
            print(f"❌ 处理文件 {file_path} 时出错: {e}")
    
    print(f"\n修复完成:")
    print(f"  修复文件数: {files_fixed}")
    print(f"  总修复数: {total_fixes}")

if __name__ == "__main__":
    fix_table_names() 