#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
专门修复mtm.py中赋值错误的工具
针对用户修改后引入的缺少等号问题
"""

import re

def fix_mtm_assignments():
    """修复mtm.py中的赋值错误"""
    file_path = "indicators/mtm.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 记录修改
    changes = []
    
    # 修复模式：变量名 + 空格 + 值（不是等号赋值的情况）
    patterns = [
        # 基本变量赋值
        (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*(?:\([^)]*\))?)', r'\1\2 = \3'),
        # df_copy df.copy() 类型
        (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+(df\.[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\))', r'\1\2 = \3'),
        # 数字赋值
        (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+(\d+(?:\.\d+)?)', r'\1\2 = \3'),
        # 字符串赋值
        (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+(".*?")', r'\1\2 = \3'),
        # None、True、False赋值
        (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+(None|True|False)', r'\1\2 = \3'),
        # 复杂表达式赋值
        (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+([\(].*)', r'\1\2 = \3'),
        # 数组/字典索引赋值
        (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*\[[^\]]+\])\s+([a-zA-Z_].*)', r'\1\2 = \3'),
        # self.属性赋值
        (r'^(\s+)(self\.[a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_].*)', r'\1\2 = \3'),
    ]
    
    lines = content.split('\n')
    modified_lines = []
    
    for i, line in enumerate(lines):
        original_line = line
        
        # 跳过注释行、空行、包含等号的行
        if line.strip().startswith('#') or not line.strip() or '=' in line:
            modified_lines.append(line)
            continue
            
        # 跳过函数定义、类定义等
        if any(keyword in line for keyword in ['def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ', 'return']):
            modified_lines.append(line)
            continue
        
        # 应用修复模式
        for pattern, replacement in patterns:
            new_line = re.sub(pattern, replacement, line)
            if new_line != line:
                changes.append(f"Line {i+1}: {line.strip()} → {new_line.strip()}")
                line = new_line
                break
        
        modified_lines.append(line)
    
    # 写回文件
    new_content = '\n'.join(modified_lines)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ 修复完成，共修改 {len(changes)} 行")
    for change in changes[:10]:  # 只显示前10个修改
        print(f"  📝 {change}")
    if len(changes) > 10:
        print(f"  ... 还有 {len(changes) - 10} 个修改")

if __name__ == "__main__":
    fix_mtm_assignments()
