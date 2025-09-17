#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTM文件完整修复工具
基于用户修改，系统性修复所有剩余的语法错误
"""

import re
import subprocess

def fix_mtm_file():
    """完整修复mtm.py文件"""
    file_path = "indicators/mtm.py"
    
    print("🔧 开始完整修复 mtm.py 文件...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 记录修改
    changes = []
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        original_line = line
        
        # 跳过已经正确的行（包含=的赋值、注释、函数定义等）
        if (line.strip().startswith('#') or 
            not line.strip() or 
            ' = ' in line or
            any(keyword in line for keyword in ['def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ', 'return', 'import', 'from '])):
            fixed_lines.append(line)
            continue
        
        # 应用赋值修复模式
        patterns = [
            # 基本变量赋值
            (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*(?:\([^)]*\))?)', r'\1\2 = \3'),
            # pandas操作
            (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*\[[^\]]+\])\s+([^=].*)', r'\1\2 = \3'),
            # self属性赋值
            (r'^(\s+)(self\.[a-zA-Z_][a-zA-Z0-9_]*)\s+([^=].*)', r'\1\2 = \3'),
            # 数字和布尔值赋值
            (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+(True|False|None|\d+(?:\.\d+)?)', r'\1\2 = \3'),
            # 复杂表达式
            (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+(\(.*)', r'\1\2 = \3'),
            # 字符串赋值
            (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+(".*?")', r'\1\2 = \3'),
        ]
        
        for pattern, replacement in patterns:
            new_line = re.sub(pattern, replacement, line)
            if new_line != line:
                changes.append(f"第{i+1}行: {original_line.strip()} → {new_line.strip()}")
                line = new_line
                break
        
        fixed_lines.append(line)
    
    # 写回文件
    new_content = '\n'.join(fixed_lines)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ 修复完成，共修改 {len(changes)} 行")
    for change in changes[:10]:  # 显示前10个修改
        print(f"  📝 {change}")
    if len(changes) > 10:
        print(f"  ... 还有 {len(changes) - 10} 个修改")
    
    # 验证语法
    result = subprocess.run(['python3', '-m', 'py_compile', file_path], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {file_path} 语法修复成功！")
        return True
    else:
        print(f"⚠️  {file_path} 修复后仍有语法错误:")
        print(f"   {result.stderr.strip()[:200]}...")
        return False

if __name__ == "__main__":
    fix_mtm_file()
