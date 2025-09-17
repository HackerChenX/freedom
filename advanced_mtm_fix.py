#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTM高级修复工具 - 处理复杂表达式赋值错误
"""

import re
import subprocess

def advanced_mtm_fix():
    """高级修复MTM文件中的复杂赋值错误"""
    file_path = "indicators/mtm.py"
    
    print("🔧 MTM高级修复开始...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    changes_count = 0
    
    for i, line in enumerate(lines):
        original_line = line
        
        # 跳过已经正确的行
        if (line.strip().startswith('#') or 
            not line.strip() or 
            ' = ' in line or
            any(keyword in line for keyword in ['def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ', 'return', 'import', 'from '])):
            fixed_lines.append(line)
            continue
        
        # 高级模式修复
        
        # 1. ratio min(...) -> ratio = min(...)
        if re.search(r'^(\s+)(ratio)\s+(min\(.+\))$', line):
            line = re.sub(r'^(\s+)(ratio)\s+(min\(.+\))$', r'\1\2 = \3', line)
            
        # 2. score.iloc[i] expression -> score.iloc[i] = expression
        elif re.search(r'^(\s+)(score\.iloc\[\w+\])\s+(.+)$', line):
            line = re.sub(r'^(\s+)(score\.iloc\[\w+\])\s+(.+)$', r'\1\2 = \3', line)
            
        # 3. 一般变量赋值模式
        elif re.search(r'^(\s+)([a-zA-Z_]\w*)\s+(.+)$', line):
            line = re.sub(r'^(\s+)([a-zA-Z_]\w*)\s+(.+)$', r'\1\2 = \3', line)
        
        if line != original_line:
            changes_count += 1
            print(f"  行 {i+1}: 修复高级赋值错误")
        
        fixed_lines.append(line)
    
    # 写入修复后的内容
    fixed_content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"✅ MTM高级修复完成，共修复 {changes_count} 行")
    
    # 验证语法
    try:
        subprocess.run(['python3', '-m', 'py_compile', file_path], check=True, capture_output=True)
        print("✅ MTM语法检查通过")
        return True
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8')
        print(f"❌ MTM语法检查失败:\n{error_output}")
        return False

if __name__ == "__main__":
    advanced_mtm_fix()
