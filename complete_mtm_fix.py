#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTM文件100%完整修复工具
基于质量优先策略，确保彻底修复所有剩余的语法错误
"""

import re
import subprocess

def complete_mtm_fix():
    """100%完成MTM文件修复"""
    file_path = "indicators/mtm.py"
    
    print("🚀 MTM文件100%完整修复开始...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 记录修改
    lines = content.split('\n')
    fixed_lines = []
    changes_count = 0
    
    for i, line in enumerate(lines):
        original_line = line
        
        # 跳过已经正确的行
        if (line.strip().startswith('#') or 
            not line.strip() or 
            ' = ' in line or
            ' == ' in line or
            ' != ' in line or
            ' <= ' in line or
            ' >= ' in line or
            any(keyword in line for keyword in ['def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ', 'return', 'import', 'from ', 'raise', 'assert', 'yield', 'lambda', 'global', 'nonlocal', '@'])):
            fixed_lines.append(line)
            continue
        
        # 系统性修复各种赋值错误模式
        
        # 1. signals['xxx'].iloc[i] True -> signals['xxx'].iloc[i] = True
        if re.search(r"signals\['\w+'\]\.iloc\[\w+\]\s+True", line):
            line = re.sub(r"(signals\['\w+'\]\.iloc\[\w+\])\s+(True)", r'\1 = \2', line)
        
        # 2. signals['xxx'].iloc[i] False -> signals['xxx'].iloc[i] = False  
        elif re.search(r"signals\['\w+'\]\.iloc\[\w+\]\s+False", line):
            line = re.sub(r"(signals\['\w+'\]\.iloc\[\w+\])\s+(False)", r'\1 = \2', line)
            
        # 3. signals['signal_strength'].iloc[i] float(...) -> signals['signal_strength'].iloc[i] = float(...)
        elif re.search(r"signals\['signal_strength'\]\.iloc\[\w+\]\s+float\(", line):
            line = re.sub(r"(signals\['signal_strength'\]\.iloc\[\w+\])\s+(float\(.*\))", r'\1 = \2', line)
            
        # 4. 基本变量赋值: var value -> var = value
        elif re.search(r'^(\s+)([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*(?:\.\w+)*(?:\([^)]*\))?|\d+(?:\.\d+)?|".*?"|\'.*?\'|True|False|None)$', line):
            line = re.sub(r'^(\s+)([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*(?:\.\w+)*(?:\([^)]*\))?|\d+(?:\.\d+)?|".*?"|\'.*?\'|True|False|None)$', r'\1\2 = \3', line)
            
        # 5. 复杂表达式赋值: var complex_expression -> var = complex_expression
        elif re.search(r'^(\s+)([a-zA-Z_]\w*)\s+(.+)$', line) and not any(keyword in line for keyword in ['if ', 'elif ', 'for ', 'while ', 'def ', 'class ', 'try:', 'except', 'finally:', 'with ']):
            line = re.sub(r'^(\s+)([a-zA-Z_]\w*)\s+(.+)$', r'\1\2 = \3', line)
        
        if line != original_line:
            changes_count += 1
            print(f"  行 {i+1}: 修复赋值错误")
        
        fixed_lines.append(line)
    
    # 写入修复后的内容
    fixed_content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"✅ MTM完整修复完成，共修复 {changes_count} 行")
    
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
    complete_mtm_fix()
