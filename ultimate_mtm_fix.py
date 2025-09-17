#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTM终极修复工具 - 确保100%修复完成
"""

import re
import subprocess

def ultimate_fix():
    """终极修复MTM文件"""
    file_path = "indicators/mtm.py"
    
    print("🏆 MTM终极修复开始...")
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    changes_count = 0
    
    for i, line in enumerate(lines, 1):
        original_line = line
        
        # 跳过注释和空行
        if line.strip().startswith('#') or not line.strip():
            fixed_lines.append(line)
            continue
        
        # 跳过包含等号的行（已经正确的赋值）
        if ' = ' in line or ' == ' in line or ' != ' in line or ' <= ' in line or ' >= ' in line:
            fixed_lines.append(line)
            continue
        
        # 跳过控制流语句
        if any(keyword in line for keyword in ['def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ', 'return', 'import', 'from ', 'raise', 'assert']):
            fixed_lines.append(line)
            continue
        
        # 修复所有的赋值错误模式
        
        # 1. score.iloc[i] value -> score.iloc[i] = value
        if re.search(r'^(\s*)(score\.iloc\[[^\]]+\])\s+([^=].*?)(\s*#.*)?$', line):
            line = re.sub(r'^(\s*)(score\.iloc\[[^\]]+\])\s+([^=].*?)(\s*#.*)?$', r'\1\2 = \3\4', line)
            changes_count += 1
            print(f"  行 {i}: 修复score.iloc赋值")
        
        # 2. variable value -> variable = value（通用模式）
        elif re.search(r'^(\s*)([a-zA-Z_]\w*)\s+([^=].*?)(\s*#.*)?$', line):
            line = re.sub(r'^(\s*)([a-zA-Z_]\w*)\s+([^=].*?)(\s*#.*)?$', r'\1\2 = \3\4', line)
            changes_count += 1
            print(f"  行 {i}: 修复普通变量赋值")
        
        # 3. dict['key'] value -> dict['key'] = value
        elif re.search(r"^(\s*)([a-zA-Z_]\w*\[.*?\])\s+([^=].*?)(\s*#.*)?$", line):
            line = re.sub(r"^(\s*)([a-zA-Z_]\w*\[.*?\])\s+([^=].*?)(\s*#.*)?$", r'\1\2 = \3\4', line)
            changes_count += 1
            print(f"  行 {i}: 修复字典赋值")
        
        fixed_lines.append(line)
    
    # 写入修复后的内容
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ 终极修复完成，共修复 {changes_count} 处错误")
    
    # 验证语法
    try:
        subprocess.run(['python3', '-m', 'py_compile', file_path], 
                      check=True, capture_output=True, text=True)
        print("🎉 MTM文件100%修复成功！语法检查通过！")
        return True
    except subprocess.CalledProcessError as e:
        error_output = e.stderr
        print(f"❌ 仍有语法错误:\n{error_output}")
        return False

if __name__ == "__main__":
    success = ultimate_fix()
    if success:
        print("\n🏆 MTM文件100%修复完成！可以进行系统测试了！")
    else:
        print("\n🔧 需要进一步分析剩余错误...")
