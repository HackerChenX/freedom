#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
专门修复字典赋值错误的工具
处理 dict['key'] value 模式
"""

import re
import subprocess

def fix_dict_assignments():
    """修复字典赋值错误"""
    file_path = "indicators/pvt.py"
    
    print("🔧 修复字典赋值错误...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    changes_count = 0
    
    for i, line in enumerate(lines):
        original_line = line
        
        # 检查字典赋值模式: dict['key'] value -> dict['key'] = value
        if re.search(r"^(\s+)([a-zA-Z_]\w*(?:\[[^\]]+\])*)\s+([^=].+)$", line) and not any(keyword in line for keyword in ['if ', 'elif ', 'for ', 'while ', 'def ', 'class ', 'try:', 'except', 'finally:', 'with ', 'return', 'import', 'from ']):
            line = re.sub(r"^(\s+)([a-zA-Z_]\w*(?:\[[^\]]+\])*)\s+([^=].+)$", r'\1\2 = \3', line)
            changes_count += 1
            print(f"  行 {i+1}: 修复字典赋值错误")
        
        fixed_lines.append(line)
    
    # 写入修复后的内容
    fixed_content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"✅ 字典赋值修复完成，共修复 {changes_count} 行")
    
    # 验证语法
    try:
        subprocess.run(['python3', '-m', 'py_compile', file_path], check=True, capture_output=True)
        print("🎉 PVT语法检查通过！")
        return True
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8')
        print(f"❌ PVT语法检查失败:\n{error_output}")
        return False

if __name__ == "__main__":
    fix_dict_assignments()
