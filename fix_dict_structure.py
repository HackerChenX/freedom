#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复MTM文件中的字典结构问题
专门处理缩进和字典语法错误
"""

import re
import subprocess

def fix_dict_structure():
    """修复字典结构问题"""
    file_path = "indicators/mtm.py"
    
    print("🔧 修复字典结构问题...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    changes_count = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 检查是否是错误的字典模式: "key": {}, 然后下一行是 "id": "key",
        if (re.match(r'^(\s*)"([^"]+)":\s*\{\}\s*,?\s*$', line) and 
            i + 1 < len(lines) and 
            re.match(r'^\s*"id":\s*"[^"]*",?\s*$', lines[i + 1])):
            
            # 提取缩进和键名
            match = re.match(r'^(\s*)"([^"]+)":\s*\{\}\s*,?\s*$', line)
            indent = match.group(1)
            key_name = match.group(2)
            
            # 开始收集字典内容
            dict_content = []
            j = i + 1
            
            # 收集所有属于这个字典的行
            while j < len(lines):
                next_line = lines[j]
                
                # 如果是下一个字典项或者缩进减少，停止收集
                if (re.match(r'^\s*"[^"]*":\s*\{', next_line) or 
                    re.match(r'^\s*\}', next_line) or
                    (next_line.strip() and len(next_line) - len(next_line.lstrip()) <= len(indent))):
                    break
                
                # 如果是字典属性行，收集它
                if re.match(r'^\s*"[^"]*":\s*.+,?\s*$', next_line):
                    dict_content.append(next_line.strip().rstrip(','))
                
                j += 1
            
            # 重建字典
            if dict_content:
                fixed_lines.append(f'{indent}"{key_name}": {{')
                for k, dict_line in enumerate(dict_content):
                    comma = ',' if k < len(dict_content) - 1 else ''
                    fixed_lines.append(f'{indent}    {dict_line}{comma}')
                fixed_lines.append(f'{indent}}},')
                changes_count += 1
                print(f"  修复字典: {key_name}")
            else:
                fixed_lines.append(line)
            
            i = j
        else:
            fixed_lines.append(line)
            i += 1
    
    # 写入修复后的内容
    fixed_content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"✅ 字典结构修复完成，共修复 {changes_count} 个字典")
    
    # 验证语法
    try:
        subprocess.run(['python3', '-c', 'import py_compile; py_compile.compile("indicators/mtm.py", doraise=True)'], 
                      check=True, capture_output=True)
        print("🎉 MTM语法检查通过！")
        return True
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8')
        print(f"❌ MTM语法检查失败:\n{error_output}")
        return False

if __name__ == "__main__":
    fix_dict_structure()
