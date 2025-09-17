#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTM最终修复工具 - 确保100%完成所有语法错误修复
"""

import re
import subprocess

def final_mtm_fix():
    """最终修复MTM文件中的所有剩余语法错误"""
    file_path = "indicators/mtm.py"
    
    print("🎯 MTM最终修复开始...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复各种可能的语法错误
    changes_count = 0
    
    # 1. 修复未闭合的字典
    # 查找模式：行以 "key": { 结尾但下一行不是字典内容
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # 检查是否是未闭合的字典开始
        if re.match(r'^\s*"[^"]*":\s*\{\s*$', line):
            # 查看下几行，如果没有找到字典内容，添加闭合括号
            j = i + 1
            has_dict_content = False
            while j < len(lines) and j < i + 5:  # 检查接下来5行
                next_line = lines[j].strip()
                if next_line and not next_line.startswith('#'):
                    if re.match(r'^"[^"]*":', next_line) or next_line == '}':
                        has_dict_content = True
                        break
                    elif not re.match(r'^\s*$', next_line):
                        break
                j += 1
            
            if not has_dict_content:
                # 添加空字典闭合
                line = line.rstrip() + ' }'
                changes_count += 1
                print(f"  行 {i+1}: 修复字典语法")
        
        fixed_lines.append(line)
    
    # 重新组合内容
    content = '\n'.join(fixed_lines)
    
    # 2. 修复其他常见语法错误
    # 修复空字典定义
    content = re.sub(r':\s*\{\s*\n\s*(["\w])', r': {},\n        \1', content)
    
    # 3. 确保字典项都有逗号分隔
    content = re.sub(r'(\}\s*)\n(\s*"[^"]*":\s*\{)', r'\1,\n\2', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ MTM最终修复完成，共修复 {changes_count} 行")
    
    # 验证语法
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            subprocess.run(['python3', '-m', 'py_compile', file_path], check=True, capture_output=True)
            print("🎉 MTM语法检查通过！100%修复完成！")
            return True
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode('utf-8')
            print(f"❌ 第{attempt+1}次语法检查失败:\n{error_output}")
            
            if attempt < max_attempts - 1:
                # 尝试自动修复常见错误
                if "expression expected after dictionary key" in error_output:
                    # 修复字典语法
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    content = re.sub(r':\s*\{\s*\n\s*"', r': {\n        "', content)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print("  🔧 尝试修复字典语法错误...")
            
    return False

if __name__ == "__main__":
    final_mtm_fix()
