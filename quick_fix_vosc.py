#!/usr/bin/env python3
"""
快速修复VOSC文件中剩余的语法错误
"""

import re

def fix_vosc_file():
    """修复VOSC文件"""
    file_path = "indicators/vosc.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # 需要修复的模式列表
    fixes = [
        (r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+(pd\.Series\([^)]+\))\s*$', r'\1\2 = \3'),
        (r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+(self\._result\[[^]]+\])\s*$', r'\1\2 = \3'),
        (r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+(data\[[^]]+\])\s*$', r'\1\2 = \3'),
        (r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+(.*\.tail\([^)]+\))\s*$', r'\1\2 = \3'),
        (r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+(.*\.iloc\[[^]]+\])\s*$', r'\1\2 = \3'),
        (r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+(.*\s*[><=]+\s*.*)\s*$', r'\1\2 = \3'),
        (r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+(\[\])\s*$', r'\1\2 = \3'),
        (r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+(min\([^)]+\))\s*$', r'\1\2 = \3'),
        (r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+([A-Z][a-zA-Z0-9_]*\([^)]*\))\s*$', r'\1\2 = \3'),
        (r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-z_][a-zA-Z0-9_]*\([^)]*\))\s*$', r'\1\2 = \3'),
        (r'^(\s*)return\s+"([a-zA-Z_][a-zA-Z0-9_]*)"', r'\1return \2'),
        (r'^([A-Z_][A-Z0-9_]*)\s+([A-Z][a-zA-Z0-9_]*)$', r'\1 = \2'),
    ]
    
    modified = False
    
    for i, line in enumerate(lines):
        original_line = line
        
        # 跳过注释行、空行、关键字行
        stripped = line.strip()
        if (not stripped or stripped.startswith('#') or 
            any(keyword in line for keyword in [
                'class ', 'def ', 'import ', 'from ', 'if ', 'elif ', 'else:', 
                'try:', 'except', 'finally:', 'with ', 'for ', 'while ', 'return ',
                'raise ', 'assert ', 'yield ', 'del ', 'global ', 'nonlocal ',
                'pass', 'break', 'continue', '@'
            ]) or '=' in line):
            continue
        
        # 应用修复模式
        for pattern, replacement in fixes:
            new_line = re.sub(pattern, replacement, line)
            if new_line != line:
                lines[i] = new_line
                print(f"第{i+1}行修复:")
                print(f"  原始: {repr(line)}")
                print(f"  修复: {repr(new_line)}")
                modified = True
                break
    
    if modified:
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"✅ 文件已修复并保存")
    else:
        print("ℹ️  没有发现需要修复的内容")
    
    return modified

if __name__ == '__main__':
    fix_vosc_file()
