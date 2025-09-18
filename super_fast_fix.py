#!/usr/bin/env python3
"""
超级快速修复缺失等号的脚本
专门针对最常见的语法错误模式
"""

import re
import sys

def fast_fix_file(file_path):
    """快速修复单个文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_count = 0
    
    for i, line in enumerate(lines):
        original_line = line
        
        # 跳过明显不需要修复的行
        stripped = line.strip()
        if (not stripped or stripped.startswith('#') or 
            '=' in line or ':' in stripped or
            any(kw in line for kw in ['def ', 'class ', 'import ', 'from ', 'if ', 'elif ', 'else', 
                                     'try:', 'except', 'finally:', 'with ', 'for ', 'while ', 
                                     'return ', 'raise ', 'yield ', '@'])):
            continue
        
        # 模式1: variable_name pd.Series(...) 或类似函数调用
        if re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*$', line):
            lines[i] = re.sub(r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\))\s*$', 
                             r'\1\2 = \3', line)
            if lines[i] != original_line:
                print(f"第{i+1}行修复: {original_line.strip()} → {lines[i].strip()}")
                fixed_count += 1
                continue
        
        # 模式2: variable_name self._result['key'] 或字典/列表访问
        if re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_\.]*\[[^\]]+\]\s*$', line):
            lines[i] = re.sub(r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_\.]*\[[^\]]+\])\s*$', 
                             r'\1\2 = \3', line)
            if lines[i] != original_line:
                print(f"第{i+1}行修复: {original_line.strip()} → {lines[i].strip()}")
                fixed_count += 1
                continue
        
        # 模式3: variable_name [] 或简单字面量
        if re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s+(\[\]|\{\}|[0-9]+|True|False|None)\s*$', line):
            lines[i] = re.sub(r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+(.*)\s*$', 
                             r'\1\2 = \3', line)
            if lines[i] != original_line:
                print(f"第{i+1}行修复: {original_line.strip()} → {lines[i].strip()}")
                fixed_count += 1
                continue
        
        # 模式4: variable_name expression (比较、算术等)
        if re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s+.*(>|<|>=|<=|\+|\-|\*|/|%|&|\|)\s*.*$', line):
            lines[i] = re.sub(r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+(.*)\s*$', 
                             r'\1\2 = \3', line)
            if lines[i] != original_line:
                print(f"第{i+1}行修复: {original_line.strip()} → {lines[i].strip()}")
                fixed_count += 1
                continue
        
        # 模式5: return "variable_name" 应该是 return variable_name
        if re.match(r'^\s*return\s+"[a-zA-Z_][a-zA-Z0-9_]*"\s*$', line):
            lines[i] = re.sub(r'^(\s*return\s+)"([a-zA-Z_][a-zA-Z0-9_]*)"(\s*)$', 
                             r'\1\2\3', line)
            if lines[i] != original_line:
                print(f"第{i+1}行修复: {original_line.strip()} → {lines[i].strip()}")
                fixed_count += 1
                continue
        
        # 模式6: CLASS_NAME ClassName (类别名定义)
        if re.match(r'^[A-Z_][A-Z0-9_]*\s+[A-Z][a-zA-Z0-9_]*\s*$', line.strip()):
            lines[i] = re.sub(r'^([A-Z_][A-Z0-9_]*)\s+([A-Z][a-zA-Z0-9_]*)\s*$', 
                             r'\1 = \2', line)
            if lines[i] != original_line:
                print(f"第{i+1}行修复: {original_line.strip()} → {lines[i].strip()}")
                fixed_count += 1
                continue
    
    if fixed_count > 0:
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"✅ 文件 {file_path} 修复完成，共修复 {fixed_count} 处")
    else:
        print(f"ℹ️  文件 {file_path} 没有发现需要修复的内容")
    
    return fixed_count

def main():
    if len(sys.argv) < 2:
        print("用法: python super_fast_fix.py <文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print(f"🚀 开始快速修复文件: {file_path}")
    
    try:
        fixed_count = fast_fix_file(file_path)
        
        # 测试语法
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'py_compile', file_path], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 语法检查通过！修复了 {fixed_count} 处错误")
        else:
            print(f"❌ 语法检查失败，还有其他错误需要修复:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ 修复过程中出错: {e}")

if __name__ == '__main__':
    main()
