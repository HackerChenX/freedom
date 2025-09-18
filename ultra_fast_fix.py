#!/usr/bin/env python3
"""
终极快速修复脚本 - 针对所有缺失等号的语法错误
"""

import re

def ultra_fast_fix(file_path):
    """终极快速修复"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 直接使用正则表达式批量替换所有模式
    original_content = content
    
    # 模式1: variable_name something (最常见的模式)
    # 但排除已有等号、关键字、注释等
    lines = content.split('\n')
    new_lines = []
    fixed_count = 0
    
    for i, line in enumerate(lines):
        original_line = line
        
        # 跳过不需要修复的行
        stripped = line.strip()
        if (not stripped or stripped.startswith('#') or 
            '=' in line or line.strip().endswith(':') or
            any(kw in line for kw in ['def ', 'class ', 'import ', 'from ', 'if ', 'elif ', 
                                     'else', 'try:', 'except', 'finally:', 'with ', 'for ', 
                                     'while ', 'return ', 'raise ', 'yield ', '@', 'and ', 'or '])):
            new_lines.append(line)
            continue
        
        # 通用模式: 以字母开头的变量名 + 空格 + 非等号开头的表达式
        pattern = r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+([^=\s].*)\s*$'
        match = re.match(pattern, line)
        
        if match:
            indent, var_name, expression = match.groups()
            # 额外检查：确保不是其他语法结构
            if (not any(word in expression for word in ['def ', 'class ', 'lambda']) and
                not expression.strip().startswith(('(', 'and', 'or')) and
                len(var_name) > 1):  # 避免单字符变量的误报
                
                new_line = f"{indent}{var_name} = {expression}"
                new_lines.append(new_line)
                print(f"第{i+1}行修复: {line.strip()} → {new_line.strip()}")
                fixed_count += 1
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    if fixed_count > 0:
        # 写回文件
        new_content = '\n'.join(new_lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ 共修复 {fixed_count} 处语法错误")
    else:
        print("ℹ️  没有发现需要修复的语法错误")
    
    return fixed_count

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("用法: python ultra_fast_fix.py <文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(f"🚀 开始终极快速修复: {file_path}")
    
    fixed_count = ultra_fast_fix(file_path)
    
    # 测试语法
    import subprocess
    result = subprocess.run([sys.executable, '-m', 'py_compile', file_path], 
                           capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"🎉 语法检查通过！文件修复成功！")
    else:
        print(f"⚠️  还有其他语法错误需要处理:")
        error_lines = result.stderr.strip().split('\n')
        for line in error_lines[-3:]:  # 显示最后几行错误信息
            print(f"   {line}")
