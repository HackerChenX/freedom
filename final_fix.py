#!/usr/bin/env python3
"""
最终修复脚本 - 专门处理所有类型的缺失等号错误
"""

import re

def final_fix(file_path):
    """最终修复方案"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用更简单直接的方法：逐行检查和修复
    lines = content.split('\n')
    fixed_count = 0
    
    for i, line in enumerate(lines):
        original_line = line
        stripped = line.strip()
        
        # 跳过不需要处理的行
        if (not stripped or 
            stripped.startswith('#') or 
            '=' in line or
            stripped.startswith(('def ', 'class ', 'import ', 'from ', 'if ', 'elif ', 
                               'else:', 'try:', 'except:', 'finally:', 'with ', 'for ', 
                               'while ', 'return ', 'raise ', 'yield ', '@')) or
            stripped.endswith(':')):
            continue
        
        # 检查是否是"标识符 + 空格 + 其他内容"的模式
        # 使用更宽松的正则表达式
        if re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_\[\]\'\"]*\s+[^\s]', line):
            # 找到第一个标识符和后面的内容
            match = re.match(r'^(\s*)([a-zA-Z_][a-zA-Z0-9_\[\]\'\"]*)\s+(.*)\s*$', line)
            if match:
                indent, identifier, rest = match.groups()
                
                # 排除一些特殊语法
                if not any(word in rest for word in ['and ', 'or ', 'not ', 'in ', 'is ', 'lambda']):
                    new_line = f"{indent}{identifier} = {rest}"
                    lines[i] = new_line
                    print(f"第{i+1}行: {original_line.strip()}")
                    print(f"   ↓")
                    print(f"   {new_line.strip()}")
                    print()
                    fixed_count += 1
    
    if fixed_count > 0:
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"✅ 总共修复了 {fixed_count} 处语法错误")
    else:
        print("ℹ️  没有发现需要修复的语法错误")
    
    return fixed_count

def test_syntax(file_path):
    """测试语法"""
    import subprocess
    import sys
    
    result = subprocess.run([sys.executable, '-m', 'py_compile', file_path], 
                           capture_output=True, text=True)
    return result.returncode == 0, result.stderr

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python final_fix.py <文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(f"🎯 最终修复开始: {file_path}")
    print("=" * 50)
    
    # 执行修复
    fixed_count = final_fix(file_path)
    
    print("=" * 50)
    print("🔍 执行语法检查...")
    
    # 测试语法
    syntax_ok, error_msg = test_syntax(file_path)
    
    if syntax_ok:
        print(f"🎉 完美！语法检查完全通过！")
        print(f"📊 修复统计: {fixed_count} 处语法错误已全部修复")
    else:
        print(f"⚠️  语法检查发现剩余错误:")
        print(error_msg)
        print(f"📊 修复统计: {fixed_count} 处错误已修复，但还需处理其他类型的错误")
