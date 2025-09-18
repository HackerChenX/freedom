#!/usr/bin/env python3
"""
激进式快速修复脚本 - 修复所有明显的缺失等号错误
"""

import re

def aggressive_fix(file_path):
    """激进式修复所有缺失等号的错误"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_count = 0
    
    for i, line in enumerate(lines):
        original_line = line
        stripped = line.strip()
        
        # 跳过空行、注释、已有等号的行
        if not stripped or stripped.startswith('#') or '=' in line:
            continue
        
        # 跳过明显的控制结构
        if any(kw in line for kw in ['def ', 'class ', 'import ', 'from ', 'if ', 'elif ', 
                                    'else:', 'try:', 'except:', 'finally:', 'with ', 'for ', 
                                    'while ', 'return ', 'raise ', 'yield ', '@']):
            continue
        
        # 激进模式：任何以标识符开头，后跟空格和非控制字符的行
        match = re.match(r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+([^\s].*)$', line)
        if match:
            indent, var_name, rest = match.groups()
            
            # 排除一些特殊情况
            if (not rest.startswith(('and ', 'or ', 'not ', 'in ', 'is ', 'lambda')) and
                not line.strip().endswith(':') and
                not rest.strip().startswith('=')):
                
                new_line = f"{indent}{var_name} = {rest}"
                lines[i] = new_line
                print(f"第{i+1}行修复: {original_line.strip()}")
                print(f"         → {new_line.strip()}")
                fixed_count += 1
    
    if fixed_count > 0:
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"\n✅ 总共修复 {fixed_count} 处语法错误")
    else:
        print("ℹ️  没有发现需要修复的语法错误")
    
    return fixed_count

if __name__ == '__main__':
    import sys
    import subprocess
    
    if len(sys.argv) < 2:
        print("用法: python aggressive_fix.py <文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(f"🚀 开始激进式修复: {file_path}")
    print("=" * 60)
    
    fixed_count = aggressive_fix(file_path)
    
    print("=" * 60)
    print("🔍 语法检查中...")
    
    # 测试语法
    result = subprocess.run([sys.executable, '-m', 'py_compile', file_path], 
                           capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"🎉 恭喜！语法检查完全通过！")
        print(f"📊 本次修复: {fixed_count} 处错误")
    else:
        print(f"⚠️  还剩余一些语法错误:")
        print(result.stderr)
        print(f"📊 本次修复: {fixed_count} 处错误，还需继续处理其他错误")
