#!/usr/bin/env python3
"""
专门修复缺少逗号的语法错误脚本
"""

import re

def fix_comma_errors(file_path):
    """修复函数参数列表中缺少逗号的错误"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_count = 0
    
    for i, line in enumerate(lines):
        original_line = line
        
        # 匹配模式：parameter=value  # comment,  # comment
        # 这种情况下，第一个逗号后面应该没有第二个逗号
        pattern = r'^(\s*\w+\s*=\s*[^,]+)\s*#[^,]*,\s*#.*$'
        match = re.match(pattern, line)
        
        if match:
            # 提取参数部分，添加逗号
            param_part = match.group(1)
            # 移除重复的注释和逗号，只保留一个简洁的注释
            new_line = f"{param_part},  # TODO: 将魔法数字提取到配置中"
            lines[i] = new_line
            print(f"第{i+1}行修复:")
            print(f"  原始: {original_line.strip()}")
            print(f"  修复: {new_line.strip()}")
            print()
            fixed_count += 1
    
    if fixed_count > 0:
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"✅ 总共修复 {fixed_count} 处逗号错误")
    else:
        print("ℹ️  没有发现需要修复的逗号错误")
    
    return fixed_count

if __name__ == '__main__':
    import sys
    import subprocess
    
    if len(sys.argv) < 2:
        print("用法: python fix_comma_errors.py <文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(f"🔧 开始修复逗号错误: {file_path}")
    print("=" * 50)
    
    fixed_count = fix_comma_errors(file_path)
    
    print("=" * 50)
    print("🔍 语法检查中...")
    
    # 测试语法
    result = subprocess.run([sys.executable, '-m', 'py_compile', file_path], 
                           capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"🎉 语法检查完全通过！")
        print(f"📊 本次修复: {fixed_count} 处逗号错误")
    else:
        print(f"⚠️  还有其他语法错误:")
        print(result.stderr)
        print(f"📊 本次修复: {fixed_count} 处逗号错误")
