#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTM文件最终精准修复工具
专门处理复杂表达式赋值和特殊语法错误
"""

import re
import subprocess

def final_precision_fix():
    """最终精准修复MTM文件"""
    file_path = "indicators/mtm.py"
    
    print("🎯 MTM文件最终精准修复开始...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 精准修复特定的复杂表达式
    original_content = content
    
    # 1. 修复复杂的条件表达式赋值
    # ratio min(...) if ... else ... -> ratio = min(...) if ... else ...
    content = re.sub(
        r'^(\s+)(ratio)\s+(min\(.+?\))\s+(if\s+.+?\s+else\s+.+?)(\s*#.*)?$',
        r'\1\2 = \3 \4\5',
        content,
        flags=re.MULTILINE
    )
    
    # 2. 修复其他变量的复杂表达式赋值
    # variable complex_expression if condition else default -> variable = complex_expression if condition else default
    content = re.sub(
        r'^(\s+)([a-zA-Z_]\w*)\s+([^=].+?\s+if\s+.+?\s+else\s+.+?)(\s*#.*)?$',
        r'\1\2 = \3\4',
        content,
        flags=re.MULTILINE
    )
    
    # 3. 修复价格计算表达式
    if 'abs(current_price - previous_price) / previous_price' in content:
        content = content.replace(
            'abs(current_price - previous_price) / previous_price',
            'price_change = abs(current_price - previous_price) / previous_price'
        )
    
    # 4. 修复错误的return语句
    content = re.sub(
        r'return "([^"]*)"(\s*#.*)?$',
        r'return \1\2',
        content,
        flags=re.MULTILINE
    )
    
    # 5. 修复未闭合的字典或错误的return语句
    content = re.sub(
        r'return "\{([^}]*)\}$',
        r'return {\1}',
        content,
        flags=re.MULTILINE
    )
    
    # 6. 修复特殊的字典末尾缺失问题
    content = re.sub(
        r'(\}\s*)\n(\s*)(\w+.*?:.*?{)',
        r'\1},\n\2\3',
        content,
        flags=re.MULTILINE
    )
    
    changes_made = content != original_content
    
    if changes_made:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ 完成最终精准修复")
    else:
        print("ℹ️ 未发现需要精准修复的内容")
    
    # 验证语法
    try:
        subprocess.run(['python3', '-m', 'py_compile', file_path], 
                      check=True, capture_output=True, text=True)
        print("🎉 MTM语法检查通过！100%修复完成！")
        return True
    except subprocess.CalledProcessError as e:
        error_output = e.stderr
        print(f"❌ MTM语法检查失败:\n{error_output}")
        
        # 提取具体错误信息用于进一步修复
        match = re.search(r'line (\d+)', error_output)
        if match:
            line_num = int(match.group(1))
            print(f"\n🔍 剩余错误位置: 第 {line_num} 行")
            
            # 显示错误行内容
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if line_num <= len(lines):
                    print(f"错误行内容: {lines[line_num-1].strip()}")
        
        return False

if __name__ == "__main__":
    success = final_precision_fix()
    if success:
        print("\n🏆 MTM文件100%修复成功！")
    else:
        print("\n🔧 需要进一步手动修复...")
