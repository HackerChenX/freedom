#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复复合赋值运算符错误的工具
专门处理 "= +=" 这类错误模式
"""

import re
import subprocess

def fix_compound_assignment():
    """修复复合赋值运算符错误"""
    file_path = "indicators/mtm.py"
    
    print("🔧 修复复合赋值运算符错误...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复各种复合赋值错误
    original_content = content
    
    # 修复 "= +=" -> "+="
    content = re.sub(r' = \+= ', ' += ', content)
    content = re.sub(r' = -= ', ' -= ', content)
    content = re.sub(r' = \*= ', ' *= ', content)
    content = re.sub(r' = /= ', ' /= ', content)
    
    changes = original_content != content
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    if changes:
        print("✅ 修复了复合赋值运算符错误")
    else:
        print("ℹ️ 未发现复合赋值运算符错误")
    
    # 验证语法
    try:
        subprocess.run(['python3', '-m', 'py_compile', file_path], check=True, capture_output=True)
        print("✅ MTM语法检查通过")
        return True
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8')
        print(f"❌ MTM语法检查失败:\n{error_output}")
        return False

if __name__ == "__main__":
    fix_compound_assignment()
