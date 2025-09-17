#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复双等号赋值错误的工具
处理过度替换导致的 = = 问题
"""

import re

def fix_double_equals():
    """修复mtm.py中的双等号问题"""
    file_path = "indicators/mtm.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复双等号问题
    # = = -> =
    content = re.sub(r' = = ', ' = ', content)
    content = re.sub(r'= = ', '= ', content)
    content = re.sub(r' = =', ' =', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 双等号问题修复完成")

if __name__ == "__main__":
    fix_double_equals()
