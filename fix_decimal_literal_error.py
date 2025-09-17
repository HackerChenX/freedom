#!/usr/bin/env python3
"""
修复decimal literal错误
"""

import os
import re

def fix_decimal_literal_error():
    """修复decimal literal错误"""
    file_path = 'db/services/integrated/intelligent_query_optimizer.py'
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换可能导致问题的数字格式
        content = content.replace('18个方法', '十八个方法')
        content = content.replace('6个方法', '六个方法')
        content = content.replace('1.', '第一.')
        content = content.replace('2.', '第二.')
        content = content.replace('3.', '第三.')
        content = content.replace('4.', '第四.')
        content = content.replace('5.', '第五.')
        
        # 修复其他可能的数字问题
        content = re.sub(r'(\d+)个方法', r'\1 个方法', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 修复decimal literal错误完成")
        
    except Exception as e:
        print(f"❌ 修复decimal literal错误失败: {e}")

if __name__ == "__main__":
    fix_decimal_literal_error()
