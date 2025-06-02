#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试文件替代修复工具

直接替换导入部分来解决问题
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# 需要修复的文件列表
TEST_FILES = [
    "test_indicators_and_backtest.py",
    "test_pattern_recognition.py",
    "test_multi_period_analysis.py"
]

# 导入部分替换
IMPORT_REPLACEMENT = """import sys
import os
import time
import json
import pandas as pd
import numpy as np
import unittest
from datetime import datetime, timedelta
from typing import List, Dict, Optional
# 单独导入Any类型
from typing import Any

"""

def main():
    """主函数"""
    # 测试目录
    review_dir = os.path.join(root_dir, "tests", "review")
    
    # 修复每个文件
    fixed_count = 0
    for file_name in TEST_FILES:
        file_path = os.path.join(review_dir, file_name)
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 定位导入部分
            import_start = content.find("import sys")
            if import_start == -1:
                print(f"文件 {file_path} 中找不到导入部分")
                continue
                
            # 找到导入部分的结束位置
            import_end = content.find("# 添加项目根目录到Python路径", import_start)
            if import_end == -1:
                print(f"文件 {file_path} 中找不到导入部分结束位置")
                continue
            
            # 替换导入部分
            new_content = content[:import_start] + IMPORT_REPLACEMENT + content[import_end:]
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"文件 {file_path} 已修复")
            fixed_count += 1
            
        except Exception as e:
            print(f"修复文件 {file_path} 时出错: {e}")
    
    print(f"修复完成，共修复了 {fixed_count} 个文件")

if __name__ == "__main__":
    main() 