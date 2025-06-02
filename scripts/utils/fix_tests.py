#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试文件修复工具

修复测试文件中的导入和其他常见问题
"""

import os
import re
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger

# 获取日志记录器
logger = get_logger("test_fixer")

def fix_any_import(file_path):
    """
    修复文件中的Any导入问题
    
    Args:
        file_path: 要修复的文件路径
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否有from typing import ... 行
    typing_import_pattern = r'from\s+typing\s+import\s+([^\\]+?)(?:\n|$)'
    typing_imports = re.search(typing_import_pattern, content)
    
    if typing_imports:
        import_items = typing_imports.group(1)
        # 如果已经导入了Any，就不做任何修改
        if 'Any' in import_items:
            logger.info(f"文件 {file_path} 已经导入了Any，无需修改")
            return False
        
        # 添加Any到导入列表
        new_import_items = import_items.strip()
        if new_import_items.endswith(','):
            new_import_items += ' Any'
        else:
            new_import_items += ', Any'
        
        # 替换导入行
        new_content = content.replace(
            f"from typing import {import_items}",
            f"from typing import {new_import_items}"
        )
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.info(f"文件 {file_path} 已修复，添加了Any导入")
        return True
    else:
        # 如果没有找到typing导入行，添加一个新的导入行
        new_content = content.replace(
            "import unittest", 
            "import unittest\nfrom typing import Any"
        )
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.info(f"文件 {file_path} 已修复，添加了typing.Any导入")
        return True

def main():
    """主函数"""
    # 测试目录
    review_dir = os.path.join(root_dir, "tests", "review")
    
    # 要修复的文件列表
    test_files = [
        "run_all_tests.py",
        "test_indicators_and_backtest.py",
        "test_pattern_recognition.py",
        "test_multi_period_analysis.py"
    ]
    
    # 修复每个文件
    fixed_count = 0
    for file_name in test_files:
        file_path = os.path.join(review_dir, file_name)
        if os.path.exists(file_path):
            if fix_any_import(file_path):
                fixed_count += 1
        else:
            logger.warning(f"文件不存在: {file_path}")
    
    logger.info(f"修复完成，共修复了 {fixed_count} 个文件")

if __name__ == "__main__":
    main() 