#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试文件直接修复工具

使用直接字符串替换的方式修复测试文件
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger

# 获取日志记录器
logger = get_logger("test_fixer")

def fix_file(file_path):
    """
    修复文件
    
    Args:
        file_path: 要修复的文件路径
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换 Any 导入
        old_import = "from typing import List, Dict, Optional, Any"
        new_import = "from typing import List, Dict, Optional\nfrom typing import Any"
        
        new_content = content.replace(old_import, new_import)
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.info(f"文件 {file_path} 已修复")
        return True
    except Exception as e:
        logger.error(f"修复文件 {file_path} 时出错: {e}")
        return False

def main():
    """主函数"""
    # 测试目录
    review_dir = os.path.join(root_dir, "tests", "review")
    
    # 要修复的文件列表
    test_files = [
        "test_indicators_and_backtest.py",
        "test_pattern_recognition.py",
        "test_multi_period_analysis.py"
    ]
    
    # 修复每个文件
    fixed_count = 0
    for file_name in test_files:
        file_path = os.path.join(review_dir, file_name)
        if os.path.exists(file_path):
            if fix_file(file_path):
                fixed_count += 1
        else:
            logger.warning(f"文件不存在: {file_path}")
    
    logger.info(f"修复完成，共修复了 {fixed_count} 个文件")

if __name__ == "__main__":
    main() 