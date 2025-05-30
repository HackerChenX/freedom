#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试运行脚本

运行指定的测试文件或测试套件
"""

import os
import sys
import unittest

# 将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def run_test(test_path):
    """
    运行指定的测试文件或测试套件
    
    Args:
        test_path: 测试文件路径或测试模块名称
    """
    # 如果是文件路径，将其转换为模块名称
    if test_path.endswith('.py'):
        # 移除.py后缀
        test_path = test_path[:-3]
        # 将路径分隔符替换为模块分隔符
        test_path = test_path.replace('/', '.')
        test_path = test_path.replace('\\', '.')
    
    # 加载测试套件
    suite = unittest.defaultTestLoader.loadTestsFromName(test_path)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python run_tests.py <test_path>")
        sys.exit(1)
    
    test_path = sys.argv[1]
    success = run_test(test_path)
    
    # 根据测试结果设置退出码
    sys.exit(0 if success else 1) 