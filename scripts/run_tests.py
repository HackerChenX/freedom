#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单元测试运行脚本

运行项目中的所有单元测试，可以指定特定模块或测试类
"""

import os
import sys
import unittest
import argparse

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)


def run_tests(test_module=None, verbose=True):
    """
    运行单元测试
    
    Args:
        test_module: 要测试的特定模块，例如'tests.indicators.test_new_indicators'
        verbose: 是否显示详细信息
    """
    if test_module:
        # 运行指定模块的测试
        try:
            suite = unittest.defaultTestLoader.loadTestsFromName(test_module)
        except Exception as e:
            print(f"无法加载测试模块 {test_module}: {e}")
            return False
    else:
        # 运行tests目录下的所有测试
        tests_dir = os.path.join(root_dir, 'tests')
        suite = unittest.defaultTestLoader.discover(tests_dir)
    
    # 运行测试
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行单元测试')
    parser.add_argument('-m', '--module', type=str, help='要测试的特定模块')
    parser.add_argument('-q', '--quiet', action='store_true', help='不显示详细测试信息')
    args = parser.parse_args()
    
    print("开始运行单元测试...")
    success = run_tests(args.module, not args.quiet)
    
    if success:
        print("\n所有测试通过！")
        sys.exit(0)
    else:
        print("\n测试失败！")
        sys.exit(1)


if __name__ == '__main__':
    main() 