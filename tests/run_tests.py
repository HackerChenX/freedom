#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
测试运行器

用于执行所有测试并生成测试报告
"""

import os
import sys
import unittest
import argparse
import time
from datetime import datetime
import coverage
import HTMLTestRunner

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试运行器")
    
    parser.add_argument("--type", "-t", 
                        choices=["unit", "integration", "performance", "all"], 
                        default="all",
                        help="要运行的测试类型")
    
    parser.add_argument("--pattern", "-p", 
                        default="test_*.py",
                        help="测试文件匹配模式")
    
    parser.add_argument("--report", "-r", 
                        action="store_true",
                        help="生成HTML测试报告")
    
    parser.add_argument("--coverage", "-c", 
                        action="store_true",
                        help="生成覆盖率报告")
    
    parser.add_argument("--output-dir", "-o", 
                        default=os.path.join(root_dir, "tests", "reports"),
                        help="报告输出目录")
    
    return parser.parse_args()

def run_tests(test_type, pattern, output_dir, generate_report, measure_coverage):
    """
    运行指定类型的测试
    
    Args:
        test_type: 测试类型 (unit, integration, performance, all)
        pattern: 测试文件匹配模式
        output_dir: 报告输出目录
        generate_report: 是否生成HTML报告
        measure_coverage: 是否生成覆盖率报告
    
    Returns:
        测试结果
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定测试目录
    if test_type == "all":
        test_dirs = ["unit", "integration", "performance"]
    else:
        test_dirs = [test_type]
    
    # 收集所有测试
    test_suite = unittest.TestSuite()
    for test_dir in test_dirs:
        dir_path = os.path.join(os.path.dirname(__file__), test_dir)
        if os.path.exists(dir_path):
            print(f"加载测试: {dir_path}")
            discover = unittest.defaultTestLoader.discover(
                dir_path, pattern=pattern
            )
            test_suite.addTests(discover)
    
    # 设置覆盖率
    cov = None
    if measure_coverage:
        cov = coverage.Coverage(
            source=["strategy", "indicators", "db", "utils"],
            omit=["*/test*", "*/migrations/*", "*/venv/*"]
        )
        cov.start()
    
    # 运行测试
    if generate_report:
        # 准备HTML报告
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        report_file = os.path.join(
            output_dir, f"test_report_{test_type}_{timestamp}.html"
        )
        with open(report_file, "wb") as f:
            runner = HTMLTestRunner.HTMLTestRunner(
                stream=f,
                title=f"可配置选股系统 {test_type.capitalize()} 测试报告",
                description=f"测试执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            result = runner.run(test_suite)
        print(f"HTML测试报告已生成: {report_file}")
    else:
        # 使用普通测试运行器
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
    
    # 生成覆盖率报告
    if measure_coverage:
        cov.stop()
        cov.save()
        
        # 生成覆盖率报告
        cov_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        coverage_html_dir = os.path.join(
            output_dir, f"coverage_{test_type}_{cov_timestamp}"
        )
        cov.html_report(directory=coverage_html_dir)
        
        # 输出覆盖率摘要
        print("\n覆盖率报告:")
        cov.report()
        print(f"HTML覆盖率报告已生成: {coverage_html_dir}")
    
    return result

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行测试
    print(f"开始运行{args.type}测试...")
    result = run_tests(
        test_type=args.type,
        pattern=args.pattern,
        output_dir=args.output_dir,
        generate_report=args.report,
        measure_coverage=args.coverage
    )
    
    # 计算运行时间
    elapsed_time = time.time() - start_time
    
    # 输出摘要
    print("\n测试摘要:")
    print(f"运行时间: {elapsed_time:.2f}秒")
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    # 设置退出代码
    if result.wasSuccessful():
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main()) 