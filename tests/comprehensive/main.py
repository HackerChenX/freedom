#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
综合选股测试系统主脚本

运行全面测试、分析结果并修复问题
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def run_command(command, cwd=None):
    """
    运行命令
    
    Args:
        command: 命令
        cwd: 工作目录
        
    Returns:
        int: 返回码
    """
    print(f"执行命令: {command}")
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=cwd
    )
    
    # 实时输出
    while True:
        stdout_line = process.stdout.readline()
        stderr_line = process.stderr.readline()
        
        if stdout_line:
            print(stdout_line.strip())
        
        if stderr_line:
            print(stderr_line.strip(), file=sys.stderr)
        
        if not stdout_line and not stderr_line and process.poll() is not None:
            break
    
    return process.returncode


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="综合选股测试系统主脚本")
    parser.add_argument('--workspace', default="test_workspace", help="工作空间目录")
    parser.add_argument('--dashboard', action='store_true', help="启用监控仪表板")
    parser.add_argument('--skip-tests', action='store_true', help="跳过测试")
    parser.add_argument('--skip-fixes', action='store_true', help="跳过修复")
    args = parser.parse_args()
    
    # 创建工作空间目录
    workspace_dir = Path(args.workspace)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行测试
    if not args.skip_tests:
        print("\n=== 运行测试 ===")
        
        dashboard_arg = "--dashboard" if args.dashboard else ""
        test_command = f"python -m tests.comprehensive.run_tests --workspace {args.workspace} {dashboard_arg}"
        
        test_result = run_command(test_command)
        
        if test_result != 0:
            print(f"测试失败，返回码: {test_result}")
            if args.skip_fixes:
                return test_result
        else:
            print("测试成功")
    
    # 修复问题
    if not args.skip_fixes:
        print("\n=== 修复问题 ===")
        
        fix_command = f"python -m tests.comprehensive.fix_issues --workspace {args.workspace}"
        
        fix_result = run_command(fix_command)
        
        if fix_result != 0:
            print(f"修复失败，返回码: {fix_result}")
            return fix_result
        else:
            print("修复成功")
    
    # 记录结束时间
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration:.2f}秒")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())