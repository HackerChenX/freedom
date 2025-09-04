#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复装饰器参数兼容性问题

将所有使用 threshold_seconds 的 performance_monitor 装饰器
修改为使用 threshold 参数。

Author: AI Assistant
Date: 2025-07-19
"""

import os
import re
import sys
from pathlib import Path

def fix_decorator_parameters():
    """修复装饰器参数"""
    
    # 需要修复的文件列表
    files_to_fix = [
        './analysis/engines/date_manager.py',
        './analysis/market/a_stock_market_analysis.py',
        './bin/zxm_analysis.py',
        './bin/run_advanced_backtest.py',
        './bin/indicator_scoring.py',
        './bin/main.py',
        './bin/init_db.py',
        './tests/comprehensive/end_to_end_integration_tester.py',
        './tests/comprehensive/unit_test_framework.py',
        './tests/comprehensive/indicator_accuracy_validator.py',
        './tests/comprehensive/exception_scenario_tester.py',
        './tests/comprehensive/test_infrastructure.py',
        './tests/comprehensive/pattern_recognition_tester.py',
        './tests/comprehensive/parameter_combination_tester.py',
        './tests/comprehensive/strategy_coverage_tester.py',
        './tests/comprehensive/indicator_tester.py',
        './tests/comprehensive/boundary_condition_tester.py',
        './utils/trading_halt_processor.py',
        './scripts/production_strategy_validator.py',
        './db/advanced_data_quality_manager.py'
    ]
    
    fixed_count = 0
    error_count = 0
    
    for file_path in files_to_fix:
        try:
            if os.path.exists(file_path):
                print(f"修复文件: {file_path}")
                
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 备份原文件
                backup_path = file_path + '.backup'
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # 修复装饰器参数
                # 匹配 @performance_monitor(threshold_seconds=数值)
                pattern = r'@performance_monitor\(threshold_seconds=([0-9.]+)\)'
                replacement = r'@performance_monitor(threshold=\1)'
                
                new_content = re.sub(pattern, replacement, content)
                
                # 检查是否有修改
                if new_content != content:
                    # 写入修复后的内容
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    print(f"  ✅ 已修复装饰器参数")
                    fixed_count += 1
                else:
                    print(f"  ℹ️ 无需修复")
                    # 删除不必要的备份
                    os.remove(backup_path)
            else:
                print(f"  ⚠️ 文件不存在: {file_path}")
                
        except Exception as e:
            print(f"  ❌ 修复失败: {e}")
            error_count += 1
    
    print(f"\n修复完成:")
    print(f"  成功修复: {fixed_count} 个文件")
    print(f"  修复失败: {error_count} 个文件")
    
    return fixed_count, error_count

def verify_fixes():
    """验证修复结果"""
    print("\n验证修复结果...")
    
    # 检查是否还有 threshold_seconds 参数
    result = os.system("grep -r 'threshold_seconds' . --include='*.py' | grep -v '.backup' | grep -v 'fix_decorator_parameters.py'")
    
    if result == 0:
        print("⚠️ 仍有文件使用 threshold_seconds 参数")
        return False
    else:
        print("✅ 所有 threshold_seconds 参数已修复")
        return True

def main():
    """主函数"""
    print("🔧 开始修复装饰器参数兼容性问题")
    print("=" * 60)
    
    # 修复装饰器参数
    fixed_count, error_count = fix_decorator_parameters()
    
    # 验证修复结果
    success = verify_fixes()
    
    print("=" * 60)
    if success and error_count == 0:
        print("🎉 装饰器参数修复完成！")
        return True
    else:
        print("❌ 修复过程中遇到问题，请检查错误信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
