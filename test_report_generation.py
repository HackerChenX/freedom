#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试报告生成是否包含数据污染
"""

import pandas as pd
import numpy as np
from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer

def test_report_generation():
    """测试报告生成是否包含非形态数据"""
    print("=== 测试报告生成数据污染修复 ===\n")
    
    # 创建批量分析器
    analyzer = BuyPointBatchAnalyzer()
    
    # 模拟一些买点数据，只包含形态结果
    mock_buypoints = [
        {
            'code': '000001.SZ',
            'name': '平安银行',
            'date': '2024-01-15',
            'close': 10.5,
            'indicator_results': {
                'chaikin': [
                    {'pattern_id': 'CHAIKIN_CROSS_UP_ZERO', 'strength': 75, 'duration': 3},
                    {'pattern_id': 'CHAIKIN_RISING', 'strength': 60, 'duration': 2}
                ],
                'macd': [
                    {'pattern_id': 'MACD_GOLDEN_CROSS', 'strength': 80, 'duration': 5},
                    {'pattern_id': 'MACD_BULLISH_DIVERGENCE', 'strength': 70, 'duration': 4}
                ],
                'cci': [
                    {'pattern_id': 'CCI_CROSS_UP_OVERSOLD', 'strength': 65, 'duration': 3}
                ]
            }
        },
        {
            'code': '000002.SZ',
            'name': '万科A',
            'date': '2024-01-20',
            'close': 8.2,
            'indicator_results': {
                'bias': [
                    {'pattern_id': 'BIAS_EXTREME_LOW', 'strength': 85, 'duration': 6}
                ],
                'trix': [
                    {'pattern_id': 'TRIX_GOLDEN_CROSS', 'strength': 75, 'duration': 4}
                ]
            }
        }
    ]
    
    try:
        # 创建临时报告文件
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
            temp_report_path = temp_file.name

        try:
            # 生成报告
            print("生成指标报告...")

            # 首先需要提取共性指标
            common_indicators = {}
            for buypoint in mock_buypoints:
                for indicator_name, patterns in buypoint['indicator_results'].items():
                    if indicator_name not in common_indicators:
                        common_indicators[indicator_name] = []
                    common_indicators[indicator_name].extend(patterns)

            analyzer._generate_indicators_report(common_indicators, temp_report_path)

            # 读取生成的报告
            with open(temp_report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()

            print(f"✅ 报告生成成功，长度: {len(report_content)} 字符")

            # 检查报告内容是否包含非形态数据列
            problematic_terms = ['code', 'name', 'date', 'open', 'high', 'low', 'close', 'volume',
                               'macd_line', 'macd_signal', 'macd_histogram', 'chaikin_oscillator']
            found_issues = []

            lines = report_content.split('\n')
            for line_num, line in enumerate(lines, 1):
                for term in problematic_terms:
                    # 检查是否作为表格列出现
                    if f'| {term} |' in line or f'|{term}|' in line:
                        found_issues.append(f"第{line_num}行: {term}")

            if found_issues:
                print(f"❌ 报告中仍包含非形态数据列:")
                for issue in found_issues[:10]:  # 只显示前10个问题
                    print(f"  - {issue}")
                if len(found_issues) > 10:
                    print(f"  - ... 还有 {len(found_issues) - 10} 个问题")

                # 显示报告的前几行用于调试
                print("\n报告前20行:")
                for i, line in enumerate(lines[:20], 1):
                    print(f"{i:2d}: {line}")

                return False
            else:
                print("✅ 报告中没有发现非形态数据列")

                # 显示报告的一些统计信息
                pattern_count = report_content.count('pattern_id')
                table_count = report_content.count('|')
                print(f"  - 形态数量: {pattern_count}")
                print(f"  - 表格元素数量: {table_count}")

                # 显示报告的前几行
                print("\n报告前10行:")
                for i, line in enumerate(lines[:10], 1):
                    print(f"{i:2d}: {line}")

                return True

        finally:
            # 清理临时文件
            if os.path.exists(temp_report_path):
                os.unlink(temp_report_path)
            
    except Exception as e:
        print(f"❌ 报告生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始测试报告生成数据污染修复效果...\n")
    
    # 测试报告生成
    report_test_passed = test_report_generation()
    
    print("\n=== 测试总结 ===")
    if report_test_passed:
        print("✅ 报告生成测试通过！数据污染问题已修复。")
    else:
        print("❌ 报告生成测试失败，仍存在数据污染问题。")

if __name__ == "__main__":
    main()
