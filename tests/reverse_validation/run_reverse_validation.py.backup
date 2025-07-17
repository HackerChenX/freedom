#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
选股系统反向验证测试主执行脚本

运行完整的反向验证测试，生成详细报告
"""

import sys
import os
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.reverse_validation.reverse_validation_framework import Reverse_validation_framework


def main_runreversevalidation():
    """主函数"""
    parser = argparse.ArgumentParser(description='选股系统反向验证测试')
    parser.add_argument('--indicators', nargs='+',
                       choices=['KDJ', 'RSI', 'MACD', 'BOLL', 'MA', 'EMA'],
                       help='要测试的指标列表，默认测试所有核心指标')
    parser.add_argument('--output-dir', default='tests/reverse_validation/results',
                       help='输出目录，默认为 tests/reverse_validation/results')
    parser.add_argument('--report-format', choices=['markdown', 'json', 'both'],
                       default='both', help='报告格式')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化验证框架
    print("初始化反向验证测试框架...")
    framework = Reverse_validation_framework()

    # 运行批量验证
    print("开始运行反向验证测试...")
    print(f"测试指标: {args.indicators if args.indicators else '所有核心指标'}")
    print("-" * 60)

    batch_results = framework.run_batch_validation(indicators=args.indicators)

    print("-" * 60)
    print("测试完成！")

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 生成报告
    if args.report_format in ['markdown', 'both']:
        report_file = os.path.join(args.output_dir, f'reverse_validation_report_{timestamp}.md')
        report_content = framework.generate_detailed_report(batch_results, report_file)

        if args.verbose:
            print("\n" + "="*80)
            print("详细报告:")
            print("="*80)
            print(report_content)

    if args.report_format in ['json', 'both']:
        json_file = os.path.join(args.output_dir, f'reverse_validation_results_{timestamp}.json')
        framework.save_results_to_json(batch_results, json_file)

    # 输出摘要
    print("\n" + "="*60)
    print("测试摘要:")
    print("="*60)
    print(f"总测试数: {batch_results['total_tests']}")
    print(f"成功测试数: {batch_results['successful_tests']}")
    print(f"失败测试数: {batch_results['failed_tests']}")
    print(f"成功率: {batch_results['summary']['success_rate']:.2%}")
    print(f"平均匹配分: {batch_results['average_match_score']:.3f}")

    if batch_results['summary']['best_indicator']:
        best = batch_results['summary']['best_indicator']
        print(f"最佳指标: {best['name']} (分数: {best['score']:.3f})")

    if batch_results['summary']['worst_indicator']:
        worst = batch_results['summary']['worst_indicator']
        print(f"需要改进的指标: {worst['name']} (分数: {worst['score']:.3f})")

    # 输出建议
    if batch_results['summary']['recommendations']:
        print("\n建议:")
        for i, recommendation in enumerate(batch_results['summary']['recommendations'], 1):
            print(f"{i}. {recommendation}")

    print(f"\n详细结果已保存到: {args.output_dir}")

    # 返回退出码
    success_rate = batch_results['summary']['success_rate']
    if success_rate >= 0.8:
        print("✅ 测试结果优秀")
        return 0
    elif success_rate >= 0.6:
        print("⚠️ 测试结果良好，但有改进空间")
        return 0
    else:
        print("❌ 测试结果需要改进")
        return 1


if __name__ == '__main__':
    exit_code = main_runreversevalidation()
    sys.exit(exit_code)