#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行更新后的反向验证测试

基于重构后的系统架构，验证技术指标形态识别的准确性
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tests.reverse_validation.reverse_validation_framework import Reverse_validation_framework
from utils.logger import getLogger

logger = getLogger(__name__)


async def main():
    """主函数"""
    print("=" * 80)
    print("股票选股系统 - 反向验证测试 (重构版本)")
    print("=" * 80)
    
    try:
        # 初始化测试框架
        print("\n1. 初始化测试框架...")
        framework = Reverse_validation_framework()
        
        # 验证系统状态
        print("\n2. 验证重构后的系统状态...")
        system_validation = framework.validate_refactored_system()
        
        print(f"   买点分析器: {'✓' if system_validation['buypoint_analyzer'] else '✗'}")
        print(f"   形态注册表: {'✓' if system_validation['pattern_registry'] else '✗'}")
        print(f"   数据访问层: {'✓' if system_validation['data_access'] else '✗'}")
        print(f"   形态生成器: {'✓' if system_validation['pattern_generator'] else '✗'}")
        print(f"   整体状态: {'✓' if system_validation['overall_status'] else '✗'}")
        
        if not system_validation['overall_status']:
            print("\n❌ 系统验证失败，请检查系统配置")
            if 'error' in system_validation:
                print(f"错误信息: {system_validation['error']}")
            return
        
        # 运行单个指标测试
        print("\n3. 运行单个指标测试示例...")
        test_indicators = ['MACD', 'RSI', 'KDJ']
        
        for indicator in test_indicators:
            print(f"\n   测试指标: {indicator}")
            
            # 获取该指标的形态
            patterns = framework._get_patterns_for_indicator(indicator)
            print(f"   发现 {len(patterns)} 个相关形态: {patterns[:3]}...")
            
            # 测试第一个形态
            if patterns:
                pattern_name = patterns[0]
                try:
                    # 生成测试数据
                    pattern_data = framework.pattern_generator.generate_pattern_data(
                        pattern_type=pattern_name,
                        data_points=60,
                        stock_code=f"TEST_{indicator}"
                    )
                    
                    if pattern_data is not None and not pattern_data.empty:
                        # 运行验证
                        result = framework.run_single_pattern_validation(
                            indicator, pattern_name, pattern_data
                        )
                        
                        status = "✓" if result['is_successful'] else "✗"
                        score = result['match_score']
                        print(f"   {pattern_name}: {status} (匹配度: {score:.2f})")
                        
                        if result['identified_patterns']:
                            print(f"   识别出的形态: {result['identified_patterns'][:3]}")
                    else:
                        print(f"   {pattern_name}: ⚠ 无法生成测试数据")
                        
                except Exception as e:
                    print(f"   {pattern_name}: ✗ 测试失败 - {e}")
        
        # 运行全面验证测试
        print("\n4. 运行全面验证测试...")
        comprehensive_result = await framework.run_comprehensive_validation_async(['MACD', 'RSI'])
        
        summary = comprehensive_result['test_summary']
        print(f"   总测试数: {summary['total_tests']}")
        print(f"   成功测试: {summary['successful_tests']}")
        print(f"   失败测试: {summary['failed_tests']}")
        print(f"   成功率: {summary['success_rate']:.1%}")
        print(f"   耗时: {summary['duration_seconds']:.2f}秒")
        
        # 保存结果
        print("\n5. 保存测试结果...")
        output_dir = Path("tests/data/result")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        json_file = output_dir / f"reverse_validation_result_{timestamp}.json"
        framework.save_results_to_json(comprehensive_result, str(json_file))
        
        # 生成报告
        report_file = output_dir / f"reverse_validation_report_{timestamp}.md"
        framework.generate_validation_report(comprehensive_result, str(report_file))
        
        print(f"   结果文件: {json_file}")
        print(f"   报告文件: {report_file}")
        
        # 显示总结
        print("\n" + "=" * 80)
        print("测试完成总结:")
        print(f"✓ 系统验证: {'通过' if system_validation['overall_status'] else '失败'}")
        print(f"✓ 全面测试: {summary['successful_tests']}/{summary['total_tests']} 通过")
        print(f"✓ 整体成功率: {summary['success_rate']:.1%}")
        print(f"✓ 性能指标: {comprehensive_result['performance_metrics']['tests_per_second']:.1f} 测试/秒")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        print(f"\n❌ 测试执行失败: {e}")
        return 1
    
    return 0


def run_quick_test():
    """运行快速测试"""
    print("运行快速反向验证测试...")
    
    try:
        framework = Reverse_validation_framework()
        
        # 验证系统
        system_status = framework.validate_refactored_system()
        if not system_status['overall_status']:
            print("❌ 系统验证失败")
            return False
        
        # 测试单个形态
        pattern_data = framework.pattern_generator.generate_pattern_data(
            "MACD_GOLDEN_CROSS", 30, "QUICK_TEST"
        )
        
        if pattern_data is not None:
            result = framework.run_single_pattern_validation(
                "MACD", "MACD_GOLDEN_CROSS", pattern_data
            )
            
            print(f"快速测试结果: {'✓' if result['is_successful'] else '✗'}")
            print(f"匹配度: {result['match_score']:.2f}")
            return result['is_successful']
        else:
            print("❌ 无法生成测试数据")
            return False
            
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="反向验证测试")
    parser.add_argument("--quick", action="store_true", help="运行快速测试")
    parser.add_argument("--indicators", nargs="+", help="指定要测试的指标")
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        # 运行完整测试
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
