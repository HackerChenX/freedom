#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
统一指标测试执行脚本

快速启动统一测试框架，支持多种测试模式
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Optional

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from unified_indicator_tester import UnifiedIndicatorTester
from utils.logger import getLogger

logger = getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="统一指标测试执行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 测试所有已修复指标
  python run_unified_tests.py --mode all
  
  # 测试特定指标
  python run_unified_tests.py --mode single --indicator MACD
  
  # 测试多个指标
  python run_unified_tests.py --mode multiple --indicators MACD,RSI,KDJ
  
  # 快速测试（仅核心功能）
  python run_unified_tests.py --mode quick
  
  # 性能测试
  python run_unified_tests.py --mode performance --scale 4000
  
  # 复杂条件测试
  python run_unified_tests.py --mode complex
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["all", "single", "multiple", "quick", "performance", "complex"],
        default="all",
        help="测试模式"
    )
    
    parser.add_argument(
        "--indicator", "-i",
        help="单个指标测试时指定的指标名称"
    )
    
    parser.add_argument(
        "--indicators",
        help="多个指标测试时指定的指标名称（逗号分隔）"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="tests/unified_indicator_testing/config.yaml",
        help="测试配置文件路径"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="测试结果输出目录"
    )
    
    parser.add_argument(
        "--scale",
        type=int,
        default=100,
        help="性能测试时的数据规模"
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：要求100%通过率"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行：只显示将要执行的测试，不实际执行"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 统一指标测试框架")
    print("=" * 50)
    print(f"测试模式: {args.mode}")
    print(f"配置文件: {args.config}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        # 创建测试器实例
        tester = UnifiedIndicatorTester(config_path=args.config)
        
        # 根据模式执行不同的测试
        if args.mode == "all":
            results = run_all_tests(tester, args)
        elif args.mode == "single":
            results = run_single_test(tester, args)
        elif args.mode == "multiple":
            results = run_multiple_tests(tester, args)
        elif args.mode == "quick":
            results = run_quick_tests(tester, args)
        elif args.mode == "performance":
            results = run_performance_tests(tester, args)
        elif args.mode == "complex":
            results = run_complex_tests(tester, args)
        else:
            raise ValueError(f"不支持的测试模式: {args.mode}")
        
        # 输出测试结果摘要
        print_test_summary(results)
        
        # 保存测试结果
        if args.output:
            save_test_results(results, args.output)
        
        # 检查测试是否通过
        if args.strict:
            check_strict_mode_results(results)
            
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        print(f"❌ 测试执行失败: {e}")
        sys.exit(1)


def run_all_tests(tester: UnifiedIndicatorTester, args) -> dict:
    """运行所有指标测试"""
    print("📊 开始测试所有已修复指标...")
    
    if args.dry_run:
        indicators = list(tester.config['indicators_test_matrix']['completed_indicators'].keys())
        print(f"将测试以下指标: {', '.join(indicators)}")
        return {}
    
    return tester.test_all_indicators()


def run_single_test(tester: UnifiedIndicatorTester, args) -> dict:
    """运行单个指标测试"""
    if not args.indicator:
        raise ValueError("单个指标测试模式需要指定 --indicator 参数")
    
    print(f"📈 开始测试指标: {args.indicator}")
    
    if args.dry_run:
        print(f"将测试指标: {args.indicator}")
        return {}
    
    result = tester.test_indicator_comprehensive(args.indicator)
    return {args.indicator: result}


def run_multiple_tests(tester: UnifiedIndicatorTester, args) -> dict:
    """运行多个指标测试"""
    if not args.indicators:
        raise ValueError("多个指标测试模式需要指定 --indicators 参数")
    
    indicators = [ind.strip() for ind in args.indicators.split(',')]
    print(f"📊 开始测试指标: {', '.join(indicators)}")
    
    if args.dry_run:
        print(f"将测试以下指标: {', '.join(indicators)}")
        return {}
    
    results = {}
    for indicator in indicators:
        print(f"正在测试: {indicator}")
        results[indicator] = tester.test_indicator_comprehensive(indicator)
    
    return results


def run_quick_tests(tester: UnifiedIndicatorTester, args) -> dict:
    """运行快速测试（仅核心功能）"""
    print("⚡ 开始快速测试（核心指标）...")
    
    # 选择几个核心指标进行快速测试
    core_indicators = ['MACD', 'RSI', 'KDJ']
    
    if args.dry_run:
        print(f"将快速测试以下核心指标: {', '.join(core_indicators)}")
        return {}
    
    results = {}
    for indicator in core_indicators:
        if indicator in tester.config['indicators_test_matrix']['completed_indicators']:
            print(f"快速测试: {indicator}")
            results[indicator] = tester.test_indicator_comprehensive(indicator)
    
    return results


def run_performance_tests(tester: UnifiedIndicatorTester, args) -> dict:
    """运行性能测试"""
    print(f"🚀 开始性能测试（规模: {args.scale}）...")
    
    if args.dry_run:
        print(f"将进行性能测试，数据规模: {args.scale}")
        return {}
    
    # 这里实现性能测试逻辑
    # 暂时返回占位符结果
    return {
        'performance_test': {
            'scale': args.scale,
            'status': 'completed',
            'results': 'Performance test implementation needed'
        }
    }


def run_complex_tests(tester: UnifiedIndicatorTester, args) -> dict:
    """运行复杂条件测试"""
    print("🔧 开始复杂条件组合测试...")
    
    if args.dry_run:
        print("将进行复杂条件组合测试")
        return {}
    
    # 这里实现复杂条件测试逻辑
    # 暂时返回占位符结果
    return {
        'complex_condition_test': {
            'status': 'completed',
            'results': 'Complex condition test implementation needed'
        }
    }


def print_test_summary(results: dict):
    """打印测试结果摘要"""
    print("\n" + "=" * 50)
    print("📋 测试结果摘要")
    print("=" * 50)
    
    if not results:
        print("没有测试结果")
        return
    
    total_indicators = len(results)
    passed_indicators = 0
    failed_indicators = 0
    
    for indicator_name, result in results.items():
        if isinstance(result, dict) and 'overall_score' in result:
            score = result['overall_score']
            if score >= 1.0:
                status = "✅ 通过"
                passed_indicators += 1
            else:
                status = "❌ 失败"
                failed_indicators += 1
            print(f"{indicator_name:15} {status} (评分: {score:.2f})")
        else:
            print(f"{indicator_name:15} ❌ 错误")
            failed_indicators += 1
    
    print("-" * 50)
    print(f"总计: {total_indicators} 个指标")
    print(f"通过: {passed_indicators} 个")
    print(f"失败: {failed_indicators} 个")
    print(f"通过率: {passed_indicators/total_indicators*100:.1f}%")
    
    if passed_indicators == total_indicators:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  存在测试失败")


def save_test_results(results: dict, output_dir: str):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"test_results_{timestamp}.json")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"📁 测试结果已保存到: {result_file}")


def check_strict_mode_results(results: dict):
    """检查严格模式结果"""
    failed_indicators = []
    
    for indicator_name, result in results.items():
        if isinstance(result, dict) and 'overall_score' in result:
            if result['overall_score'] < 1.0:
                failed_indicators.append(indicator_name)
        else:
            failed_indicators.append(indicator_name)
    
    if failed_indicators:
        print(f"\n❌ 严格模式检查失败！以下指标未达到100%通过率:")
        for indicator in failed_indicators:
            print(f"  - {indicator}")
        sys.exit(1)
    else:
        print("\n✅ 严格模式检查通过！所有指标都达到100%通过率")


if __name__ == "__main__":
    main()
