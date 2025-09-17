#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全面指标形态策略测试启动脚本

快速启动全面的指标形态策略测试和验证流程。

使用方法:
    python run_comprehensive_test.py [选项]

选项:
    --stock-pool-size: 测试股票池大小 (默认: 100)
    --max-time: 最大执行时间(秒) (默认: 300)
    --enable-validation: 启用闭环验证 (默认: True)
    --output-dir: 输出目录 (默认: data/comprehensive_test_results)

Author: AI Assistant
Date: 2025-07-19
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from comprehensive_indicator_pattern_strategy_tester import ComprehensiveIndicatorPatternStrategyTester
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='全面指标形态策略测试系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    # 基本测试 (100只股票)
    python run_comprehensive_test.py
    
    # 大规模测试 (1000只股票)
    python run_comprehensive_test.py --stock-pool-size 1000
    
    # 快速测试 (50只股票，2分钟限制)
    python run_comprehensive_test.py --stock-pool-size 50 --max-time 120
    
    # 禁用闭环验证的测试
    python run_comprehensive_test.py --disable-validation
        """
    )
    
    parser.add_argument(
        '--stock-pool-size',
        type=int,
        default=100,
        help='测试股票池大小 (默认: 100)'
    )
    
    parser.add_argument(
        '--max-time',
        type=int,
        default=300,
        help='最大执行时间(秒) (默认: 300)'
    )
    
    parser.add_argument(
        '--disable-validation',
        action='store_true',
        help='禁用闭环验证'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/comprehensive_test_results',
        help='输出目录 (默认: data/comprehensive_test_results)'
    )
    
    parser.add_argument(
        '--parallel-workers',
        type=int,
        default=8,
        help='并行工作线程数 (默认: 8)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='批处理大小 (默认: 50)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出模式'
    )
    
    return parser.parse_args()


def create_test_config(args):
    """创建测试配置"""
    config = {
        'testing': {
            'max_execution_time': args.max_time,
            'stock_pool_size': args.stock_pool_size,
            'parallel_workers': args.parallel_workers,
            'batch_size': args.batch_size,
            'early_stop_enabled': True,
            'memory_optimization': True
        },
        'validation': {
            'min_selection_count': 1,
            'max_selection_ratio': 0.1,
            'closed_loop_validation': not args.disable_validation,
            'buypoint_analysis_enabled': not args.disable_validation
        },
        'performance': {
            'query_timeout': 30,
            'connection_pool_size': 20,
            'cache_enabled': True,
            'vectorization_enabled': True
        },
        'output': {
            'results_dir': args.output_dir,
            'detailed_report': True,
            'csv_export': True,
            'json_export': True
        }
    }
    
    return config


def print_test_info(config):
    """打印测试信息"""
    print("\n" + "="*80)
    print("🎯 全面指标形态策略测试系统")
    print("="*80)
    print(f"📊 测试股票池大小: {config['testing']['stock_pool_size']:,} 只")
    print(f"⏰ 最大执行时间: {config['testing']['max_execution_time']} 秒")
    print(f"🔄 并行工作线程: {config['testing']['parallel_workers']} 个")
    print(f"📦 批处理大小: {config['testing']['batch_size']} 个")
    print(f"🔍 闭环验证: {'启用' if config['validation']['closed_loop_validation'] else '禁用'}")
    print(f"💾 输出目录: {config['output']['results_dir']}")
    print("="*80)
    print()


def check_system_requirements():
    """检查系统要求"""
    try:
        # 检查Python版本
        if sys.version_info < (3, 8):
            logger.error("❌ 需要Python 3.8或更高版本")
            return False
        
        # 检查必要的包
        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"❌ 缺少必要的包: {', '.join(missing_packages)}")
            logger.info("请运行: pip install pandas numpy matplotlib seaborn")
            return False
        
        # 检查输出目录权限
        test_dir = Path('data/test_permissions')
        try:
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file = test_dir / 'test.txt'
            test_file.write_text('test')
            test_file.unlink()
            test_dir.rmdir()
        except Exception as e:
            logger.error(f"❌ 输出目录权限检查失败: {e}")
            return False
        
        logger.info("✅ 系统要求检查通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 系统要求检查失败: {e}")
        return False


def run_test(config):
    """运行测试"""
    try:
        logger.info("🚀 启动全面指标形态策略测试")
        
        # 创建测试器
        tester = ComprehensiveIndicatorPatternStrategyTester(config)
        
        # 运行测试
        start_time = time.time()
        report = tester.run_comprehensive_test()
        total_time = time.time() - start_time
        
        # 输出结果
        print_test_results(report, total_time)
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("⚠️ 用户中断测试")
        print("\n测试被用户中断")
        return False
        
    except Exception as e:
        logger.error(f"❌ 测试执行失败: {e}")
        print(f"\n测试失败: {e}")
        return False


def print_test_results(report, total_time):
    """打印测试结果"""
    try:
        print("\n" + "="*80)
        print("📋 测试完成 - 结果汇总")
        print("="*80)
        
        # 基本信息
        test_summary = report.get('test_summary', {})
        coverage = report.get('coverage_metrics', {})
        success = report.get('success_metrics', {})
        performance = report.get('performance_metrics', {})
        
        print(f"⏱️  总执行时间: {total_time:.2f} 秒")
        print(f"🎯 性能达标: {'✅ 是' if performance.get('performance_compliant', False) else '❌ 否'}")
        print()
        
        print("📊 测试覆盖:")
        print(f"   技术指标: {coverage.get('total_indicators', 0):,} 个")
        print(f"   形态模式: {coverage.get('total_patterns', 0):,} 个")
        print(f"   生成策略: {coverage.get('total_strategies', 0):,} 个")
        print(f"   测试股票: {coverage.get('total_stocks_tested', 0):,} 只")
        print()
        
        print("🎯 成功指标:")
        print(f"   策略成功率: {success.get('success_rate', 0):.1f}%")
        print(f"   选股成功率: {success.get('selection_rate', 0):.1f}%")
        print(f"   验证成功率: {success.get('validation_rate', 0):.1f}%")
        print()
        
        # 改进建议
        recommendations = report.get('recommendations', [])
        if recommendations:
            print("💡 改进建议:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec}")
            if len(recommendations) > 5:
                print(f"   ... 还有 {len(recommendations) - 5} 条建议")
        
        print("="*80)
        print(f"📁 详细报告已保存到: {report.get('output_dir', 'data/comprehensive_test_results')}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ 打印测试结果失败: {e}")
        print(f"结果显示失败: {e}")


def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_arguments()
        
        # 创建配置
        config = create_test_config(args)
        
        # 打印测试信息
        print_test_info(config)
        
        # 检查系统要求
        if not check_system_requirements():
            sys.exit(1)
        
        # 确认开始测试
        if not args.verbose:
            response = input("是否开始测试? (y/N): ")
            if response.lower() not in ['y', 'yes', '是']:
                print("测试已取消")
                sys.exit(0)
        
        # 运行测试
        success = run_test(config)
        
        if success:
            print("\n🎉 测试成功完成!")
            sys.exit(0)
        else:
            print("\n❌ 测试失败")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        print(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
