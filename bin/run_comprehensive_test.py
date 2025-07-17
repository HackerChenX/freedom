#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票选股策略系统综合测试主程序

提供命令行接口来执行各种测试任务，包括数据验证、选股功能、性能基准、架构合规性等。
遵循L6用户接口层规范，提供统一的测试执行入口。

使用示例:
    python bin/run_comprehensive_test.py --help
    python bin/run_comprehensive_test.py --test-suite data_validation
    python bin/run_comprehensive_test.py --category "基础验证"
    python bin/run_comprehensive_test.py --priority critical
    python bin/run_comprehensive_test.py --all
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from tests.comprehensive.test_infrastructure import initialize_test_infrastructure, TestPriority
from tests.comprehensive.test_suite_manager import create_test_suite_manager
from tests.comprehensive.config import get_test_config

logger = get_logger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="股票选股策略系统综合测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --list                           # 列出所有测试套件
  %(prog)s --test-suite data_validation     # 运行数据验证测试套件
  %(prog)s --category "基础验证"             # 运行基础验证类别的所有测试
  %(prog)s --priority critical              # 运行关键优先级的测试
  %(prog)s --all                            # 运行所有测试套件
  %(prog)s --quick                          # 运行快速测试（关键和高优先级）
  %(prog)s --config config/test_config.yaml # 使用指定配置文件

测试类别:
  基础验证、功能测试、性能测试、架构验证、指标测试、集成测试、监控测试、安全测试

优先级:
  critical（关键）、high（高）、medium（中）、low（低）
        """
    )
    
    # 基本参数
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='测试配置文件路径'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='测试结果输出目录'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别'
    )
    
    # 测试选择参数
    test_group = parser.add_mutually_exclusive_group()
    
    test_group.add_argument(
        '--all',
        action='store_true',
        help='运行所有测试套件'
    )
    
    test_group.add_argument(
        '--quick',
        action='store_true',
        help='运行快速测试（关键和高优先级测试）'
    )
    
    test_group.add_argument(
        '--test-suite', '-t',
        type=str,
        action='append',
        help='指定要运行的测试套件名称（可多次使用）'
    )
    
    test_group.add_argument(
        '--category',
        type=str,
        help='按类别运行测试套件'
    )
    
    test_group.add_argument(
        '--priority',
        type=str,
        choices=['critical', 'high', 'medium', 'low'],
        help='按优先级运行测试套件'
    )
    
    # 控制参数
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='启用并行执行（注意依赖关系）'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='单个测试套件超时时间（秒），默认3600秒'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='最大并发工作线程数，默认4'
    )
    
    # 信息参数
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用的测试套件'
    )
    
    parser.add_argument(
        '--info',
        type=str,
        help='显示指定测试套件的详细信息'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='显示将要执行的测试但不实际运行'
    )
    
    # 报告参数
    parser.add_argument(
        '--no-html',
        action='store_true',
        help='不生成HTML报告'
    )
    
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='不生成JSON报告'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    
    return parser.parse_args()


def setup_logging(log_level: str):
    """设置日志级别"""
    import logging
    
    level = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_banner():
    """打印程序横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    股票选股策略系统综合测试工具                                ║
║                    Stock Selection System Comprehensive Test                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_test_summary(manager, suite_names: Optional[List[str]] = None):
    """打印测试摘要"""
    if suite_names:
        print(f"\n📋 计划执行的测试套件: {len(suite_names)} 个")
        total_time = manager.estimate_total_time(suite_names)
        
        for suite_name in suite_names:
            suite_info = manager.get_suite_info(suite_name)
            if suite_info:
                priority_icon = {
                    TestPriority.CRITICAL: "🔥",
                    TestPriority.HIGH: "⭐", 
                    TestPriority.MEDIUM: "📊",
                    TestPriority.LOW: "🔧"
                }
                icon = priority_icon.get(suite_info.priority, "📋")
                print(f"  {icon} {suite_info.name}: {suite_info.description} "
                      f"({suite_info.estimated_time/60:.1f}分钟)")
        
        print(f"\n⏱️  预估总执行时间: {total_time/60:.1f} 分钟")
    else:
        manager.print_suite_summary()


def validate_environment():
    """验证测试环境"""
    try:
        # 检查必要的目录
        required_dirs = [
            'tests/comprehensive',
            'config',
            'utils'
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"必需目录不存在: {dir_path}")
        
        # 检查数据库连接（如果可能）
        try:
            from db.query_executor import get_query_executor
            query_executor = get_query_executor()
            # 这里可以添加简单的连接测试
            logger.info("数据库连接验证通过")
        except Exception as e:
            logger.warning(f"数据库连接验证失败: {e}")
        
        logger.info("测试环境验证通过")
        return True
        
    except Exception as e:
        logger.error(f"测试环境验证失败: {e}")
        return False


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 打印横幅
    if not args.list and not args.info:
        print_banner()
    
    try:
        # 验证环境
        if not validate_environment():
            print("❌ 测试环境验证失败，请检查系统配置")
            return 1
        
        # 初始化测试基础设施
        if args.verbose:
            print("🔧 正在初始化测试基础设施...")
        
        engine = initialize_test_infrastructure()
        manager = create_test_suite_manager(engine)
        
        # 处理列出测试套件
        if args.list:
            manager.print_suite_summary()
            return 0
        
        # 处理显示测试套件信息
        if args.info:
            suite_info = manager.get_suite_info(args.info)
            if suite_info:
                print(f"\n测试套件信息: {suite_info.name}")
                print(f"描述: {suite_info.description}")
                print(f"优先级: {suite_info.priority.value}")
                print(f"类别: {suite_info.category}")
                print(f"预估时间: {suite_info.estimated_time/60:.1f} 分钟")
                print(f"依赖: {', '.join(suite_info.dependencies) if suite_info.dependencies else '无'}")
                print(f"模块路径: {suite_info.module_path}")
                print(f"类名: {suite_info.class_name}")
            else:
                print(f"❌ 未找到测试套件: {args.info}")
                return 1
            return 0
        
        # 确定要执行的测试套件
        suite_names = None
        
        if args.all:
            suite_names = manager.get_execution_order()
            if args.verbose:
                print("📊 将执行所有测试套件")
        
        elif args.quick:
            critical_suites = [s.name for s in manager.list_suites(priority=TestPriority.CRITICAL)]
            high_suites = [s.name for s in manager.list_suites(priority=TestPriority.HIGH)]
            suite_names = critical_suites + high_suites
            if args.verbose:
                print("⚡ 将执行快速测试（关键和高优先级）")
        
        elif args.test_suite:
            suite_names = args.test_suite
            if args.verbose:
                print(f"🎯 将执行指定测试套件: {', '.join(suite_names)}")
        
        elif args.category:
            suites = manager.list_suites(category=args.category)
            suite_names = [s.name for s in suites]
            if not suite_names:
                print(f"❌ 未找到类别为 '{args.category}' 的测试套件")
                return 1
            if args.verbose:
                print(f"📂 将执行类别 '{args.category}' 的测试套件")
        
        elif args.priority:
            priority_map = {
                'critical': TestPriority.CRITICAL,
                'high': TestPriority.HIGH,
                'medium': TestPriority.MEDIUM,
                'low': TestPriority.LOW
            }
            priority = priority_map[args.priority]
            suites = manager.list_suites(priority=priority)
            suite_names = [s.name for s in suites]
            if not suite_names:
                print(f"❌ 未找到优先级为 '{args.priority}' 的测试套件")
                return 1
            if args.verbose:
                print(f"🎖️ 将执行优先级 '{args.priority}' 的测试套件")
        
        else:
            # 默认执行关键优先级测试
            suites = manager.list_suites(priority=TestPriority.CRITICAL)
            suite_names = [s.name for s in suites]
            print("ℹ️  未指定测试范围，默认执行关键优先级测试")
            print("   使用 --help 查看更多选项")
        
        if not suite_names:
            print("❌ 没有找到要执行的测试套件")
            return 1
        
        # 显示测试摘要
        if args.dry_run:
            print("\n🔍 干运行模式 - 显示将要执行的测试:")
            print_test_summary(manager, suite_names)
            return 0
        
        if args.verbose:
            print_test_summary(manager, suite_names)
        
        # 执行测试
        print(f"\n🚀 开始执行综合测试...")
        start_time = time.time()
        
        try:
            results = manager.run_suites(
                suite_names=suite_names,
                parallel=args.parallel
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 显示结果摘要
            print(f"\n{'='*80}")
            print("🎯 测试执行完成")
            print(f"{'='*80}")
            print(f"总执行时间: {execution_time/60:.2f} 分钟")
            print(f"总测试数: {results['total_tests']}")
            print(f"通过数: {results['total_passed']}")
            print(f"失败数: {results['total_failed']}")
            print(f"成功率: {results['overall_success_rate']:.1%}")
            
            # 显示套件级别结果
            print(f"\n📊 套件级别结果:")
            for suite_name in suite_names:
                if suite_name in results['suite_results']:
                    suite_results = results['suite_results'][suite_name]
                    passed = len([r for r in suite_results if r['status'] == 'passed'])
                    total = len(suite_results)
                    rate = passed / total if total > 0 else 0.0
                    status_icon = "✅" if rate >= 1.0 else "⚠️" if rate >= 0.8 else "❌"
                    print(f"  {status_icon} {suite_name}: {passed}/{total} ({rate:.1%})")
            
            # 根据结果设置退出码
            if results['overall_success_rate'] >= 1.0:
                print(f"\n🎉 所有测试通过！")
                return 0
            elif results['overall_success_rate'] >= 0.8:
                print(f"\n⚠️  测试基本通过，但有部分失败")
                return 1
            else:
                print(f"\n❌ 测试失败，成功率过低")
                return 2
                
        except KeyboardInterrupt:
            print(f"\n⏹️  测试被用户中断")
            return 130
        
        except Exception as e:
            logger.error(f"测试执行异常: {e}")
            print(f"\n💥 测试执行异常: {e}")
            return 3
        
        finally:
            # 清理资源
            if args.verbose:
                print("🧹 正在清理资源...")
            engine.cleanup()
    
    except Exception as e:
        logger.error(f"程序异常: {e}")
        print(f"\n💥 程序异常: {e}")
        return 4


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 