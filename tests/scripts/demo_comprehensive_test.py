#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全面指标形态策略测试演示脚本

快速演示全面指标形态策略测试系统的功能。
使用较小的数据集进行快速测试，验证系统的完整性。

Author: AI Assistant
Date: 2025-07-19
"""

import os
import sys
import time
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger

logger = get_logger(__name__)


def create_demo_config():
    """创建演示配置"""
    return {
        'testing': {
            'max_execution_time': 120,  # 2分钟演示
            'stock_pool_size': 20,      # 20只股票演示
            'parallel_workers': 4,      # 4个工作线程
            'batch_size': 5,           # 小批次处理
            'early_stop_enabled': True,
            'memory_optimization': True
        },
        'validation': {
            'min_selection_count': 1,
            'max_selection_ratio': 0.2,  # 更宽松的选股比例
            'closed_loop_validation': True,
            'buypoint_analysis_enabled': True
        },
        'performance': {
            'query_timeout': 15,
            'connection_pool_size': 10,
            'cache_enabled': True,
            'vectorization_enabled': True
        },
        'output': {
            'results_dir': 'data/demo_test_results',
            'detailed_report': True,
            'csv_export': True,
            'json_export': True
        }
    }


def run_demo():
    """运行演示"""
    try:
        print("\n" + "="*60)
        print("🎯 全面指标形态策略测试系统 - 演示模式")
        print("="*60)
        print("📊 演示配置:")
        print("   - 测试股票: 20只")
        print("   - 最大时间: 2分钟")
        print("   - 并行线程: 4个")
        print("   - 启用闭环验证")
        print("="*60)
        
        # 导入测试器
        try:
            from comprehensive_indicator_pattern_strategy_tester import ComprehensiveIndicatorPatternStrategyTester
        except ImportError as e:
            logger.error(f"❌ 导入测试器失败: {e}")
            print("请确保所有依赖文件都在正确位置")
            return False
        
        # 创建配置
        config = create_demo_config()
        
        # 创建测试器
        print("🔧 初始化测试器...")
        tester = ComprehensiveIndicatorPatternStrategyTester(config)
        
        # 运行测试
        print("🚀 开始演示测试...")
        start_time = time.time()
        
        try:
            report = tester.run_comprehensive_test()
            total_time = time.time() - start_time
            
            # 显示结果
            print_demo_results(report, total_time)
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试执行失败: {e}")
            print(f"测试执行失败: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 演示运行失败: {e}")
        print(f"演示运行失败: {e}")
        return False


def print_demo_results(report, total_time):
    """打印演示结果"""
    try:
        print("\n" + "="*60)
        print("📋 演示测试完成 - 结果汇总")
        print("="*60)
        
        # 获取结果数据
        test_summary = report.get('test_summary', {})
        coverage = report.get('coverage_metrics', {})
        success = report.get('success_metrics', {})
        performance = report.get('performance_metrics', {})
        issue_summary = report.get('issue_summary', {})
        recommendations = report.get('recommendations', [])
        
        # 基本信息
        print(f"⏱️  执行时间: {total_time:.2f} 秒")
        print(f"🎯 性能状态: {'✅ 达标' if performance.get('performance_compliant', False) else '❌ 超时'}")
        print()
        
        # 测试覆盖
        print("📊 测试覆盖:")
        print(f"   技术指标: {coverage.get('total_indicators', 0)} 个")
        print(f"   形态模式: {coverage.get('total_patterns', 0)} 个")
        print(f"   生成策略: {coverage.get('total_strategies', 0)} 个")
        print(f"   测试股票: {coverage.get('total_stocks_tested', 0)} 只")
        print()
        
        # 成功指标
        print("🎯 成功指标:")
        success_rate = success.get('success_rate', 0)
        selection_rate = success.get('selection_rate', 0)
        validation_rate = success.get('validation_rate', 0)
        
        print(f"   策略成功率: {success_rate:.1f}% {get_status_emoji(success_rate, 70)}")
        print(f"   选股成功率: {selection_rate:.1f}% {get_status_emoji(selection_rate, 50)}")
        print(f"   验证成功率: {validation_rate:.1f}% {get_status_emoji(validation_rate, 60)}")
        print()
        
        # 问题统计
        total_issues = issue_summary.get('total_issues', 0)
        print(f"🔧 发现问题: {total_issues} 个 {get_issue_emoji(total_issues)}")
        
        if total_issues > 0:
            issues_by_type = issue_summary.get('issues_by_type', {})
            for issue_type, count in issues_by_type.items():
                print(f"   {issue_type}: {count} 个")
        print()
        
        # 改进建议
        if recommendations:
            print("💡 主要建议:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
            if len(recommendations) > 3:
                print(f"   ... 还有 {len(recommendations) - 3} 条建议")
        else:
            print("💡 暂无改进建议")
        
        print("="*60)
        
        # 输出文件信息
        output_dir = config.get('output', {}).get('results_dir', 'data/demo_test_results')
        print(f"📁 详细报告保存在: {output_dir}")
        
        # 演示总结
        print("\n🎉 演示完成!")
        if success_rate >= 70 and selection_rate >= 50:
            print("✅ 系统运行良好，可以进行大规模测试")
        elif total_issues == 0:
            print("⚠️ 系统基本正常，建议优化策略参数")
        else:
            print("❌ 发现问题，建议先修复后再进行大规模测试")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ 打印演示结果失败: {e}")
        print(f"结果显示失败: {e}")


def get_status_emoji(rate, threshold):
    """获取状态表情符号"""
    if rate >= threshold:
        return "✅"
    elif rate >= threshold * 0.7:
        return "⚠️"
    else:
        return "❌"


def get_issue_emoji(count):
    """获取问题表情符号"""
    if count == 0:
        return "✅"
    elif count <= 5:
        return "⚠️"
    else:
        return "❌"


def check_demo_requirements():
    """检查演示要求"""
    try:
        print("🔍 检查演示环境...")
        
        # 检查Python版本
        if sys.version_info < (3, 8):
            print("❌ 需要Python 3.8或更高版本")
            return False
        
        # 检查基本包
        required_packages = ['pandas', 'numpy']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ 缺少必要的包: {', '.join(missing_packages)}")
            print("请运行: pip install pandas numpy")
            return False
        
        # 检查项目文件
        required_files = [
            'comprehensive_indicator_pattern_strategy_tester.py',
            'pattern_strategy_generator.py',
            'performance_optimizer.py',
            'closed_loop_validator.py',
            'test_report_generator.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ 缺少必要的文件: {', '.join(missing_files)}")
            return False
        
        print("✅ 演示环境检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 环境检查失败: {e}")
        return False


def main():
    """主函数"""
    try:
        print("🎯 全面指标形态策略测试系统 - 演示模式")
        
        # 检查环境
        if not check_demo_requirements():
            print("\n❌ 环境检查失败，无法运行演示")
            sys.exit(1)
        
        # 运行演示
        success = run_demo()
        
        if success:
            print("\n🎉 演示成功完成!")
            print("\n要运行完整测试，请使用:")
            print("python run_comprehensive_test.py --stock-pool-size 1000")
        else:
            print("\n❌ 演示失败")
            print("请检查错误信息并修复问题")
        
    except KeyboardInterrupt:
        print("\n⚠️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示程序失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
