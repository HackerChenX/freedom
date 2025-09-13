#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生产级股票选股系统测试脚本

验证完整的端到端工作流程
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from strategy.historical_buypoint_strategy_generator import (
    HistoricalBuyPointStrategyGenerator, BuyPointInput,
    StrategyGenerationMode, PatternRecognitionMethod
)


def test_strategy_generation():
    """测试策略生成功能"""
    print("🧪 测试历史买点策略生成功能...")

    # 创建测试买点数据
    test_buypoints = [
        BuyPointInput("000001", "2024-01-15", expected_return=8.5),
        BuyPointInput("000002", "2024-01-16", expected_return=6.8),
        BuyPointInput("000858", "2024-01-18", expected_return=12.3),
        BuyPointInput("002415", "2024-01-20", expected_return=9.2),
        BuyPointInput("600036", "2024-01-22", expected_return=7.1)
    ]

    # 初始化策略生成器
    generator = HistoricalBuyPointStrategyGenerator(
        generation_mode=StrategyGenerationMode.BALANCED,
        pattern_method=PatternRecognitionMethod.STATISTICAL
    )

    try:
        # 生成策略
        print(f"  📊 输入买点数量: {len(test_buypoints)}")
        strategy = generator.generate_strategy_from_buypoints(
            test_buypoints,
            "测试策略"
        )

        if strategy:
            print("  ✅ 策略生成成功!")
            print(f"     策略名称: {strategy.strategy_name}")
            print(f"     技术模式数量: {len(strategy.technical_patterns)}")
            print(f"     预期成功率: {strategy.expected_success_rate:.2%}")
            print(f"     预期收益: {strategy.expected_return:.2f}%")
            print(f"     风险级别: {strategy.risk_level}")

            # 显示技术模式
            print("     技术模式:")
            for i, pattern in enumerate(strategy.technical_patterns, 1):
                print(f"       {i}. {pattern.to_condition_string()} (置信度: {pattern.confidence:.2f})")

            return True
        else:
            print("  ❌ 策略生成失败")
            return False

    except Exception as e:
        print(f"  ❌ 策略生成异常: {e}")
        return False


def test_command_line_tool():
    """测试命令行工具"""
    print("\n🧪 测试统一命令行工具...")

    # 创建临时买点文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = [
            {
                "stock_code": "000001",
                "buypoint_date": "2024-01-15",
                "expected_return": 8.5,
                "note": "测试买点1"
            },
            {
                "stock_code": "000002",
                "buypoint_date": "2024-01-16",
                "expected_return": 6.8,
                "note": "测试买点2"
            },
            {
                "stock_code": "000858",
                "buypoint_date": "2024-01-18",
                "expected_return": 12.3,
                "note": "测试买点3"
            }
        ]
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        temp_file = f.name

    # 创建临时输出目录
    temp_output_dir = tempfile.mkdtemp()

    try:
        # 测试命令行工具
        from bin.production_stock_selector import ProductionStockSelector

        selector = ProductionStockSelector()
        results = selector.execute_full_pipeline(
            buypoints_file=temp_file,
            output_dir=temp_output_dir,
            strategy_name="命令行测试策略"
        )

        if results['status'] == 'success':
            print("  ✅ 命令行工具测试成功!")
            print(f"     执行耗时: {results.get('total_duration', '未知')}")

            # 检查各步骤状态
            steps = results['steps']
            for step_name, step_info in steps.items():
                status_icon = "✅" if step_info['status'] == 'success' else "⚠️"
                print(f"     {status_icon} {step_name}: {step_info['message']}")

            return True
        else:
            print(f"  ❌ 命令行工具测试失败: {results.get('error', '未知错误')}")
            return False

    except Exception as e:
        print(f"  ❌ 命令行工具测试异常: {e}")
        return False

    finally:
        # 清理临时文件
        try:
            os.unlink(temp_file)
            import shutil
            shutil.rmtree(temp_output_dir, ignore_errors=True)
        except:
            pass


def test_data_loading():
    """测试数据加载功能"""
    print("\n🧪 测试数据加载功能...")

    try:
        from bin.production_stock_selector import ProductionStockSelector

        selector = ProductionStockSelector()

        # 创建测试CSV文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("stock_code,buypoint_date,expected_return,note\n")
            f.write("000001,2024-01-15,8.5,测试买点1\n")
            f.write("000002,2024-01-16,6.8,测试买点2\n")
            temp_csv_file = f.name

        # 测试CSV加载
        buypoints = selector._load_buypoint_data(temp_csv_file)

        if len(buypoints) == 2:
            print("  ✅ CSV数据加载成功!")
            print(f"     加载买点数量: {len(buypoints)}")
            for bp in buypoints:
                print(f"     - {bp.stock_code} {bp.buypoint_date}")
            return True
        else:
            print(f"  ❌ CSV数据加载失败: 期望2个，实际{len(buypoints)}个")
            return False

    except Exception as e:
        print(f"  ❌ 数据加载测试异常: {e}")
        return False

    finally:
        try:
            os.unlink(temp_csv_file)
        except:
            pass


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始生产级股票选股系统测试")
    print("=" * 50)

    tests = [
        ("数据加载功能", test_data_loading),
        ("策略生成功能", test_strategy_generation),
        ("命令行工具", test_command_line_tool)
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 测试项目: {test_name}")
        print("-" * 30)
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"  ❌ {test_name} 测试失败")
        except Exception as e:
            print(f"  💥 {test_name} 测试异常: {e}")

    # 测试总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结")
    print("=" * 50)
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"通过率: {passed_tests/total_tests:.2%}")

    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！系统功能正常。")
        return True
    else:
        print(f"\n⚠️ 有 {total_tests - passed_tests} 个测试失败，请检查系统。")
        return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='生产级股票选股系统测试脚本')
    parser.add_argument('--test', choices=['all', 'data', 'strategy', 'cli'],
                       default='all', help='指定测试项目')

    args = parser.parse_args()

    if args.test == 'all':
        success = run_all_tests()
    elif args.test == 'data':
        success = test_data_loading()
    elif args.test == 'strategy':
        success = test_strategy_generation()
    elif args.test == 'cli':
        success = test_command_line_tool()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())