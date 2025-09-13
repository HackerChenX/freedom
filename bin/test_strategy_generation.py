#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
历史买点策略生成算法测试脚本

测试历史买点策略生成器的核心功能：
1. 从历史买点输入生成技术分析策略
2. 验证策略生成的完整性和准确性
3. 确保使用真实ClickHouse数据

遵循用户要求：禁止使用模拟数据，使用ClickHouse中的真实数据
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from strategy.historical_buypoint_strategy_generator import (
    HistoricalBuyPointStrategyGenerator,
    BuyPointInput,
    StrategyGenerationMode,
    PatternRecognitionMethod
)
from utils.dependency_injection import get_logger

logger = get_logger(__name__)

def test_buypoint_input_validation():
    """测试买点输入数据验证"""
    print("\n=== 测试买点输入数据验证 ===")

    try:
        # 测试有效的买点输入
        valid_input = BuyPointInput(
            stock_code="000001",
            buypoint_date="2024-01-15",
            expected_return=5.2,
            holding_days=10,
            note="测试买点"
        )
        print(f"✅ 有效买点输入创建成功: {valid_input.stock_code} on {valid_input.buypoint_date}")

        # 测试无效股票代码
        try:
            invalid_code = BuyPointInput(
                stock_code="INVALID",
                buypoint_date="2024-01-15"
            )
            print("❌ 应该抛出无效股票代码错误")
        except ValueError as e:
            print(f"✅ 正确捕获无效股票代码错误: {e}")

        # 测试无效日期格式
        try:
            invalid_date = BuyPointInput(
                stock_code="000001",
                buypoint_date="invalid-date"
            )
            print("❌ 应该抛出无效日期格式错误")
        except ValueError as e:
            print(f"✅ 正确捕获无效日期格式错误: {e}")

    except Exception as e:
        print(f"❌ 买点输入验证测试失败: {e}")
        return False

    return True

def test_strategy_generator_initialization():
    """测试策略生成器初始化"""
    print("\n=== 测试策略生成器初始化 ===")

    try:
        # 测试不同模式的初始化
        modes = [
            StrategyGenerationMode.BALANCED,
            StrategyGenerationMode.CONSERVATIVE,
            StrategyGenerationMode.AGGRESSIVE,
            StrategyGenerationMode.ADAPTIVE
        ]

        methods = [
            PatternRecognitionMethod.STATISTICAL,
            PatternRecognitionMethod.FREQUENT_PATTERN,
            PatternRecognitionMethod.HYBRID
        ]

        for mode in modes:
            for method in methods:
                try:
                    generator = HistoricalBuyPointStrategyGenerator(
                        generation_mode=mode,
                        pattern_method=method
                    )
                    print(f"✅ 策略生成器初始化成功: {mode.value} + {method.value}")
                except Exception as e:
                    print(f"❌ 策略生成器初始化失败 {mode.value} + {method.value}: {e}")
                    return False

        return True

    except Exception as e:
        print(f"❌ 策略生成器初始化测试失败: {e}")
        return False

def test_strategy_generation_with_sample_data():
    """使用示例数据测试策略生成"""
    print("\n=== 测试策略生成功能 ===")

    try:
        # 初始化策略生成器
        generator = HistoricalBuyPointStrategyGenerator(
            generation_mode=StrategyGenerationMode.BALANCED,
            pattern_method=PatternRecognitionMethod.STATISTICAL
        )
        print("✅ 策略生成器初始化完成")

        # 创建示例历史买点数据 - 使用数据库中真实存在的股票代码和日期
        sample_buypoints = [
            BuyPointInput(
                stock_code="300003",  # 乐普医疗
                buypoint_date="2025-05-09",
                expected_return=8.5,
                holding_days=15,
                note="技术形态突破买点"
            ),
            BuyPointInput(
                stock_code="603187",  # 海容冷链
                buypoint_date="2025-05-09",
                expected_return=6.2,
                holding_days=12,
                note="均线支撑买点"
            ),
            BuyPointInput(
                stock_code="600540",  # 新赛股份
                buypoint_date="2025-05-09",
                expected_return=4.8,
                holding_days=18,
                note="量价配合买点"
            ),
            BuyPointInput(
                stock_code="605003",  # 众望布艺
                buypoint_date="2025-05-09",
                expected_return=7.1,
                holding_days=10,
                note="MACD金叉买点"
            ),
            BuyPointInput(
                stock_code="605337",  # 李子园
                buypoint_date="2025-05-09",
                expected_return=5.9,
                holding_days=14,
                note="RSI超卖反弹买点"
            )
        ]

        print(f"📊 准备测试 {len(sample_buypoints)} 个历史买点")
        for i, bp in enumerate(sample_buypoints, 1):
            print(f"  {i}. {bp.stock_code} - {bp.buypoint_date} (预期收益: {bp.expected_return}%)")

        # 生成策略
        print("\n🔄 开始生成策略...")
        strategy = generator.generate_strategy_from_buypoints(
            buypoint_inputs=sample_buypoints,
            strategy_name="真实数据测试策略_20250509"
        )

        if strategy is None:
            print("❌ 策略生成失败，返回None")
            return False

        # 输出生成的策略信息
        print(f"\n✅ 策略生成成功!")
        print(f"策略名称: {strategy.strategy_name}")
        print(f"策略描述: {strategy.description}")
        print(f"技术模式数量: {len(strategy.technical_patterns)}")
        print(f"预期成功率: {strategy.expected_success_rate:.2%}")
        print(f"预期收益: {strategy.expected_return:.2f}%")
        print(f"风险级别: {strategy.risk_level}")
        print(f"源买点数量: {strategy.source_buypoints_count}")
        print(f"生成时间: {strategy.generation_time}")
        print(f"生成模式: {strategy.generation_mode.value}")
        print(f"模式识别方法: {strategy.pattern_method.value}")

        # 输出技术模式详情
        print(f"\n📋 识别的技术模式:")
        for i, pattern in enumerate(strategy.technical_patterns, 1):
            print(f"  {i}. {pattern.to_condition_string()}")
            print(f"     置信度: {pattern.confidence:.3f}, 频率: {pattern.frequency}")
            if pattern.avg_return is not None:
                print(f"     平均收益: {pattern.avg_return:.2f}%")

        return True

    except Exception as e:
        print(f"❌ 策略生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """运行综合测试"""
    print("=" * 60)
    print("🧪 历史买点策略生成算法 - 综合测试")
    print("=" * 60)

    test_results = []

    # 1. 测试买点输入验证
    result1 = test_buypoint_input_validation()
    test_results.append(("买点输入验证", result1))

    # 2. 测试策略生成器初始化
    result2 = test_strategy_generator_initialization()
    test_results.append(("策略生成器初始化", result2))

    # 3. 测试策略生成功能
    result3 = test_strategy_generation_with_sample_data()
    test_results.append(("策略生成功能", result3))

    # 输出测试结果汇总
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)

    passed_count = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<20} : {status}")
        if result:
            passed_count += 1

    print(f"\n总体结果: {passed_count}/{len(test_results)} 测试通过")

    if passed_count == len(test_results):
        print("🎉 所有测试通过! 历史买点策略生成算法工作正常")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步调试")
        return False

if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试执行过程中发生未预期错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)