#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
历史买点自动策略生成系统 - 测试示例

演示系统的完整使用流程，包括：
1. 准备测试数据
2. 生成策略
3. 验证策略
4. 分析结果
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def create_sample_buypoint_data(num_stocks=20, num_buypoints_per_stock=3):
    """
    创建示例买点数据

    Args:
        num_stocks: 股票数量
        num_buypoints_per_stock: 每只股票的买点数量

    Returns:
        pd.DataFrame: 买点数据
    """
    print("创建示例买点数据...")

    # 生成股票代码
    stock_codes = []
    for i in range(num_stocks):
        if i < 10:
            code = f"00000{i+1}"
        else:
            code = f"60000{i-9}"
        stock_codes.append(code)

    # 生成买点日期
    base_date = datetime(2024, 1, 1)
    buypoint_data = []

    for stock_code in stock_codes:
        for j in range(num_buypoints_per_stock):
            # 随机生成买点日期（2024年1月到8月之间）
            days_offset = np.random.randint(0, 240)  # 8个月约240天
            buypoint_date = (base_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')

            buypoint_data.append({
                'stock_code': stock_code,
                'buypoint_date': buypoint_date,
                'buy_reason': f'技术买点{j+1}',  # 可选字段
                'expected_return': np.random.uniform(0.05, 0.25)  # 可选字段
            })

    df = pd.DataFrame(buypoint_data)
    print(f"生成买点数据: {len(df)} 条记录，涉及 {len(stock_codes)} 只股票")

    return df

def run_example_workflow():
    """运行完整的示例工作流程"""

    print("=" * 60)
    print("历史买点自动策略生成系统 - 测试示例")
    print("=" * 60)

    try:
        # 步骤1: 创建测试数据
        print("\n步骤1: 准备测试数据")
        print("-" * 30)

        buypoint_df = create_sample_buypoint_data(num_stocks=15, num_buypoints_per_stock=4)

        # 保存测试数据
        test_data_file = "./test_buypoint_data.csv"
        buypoint_df.to_csv(test_data_file, index=False)
        print(f"测试数据已保存到: {test_data_file}")

        # 步骤2: 初始化系统
        print("\n步骤2: 初始化系统")
        print("-" * 30)

        from strategy.historical_buypoint_strategy_system import HistoricalBuyPointStrategySystem

        system = HistoricalBuyPointStrategySystem()
        print("系统初始化完成")

        # 准备买点数据列表
        buypoint_data = [(row['stock_code'], row['buypoint_date'])
                        for _, row in buypoint_df.iterrows()]

        print(f"准备分析 {len(buypoint_data)} 个历史买点")

        # 步骤3: 生成策略
        print("\n步骤3: 生成策略")
        print("-" * 30)

        generation_config = {
            'pattern_method': 'hybrid',
            'generation_mode': 'balanced',
            'min_pattern_frequency': 2,  # 降低频率要求便于测试
            'min_success_rate': 0.4
        }

        strategy = system.generate_strategy_from_historical_data(
            buypoint_data=buypoint_data,
            strategy_name="TestStrategy_Example",
            generation_config=generation_config
        )

        print(f"策略生成成功: {strategy.name}")
        print(f"包含模式数量: {len(strategy.generated_strategy.pattern_templates)}")

        # 步骤4: 验证策略
        print("\n步骤4: 验证策略")
        print("-" * 30)

        validation_config = {
            'pool_size': 50,  # 减小测试股票池大小
            'validation_period_days': 20
        }

        validation_report = system.validate_strategy_performance(
            strategy=strategy,
            validation_config=validation_config
        )

        print(f"策略验证完成")
        print(f"整体评分: {validation_report.overall_score:.2f}/100")
        print(f"验证结果: {validation_report.overall_result.value}")

        # 步骤5: 分析结果
        print("\n步骤5: 分析结果")
        print("-" * 30)

        print("选股摘要:")
        selection_summary = validation_report.selection_summary
        print(f"  - 测试股票池: {selection_summary['total_universe']} 只")
        print(f"  - 选出股票: {selection_summary['selected_count']} 只")
        print(f"  - 选股率: {selection_summary['selection_rate']:.2%}")
        print(f"  - 平均评分: {selection_summary['avg_score']:.2f}")

        print("\n买点验证摘要:")
        validation_summary = validation_report.validation_summary
        print(f"  - 验证股票: {validation_summary['total_validations']} 只")
        print(f"  - 通过验证: {validation_summary['passed_count']} 只")
        print(f"  - 成功率: {validation_summary['success_rate']:.2%}")
        print(f"  - 平均质量: {validation_summary['avg_quality_score']:.2f}")

        print("\n优化建议:")
        for i, suggestion in enumerate(validation_report.optimization_suggestions, 1):
            print(f"  {i}. {suggestion}")

        # 步骤6: 保存结果
        print("\n步骤6: 保存结果")
        print("-" * 30)

        output_dir = "./example_output"
        file_paths = system.save_strategy_and_report(
            strategy=strategy,
            validation_report=validation_report,
            output_dir=output_dir
        )

        print("结果文件:")
        for file_type, file_path in file_paths.items():
            print(f"  - {file_type}: {file_path}")

        # 步骤7: 展示策略信息
        print("\n步骤7: 策略详细信息")
        print("-" * 30)

        strategy_info = strategy.get_strategy_info()
        print("策略基本信息:")
        print(f"  - 策略名称: {strategy_info['strategy_name']}")
        print(f"  - 模式数量: {strategy_info['pattern_count']}")

        expected_performance = strategy.generated_strategy.expected_performance
        print("\n预期性能:")
        print(f"  - 预期收益: {expected_performance.get('expected_return', 0):.2%}")
        print(f"  - 预期成功率: {expected_performance.get('expected_success_rate', 0):.2%}")
        print(f"  - 置信水平: {expected_performance.get('confidence_level', 0):.2%}")

        # 清理测试文件
        if os.path.exists(test_data_file):
            os.remove(test_data_file)
            print(f"\n已清理测试数据文件: {test_data_file}")

        print("\n" + "=" * 60)
        print("测试示例执行完成！")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_optimization_example():
    """运行优化示例"""

    print("\n" + "=" * 60)
    print("优化模式测试示例")
    print("=" * 60)

    try:
        from strategy.historical_buypoint_strategy_system import HistoricalBuyPointStrategySystem

        # 创建更多测试数据用于优化
        buypoint_df = create_sample_buypoint_data(num_stocks=25, num_buypoints_per_stock=5)
        buypoint_data = [(row['stock_code'], row['buypoint_date'])
                        for _, row in buypoint_df.iterrows()]

        print(f"优化测试数据: {len(buypoint_data)} 个买点")

        system = HistoricalBuyPointStrategySystem()

        # 优化配置
        optimization_config = {
            'max_iterations': 2,  # 减少迭代次数便于测试
            'target_performance': 70.0
        }

        print("开始迭代优化...")
        best_strategy, best_report = system.optimize_strategy_iteratively(
            buypoint_data=buypoint_data,
            strategy_name="OptimizedStrategy_Example",
            optimization_config=optimization_config
        )

        print(f"优化完成！")
        print(f"最佳策略: {best_strategy.name}")
        print(f"最佳评分: {best_report.overall_score:.2f}/100")
        print(f"验证结果: {best_report.overall_result.value}")

        # 保存优化结果
        output_dir = "./optimization_output"
        file_paths = system.save_strategy_and_report(
            strategy=best_strategy,
            validation_report=best_report,
            output_dir=output_dir
        )

        print("\n优化结果已保存:")
        for file_type, file_path in file_paths.items():
            print(f"  - {file_type}: {file_path}")

        return True

    except Exception as e:
        print(f"优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_api_usage():
    """演示API使用方式"""

    print("\n" + "=" * 60)
    print("API使用方式演示")
    print("=" * 60)

    # 创建配置文件示例
    config_example = {
        "generation": {
            "pattern_method": "statistical",
            "generation_mode": "conservative",
            "min_pattern_frequency": 3,
            "min_success_rate": 0.7
        },
        "validation": {
            "pool_size": 100,
            "validation_period_days": 15
        }
    }

    config_file = "./api_example_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_example, f, indent=2, ensure_ascii=False)

    print(f"配置文件示例已创建: {config_file}")

    # 创建买点数据示例
    sample_data = [
        ("000001", "2024-01-15"),
        ("000001", "2024-03-10"),
        ("000002", "2024-01-20"),
        ("000002", "2024-02-25"),
        ("600000", "2024-01-18"),
        ("600000", "2024-04-05"),
        ("600036", "2024-02-01"),
        ("600036", "2024-03-15"),
    ]

    buypoint_csv = "./api_example_buypoints.csv"
    df = pd.DataFrame(sample_data, columns=['stock_code', 'buypoint_date'])
    df.to_csv(buypoint_csv, index=False)

    print(f"买点数据示例已创建: {buypoint_csv}")

    print("\n可以使用以下命令运行:")
    print(f"python strategy/historical_buypoint_strategy_system.py \\")
    print(f"    --input {buypoint_csv} \\")
    print(f"    --output ./api_output \\")
    print(f"    --strategy-name 'APIExample' \\")
    print(f"    --config {config_file}")

    print("\n或者使用优化模式:")
    print(f"python strategy/historical_buypoint_strategy_system.py \\")
    print(f"    --input {buypoint_csv} \\")
    print(f"    --output ./api_output \\")
    print(f"    --strategy-name 'OptimizedAPIExample' \\")
    print(f"    --optimize \\")
    print(f"    --config {config_file}")

    # 清理示例文件
    for file_path in [config_file, buypoint_csv]:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    print("历史买点自动策略生成系统 - 完整测试示例")

    # 运行基本示例
    success1 = run_example_workflow()

    # 运行优化示例
    if success1:
        success2 = run_optimization_example()
    else:
        success2 = False

    # 演示API使用
    demonstrate_api_usage()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"基本功能测试: {'通过' if success1 else '失败'}")
    print(f"优化功能测试: {'通过' if success2 else '失败'}")
    print(f"整体测试结果: {'成功' if success1 and success2 else '部分失败'}")

    if success1 and success2:
        print("\n✅ 系统测试全部通过！")
        print("系统已准备好用于生产环境部署。")
    else:
        print("\n⚠️ 部分测试失败，请检查系统配置和依赖。")

    print("\n测试文件位置:")
    print("- 基本测试结果: ./example_output/")
    print("- 优化测试结果: ./optimization_output/")
    print("- 系统文档: ./docs/historical_buypoint_strategy_system_guide.md")
    print("- 配置示例: ./config/historical_buypoint_strategy_config.json")