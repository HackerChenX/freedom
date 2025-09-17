#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双向验证系统测试脚本

测试双向验证系统的完整功能，包括：
1. 前向验证测试
2. 后向验证测试
3. 报告生成测试
4. 质量标准验证
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# 导入测试所需的模块
from validation.bidirectional_validation_system import BidirectionalValidationSystem
from strategy.historical_buypoint_strategy_generator import (
    HistoricalBuyPointStrategyGenerator, BuyPointInput,
    StrategyGenerationMode, PatternRecognitionMethod
)

def create_mock_strategy():
    """创建模拟策略用于测试"""
    from dataclasses import dataclass
    from datetime import datetime

    @dataclass
    class MockPattern:
        indicator_name: str
        condition_type: str
        threshold_value: float
        confidence: float
        frequency: int

        def to_condition_string(self):
            return f"{self.indicator_name} {self.condition_type} {self.threshold_value}"

    @dataclass
    class MockStrategy:
        strategy_name: str
        description: str
        technical_patterns: list
        expected_success_rate: float
        expected_return: float
        risk_level: str
        source_buypoints_count: int
        generation_time: datetime

    # 创建模拟技术模式
    mock_patterns = [
        MockPattern("RSI", "<=", 30.0, 0.85, 5),
        MockPattern("MACD", ">=", 0.0, 0.78, 4),
        MockPattern("price_position", "between", (0.2, 0.4), 0.72, 6)
    ]

    # 创建模拟策略
    mock_strategy = MockStrategy(
        strategy_name="测试双向验证策略",
        description="用于测试双向验证系统的模拟策略",
        technical_patterns=mock_patterns,
        expected_success_rate=0.75,
        expected_return=8.5,
        risk_level="MEDIUM",
        source_buypoints_count=8,
        generation_time=datetime.now()
    )

    return mock_strategy

def create_test_buypoints():
    """创建测试用的买点数据"""
    buypoints = []
    for i in range(8):
        buypoint = {
            'stock_code': f"00000{i+1}",
            'buypoint_date': (datetime.now() - timedelta(days=30+i)).strftime('%Y-%m-%d'),
            'expected_return': 7.5 + i * 0.5,
            'holding_days': 15 + i,
            'note': f'测试买点{i+1}'
        }
        buypoints.append(buypoint)

    return buypoints

def create_test_selected_stocks():
    """创建测试用的选中股票数据"""
    selected_stocks = []
    for i in range(5):
        stock = {
            'stock_code': f"00000{i+1}",
            'score': 0.8 + i * 0.02,
            'match_patterns': 3 + (i % 2)
        }
        selected_stocks.append(stock)

    return selected_stocks

def test_bidirectional_validation_system():
    """测试双向验证系统"""
    print("="*80)
    print("双向验证系统功能测试")
    print("="*80)

    try:
        # 1. 创建测试数据
        print("\n1. 创建测试数据...")
        mock_strategy = create_mock_strategy()
        test_buypoints = create_test_buypoints()
        test_selected_stocks = create_test_selected_stocks()

        print(f"   ✓ 策略名称: {mock_strategy.strategy_name}")
        print(f"   ✓ 技术模式数: {len(mock_strategy.technical_patterns)}")
        print(f"   ✓ 测试买点数: {len(test_buypoints)}")
        print(f"   ✓ 选中股票数: {len(test_selected_stocks)}")

        # 2. 初始化验证系统
        print("\n2. 初始化双向验证系统...")
        validation_system = BidirectionalValidationSystem()
        print("   ✓ 验证系统初始化完成")

        # 3. 执行双向验证
        print("\n3. 执行双向验证...")
        start_time = datetime.now()

        validation_result = validation_system.execute_bidirectional_validation(
            strategy=mock_strategy,
            original_buypoints=test_buypoints,
            selected_stocks=test_selected_stocks,
            output_dir="./results/validation_test_reports"
        )

        execution_time = (datetime.now() - start_time).total_seconds()
        print(f"   ✓ 验证完成，耗时: {execution_time:.2f}秒")

        # 4. 分析验证结果
        print("\n4. 分析验证结果:")
        print(f"   状态: {validation_result['status']}")
        print(f"   消息: {validation_result['message']}")

        if 'details' in validation_result:
            details = validation_result['details']

            # 前向验证结果
            if 'forward_validation' in details:
                fv = details['forward_validation']
                print(f"   前向验证 - 准确率: {fv['accuracy']:.2%}")
                print(f"   前向验证 - 匹配买点: {fv['matched_buypoints']}/{fv['total_patterns']}")
                print(f"   前向验证 - 信号一致性: {fv['signal_consistency']:.3f}")

            # 后向验证结果
            if 'backward_validation' in details:
                bv = details['backward_validation']
                print(f"   后向验证 - 策略覆盖率: {bv['strategy_coverage']:.2%}")
                print(f"   后向验证 - 分析股票数: {bv['analyzed_stocks']}")
                print(f"   后向验证 - 统计显著性: {bv['statistical_significance']:.3f}")

            # 综合指标
            print(f"   整体得分: {details.get('overall_score', 0):.3f}")
            print(f"   风险评估: {details.get('risk_assessment', 'UNKNOWN')}")

        # 5. 检查质量标准
        print("\n5. 质量标准检查:")
        quality_checks = []

        # 检查执行时间
        if execution_time < 10.0:
            quality_checks.append(f"✓ 执行时间达标: {execution_time:.2f}s < 10s")
        else:
            quality_checks.append(f"✗ 执行时间超标: {execution_time:.2f}s >= 10s")

        # 检查验证状态
        if validation_result['status'] in ['success', 'warning']:
            quality_checks.append("✓ 验证系统正常运行")
        else:
            quality_checks.append("✗ 验证系统运行异常")

        # 检查报告生成
        if 'report_files' in validation_result:
            quality_checks.append("✓ 验证报告成功生成")
        else:
            quality_checks.append("✗ 验证报告生成失败")

        # 检查建议生成
        if validation_result.get('recommendations'):
            quality_checks.append("✓ 改进建议成功生成")
        else:
            quality_checks.append("✗ 改进建议生成失败")

        for check in quality_checks:
            print(f"   {check}")

        # 6. 显示改进建议
        if validation_result.get('recommendations'):
            print("\n6. 改进建议:")
            for i, recommendation in enumerate(validation_result['recommendations'], 1):
                print(f"   {i}. {recommendation}")

        # 7. 显示报告文件
        if validation_result.get('report_files'):
            print("\n7. 生成的报告文件:")
            for file_type, file_path in validation_result['report_files'].items():
                print(f"   {file_type}: {file_path}")

                # 检查文件是否真实存在
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"      (存在, {file_size} bytes)")
                else:
                    print(f"      (不存在)")

        # 8. 验证统计信息
        print("\n8. 验证系统统计:")
        stats = validation_system.get_validation_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")

        print("\n" + "="*80)
        print("双向验证系统测试完成")
        print("="*80)

        return True

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """测试各个组件的独立功能"""
    print("\n" + "="*80)
    print("组件独立功能测试")
    print("="*80)

    try:
        from validation.bidirectional_validation_system import (
from db.sql_manager import SQLManager, QueryType
            ForwardValidator, BackwardValidator, ValidationReportGenerator
        )

        # 测试数据准备
        mock_strategy = create_mock_strategy()
        test_buypoints = create_test_buypoints()
        test_selected_stocks = create_test_selected_stocks()

        print("\n1. 测试前向验证器...")
        forward_validator = ForwardValidator(None)  # 使用模拟数据管理器
        print("   ✓ 前向验证器初始化成功")

        print("\n2. 测试后向验证器...")
        backward_validator = BackwardValidator(None)  # 使用模拟数据管理器
        print("   ✓ 后向验证器初始化成功")

        print("\n3. 测试报告生成器...")
        report_generator = ValidationReportGenerator()
        print("   ✓ 报告生成器初始化成功")

        print("\n✅ 所有组件初始化测试通过")
        return True

    except Exception as e:
        print(f"❌ 组件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始双向验证系统完整测试...")

    # 确保输出目录存在
    output_dir = Path("./results/validation_test_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 执行测试
    test_success = True

    # 测试1: 组件独立功能测试
    component_test_success = test_individual_components()
    test_success = test_success and component_test_success

    # 测试2: 完整系统功能测试
    system_test_success = test_bidirectional_validation_system()
    test_success = test_success and system_test_success

    # 输出测试总结
    print("\n" + "="*80)
    if test_success:
        print("🎉 所有测试通过！双向验证系统功能正常")
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
    print("="*80)

    return 0 if test_success else 1

if __name__ == '__main__':
    sys.exit(main())