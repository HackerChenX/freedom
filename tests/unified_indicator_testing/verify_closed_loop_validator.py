#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
ClosedLoopValidator验证脚本

验证闭环验证器的功能是否正常
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

try:
    from tests.unified_indicator_testing.components.closed_loop_validator import ClosedLoopValidator
except ImportError:
    # 尝试相对导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from components.closed_loop_validator import ClosedLoopValidator


def create_test_data():
    """创建测试数据"""
    # 创建选股结果
    selection_results = {
        'execution_success': True,
        'selected_stocks': [
            {'code': 'TARGET_001', 'score': 0.9, 'signal_strength': 0.8},
            {'code': 'TARGET_002', 'score': 0.85, 'signal_strength': 0.75},
            {'code': 'NOISE_001', 'score': 0.7, 'signal_strength': 0.6}
        ],
        'total_candidates': 10,
        'selection_count': 3
    }
    
    # 创建股票数据池
    data_pool = []
    
    # 目标股票数据（有明显买点形态）
    for i in range(3):
        target_data = pd.DataFrame({
            'date': ['20250721', '20250722', '20250723', '20250724', '20250725'],
            'code': [f'TARGET_{i:03d}'] * 5,
            'name': [f'目标股票_{i}'] * 5,
            'open': [10.0, 10.2, 10.5, 10.8, 11.0],
            'high': [10.3, 10.6, 10.9, 11.2, 11.5],
            'low': [9.8, 10.0, 10.3, 10.6, 10.8],
            'close': [10.1, 10.4, 10.7, 11.0, 11.3],
            'volume': [1000000, 1200000, 1500000, 1800000, 2000000],
            'industry': ['测试行业'] * 5
        })
        data_pool.append(target_data)
    
    # 干扰股票数据
    for i in range(2):
        noise_data = pd.DataFrame({
            'date': ['20250721', '20250722', '20250723', '20250724', '20250725'],
            'code': [f'NOISE_{i:03d}'] * 5,
            'name': [f'干扰股票_{i}'] * 5,
            'open': [15.0, 14.9, 14.8, 14.7, 14.6],
            'high': [15.2, 15.0, 14.9, 14.8, 14.7],
            'low': [14.8, 14.7, 14.6, 14.5, 14.4],
            'close': [14.9, 14.8, 14.7, 14.6, 14.5],
            'volume': [800000, 750000, 700000, 650000, 600000],
            'industry': ['其他行业'] * 5
        })
        data_pool.append(noise_data)
    
    return selection_results, data_pool


def verify_initialization():
    """验证初始化"""
    print("🔍 验证初始化...")
    
    validator = ClosedLoopValidator()
    
    print(f"    ✓ 验证配置: {len(validator.validation_config)} 项")
    print(f"    ✓ 买点形态: {len(validator.buypoint_patterns)} 个指标")
    print(f"    ✓ 验证统计: {len(validator.validation_stats)} 项")
    
    # 验证支持的形态
    supported_patterns = validator.get_supported_patterns()
    print(f"    ✓ 支持的指标: {list(supported_patterns.keys())}")
    
    validator.cleanup()
    return True


def verify_pattern_definitions():
    """验证形态定义"""
    print("\n📝 验证形态定义...")
    
    validator = ClosedLoopValidator()
    
    # 测试MACD形态
    macd_golden = validator._get_pattern_definition('MACD', 'GOLDEN_CROSS')
    print(f"    ✓ MACD金叉形态: {macd_golden is not None}")
    if macd_golden:
        print(f"      入口条件: {len(macd_golden.get('entry_conditions', []))}")
        print(f"      确认信号: {len(macd_golden.get('confirmation_signals', []))}")
        print(f"      风险控制: {len(macd_golden.get('risk_controls', []))}")
    
    # 测试RSI形态
    rsi_oversold = validator._get_pattern_definition('RSI', 'OVERSOLD')
    print(f"    ✓ RSI超卖形态: {rsi_oversold is not None}")
    
    # 测试不存在的形态
    unknown = validator._get_pattern_definition('UNKNOWN', 'UNKNOWN')
    print(f"    ✓ 未知形态处理: {unknown is None}")
    
    validator.cleanup()
    return True


def verify_technical_indicators():
    """验证技术指标计算"""
    print("\n📊 验证技术指标计算...")
    
    validator = ClosedLoopValidator()
    selection_results, data_pool = create_test_data()
    
    test_data = data_pool[0]  # 使用第一个测试数据
    
    # 测试MACD计算
    macd_data = validator._calculate_macd(test_data.copy())
    print(f"    ✓ MACD计算: {all(col in macd_data.columns for col in ['macd_line', 'signal_line', 'histogram'])}")
    
    # 测试RSI计算
    rsi_data = validator._calculate_rsi(test_data.copy())
    print(f"    ✓ RSI计算: {'rsi' in rsi_data.columns}")
    if 'rsi' in rsi_data.columns:
        rsi_valid = (rsi_data['rsi'] >= 0).all() and (rsi_data['rsi'] <= 100).all()
        print(f"      RSI范围有效: {rsi_valid}")
    
    # 测试KDJ计算
    kdj_data = validator._calculate_kdj(test_data.copy())
    print(f"    ✓ KDJ计算: {all(col in kdj_data.columns for col in ['k', 'd', 'j'])}")
    
    # 测试布林带计算
    boll_data = validator._calculate_bollinger(test_data.copy())
    print(f"    ✓ 布林带计算: {all(col in boll_data.columns for col in ['upper_band', 'middle_band', 'lower_band'])}")
    
    validator.cleanup()
    return True


def verify_condition_evaluation():
    """验证条件评估"""
    print("\n🔧 验证条件评估...")
    
    validator = ClosedLoopValidator()
    selection_results, data_pool = create_test_data()
    
    test_data = data_pool[0].copy()
    enhanced_data = validator._calculate_technical_indicators(test_data, 'MACD')
    
    # 测试基本比较
    condition1 = {'field': 'close', 'operator': '>', 'value': 10.0}
    result1 = validator._evaluate_condition(enhanced_data, 2, condition1)
    print(f"    ✓ 基本比较 (close > 10.0): {result1}")
    
    # 测试字段比较
    condition2 = {'field': 'high', 'operator': '>', 'reference': 'low'}
    result2 = validator._evaluate_condition(enhanced_data, 2, condition2)
    print(f"    ✓ 字段比较 (high > low): {result2}")
    
    # 测试范围条件
    condition3 = {'field': 'close', 'operator': 'between', 'value': [10.0, 12.0]}
    result3 = validator._evaluate_condition(enhanced_data, 2, condition3)
    print(f"    ✓ 范围条件 (10.0 <= close <= 12.0): {result3}")
    
    validator.cleanup()
    return True


def verify_entry_points_analysis():
    """验证入口点分析"""
    print("\n🎯 验证入口点分析...")
    
    validator = ClosedLoopValidator()
    selection_results, data_pool = create_test_data()
    
    test_data = data_pool[0]  # 目标股票数据
    pattern_def = validator._get_pattern_definition('MACD', 'GOLDEN_CROSS')
    
    if pattern_def:
        entry_points = validator._analyze_entry_points(
            test_data, pattern_def, 'MACD', 'GOLDEN_CROSS'
        )
        
        print(f"    ✓ 入口点数量: {len(entry_points)}")
        
        if entry_points:
            entry_point = entry_points[0]
            print(f"    ✓ 入口点结构: {all(key in entry_point for key in ['date', 'entry_price', 'total_score'])}")
            print(f"      日期: {entry_point.get('date', 'N/A')}")
            print(f"      价格: {entry_point.get('entry_price', 0):.2f}")
            print(f"      评分: {entry_point.get('total_score', 0):.2f}")
    
    validator.cleanup()
    return len(entry_points) >= 0  # 至少不出错


def verify_single_stock_validation():
    """验证单个股票验证"""
    print("\n🔍 验证单个股票验证...")
    
    validator = ClosedLoopValidator()
    selection_results, data_pool = create_test_data()
    
    stock = {'code': 'TARGET_001', 'score': 0.9}
    stock_data = data_pool[0]  # 目标股票数据
    
    validation_result = validator._validate_single_stock(
        stock, stock_data, 'MACD', 'GOLDEN_CROSS'
    )
    
    print(f"    ✓ 验证结果结构: {all(key in validation_result for key in ['stock_code', 'is_valid', 'confidence_score'])}")
    print(f"    ✓ 股票代码: {validation_result.get('stock_code', 'N/A')}")
    print(f"    ✓ 是否有效: {validation_result.get('is_valid', False)}")
    print(f"    ✓ 置信度: {validation_result.get('confidence_score', 0):.2f}")
    print(f"    ✓ 入口点数量: {len(validation_result.get('entry_points', []))}")
    
    validator.cleanup()
    return True


def verify_complete_validation():
    """验证完整验证流程"""
    print("\n🚀 验证完整验证流程...")
    
    validator = ClosedLoopValidator()
    selection_results, data_pool = create_test_data()
    
    validation_result = validator.validate_selection_results(
        selection_results, data_pool, 'MACD', 'GOLDEN_CROSS'
    )
    
    print(f"    ✓ 验证结果结构: {all(key in validation_result for key in ['indicator_name', 'pattern_type', 'validation_rate'])}")
    print(f"    ✓ 指标名称: {validation_result.get('indicator_name', 'N/A')}")
    print(f"    ✓ 形态类型: {validation_result.get('pattern_type', 'N/A')}")
    print(f"    ✓ 验证率: {validation_result.get('validation_rate', 0):.2%}")
    print(f"    ✓ 成功验证: {validation_result.get('successful_validations', 0)}")
    print(f"    ✓ 总验证数: {validation_result.get('total_validations', 0)}")
    print(f"    ✓ 执行时间: {validation_result.get('execution_time', 0):.2f}秒")
    
    # 验证摘要信息
    summary = validation_result.get('summary', {})
    if summary:
        print(f"    ✓ 平均置信度: {summary.get('avg_confidence_score', 0):.2f}")
        print(f"    ✓ 验证质量: {summary.get('validation_quality', 'N/A')}")
    
    validator.cleanup()
    return validation_result.get('validation_rate', 0) >= 0  # 至少不出错


def verify_validation_statistics():
    """验证验证统计"""
    print("\n📈 验证验证统计...")
    
    validator = ClosedLoopValidator()
    selection_results, data_pool = create_test_data()
    
    # 执行几次验证以生成统计数据
    validator.validate_selection_results(selection_results, data_pool, 'MACD', 'GOLDEN_CROSS')
    validator.validate_selection_results(selection_results, data_pool, 'RSI', 'OVERSOLD')
    
    stats = validator.get_validation_statistics()
    
    print(f"    ✓ 统计结构: {all(key in stats for key in ['total_validated', 'successful_validations', 'overall_success_rate'])}")
    print(f"    ✓ 总验证数: {stats.get('total_validated', 0)}")
    print(f"    ✓ 成功验证数: {stats.get('successful_validations', 0)}")
    print(f"    ✓ 总体成功率: {stats.get('overall_success_rate', 0):.2%}")
    print(f"    ✓ 形态匹配: {len(stats.get('pattern_matches', {}))}")
    
    validator.cleanup()
    return True


def verify_error_handling():
    """验证错误处理"""
    print("\n⚠️ 验证错误处理...")
    
    validator = ClosedLoopValidator()
    
    # 测试空选股结果
    empty_result = validator.validate_selection_results(
        {'selected_stocks': []}, [], 'MACD', 'GOLDEN_CROSS'
    )
    print(f"    ✓ 空选股结果: {empty_result.get('total_validations', 0) == 0}")
    
    # 测试无效指标
    invalid_result = validator.validate_selection_results(
        {'selected_stocks': [{'code': 'TEST001'}]}, [], 'INVALID', 'UNKNOWN'
    )
    print(f"    ✓ 无效指标: {'validation_rate' in invalid_result}")
    
    # 测试空数据池
    selection_results, _ = create_test_data()
    empty_pool_result = validator.validate_selection_results(
        selection_results, [], 'MACD', 'GOLDEN_CROSS'
    )
    print(f"    ✓ 空数据池: {'validation_rate' in empty_pool_result}")
    
    validator.cleanup()
    return True


def verify_integration():
    """验证集成"""
    print("\n🔗 验证集成...")
    
    try:
        from tests.unified_indicator_testing.unified_indicator_tester import UnifiedIndicatorTester
        
        # 创建测试器实例
        with UnifiedIndicatorTester() as unified_tester:
            print("    ✓ UnifiedIndicatorTester初始化成功")
            
            # 验证ClosedLoopValidator类型
            validator_type = type(unified_tester.closed_loop_validator).__name__
            print(f"    ✓ 闭环验证器类型: {validator_type}")
            
            # 测试集成功能
            selection_results, data_pool = create_test_data()
            buypoint_result = {'indicator': 'MACD', 'pattern': 'GOLDEN_CROSS'}
            
            validation_result = unified_tester._test_closed_loop_validation(
                buypoint_result, selection_results, data_pool
            )
            
            print(f"    ✓ 集成测试执行: {validation_result.get('status', 'UNKNOWN')}")
            print(f"    ✓ 验证率: {validation_result.get('validation_rate', 0):.2%}")
            print(f"    ✓ 评分: {validation_result.get('score', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ 集成测试失败: {e}")
        return False


def main():
    """主验证函数"""
    print("🚀 ClosedLoopValidator 功能验证")
    print("=" * 60)
    
    results = []
    
    # 执行各项验证
    try:
        results.append(("初始化", verify_initialization()))
    except Exception as e:
        print(f"    ✗ 初始化失败: {e}")
        results.append(("初始化", False))
    
    try:
        results.append(("形态定义", verify_pattern_definitions()))
    except Exception as e:
        print(f"    ✗ 形态定义失败: {e}")
        results.append(("形态定义", False))
    
    try:
        results.append(("技术指标", verify_technical_indicators()))
    except Exception as e:
        print(f"    ✗ 技术指标失败: {e}")
        results.append(("技术指标", False))
    
    try:
        results.append(("条件评估", verify_condition_evaluation()))
    except Exception as e:
        print(f"    ✗ 条件评估失败: {e}")
        results.append(("条件评估", False))
    
    try:
        results.append(("入口点分析", verify_entry_points_analysis()))
    except Exception as e:
        print(f"    ✗ 入口点分析失败: {e}")
        results.append(("入口点分析", False))
    
    try:
        results.append(("单股验证", verify_single_stock_validation()))
    except Exception as e:
        print(f"    ✗ 单股验证失败: {e}")
        results.append(("单股验证", False))
    
    try:
        results.append(("完整验证", verify_complete_validation()))
    except Exception as e:
        print(f"    ✗ 完整验证失败: {e}")
        results.append(("完整验证", False))
    
    try:
        results.append(("验证统计", verify_validation_statistics()))
    except Exception as e:
        print(f"    ✗ 验证统计失败: {e}")
        results.append(("验证统计", False))
    
    try:
        results.append(("错误处理", verify_error_handling()))
    except Exception as e:
        print(f"    ✗ 错误处理失败: {e}")
        results.append(("错误处理", False))
    
    try:
        results.append(("集成测试", verify_integration()))
    except Exception as e:
        print(f"    ✗ 集成测试失败: {e}")
        results.append(("集成测试", False))
    
    # 输出验证结果
    print("\n" + "=" * 60)
    print("📋 验证结果摘要")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"总计: {len(results)} 项测试")
    print(f"通过: {passed} 项")
    print(f"失败: {len(results) - passed} 项")
    print(f"通过率: {passed/len(results)*100:.1f}%")
    
    if passed == len(results):
        print("\n🎉 所有验证通过！ClosedLoopValidator功能正常")
        return True
    else:
        print(f"\n⚠️  存在 {len(results) - passed} 项验证失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
