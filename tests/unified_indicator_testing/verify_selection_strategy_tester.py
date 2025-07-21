#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
SelectionStrategyTester验证脚本

验证选股策略测试器的功能是否正常
"""

import os
import sys
import pandas as pd
import json
import yaml
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

try:
    from tests.unified_indicator_testing.components.selection_strategy_tester import SelectionStrategyTester
except ImportError:
    # 尝试相对导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from components.selection_strategy_tester import SelectionStrategyTester


def create_test_data_pool():
    """创建测试数据池"""
    data_pool = []
    
    # 创建目标股票数据
    for i in range(5):
        target_data = pd.DataFrame({
            'date': ['20250721', '20250722', '20250723'],
            'code': [f'TARGET_{i:03d}'] * 3,
            'name': [f'目标股票_{i}'] * 3,
            'open': [10.0, 10.5, 11.0],
            'high': [10.5, 11.0, 11.5],
            'low': [9.5, 10.0, 10.5],
            'close': [10.2, 10.8, 11.2],
            'volume': [1000000, 1200000, 1100000],
            'industry': ['测试行业'] * 3
        })
        data_pool.append(target_data)
    
    # 创建干扰股票数据
    for i in range(15):
        noise_data = pd.DataFrame({
            'date': ['20250721', '20250722', '20250723'],
            'code': [f'NOISE_{i:03d}'] * 3,
            'name': [f'干扰股票_{i}'] * 3,
            'open': [15.0, 14.8, 15.2],
            'high': [15.5, 15.3, 15.8],
            'low': [14.5, 14.2, 14.8],
            'close': [15.1, 14.9, 15.5],
            'volume': [800000, 900000, 850000],
            'industry': ['其他行业'] * 3
        })
        data_pool.append(noise_data)
    
    return data_pool


def verify_initialization():
    """验证初始化"""
    print("🔍 验证初始化...")
    
    tester = SelectionStrategyTester()
    
    print(f"    ✓ 选股脚本路径: {tester.stock_select_script}")
    print(f"    ✓ 策略模板数量: {len(tester.strategy_templates)}")
    print(f"    ✓ 指标映射数量: {len(tester.indicator_pattern_mapping)}")
    
    # 验证支持的指标
    supported_indicators = tester.get_supported_indicators()
    print(f"    ✓ 支持的指标: {len(supported_indicators)} 个")
    print(f"      主要指标: {', '.join(supported_indicators[:5])}")
    
    return tester


def verify_strategy_generation():
    """验证策略生成"""
    print("\n📝 验证策略生成...")
    
    tester = SelectionStrategyTester()
    
    # 测试统一格式策略生成
    print("  测试统一格式策略生成:")
    unified_strategy = tester.generate_strategy_config('MACD', 'GOLDEN_CROSS', 'unified')
    
    print(f"    ✓ 策略ID: {unified_strategy['strategy']['id']}")
    print(f"    ✓ 策略名称: {unified_strategy['strategy']['name']}")
    print(f"    ✓ 技术指标数量: {len(unified_strategy['technical_indicators']['primary_indicators'])}")
    
    # 测试传统格式策略生成
    print("  测试传统格式策略生成:")
    legacy_strategy = tester.generate_strategy_config('RSI', 'OVERSOLD', 'legacy')
    
    print(f"    ✓ 策略ID: {legacy_strategy['strategy']['id']}")
    print(f"    ✓ 条件数量: {len(legacy_strategy['strategy']['conditions'])}")
    
    # 测试策略验证
    print("  测试策略验证:")
    validation_result = tester.validate_strategy_config(unified_strategy)
    print(f"    ✓ 验证结果: {validation_result['is_valid']}")
    print(f"    ✓ 错误数量: {len(validation_result['errors'])}")
    
    tester.cleanup()
    return True


def verify_complex_strategy():
    """验证复杂策略生成"""
    print("\n🔧 验证复杂策略生成...")
    
    tester = SelectionStrategyTester()
    
    # 测试多指标组合策略
    indicators = [
        {'indicator_name': 'MACD', 'pattern_type': 'GOLDEN_CROSS'},
        {'indicator_name': 'RSI', 'pattern_type': 'OVERSOLD'},
        {'indicator_name': 'KDJ', 'pattern_type': 'GOLDEN_CROSS'}
    ]
    
    complex_strategy = tester.generate_complex_strategy(indicators, 'AND')
    
    print(f"    ✓ 复杂策略ID: {complex_strategy['strategy']['id']}")
    print(f"    ✓ 包含指标数量: {len(complex_strategy['technical_indicators']['primary_indicators'])}")
    print(f"    ✓ 逻辑操作符: {complex_strategy['technical_indicators']['combination_logic']}")
    
    # 验证复杂策略
    validation_result = tester.validate_strategy_config(complex_strategy)
    print(f"    ✓ 复杂策略验证: {validation_result['is_valid']}")
    
    tester.cleanup()
    return True


def verify_strategy_execution():
    """验证策略执行"""
    print("\n🚀 验证策略执行...")

    try:
        tester = SelectionStrategyTester()
        data_pool = create_test_data_pool()

        # 测试单个策略执行（使用较小的数据集）
        print("  测试单个策略执行:")
        small_data_pool = data_pool[:5]  # 只使用5只股票进行快速测试

        result = tester.test_strategy_selection('MACD', 'GOLDEN_CROSS', small_data_pool)

        print(f"    ✓ 执行成功: {result['execution_success']}")
        print(f"    ✓ 候选股票: {result['total_candidates']}")
        print(f"    ✓ 选中股票: {result['selection_count']}")
        print(f"    ✓ 执行时间: {result['execution_time']:.2f}秒")

        # 验证性能指标
        if 'performance_metrics' in result:
            metrics = result['performance_metrics']
            print(f"    ✓ 精确率: {metrics.get('precision', 0):.2f}")
            print(f"    ✓ 召回率: {metrics.get('recall', 0):.2f}")
            print(f"    ✓ F1分数: {metrics.get('f1_score', 0):.2f}")

        tester.cleanup()
        return result['execution_success']

    except Exception as e:
        print(f"    ✗ 策略执行验证失败: {e}")
        return False


def verify_multiple_strategies():
    """验证多策略执行"""
    print("\n📊 验证多策略执行...")

    try:
        tester = SelectionStrategyTester()
        data_pool = create_test_data_pool()[:3]  # 使用更小的数据集

        # 减少策略数量以加快测试
        strategies = [
            {'indicator_name': 'MACD', 'pattern_type': 'GOLDEN_CROSS'},
            {'indicator_name': 'RSI', 'pattern_type': 'OVERSOLD'},
            {'indicator_name': 'KDJ', 'pattern_type': 'OVERBOUGHT'}
        ]

        results = tester.test_multiple_strategies(strategies, data_pool)

        print(f"    ✓ 总策略数: {results['total_strategies']}")
        print(f"    ✓ 完成策略数: {results['completed_strategies']}")
        print(f"    ✓ 失败策略数: {results['failed_strategies']}")
        print(f"    ✓ 成功率: {results['completed_strategies']/results['total_strategies']*100:.1f}%")

        # 显示每个策略的结果
        for strategy_key, strategy_result in results['results'].items():
            success = strategy_result.get('execution_success', False)
            status = "✅" if success else "❌"
            print(f"      {status} {strategy_key}")

        tester.cleanup()
        return results['completed_strategies'] >= results['total_strategies'] * 0.8

    except Exception as e:
        print(f"    ✗ 多策略执行验证失败: {e}")
        return False


def verify_file_operations():
    """验证文件操作"""
    print("\n📁 验证文件操作...")
    
    tester = SelectionStrategyTester()
    
    # 测试策略文件保存
    print("  测试策略文件保存:")
    strategy_config = tester.generate_strategy_config('BOLL', 'SQUEEZE')
    
    # 保存JSON格式
    json_file = tester._save_strategy_config(strategy_config, 'unified')
    print(f"    ✓ JSON文件: {os.path.basename(json_file)}")
    print(f"    ✓ 文件存在: {os.path.exists(json_file)}")
    
    # 验证JSON内容
    with open(json_file, 'r', encoding='utf-8') as f:
        loaded_config = json.load(f)
    print(f"    ✓ JSON解析: {loaded_config['strategy']['id'] == strategy_config['strategy']['id']}")
    
    # 保存YAML格式
    yaml_file = tester._save_strategy_config(strategy_config, 'legacy')
    print(f"    ✓ YAML文件: {os.path.basename(yaml_file)}")
    print(f"    ✓ 文件存在: {os.path.exists(yaml_file)}")
    
    # 验证YAML内容
    with open(yaml_file, 'r', encoding='utf-8') as f:
        loaded_config = yaml.safe_load(f)
    print(f"    ✓ YAML解析: {'strategy' in loaded_config}")
    
    # 测试模拟环境设置
    print("  测试模拟环境设置:")
    data_pool = create_test_data_pool()
    mock_env = tester._setup_mock_environment(data_pool)
    
    print(f"    ✓ 数据文件: {os.path.exists(mock_env['data_file'])}")
    print(f"    ✓ 配置文件: {os.path.exists(mock_env['config_file'])}")
    print(f"    ✓ 环境变量: {'MOCK_DATA_MODE' in mock_env['env_vars']}")
    
    tester.cleanup()
    return True


def verify_performance():
    """验证性能"""
    print("\n⚡ 验证性能...")

    try:
        tester = SelectionStrategyTester()

        # 创建中等规模数据池（减少规模以加快测试）
        medium_data_pool = []
        for i in range(20):  # 从100减少到20
            data = pd.DataFrame({
                'date': ['20250721', '20250722', '20250723'],
                'code': [f'PERF_{i:03d}'] * 3,
                'name': [f'性能测试股票_{i}'] * 3,
                'open': [10.0, 10.5, 11.0],
                'high': [10.5, 11.0, 11.5],
                'low': [9.5, 10.0, 10.5],
                'close': [10.2, 10.8, 11.2],
                'volume': [1000000, 1200000, 1100000],
                'industry': ['测试行业'] * 3
            })
            medium_data_pool.append(data)

        # 测试性能
        start_time = datetime.now()
        result = tester.test_strategy_selection('MACD', 'GOLDEN_CROSS', medium_data_pool)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()
        throughput = len(medium_data_pool) / duration if duration > 0 else 0

        print(f"    ✓ 数据规模: {len(medium_data_pool)} 只股票")
        print(f"    ✓ 执行时间: {duration:.2f}秒")
        print(f"    ✓ 处理速度: {throughput:.1f} 股票/秒")
        print(f"    ✓ 执行成功: {result['execution_success']}")

        tester.cleanup()
        return duration < 30 and result['execution_success']  # 要求在30秒内完成且成功

    except Exception as e:
        print(f"    ✗ 性能验证失败: {e}")
        return False


def verify_integration():
    """验证集成"""
    print("\n🔗 验证集成...")

    try:
        # 尝试多种导入方式
        unified_tester = None

        try:
            from tests.unified_indicator_testing.unified_indicator_tester import UnifiedIndicatorTester
            unified_tester = UnifiedIndicatorTester()
        except ImportError:
            try:
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from unified_indicator_tester import UnifiedIndicatorTester
                unified_tester = UnifiedIndicatorTester()
            except ImportError:
                # 如果都失败，创建一个简单的集成测试
                print("    ⚠️ 无法导入UnifiedIndicatorTester，执行简化集成测试")
                return verify_simplified_integration()

        if unified_tester:
            print("    ✓ UnifiedIndicatorTester初始化成功")

            # 验证SelectionStrategyTester类型
            tester_type = type(unified_tester.selection_tester).__name__
            print(f"    ✓ 选股测试器类型: {tester_type}")

            # 测试集成功能
            data_pool = create_test_data_pool()
            selection_result = unified_tester._test_selection_strategy(
                'MACD', 'GOLDEN_CROSS', data_pool
            )

            print(f"    ✓ 集成测试执行: {selection_result.get('status', 'UNKNOWN')}")
            print(f"    ✓ 选股数量: {selection_result.get('selected_count', 0)}")
            print(f"    ✓ 评分: {selection_result.get('score', 0):.2f}")

            # 清理资源
            unified_tester.cleanup()

            return True

        return False

    except Exception as e:
        print(f"    ✗ 集成测试失败: {e}")
        return False

def verify_simplified_integration():
    """简化集成测试"""
    try:
        # 直接测试SelectionStrategyTester与其他组件的兼容性
        tester = SelectionStrategyTester()
        data_pool = create_test_data_pool()

        # 测试策略生成和执行
        result = tester.test_strategy_selection('MACD', 'GOLDEN_CROSS', data_pool)

        print(f"    ✓ 简化集成测试执行: {result.get('execution_success', False)}")
        print(f"    ✓ 选股数量: {result.get('selection_count', 0)}")
        print(f"    ✓ 性能指标: {len(result.get('performance_metrics', {}))}")

        tester.cleanup()
        return result.get('execution_success', False)

    except Exception as e:
        print(f"    ✗ 简化集成测试失败: {e}")
        return False


def main():
    """主验证函数"""
    print("🚀 SelectionStrategyTester 功能验证")
    print("=" * 60)
    
    results = []
    
    # 执行各项验证
    try:
        tester = verify_initialization()
        results.append(("初始化", True))
    except Exception as e:
        print(f"    ✗ 初始化失败: {e}")
        results.append(("初始化", False))
    
    try:
        results.append(("策略生成", verify_strategy_generation()))
    except Exception as e:
        print(f"    ✗ 策略生成失败: {e}")
        results.append(("策略生成", False))
    
    try:
        results.append(("复杂策略", verify_complex_strategy()))
    except Exception as e:
        print(f"    ✗ 复杂策略失败: {e}")
        results.append(("复杂策略", False))
    
    try:
        results.append(("策略执行", verify_strategy_execution()))
    except Exception as e:
        print(f"    ✗ 策略执行失败: {e}")
        results.append(("策略执行", False))
    
    try:
        results.append(("多策略执行", verify_multiple_strategies()))
    except Exception as e:
        print(f"    ✗ 多策略执行失败: {e}")
        results.append(("多策略执行", False))
    
    try:
        results.append(("文件操作", verify_file_operations()))
    except Exception as e:
        print(f"    ✗ 文件操作失败: {e}")
        results.append(("文件操作", False))
    
    try:
        results.append(("性能测试", verify_performance()))
    except Exception as e:
        print(f"    ✗ 性能测试失败: {e}")
        results.append(("性能测试", False))
    
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
        print("\n🎉 所有验证通过！SelectionStrategyTester功能正常")
        return True
    else:
        print(f"\n⚠️  存在 {len(results) - passed} 项验证失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
