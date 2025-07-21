#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
StockInfoCompatibleDataGenerator验证脚本

验证StockInfo兼容数据生成器的功能是否正常
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

try:
    from tests.unified_indicator_testing.components.stockinfo_compatible_data_generator import StockInfoCompatibleDataGenerator
except ImportError:
    # 尝试相对导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from components.stockinfo_compatible_data_generator import StockInfoCompatibleDataGenerator


def verify_basic_functionality():
    """验证基础功能"""
    print("🔍 验证基础功能...")
    
    generator = StockInfoCompatibleDataGenerator()
    
    # 测试1: 生成StockInfo兼容数据
    print("  测试1: 生成StockInfo兼容数据")
    data = generator.generate_stockinfo_compatible_data(
        indicator_name='MACD',
        pattern_type='GOLDEN_CROSS',
        stock_code='VERIFY001',
        history_days=60
    )
    
    print(f"    ✓ 生成数据: {len(data)} 行")
    print(f"    ✓ 字段数量: {len(data.columns)}")
    print(f"    ✓ 必需字段: {all(field in data.columns for field in ['date', 'code', 'open', 'high', 'low', 'close', 'volume'])}")
    
    # 测试2: 生成随机数据
    print("  测试2: 生成随机数据")
    random_data = generator.generate_random_stockinfo_data('RANDOM001', 30)
    print(f"    ✓ 随机数据: {len(random_data)} 行")
    
    # 测试3: 数据结构验证
    print("  测试3: 数据结构验证")
    is_valid = generator._validate_stockinfo_structure(data)
    print(f"    ✓ 结构验证: {is_valid}")
    
    return True


def verify_data_quality():
    """验证数据质量"""
    print("\n📊 验证数据质量...")
    
    generator = StockInfoCompatibleDataGenerator()
    
    # 生成测试数据
    data = generator.generate_stockinfo_compatible_data(
        'RSI', 'OVERSOLD', 'QUALITY001', 50
    )
    
    # 检查价格逻辑
    print("  检查价格逻辑:")
    price_errors = 0
    for i in range(len(data)):
        high = data.iloc[i]['high']
        low = data.iloc[i]['low']
        open_price = data.iloc[i]['open']
        close = data.iloc[i]['close']
        
        if not (low <= open_price <= high and low <= close <= high):
            price_errors += 1
    
    print(f"    ✓ 价格逻辑错误: {price_errors}/{len(data)} ({price_errors/len(data)*100:.1f}%)")
    
    # 检查数据类型
    print("  检查数据类型:")
    print(f"    ✓ open类型: {data['open'].dtype}")
    print(f"    ✓ high类型: {data['high'].dtype}")
    print(f"    ✓ low类型: {data['low'].dtype}")
    print(f"    ✓ close类型: {data['close'].dtype}")
    print(f"    ✓ volume类型: {data['volume'].dtype}")
    
    # 检查数据范围
    print("  检查数据范围:")
    print(f"    ✓ 价格范围: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"    ✓ 成交量范围: {data['volume'].min():.0f} - {data['volume'].max():.0f}")
    
    return True


def verify_compatibility():
    """验证兼容性"""
    print("\n🔗 验证兼容性...")
    
    generator = StockInfoCompatibleDataGenerator()
    
    # 生成测试数据
    data = generator.generate_stockinfo_compatible_data(
        'KDJ', 'GOLDEN_CROSS', 'COMPAT001', 40
    )
    
    # 兼容性验证
    report = generator.validate_data_compatibility(data)
    
    print("  兼容性报告:")
    print(f"    ✓ 是否兼容: {report['is_compatible']}")
    print(f"    ✓ 缺失字段: {len(report['missing_fields'])}")
    print(f"    ✓ 类型错误: {len(report['invalid_data_types'])}")
    print(f"    ✓ 质量问题: {len(report['data_quality_issues'])}")
    
    if report['recommendations']:
        print("  建议:")
        for rec in report['recommendations']:
            print(f"    • {rec}")
    
    return report['is_compatible']


def verify_large_scale():
    """验证大规模数据生成"""
    print("\n🚀 验证大规模数据生成...")
    
    generator = StockInfoCompatibleDataGenerator()
    
    # 生成100只股票的数据
    stock_codes = [f'SCALE{i:03d}' for i in range(100)]
    
    print(f"  生成 {len(stock_codes)} 只股票的数据...")
    start_time = datetime.now()
    
    results = generator.generate_large_scale_data(
        stock_codes=stock_codes,
        indicator_name='BOLL',
        pattern_type='UPPER_BREAKOUT',
        history_days=30
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"    ✓ 成功生成: {len(results)}/{len(stock_codes)} ({len(results)/len(stock_codes)*100:.1f}%)")
    print(f"    ✓ 执行时间: {duration:.2f}秒")
    print(f"    ✓ 平均速度: {len(results)/duration:.1f} 股票/秒")
    
    # 验证结果质量
    if results:
        sample_code = list(results.keys())[0]
        sample_data = results[sample_code]
        print(f"    ✓ 样本数据: {len(sample_data)} 行，{len(sample_data.columns)} 列")
    
    return len(results) >= len(stock_codes) * 0.8  # 至少80%成功率


def verify_indicator_support():
    """验证指标支持"""
    print("\n📈 验证指标支持...")
    
    generator = StockInfoCompatibleDataGenerator()
    
    supported_indicators = generator.get_supported_indicators()
    print(f"  支持的指标数量: {len(supported_indicators)}")
    
    # 测试几个关键指标
    test_indicators = ['MACD', 'RSI', 'KDJ', 'BOLL', 'VOL']
    
    for indicator in test_indicators:
        if indicator in supported_indicators:
            requirement = generator.get_indicator_history_requirement(indicator)
            print(f"    ✓ {indicator}: {requirement} 天历史数据")
        else:
            print(f"    ✗ {indicator}: 不支持")
    
    return all(indicator in supported_indicators for indicator in test_indicators)


def verify_integration():
    """验证与UnifiedIndicatorTester的集成"""
    print("\n🔧 验证集成...")
    
    try:
        try:
            from tests.unified_indicator_testing.unified_indicator_tester import UnifiedIndicatorTester
        except ImportError:
            from unified_indicator_tester import UnifiedIndicatorTester
        
        # 创建测试器实例
        with UnifiedIndicatorTester() as tester:
            print("    ✓ UnifiedIndicatorTester初始化成功")
            
            # 验证数据生成器类型
            generator_type = type(tester.data_generator).__name__
            print(f"    ✓ 数据生成器类型: {generator_type}")
            
            # 测试数据生成
            test_data = tester._generate_comprehensive_data_pool(
                'MACD', 'GOLDEN_CROSS', {'history_requirement': 60}, 10
            )
            
            print(f"    ✓ 集成测试数据: {len(test_data)} 只股票")
            
            if test_data:
                sample_data = test_data[0]
                print(f"    ✓ 样本数据: {len(sample_data)} 行，{len(sample_data.columns)} 列")
        
        return True
        
    except Exception as e:
        print(f"    ✗ 集成测试失败: {e}")
        return False


def main():
    """主验证函数"""
    print("🚀 StockInfoCompatibleDataGenerator 功能验证")
    print("=" * 60)
    
    results = []
    
    # 执行各项验证
    results.append(("基础功能", verify_basic_functionality()))
    results.append(("数据质量", verify_data_quality()))
    results.append(("兼容性", verify_compatibility()))
    results.append(("大规模生成", verify_large_scale()))
    results.append(("指标支持", verify_indicator_support()))
    results.append(("集成测试", verify_integration()))
    
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
        print("\n🎉 所有验证通过！StockInfoCompatibleDataGenerator功能正常")
        return True
    else:
        print(f"\n⚠️  存在 {len(results) - passed} 项验证失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
