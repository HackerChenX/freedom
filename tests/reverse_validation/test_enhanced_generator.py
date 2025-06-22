#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试增强版数据生成器

验证生成的数据是否真正具备目标技术形态特征
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from enhanced_pattern_generator import EnhancedPatternGenerator
from technical_indicators import TechnicalIndicators, PatternRecognizer


def test_enhanced_rsi_patterns():
    """测试增强版RSI形态生成"""
    print("测试增强版RSI形态生成...")

    generator = EnhancedPatternGenerator()
    indicators = TechnicalIndicators()
    recognizer = PatternRecognizer()

    # 测试RSI超买形态
    print("\n1. 测试RSI超买形态")
    overbought_data = generator.generate_rsi_overbought_data()
    rsi = indicators.calculate_rsi(overbought_data)
    patterns = recognizer.detect_rsi_patterns(overbought_data)

    print(f"   数据点数: {len(overbought_data)}")
    print(f"   最终RSI值: {rsi.iloc[-1]:.2f}")
    print(f"   RSI超买检测: {patterns.get('RSI_OVERBOUGHT', False)}")
    print(f"   价格变化: {overbought_data['close'].iloc[0]:.2f} -> {overbought_data['close'].iloc[-1]:.2f}")

    # 测试RSI超卖形态
    print("\n2. 测试RSI超卖形态")
    oversold_data = generator.generate_rsi_oversold_data()
    rsi = indicators.calculate_rsi(oversold_data)
    patterns = recognizer.detect_rsi_patterns(oversold_data)

    print(f"   数据点数: {len(oversold_data)}")
    print(f"   最终RSI值: {rsi.iloc[-1]:.2f}")
    print(f"   RSI超卖检测: {patterns.get('RSI_OVERSOLD', False)}")
    print(f"   价格变化: {oversold_data['close'].iloc[0]:.2f} -> {oversold_data['close'].iloc[-1]:.2f}")

    # 测试RSI金叉形态
    print("\n3. 测试RSI金叉形态")
    golden_cross_data = generator.generate_rsi_golden_cross_data()
    rsi = indicators.calculate_rsi(golden_cross_data)
    patterns = recognizer.detect_rsi_patterns(golden_cross_data)

    print(f"   数据点数: {len(golden_cross_data)}")
    print(f"   最终RSI值: {rsi.iloc[-1]:.2f}")
    print(f"   RSI金叉检测: {patterns.get('RSI_GOLDEN_CROSS', False)}")
    print(f"   价格变化: {golden_cross_data['close'].iloc[0]:.2f} -> {golden_cross_data['close'].iloc[-1]:.2f}")

    # 测试RSI死叉形态
    print("\n4. 测试RSI死叉形态")
    death_cross_data = generator.generate_rsi_death_cross_data()
    rsi = indicators.calculate_rsi(death_cross_data)
    patterns = recognizer.detect_rsi_patterns(death_cross_data)

    print(f"   数据点数: {len(death_cross_data)}")
    print(f"   最终RSI值: {rsi.iloc[-1]:.2f}")
    print(f"   RSI死叉检测: {patterns.get('RSI_DEATH_CROSS', False)}")
    print(f"   价格变化: {death_cross_data['close'].iloc[0]:.2f} -> {death_cross_data['close'].iloc[-1]:.2f}")

    # 测试RSI背离形态
    print("\n5. 测试RSI背离形态")
    divergence_data = generator.generate_rsi_divergence_data()
    rsi = indicators.calculate_rsi(divergence_data)
    patterns = recognizer.detect_rsi_patterns(divergence_data)

    print(f"   数据点数: {len(divergence_data)}")
    print(f"   最终RSI值: {rsi.iloc[-1]:.2f}")
    print(f"   RSI背离检测: {patterns.get('RSI_DIVERGENCE', False)}")
    print(f"   价格变化: {divergence_data['close'].iloc[0]:.2f} -> {divergence_data['close'].iloc[-1]:.2f}")

    # 重新计算正确的成功率
    success_count = 0
    test_data = [
        (overbought_data, 'RSI_OVERBOUGHT'),
        (oversold_data, 'RSI_OVERSOLD'),
        (golden_cross_data, 'RSI_GOLDEN_CROSS'),
        (death_cross_data, 'RSI_DEATH_CROSS'),
        (divergence_data, 'RSI_DIVERGENCE')
    ]

    for data, expected_pattern in test_data:
        detected_patterns = recognizer.detect_rsi_patterns(data)
        if detected_patterns.get(expected_pattern, False):
            success_count += 1

    success_rate = success_count / len(test_data)

    print(f"\n增强版RSI形态生成测试结果:")
    print(f"成功识别: {success_count}/{len(test_data)}")
    print(f"成功率: {success_rate:.2%}")

    return success_rate >= 0.8  # 80%以上认为成功


def main():
    """主测试函数"""
    print("=" * 60)
    print("增强版数据生成器测试")
    print("=" * 60)

    tests = [
        test_enhanced_rsi_patterns,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} 通过")
            else:
                print(f"❌ {test_func.__name__} 失败")
        except Exception as e:
            print(f"❌ {test_func.__name__} 异常: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！增强版数据生成器工作正常。")
        return 0
    else:
        print("⚠️ 部分测试失败，需要进一步优化。")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)