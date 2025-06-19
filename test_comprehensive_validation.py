#!/usr/bin/env python3
"""
综合验证测试 - 验证形态重构系统的完整性和向后兼容性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger
from indicators.pattern_registry import PatternRegistry
from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer
from indicators.kdj import KDJ
from indicators.rsi import RSI
from indicators.trix import TRIX
from indicators.roc import ROC
from indicators.cmo import CMO
from indicators.vol import VOL
from indicators.atr import ATR
from indicators.kc import KC
from indicators.mfi import MFI
from indicators.vortex import Vortex

# 设置日志级别
import logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

def test_pattern_registry_completeness():
    """测试PatternRegistry的完整性"""
    print("\n=== 测试PatternRegistry完整性 ===")
    
    # 创建指标实例并初始化
    indicators = [
        ('KDJ', KDJ()),
        ('RSI', RSI()),
        ('TRIX', TRIX()),
        ('ROC', ROC()),
        ('CMO', CMO()),
        ('VOL', VOL()),
        ('ATR', ATR()),
        ('KC', KC()),
        ('MFI', MFI()),
        ('Vortex', Vortex())
    ]
    
    registry = PatternRegistry()
    total_patterns = 0
    
    for name, indicator in indicators:
        try:
            indicator.initialize()
            patterns = registry.get_patterns_by_indicator(name)
            pattern_count = len(patterns)
            total_patterns += pattern_count
            print(f"{name}: {pattern_count} 个形态")
            
            # 显示前3个形态作为示例
            if patterns:
                if isinstance(patterns, dict):
                    pattern_items = list(patterns.items())[:3]
                    for pattern_id, info in pattern_items:
                        print(f"  - {pattern_id}: {info.get('display_name', 'N/A')}")
                elif isinstance(patterns, list):
                    pattern_items = patterns[:3]
                    for info in pattern_items:
                        if isinstance(info, dict):
                            print(f"  - {info.get('pattern_id', 'N/A')}: {info.get('display_name', 'N/A')}")
                        else:
                            print(f"  - {info}")
                    
        except Exception as e:
            print(f"❌ {name} 初始化失败: {e}")
    
    print(f"\n总计: {total_patterns} 个形态已注册到PatternRegistry")
    return total_patterns > 50  # 期望至少有50个形态

def test_pattern_retrieval_consistency():
    """测试形态检索的一致性"""
    print("\n=== 测试形态检索一致性 ===")

    # 创建分析器实例
    analyzer = BuyPointBatchAnalyzer()

    test_cases = [
        ('KDJ', 'KDJ_GOLDEN_CROSS'),
        ('RSI', 'RSI_OVERSOLD'),
        ('TRIX', 'TRIX_ABOVE_ZERO'),
        ('ROC', 'ROC_ABOVE_ZERO'),
        ('CMO', 'CMO_RISING'),
        ('VOL', 'VOL_FALLING'),
        ('ATR', 'ATR_UPWARD_BREAKOUT'),
        ('KC', 'KC_ABOVE_MIDDLE'),
        ('MFI', 'MFI_RISING'),
        ('Vortex', 'VORTEX_BULLISH_CROSS')
    ]

    success_count = 0

    for indicator_name, pattern_id in test_cases:
        try:
            pattern_info = analyzer.get_precise_pattern_info(indicator_name, pattern_id, "")

            if pattern_info and pattern_info.get('name'):
                print(f"✅ {indicator_name} - {pattern_id}: {pattern_info['name']}")
                success_count += 1
            else:
                print(f"❌ {indicator_name} - {pattern_id}: 未找到形态信息")

        except Exception as e:
            print(f"❌ {indicator_name} - {pattern_id}: 检索失败 - {e}")

    success_rate = success_count / len(test_cases)
    print(f"\n形态检索成功率: {success_rate:.1%} ({success_count}/{len(test_cases)})")
    return success_rate >= 0.8  # 期望至少80%成功率

def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n=== 测试向后兼容性 ===")

    # 创建分析器实例
    analyzer = BuyPointBatchAnalyzer()

    # 测试一些可能仍在centralized mapping中的形态
    legacy_patterns = [
        ('MA', 'bullish_arrangement'),
        ('EMA', 'EMA_BULLISH_ARRANGEMENT'),
        ('SAR', 'SAR_UPTREND'),
        ('ADX', 'ADX_UPTREND'),
        ('PSY', 'PSY_ABOVE_50'),
        ('OBV', 'OBV_RISING'),
        ('VR', 'VR_NORMAL'),
        ('VOSC', 'VOSC_RISING')
    ]

    success_count = 0

    for indicator_name, pattern_id in legacy_patterns:
        try:
            pattern_info = analyzer.get_precise_pattern_info(indicator_name, pattern_id, "")

            if pattern_info and pattern_info.get('name'):
                print(f"✅ {indicator_name} - {pattern_id}: {pattern_info['name']}")
                success_count += 1
            else:
                print(f"⚠️ {indicator_name} - {pattern_id}: 使用默认形态信息")
                success_count += 0.5  # 部分成功

        except Exception as e:
            print(f"❌ {indicator_name} - {pattern_id}: 兼容性测试失败 - {e}")

    success_rate = success_count / len(legacy_patterns)
    print(f"\n向后兼容性测试成功率: {success_rate:.1%}")
    return success_rate >= 0.7  # 期望至少70%兼容

def test_chinese_naming_standards():
    """测试中文命名标准"""
    print("\n=== 测试中文命名标准 ===")
    
    registry = PatternRegistry()
    
    # 检查一些关键指标的中文命名
    test_indicators = ['KDJ', 'RSI', 'TRIX', 'ROC', 'CMO']
    
    chinese_pattern_count = 0
    total_pattern_count = 0
    
    for indicator_name in test_indicators:
        patterns = registry.get_patterns_by_indicator(indicator_name)

        if isinstance(patterns, dict):
            for pattern_id, info in patterns.items():
                total_pattern_count += 1
                display_name = info.get('display_name', '')

                # 检查是否包含中文字符
                if any('\u4e00' <= char <= '\u9fff' for char in display_name):
                    chinese_pattern_count += 1
                    print(f"✅ {indicator_name} - {pattern_id}: {display_name}")
                else:
                    print(f"⚠️ {indicator_name} - {pattern_id}: {display_name} (非中文)")
        elif isinstance(patterns, list):
            for info in patterns:
                if isinstance(info, dict):
                    total_pattern_count += 1
                    display_name = info.get('display_name', '')
                    pattern_id = info.get('pattern_id', 'N/A')

                    # 检查是否包含中文字符
                    if any('\u4e00' <= char <= '\u9fff' for char in display_name):
                        chinese_pattern_count += 1
                        print(f"✅ {indicator_name} - {pattern_id}: {display_name}")
                    else:
                        print(f"⚠️ {indicator_name} - {pattern_id}: {display_name} (非中文)")
    
    if total_pattern_count > 0:
        chinese_rate = chinese_pattern_count / total_pattern_count
        print(f"\n中文命名覆盖率: {chinese_rate:.1%} ({chinese_pattern_count}/{total_pattern_count})")
        return chinese_rate >= 0.8  # 期望至少80%使用中文命名
    else:
        print("⚠️ 未找到任何形态进行中文命名测试")
        return False

def main():
    """主测试函数"""
    print("开始综合验证测试...")
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("PatternRegistry完整性", test_pattern_registry_completeness()))
    test_results.append(("形态检索一致性", test_pattern_retrieval_consistency()))
    test_results.append(("向后兼容性", test_backward_compatibility()))
    test_results.append(("中文命名标准", test_chinese_naming_standards()))
    
    # 汇总测试结果
    print("\n" + "="*50)
    print("综合验证测试结果汇总")
    print("="*50)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    overall_success_rate = passed_tests / total_tests
    print(f"\n总体测试通过率: {overall_success_rate:.1%} ({passed_tests}/{total_tests})")
    
    if overall_success_rate >= 0.75:
        print("\n🎉 综合验证测试通过！形态重构系统运行良好。")
        return True
    else:
        print("\n⚠️ 综合验证测试未完全通过，需要进一步优化。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
