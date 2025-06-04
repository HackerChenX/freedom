#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试PatternRegistry功能
"""

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength, PatternInfo

def test_pattern_registry():
    """测试PatternRegistry的基本功能"""
    print("测试PatternRegistry功能...")
    
    # 获取单例实例
    registry = PatternRegistry()
    
    # 测试注册单个形态
    print("\n1. 测试注册单个形态")
    registry.register(
        pattern_id="TEST_PATTERN",
        display_name="测试形态",
        indicator_id="TEST",
        pattern_type=PatternType.BULLISH,
        default_strength=PatternStrength.STRONG
    )
    
    # 查询形态
    pattern = registry.get_pattern("TEST_PATTERN")
    print(f"注册的形态: {pattern}")
    
    # 测试按指标查询
    print("\n2. 测试按指标查询形态")
    patterns = registry.get_patterns_by_indicator("TEST")
    print(f"TEST指标的形态: {patterns}")
    
    # 测试批量注册
    print("\n3. 测试批量注册形态")
    patterns_batch = [
        PatternInfo(
            pattern_id="TEST_PATTERN_1",
            display_name="测试形态1",
            indicator_id="TEST",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=10
        ),
        PatternInfo(
            pattern_id="TEST_PATTERN_2",
            display_name="测试形态2",
            indicator_id="TEST",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-10
        )
    ]
    
    registry.register_patterns_batch(patterns_batch)
    
    # 查询批量注册的形态
    for pattern_id in ["TEST_PATTERN_1", "TEST_PATTERN_2"]:
        pattern = registry.get_pattern(pattern_id)
        print(f"批量注册的形态 {pattern_id}: {pattern}")
    
    # 再次查询指标下的所有形态
    patterns = registry.get_patterns_by_indicator("TEST")
    print(f"TEST指标的所有形态: {patterns}")
    
    # 测试计算组合评分影响
    print("\n4. 测试计算组合评分影响")
    # 确保在计算之前这些形态已经被注册并设置了正确的评分影响值
    patterns_with_impact = [
        PatternInfo("TEST_IMPACT_1", "测试影响1", "TEST", PatternType.BULLISH, score_impact=8.5),
        PatternInfo("TEST_IMPACT_2", "测试影响2", "TEST", PatternType.BEARISH, score_impact=-5.5)
    ]
    registry.register_patterns_batch(patterns_with_impact)
    
    # 计算组合评分影响
    combined_impact = PatternRegistry.calculate_combined_score_impact(["TEST_IMPACT_1", "TEST_IMPACT_2"])
    print(f"组合评分影响: {combined_impact}")
    
    # 验证结果是否符合预期
    expected = 3.0  # 8.5 + (-5.5) = 3.0
    if abs(combined_impact - expected) < 0.001:
        print("✓ 评分影响计算正确!")
    else:
        print(f"✗ 评分影响计算错误，期望 {expected}，实际得到 {combined_impact}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_pattern_registry() 