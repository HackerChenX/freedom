#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PatternRegistry使用示例
"""

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength, PatternInfo

def pattern_registry_example():
    """PatternRegistry使用示例"""
    print("=== PatternRegistry使用示例 ===")
    
    # 获取单例实例
    registry = PatternRegistry()
    
    # 注册多个形态
    patterns = [
        PatternInfo("BOLL_SQUEEZE", "布林带挤压", "BOLL", PatternType.NEUTRAL, score_impact=0),
        PatternInfo("BOLL_BREAKOUT_UP", "布林带向上突破", "BOLL", PatternType.BULLISH, score_impact=15),
        PatternInfo("BOLL_BREAKOUT_DOWN", "布林带向下突破", "BOLL", PatternType.BEARISH, score_impact=-15)
    ]
    
    registry.register_patterns_batch(patterns)
    
    # 查询指标的所有形态
    boll_patterns = registry.get_patterns_by_indicator("BOLL")
    print(f"BOLL指标的所有形态: {boll_patterns}")
    
    # 获取特定形态的详细信息
    breakout_pattern = registry.get_pattern("BOLL_BREAKOUT_UP")
    print(f"向上突破形态详情: {breakout_pattern}")
    
    # 计算组合评分影响
    combined_impact = PatternRegistry.calculate_combined_score_impact(["BOLL_BREAKOUT_UP", "BOLL_SQUEEZE"])
    print(f"组合评分影响: {combined_impact}")
    
    # 测试全局控制覆盖行为
    print("\n测试覆盖行为:")
    # 默认不允许覆盖
    try:
        registry.register(
            pattern_id="BOLL_SQUEEZE",
            display_name="布林带挤压-更新",
            indicator_id="BOLL",
            pattern_type=PatternType.NEUTRAL
        )
    except Exception as e:
        print(f"尝试覆盖失败: {e}")
    
    # 设置全局允许覆盖
    PatternRegistry.set_allow_override(True)
    registry.register(
        pattern_id="BOLL_SQUEEZE",
        display_name="布林带挤压-已更新",
        indicator_id="BOLL",
        pattern_type=PatternType.NEUTRAL
    )
    
    # 验证更新后的形态
    updated_pattern = registry.get_pattern("BOLL_SQUEEZE")
    print(f"更新后的形态: {updated_pattern}")
    
    print("=== 示例结束 ===")

if __name__ == "__main__":
    pattern_registry_example() 