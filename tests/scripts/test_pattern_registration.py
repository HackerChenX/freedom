#!/usr/bin/env python3
"""
测试形态注册功能
"""
import sys
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.kdj import KDJ
from indicators.pattern_registry import Pattern_registry

def test_manual_pattern_registration():
    """手动测试形态注册"""
    print("=== 手动形态注册测试 ===")
    
    # 清空注册表
    Pattern_registry.clear_registry()
    
    # 创建KDJ实例
    kdj = KDJ(k_period=9, d_period=3, j_period=3)
    
    # 获取PatternRegistry实例
    registry = Pattern_registry()
    
    print(f"注册前形态数量: {len(registry.get_all_patterns())}")
    
    # 手动注册一个形态
    try:
        kdj.register_pattern_to_registry(
            pattern_id="TEST_PATTERN",
            display_name="测试形态",
            description="这是一个测试形态",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0,
            polarity="POSITIVE"
        )
        print("✅ 手动注册成功")
    except Exception as e:
        print(f"❌ 手动注册失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"注册后形态数量: {len(registry.get_all_patterns())}")
    
    # 检查形态是否注册成功
    test_pattern = registry.get_pattern_info("TEST_PATTERN")
    if test_pattern:
        print(f"✅ 形态注册成功: {test_pattern}")
    else:
        print("❌ 形态注册失败")
    
    return registry

def test_kdj_automatic_registration():
    """测试KDJ自动注册"""
    print("\n=== KDJ自动注册测试 ===")
    
    # 清空注册表
    Pattern_registry.clear_registry()
    
    # 获取PatternRegistry实例
    registry = Pattern_registry()
    print(f"清空后形态数量: {len(registry.get_all_patterns())}")
    
    # 创建KDJ实例（应该自动注册形态）
    kdj = KDJ(k_period=9, d_period=3, j_period=3)
    
    print(f"创建KDJ后形态数量: {len(registry.get_all_patterns())}")
    
    # 手动调用注册方法
    try:
        kdj._register_patterns()
        print("✅ 手动调用_register_patterns成功")
    except Exception as e:
        print(f"❌ 手动调用_register_patterns失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"手动注册后形态数量: {len(registry.get_all_patterns())}")
    
    # 检查KDJ形态
    kdj_patterns = [p for p in registry.get_all_patterns().keys() if 'KDJ' in p.upper()]
    print(f"KDJ形态: {kdj_patterns}")
    
    return registry

def test_pattern_registry_methods():
    """测试PatternRegistry方法"""
    print("\n=== PatternRegistry方法测试 ===")
    
    # 清空注册表
    Pattern_registry.clear_registry()
    
    # 获取PatternRegistry实例
    registry = Pattern_registry()
    
    # 直接使用registry.register方法
    try:
        from indicators.pattern_registry import Pattern_type, Pattern_strength, Pattern_polarity
        
        registry.register(
            pattern_id="DIRECT_TEST",
            display_name="直接测试形态",
            description="直接使用registry.register方法",
            indicator_id="TEST",
            pattern_type=Pattern_type.BULLISH,
            default_strength=Pattern_strength.STRONG,
            score_impact=20.0,
            polarity=Pattern_polarity.POSITIVE
        )
        print("✅ 直接注册成功")
    except Exception as e:
        print(f"❌ 直接注册失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"直接注册后形态数量: {len(registry.get_all_patterns())}")
    
    # 检查形态
    direct_pattern = registry.get_pattern_info("DIRECT_TEST")
    if direct_pattern:
        print(f"✅ 直接注册的形态: {direct_pattern}")
    else:
        print("❌ 直接注册的形态未找到")
    
    return registry

if __name__ == "__main__":
    try:
        # 测试手动注册
        registry1 = test_manual_pattern_registration()
        
        # 测试KDJ自动注册
        registry2 = test_kdj_automatic_registration()
        
        # 测试PatternRegistry方法
        registry3 = test_pattern_registry_methods()
        
        print("\n=== 总结 ===")
        print(f"最终形态数量: {len(registry3.get_all_patterns())}")
        all_patterns = registry3.get_all_patterns()
        for pattern_id, pattern_info in all_patterns.items():
            print(f"  {pattern_id}: {pattern_info.get('display_name', 'N/A')}")
        
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc() 