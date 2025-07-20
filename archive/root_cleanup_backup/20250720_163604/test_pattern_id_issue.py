#!/usr/bin/env python3
"""
测试形态ID规范化问题
"""
import sys
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.kdj import KDJ
from indicators.pattern_registry import Pattern_registry

def test_pattern_id_normalization():
    """测试形态ID规范化"""
    print("=== 形态ID规范化测试 ===")
    
    # 清空注册表
    Pattern_registry.clear_registry()
    
    # 创建KDJ实例
    kdj = KDJ(k_period=9, d_period=3, j_period=3)
    
    # 获取PatternRegistry实例
    registry = Pattern_registry()
    
    print(f"KDJ指标类型: {kdj.get_indicator_type()}")
    
    # 测试规范化方法
    pattern_id = "TEST_PATTERN"
    indicator_id = "KDJ"
    normalized_id = Pattern_registry._normalize_pattern_id(pattern_id, indicator_id)
    print(f"原始ID: {pattern_id}")
    print(f"指标ID: {indicator_id}")
    print(f"规范化ID: {normalized_id}")
    
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
    
    # 查看实际注册的形态
    all_patterns = registry.get_all_patterns()
    print(f"注册的形态数量: {len(all_patterns)}")
    for pattern_id, pattern_info in all_patterns.items():
        print(f"  形态ID: {pattern_id}")
        print(f"  显示名称: {pattern_info.get('display_name', 'N/A')}")
        print(f"  指标ID: {pattern_info.get('indicator_id', 'N/A')}")
    
    # 尝试用不同的ID查找
    test_ids = [
        "TEST_PATTERN",
        "KDJ_TEST_PATTERN", 
        "test_pattern",
        "kdj_test_pattern"
    ]
    
    for test_id in test_ids:
        pattern_info = registry.get_pattern_info(test_id)
        print(f"查找 {test_id}: {'✅ 找到' if pattern_info else '❌ 未找到'}")
    
    return all_patterns

def test_kdj_pattern_registration_detailed():
    """详细测试KDJ形态注册"""
    print("\n=== KDJ形态注册详细测试 ===")
    
    # 清空注册表
    Pattern_registry.clear_registry()
    
    # 创建KDJ实例
    kdj = KDJ(k_period=9, d_period=3, j_period=3)
    
    # 获取PatternRegistry实例
    registry = Pattern_registry()
    
    # 手动调用注册方法
    print("调用 _register_patterns() 前...")
    print(f"形态数量: {len(registry.get_all_patterns())}")
    
    kdj._register_patterns()
    
    print("调用 _register_patterns() 后...")
    all_patterns = registry.get_all_patterns()
    print(f"形态数量: {len(all_patterns)}")
    
    # 详细查看每个形态
    for pattern_id, pattern_info in all_patterns.items():
        print(f"\n形态: {pattern_id}")
        print(f"  显示名称: {pattern_info.get('display_name', 'N/A')}")
        print(f"  指标ID: {pattern_info.get('indicator_id', 'N/A')}")
        print(f"  评分影响: {pattern_info.get('score_impact', 0)}")
        print(f"  形态类型: {pattern_info.get('pattern_type', 'N/A')}")
    
    # 测试查找KDJ形态
    kdj_patterns_to_test = [
        "KDJ_GOLDEN_CROSS",
        "KDJ_DEATH_CROSS", 
        "KDJ_OVERBOUGHT",
        "KDJ_OVERSOLD"
    ]
    
    print(f"\n测试查找KDJ形态:")
    for pattern_id in kdj_patterns_to_test:
        pattern_info = registry.get_pattern_info(pattern_id)
        if pattern_info:
            print(f"✅ {pattern_id}: 评分影响 {pattern_info.get('score_impact', 0)}")
        else:
            print(f"❌ {pattern_id}: 未找到")
    
    return all_patterns

def test_kdj_scoring_with_registered_patterns():
    """测试KDJ评分与已注册形态"""
    print("\n=== KDJ评分与已注册形态测试 ===")
    
    # 使用前面注册的形态
    kdj = KDJ(k_period=9, d_period=3, j_period=3)
    registry = Pattern_registry()
    
    # 创建测试数据
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    
    data = pd.DataFrame({
        'date': dates,
        'high': 100 + np.random.normal(0, 2, 50),
        'low': 100 + np.random.normal(0, 2, 50),
        'close': 100 + np.random.normal(0, 2, 50),
        'volume': np.random.randint(1000, 10000, 50)
    })
    data.set_index('date', inplace=True)
    
    # 计算KDJ并获取形态
    kdj_result = kdj.calculate(data)
    patterns = kdj.get_patterns(data)
    
    print(f"KDJ计算结果列: {list(kdj_result.columns)}")
    print(f"形态检测结果列: {list(patterns.columns)}")
    
    # 检查形态检测结果
    for col in patterns.columns:
        if patterns[col].any():
            print(f"检测到形态 {col}: {patterns[col].sum()} 次")
            # 检查注册表中是否有这个形态
            pattern_info = registry.get_pattern_info(col)
            if pattern_info:
                print(f"  注册表中的评分影响: {pattern_info.get('score_impact', 0)}")
            else:
                print(f"  ⚠️ 注册表中未找到此形态")
    
    # 计算评分
    raw_scores = kdj.calculate_raw_score(data)
    print(f"\n原始评分统计:")
    print(f"  平均值: {raw_scores.mean():.2f}")
    print(f"  标准差: {raw_scores.std():.2f}")
    print(f"  唯一值数量: {len(raw_scores.unique())}")
    
    return raw_scores

if __name__ == "__main__":
    try:
        # 测试形态ID规范化
        patterns1 = test_pattern_id_normalization()
        
        # 测试KDJ形态注册
        patterns2 = test_kdj_pattern_registration_detailed()
        
        # 测试评分
        scores = test_kdj_scoring_with_registered_patterns()
        
        print(f"\n=== 总结 ===")
        print(f"手动注册形态数量: {len(patterns1)}")
        print(f"KDJ形态数量: {len(patterns2)}")
        print(f"评分是否动态: {len(scores.unique()) > 1}")
        
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc() 