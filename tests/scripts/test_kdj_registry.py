#!/usr/bin/env python3
"""
测试KDJ形态注册问题
"""
import sys
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

import pandas as pd
import numpy as np
from indicators.kdj import KDJ
from indicators.pattern_registry import Pattern_registry

def test_kdj_pattern_registration():
    """测试KDJ形态注册"""
    print("=== KDJ形态注册测试 ===")
    
    # 创建KDJ实例
    kdj = KDJ(k_period=9, d_period=3, j_period=3)
    
    # 获取PatternRegistry实例
    registry = Pattern_registry()
    
    # 检查KDJ形态是否已注册
    print(f"KDJ指标类型: {kdj.get_indicator_type()}")
    
    # 获取所有注册的形态
    all_patterns = registry.get_all_patterns()
    print(f"总共注册的形态数量: {len(all_patterns)}")
    
    # 查找KDJ相关形态
    kdj_patterns = [p for p in all_patterns.keys() if 'KDJ' in p.upper()]
    print(f"KDJ相关形态: {kdj_patterns}")
    
    # 检查具体的KDJ形态
    test_patterns = [
        'KDJ_GOLDEN_CROSS',
        'KDJ_DEATH_CROSS',
        'KDJ_OVERBOUGHT',
        'KDJ_OVERSOLD',
        'KDJ_BOTTOM_DIVERGENCE',
        'KDJ_TOP_DIVERGENCE'
    ]
    
    for pattern in test_patterns:
        pattern_info = registry.get_pattern_info(pattern)
        print(f"形态 {pattern}: {pattern_info is not None}")
        if pattern_info:
            print(f"  - 显示名称: {pattern_info.get('display_name', 'N/A')}")
            print(f"  - 评分影响: {pattern_info.get('score_impact', 0)}")
    
    return kdj_patterns

def test_kdj_pattern_detection():
    """测试KDJ形态检测"""
    print("\n=== KDJ形态检测测试 ===")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    
    # 创建有趋势的数据
    base_price = 100
    trend = np.linspace(0, 10, 50)  # 上升趋势
    noise = np.random.normal(0, 1, 50)
    
    data = pd.DataFrame({
        'date': dates,
        'high': base_price + trend + noise + np.abs(noise),
        'low': base_price + trend + noise - np.abs(noise),
        'close': base_price + trend + noise,
        'volume': np.random.randint(1000, 10000, 50)
    })
    data.set_index('date', inplace=True)
    
    # 创建KDJ实例
    kdj = KDJ(k_period=9, d_period=3, j_period=3)
    
    # 计算KDJ
    result = kdj.calculate(data)
    print(f"KDJ计算结果列: {list(result.columns)}")
    
    # 检测形态
    try:
        patterns = kdj.get_patterns(data)
        print(f"检测到的形态类型: {type(patterns)}")
        if isinstance(patterns, pd.DataFrame):
            print(f"形态DataFrame形状: {patterns.shape}")
            print(f"形态DataFrame列: {list(patterns.columns)}")
            # 检查是否有形态被检测到
            pattern_cols = [col for col in patterns.columns if 'pattern' in col.lower()]
            for col in pattern_cols:
                if patterns[col].any():
                    print(f"检测到形态 {col}: {patterns[col].sum()} 次")
        elif isinstance(patterns, list):
            print(f"检测到的形态列表: {patterns}")
    except Exception as e:
        print(f"形态检测出错: {e}")
    
    return result

def test_kdj_scoring_with_patterns():
    """测试KDJ评分与形态的关系"""
    print("\n=== KDJ评分与形态关系测试 ===")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    
    # 创建有明显金叉死叉的数据
    base_price = 100
    data = pd.DataFrame({
        'date': dates,
        'high': base_price + np.random.normal(0, 2, 50),
        'low': base_price + np.random.normal(0, 2, 50),
        'close': base_price + np.random.normal(0, 2, 50),
        'volume': np.random.randint(1000, 10000, 50)
    })
    data.set_index('date', inplace=True)
    
    # 创建KDJ实例
    kdj = KDJ(k_period=9, d_period=3, j_period=3)
    
    # 计算评分
    score_result = kdj.calculate_score(data)
    print(f"评分结果: {score_result}")
    
    # 计算原始评分
    raw_scores = kdj.calculate_raw_score(data)
    print(f"原始评分统计:")
    print(f"  平均值: {raw_scores.mean():.2f}")
    print(f"  标准差: {raw_scores.std():.2f}")
    print(f"  最小值: {raw_scores.min():.2f}")
    print(f"  最大值: {raw_scores.max():.2f}")
    print(f"  唯一值数量: {len(raw_scores.unique())}")
    
    # 检查是否是固定值
    if len(raw_scores.unique()) == 1:
        print(f"⚠️  评分是固定值: {raw_scores.iloc[0]}")
    else:
        print("✅ 评分是动态的")
    
    return raw_scores

if __name__ == "__main__":
    try:
        # 测试形态注册
        kdj_patterns = test_kdj_pattern_registration()
        
        # 测试形态检测
        kdj_result = test_kdj_pattern_detection()
        
        # 测试评分
        raw_scores = test_kdj_scoring_with_patterns()
        
        print("\n=== 总结 ===")
        print(f"KDJ形态注册数量: {len(kdj_patterns)}")
        print(f"KDJ计算是否成功: {kdj_result is not None}")
        print(f"KDJ评分是否动态: {len(raw_scores.unique()) > 1}")
        
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc() 