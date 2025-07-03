#!/usr/bin/env python3
"""
测试KDJ指标中get_pattern方法的调用
"""
import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.kdj import KDJ
from indicators.pattern_registry import PatternRegistry

def test_kdj_get_pattern():
    """测试KDJ指标中get_pattern方法的调用"""
    print("=== 测试KDJ指标中get_pattern方法的调用 ===")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # 创建有趋势的价格数据
    base_price = 100
    trend = np.linspace(0, 20, 100)
    noise = np.random.normal(0, 2, 100)
    
    data = pd.DataFrame({
        'date': dates,
        'open': base_price + trend + noise,
        'high': base_price + trend + noise + np.abs(np.random.normal(0, 1, 100)),
        'low': base_price + trend + noise - np.abs(np.random.normal(0, 1, 100)),
        'close': base_price + trend + noise + np.random.normal(0, 0.5, 100),
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # 确保high >= low, close在high和low之间
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    data['close'] = np.clip(data['close'], data['low'], data['high'])
    
    # 创建KDJ指标实例
    kdj = KDJ(n=9, m1=3, m2=3)
    
    # 确保形态已注册
    kdj._register_patterns()
    
    # 获取形态检测结果
    patterns = kdj.get_patterns(data)
    print(f"检测到的形态列: {list(patterns.columns)}")
    
    # 测试PatternRegistry的get_pattern方法
    registry = PatternRegistry()
    
    print("\n=== 测试PatternRegistry中的形态信息 ===")
    all_patterns = registry.get_all_pattern_ids()
    print(f"注册表中的所有形态: {len(all_patterns)}")
    
    # 查找KDJ相关形态
    kdj_patterns = [p for p in all_patterns if 'KDJ' in p]
    print(f"KDJ相关形态: {kdj_patterns}")
    
    # 测试get_pattern方法
    print("\n=== 测试get_pattern方法 ===")
    for pattern_col in patterns.columns:
        print(f"\n测试形态: {pattern_col}")
        
        # 使用get_pattern方法
        pattern_info = registry.get_pattern(pattern_col)
        print(f"  get_pattern结果: {pattern_info}")
        
        # 使用get_pattern_info方法进行对比
        pattern_info2 = registry.get_pattern_info(pattern_col)
        print(f"  get_pattern_info结果: {pattern_info2}")
        
        # 检查是否相同
        if pattern_info == pattern_info2:
            print("  ✅ 两种方法结果一致")
        else:
            print("  ❌ 两种方法结果不一致")
    
    # 测试calculate_raw_score方法
    print("\n=== 测试calculate_raw_score方法 ===")
    try:
        score = kdj.calculate_raw_score(data)
        print(f"评分计算成功:")
        print(f"  评分范围: {score.min():.2f} - {score.max():.2f}")
        print(f"  评分平均值: {score.mean():.2f}")
        print(f"  评分标准差: {score.std():.2f}")
        print(f"  唯一值数量: {score.nunique()}")
        
        # 检查是否固定50分
        if score.nunique() == 1 and score.iloc[0] == 50.0:
            print("  ❌ 评分固定为50分")
        else:
            print("  ✅ 评分动态变化")
            
    except Exception as e:
        print(f"  ❌ 评分计算失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_kdj_get_pattern() 