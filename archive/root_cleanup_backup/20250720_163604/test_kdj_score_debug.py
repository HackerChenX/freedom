#!/usr/bin/env python3
"""
详细调试KDJ评分计算过程
"""
import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.kdj import KDJ
from indicators.pattern_registry import Pattern_registry

def debug_kdj_scoring():
    """详细调试KDJ评分计算过程"""
    print("=== 详细调试KDJ评分计算过程 ===")
    
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
    kdj._register_patterns()
    
    # 首先计算KDJ指标
    kdj_result = kdj.calculate(data)
    print(f"KDJ计算结果列: {list(kdj_result.columns)}")
    print(f"KDJ数据样本:")
    print(kdj_result[['K', 'D', 'J']].tail())
    
    # 获取形态检测结果
    patterns = kdj.get_patterns(data)
    print(f"\n形态检测结果:")
    print(f"形态列: {list(patterns.columns)}")
    
    # 检查形态是否被检测到
    for col in patterns.columns:
        pattern_count = patterns[col].sum()
        print(f"  {col}: {pattern_count} 次检测到")
    
    # 手动模拟calculate_raw_score的计算过程
    print(f"\n=== 手动模拟评分计算过程 ===")
    
    # 1. 位置分计算
    k_values = kdj_result['K']
    d_values = kdj_result['D']
    j_values = kdj_result['J']
    
    position_score = pd.Series(50.0, index=data.index)
    
    # 超买区域(K>80)降分
    overbought_mask = k_values > 80
    position_score[overbought_mask] -= 10
    print(f"超买区域数量: {overbought_mask.sum()}")
    
    # 超卖区域(K<20)加分
    oversold_mask = k_values < 20
    position_score[oversold_mask] += 10
    print(f"超卖区域数量: {oversold_mask.sum()}")
    
    print(f"位置分范围: {position_score.min():.2f} - {position_score.max():.2f}")
    
    # 2. 趋势分计算
    trend_score = pd.Series(0.0, index=data.index)
    
    # K线上升趋势
    k_trend = k_values.diff()
    rising_k = k_trend > 0
    trend_score[rising_k] += 5
    print(f"K线上升数量: {rising_k.sum()}")
    
    # D线上升趋势
    d_trend = d_values.diff()
    rising_d = d_trend > 0
    trend_score[rising_d] += 3
    print(f"D线上升数量: {rising_d.sum()}")
    
    print(f"趋势分范围: {trend_score.min():.2f} - {trend_score.max():.2f}")
    
    # 3. 交叉分计算
    cross_score = pd.Series(0.0, index=data.index)
    
    # 金叉检测
    k_above_d = k_values > d_values
    k_above_d_prev = k_above_d.shift(1)
    # 处理NaN值
    k_above_d_prev = k_above_d_prev.fillna(False)
    golden_cross = (~k_above_d_prev) & k_above_d
    cross_score[golden_cross] += 15
    print(f"金叉数量: {golden_cross.sum()}")
    
    # 死叉检测
    death_cross = k_above_d_prev & (~k_above_d)
    cross_score[death_cross] -= 15
    print(f"死叉数量: {death_cross.sum()}")
    
    print(f"交叉分范围: {cross_score.min():.2f} - {cross_score.max():.2f}")
    
    # 4. J值影响分计算
    j_impact = pd.Series(0.0, index=data.index)
    
    # J值极值影响
    j_extreme_high = j_values > 100
    j_impact[j_extreme_high] -= 5
    print(f"J值极高数量: {j_extreme_high.sum()}")
    
    j_extreme_low = j_values < 0
    j_impact[j_extreme_low] += 5
    print(f"J值极低数量: {j_extreme_low.sum()}")
    
    print(f"J值影响分范围: {j_impact.min():.2f} - {j_impact.max():.2f}")
    
    # 5. 计算基础评分
    raw_score = position_score + trend_score + cross_score + j_impact
    print(f"基础评分范围: {raw_score.min():.2f} - {raw_score.max():.2f}")
    
    # 6. 形态调整分计算
    print(f"\n=== 形态调整分计算 ===")
    pattern_adjustment = pd.Series(0.0, index=data.index)
    
    registry = Pattern_registry()
    
    # 遍历所有检测到的形态
    for pattern_col in patterns.columns:
        pattern_info = registry.get_pattern(pattern_col)
        if pattern_info and 'score_impact' in pattern_info:
            score_impact = pattern_info['score_impact']
            print(f"形态 {pattern_col}: score_impact = {score_impact}")
            
            # 检查在哪些时间点检测到了形态
            pattern_detected = patterns[pattern_col]
            detected_indices = pattern_detected[pattern_detected].index
            print(f"  检测到的时间点数量: {len(detected_indices)}")
            
            if len(detected_indices) > 0:
                print(f"  检测到的时间点: {detected_indices.tolist()[:5]}...")  # 只显示前5个
                
                # 应用形态调整
                for idx in detected_indices:
                    adjustment = np.clip(score_impact, -15, 15)
                    pattern_adjustment.at[idx] += adjustment
                    print(f"    在 {idx} 应用调整: {adjustment}")
    
    print(f"形态调整分范围: {pattern_adjustment.min():.2f} - {pattern_adjustment.max():.2f}")
    print(f"形态调整分非零数量: {(pattern_adjustment != 0).sum()}")
    
    # 7. 最终评分
    final_score = raw_score + pattern_adjustment
    final_score = np.clip(final_score, 0, 100)
    
    print(f"\n=== 最终评分 ===")
    print(f"最终评分范围: {final_score.min():.2f} - {final_score.max():.2f}")
    print(f"最终评分平均值: {final_score.mean():.2f}")
    print(f"最终评分标准差: {final_score.std():.2f}")
    print(f"唯一值数量: {final_score.nunique()}")
    
    # 8. 与实际方法对比
    print(f"\n=== 与实际方法对比 ===")
    actual_score = kdj.calculate_raw_score(data)
    print(f"实际方法评分范围: {actual_score.min():.2f} - {actual_score.max():.2f}")
    print(f"实际方法评分平均值: {actual_score.mean():.2f}")
    print(f"实际方法评分标准差: {actual_score.std():.2f}")
    print(f"实际方法唯一值数量: {actual_score.nunique()}")
    
    # 检查是否一致
    if np.allclose(final_score, actual_score):
        print("✅ 手动计算与实际方法一致")
    else:
        print("❌ 手动计算与实际方法不一致")
        
        # 显示差异
        diff = final_score - actual_score
        print(f"差异范围: {diff.min():.2f} - {diff.max():.2f}")
        print(f"差异平均值: {diff.mean():.2f}")

if __name__ == "__main__":
    debug_kdj_scoring() 