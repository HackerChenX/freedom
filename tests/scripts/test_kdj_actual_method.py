#!/usr/bin/env python3
"""
调试KDJ的实际calculate_raw_score方法
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

def debug_kdj_actual_method():
    """调试KDJ的实际calculate_raw_score方法"""
    print("=== 调试KDJ的实际calculate_raw_score方法 ===")
    
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
    print(f"KDJ计算结果: {kdj.has_result()}")
    print(f"KDJ结果列: {list(kdj_result.columns) if kdj_result is not None else 'None'}")
    
    # 检查_result属性
    print(f"KDJ _result: {kdj._result is not None}")
    if kdj._result is not None:
        print(f"KDJ _result列: {list(kdj._result.columns)}")
        print(f"KDJ _result中是否有K,D,J: {all(col in kdj._result.columns for col in ['K', 'D', 'J'])}")
        
        # 显示KDJ值
        k = kdj._result['K']
        d = kdj._result['D']
        j = kdj._result['J']
        
        print(f"K值范围: {k.min():.2f} - {k.max():.2f}")
        print(f"D值范围: {d.min():.2f} - {d.max():.2f}")
        print(f"J值范围: {j.min():.2f} - {j.max():.2f}")
        
        # 手动执行calculate_raw_score的步骤
        print(f"\n=== 手动执行calculate_raw_score的步骤 ===")
        
        # 1. 位置分：基于K值的位置
        position_score = k.copy()
        print(f"原始K值作为位置分: {position_score.min():.2f} - {position_score.max():.2f}")
        
        # 调整曲线，使得中间值(50)得分为50分，两端得分递减
        position_score = 50 - 40 * np.abs(position_score - 50) / 50
        print(f"调整后位置分: {position_score.min():.2f} - {position_score.max():.2f}")
        
        # 在超买超卖区域有所调整
        position_score = np.where(k <= 20, 40 + (20 - k) * 1.5, position_score)  # 超卖区加分
        position_score = np.where(k >= 80, 40 - (k - 80) * 1.5, position_score)  # 超买区减分
        print(f"超买超卖调整后位置分: {position_score.min():.2f} - {position_score.max():.2f}")
        
        # 2. 趋势分：基于K和D的变化趋势
        k_trend = k - k.shift(3)
        d_trend = d - d.shift(3)
        print(f"K趋势范围: {k_trend.min():.2f} - {k_trend.max():.2f}")
        print(f"D趋势范围: {d_trend.min():.2f} - {d_trend.max():.2f}")
        
        trend_score = 50 + (k_trend + d_trend) * 3
        trend_score = np.clip(trend_score, 0, 100)
        print(f"趋势分: {trend_score.min():.2f} - {trend_score.max():.2f}")
        
        # 3. 金叉死叉分
        golden_cross = (k > d) & (k.shift(1) <= d.shift(1))
        death_cross = (k < d) & (k.shift(1) >= d.shift(1))
        print(f"金叉次数: {golden_cross.sum()}")
        print(f"死叉次数: {death_cross.sum()}")
        
        # 初始化交叉得分为50分（中性）
        cross_score = pd.Series(50.0, index=data.index)
        
        # 金叉加分，最近越近影响越大
        for i in range(5):
            mask = golden_cross.shift(i).fillna(False).astype(bool)
            score_boost = 30 * (0.8 ** i)  # 随距离衰减
            cross_score = np.where(mask, 50 + score_boost, cross_score)
        
        # 死叉减分，最近越近影响越大
        for i in range(5):
            mask = death_cross.shift(i).fillna(False).astype(bool)
            score_drop = 30 * (0.8 ** i)  # 随距离衰减
            cross_score = np.where(mask, 50 - score_drop, cross_score)
        
        print(f"交叉分: {cross_score.min():.2f} - {cross_score.max():.2f}")
        
        # 4. J值影响分
        j_score = 50 + (j - 50) * 0.2
        j_score = np.clip(j_score, 0, 100)
        print(f"J值影响分: {j_score.min():.2f} - {j_score.max():.2f}")
        
        # 合并各部分得分，按权重加权平均
        raw_score = (
            position_score * 0.4 +  # 位置分权重40%
            trend_score * 0.3 +     # 趋势分权重30%
            cross_score * 0.2 +     # 金叉死叉分权重20%
            j_score * 0.1           # J值影响分权重10%
        )
        print(f"加权平均后原始分: {raw_score.min():.2f} - {raw_score.max():.2f}")
        
        # 形态调整
        patterns = kdj.get_patterns(data)
        pattern_adjustment = pd.Series(0.0, index=data.index)
        
        registry = Pattern_registry()
        
        # 遍历所有检测到的形态
        for pattern_col in patterns.columns:
            pattern_info = registry.get_pattern(pattern_col)
            if pattern_info and 'score_impact' in pattern_info:
                score_impact = pattern_info['score_impact']
                # 对于每个时间点，如果形态存在，则应用调整
                for idx in patterns.index:
                    if patterns.at[idx, pattern_col]:
                        pattern_adjustment.at[idx] += np.clip(score_impact, -15, 15)
        
        # 应用形态调整（最多±15分）
        pattern_adjustment = np.clip(pattern_adjustment, -15, 15)
        raw_score += pattern_adjustment
        
        print(f"形态调整后: {raw_score.min():.2f} - {raw_score.max():.2f}")
        
        # 确保最终分数在0-100范围内
        final_score = np.clip(raw_score, 0, 100)
        print(f"最终分数: {final_score.min():.2f} - {final_score.max():.2f}")
        print(f"最终分数平均值: {final_score.mean():.2f}")
        print(f"最终分数标准差: {final_score.std():.2f}")
        print(f"最终分数唯一值数量: {final_score.nunique()}")
        
        # 与实际方法对比
        print(f"\n=== 与实际方法对比 ===")
        actual_score = kdj.calculate_raw_score(data)
        print(f"实际方法结果: {actual_score.min():.2f} - {actual_score.max():.2f}")
        print(f"实际方法平均值: {actual_score.mean():.2f}")
        print(f"实际方法标准差: {actual_score.std():.2f}")
        print(f"实际方法唯一值数量: {actual_score.nunique()}")
        
        # 检查是否一致
        if np.allclose(final_score, actual_score, rtol=1e-10):
            print("✅ 手动计算与实际方法一致")
        else:
            print("❌ 手动计算与实际方法不一致")
            
            # 显示差异
            diff = final_score - actual_score
            print(f"差异范围: {diff.min():.2f} - {diff.max():.2f}")
            print(f"差异平均值: {diff.mean():.2f}")
            print(f"差异标准差: {diff.std():.2f}")
            
            # 检查是否有NaN
            print(f"手动计算有NaN: {final_score.isna().sum()}")
            print(f"实际方法有NaN: {actual_score.isna().sum()}")
    
    else:
        print("❌ KDJ _result为None")

if __name__ == "__main__":
    debug_kdj_actual_method() 