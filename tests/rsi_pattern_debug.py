#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSI形态检测调试脚本

专门用于调试和修复RSI复杂形态检测问题
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from utils.technical_utils import calculate_rsi_Utils
except ImportError as e:
    print(f"导入错误: {e}")

def generate_golden_cross_data():
    """生成确保产生金叉的数据"""
    np.random.seed(42)
    n_points = 100
    base_price = 15.0
    prices = [base_price]
    
    for i in range(1, n_points):
        if i < 30:
            # 前30天：明显下降，RSI降到30以下
            trend = -0.015 + np.random.normal(0, 0.003)
        elif i < 50:
            # 中间20天：震荡，RSI在30-40区间
            trend = np.random.normal(0, 0.005)
        elif i < 70:
            # 接下来20天：缓慢上升，RSI开始回升
            trend = 0.008 + np.random.normal(0, 0.003)
        else:
            # 最后30天：加速上升，确保RSI快速上升形成金叉
            trend = 0.015 + np.random.normal(0, 0.002)
        
        new_price = prices[-1] * (1 + trend)
        prices.append(new_price)
    
    return np.array(prices)

def analyze_rsi_pattern(prices, pattern_name):
    """分析RSI形态"""
    print(f"\n🔍 分析{pattern_name}形态")
    print("=" * 60)
    
    # 计算RSI
    price_series = pd.Series(prices)
    rsi_values = calculate_rsi_Utils(price_series, 14)
    
    if rsi_values is None or rsi_values.empty:
        print("❌ RSI计算失败")
        return False
    
    print(f"📊 数据点数: {len(prices)}")
    print(f"📊 RSI数据点数: {len(rsi_values)}")
    print(f"📊 RSI范围: {rsi_values.min():.2f} - {rsi_values.max():.2f}")
    
    # 计算RSI移动平均
    rsi_ma5 = rsi_values.rolling(window=5).mean()
    rsi_ma10 = rsi_values.rolling(window=10).mean()
    
    print(f"📊 RSI MA5范围: {rsi_ma5.min():.2f} - {rsi_ma5.max():.2f}")
    print(f"📊 RSI MA10范围: {rsi_ma10.min():.2f} - {rsi_ma10.max():.2f}")
    
    # 检查金叉条件
    golden_crosses = []
    for i in range(15, len(rsi_ma5)):
        if (not pd.isna(rsi_ma5.iloc[i]) and not pd.isna(rsi_ma10.iloc[i]) and
            not pd.isna(rsi_ma5.iloc[i-1]) and not pd.isna(rsi_ma10.iloc[i-1])):
            
            current_short = rsi_ma5.iloc[i]
            current_long = rsi_ma10.iloc[i]
            prev_short = rsi_ma5.iloc[i-1]
            prev_long = rsi_ma10.iloc[i-1]
            
            # 金叉条件
            if prev_short <= prev_long and current_short > current_long:
                cross_strength = abs(current_short - current_long) - abs(prev_short - prev_long)
                golden_crosses.append({
                    'index': i,
                    'prev_short': prev_short,
                    'prev_long': prev_long,
                    'current_short': current_short,
                    'current_long': current_long,
                    'cross_strength': cross_strength
                })
                print(f"🎯 发现金叉 第{i}天: MA5={current_short:.2f}, MA10={current_long:.2f}, 强度={cross_strength:.3f}")
    
    print(f"📈 总共发现 {len(golden_crosses)} 个金叉")
    
    # 检查死叉条件
    death_crosses = []
    for i in range(15, len(rsi_ma5)):
        if (not pd.isna(rsi_ma5.iloc[i]) and not pd.isna(rsi_ma10.iloc[i]) and
            not pd.isna(rsi_ma5.iloc[i-1]) and not pd.isna(rsi_ma10.iloc[i-1])):
            
            current_short = rsi_ma5.iloc[i]
            current_long = rsi_ma10.iloc[i]
            prev_short = rsi_ma5.iloc[i-1]
            prev_long = rsi_ma10.iloc[i-1]
            
            # 死叉条件
            if prev_short >= prev_long and current_short < current_long:
                cross_strength = abs(current_short - current_long) - abs(prev_short - prev_long)
                death_crosses.append({
                    'index': i,
                    'prev_short': prev_short,
                    'prev_long': prev_long,
                    'current_short': current_short,
                    'current_long': current_long,
                    'cross_strength': cross_strength
                })
                print(f"📉 发现死叉 第{i}天: MA5={current_short:.2f}, MA10={current_long:.2f}, 强度={cross_strength:.3f}")
    
    print(f"📉 总共发现 {len(death_crosses)} 个死叉")
    
    return len(golden_crosses) > 0 or len(death_crosses) > 0

def test_enhanced_detection():
    """测试增强的检测算法"""
    print("🎯 RSI形态检测调试分析")
    print("=" * 80)
    
    # 测试金叉数据
    golden_cross_prices = generate_golden_cross_data()
    result1 = analyze_rsi_pattern(golden_cross_prices, "金叉")
    
    # 测试死叉数据（反转金叉数据）
    death_cross_prices = golden_cross_prices[::-1]  # 反转数据
    result2 = analyze_rsi_pattern(death_cross_prices, "死叉")
    
    print(f"\n🏆 测试结果总结")
    print("=" * 60)
    print(f"金叉检测: {'✅ 成功' if result1 else '❌ 失败'}")
    print(f"死叉检测: {'✅ 成功' if result2 else '❌ 失败'}")
    
    return result1, result2

def enhanced_golden_cross_detection(rsi_data: pd.Series) -> bool:
    """增强的RSI金叉检测（调试版）"""
    if len(rsi_data) < 15:
        print("❌ 数据长度不足")
        return False
    
    try:
        # 计算RSI的短期和长期移动平均
        rsi_ma5 = rsi_data.rolling(window=5).mean()
        rsi_ma10 = rsi_data.rolling(window=10).mean()
        
        print(f"📊 MA5最后5个值: {rsi_ma5.tail(5).values}")
        print(f"📊 MA10最后5个值: {rsi_ma10.tail(5).values}")
        
        # 检查最近5天内是否形成金叉
        for i in range(1, min(6, len(rsi_ma5))):
            if len(rsi_ma5) >= i+1 and len(rsi_ma10) >= i+1:
                current_short = rsi_ma5.iloc[-i]
                current_long = rsi_ma10.iloc[-i]
                prev_short = rsi_ma5.iloc[-i-1]
                prev_long = rsi_ma10.iloc[-i-1]
                
                print(f"🔍 检查第{i}天: 前MA5={prev_short:.3f}, 前MA10={prev_long:.3f}, 当前MA5={current_short:.3f}, 当前MA10={current_long:.3f}")
                
                # 金叉条件：短期均线上穿长期均线
                if (prev_short <= prev_long and current_short > current_long and 
                    not pd.isna(current_short) and not pd.isna(current_long) and
                    not pd.isna(prev_short) and not pd.isna(prev_long)):
                    
                    # 计算穿越强度
                    cross_strength = abs(current_short - current_long) - abs(prev_short - prev_long)
                    print(f"🎯 发现金叉! 穿越强度: {cross_strength:.3f}")
                    
                    if cross_strength > 0.05:  # 降低阈值
                        print(f"✅ 金叉确认! 强度足够")
                        return True
                    else:
                        print(f"⚠️ 金叉强度不足: {cross_strength:.3f} < 0.05")
        
        print("❌ 未发现金叉")
        return False
        
    except Exception as e:
        print(f"❌ RSI金叉检测异常: {e}")
        return False

def test_specific_detection():
    """测试特定的检测逻辑"""
    print("\n🔬 测试特定检测逻辑")
    print("=" * 60)
    
    # 生成测试数据
    prices = generate_golden_cross_data()
    price_series = pd.Series(prices)
    rsi_values = calculate_rsi_Utils(price_series, 14)
    
    if rsi_values is None or rsi_values.empty:
        print("❌ RSI计算失败")
        return False
    
    # 使用增强的检测逻辑
    result = enhanced_golden_cross_detection(rsi_values)
    
    print(f"🏆 增强检测结果: {'✅ 成功' if result else '❌ 失败'}")
    return result

if __name__ == "__main__":
    # 运行调试测试
    test_enhanced_detection()
    test_specific_detection()
