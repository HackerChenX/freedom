#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD零轴穿越调试脚本

专门调试零轴穿越检测问题
"""

import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
except ImportError as e:
    print(f"导入错误: {e}")

def debug_zero_cross():
    """调试零轴穿越"""
    
    print("🔍 MACD零轴穿越专项调试")
    print("=" * 60)
    
    # 创建明显的零轴穿越数据
    dates = pd.date_range('2025-01-01', periods=60, freq='D')
    
    # 设计价格数据：前30天下跌（MACD负值），后30天上涨（MACD正值）
    prices = []
    for i in range(60):
        if i < 30:
            # 下跌阶段，确保MACD为负
            price = 100 - i * 2.0  # 大幅下跌
        else:
            # 上涨阶段，确保MACD为正
            price = 40 + (i - 30) * 3.0  # 大幅上涨
        prices.append(price)
    
    test_data = pd.DataFrame({
        'date': dates,
        'open': [p * 0.99 for p in prices],
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [1000000] * 60
    })
    
    print(f"📊 创建了{len(test_data)}条测试数据")
    print(f"价格范围: {min(prices):.2f} - {max(prices):.2f}")
    print(f"前30天: {prices[0]:.2f} -> {prices[29]:.2f} (下跌)")
    print(f"后30天: {prices[30]:.2f} -> {prices[59]:.2f} (上涨)")
    
    # 初始化MACD指标
    macd_indicator = MacdMacd()
    
    # 计算MACD数据
    print(f"\n🔧 计算MACD数据")
    macd_data = macd_indicator._calculate_macd(test_data)
    
    if macd_data is not None and not macd_data.empty:
        print(f"✅ MACD计算成功，形状: {macd_data.shape}")
        
        if 'macd_line' in macd_data.columns:
            dif = macd_data['macd_line'].dropna()
            print(f"📊 DIF数据点: {len(dif)}")
            
            if len(dif) > 10:
                print(f"DIF前5个值: {dif.head().tolist()}")
                print(f"DIF后5个值: {dif.tail().tolist()}")
                
                # 检查零轴穿越
                zero_crosses_up = []
                zero_crosses_down = []
                
                for i in range(1, len(dif)):
                    prev_val = dif.iloc[i-1]
                    curr_val = dif.iloc[i]
                    
                    if not pd.isna(prev_val) and not pd.isna(curr_val):
                        # 向上穿越零轴
                        if prev_val <= 0 and curr_val > 0:
                            zero_crosses_up.append(i)
                            print(f"  🔺 向上穿越零轴: 索引{i}, {prev_val:.6f} -> {curr_val:.6f}")
                        
                        # 向下穿越零轴
                        if prev_val >= 0 and curr_val < 0:
                            zero_crosses_down.append(i)
                            print(f"  🔻 向下穿越零轴: 索引{i}, {prev_val:.6f} -> {curr_val:.6f}")
                
                print(f"\n📊 手动检测结果:")
                print(f"  向上穿越: {len(zero_crosses_up)}次")
                print(f"  向下穿越: {len(zero_crosses_down)}次")
    
    # 获取形态识别结果
    print(f"\n🔧 获取形态识别结果")
    patterns_result = macd_indicator.get_patterns_Macd(test_data)
    
    if patterns_result is not None and not patterns_result.empty:
        print(f"✅ 形态识别成功，形状: {patterns_result.shape}")
        
        # 检查零轴穿越相关列
        zero_columns = [col for col in patterns_result.columns if 'ZERO_CROSS' in col]
        print(f"📋 零轴穿越相关列: {zero_columns}")
        
        for col in zero_columns:
            if col in patterns_result.columns:
                signals = patterns_result[col].dropna()
                true_signals = signals[signals == True]
                print(f"  {col}: {len(true_signals)}个信号")
                
                if len(true_signals) > 0:
                    print(f"    信号位置: {list(true_signals.index)}")
        
        # 显示所有形态的信号数量
        print(f"\n📊 所有形态信号统计:")
        for col in patterns_result.columns:
            signals = patterns_result[col].dropna()
            true_signals = signals[signals == True]
            print(f"  {col}: {len(true_signals)}个信号")
    
    else:
        print(f"❌ 形态识别失败")

def test_simple_zero_cross():
    """测试简单的零轴穿越"""
    
    print(f"\n🔧 测试简单零轴穿越")
    print("=" * 60)
    
    # 创建极简的零轴穿越数据
    dates = pd.date_range('2025-01-01', periods=40, freq='D')
    
    # 简单的V型反转：下跌然后上涨
    prices = []
    for i in range(40):
        if i < 20:
            price = 100 - i * 1.5  # 下跌到70
        else:
            price = 70 + (i - 20) * 2.0  # 上涨到110
        prices.append(price)
    
    simple_data = pd.DataFrame({
        'date': dates,
        'open': [p * 0.995 for p in prices],
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [2000000] * 40
    })
    
    print(f"📊 简单数据: {len(simple_data)}条")
    print(f"价格变化: {prices[0]:.1f} -> {prices[19]:.1f} -> {prices[39]:.1f}")
    
    # 测试MACD
    macd_indicator = MacdMacd()
    
    # 计算MACD
    macd_data = macd_indicator._calculate_macd(simple_data)
    
    if macd_data is not None and 'macd_line' in macd_data.columns:
        dif = macd_data['macd_line'].dropna()
        
        if len(dif) > 5:
            print(f"📊 DIF值范围: {dif.min():.6f} 到 {dif.max():.6f}")
            
            # 查找零轴穿越
            for i in range(1, len(dif)):
                prev_val = dif.iloc[i-1]
                curr_val = dif.iloc[i]
                
                if not pd.isna(prev_val) and not pd.isna(curr_val):
                    if (prev_val <= 0 and curr_val > 0) or (prev_val >= 0 and curr_val < 0):
                        print(f"  零轴穿越: 索引{i}, {prev_val:.6f} -> {curr_val:.6f}")
    
    # 测试形态识别
    patterns_result = macd_indicator.get_patterns_Macd(simple_data)
    
    if patterns_result is not None and not patterns_result.empty:
        zero_columns = [col for col in patterns_result.columns if 'ZERO_CROSS' in col]
        
        for col in zero_columns:
            signals = patterns_result[col].dropna()
            true_signals = signals[signals == True]
            print(f"  {col}: {len(true_signals)}个信号")

if __name__ == "__main__":
    debug_zero_cross()
    test_simple_zero_cross()
