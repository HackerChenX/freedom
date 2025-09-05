#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试MACD形态识别问题

分析MACD指标的实际输出，找出形态识别失败的原因
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from indicators.macd import MacdMacd
from tests.unified_indicator_testing.components.stockinfo_compatible_data_generator import StockInfoCompatibleDataGenerator

def debug_macd_patterns():
    """调试MACD形态识别"""
    
    print("🔍 开始调试MACD形态识别")
    print("=" * 60)
    
    # 初始化
    macd = MacdMacd()
    generator = StockInfoCompatibleDataGenerator()
    
    # 测试每个形态
    patterns_to_test = ['GOLDEN_CROSS', 'DEATH_CROSS', 'BEARISH_DIVERGENCE']
    
    for pattern in patterns_to_test:
        print(f"\n📊 测试形态: {pattern}")
        print("-" * 40)
        
        # 生成测试数据
        test_data = generator.generate_stockinfo_compatible_data(
            indicator_name='MACD',
            pattern_type=pattern,
            stock_code=f'DEBUG_{pattern}',
            history_days=60
        )
        
        if test_data is None:
            print(f"❌ 数据生成失败: {pattern}")
            continue
        
        print(f"✅ 数据生成成功: {len(test_data)} 行")
        
        # 计算MACD指标
        calc_result = macd.calculate(test_data)
        if calc_result is None:
            print(f"❌ MACD计算失败: {pattern}")
            continue
        
        print(f"✅ MACD计算成功")
        print(f"   计算结果列: {list(calc_result.columns)}")
        
        # 获取形态识别结果
        patterns_result = macd.get_patterns(test_data)
        if patterns_result is None:
            print(f"❌ 形态识别失败: {pattern}")
            continue
        
        print(f"✅ 形态识别成功")
        print(f"   识别的形态: {list(patterns_result.columns)}")
        
        # 分析每个形态的识别情况
        for col in patterns_result.columns:
            detections = patterns_result[col].sum()
            print(f"   {col}: {detections} 次检测")
            
            if detections > 0:
                # 显示检测到的位置
                detection_indices = patterns_result[patterns_result[col]].index.tolist()
                print(f"     检测位置: {detection_indices[-5:]}")  # 显示最后5个位置
        
        # 分析MACD数值
        if calc_result is not None:
            print(f"\n📈 MACD数值分析 (最后10行):")
            macd_cols = [col for col in calc_result.columns if 'macd' in col.lower()]
            if macd_cols:
                recent_data = calc_result[macd_cols].tail(10)
                print(recent_data.to_string())
    
    # 测试噪声数据
    print(f"\n🔊 测试噪声数据")
    print("-" * 40)
    
    noise_data = generate_simple_noise_data()
    print(f"✅ 噪声数据生成: {len(noise_data)} 行")
    
    noise_patterns = macd.get_patterns(noise_data)
    if noise_patterns is not None:
        print(f"✅ 噪声数据形态识别完成")
        
        total_detections = 0
        for col in noise_patterns.columns:
            detections = noise_patterns[col].sum()
            total_detections += detections
            print(f"   {col}: {detections} 次检测")
        
        print(f"📊 总假阳性: {total_detections}")
        
        if total_detections > 0:
            print(f"\n⚠️ 噪声数据产生了假阳性，需要进一步优化过滤条件")
    else:
        print(f"❌ 噪声数据形态识别失败")

def generate_simple_noise_data():
    """生成简单的噪声数据"""
    
    # 生成随机价格数据
    days = 60
    base_price = 10.0
    noise_data = []
    
    for i in range(days):
        # 添加随机噪声
        price_change = np.random.normal(0, 0.01)  # 1%标准差
        base_price *= (1 + price_change)
        
        noise_data.append({
            'date': pd.Timestamp.now() - pd.Timedelta(days=days-i),
            'open': base_price * (1 + np.random.normal(0, 0.005)),
            'high': base_price * (1 + abs(np.random.normal(0, 0.01))),
            'low': base_price * (1 - abs(np.random.normal(0, 0.01))),
            'close': base_price,
            'volume': np.random.randint(1000000, 10000000)
        })
    
    return pd.DataFrame(noise_data)

def test_macd_above_zero_golden():
    """专门测试MACD_ABOVE_ZERO_GOLDEN形态"""
    
    print(f"\n🎯 专门测试MACD_ABOVE_ZERO_GOLDEN")
    print("=" * 60)
    
    macd = MacdMacd()
    generator = StockInfoCompatibleDataGenerator()
    
    # 生成金叉数据
    test_data = generator.generate_stockinfo_compatible_data(
        indicator_name='MACD',
        pattern_type='GOLDEN_CROSS',
        stock_code='TEST_ABOVE_ZERO',
        history_days=60
    )
    
    if test_data is None:
        print("❌ 数据生成失败")
        return
    
    # 计算MACD
    calc_result = macd.calculate(test_data)
    if calc_result is None:
        print("❌ MACD计算失败")
        return
    
    # 获取形态
    patterns_result = macd.get_patterns(test_data)
    if patterns_result is None:
        print("❌ 形态识别失败")
        return
    
    print("✅ 数据处理成功")
    
    # 分析MACD值和零轴关系
    macd_line = calc_result.get('macd_line', calc_result.get('MACD', None))
    macd_signal = calc_result.get('macd_signal', calc_result.get('MACD_SIGNAL', None))
    
    if macd_line is not None and macd_signal is not None:
        print(f"\n📊 MACD线和信号线分析 (最后10行):")
        analysis_df = pd.DataFrame({
            'MACD_LINE': macd_line,
            'MACD_SIGNAL': macd_signal,
            'MACD_ABOVE_ZERO': macd_line > 0,
            'SIGNAL_ABOVE_ZERO': macd_signal > -0.001,
            'GOLDEN_CROSS': patterns_result.get('GOLDEN_CROSS', False),
            'ABOVE_ZERO_GOLDEN': patterns_result.get('MACD_ABOVE_ZERO_GOLDEN', False)
        }).tail(10)
        
        print(analysis_df.to_string())
        
        # 检查零轴上方金叉的条件
        above_zero_opportunities = (
            (macd_line > 0) & 
            (macd_signal > -0.001) & 
            patterns_result.get('GOLDEN_CROSS', False)
        ).sum()
        
        print(f"\n📈 零轴上方金叉机会: {above_zero_opportunities} 次")
        
        if above_zero_opportunities > 0:
            print("✅ 存在零轴上方金叉机会，但未被识别")
            print("💡 可能需要调整MACD_ABOVE_ZERO_GOLDEN的识别逻辑")
        else:
            print("ℹ️ 当前数据中没有零轴上方金叉机会")
    else:
        print("❌ 无法获取MACD线和信号线数据")
        print(f"可用列: {list(calc_result.columns)}")

if __name__ == "__main__":
    debug_macd_patterns()
    test_macd_above_zero_golden()
