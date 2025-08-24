#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD形态识别调试脚本

检查MACD指标实际输出的形态列，调试阶段2验证问题
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from clickhouse_driver import Client
except ImportError as e:
    print(f"导入错误: {e}")

def debug_macd_patterns():
    """调试MACD形态识别"""
    
    print("🔍 MACD形态识别调试")
    print("=" * 60)
    
    # 初始化MACD指标
    macd_indicator = MacdMacd()
    
    # 连接数据库获取真实数据
    client = Client(
        host='localhost',
        port=9000,
        database='stock',
        user='default',
        password='123456'
    )
    
    # 获取测试数据
    query = """
    SELECT date, open, high, low, close, volume
    FROM stock_info
    WHERE code = '002578'
    AND level = '日线'
    AND date >= '2024-01-01'
    ORDER BY date ASC
    LIMIT 100
    """
    
    result = client.execute(query)
    
    if result:
        df = pd.DataFrame(result, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df_sorted = df.sort_values('date').reset_index(drop=True)
        print(f"📊 获取到{len(df_sorted)}条数据")
        
        # 获取MACD形态识别结果
        print(f"\n🔧 调用get_patterns_Macd方法")
        patterns_result = macd_indicator.get_patterns_Macd(df_sorted)
        
        if patterns_result is not None and not patterns_result.empty:
            print(f"✅ 形态识别成功，结果形状: {patterns_result.shape}")
            
            # 显示所有列名
            print(f"\n📋 所有形态列:")
            for i, col in enumerate(patterns_result.columns, 1):
                print(f"  {i:2d}. {col}")
            
            # 分析每个形态列的信号数量
            print(f"\n📊 各形态信号统计:")
            for col in patterns_result.columns:
                if col in patterns_result.columns:
                    signals = patterns_result[col].dropna()
                    true_signals = signals[signals == True] if len(signals) > 0 else pd.Series()
                    print(f"  {col}: {len(true_signals)}个信号 (总数据{len(signals)})")
            
            # 检查特定形态
            target_patterns = [
                'GOLDEN_CROSS', 'DEATH_CROSS',
                'MACD_BULLISH_DIVERGENCE', 'MACD_BEARISH_DIVERGENCE',
                'MACD_ZERO_CROSS_ABOVE', 'MACD_ZERO_CROSS_BELOW',
                'MACD_HISTOGRAM_EXPANDING', 'MACD_HISTOGRAM_CONTRACTING'
            ]
            
            print(f"\n🎯 目标形态检查:")
            for pattern in target_patterns:
                if pattern in patterns_result.columns:
                    signals = patterns_result[pattern].dropna()
                    true_signals = signals[signals == True]
                    print(f"  ✅ {pattern}: {len(true_signals)}个信号")
                else:
                    print(f"  ❌ {pattern}: 列不存在")
            
            # 显示最近的信号
            print(f"\n📈 最近的形态信号:")
            for col in patterns_result.columns:
                signals = patterns_result[col].dropna()
                true_signals = signals[signals == True]
                if len(true_signals) > 0:
                    latest_signal_idx = true_signals.index[-1]
                    print(f"  {col}: 最新信号在索引{latest_signal_idx}")
        
        else:
            print(f"❌ 形态识别失败或无结果")
            
        # 测试MACD计算结果
        print(f"\n🔧 调用_calculate_macd方法")
        macd_data = macd_indicator._calculate_macd(df_sorted)
        
        if macd_data is not None and not macd_data.empty:
            print(f"✅ MACD计算成功，结果形状: {macd_data.shape}")
            print(f"📋 MACD数据列: {list(macd_data.columns)}")
            
            # 显示最新的MACD值
            if 'macd_line' in macd_data.columns:
                dif = macd_data['macd_line'].dropna()
                if len(dif) > 0:
                    print(f"  最新DIF: {dif.iloc[-1]:.6f}")
            
            if 'macd_signal' in macd_data.columns:
                dea = macd_data['macd_signal'].dropna()
                if len(dea) > 0:
                    print(f"  最新DEA: {dea.iloc[-1]:.6f}")
            
            if 'macd_histogram' in macd_data.columns:
                hist = macd_data['macd_histogram'].dropna()
                if len(hist) > 0:
                    print(f"  最新柱状图: {hist.iloc[-1]:.6f}")
        else:
            print(f"❌ MACD计算失败")
    
    else:
        print(f"❌ 无法获取测试数据")

def create_simple_test_data():
    """创建简单的测试数据"""
    
    print(f"\n🔧 创建简单测试数据")
    
    # 创建明显的金叉形态数据
    dates = pd.date_range('2025-01-01', periods=50, freq='D')
    
    # 前25天：下跌趋势，DIF < DEA
    # 后25天：上涨趋势，DIF > DEA，形成金叉
    prices = []
    for i in range(50):
        if i < 25:
            # 下跌阶段
            price = 100 - i * 0.5
        else:
            # 上涨阶段，形成金叉
            price = 87.5 + (i - 25) * 1.0
        prices.append(price)
    
    test_data = pd.DataFrame({
        'date': dates,
        'open': [p * 0.99 for p in prices],
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [1000000] * 50
    })
    
    print(f"📊 创建了{len(test_data)}条测试数据")
    print(f"价格范围: {min(prices):.2f} - {max(prices):.2f}")
    
    # 测试MACD形态识别
    macd_indicator = MacdMacd()
    
    print(f"\n🔧 测试简单数据的形态识别")
    patterns_result = macd_indicator.get_patterns_Macd(test_data)
    
    if patterns_result is not None and not patterns_result.empty:
        print(f"✅ 简单数据形态识别成功")
        
        # 检查金叉信号
        if 'GOLDEN_CROSS' in patterns_result.columns:
            golden_cross = patterns_result['GOLDEN_CROSS'].dropna()
            true_signals = golden_cross[golden_cross == True]
            print(f"  金叉信号: {len(true_signals)}个")
            
            if len(true_signals) > 0:
                print(f"  金叉位置: {list(true_signals.index)}")
        else:
            print(f"  ❌ 未找到GOLDEN_CROSS列")
    else:
        print(f"❌ 简单数据形态识别失败")

if __name__ == "__main__":
    debug_macd_patterns()
    create_simple_test_data()
