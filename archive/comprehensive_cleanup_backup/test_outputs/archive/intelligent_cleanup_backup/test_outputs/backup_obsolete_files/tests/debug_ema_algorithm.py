#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试EMA算法准确性
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def debug_ema_algorithm():
    """调试EMA算法准确性"""
    print("🔍 调试EMA算法准确性...")
    
    try:
        from indicators.ema import EmaEma
        ema = EmaEma()
        
        # 创建简单的测试数据
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        dates = pd.date_range(start='2024-01-01', periods=len(prices), freq='D')
        
        test_data = pd.DataFrame({
            'date': dates,
            'code': ['TEST'] * len(prices),
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * len(prices)
        })
        
        print(f"📊 测试数据: {prices}")
        
        # 测试EMA计算
        period = 5
        ema.set_parameters(period=period)
        result = ema.calculate(test_data)
        
        if result is not None:
            print(f"✅ EMA计算成功")
            print(f"📋 返回的列名: {list(result.columns)}")
            
            ema_col = f'EMA_Ema{period}'
            if ema_col in result.columns:
                ema_values = result[ema_col].dropna()
                print(f"🎯 EMA{period}值: {ema_values.tolist()}")
                
                # 手动计算EMA
                manual_ema = calculate_manual_ema(prices, period)
                print(f"📐 手动EMA值: {manual_ema}")
                
                # 比较结果
                if len(ema_values) > 0 and len(manual_ema) > 0:
                    min_len = min(len(ema_values), len(manual_ema))
                    ema_array = ema_values.iloc[-min_len:].values
                    manual_array = manual_ema[-min_len:]
                    
                    print(f"\n🔍 比较结果 (最后{min_len}个值):")
                    print(f"EMA指标: {ema_array}")
                    print(f"手动计算: {manual_array}")
                    
                    diff = np.abs(ema_array - manual_array)
                    max_diff = np.max(diff)
                    print(f"最大差异: {max_diff}")
                    print(f"准确性: {'✅ 通过' if max_diff < 1e-6 else '❌ 失败'}")
                    
                    # 详细比较
                    for i in range(min_len):
                        print(f"  位置{i}: EMA={ema_array[i]:.6f}, 手动={manual_array[i]:.6f}, 差异={diff[i]:.10f}")
            else:
                print(f"❌ 找不到列 {ema_col}")
        else:
            print(f"❌ EMA计算失败")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

def calculate_manual_ema(prices, period):
    """手动计算EMA，使用pandas ewm方法"""
    import pandas as pd
    price_series = pd.Series(prices)
    manual_ema = price_series.ewm(span=period, adjust=True).mean()
    return manual_ema.tolist()

if __name__ == "__main__":
    debug_ema_algorithm()
