#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试BOLL计算问题
"""

import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def debug_boll():
    """调试BOLL计算"""
    print("🔍 调试BOLL计算问题")
    print("=" * 50)
    
    try:
        from indicators.boll import BollBoll
        boll = BollBoll()
        
        # 创建测试数据
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        np.random.seed(42)
        
        base_price = 100
        price_changes = np.random.normal(0.1, 2, 30)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change / 100)
            prices.append(max(new_price, 1))
        
        test_data = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        
        print(f"📊 测试数据前10行:")
        print(test_data.head(10))
        
        # 执行计算
        result = boll.calculate(test_data)
        
        print(f"\n📊 BOLL计算结果前25行:")
        print(result[['close', 'middle', 'upper', 'lower']].head(25))
        
        # 检查前20行的上下轨关系
        print(f"\n🔍 前20行上下轨关系检查:")
        for i in range(min(20, len(result))):
            close = result.iloc[i]['close']
            middle = result.iloc[i]['middle']
            upper = result.iloc[i]['upper']
            lower = result.iloc[i]['lower']
            
            upper_ok = pd.isna(upper) or upper >= middle
            lower_ok = pd.isna(lower) or lower <= middle
            
            print(f"  第{i+1}行: close={close:.2f}, middle={middle:.2f}, upper={upper:.2f}, lower={lower:.2f}, "
                  f"上轨{'✅' if upper_ok else '❌'}, 下轨{'✅' if lower_ok else '❌'}")
        
        # 手动计算SMA和标准差验证
        print(f"\n🔍 手动验证SMA计算:")
        manual_sma = test_data['close'].rolling(window=20, min_periods=20).mean()
        manual_std = test_data['close'].rolling(window=20, min_periods=20).std()
        
        print("前25行手动SMA:")
        for i in range(min(25, len(manual_sma))):
            sma_val = manual_sma.iloc[i]
            std_val = manual_std.iloc[i]
            boll_middle = result.iloc[i]['middle']
            
            print(f"  第{i+1}行: 手动SMA={sma_val:.2f}, BOLL中轨={boll_middle:.2f}, "
                  f"标准差={std_val:.2f}, 匹配={'✅' if abs(sma_val - boll_middle) < 0.01 or (pd.isna(sma_val) and pd.isna(boll_middle)) else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"💥 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_boll()
