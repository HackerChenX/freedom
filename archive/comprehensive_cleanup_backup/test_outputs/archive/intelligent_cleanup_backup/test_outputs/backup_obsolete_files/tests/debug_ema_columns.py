#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试EMA指标返回的列名
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def debug_ema_columns():
    """调试EMA指标返回的列名"""
    print("🔍 调试EMA指标返回的列名...")
    
    try:
        from indicators.ema import EmaEma
        ema = EmaEma()
        
        # 创建测试数据
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        prices = [100 + i * 0.5 + np.random.normal(0, 1) for i in range(50)]
        
        test_data = pd.DataFrame({
            'date': dates,
            'code': ['TEST'] * 50,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * 50
        })
        
        print(f"📊 测试数据创建完成，{len(test_data)}行")
        
        # 测试不同周期
        for period in [5, 10, 20]:
            print(f"\n🔍 测试周期 {period}:")
            
            ema.set_parameters(period=period)
            result = ema.calculate(test_data)
            
            if result is not None:
                print(f"✅ 计算成功")
                print(f"📋 返回的列名: {list(result.columns)}")
                
                # 查找EMA相关的列
                ema_columns = [col for col in result.columns if 'ema' in col.lower()]
                print(f"🎯 EMA相关列: {ema_columns}")
                
                # 显示前几行数据
                if ema_columns:
                    for col in ema_columns:
                        values = result[col].dropna()
                        if len(values) > 0:
                            print(f"  {col}: {values.head(3).tolist()}")
            else:
                print(f"❌ 计算失败")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ema_columns()
