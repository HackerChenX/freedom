#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试MA指标的列名格式
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def debug_ma_columns():
    """调试MA指标的列名格式"""
    print("🔍 调试MA指标的列名格式...")
    
    try:
        from indicators.ma import MaMa
        ma = MaMa()
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'code': ['TEST'] * 50,
            'open': [100 + i for i in range(50)],
            'high': [101 + i for i in range(50)],
            'low': [99 + i for i in range(50)],
            'close': [100 + i for i in range(50)],
            'volume': [1000000] * 50
        })
        
        print(f"📊 测试数据: {len(test_data)}条")
        
        # 测试默认参数
        print(f"\n📋 默认参数:")
        if hasattr(ma, '_get_default_parameters'):
            default_params = ma._get_default_parameters()
            print(f"  - 默认参数: {default_params}")
        
        # 测试不同周期
        periods = [5, 10, 20]
        for period in periods:
            print(f"\n🔧 测试周期 {period}:")
            ma.set_parameters_Ma(period=period)
            result = ma.calculate(test_data)
            
            if result is not None:
                print(f"  - 计算成功")
                print(f"  - 返回列名: {list(result.columns)}")
                
                # 查找MA相关的列
                ma_columns = [col for col in result.columns if 'MA' in col or 'ma' in col]
                print(f"  - MA相关列: {ma_columns}")
                
                if ma_columns:
                    for col in ma_columns:
                        values = result[col].dropna()
                        if len(values) > 0:
                            print(f"    - {col}: {len(values)}个值, 示例: {values.iloc[:3].tolist()}")
            else:
                print(f"  - 计算失败")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ma_columns()
