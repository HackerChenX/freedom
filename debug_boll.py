#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试BOLL指标的布林带关系问题
"""

import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from indicators.boll import BollBoll

def debug_boll():
    """调试BOLL指标"""
    print("🔍 调试BOLL指标的布林带关系问题")
    print("=" * 50)
    
    # 创建简单的测试数据
    np.random.seed(42)
    prices = [100 + i + np.random.normal(0, 2) for i in range(50)]
    
    data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=50),
        'code': 'TEST001',
        'open': prices,
        'high': [p + np.random.uniform(0, 2) for p in prices],
        'low': [p - np.random.uniform(0, 2) for p in prices],
        'close': prices,
        'volume': [100000] * 50
    })
    
    # 创建BOLL指标
    boll = BollBoll()
    result = boll.calculate(data)
    
    if result is not None:
        print(f"数据行数: {len(result)}")
        print(f"列名: {result.columns.tolist()}")
        
        # 检查前几行数据
        print("\n前10行数据:")
        cols_to_show = ['close', 'middle', 'upper', 'lower']
        if all(col in result.columns for col in cols_to_show):
            print(result[cols_to_show].head(10))
            
            # 检查布林带关系
            print("\n布林带关系检查:")
            upper_ge_middle = result['upper'] >= result['middle']
            middle_ge_lower = result['middle'] >= result['lower']
            
            print(f"上轨 >= 中轨: {upper_ge_middle.sum()}/{len(result)} ({upper_ge_middle.mean():.4f})")
            print(f"中轨 >= 下轨: {middle_ge_lower.sum()}/{len(result)} ({middle_ge_lower.mean():.4f})")
            
            # 找出违反关系的行
            upper_violations = ~upper_ge_middle
            lower_violations = ~middle_ge_lower
            
            if upper_violations.any():
                print(f"\n上轨 < 中轨的行:")
                violation_rows = result[upper_violations][cols_to_show]
                print(violation_rows)
            
            if lower_violations.any():
                print(f"\n中轨 < 下轨的行:")
                violation_rows = result[lower_violations][cols_to_show]
                print(violation_rows)
                
            # 检查NaN值
            nan_counts = result[cols_to_show].isna().sum()
            print(f"\nNaN值统计:")
            print(nan_counts)
        else:
            print("缺少必要的列")
    else:
        print("BOLL计算失败")

if __name__ == "__main__":
    debug_boll()
