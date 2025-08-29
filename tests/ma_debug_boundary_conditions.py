#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MA指标边界条件调试脚本
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def test_boundary_conditions():
    """测试边界条件"""
    from indicators.ma import MaMa
    
    print("🔍 测试MA边界条件...")
    
    boundary_tests = [
        ('minimum_data', 5, 5),      # 最小数据量
        ('exact_period', 20, 20),    # 精确周期数据
        ('large_period', 100, 50),   # 大周期小数据
        ('single_value', 1, 10),     # 单个数据点
    ]
    
    for test_name, data_size, period in boundary_tests:
        print(f"\n--- 测试 {test_name}: 数据{data_size}条, 周期{period} ---")
        
        try:
            # 创建测试数据
            dates = pd.date_range(start='2024-01-01', periods=data_size, freq='D')
            prices = [100 + i for i in range(data_size)]
            
            test_data = pd.DataFrame({
                'date': dates,
                'close': prices,
                'high': [p * 1.02 for p in prices],
                'low': [p * 0.98 for p in prices],
                'volume': [1000000] * data_size
            })
            
            print(f"测试数据: {test_data.shape}")
            print(f"Close值: {test_data['close'].tolist()}")
            
            # 创建MA实例并设置参数
            ma = MaMa()
            ma.set_parameters(period=period, ma_type='SMA')
            
            print(f"设置参数: period={period}, ma_type=SMA")
            print(f"MA实例period: {getattr(ma, 'period', 'NOT_SET')}")
            
            # 计算MA
            result = ma.calculate(test_data)
            
            print(f"计算结果: {type(result)}")
            if result is not None:
                print(f"结果形状: {result.shape}")
                print(f"结果列: {list(result.columns)}")
                
                if 'ma' in result.columns:
                    ma_values = result['ma']
                    print(f"MA值: {ma_values.tolist()}")
                    print(f"有效值数量: {ma_values.notna().sum()}")
                    print(f"NaN值数量: {ma_values.isna().sum()}")
                    
                    # 验证结果合理性
                    if data_size >= period:
                        print("✅ 数据足够，应该有有效值")
                        if ma_values.notna().sum() > 0:
                            print("✅ 有有效值")
                        else:
                            print("❌ 没有有效值")
                    else:
                        print("⚠️ 数据不足，应该全为NaN或合理处理")
                        if ma_values.isna().all():
                            print("✅ 全为NaN，处理正确")
                        else:
                            print("⚠️ 不全为NaN，但可能是合理的处理")
                else:
                    print("❌ 缺少ma列")
            else:
                print("❌ 结果为None")
                
        except Exception as e:
            print(f"❌ 异常: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_boundary_conditions()
