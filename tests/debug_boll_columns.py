#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试BOLL指标的列名格式和参数方法
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def debug_boll_columns():
    """调试BOLL指标的列名格式和参数方法"""
    print("🔍 调试BOLL指标的列名格式和参数方法...")
    
    try:
        from indicators.boll import BollBoll
        boll = BollBoll()
        
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
        
        # 检查可用方法
        print(f"\n📋 BOLL指标可用方法:")
        methods = [method for method in dir(boll) if not method.startswith('__')]
        for method in methods:
            if 'set_parameters' in method.lower() or 'parameter' in method.lower():
                print(f"  - {method}")
        
        # 测试默认参数
        print(f"\n📋 默认参数:")
        if hasattr(boll, '_get_default_parameters'):
            try:
                default_params = boll._get_default_parameters()
                print(f"  - 默认参数: {default_params}")
            except Exception as e:
                print(f"  - 获取默认参数失败: {e}")
        
        # 测试计算
        print(f"\n🔧 测试默认计算:")
        try:
            result = boll.calculate(test_data)
            
            if result is not None:
                print(f"  - 计算成功")
                print(f"  - 返回列名: {list(result.columns)}")
                
                # 查找BOLL相关的列
                boll_columns = [col for col in result.columns if 'BOLL' in col or 'boll' in col]
                print(f"  - BOLL相关列: {boll_columns}")
                
                if boll_columns:
                    for col in boll_columns:
                        values = result[col].dropna()
                        if len(values) > 0:
                            print(f"    - {col}: {len(values)}个值, 示例: {values.iloc[:3].tolist()}")
            else:
                print(f"  - 计算失败")
        except Exception as e:
            print(f"  - 计算异常: {e}")
        
        # 测试参数设置
        print(f"\n🔧 测试参数设置:")
        try:
            if hasattr(boll, 'set_parameters_Boll'):
                boll.set_parameters_Boll(period=20, std_dev=2)
                print(f"  - set_parameters_Boll 成功")
            elif hasattr(boll, 'set_parameters'):
                boll.set_parameters(period=20, std_dev=2)
                print(f"  - set_parameters 成功")
            else:
                print(f"  - 未找到参数设置方法")
            
            result = boll.calculate(test_data)
            if result is not None:
                boll_columns = [col for col in result.columns if 'BOLL' in col or 'boll' in col]
                print(f"  - 参数设置后BOLL列: {boll_columns}")
            
        except Exception as e:
            print(f"  - 参数设置异常: {e}")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_boll_columns()
