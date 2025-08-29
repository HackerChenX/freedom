#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试KDJ指标修复效果
"""

import sys
import traceback
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def test_kdj_fixes():
    """测试KDJ指标修复效果"""
    print("🔍 测试KDJ指标修复效果")
    print("=" * 50)
    
    try:
        # 测试1: 实例化
        print("\n📋 测试1: KDJ实例化")
        try:
            from indicators.kdj import KdjKdj
            kdj = KdjKdj()
            print("✅ KDJ实例化成功")
        except Exception as e:
            print(f"❌ KDJ实例化失败: {e}")
            traceback.print_exc()
            return False
        
        # 测试2: minimum_periods属性
        print("\n📋 测试2: minimum_periods属性")
        try:
            min_periods = kdj.minimum_periods
            print(f"✅ minimum_periods: {min_periods}")
        except Exception as e:
            print(f"❌ minimum_periods测试失败: {e}")
            traceback.print_exc()
        
        # 测试3: 默认参数
        print("\n📋 测试3: 默认参数")
        try:
            default_params = kdj._get_default_parameters()
            print(f"✅ 默认参数: {default_params}")
        except Exception as e:
            print(f"❌ 默认参数测试失败: {e}")
            traceback.print_exc()
        
        # 测试4: 参数设置
        print("\n📋 测试4: 参数设置")
        try:
            kdj.set_parameters(n=14, m1=5, m2=5)
            print("✅ 参数设置成功")
        except Exception as e:
            print(f"❌ 参数设置失败: {e}")
            traceback.print_exc()
        
        # 测试5: 空数据处理
        print("\n📋 测试5: 空数据处理")
        try:
            empty_result = kdj.calculate(pd.DataFrame())
            print(f"✅ 空数据处理成功，结果类型: {type(empty_result)}")
        except Exception as e:
            print(f"❌ 空数据处理失败: {e}")
            traceback.print_exc()
        
        # 测试6: 正常计算
        print("\n📋 测试6: 正常计算")
        try:
            # 创建测试数据
            dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
            np.random.seed(42)
            base_price = 100
            price_changes = np.random.normal(0.1, 2, 50)
            prices = [base_price]
            
            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change / 100)
                prices.append(max(new_price, 1))
            
            highs = [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices]
            lows = [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices]
            
            test_data = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': highs,
                'low': lows,
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, 50)
            })
            
            result = kdj.calculate(test_data)
            print(f"✅ 正常计算成功，结果形状: {result.shape}")
            print(f"   结果列: {list(result.columns)}")
            
            # 检查KDJ值是否合理
            if 'K' in result.columns and 'D' in result.columns and 'J' in result.columns:
                k_valid = result['K'].between(0, 100).sum() > len(result) * 0.8  # 80%的K值在合理范围
                d_valid = result['D'].between(0, 100).sum() > len(result) * 0.8  # 80%的D值在合理范围
                print(f"   K值合理性: {'✅' if k_valid else '❌'}")
                print(f"   D值合理性: {'✅' if d_valid else '❌'}")
            
        except Exception as e:
            print(f"❌ 正常计算失败: {e}")
            traceback.print_exc()
        
        # 测试7: 形态识别
        print("\n📋 测试7: 形态识别")
        try:
            patterns = kdj.get_patterns(test_data)
            print(f"✅ 形态识别成功，结果形状: {patterns.shape}")
        except Exception as e:
            print(f"❌ 形态识别失败: {e}")
            traceback.print_exc()
        
        print("\n🎉 KDJ修复测试完成!")
        return True
        
    except Exception as e:
        print(f"💥 测试过程中发生异常: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kdj_fixes()
    if success:
        print("\n✅ 测试成功")
    else:
        print("\n❌ 测试失败")
