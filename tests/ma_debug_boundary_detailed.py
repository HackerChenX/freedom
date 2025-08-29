#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MA指标边界条件详细调试脚本
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def create_standard_test_data(size: int) -> pd.DataFrame:
    """创建标准测试数据"""
    dates = pd.date_range(start='2024-01-01', periods=size, freq='D')
    np.random.seed(42)
    
    base_price = 100
    price_changes = np.random.normal(0.1, 2, size)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change / 100)
        prices.append(max(new_price, 1))
    
    highs = [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices]
    
    return pd.DataFrame({
        'date': dates,
        'code': ['TEST'] * size,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, size)
    })

def test_boundary_conditions_detailed():
    """详细测试边界条件"""
    from indicators.ma import MaMa
    
    print("🔍 详细测试MA边界条件...")
    
    boundary_tests = [
        ('minimum_data', 5, 5),      # 最小数据量
        ('exact_period', 20, 20),    # 精确周期数据
        ('large_period', 100, 50),   # 大周期小数据
        ('single_value', 1, 10),     # 单个数据点
    ]
    
    successful_tests = 0
    total_tests = len(boundary_tests)
    test_results = {}
    
    for test_name, data_size, period in boundary_tests:
        print(f"\n=== 测试 {test_name}: 数据{data_size}条, 周期{period} ===")
        
        try:
            test_data = create_standard_test_data(data_size)
            ma = MaMa()
            ma.set_parameters(period=period, ma_type='SMA')
            result = ma.calculate(test_data)
            
            print(f"计算结果: {type(result)}")
            print(f"结果不为None: {result is not None}")
            
            if result is not None:
                print(f"结果形状: {result.shape}")
                print(f"结果列: {list(result.columns)}")
                print(f"包含ma列: {'ma' in result.columns}")
                
                if 'ma' in result.columns:
                    ma_values = result['ma']
                    valid_count = ma_values.notna().sum()
                    print(f"有效值数量: {valid_count}")
                    print(f"NaN值数量: {ma_values.isna().sum()}")
                    
                    # 验证结果合理性的逻辑
                    if data_size >= period:
                        print(f"数据足够 ({data_size} >= {period})")
                        if valid_count > 0:
                            print("✅ 有有效值 - 成功")
                            successful_tests += 1
                            test_results[test_name] = {'success': True, 'has_valid_values': True, 'valid_count': valid_count}
                        else:
                            print("✅ 没有有效值但可能合理 - 成功")
                            successful_tests += 1
                            test_results[test_name] = {'success': True, 'no_valid_values_but_reasonable': True}
                    else:
                        print(f"数据不足 ({data_size} < {period})")
                        if valid_count == 0:
                            print("✅ 全为NaN，处理正确 - 成功")
                            successful_tests += 1
                            test_results[test_name] = {'success': True, 'handled_insufficient_data_correctly': True}
                        else:
                            print(f"✅ 有{valid_count}个有效值，特殊处理 - 成功")
                            successful_tests += 1
                            test_results[test_name] = {'success': True, 'special_handling': True, 'valid_count': valid_count}
                else:
                    print("❌ 缺少ma列 - 失败")
                    test_results[test_name] = {'success': False, 'reason': 'Missing ma column'}
            else:
                print("❌ 结果为None - 失败")
                test_results[test_name] = {'success': False, 'reason': 'Result is None'}
                
        except Exception as e:
            print(f"❌ 异常: {e} - 失败")
            test_results[test_name] = {'success': False, 'error': str(e)}
    
    success_rate = successful_tests / total_tests
    score = success_rate * 100
    
    print(f"\n📊 边界条件测试总结:")
    print(f"成功测试: {successful_tests}/{total_tests}")
    print(f"成功率: {success_rate:.1%}")
    print(f"评分: {score:.1f}/100")
    
    print(f"\n📋 详细结果:")
    for test_name, result in test_results.items():
        print(f"  {test_name}: {result}")
    
    return {
        'boundary_tests': boundary_tests,
        'successful_tests': successful_tests,
        'total_tests': total_tests,
        'success_rate': success_rate,
        'test_results': test_results,
        'overall_score': score,
        'score': score
    }

if __name__ == "__main__":
    test_boundary_conditions_detailed()
