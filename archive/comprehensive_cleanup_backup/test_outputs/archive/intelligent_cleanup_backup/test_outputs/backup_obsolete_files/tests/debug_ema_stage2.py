#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试EMA指标阶段2基础功能验证问题
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def debug_ema_stage2():
    """调试EMA指标阶段2基础功能验证"""
    print("🔍 调试EMA指标阶段2基础功能验证...")
    
    try:
        from indicators.ema import EmaEma
        ema = EmaEma()
        
        print("📊 测试1: 参数管理")
        # 测试参数管理
        print(f"  - 有set_parameters方法: {hasattr(ema, 'set_parameters')}")
        print(f"  - 有_get_default_parameters方法: {hasattr(ema, '_get_default_parameters')}")
        print(f"  - 有minimum_periods属性: {hasattr(ema, 'minimum_periods')}")
        
        if hasattr(ema, '_get_default_parameters'):
            default_params = ema._get_default_parameters()
            print(f"  - 默认参数类型: {type(default_params)}")
            print(f"  - 默认参数内容: {default_params}")
        
        if hasattr(ema, 'minimum_periods'):
            min_periods = ema.minimum_periods
            print(f"  - minimum_periods类型: {type(min_periods)}")
            print(f"  - minimum_periods值: {min_periods}")
        
        print("\n📊 测试2: 错误处理")
        # 测试错误处理
        error_scenarios = [
            ('empty_dataframe', pd.DataFrame()),
            ('invalid_columns', pd.DataFrame({'invalid': [1, 2, 3]})),
            ('insufficient_data', pd.DataFrame({'close': [100, 101]})),
            ('nan_values', pd.DataFrame({'close': [100, np.nan, 102, np.nan, 104]}))
        ]
        
        handled_errors = 0
        for scenario_name, test_input in error_scenarios:
            try:
                result = ema.calculate(test_input)
                if result is not None:
                    print(f"  - {scenario_name}: 处理成功（返回结果）")
                    handled_errors += 1
                else:
                    print(f"  - {scenario_name}: 处理成功（返回None）")
                    handled_errors += 1
            except Exception as e:
                if any(keyword in str(e).lower() for keyword in ['数据', '列', '长度', 'data', 'column']):
                    print(f"  - {scenario_name}: 处理成功（合理异常: {e}）")
                    handled_errors += 1
                else:
                    print(f"  - {scenario_name}: 处理失败（异常: {e}）")
        
        print(f"  - 错误处理成功率: {handled_errors}/{len(error_scenarios)} = {handled_errors/len(error_scenarios)*100:.1f}%")
        
        print("\n📊 测试3: 边界条件")
        # 测试边界条件
        boundary_tests = []
        
        # 最小数据量
        min_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'code': ['TEST'] * 5,
            'close': [100, 101, 102, 103, 104],
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'volume': [1000000] * 5
        })
        
        try:
            result = ema.calculate(min_data)
            boundary_tests.append(result is not None)
            print(f"  - 最小数据量测试: {'通过' if result is not None else '失败'}")
        except Exception as e:
            boundary_tests.append(False)
            print(f"  - 最小数据量测试: 失败 ({e})")
        
        # 大数据量（简化版）
        large_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=1000),
            'code': ['TEST'] * 1000,
            'close': [100 + i * 0.1 for i in range(1000)],
            'open': [100 + i * 0.1 for i in range(1000)],
            'high': [101 + i * 0.1 for i in range(1000)],
            'low': [99 + i * 0.1 for i in range(1000)],
            'volume': [1000000] * 1000
        })
        
        try:
            result = ema.calculate(large_data)
            boundary_tests.append(result is not None and len(result) > 0)
            print(f"  - 大数据量测试: {'通过' if result is not None and len(result) > 0 else '失败'}")
        except Exception as e:
            boundary_tests.append(False)
            print(f"  - 大数据量测试: 失败 ({e})")
        
        print(f"  - 边界条件成功率: {sum(boundary_tests)}/{len(boundary_tests)} = {sum(boundary_tests)/len(boundary_tests)*100:.1f}%")
        
        print("\n📊 测试4: 数据类型处理")
        # 测试数据类型处理
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'code': ['TEST'] * 50,
            'close': [100 + i * 0.5 for i in range(50)],
            'open': [100 + i * 0.5 for i in range(50)],
            'high': [101 + i * 0.5 for i in range(50)],
            'low': [99 + i * 0.5 for i in range(50)],
            'volume': [1000000] * 50
        })
        
        type_tests = []
        
        # 整数价格
        int_data = test_data.copy()
        int_data['close'] = int_data['close'].astype(int)
        try:
            result = ema.calculate(int_data)
            type_tests.append(result is not None)
            print(f"  - 整数价格测试: {'通过' if result is not None else '失败'}")
        except Exception as e:
            type_tests.append(False)
            print(f"  - 整数价格测试: 失败 ({e})")
        
        # 浮点价格
        float_data = test_data.copy()
        float_data['close'] = float_data['close'].astype(float)
        try:
            result = ema.calculate(float_data)
            type_tests.append(result is not None)
            print(f"  - 浮点价格测试: {'通过' if result is not None else '失败'}")
        except Exception as e:
            type_tests.append(False)
            print(f"  - 浮点价格测试: 失败 ({e})")
        
        print(f"  - 数据类型处理成功率: {sum(type_tests)}/{len(type_tests)} = {sum(type_tests)/len(type_tests)*100:.1f}%")
        
        # 计算总体评分
        param_score = 100 if all([
            hasattr(ema, 'set_parameters'),
            hasattr(ema, '_get_default_parameters'),
            hasattr(ema, 'minimum_periods'),
            isinstance(ema._get_default_parameters(), dict) if hasattr(ema, '_get_default_parameters') else False,
            isinstance(ema.minimum_periods, int) if hasattr(ema, 'minimum_periods') else False
        ]) else 0
        
        error_score = (handled_errors / len(error_scenarios)) * 100
        boundary_score = (sum(boundary_tests) / len(boundary_tests)) * 100
        type_score = (sum(type_tests) / len(type_tests)) * 100
        
        overall_score = (param_score + error_score + boundary_score + type_score) / 4
        
        print(f"\n📊 评分总结:")
        print(f"  - 参数管理: {param_score:.1f}/100")
        print(f"  - 错误处理: {error_score:.1f}/100")
        print(f"  - 边界条件: {boundary_score:.1f}/100")
        print(f"  - 数据类型: {type_score:.1f}/100")
        print(f"  - 总体评分: {overall_score:.1f}/100")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ema_stage2()
