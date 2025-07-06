#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试条件评估器
验证条件评估器是否能正确处理不同类型的条件
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.unified_data_manager import get_unified_data_manager
from strategy.strategy_condition_evaluator import Strategy_condition_evaluator
import pandas as pd

def test_condition_evaluation():
    print("=== 条件评估器测试 ===")
    
    try:
        # 1. 初始化组件
        print("\n1. 初始化组件...")
        data_manager = get_unified_data_manager()
        evaluator = Strategy_condition_evaluator()
        print("✅ 组件初始化成功")
        
        # 2. 获取测试股票数据
        print("\n2. 获取测试数据...")
        test_stock = '000001'
        test_date = '2025-05-23'
        
        stock_data = data_manager.get_stock_data(
            stock_code=test_stock,
            end_date=test_date,
            lookback_days=60
        )
        
        print(f"✅ 获取到 {len(stock_data)} 条数据")
        print(f"✅ 数据类型: {type(stock_data)}")
        print(f"✅ 数据列: {list(stock_data.columns)}")
        
        # 3. 测试不同类型的条件
        test_conditions = [
            # 价格条件
            {
                'type': 'price',
                'field': 'close',
                'operator': '>',
                'value': 10.0,
                'description': '价格条件：收盘价大于10元'
            },
            
            # 成交量条件
            {
                'type': 'volume',
                'operator': '>',
                'value': 100000,
                'description': '成交量条件：成交量大于10万'
            },
            
            # 指标条件（RSI）
            {
                'type': 'indicator',
                'indicator_id': 'RSI',
                'period': 'daily',
                'condition': 'RSI < 70',
                'description': 'RSI指标小于70'
            },
            
            # 指标条件（MA）
            {
                'type': 'indicator',
                'indicator_id': 'MA',
                'period': 'daily',
                'condition': 'close > 0',
                'description': '收盘价大于0'
            }
        ]
        
        print("\n3. 测试条件评估...")
        for i, condition in enumerate(test_conditions, 1):
            print(f"\n--- 测试条件 {i}: {condition['description']} ---")
            print(f"条件配置: {condition}")
            
            try:
                result = evaluator.evaluate_condition(condition, stock_data, test_date)
                print(f"✅ 评估结果: {result}")
                
                # 如果是指标条件，显示更多调试信息
                if condition['type'] == 'indicator':
                    indicator_name = condition.get('indicator_id', '')
                    print(f"   指标名称: {indicator_name}")
                    
            except Exception as e:
                print(f"❌ 评估失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 4. 测试复合条件
        print("\n4. 测试复合条件...")
        complex_conditions = [
            {
                'type': 'indicator',
                'indicator_id': 'RSI',
                'period': 'daily',
                'condition': 'RSI < 80',
                'description': 'RSI指标小于80'
            },
            {
                'type': 'logic',
                'value': 'AND'
            },
            {
                'type': 'indicator',
                'indicator_id': 'RSI',
                'period': 'daily',
                'condition': 'RSI > 20',
                'description': 'RSI指标大于20'
            }
        ]
        
        try:
            result = evaluator.evaluate_conditions(complex_conditions, stock_data, test_date)
            print(f"✅ 复合条件评估结果: {result}")
        except Exception as e:
            print(f"❌ 复合条件评估失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n=== 条件评估器测试完成 ===")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_condition_evaluation() 