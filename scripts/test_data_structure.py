#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
检查数据结构
验证DataFrame的索引和日期列格式
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.unified_data_manager import get_unified_data_manager
import pandas as pd

def check_data_structure():
    print("=== 检查数据结构 ===")
    
    try:
        # 1. 获取数据
        print("\n1. 获取测试数据...")
        data_manager = get_unified_data_manager()
        
        stock_data = data_manager.get_stock_data(
            stock_code='000001',
            end_date='2025-05-23',
            lookback_days=10
        )
        
        print(f"✅ 获取到 {len(stock_data)} 条数据")
        
        # 2. 检查数据结构
        print("\n2. 检查数据结构...")
        print(f"数据类型: {type(stock_data)}")
        print(f"数据形状: {stock_data.shape}")
        print(f"索引类型: {type(stock_data.index)}")
        print(f"索引内容: {stock_data.index}")
        print(f"列名: {list(stock_data.columns)}")
        
        # 3. 检查日期列
        print("\n3. 检查日期列...")
        if 'date' in stock_data.columns:
            print(f"日期列类型: {stock_data['date'].dtype}")
            print(f"日期列样本: {stock_data['date'].head()}")
            print(f"最新日期: {stock_data['date'].iloc[-1]}")
            print(f"最早日期: {stock_data['date'].iloc[0]}")
        
        # 4. 检查价格数据
        print("\n4. 检查价格数据...")
        if 'close' in stock_data.columns:
            print(f"收盘价类型: {stock_data['close'].dtype}")
            print(f"收盘价样本: {stock_data['close'].head()}")
            print(f"最新收盘价: {stock_data['close'].iloc[-1]}")
        
        # 5. 模拟条件评估器的数据访问
        print("\n5. 模拟条件评估器的数据访问...")
        test_date = '2025-05-23'
        
        # 方法1：按索引查找（当前的错误方法）
        print(f"按索引查找日期 {test_date}:")
        try:
            if test_date in stock_data.index:
                print(f"  索引中找到日期: {test_date}")
            else:
                print(f"  索引中未找到日期: {test_date}")
        except Exception as e:
            print(f"  索引查找出错: {e}")
        
        # 方法2：按日期列查找（正确方法）
        print(f"按日期列查找日期 {test_date}:")
        try:
            matching_rows = stock_data[stock_data['date'] == test_date]
            if not matching_rows.empty:
                close_price = matching_rows['close'].iloc[0]
                print(f"  找到匹配行，收盘价: {close_price}")
            else:
                print(f"  未找到匹配的日期")
                # 找最接近的日期
                stock_data_sorted = stock_data.sort_values('date')
                earlier_dates = stock_data_sorted[stock_data_sorted['date'] <= test_date]
                if not earlier_dates.empty:
                    latest_date = earlier_dates['date'].iloc[-1]
                    close_price = earlier_dates[earlier_dates['date'] == latest_date]['close'].iloc[0]
                    print(f"  找到最接近的日期: {latest_date}, 收盘价: {close_price}")
        except Exception as e:
            print(f"  日期列查找出错: {e}")
        
        print("\n=== 数据结构检查完成 ===")
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_data_structure() 