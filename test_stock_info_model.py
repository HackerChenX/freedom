#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试StockInfo模型和get_stock_info方法

验证新的StockInfo对象模型与数据库层的集成
"""

import sys
import os
import datetime
import pandas as pd
from db.clickhouse_db import get_clickhouse_db
from models.stock_info import StockInfo
from enums.period import Period
from db.data_manager import DataManager

def test_stock_info_model():
    """测试StockInfo模型基本功能"""
    
    # 测试空构造
    empty_stock = StockInfo()
    print(f"空StockInfo对象: {empty_stock}")
    
    # 测试从字典构造
    data_dict = {
        'code': '000001',
        'name': '平安银行',
        'date': '2023-01-01',
        'level': '日线',
        'open': 10.5,
        'high': 11.2,
        'low': 10.1,
        'close': 10.8,
        'volume': 123456,
        'turnover_rate': 2.5,
        'price_change': 0.3,
        'price_range': 1.1,
        'industry': '银行',
        'datetime': '2023-01-01 15:00:00',
        'seq': 1
    }
    
    stock = StockInfo(data_dict)
    print(f"从字典构造的StockInfo对象: {stock}")
    print(f"股票代码: {stock.code}")
    print(f"股票名称: {stock.name}")
    print(f"收盘价: {stock.close}")
    print(f"日期时间: {stock.datetime_value}")
    
    # 测试属性修改
    stock.close = 11.0
    print(f"修改后的收盘价: {stock.close}")
    
    # 测试转换回字典
    stock_dict = stock.to_dict()
    print(f"转换回字典: {stock_dict}")
    
    # 测试从DataFrame构造
    df = pd.DataFrame([data_dict, {**data_dict, 'date': '2023-01-02', 'close': 11.2}])
    stock_collection = StockInfo(df)
    print(f"从DataFrame构造的StockInfo集合: {stock_collection}")
    print(f"集合大小: {len(stock_collection)}")
    print(f"第一条数据: {stock_collection[0]}")
    print(f"第二条数据收盘价: {stock_collection[1].close}")
    
    # 测试转换为DataFrame
    df_out = stock_collection.to_dataframe()
    print(f"转换为DataFrame，形状: {df_out.shape}")
    print(f"DataFrame前2行:\n{df_out.head(2)}")
    
    # 测试迭代
    print("迭代测试:")
    for i, item in enumerate(stock_collection):
        print(f"  项目{i+1}: {item.date}, 收盘价: {item.close}")

def test_get_stock_info():
    """测试数据库get_stock_info方法返回StockInfo对象"""
    
    # 获取数据库连接
    db = get_clickhouse_db()
    
    # 测试获取股票数据
    stock_code = '000001'  # 平安银行
    level = Period.DAILY
    start_date = '20230101'
    end_date = '20230110'
    
    print(f"获取股票 {stock_code} 的 {level.value} 数据, 从 {start_date} 到 {end_date}")
    stock_info = db.get_stock_info(stock_code, level, start_date, end_date)
    
    print(f"返回类型: {type(stock_info)}")
    print(f"返回对象: {stock_info}")
    
    if stock_info.is_collection:
        print(f"数据条数: {len(stock_info)}")
        if len(stock_info) > 0:
            first_item = stock_info[0]
            print(f"首条数据: {first_item}")
            print(f"  股票代码: {first_item.code}")
            print(f"  股票名称: {first_item.name}")
            print(f"  日期: {first_item.date}")
            print(f"  K线周期: {first_item.level}")
            print(f"  开盘价: {first_item.open}")
            print(f"  最高价: {first_item.high}")
            print(f"  最低价: {first_item.low}")
            print(f"  收盘价: {first_item.close}")
            print(f"  成交量: {first_item.volume}")
            print(f"  日期时间: {first_item.datetime_value}")
        
        # 测试转换为DataFrame
        df = stock_info.to_dataframe()
        print(f"DataFrame形状: {df.shape}")
        print(f"DataFrame前3行:\n{df.head(3)}")
    else:
        print("返回的是单条数据对象")
        print(f"  股票代码: {stock_info.code}")
        print(f"  股票名称: {stock_info.name}")
        print(f"  K线周期: {stock_info.level}")

def test_data_manager():
    """测试DataManager.get_stock_info方法返回StockInfo对象"""
    
    # 获取数据管理器
    data_manager = DataManager()
    
    # 测试获取股票数据
    stock_code = '000001'  # 平安银行
    level = 'day'
    start_date = '2023-01-01'
    end_date = '2023-01-10'
    
    print(f"通过DataManager获取股票 {stock_code} 的 {level} 数据, 从 {start_date} 到 {end_date}")
    stock_info = data_manager.get_stock_info(stock_code, level, start_date, end_date)
    
    print(f"返回类型: {type(stock_info)}")
    print(f"返回对象: {stock_info}")
    
    if hasattr(stock_info, 'is_collection') and stock_info.is_collection:
        print(f"数据条数: {len(stock_info)}")
        if len(stock_info) > 0:
            first_item = stock_info[0]
            print(f"首条数据: {first_item}")
            print(f"  股票代码: {first_item.code}")
            print(f"  股票名称: {first_item.name}")
            print(f"  收盘价: {first_item.close}")
    else:
        print("返回的是单条数据对象或DataFrame")
        if isinstance(stock_info, StockInfo):
            print(f"  股票代码: {stock_info.code}")
            print(f"  股票名称: {stock_info.name}")
            if hasattr(stock_info, 'close'):
                print(f"  收盘价: {stock_info.close}")

def main():
    """主函数"""
    print("\n===== 测试StockInfo模型 =====")
    test_stock_info_model()
    
    print("\n===== 测试get_stock_info方法 =====")
    test_get_stock_info()
    
    print("\n===== 测试DataManager.get_stock_info方法 =====")
    test_data_manager()

if __name__ == "__main__":
    main() 