#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试项目数据库模块连接
"""

import os
import sys
import pandas as pd

# 将项目根目录添加到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from db.clickhouse_db import get_clickhouse_db

def test_db_module():
    """测试使用项目数据库模块连接到ClickHouse"""
    try:
        # 获取数据库连接
        print("正在通过项目模块连接到ClickHouse数据库...")
        db = get_clickhouse_db()
        
        # 测试简单查询
        print("执行测试查询...")
        df = db.query("SELECT 1 AS test")
        print(f"查询结果: \n{df}")
        
        # 获取股票列表
        print("\n获取股票列表...")
        try:
            stocks_df = db.get_stock_list()
            print(f"股票列表前5行: \n{stocks_df.head()}")
            print(f"总共 {len(stocks_df)} 支股票")
        except Exception as e:
            print(f"获取股票列表出错: {e}")
        
        # 获取K线数据
        print("\n获取K线数据示例...")
        try:
            # 获取第一个股票代码
            stock_code = "000001.SZ"  # 尝试一个常见的股票代码
            
            import datetime
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=30)
            
            kline_df = db.get_kline_data(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                period='day'
            )
            
            if not kline_df.empty:
                print(f"{stock_code} K线数据前5行: \n{kline_df.head()}")
                print(f"总共 {len(kline_df)} 条K线记录")
            else:
                print(f"未找到 {stock_code} 的K线数据")
        except Exception as e:
            print(f"获取K线数据出错: {e}")
        
        print("\n数据库模块测试成功!")
        return True
    
    except Exception as e:
        print(f"数据库模块测试失败: {e}")
        return False

if __name__ == "__main__":
    test_db_module() 