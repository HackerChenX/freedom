#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试项目数据库模块连接
使用依赖注入架构
"""

import os
import sys
import pandas as pd

# 将项目根目录添加到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_service

def test_db_module():
    """测试使用项目数据库模块连接到ClickHouse"""
    try:
        # 通过容器获取数据访问接口
        print("正在通过依赖注入容器连接到数据库...")
        container = get_container()
        data_access = container.get_data_access()
        
        # 测试连接
        print("测试数据库连接...")
        if data_access.test_connection():
            print("数据库连接成功!")
        else:
            print("数据库连接失败!")
            return False
        
        # 获取股票列表
        print("\n获取股票列表...")
        try:
            stocks_df = data_access.get_stock_list()
            print(f"股票列表前5行: \n{stocks_df.head()}")
            print(f"总共 {len(stocks_df)} 支股票")
        except Exception as e:
            print(f"获取股票列表出错: {e}")

        # 测试get_stock_info方法
        print("\n测试get_stock_info方法...")
        try:
            stock_info WHERE 1=1 = data_access.get_stock_info(stock_code='000001', limit=5)
            print(f"股票信息类型: {type(stock_info)}")
            if hasattr(stock_info, 'to_dataframe'):
                df = stock_info.to_dataframe()
                print(f"数据框形状: {df.shape}")
                if not df.empty:
                    print(f"前5行数据: \n{df.head()}")
        except Exception as e:
            print(f"获取股票信息出错: {e}")
        
        # 获取K线数据
        print("\n获取K线数据示例...")
        try:
            # 获取第一个股票代码
            stock_code = "000001"  # 尝试一个常见的股票代码
            
            import datetime
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
            
            stock_info WHERE 1=1 = data_access.get_stock_info(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                level='日线',
                limit=5
            )
            
            kline_df = stock_info.to_dataframe()
            
            if not kline_df.empty:
                print(f"{stock_code} K线数据前5行: \n{kline_df.head()}")
                print(f"总共 {len(kline_df)} 条K线记录")
            else:
                print(f"未找到 {stock_code} 的K线数据")
        except Exception as e:
            print(f"获取K线数据出错: {e}")
        
        # 测试行业列表
        print("\n测试获取行业列表...")
        try:
            industry_df = data_access.get_industry_list()
            if not industry_df.empty:
                print(f"行业列表前5行: \n{industry_df.head()}")
                print(f"总共 {len(industry_df)} 个行业")
            else:
                print("未找到行业数据")
        except Exception as e:
            print(f"获取行业列表出错: {e}")
        
        print("\n数据库模块测试成功!")
        return True
    
    except Exception as e:
        print(f"数据库模块测试失败: {e}")
        return False

if __name__ == "__main__":
    test_db_module() 