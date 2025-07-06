#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试Click_house数据库连接
使用依赖注入架构
"""

import os
import sys
import pandas as pd
import datetime

# 将项目根目录添加到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_service

def test_connection_Connection():
    """测试数据库连接并执行简单查询"""
    try:
        # 通过容器获取数据访问接口
        print("正在通过依赖注入容器连接到数据库...")
        container = get_container()
        data_access = container.get_data_access()
        
        # 测试连接
        print("测试数据库连接...")
        if data_access.test_connection_Connection():
            print("数据库连接成功!")
        else:
            print("数据库连接失败!")
            return False
        
        # 获取股票列表
        print("获取股票列表...")
        stocks_df = data_access.get_stock_list(limit=10)
        if not stocks_df.empty:
            print(f"股票列表 (前10条):\n{stocks_df}")
        else:
            print("未找到股票数据")
        
        # 获取行业列表
        print("\n获取行业列表...")
        try:
            industry_df = data_access.get_industry_list()
            if not industry_df.empty:
                print(f"行业列表 (前5条):\n{industry_df.head()}")
                print(f"总共 {len(industry_df)} 个行业")
            else:
                print("未找到行业数据")
        except Exception as e:
            print(f"获取行业列表失败: {e}")
        
        # 测试获取股票数据
        print("\n测试获取股票数据...")
        try:
            stock_info WHERE 1=1 = data_access.get_stock_info(
                stock_code='000001',
                level='日线',
                limit=5
            )
            
            df = stock_info.to_dataframe()
            if not df.empty:
                print(f"000001 股票数据 (前5条):\n{df}")
            else:
                print("未找到000001的股票数据")
        except Exception as e:
            print(f"获取股票数据失败: {e}")
        
        print("\n数据库连接测试成功!")
        return True
    
    except Exception as e:
        print(f"连接测试失败: {e}")
        return False

if __name__ == "__main__":
    test_connection_Connection() 