#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试ClickHouse数据库连接
"""

import os
import sys
import pandas as pd
import datetime

# 将项目根目录添加到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from db.clickhouse_db import get_clickhouse_db

def test_connection():
    """测试数据库连接并执行简单查询"""
    try:
        # 获取数据库连接
        print("正在连接到ClickHouse数据库...")
        db = get_clickhouse_db()
        
        # 测试简单查询
        print("执行测试查询...")
        df = db.query("SELECT 1")
        print(f"查询结果: {df}")
        
        # 获取表列表
        print("获取数据库表列表...")
        tables_df = db.query("SHOW TABLES")
        print(f"数据库中的表:\n{tables_df}")
        
        # 如果有表，查询第一个表的结构
        if not tables_df.empty:
            first_table = tables_df.iloc[0, 0]
            print(f"\n获取表 '{first_table}' 的结构...")
            structure_df = db.query(f"DESCRIBE TABLE {first_table}")
            print(f"表结构:\n{structure_df}")
            
            # 获取第一个表的行数
            count_df = db.query(f"SELECT COUNT(*) FROM {first_table}")
            print(f"\n表 '{first_table}' 中的行数: {count_df.iloc[0, 0]}")
            
            # 获取第一个表的前5行数据
            print(f"\n表 '{first_table}' 的前5行数据:")
            data_df = db.query(f"SELECT * FROM {first_table} LIMIT 5")
            print(data_df)
        
        print("\nClickHouse连接测试成功!")
        return True
    
    except Exception as e:
        print(f"连接测试失败: {e}")
        return False

if __name__ == "__main__":
    test_connection() 