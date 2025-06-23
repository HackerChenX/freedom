#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
简单的ClickHouse连接测试
"""

from clickhouse_driver import Client
import pandas as pd
import sys

# 数据库配置选项
configs = [
    # 默认配置
    {
        'host': 'localhost',
        'port': 9000,
        'user': 'default',
        'password': '123456',  # 从Docker环境变量中获取的密码
        'database': 'stock'
    },
    # 不指定数据库名
    {
        'host': 'localhost',
        'port': 9000,
        'user': 'default',
        'password': '123456'  # 从Docker环境变量中获取的密码
    },
    # 使用默认系统数据库
    {
        'host': 'localhost',
        'port': 9000,
        'user': 'default',
        'password': '123456',  # 从Docker环境变量中获取的密码
        'database': 'default'
    }
]

def test_connection(config):
    """测试与ClickHouse的连接"""
    try:
        print(f"尝试使用配置: {config}")
        # 创建客户端
        print("正在连接到ClickHouse数据库...")
        client = Client(**config)
        
        # 测试简单查询
        print("执行测试查询...")
        result = client.execute("SELECT 1")
        print(f"查询结果: {result}")
        
        # 获取数据库列表
        print("获取数据库列表...")
        databases = client.execute("SHOW DATABASES")
        print("可用的数据库:")
        for db in databases:
            print(f"- {db[0]}")
        
        # 如果指定了数据库，尝试获取表列表
        if 'database' in config:
            print(f"\n获取数据库 '{config['database']}' 的表列表...")
            try:
                tables = client.execute(f"SHOW TABLES FROM {config['database']}")
                print(f"数据库 '{config['database']}' 中的表:")
                for table in tables:
                    print(f"- {table[0]}")
                
                # 如果有表，查询第一个表的结构
                if tables:
                    first_table = tables[0][0]
                    print(f"\n获取表 '{first_table}' 的结构...")
                    structure = client.execute(f"DESCRIBE TABLE {config['database']}.{first_table}")
                    
                    # 转换为DataFrame以更好地显示
                    structure_df = pd.DataFrame(structure, columns=['name', 'type', 'default_type', 'default_expression'])
                    print("表结构:")
                    print(structure_df)
                    
                    # 获取第一个表的行数
                    count = client.execute(f"SELECT COUNT(*) FROM {config['database']}.{first_table}")
                    print(f"\n表 '{first_table}' 中的行数: {count[0][0]}")
            except Exception as e:
                print(f"获取表信息时出错: {e}")
        
        print("\nClickHouse连接测试成功!")
        return True
    
    except Exception as e:
        print(f"连接测试失败: {e}")
        return False

if __name__ == "__main__":
    success = False
    
    # 尝试所有配置
    for config in configs:
        print("\n" + "="*50)
        if test_connection(config):
            success = True
            break
        print("="*50 + "\n")
    
    if not success:
        print("\n所有连接尝试均失败")
        sys.exit(1) 