#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
ClickHouse连接测试总结
"""

from clickhouse_driver import Client
import pandas as pd

def test_clickhouse_connection():
    """测试ClickHouse连接并总结"""
    print("=="*30)
    print("ClickHouse连接测试总结")
    print("=="*30)
    
    # 连接配置
    config = {
        'host': 'localhost',
        'port': 9000,
        'user': 'default',
        'password': '123456',
        'database': 'stock'
    }
    
    try:
        print("\n1. 尝试连接到ClickHouse...")
        client = Client(**config)
        print("   ✓ 连接成功！")
        
        print("\n2. 测试基本查询...")
        result = client.execute("SELECT 1 AS test")
        print(f"   ✓ 查询成功: {result}")
        
        print("\n3. 获取数据库列表...")
        databases = client.execute("SHOW DATABASES")
        print("   ✓ 数据库列表:")
        for db in databases:
            print(f"     - {db[0]}")
        
        print("\n4. 检查stock数据库...")
        if 'stock' in [db[0] for db in databases]:
            print("   ✓ stock数据库存在")
            
            print("\n5. 获取stock数据库中的表...")
            tables = client.execute("SHOW TABLES FROM stock")
            if tables:
                print("   ✓ 表列表:")
                for table in tables:
                    print(f"     - {table[0]}")
                
                print("\n6. 表详情:")
                for table in tables:
                    table_name = table[0]
                    row_count = client.execute(f"SELECT COUNT(*) FROM stock.{table_name}")[0][0]
                    print(f"   ✓ 表 '{table_name}' 包含 {row_count} 行数据")
            else:
                print("   ✗ stock数据库中没有表")
        else:
            print("   ✗ 未找到stock数据库")
        
        print("\n总结: ClickHouse连接测试成功！可以在项目中正常使用。")
        print("请确保在config/user_config.json中配置了正确的密码: '123456'")
        print("=="*30)
        
    except Exception as e:
        print(f"\n   ✗ 测试失败: {e}")
        print("\n总结: ClickHouse连接测试失败。请检查Docker容器是否正在运行，以及连接配置是否正确。")
        print("=="*30)

if __name__ == "__main__":
    test_clickhouse_connection() 