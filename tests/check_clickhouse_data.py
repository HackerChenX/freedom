#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查ClickHouse数据库中的实际数据
"""

import sys
import traceback

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def check_clickhouse_data():
    """检查ClickHouse数据库中的实际数据"""
    print("🔍 检查ClickHouse数据库中的实际数据")
    print("=" * 50)
    
    try:
        from clickhouse_driver import Client
from db.sql_manager import SQLManager, QueryType
        
        # 连接ClickHouse
        client = Client(
            host='localhost',
            port=9000,
            database='stock',
            user='default',
            password='123456'
        )
        
        print("✅ ClickHouse连接成功")
        
        # 1. 检查数据库
        print("\n📊 检查数据库:")
        databases = client.execute("SHOW DATABASES")
        for db in databases:
            print(f"  - {db[0]}")
        
        # 2. 检查表
        print("\n📋 检查stock数据库中的表:")
        tables = client.execute("SHOW TABLES FROM stock")
        for table in tables:
            print(f"  - {table[0]}")
        
        # 3. 检查stock_info表结构
        if tables:
            table_name = tables[0][0]  # 使用第一个表
            print(f"\n🏗️ 检查表 '{table_name}' 的结构:")
            desc = client.execute(f"DESCRIBE {table_name}")
            for col in desc:
                print(f"  - {col[0]}: {col[1]}")
            
            # 4. 检查数据量
            print(f"\n📈 检查表 '{table_name}' 的数据量:")
            count = client.execute(f"SELECT COUNT(*) FROM {table_name}")
            print(f"  总记录数: {count[0][0]:,}")
            
            # 5. 检查数据样本
            print(f"\n📄 检查表 '{table_name}' 的数据样本:")
            sample = client.execute(f"SELECT * FROM {table_name} LIMIT 5")
            for i, row in enumerate(sample, 1):
                print(f"  样本{i}: {row}")
            
            # 6. 检查日期范围
            print(f"\n📅 检查表 '{table_name}' 的日期范围:")
            try:
                date_range = client.execute(f"SELECT MIN(date), MAX(date) FROM {table_name}")
                if date_range and date_range[0]:
                    min_date, max_date = date_range[0]
                    print(f"  日期范围: {min_date} 到 {max_date}")
            except Exception as e:
                print(f"  日期范围检查失败: {e}")
            
            # 7. 检查股票代码
            print(f"\n📊 检查表 '{table_name}' 的股票代码:")
            try:
                codes = client.execute(f"SELECT DISTINCT code FROM {table_name} LIMIT 10")
                print(f"  股票代码样本:")
                for code in codes:
                    print(f"    - {code[0]}")
            except Exception as e:
                print(f"  股票代码检查失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_clickhouse_data()
