#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查ClickHouse数据库中的表
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def check_clickhouse_tables():
    """检查ClickHouse数据库中的表"""
    print("🔍 检查ClickHouse数据库中的表...")
    
    try:
        from clickhouse_driver import Client
        
        client = Client(
            host='localhost',
            port=9000,
            user='default',
            password='123456',
            database='stock'
        )
        
        # 查看所有表
        print("📊 查看所有表...")
        tables_result = client.execute("SHOW TABLES")
        
        if tables_result:
            print(f"✅ 找到{len(tables_result)}个表:")
            for table in tables_result:
                print(f"  - {table[0]}")
            
            # 如果有表，查看第一个表的结构
            if len(tables_result) > 0:
                first_table = tables_result[0][0]
                print(f"\n📋 查看表 '{first_table}' 的结构:")
                
                desc_result = client.execute(f"DESCRIBE {first_table}")
                if desc_result:
                    print("列信息:")
                    for col in desc_result:
                        print(f"  - {col[0]}: {col[1]}")
                
                # 查看数据样本
                print(f"\n📊 查看表 '{first_table}' 的数据样本:")
                sample_result = client.execute(f"SELECT * FROM {first_table} LIMIT 5")
                if sample_result:
                    print(f"样本数据 ({len(sample_result)}条):")
                    for i, row in enumerate(sample_result):
                        print(f"  行{i+1}: {row}")
                else:
                    print("  表为空")
            
            return True, tables_result
        else:
            print("⚠️ 数据库中没有表")
            return False, []
            
    except Exception as e:
        print(f"❌ 检查表失败: {e}")
        return False, []

if __name__ == "__main__":
    success, tables = check_clickhouse_tables()
    if success:
        print("🎉 ClickHouse表检查完成！")
    else:
        print("💥 ClickHouse表检查失败！")
