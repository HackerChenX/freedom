#!/usr/bin/env python
"""
ClickHouse连接测试脚本
用于验证修复后的连接参数是否正确
"""

import sys
import os

# 添加项目路径
project_root = os.path.join(os.path.dirname(__file__), "../../..")
sys.path.insert(0, os.path.join(project_root, "Core_Framework/vnpy"))

try:
    import clickhouse_connect
    
    print("🔧 测试ClickHouse连接参数...")
    
    # 测试基本连接
    client = clickhouse_connect.get_client(
        host="localhost",
        port=8123,
        username="default",
        password="",
        database="stock"
    )
    
    # 测试查询
    result = client.query("SELECT 1")
    print("✅ ClickHouse连接测试成功!")
    print(f"📊 查询结果: {result.result_rows}")
    
    # 测试表存在
    tables = client.query("SHOW TABLES")
    table_list = [row[0] for row in tables.result_rows]
    print(f"📋 数据库中的表: {table_list}")
    
    if 'stock_info' in table_list:
        print("✅ stock_info表存在")
        
        # 测试数据查询
        count_result = client.query("SELECT COUNT(*) FROM stock_info LIMIT 1")
        count = count_result.result_rows[0][0] if count_result.result_rows else 0
        print(f"📊 stock_info表记录数: {count}")
    else:
        print("⚠️ stock_info表不存在")
    
    client.close()
    print("🎉 所有测试通过!")
    
except Exception as e:
    print(f"❌ 连接测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
