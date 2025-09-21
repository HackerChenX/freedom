#!/usr/bin/env python3
"""
简化的ClickHouse模块测试
"""

import clickhouse_connect
from datetime import datetime

def test_clickhouse_connection():
    """测试ClickHouse连接和基本操作"""
    print("🧪 测试ClickHouse连接...")
    
    try:
        # 先连接到默认数据库
        client = clickhouse_connect.get_client(
            host='localhost',
            port=8123,
            username='default',
            password='123456'
        )

        print("✅ 连接成功")

        # 创建测试数据库
        try:
            client.command("CREATE DATABASE IF NOT EXISTS vnpy")
            print("✅ 数据库创建成功")
        except Exception as e:
            print(f"⚠️ 数据库创建警告: {e}")

        # 切换到vnpy数据库
        client.close()
        client = clickhouse_connect.get_client(
            host='localhost',
            port=8123,
            username='default',
            password='123456',
            database='vnpy'
        )
        
        # 创建测试表
        client.command("""
            CREATE TABLE IF NOT EXISTS test_bar_data (
                symbol String,
                exchange String,
                datetime DateTime,
                interval String,
                open_price Float64,
                high_price Float64,
                low_price Float64,
                close_price Float64,
                volume Float64,
                turnover Float64,
                open_interest Float64
            ) ENGINE = MergeTree()
            ORDER BY (symbol, exchange, interval, datetime)
        """)
        print("✅ 测试表创建成功")
        
        # 插入测试数据
        test_data = [
            ('000001', 'SSE', datetime.now(), '1m', 10.0, 10.5, 9.8, 10.2, 1000.0, 10200.0, 0.0),
            ('000002', 'SSE', datetime.now(), '1m', 20.0, 20.5, 19.8, 20.2, 2000.0, 40400.0, 0.0)
        ]
        
        client.insert('test_bar_data', test_data)
        print("✅ 测试数据插入成功")
        
        # 查询数据
        result = client.query("SELECT * FROM test_bar_data")
        print(f"✅ 查询成功，返回 {len(result.result_rows)} 条记录")
        
        for row in result.result_rows:
            print(f"   {row[0]} {row[1]} {row[2]} 价格:{row[4]}-{row[7]}")
        
        # 清理测试表
        client.command("DROP TABLE IF EXISTS test_bar_data")
        print("✅ 测试表清理完成")
        
        client.close()
        print("✅ 连接关闭")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clickhouse_connection()
    if success:
        print("\n🎉 ClickHouse基础功能测试通过！")
    else:
        print("\n💥 ClickHouse测试失败！")
