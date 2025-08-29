#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试ClickHouse连接
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def test_clickhouse_connection():
    """测试ClickHouse连接"""
    print("🔍 测试ClickHouse连接...")
    
    try:
        # 方法1: 尝试简单客户端
        print("📊 尝试简单ClickHouse客户端...")
        from db.simple_clickhouse_client import SimpleClickHouseClient
        
        client = SimpleClickHouseClient()
        
        # 测试连接
        test_query = "SELECT 1 as test"
        result = client.query_dataframe(test_query)
        
        if result is not None and not result.empty:
            print("✅ 简单客户端连接成功")
            
            # 测试股票数据查询
            stock_query = """
            SELECT 
                date,
                code,
                close,
                volume
            FROM stock_daily_data 
            WHERE date >= '2024-01-01' 
            AND volume > 0 
            AND close > 0
            ORDER BY date DESC, code
            LIMIT 10
            """
            
            stock_data = client.query_dataframe(stock_query)
            
            if stock_data is not None and not stock_data.empty:
                print(f"✅ 成功获取{len(stock_data)}条股票数据")
                print("📊 数据样本:")
                print(stock_data.head())
                return True, stock_data
            else:
                print("⚠️ 股票数据查询返回空结果")
                return False, None
        else:
            print("❌ 简单客户端连接失败")
            
    except Exception as e:
        print(f"❌ 简单客户端异常: {e}")
    
    try:
        # 方法2: 尝试主要数据库模块
        print("📊 尝试主要ClickHouse数据库模块...")
        from db.clickhouse_db import ClickHouseDb

        db = ClickHouseDb()
        
        # 测试连接
        test_query = "SELECT 1 as test"
        result = db.query_dataframe(test_query)
        
        if result is not None and not result.empty:
            print("✅ 主要数据库模块连接成功")
            
            # 测试股票数据查询
            stock_query = """
            SELECT 
                date,
                code,
                close,
                volume
            FROM stock_daily_data 
            WHERE date >= '2024-01-01' 
            AND volume > 0 
            AND close > 0
            ORDER BY date DESC, code
            LIMIT 10
            """
            
            stock_data = db.query_dataframe(stock_query)
            
            if stock_data is not None and not stock_data.empty:
                print(f"✅ 成功获取{len(stock_data)}条股票数据")
                print("📊 数据样本:")
                print(stock_data.head())
                return True, stock_data
            else:
                print("⚠️ 股票数据查询返回空结果")
                return False, None
        else:
            print("❌ 主要数据库模块连接失败")
            
    except Exception as e:
        print(f"❌ 主要数据库模块异常: {e}")
    
    try:
        # 方法3: 直接使用clickhouse_driver
        print("📊 尝试直接使用clickhouse_driver...")
        from clickhouse_driver import Client

        client = Client(
            host='localhost',
            port=9000,
            user='default',
            password='123456',
            database='stock'
        )
        
        # 测试连接
        result = client.execute("SELECT 1 as test")

        if result:
            print("✅ 直接连接成功")

            # 测试股票数据查询
            stock_result = client.execute("""
            SELECT
                date,
                code,
                close,
                volume
            FROM stock_daily_data
            WHERE date >= '2024-01-01'
            AND volume > 0
            AND close > 0
            ORDER BY date DESC, code
            LIMIT 10
            """)

            if stock_result:
                print(f"✅ 成功获取{len(stock_result)}条股票数据")
                print("📊 数据样本:")

                # 转换为DataFrame
                import pandas as pd
                columns = ['date', 'code', 'close', 'volume']
                df = pd.DataFrame(stock_result, columns=columns)
                print(df.head())
                return True, df
            else:
                print("⚠️ 股票数据查询返回空结果")
                return False, None
        else:
            print("❌ 直接连接失败")
            
    except Exception as e:
        print(f"❌ 直接连接异常: {e}")
    
    print("❌ 所有ClickHouse连接方法都失败")
    return False, None

if __name__ == "__main__":
    success, data = test_clickhouse_connection()
    if success:
        print("🎉 ClickHouse连接测试成功！")
    else:
        print("💥 ClickHouse连接测试失败！")
