#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查ClickHouse数据的日期范围
"""

import sys
import os
from clickhouse_driver import Client
from db.sql_manager import SQLManager, QueryType

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def check_date_range():
    """检查日期范围"""
    print("🔍 检查ClickHouse数据的日期范围...")
    
    try:
        client = Client(
            host='localhost',
            port=9000,
            user='default',
            password='123456',
            database='stock'
        )
        
        # 查询日期范围
        date_query = """
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(DISTINCT date) as unique_dates,
            COUNT(*) as total_records
        FROM stock_info
        """
        
        print("📊 查询日期范围...")
        result = client.execute(date_query)
        
        if result:
            min_date, max_date, unique_dates, total_records = result[0]
            print(f"✅ 日期范围查询成功:")
            print(f"最早日期: {min_date}")
            print(f"最晚日期: {max_date}")
            print(f"唯一日期数: {unique_dates}")
            print(f"总记录数: {total_records}")
            
            # 如果只有一个日期，查询所有不同的日期
            if unique_dates <= 10:
                distinct_dates_query = """
                SELECT DISTINCT date, COUNT(*) as count
                FROM stock_info
                GROUP BY date
                ORDER BY date DESC
                """
                
                print("\n📅 所有不同日期:")
                dates_result = client.execute(distinct_dates_query)
                for date, count in dates_result:
                    print(f"  {date}: {count}条记录")
        else:
            print("❌ 日期范围查询失败")
            
    except Exception as e:
        print(f"❌ 查询异常: {e}")

if __name__ == "__main__":
    check_date_range()
