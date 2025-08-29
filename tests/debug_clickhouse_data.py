#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试ClickHouse数据
"""

import sys
import os
import pandas as pd
from clickhouse_driver import Client

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def debug_clickhouse_data():
    """调试ClickHouse数据"""
    print("🔍 调试ClickHouse数据...")
    
    try:
        client = Client(
            host='localhost',
            port=9000,
            user='default',
            password='123456',
            database='stock'
        )
        
        # 查询数据样本
        query = """
        SELECT
            date,
            code,
            open,
            high,
            low,
            close,
            volume
        FROM stock_info
        WHERE volume > 0
        AND close > 0
        AND open > 0
        AND high > 0
        AND low > 0
        ORDER BY date DESC, code
        LIMIT 1000
        """
        
        print("📊 执行查询...")
        result = client.execute(query)
        
        if result:
            print(f"✅ 查询成功，获取{len(result)}条记录")
            
            # 转换为DataFrame
            columns = ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(result, columns=columns)
            
            print("📊 数据样本:")
            print(df.head())
            
            # 检查日期范围
            df['date'] = pd.to_datetime(df['date'])
            date_range = (df['date'].max() - df['date'].min()).days
            
            print(f"\n📅 日期信息:")
            print(f"最早日期: {df['date'].min()}")
            print(f"最晚日期: {df['date'].max()}")
            print(f"日期范围: {date_range}天")
            
            # 检查股票数量
            unique_stocks = df['code'].nunique()
            print(f"股票数量: {unique_stocks}")
            
            # 检查数据质量
            print(f"\n📊 数据质量:")
            print(f"数据行数: {len(df)}")
            print(f"空值数量: {df.isnull().sum().sum()}")
            
            return df
        else:
            print("❌ 查询返回空结果")
            return None
            
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return None

if __name__ == "__main__":
    debug_clickhouse_data()
