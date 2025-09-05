#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查数据库数据
检查ClickHouse数据库中是否有股票价格数据
"""

import os
import sys

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.dependency_injection import get_service

def check_database_tables():
    """检查数据库表结构"""
    print("=== 检查数据库表结构 ===")
    
    try:
        # 使用依赖注入获取数据访问服务
        data_access = get_service("IDataAccess")
        
        # 查看所有表
        tables_query = "SHOW TABLES"
        tables = data_access.query_dataframe(tables_query)
        print(f"数据库中的表: {tables}")
        
        # 检查每个表的结构和数据量
        for i in range(len(tables)):
            table_name = tables.iloc[i, 0]
                
            print(f"\n--- 表 {table_name} ---")
            
            # 查看表结构
            desc_query = f"DESCRIBE TABLE {table_name}"
            try:
                structure = data_access.query_dataframe(desc_query)
                print(f"表结构: {structure.head()}")  # 只显示前5个字段
            except Exception as e:
                print(f"查看表结构失败: {e}")
                
            # 查看数据量
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            try:
                count = data_access.query_dataframe(count_query)
                print(f"数据量: {count}")
            except Exception as e:
                print(f"查看数据量失败: {e}")
                
            # 如果是价格相关的表，查看样本数据
            if any(keyword in table_name.lower() for keyword in ['price', 'daily', 'kline', 'stock']):
                sample_query = f"SELECT code, name, date, level, open, close, high, low, volume FROM {table_name} LIMIT 5"
                try:
                    sample = data_access.query_dataframe(sample_query)
                    print(f"样本数据: {sample}")
                except Exception as e:
                    print(f"查看样本数据失败: {e}")
                    
    except Exception as e:
        print(f"检查数据库表失败: {e}")

def check_stock_data():
    """检查具体股票数据"""
    print("\n=== 检查具体股票数据 ===")
    
    try:
        # 使用依赖注入获取数据访问服务
        data_access = get_service("IDataAccess")
        
        # 尝试常见的股票数据表名
        possible_tables = [
            'daily_price', 'stock_daily', 'kline_data', 'stock_price', 
            'daily_data', 'price_data', 'stock_kline', 'daily'
        ]
        
        for table_name in possible_tables:
            print(f"\n尝试查询表: {table_name}")
            
            # 检查表是否存在
            check_query = f"SELECT COUNT(*) FROM {table_name} WHERE code = '000001' LIMIT 1"
            try:
                result = data_access.query_dataframe(check_query)
                print(f"表 {table_name} 中000001的数据量: {result}")
                
                # 如果有数据，查看样本
                if result is not None and not result.empty and result.iloc[0, 0] > 0:
                    sample_query = f"SELECT code, name, date, level, open, close, high, low, volume FROM {table_name} WHERE code = '000001' ORDER BY date DESC LIMIT 3"
                    sample = data_access.query_dataframe(sample_query)
                    print(f"样本数据: {sample}")
                    
            except Exception as e:
                print(f"查询表 {table_name} 失败: {e}")
                
    except Exception as e:
        print(f"检查股票数据失败: {e}")

def check_date_range():
    """检查数据的时间范围"""
    print("\n=== 检查数据时间范围 ===")
    
    try:
        # 使用依赖注入获取数据访问服务
        data_access = get_service("IDataAccess")
        
        # 尝试查找有日期字段的表
        tables_with_data = []
        
        # 先获取所有表
        tables_query = "SHOW TABLES"
        tables = data_access.query_dataframe(tables_query)
        
        for i in range(len(tables)):
            table_name = tables.iloc[i, 0]
                
            try:
                # 尝试查询日期范围
                date_query = f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {table_name} LIMIT 1"
                date_range = data_access.query_dataframe(date_query)
                if date_range is not None and not date_range.empty:
                    print(f"表 {table_name} 日期范围: {date_range.iloc[0].to_dict()}")
                    tables_with_data.append(table_name)
            except:
                # 如果没有date字段，尝试其他可能的日期字段
                for date_field in ['trade_date', 'trading_date', 'dt', 'timestamp']:
                    try:
                        date_query = f"SELECT MIN({date_field}) as min_date, MAX({date_field}) as max_date FROM {table_name} LIMIT 1"
                        date_range = data_access.query_dataframe(date_query)
                        if date_range is not None and not date_range.empty:
                            print(f"表 {table_name} ({date_field}) 日期范围: {date_range.iloc[0].to_dict()}")
                            tables_with_data.append(table_name)
                            break
                    except:
                        continue
                        
        print(f"\n有日期数据的表: {tables_with_data}")
                
    except Exception as e:
        print(f"检查日期范围失败: {e}")

def main_checkdatabasedata():
    """主函数"""
    print("开始检查数据库数据")
    
    check_database_tables()
    check_stock_data()
    check_date_range()
    
    print("\n数据库数据检查完成")

if __name__ == "__main__":
    main_checkdatabasedata() 