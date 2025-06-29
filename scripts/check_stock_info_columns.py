#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
检查stock_info表的列名和数据结构
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.clickhouse_db import get_clickhouse_db

def main():
    print("开始检查stock_info表结构和数据...")
    
    try:
        db = get_clickhouse_db()
        
        # 检查表结构
        print("\n=== 检查表结构 ===")
        structure_query = "DESCRIBE TABLE stock_info"
        structure_result = db.query(structure_query)
        print("表结构:")
        print(structure_result)
        
        # 获取列名
        print("\n=== 获取列名 ===")
        columns_query = "SELECT * FROM stock_info LIMIT 1"
        sample_result = db.query(columns_query)
        print("列名:")
        if hasattr(sample_result, 'columns'):
            print(list(sample_result.columns))
        else:
            print("无法获取列名")
        
        # 查看样本数据
        print("\n=== 样本数据 ===")
        sample_query = "SELECT * FROM stock_info WHERE code = '000001' ORDER BY date DESC LIMIT 3"
        sample_data = db.query(sample_query)
        print("样本数据:")
        print(sample_data)
        
        # 检查特定股票的数据量
        print("\n=== 数据量检查 ===")
        count_query = "SELECT code, COUNT(*) as count FROM stock_info WHERE code IN ('000001', '000002') GROUP BY code"
        count_result = db.query(count_query)
        print("数据量:")
        print(count_result)
        
        # 检查日期范围
        print("\n=== 日期范围检查 ===")
        date_query = """
        SELECT 
            code,
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(*) as total_records
        FROM stock_info 
        WHERE code = '000001'
        GROUP BY code
        """
        date_result = db.query(date_query)
        print("日期范围:")
        print(date_result)
        
    except Exception as e:
        print(f"检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 