#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查数据库表结构
确认stock_info表的实际字段
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.enhanced_connection_pool import ClickHouseConnectionPool
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

def check_database_structure():
    """检查数据库表结构"""
    try:
        pool = ClickHouseConnectionPool()
        
        # 检查stock_info表结构
        describe_query = "DESCRIBE stock_info"
        
        result_df = pool.query_dataframe(describe_query)
        result = result_df.values.tolist()

        print("📊 stock_info表结构:")
        print("=" * 50)

        fields = []
        for row in result:
            field_name = row[0]
            field_type = row[1]
            fields.append(field_name)
            print(f"  {field_name}: {field_type}")

        print(f"\n📋 总字段数: {len(fields)}")
        print(f"字段列表: {', '.join(fields)}")

        # 检查问题字段
        problem_fields = ['price_change', 'price_range', 'industry']
        missing_fields = []
        existing_fields = []

        for field in problem_fields:
            if field in fields:
                existing_fields.append(field)
            else:
                missing_fields.append(field)

        if missing_fields:
            print(f"\n❌ 缺失字段: {', '.join(missing_fields)}")

        if existing_fields:
            print(f"\n✅ 存在字段: {', '.join(existing_fields)}")

        # 检查数据样本
        sample_query = "SELECT code, name, date, open, high, low, close, volume, turnover_rate FROM stock_info LIMIT 5"
        sample_result_df = pool.query_dataframe(sample_query)
        sample_result = sample_result_df.values.tolist()

        print(f"\n📋 数据样本 (前5条):")
        print("=" * 50)
        for i, row in enumerate(sample_result):
            print(f"记录 {i+1}: {dict(zip(fields, row))}")

        return {
            'fields': fields,
            'missing_fields': missing_fields,
            'existing_fields': existing_fields,
            'total_fields': len(fields)
        }
            
    except Exception as e:
        logger.error(f"检查数据库结构失败: {e}")
        return None

def check_minute_data_tables():
    """检查分钟级数据表是否存在"""
    try:
        pool = ClickHouseConnectionPool()
        
        # 检查所有表
        show_tables_query = "SHOW TABLES"
        
        result_df = pool.query_dataframe(show_tables_query)
        result = result_df.values.tolist()

        tables = [row[0] for row in result]

        print(f"\n📊 数据库中的所有表:")
        print("=" * 50)
        for table in tables:
            print(f"  {table}")

        # 检查分钟级数据表
        minute_tables = [table for table in tables if 'min' in table.lower() or '分钟' in table]

        if minute_tables:
            print(f"\n✅ 发现分钟级数据表: {', '.join(minute_tables)}")
        else:
            print(f"\n❌ 未发现分钟级数据表")

        # 检查不同level的数据
        level_query = "SELECT DISTINCT level FROM stock_info ORDER BY level"
        level_result_df = pool.query_dataframe(level_query)
        level_result = level_result_df.values.tolist()

        levels = [row[0] for row in level_result]

        print(f"\n📋 stock_info表中的level类型:")
        print("=" * 50)
        for level in levels:
            # 统计每个level的数据量
            count_query = f"SELECT COUNT(*) FROM stock_info WHERE code = %(code)s AND level = '{level}'"
            count_result_df = pool.query_dataframe(count_query)
            count_result = count_result_df.values.tolist()
            count = count_result[0][0] if count_result else 0
            print(f"  {level}: {count} 条记录")

        return {
            'all_tables': tables,
            'minute_tables': minute_tables,
            'levels': levels
        }
            
    except Exception as e:
        logger.error(f"检查分钟级数据表失败: {e}")
        return None

def main():
    """主函数"""
    print("🔍 数据库结构检查")
    print("=" * 50)
    
    # 检查表结构
    structure_info = check_database_structure()
    
    # 检查分钟级数据
    minute_info = check_minute_data_tables()
    
    # 生成修复建议
    print(f"\n💡 修复建议:")
    print("=" * 50)
    
    if structure_info and structure_info['missing_fields']:
        print(f"1. 移除查询中的缺失字段: {', '.join(structure_info['missing_fields'])}")
    
    if minute_info and not minute_info['minute_tables']:
        print("2. 需要创建分钟级数据表或实现数据聚合功能")
    
    if minute_info and minute_info['levels']:
        missing_levels = []
        required_levels = ['15分钟', '30分钟', '60分钟', '日线', '周线', '月线']
        for level in required_levels:
            if level not in minute_info['levels']:
                missing_levels.append(level)
        
        if missing_levels:
            print(f"3. 缺失的数据周期: {', '.join(missing_levels)}")
    
    print(f"\n✅ 数据库结构检查完成")

if __name__ == "__main__":
    main()
