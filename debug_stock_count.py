#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
调试脚本：检查数据库中的股票数量
"""

import sys
import os

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from db.unified_data_manager import get_unified_data_manager
from utils.logger import get_logger, init_logging

# 初始化日志
init_logging(level="INFO")
logger = get_logger(__name__)

def main():
    """主函数"""
    try:
        print("=== 调试股票数量问题 ===")
        
        # 获取数据管理器
        data_manager = get_unified_data_manager()
        
        print("\n1. 测试获取股票列表（无限制）")
        stock_list = data_manager.get_stock_list()
        print(f"   获取到股票数量: {len(stock_list)}")
        if stock_list:
            print(f"   前10只股票: {stock_list[:10]}")
        
        print("\n2. 测试获取股票列表（限制100）")
        stock_list_100 = data_manager.get_stock_list(limit=100)
        print(f"   获取到股票数量: {len(stock_list_100)}")
        
        print("\n3. 测试获取股票信息（无限制）")
        stock_info = data_manager.enhanced_manager.get_stock_info()
        df = stock_info.to_dataframe()
        print(f"   获取到股票信息数量: {len(df)}")
        if not df.empty:
            print(f"   数据列: {list(df.columns)}")
            print(f"   前5行:")
            print(df.head())
        
        print("\n4. 测试获取股票信息（限制20）")
        stock_info_20 = data_manager.enhanced_manager.get_stock_info(limit=20)
        df_20 = stock_info_20.to_dataframe()
        print(f"   获取到股票信息数量: {len(df_20)}")
        
        print("\n5. 检查数据库连接和查询")
        from db.enhanced_connection_pool import get_connection_pool
        pool = get_connection_pool()
        
        with pool.get_connection() as conn:
            # 直接查询股票信息表
            result = conn.query_dataframe("SELECT COUNT(*) as total FROM stock_info")
            print(f"   数据库中stock_info表总记录数: {result.iloc[0]['total']}")
            
            # 查询不同股票代码数量
            result2 = conn.query_dataframe("SELECT COUNT(DISTINCT code) as unique_stocks FROM stock_info")
            print(f"   数据库中不同股票代码数量: {result2.iloc[0]['unique_stocks']}")
            
            # 查询最新日期的股票数量
            result3 = conn.query_dataframe("""
                SELECT date, COUNT(DISTINCT code) as stocks_count 
                FROM stock_info 
                GROUP BY date 
                ORDER BY date DESC 
                LIMIT 5
            """)
            print(f"   最近5个交易日的股票数量:")
            for _, row in result3.iterrows():
                print(f"     {row['date']}: {row['stocks_count']}只股票")
        
        print("\n=== 调试完成 ===")
        
    except Exception as e:
        logger.error(f"调试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
