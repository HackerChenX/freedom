#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查询ClickHouse数据库中的真实股票数据
了解可用的数据范围，为测试提供准确的股票代码和日期
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from db.managers.data_access_manager import get_unified_data_manager
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

def query_available_data():
    """查询数据库中可用的股票数据"""
    print("=" * 60)
    print("🔍 查询ClickHouse数据库中的真实股票数据")
    print("=" * 60)

    try:
        # 获取数据管理器
        data_manager = get_unified_data_manager()

        # 1. 查询日期范围
        print("\n📅 查询数据日期范围...")
        date_query = """
        SELECT
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(DISTINCT date) as total_days
        FROM stock_info WHERE code = %(code)s AND level = '日线'
        """

        date_result = data_manager.query_Manager_Unified_Data_Manager(date_query)
        if not date_result.empty:
            print(f"数据日期范围: {date_result.iloc[0]['min_date']} 到 {date_result.iloc[0]['max_date']}")
            print(f"总交易日数: {date_result.iloc[0]['total_days']} 天")

        # 2. 查询股票代码数量
        print("\n📊 查询股票代码数量...")
        stock_query = """
        SELECT
            COUNT(DISTINCT code) as total_stocks
        FROM stock_info WHERE code = %(code)s AND level = '日线'
        """

        stock_result = data_manager.query_Manager_Unified_Data_Manager(stock_query)
        if not stock_result.empty:
            print(f"总股票数量: {stock_result.iloc[0]['total_stocks']} 只")

        # 3. 查询最近的股票数据样本
        print("\n📋 查询最新股票数据样本...")
        recent_query = """
        SELECT
            code, name, date, close, volume, turnover_rate
        FROM stock_info WHERE code = %(code)s AND level = '日线'
        ORDER BY date DESC, code
        LIMIT 10
        """

        recent_result = data_manager.query_Manager_Unified_Data_Manager(recent_query)
        if not recent_result.empty:
            print("最新股票数据样本:")
            for _, row in recent_result.iterrows():
                print(f"  {row['code']} - {row['name']} - {row['date']} - 收盘: {row['close']}")

        # 4. 查询常见股票的数据分布
        print("\n🏦 查询热门股票的数据可用性...")
        popular_stocks = ['000001', '000002', '600000', '600036', '601318', '000858']

        for stock_code in popular_stocks:
            stock_data_query = f"""
            SELECT
                code, name,
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(*) as data_count
            FROM stock_info
            WHERE code = '{stock_code}' AND level = '日线'
            GROUP BY code, name
            """

            stock_data = data_manager.query_Manager_Unified_Data_Manager(stock_data_query)
            if not stock_data.empty:
                row = stock_data.iloc[0]
                print(f"  {row['code']} - {row['name']}: {row['min_date']} 到 {row['max_date']} ({row['data_count']} 条记录)")
            else:
                print(f"  {stock_code}: 无数据")

        # 5. 查询近期有数据的股票代码
        print("\n🎯 查询近期有数据的股票代码...")
        recent_stocks_query = """
        SELECT
            code, name, MAX(date) as latest_date
        FROM stock_info WHERE code = %(code)s AND level = '日线'
        GROUP BY code, name
        ORDER BY latest_date DESC
        LIMIT 20
        """

        recent_stocks = data_manager.query_Manager_Unified_Data_Manager(recent_stocks_query)
        if not recent_stocks.empty:
            print("近期有数据的股票代码:")
            for _, row in recent_stocks.iterrows():
                print(f"  {row['code']} - {row['name']} - 最新: {row['latest_date']}")

        # 6. 为测试生成推荐的股票代码和日期
        print("\n💡 生成测试建议...")
        if not recent_stocks.empty:
            # 选择前5只有最新数据的股票
            test_stocks = recent_stocks.head(5)
            print("建议用于测试的股票代码和日期:")

            for _, row in test_stocks.iterrows():
                latest_date = row['latest_date']
                # 往前推几天作为买点日期
                test_date_query = f"""
                SELECT date
                FROM stock_info
                WHERE code = '{row['code']}' AND level = '日线' AND date <= '{latest_date}'
                ORDER BY date DESC
                LIMIT 10, 1
                """

                test_date_result = data_manager.query_Manager_Unified_Data_Manager(test_date_query)
                if not test_date_result.empty:
                    test_date = test_date_result.iloc[0]['date']
                    print(f"  股票: {row['code']} ({row['name']}), 建议买点日期: {test_date}")

        return True

    except Exception as e:
        print(f"❌ 查询数据库失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = query_available_data()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  查询被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 查询过程中发生未预期错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)