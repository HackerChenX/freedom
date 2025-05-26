#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import argparse

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from scripts.akshare_to_clickhouse import AKShareToClickHouse

def sync_stock(stock_code, stock_name):
    """
    同步单只股票的数据
    :param stock_code: 股票代码
    :param stock_name: 股票名称
    """
    print(f"开始同步股票数据: {stock_code} - {stock_name}")
    
    # 初始化同步器
    synchronizer = AKShareToClickHouse()
    
    # 获取股票数据
    # 15分钟数据
    print(f"获取15分钟级别数据...")
    stock_data_15min = synchronizer.get_stock_15min_data(stock_code)
    if not stock_data_15min.empty:
        processed_data = synchronizer.prepare_stock_data_for_clickhouse(stock_data_15min, stock_code, stock_name)
        if not processed_data.empty:
            synchronizer.save_stock_data_to_clickhouse(processed_data, '15分钟')
    
    # 日线数据
    print(f"获取日线级别数据...")
    stock_data_daily = synchronizer.get_stock_daily_data(stock_code)
    if not stock_data_daily.empty:
        processed_data = synchronizer.prepare_stock_data_for_clickhouse(stock_data_daily, stock_code, stock_name)
        if not processed_data.empty:
            synchronizer.save_stock_data_to_clickhouse(processed_data, '日线')
    
    # 周线数据
    print(f"获取周线级别数据...")
    stock_data_weekly = synchronizer.get_stock_weekly_data(stock_code)
    if not stock_data_weekly.empty:
        processed_data = synchronizer.prepare_stock_data_for_clickhouse(stock_data_weekly, stock_code, stock_name)
        if not processed_data.empty:
            synchronizer.save_stock_data_to_clickhouse(processed_data, '周线')
    
    # 月线数据
    print(f"获取月线级别数据...")
    stock_data_monthly = synchronizer.get_stock_monthly_data(stock_code)
    if not stock_data_monthly.empty:
        processed_data = synchronizer.prepare_stock_data_for_clickhouse(stock_data_monthly, stock_code, stock_name)
        if not processed_data.empty:
            synchronizer.save_stock_data_to_clickhouse(processed_data, '月线')
    
    print(f"股票 {stock_code} - {stock_name} 数据同步完成")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='同步单只股票的数据')
    parser.add_argument('--code', type=str, required=True, help='股票代码')
    parser.add_argument('--name', type=str, required=True, help='股票名称')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 同步股票数据
    sync_stock(args.code, args.name) 