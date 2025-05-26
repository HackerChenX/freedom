#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import pandas as pd
import numpy as np
from db.clickhouse_db import get_clickhouse_db, get_default_config
from enums.kline_period import KlinePeriod
from analysis.market.a_stock_market_analysis import MarketAnalyzer, print_market_indicators
from datetime import datetime
import argparse

def get_latest_data(analyzer):
    """获取最新数据"""
    # 使用analyzer获取最新数据
    date = analyzer.date
    print(f"获取{date}数据...")
    # 分析数据
    analyzer.calculate_market_strength()
    return analyzer

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='股票市场分析工具')
    parser.add_argument('--date', type=str, help='分析日期 (YYYYMMDD格式)')
    parser.add_argument('--data_source', type=str, default='auto', 
                        choices=['akshare', 'baostock', 'auto'],
                        help='数据源 (默认: auto)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 初始化市场分析器
    analyzer = MarketAnalyzer(date=args.date, data_source=args.data_source)
    
    # 获取最新数据
    analyzer = get_latest_data(analyzer)
    
    # 打印市场指标
    print_market_indicators(analyzer)
    
    # 获取操作建议
    advice = analyzer.get_operation_advice()
    print("\n==== 操作建议 ====")
    print(advice)
    
    if args.debug:
        # 在调试模式下打印更多信息
        print("\n==== 调试信息 ====")
        print(f"使用数据源: {args.data_source}")
        print(f"分析日期: {analyzer.date}")
        # 打印更多调试信息...

if __name__ == "__main__":
    main() 