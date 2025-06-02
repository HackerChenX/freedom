#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import argparse
import datetime
from typing import List, Dict, Any, Optional

from utils.logger import get_logger
from scripts.backtest.consolidated_backtest import ConsolidatedBacktest

logger = get_logger(__name__)

def main():
    """股票综合回测主函数"""
    parser = argparse.ArgumentParser(description="股票综合回测分析工具")
    
    parser.add_argument("--csv", required=True, help="包含股票代码和日期的CSV文件路径")
    parser.add_argument("--days-before", type=int, default=20, help="分析买点前几天的数据")
    parser.add_argument("--days-after", type=int, default=10, help="分析买点后几天的数据")
    parser.add_argument("--output-dir", help="输出目录路径")
    parser.add_argument("--no-strategy", action="store_true", help="不生成选股策略")
    
    args = parser.parse_args()
    
    # 运行综合回测
    try:
        print("开始股票综合回测分析...")
        backtest = ConsolidatedBacktest()
        report_file, strategy_file = backtest.run(args.csv, args.days_before, args.days_after)
        
        if report_file:
            print(f"分析报告已生成: {report_file}")
            
            if strategy_file and not args.no_strategy:
                print(f"选股策略已生成: {strategy_file}")
                
            print("\n回测分析完成！")
        else:
            print("回测分析失败，请检查日志获取详细错误信息。")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"回测分析过程中出错: {e}")
        print(f"回测分析过程中出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 