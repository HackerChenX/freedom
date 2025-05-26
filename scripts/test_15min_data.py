#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from formula import formula
from enums.kline_period import KlinePeriod
from utils.logger import get_logger

logger = get_logger(__name__)

def test_15min_data(code="600585", date="20250408"):
    """
    测试指定股票的15分钟数据
    
    Args:
        code: 股票代码
        date: 日期，格式为YYYYMMDD
    """
    try:
        # 日期前后范围 - 使用2025年的日期
        start_date = "20250401"
        end_date = "20250430"
        
        logger.info(f"测试股票 {code} 的15分钟数据，日期范围: {start_date} - {end_date}")
        
        # 获取不同周期的数据
        logger.info("获取日线数据...")
        f_daily = formula.Formula(code, start=start_date, end=end_date)
        
        logger.info("获取15分钟数据...")
        f_15min = formula.StockData(code, KlinePeriod.MIN_15, start=start_date, end=end_date)
        
        logger.info("获取30分钟数据...")
        f_30min = formula.StockData(code, KlinePeriod.MIN_30, start=start_date, end=end_date)
        
        logger.info("获取60分钟数据...")
        f_60min = formula.StockData(code, KlinePeriod.MIN_60, start=start_date, end=end_date)
        
        # 打印结果
        logger.info(f"日线数据条数: {len(f_daily.dataDay.close)}")
        logger.info(f"15分钟数据条数: {len(f_15min.close)}")
        logger.info(f"30分钟数据条数: {len(f_30min.close)}")
        logger.info(f"60分钟数据条数: {len(f_60min.close)}")
        
        # 打印第一条记录的日期
        if len(f_daily.dataDay.history) > 0:
            logger.info(f"日线第一条记录日期: {f_daily.dataDay.history['date'][0]}")
        else:
            logger.info("日线数据为空")
            
        if f_15min.history is not None and len(f_15min.history) > 0:
            logger.info(f"15分钟第一条记录日期: {f_15min.history['date'][0]}")
            logger.info(f"15分钟数据示例: {f_15min.history.iloc[0]}")
        else:
            logger.info("15分钟数据为空")
            
        if f_30min.history is not None and len(f_30min.history) > 0:
            logger.info(f"30分钟第一条记录日期: {f_30min.history['date'][0]}")
        else:
            logger.info("30分钟数据为空")
            
        if f_60min.history is not None and len(f_60min.history) > 0:
            logger.info(f"60分钟第一条记录日期: {f_60min.history['date'][0]}")
        else:
            logger.info("60分钟数据为空")
    
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试15分钟数据")
    parser.add_argument("--code", type=str, default="600585", help="股票代码")
    parser.add_argument("--date", type=str, default="20250408", help="日期，格式为YYYYMMDD")
    
    args = parser.parse_args()
    test_15min_data(args.code, args.date) 