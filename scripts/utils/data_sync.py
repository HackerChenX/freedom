#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

import multiprocessing as mp
import time
import datetime
import logging
from formula import formula
from utils.logger import get_logger
from utils.path_utils import get_stock_code_name_file
import pandas as pd

# 设置日志
logger = get_logger(__name__)

def 同步数据(max_date=None):
    """
    多线程同步股票数据
    
    Args:
        max_date: 最大日期，默认为None，表示当前日期
    """
    stock_code_name = pd.read_csv(get_stock_code_name_file(), dtype=str)
    start_time = time.time()
    pool = mp.Pool()  # 创建线程池
    for index, row in stock_code_name.iterrows():
        pool.apply_async(同步数据_task, args=(row["code"], max_date))  # 并发执行每个任务
    pool.close()  # 关闭线程池
    pool.join()  # 等待所有线程执行结束
    end_time = time.time()
    logger.info(f"同步数据完成，耗时 {end_time - start_time}秒")


def 同步数据_task(code, max_date=None):
    """
    同步单个股票数据任务
    
    Args:
        code: 股票代码
        max_date: 最大日期，默认为None，表示当前日期
    """
    try:
        # 设置起始日期为2年前
        start_date = (datetime.datetime.now() - datetime.timedelta(days=2*365)).strftime('%Y%m%d')
        
        # 如果没有指定最大日期，则使用当前日期
        if max_date is None:
            max_date = datetime.datetime.now().strftime('%Y%m%d')
            
        logger.info(f"同步股票数据: {code} 从 {start_date} 到 {max_date}")
        
        # 同步数据
        f = formula.Formula(code, start=start_date, end=max_date, sync=True)
        
        # 记录同步结果
        if f.dataDay.history is not None and len(f.dataDay.history) > 0:
            logger.info(f"同步成功: {f.get_desc()} 获取到 {len(f.dataDay.history)} 条日线数据")
        else:
            logger.warning(f"同步失败: {code} 没有获取到数据")
    except Exception as e:
        logger.error(f"同步股票 {code} 数据失败: {e}")


def 同步板块(max_date=None):
    """
    同步行业板块数据
    
    Args:
        max_date: 最大日期，默认为None，表示当前日期
    """
    from enums.industry import Industry
    
    # 设置起始日期为2年前
    start_date = (datetime.datetime.now() - datetime.timedelta(days=2*365)).strftime('%Y%m%d')
    
    # 如果没有指定最大日期，则使用当前日期
    if max_date is None:
        max_date = datetime.datetime.now().strftime('%Y%m%d')
    
    # 获取所有行业板块
    industries = Industry.get_all_industries()
    
    logger.info(f"开始同步 {len(industries)} 个行业板块数据，从 {start_date} 到 {max_date}")
    
    # 为每个行业创建数据同步任务
    start_time = time.time()
    pool = mp.Pool()  # 创建线程池
    
    for industry_name, industry_code in industries.items():
        pool.apply_async(同步板块_task, args=(industry_code, industry_name, start_date, max_date))
    
    pool.close()  # 关闭线程池
    pool.join()  # 等待所有线程执行结束
    
    end_time = time.time()
    logger.info(f"同步行业板块数据完成，耗时 {end_time - start_time}秒")


def 同步板块_task(industry_code, industry_name, start_date, max_date):
    """
    同步单个行业板块数据任务
    
    Args:
        industry_code: 行业代码
        industry_name: 行业名称
        start_date: 开始日期
        max_date: 最大日期
    """
    try:
        from formula.stock_formula import IndustryData
        
        logger.info(f"同步行业板块数据: {industry_name}({industry_code}) 从 {start_date} 到 {max_date}")
        
        # 同步数据
        industry_data = IndustryData(industry_code, start=start_date, end=max_date, sync=True)
        
        # 记录同步结果
        if hasattr(industry_data, 'history') and industry_data.history is not None and len(industry_data.history) > 0:
            logger.info(f"同步成功: {industry_name}({industry_code}) 获取到 {len(industry_data.history)} 条数据")
        else:
            logger.warning(f"同步失败: {industry_name}({industry_code}) 没有获取到数据")
    except Exception as e:
        logger.error(f"同步行业板块 {industry_name}({industry_code}) 数据失败: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='股票数据同步工具')
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['stock', 'industry', 'all'],
                      help='同步模式: stock-同步股票数据, industry-同步行业数据, all-同步所有数据')
    parser.add_argument('--date', type=str, help='最大日期，格式为YYYYMMDD，默认为当前日期')
    parser.add_argument('--code', type=str, help='股票代码，仅在mode=stock且需要同步单个股票时使用')
    
    args = parser.parse_args()
    
    if args.mode == 'stock':
        if args.code:
            同步数据_task(args.code, args.date)
        else:
            同步数据(args.date)
    elif args.mode == 'industry':
        同步板块(args.date)
    elif args.mode == 'all':
        同步数据(args.date)
        同步板块(args.date) 