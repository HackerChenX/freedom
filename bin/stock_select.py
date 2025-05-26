#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import multiprocessing as mp
import time
import pandas as pd
from config.config import get_config
from formula import formula
from db.db_manager import DBManager
from utils.logger import get_logger
from utils.path_utils import get_stock_result_file, get_stock_code_name_file
from enums.kline_period import KlinePeriod
from strategy.strategy_factory import StrategyFactory

# 获取日志记录器
logger = get_logger(__name__)

# 使用配置系统管理全局变量
GLOBAL_DATE = get_config('stock.default_date')

# 使用数据库管理器单例获取数据库连接
db_manager = DBManager.get_instance()


def stock_select():
    """
    多线程选股
    """
    stock_code_name = pd.read_csv(get_stock_code_name_file(), dtype=str)
    start_time = time.time()
    pool = mp.Pool()  # 创建线程池
    for index, row in stock_code_name.iterrows():
        pool.apply_async(选股, args=(row["code"],))  # 并发执行每个任务
    pool.close()  # 关闭线程池
    pool.join()  # 等待所有线程执行结束
    end_time = time.time()
    logger.info(f"选股完成，耗时 {end_time - start_time}秒")


def stock_select_strategy(strategy_name, **params):
    """
    使用指定策略选股
    
    Args:
        strategy_name: 策略名称
        **params: 策略参数
    """
    logger.info(f"使用 {strategy_name} 策略选股，参数: {params}")
    
    try:
        # 创建策略实例
        strategy = StrategyFactory.create_strategy(strategy_name, **params)
        if not strategy:
            logger.error(f"无法创建策略: {strategy_name}")
            return
        
        # 获取当前日期
        today = GLOBAL_DATE or time.strftime('%Y%m%d')
        
        # 执行选股
        selected_stocks = strategy.select(db_manager, today)
        
        # 输出结果
        if selected_stocks:
            logger.info(f"选股结果: 共找到 {len(selected_stocks)} 只股票")
            for stock in selected_stocks:
                logger.info(f"股票: {stock['code']} {stock['name']} 行业: {stock['industry']}")
            
            # 保存结果到文件
            result_file = f"data/result/选股结果_{strategy_name}_{today}.csv"
            pd.DataFrame(selected_stocks).to_csv(result_file, index=False, encoding='utf-8-sig')
            logger.info(f"选股结果已保存到: {result_file}")
        else:
            logger.info("未找到符合条件的股票")
    
    except Exception as e:
        logger.error(f"选股出错: {e}", exc_info=True)


def stock_select_by_industry(industry_code):
    """
    按行业选股
    
    Args:
        industry_code: 行业代码
    """
    logger.info(f"按行业 {industry_code} 选股")
    
    try:
        # 获取当前日期
        today = GLOBAL_DATE or time.strftime('%Y%m%d')
        
        # 获取该行业的所有股票
        stocks = db_manager.get_stocks_by_industry(industry_code)
        
        if stocks:
            logger.info(f"行业 {industry_code} 共有 {len(stocks)} 只股票")
            for stock in stocks:
                logger.info(f"股票: {stock['code']} {stock['name']}")
            
            # 保存结果到文件
            result_file = f"data/result/行业股票_{industry_code}_{today}.csv"
            pd.DataFrame(stocks).to_csv(result_file, index=False, encoding='utf-8-sig')
            logger.info(f"行业股票已保存到: {result_file}")
        else:
            logger.info(f"未找到行业 {industry_code} 的股票")
    
    except Exception as e:
        logger.error(f"按行业选股出错: {e}", exc_info=True)


def stock_select_list():
    """
    从列表中选股
    """
    with open(get_stock_result_file(), "r") as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split(" ")
            code = arr[0]
            if len(code) == 6:
                选股(code)


def stock_select_single(code):
    """
    单个股票选股测试
    
    Args:
        code: 股票代码
    """
    选股(code)


def 选股(code):
    """
    单个股票选股逻辑
    
    Args:
        code: 股票代码
    """
    try:
        f = formula.Formula(code)
        
        # 没有数据，跳过
        if f.dataDay.history is not None and len(f.dataDay.history) > 0:
            # 策略条件
            if (f.换手率大于(1.5) and 
                f.弹性() and 
                f.吸筹(KlinePeriod.DAILY) and 
                f.吸筹(KlinePeriod.WEEKLY)):
                
                logger.info(f"选股结果: {f.get_desc()}")
                
                # 使用路径工具获取文件路径
                with open(get_stock_result_file(), "a") as file:
                    file.write(f"{f.get_desc()}\n")
    except Exception as e:
        logger.error(f"选股过程中出错: {code} - {e}")


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='股票选股工具')
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['all', 'strategy', 'industry', 'list', 'single'],
                      help='选股模式: all-所有股票, strategy-使用策略, industry-按行业, list-从列表, single-单个股票')
    parser.add_argument('--strategy', type=str, help='策略名称，仅在mode=strategy时使用')
    parser.add_argument('--industry', type=str, help='行业代码，仅在mode=industry时使用')
    parser.add_argument('--code', type=str, help='股票代码，仅在mode=single时使用')
    parser.add_argument('--params', type=str, help='策略参数，JSON格式，仅在mode=strategy时使用')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        stock_select()
    elif args.mode == 'strategy':
        if not args.strategy:
            parser.error("使用策略选股需要指定--strategy参数")
        
        # 解析策略参数
        import json
        params = {}
        if args.params:
            try:
                params = json.loads(args.params)
            except json.JSONDecodeError:
                parser.error("--params参数必须是有效的JSON格式")
        
        stock_select_strategy(args.strategy, **params)
    elif args.mode == 'industry':
        if not args.industry:
            parser.error("按行业选股需要指定--industry参数")
        stock_select_by_industry(args.industry)
    elif args.mode == 'list':
        stock_select_list()
    elif args.mode == 'single':
        if not args.code:
            parser.error("单个股票选股需要指定--code参数")
        stock_select_single(args.code)


if __name__ == "__main__":
    main() 