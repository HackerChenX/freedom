#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import pandas as pd
import numpy as np
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from enums.kline_period import Kline_period
from analysis.market.a_stock_market_analysis import Market_analyzer, print_market_indicators
from datetime import datetime
import argparse
from typing import Optional
from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor

logger = get_logger(__name__)


@exception_handler(reraise=False, default_return=None)
@performance_monitor(threshold=5.0)
def get_latest_data(analyzer: Market_analyzer) -> Optional[Market_analyzer]:
    """
    获取最新数据
    
    Args:
        analyzer: 市场分析器实例
        
    Returns:
        Optional[Market_analyzer]: 更新后的分析器，失败时返回None
    """
    try:
        # 使用analyzer获取最新数据
        date = analyzer.date
        logger.info(f"获取{date}数据...")
        
        # 分析数据
        analyzer.calculate_market_strength()
        
        logger.info(f"成功获取并分析{date}的市场数据")
        return analyzer
        
    except Exception as e:
        logger.error(f"获取最新数据失败: {e}")
        return None


@exception_handler(reraise=False)
def main_34():
    """
    主函数
    """
    try:
        # 初始化依赖注入容器
        container = get_container()
        data_access = get_service(Data_access_interface)
        logger.info("成功初始化数据访问服务")
        
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='股票市场分析工具')
        parser.add_argument('--date', type=str, help='分析日期 (YYYYMMDD格式)')
        parser.add_argument('--data_source', type=str, default='auto', 
                            choices=['akshare', 'baostock', 'auto'],
                            help='数据源 (默认: auto)')
        parser.add_argument('--debug', action='store_true', help='启用调试模式')
        
        args = parser.parse_args()
        
        # 初始化市场分析器，传入数据访问接口
        analyzer = Market_analyzer(
            date=args.date, 
            data_source=args.data_source,
            data_access=data_access
        )
        
        logger.info(f"开始分析市场数据，日期: {args.date or '最新'}")
        
        # 获取最新数据
        analyzer = get_latest_data(analyzer)
        
        if analyzer is None:
            logger.error("获取市场数据失败，程序退出")
            print("错误: 无法获取市场数据，请检查数据源连接")
            return
        
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
            print(f"数据访问接口: {type(data_access).__name__}")
            
            # 打印容器信息
            print(f"依赖注入容器: {type(container).__name__}")
            print(f"已注册服务数量: {len(container._services)}")
            
        logger.info("市场分析完成")
        
    except Keyboard_interrupt:
        logger.info("用户中断程序执行")
        print("\n程序被用户中断")
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        print(f"程序执行出错: {e}")
        if args and args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main_34() 