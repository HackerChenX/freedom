#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测数据管理模块

负责获取和管理回测所需的各种数据
"""

import os
import sys
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.decorators import performance_monitor, time_it
from utils.period_manager import PeriodManager
from db.db_manager import DBManager
from enums.period import Period

# 获取日志记录器
logger = get_logger(__name__)


class BacktestDataManager:
    """
    回测数据管理器
    
    负责获取和管理回测所需的各种数据，包括股票信息、行情数据等
    """

    def __init__(self):
        """
        初始化回测数据管理器
        """
        # 初始化数据库
        try:
            self.db_manager = DBManager()
            if not self.db_manager.db:
                raise Exception("数据库连接初始化失败")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise Exception("回测系统必须使用 ClickHouse 数据库，请检查数据库配置和连接状态")
        
        # 初始化周期管理器
        self.period_manager = PeriodManager()
        
        # 缓存
        self.data_cache = {}
        
        logger.info("回测数据管理器初始化完成")

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        
        Returns:
            pd.DataFrame: 股票列表
        """
        return self.db_manager.get_stock_list()
    
    def get_stock_name(self, stock_code: str) -> str:
        """
        获取股票名称
        
        Args:
            stock_code: 股票代码
            
        Returns:
            str: 股票名称
        """
        return self.db_manager.get_stock_name(stock_code)
    
    def get_stock_data(self, stock_code: str, period: Union[str, Period], 
                      start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取股票数据
        
        Args:
            stock_code: 股票代码
            period: 周期，必须是 Period 枚举值或可转换为 Period 的字符串
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 股票数据
        
        Raises:
            ValueError: 参数无效时抛出
        """
        # 转换周期为 Period 枚举
        if isinstance(period, str):
            try:
                period = Period.from_string(period)
            except ValueError as e:
                raise ValueError(f"无效的K线周期: {period}，支持的周期: {', '.join(Period.get_all_period_values())}") from e
        elif not isinstance(period, Period):
            raise ValueError(f"周期参数必须是 Period 枚举或可转换为 Period 的字符串，当前类型: {type(period)}")
        
        # 检查缓存
        cache_key = f"{stock_code}_{period.value}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # 获取数据
        data = self.db_manager.get_stock_info(
            stock_code=stock_code,
            level=period,  # 直接传递 Period 枚举
            start_date=start_date,
            end_date=end_date
        )
        
        # 缓存数据
        self.data_cache[cache_key] = data
        
        return data
    
    def get_period_data(self, stock_code: str, period: Union[str, Period], 
                       start_date: str, end_date: str) -> pd.DataFrame:
        """
        从周期管理器获取数据
        
        Args:
            stock_code: 股票代码
            period: 周期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 周期数据
        """
        return self.period_manager.get_data(
            stock_code=stock_code,
            period=period,
            start_date=start_date,
            end_date=end_date
        )
    
    @time_it
    def get_trading_dates(self, stock_code: str, start_date: str, end_date: str) -> List:
        """
        获取交易日期列表
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List: 交易日期列表
        """
        return self.db_manager.get_trading_dates(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
    
    def clear_cache(self):
        """清除缓存"""
        self.data_cache.clear()
        logger.info("数据缓存已清除")

    def get_last_close(self, stock_code: str, end_date: str, period: str = "daily") -> Optional[float]:
        """
        获取指定股票在指定日期的收盘价
        Args:
            stock_code: 股票代码
            end_date: 截止日期（字符串，格式YYYYMMDD或YYYY-MM-DD）
            period: 周期，默认为"daily"
        Returns:
            float or None: 收盘价
        """
        df = self.get_stock_data(stock_code, period, end_date, end_date)
        if df is not None and not df.empty:
            # 取最后一行的close
            return float(df.iloc[-1]["close"])
        return None


# 测试代码
if __name__ == "__main__":
    # 初始化数据管理器
    data_manager = BacktestDataManager()
    
    # 获取股票列表
    stock_list = data_manager.get_stock_list()
    print(f"股票列表: {len(stock_list)} 只股票")
    
    # 获取单只股票数据
    if not stock_list.empty:
        test_stock = stock_list.iloc[0]['code']
        data = data_manager.get_stock_data(
            stock_code=test_stock,
            period=Period.DAILY,
            start_date="20220101",
            end_date="20220131"
        )
        print(f"股票 {test_stock} 数据: {len(data)} 条记录") 