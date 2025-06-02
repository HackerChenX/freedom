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

    def __init__(self, use_mock_data: bool = False):
        """
        初始化回测数据管理器
        
        Args:
            use_mock_data: 是否使用模拟数据
        """
        self.use_mock_data = use_mock_data
        
        # 初始化数据库
        try:
            self.db_manager = DBManager()
            self.use_mock_data = False
        except Exception as e:
            logger.warning(f"数据库连接失败，将使用模拟数据: {e}")
            self.db_manager = None
            self.use_mock_data = True
            self._init_mock_data()
        
        # 初始化周期管理器
        self.period_manager = PeriodManager()
        
        # 缓存
        self.data_cache = {}
        
        logger.info("回测数据管理器初始化完成")
    
    def _init_mock_data(self):
        """初始化模拟数据，用于在数据库连接失败时提供测试数据"""
        try:
            logger.info("初始化模拟数据...")

            # 创建一些模拟股票数据
            mock_stocks = pd.DataFrame({
                'code': ['000001', '000002', '000568', '600000'],
                'name': ['平安银行', '万科A', '泸州老窖', '浦发银行'],
                'industry': ['银行', '房地产', '白酒', '银行'],
                'area': ['深圳', '深圳', '四川', '上海'],
                'list_date': ['1991-04-03', '1991-05-29', '1998-05-16', '1999-11-10']
            })

            # 创建模拟数据管理器
            class MockDataManager:
                def __init__(self):
                    self.stocks = mock_stocks

                def get_stock_list(self):
                    """获取股票列表"""
                    return self.stocks

                def get_stock_name(self, stock_code):
                    """获取股票名称"""
                    matched = self.stocks[self.stocks['code'] == stock_code]
                    if not matched.empty:
                        return matched.iloc[0]['name']
                    return stock_code

                def get_stock_info(self, stock_code, level, start_date, end_date):
                    """获取股票行情数据"""
                    import numpy as np
                    from datetime import datetime, timedelta

                    logger.info(f"生成股票 {stock_code} 的模拟数据，周期: {level}, 时间段: {start_date} - {end_date}")

                    # 解析日期
                    if isinstance(start_date, str):
                        if len(start_date) == 8:  # YYYYMMDD format
                            start = datetime.strptime(start_date, '%Y%m%d')
                        else:
                            start = datetime.strptime(start_date, '%Y-%m-%d')
                    else:
                        start = start_date

                    if isinstance(end_date, str):
                        if len(end_date) == 8:  # YYYYMMDD format
                            end = datetime.strptime(end_date, '%Y%m%d')
                        else:
                            end = datetime.strptime(end_date, '%Y-%m-%d')
                    else:
                        end = end_date

                    # 生成交易日期
                    dates = []
                    current = start
                    while current <= end:
                        # 跳过周末
                        if current.weekday() < 5:  # 0-4表示周一至周五
                            dates.append(current)
                        current += timedelta(days=1)

                    if not dates:
                        logger.warning(f"在指定时间段 {start_date} - {end_date} 内没有交易日")
                        return pd.DataFrame()

                    # 生成模拟价格数据
                    n = len(dates)
                    np.random.seed(int(stock_code[-4:]) + hash(level) % 1000)  # 使用股票代码和周期作为随机种子

                    # 起始价格为 10 + 股票代码后两位的数值/10
                    base_price = 10 + int(stock_code[-2:]) / 10

                    # 生成随机价格变动
                    price_changes = np.random.normal(0, 0.02, n)  # 每日价格变动正态分布，均值0，标准差0.02

                    # 计算累积价格
                    cumulative_changes = np.cumsum(price_changes)
                    prices = base_price * (1 + cumulative_changes)

                    # 确保价格都为正
                    prices = np.maximum(prices, 0.1)

                    # 生成OHLC数据
                    daily_volatility = 0.01  # 日内波动率

                    opens = prices * (1 + np.random.normal(0, daily_volatility / 2, n))
                    highs = np.maximum(prices * (1 + np.random.normal(daily_volatility, daily_volatility / 2, n)),
                                       opens)
                    lows = np.minimum(prices * (1 + np.random.normal(-daily_volatility, daily_volatility / 2, n)),
                                      opens)
                    closes = prices

                    # 生成成交量数据
                    volumes = np.random.lognormal(mean=12, sigma=1, size=n)  # 成交量服从对数正态分布

                    # 获取股票名称
                    stock_name = self.get_stock_name(stock_code)

                    # 获取行业
                    industry = self.stocks[self.stocks['code'] == stock_code].iloc[0]['industry'] if not self.stocks[
                        self.stocks['code'] == stock_code].empty else 'Unknown'

                    # 创建DataFrame
                    df = pd.DataFrame({
                        'code': [stock_code] * n,
                        'name': [stock_name] * n,
                        'date': dates,
                        'level': [level] * n,
                        'open': opens,
                        'high': highs,
                        'low': lows,
                        'close': closes,
                        'volume': volumes,
                        'turnover_rate': np.random.uniform(1, 5, n),
                        'price_change': np.diff(np.append([closes[0]], closes)),
                        'price_range': (highs - lows) / lows * 100,
                        'industry': [industry] * n
                    })

                    logger.info(f"成功生成 {len(df)} 条模拟数据记录")
                    return df
                
                def get_trading_dates(self, stock_code, start_date, end_date):
                    """获取交易日期列表"""
                    data = self.get_stock_info(stock_code, "daily", start_date, end_date)
                    if data.empty:
                        return []
                    return data['date'].tolist()

            # 创建实例
            self.db_manager = MockDataManager()

            logger.info("模拟数据初始化完成")
        except Exception as e:
            logger.error(f"初始化模拟数据失败: {e}")
            
            # 创建最小的模拟数据管理器
            class MinimalMockDB:
                def get_stock_name(self, code):
                    return code
                
                def get_stock_list(self):
                    return pd.DataFrame({'code': ['000001'], 'name': ['模拟股票']})
                
                def get_stock_info(self, *args, **kwargs):
                    return pd.DataFrame()
                
                def get_trading_dates(self, *args, **kwargs):
                    return []
            
            self.db_manager = MinimalMockDB()
    
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
            period: 周期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 股票数据
        """
        # 转换周期
        if isinstance(period, Period):
            period_str = period.value
        else:
            period_str = period
        
        # 检查缓存
        cache_key = f"{stock_code}_{period_str}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # 获取数据
        data = self.db_manager.get_stock_info(
            stock_code=stock_code,
            level=period_str.lower(),
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
            period="daily",
            start_date="20220101",
            end_date="20220131"
        )
        print(f"股票 {test_stock} 数据: {len(data)} 条记录") 