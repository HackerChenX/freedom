#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
ClickHouse数据库连接模拟器

用于在ClickHouse服务器不可用时提供模拟数据，确保系统测试能够继续进行。
"""

import pandas as pd
from datetime import datetime, timedelta
import random
from typing import List, Tuple, Any, Dict
import logging

logger = logging.getLogger(__name__)

class MockClickHouseConnection:
    """模拟ClickHouse连接"""
    
    def __init__(self):
        self.connected = True
        self._generate_mock_data()
    
    def _generate_mock_data(self):
        """生成模拟股票数据"""
        # 模拟股票代码
        self.stock_codes = ['000001', '000002', '300005', '603359', '300003', '600036', '000858']
        
        # 模拟表结构
        self.table_schema = [
            ('code', 'String'),
            ('name', 'String'), 
            ('date', 'Date'),
            ('level', 'String'),
            ('open', 'Float64'),
            ('high', 'Float64'),
            ('low', 'Float64'),
            ('close', 'Float64'),
            ('volume', 'UInt64'),
            ('price_change', 'Float64'),
            ('price_range', 'Float64'),
            ('industry', 'String'),
            ('datetime', 'DateTime'),
            ('seq', 'UInt64')
        ]
        
        # 生成模拟数据（优化以满足策略条件）
        self.mock_data = []
        base_date = datetime(2024, 1, 1)

        for code_idx, code in enumerate(self.stock_codes):
            # 为不同股票设置不同的价格基础，确保有些股票满足策略条件
            if code_idx % 3 == 0:  # 1/3的股票设置为上涨趋势
                base_price = random.uniform(20, 80)
                trend_factor = 1.002  # 轻微上涨趋势
            elif code_idx % 3 == 1:  # 1/3的股票设置为震荡
                base_price = random.uniform(15, 60)
                trend_factor = 1.0
            else:  # 1/3的股票设置为下跌趋势
                base_price = random.uniform(10, 40)
                trend_factor = 0.998

            current_price = base_price

            for i in range(100):  # 每个股票100天数据
                date = base_date + timedelta(days=i)

                # 应用趋势因子
                current_price *= trend_factor

                # 添加随机波动
                daily_volatility = random.uniform(0.95, 1.05)
                open_price = current_price * daily_volatility
                high_price = open_price * random.uniform(1.0, 1.08)
                low_price = open_price * random.uniform(0.92, 1.0)
                close_price = open_price * random.uniform(0.98, 1.02)

                # 确保价格逻辑正确
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)

                record = {
                    'code': code,
                    'name': f'股票{code}',
                    'date': date.strftime('%Y-%m-%d'),
                    'level': '日线',
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': random.randint(1000000, 10000000),
                    'price_change': round((close_price - open_price), 2),
                    'price_range': round(((high_price - low_price) / open_price * 100), 2),
                    'industry': '科技',
                    'datetime': date,
                    'seq': i + 1
                }
                self.mock_data.append(record)

                # 更新当前价格为收盘价
                current_price = close_price
        
        logger.info(f"生成了{len(self.mock_data)}条模拟股票数据")
    
    def execute(self, query: str, params=None) -> List[Tuple]:
        """执行SQL查询（模拟）"""
        query = query.strip().upper()
        
        # 简单连接测试
        if 'SELECT 1' in query:
            return [(1,)]
        
        # 表结构查询
        if 'DESCRIBE STOCK_INFO' in query:
            return self.table_schema
        
        # 计数查询
        if 'SELECT COUNT(*)' in query and 'FROM STOCK_INFO' in query:
            return [(len(self.mock_data),)]
        
        # 日期范围查询
        if 'SELECT MIN(DATE), MAX(DATE)' in query:
            dates = [record['date'] for record in self.mock_data]
            return [(min(dates), max(dates))]
        
        # 股票代码查询
        if 'SELECT DISTINCT CODE' in query:
            return [(code,) for code in self.stock_codes[:5]]
        
        # 股票数据查询
        if 'FROM STOCK_INFO' in query and 'WHERE' in query:
            return self._filter_stock_data(query)
        
        # 默认返回空结果
        logger.warning(f"未处理的查询: {query}")
        return []
    
    def _filter_stock_data(self, query: str) -> List[Tuple]:
        """过滤股票数据（简单实现）"""
        # 提取股票代码
        code = None
        if "CODE = '" in query:
            start = query.find("CODE = '") + 8
            end = query.find("'", start)
            code = query[start:end]
        
        # 过滤数据
        filtered_data = []
        for record in self.mock_data:
            if code and record['code'] != code:
                continue
            
            # 根据SELECT字段返回相应数据
            if 'SELECT *' in query:
                filtered_data.append(tuple(record.values()))
            else:
                # 简化处理，返回主要字段
                filtered_data.append((
                    record['code'],
                    record['name'], 
                    record['date'],
                    record['open'],
                    record['high'],
                    record['low'],
                    record['close'],
                    record['volume']
                ))
        
        return filtered_data[:50]  # 限制返回数量
    
    def query_dataframe(self, query: str) -> pd.DataFrame:
        """查询并返回DataFrame"""
        result = self.execute(query)
        
        if not result:
            return pd.DataFrame()
        
        # 根据查询类型构建DataFrame
        if 'FROM STOCK_INFO' in query.upper() and 'WHERE' in query.upper():
            columns = ['code', 'name', 'date', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(result, columns=columns)
            
            # 转换数据类型
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            return df
        
        # 其他查询返回简单DataFrame
        return pd.DataFrame(result)
    
    def close(self):
        """关闭连接"""
        self.connected = False
        logger.info("模拟数据库连接已关闭")

    def disconnect(self):
        """断开连接（兼容ClickHouse客户端接口）"""
        self.close()

    def ping(self):
        """检查连接状态"""
        return self.connected

class MockClickHouseConnectionPool:
    """模拟ClickHouse连接池"""
    
    def __init__(self):
        self.pool_size = 5
        self.connections = []
        logger.info("模拟ClickHouse连接池已初始化")
    
    def get_connection(self):
        """获取连接（上下文管理器）"""
        return MockConnectionContext()
    
    def query_dataframe(self, query: str) -> pd.DataFrame:
        """直接查询DataFrame"""
        conn = MockClickHouseConnection()
        return conn.query_dataframe(query)
    
    def close(self):
        """关闭连接池"""
        logger.info("模拟ClickHouse连接池已关闭")

class MockConnectionContext:
    """模拟连接上下文管理器"""
    
    def __enter__(self):
        self.connection = MockClickHouseConnection()
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

# 全局模拟连接池实例
_mock_pool = None

def get_mock_connection_pool():
    """获取模拟连接池"""
    global _mock_pool
    if _mock_pool is None:
        _mock_pool = MockClickHouseConnectionPool()
    return _mock_pool
