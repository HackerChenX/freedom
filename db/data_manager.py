"""
数据管理器模块

负责从数据库获取和缓存数据
"""

import pandas as pd
import time
from typing import Dict, List, Optional, Any, Union
import logging

from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger

logger = get_logger(__name__)


class DataManager:
    """
    数据管理器，负责数据的获取和缓存
    """
    
    def __init__(self, cache_enabled=True, cache_ttl=3600):
        """
        初始化数据管理器
        
        Args:
            cache_enabled: 是否启用缓存
            cache_ttl: 缓存有效期（秒）
        """
        self.db = get_clickhouse_db()
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.cache_timestamp = {}
        
    def get_kline_data(self, stock_code: str, period: str, start_date: str, end_date: str, 
                       fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            stock_code: 股票代码
            period: 周期(DAILY, WEEKLY, MONTHLY, MIN60, MIN30, MIN15, MIN5)
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表，默认为所有字段
            
        Returns:
            K线数据DataFrame
        """
        # 构建缓存键
        cache_key = f"{stock_code}_{period}_{start_date}_{end_date}_{','.join(fields or [])}"
        
        # 检查缓存
        if self.cache_enabled and cache_key in self.cache:
            # 检查缓存是否过期
            if time.time() - self.cache_timestamp.get(cache_key, 0) < self.cache_ttl:
                logger.debug(f"从缓存获取数据: {cache_key}")
                return self.cache[cache_key]
        
        # 构建查询字段
        if fields is None:
            fields = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        
        # 构建查询SQL
        table_name = self._get_table_name(period)
        field_str = ', '.join(fields)
        sql = f"""
        SELECT {field_str}
        FROM {table_name}
        WHERE stock_code = '{stock_code}'
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        ORDER BY date
        """
        
        # 执行查询
        logger.debug(f"执行SQL查询: {sql}")
        result = self.db.query(sql)
        
        # 更新缓存
        if self.cache_enabled:
            self.cache[cache_key] = result
            self.cache_timestamp[cache_key] = time.time()
        
        return result
    
    def get_stock_list(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            filters: 过滤条件
            
        Returns:
            股票列表DataFrame
        """
        # 构建查询条件
        where_clause = "WHERE 1=1"
        if filters:
            if 'market' in filters and filters['market']:
                markets = "', '".join(filters['market'])
                where_clause += f" AND market IN ('{markets}')"
            if 'industry' in filters and filters['industry']:
                industries = "', '".join(filters['industry'])
                where_clause += f" AND industry IN ('{industries}')"
            if 'market_cap' in filters:
                if 'min' in filters['market_cap']:
                    where_clause += f" AND market_cap >= {filters['market_cap']['min']}"
                if 'max' in filters['market_cap']:
                    where_clause += f" AND market_cap <= {filters['market_cap']['max']}"
            if 'price' in filters:
                if 'min' in filters['price']:
                    where_clause += f" AND latest_price >= {filters['price']['min']}"
                if 'max' in filters['price']:
                    where_clause += f" AND latest_price <= {filters['price']['max']}"
        
        # 构建查询SQL
        sql = f"""
        SELECT stock_code, stock_name, market, industry, market_cap, latest_price
        FROM stock_basic
        {where_clause}
        """
        
        # 执行查询
        logger.debug(f"执行SQL查询: {sql}")
        return self.db.query(sql)
    
    def get_stock_basic_info(self, stock_codes: List[str]) -> pd.DataFrame:
        """
        获取股票基本信息
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            股票基本信息DataFrame
        """
        if not stock_codes:
            return pd.DataFrame()
            
        # 构建股票代码列表字符串
        stock_codes_str = "', '".join(stock_codes)
        
        # 构建查询SQL
        sql = f"""
        SELECT stock_code, stock_name, market, industry, market_cap, latest_price
        FROM stock_basic
        WHERE stock_code IN ('{stock_codes_str}')
        """
        
        # 执行查询
        logger.debug(f"执行SQL查询: {sql}")
        return self.db.query(sql)
    
    def get_industry_list(self) -> pd.DataFrame:
        """
        获取行业列表
        
        Returns:
            行业列表DataFrame
        """
        sql = """
        SELECT DISTINCT industry
        FROM stock_basic
        WHERE industry != ''
        ORDER BY industry
        """
        
        return self.db.query(sql)
    
    def get_market_list(self) -> pd.DataFrame:
        """
        获取市场列表
        
        Returns:
            市场列表DataFrame
        """
        sql = """
        SELECT DISTINCT market
        FROM stock_basic
        WHERE market != ''
        ORDER BY market
        """
        
        return self.db.query(sql)
    
    def clear_cache(self):
        """清除所有缓存"""
        self.cache = {}
        self.cache_timestamp = {}
        logger.info("已清除所有数据缓存")
    
    def _get_table_name(self, period: str) -> str:
        """
        根据周期获取对应的表名
        
        Args:
            period: 周期字符串
            
        Returns:
            str: 表名
        """
        # 转换周期字符串到表名
        period_map = {
            'DAILY': 'kline_daily',
            'WEEKLY': 'kline_weekly',
            'MONTHLY': 'kline_monthly',
            'MIN60': 'kline_60min',
            'MIN30': 'kline_30min',
            'MIN15': 'kline_15min',
            'MIN5': 'kline_5min'
        }
        
        if period not in period_map:
            raise ValueError(f"不支持的周期: {period}")
            
        return period_map[period]
    
    def save_selection_result(self, result: pd.DataFrame) -> bool:
        """
        保存选股结果到数据库
        
        Args:
            result: 选股结果DataFrame
            
        Returns:
            bool: 是否保存成功
        """
        if result.empty:
            logger.warning("选股结果为空，无需保存")
            return True
            
        try:
            # 确保必要的列存在
            required_columns = ["stock_code", "stock_name", "strategy_id", "strategy_name", "selection_date", "rank"]
            for col in required_columns:
                if col not in result.columns:
                    logger.error(f"选股结果缺少必要的列: {col}")
                    return False
            
            # 构建插入SQL
            columns = ", ".join(result.columns)
            
            # 准备插入数据
            records = []
            for _, row in result.iterrows():
                values = []
                for col in result.columns:
                    value = row[col]
                    if isinstance(value, str):
                        values.append(f"'{value}'")
                    elif pd.isna(value):
                        values.append("NULL")
                    else:
                        values.append(str(value))
                
                records.append(f"({', '.join(values)})")
                
            values_str = ", ".join(records)
            
            # 执行插入
            sql = f"""
            INSERT INTO stock_selection_results ({columns})
            VALUES {values_str}
            """
            
            self.db.execute(sql)
            logger.info(f"已保存 {len(result)} 条选股结果到数据库")
            return True
        except Exception as e:
            logger.error(f"保存选股结果到数据库失败: {e}")
            return False
    
    def get_selection_history(self, strategy_id: Optional[str] = None, 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             limit: int = 100) -> pd.DataFrame:
        """
        获取选股历史记录
        
        Args:
            strategy_id: 策略ID
            start_date: 开始日期
            end_date: 结束日期
            limit: 限制返回的记录数
            
        Returns:
            选股历史记录DataFrame
        """
        # 构建查询条件
        where_clause = "WHERE 1=1"
        if strategy_id:
            where_clause += f" AND strategy_id = '{strategy_id}'"
        if start_date:
            where_clause += f" AND selection_date >= '{start_date}'"
        if end_date:
            where_clause += f" AND selection_date <= '{end_date}'"
            
        # 构建查询SQL
        sql = f"""
        SELECT stock_code, stock_name, strategy_id, strategy_name, selection_date, rank
        FROM stock_selection_results
        {where_clause}
        ORDER BY selection_date DESC, rank ASC
        LIMIT {limit}
        """
        
        # 执行查询
        return self.db.query(sql) 