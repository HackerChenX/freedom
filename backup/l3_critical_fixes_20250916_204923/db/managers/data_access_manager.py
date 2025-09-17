"""
数据访问管理器 - 修复版本
实现统一的数据访问接口，严格遵循L3数据服务层规范
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging

from db.interfaces.data_access_interface import DataAccessInterface
from db.enhanced_connection_pool import get_connection_pool
from db.sql_manager import SQLManager, QueryType
from utils.enhanced_exception_handler import exception_handler
from utils.enhanced_performance_monitor import performance_monitor
from utils.logger import get_logger

logger = get_logger(__name__)


class DataAccessManager(DataAccessInterface):
    """
    数据访问管理器
    实现统一的数据访问接口，提供高性能的数据查询服务
    """
    
    def __init__(self):
        """初始化数据访问管理器"""
        self.logger = logger
        self.connection_pool = get_connection_pool()
        self.sql_manager = SQLManager()
        self.cache_service = None  # 可选的缓存服务
        
        self.logger.info("数据访问管理器初始化完成")

    # 实现抽象接口方法
    def get_stock_data_data_access_interface(self, code: str, start_date: str, end_date: str,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """实现接口方法：获取股票数据"""
        return self.get_stock_data(code, start_date, end_date, columns)

    def get_stocks_data_batch_data_access_interface(self, codes: List[str], start_date: str, end_date: str,
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """实现接口方法：批量获取股票数据"""
        return self.get_stocks_data_batch(codes, start_date, end_date, columns)

    def get_indicator_data_data_access_interface(self, code: str, indicator: str, start_date: str, end_date: str,
                          params: Optional[Dict] = None) -> pd.DataFrame:
        """实现接口方法：获取指标数据"""
        return self.get_indicator_data(code, indicator, start_date, end_date, params)

    def check_data_exists_data_access_interface(self, table: str, conditions: Dict) -> bool:
        """实现接口方法：检查数据是否存在"""
        return self.check_data_exists(table, conditions)

    def get_latest_data_data_access_interface(self, table: str, code: str, columns: Optional[List[str]] = None) -> Optional[Dict]:
        """实现接口方法：获取最新数据"""
        return self.get_latest_data(table, code, columns)

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=2.0)
    def get_stock_data(self,
                      code: str,
                      start_date: str,
                      end_date: str,
                      level: str = '日线',
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取股票数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            level: 数据级别
            
        Returns:
            股票数据DataFrame
        """
        params = {
            'code': code,
            'start_date': start_date,
            'end_date': end_date,
            'level': level
        }
        
        query = self.sql_manager.get_query(QueryType.STOCK_DATA)
        return self.connection_pool.query_dataframe(query, params)
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=3.0)
    def get_batch_stock_data(self, 
                            codes: List[str], 
                            start_date: str, 
                            end_date: str, 
                            level: str = '日线') -> pd.DataFrame:
        """
        批量获取股票数据
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            level: 数据级别
            
        Returns:
            批量股票数据DataFrame
        """
        params = {
            'codes': codes,
            'start_date': start_date,
            'end_date': end_date,
            'level': level
        }
        
        query = self.sql_manager.get_query(QueryType.BATCH_STOCK_DATA)
        return self.connection_pool.query_dataframe(query, params)
    
    @exception_handler(reraise=False, default_return=[])
    @performance_monitor(threshold_seconds=1.0)
    def get_stock_list(self, level: str = '日线') -> List[str]:
        """
        获取股票列表
        
        Args:
            level: 数据级别
            
        Returns:
            股票代码列表
        """
        params = {'level': level}
        
        query = self.sql_manager.get_query(QueryType.STOCK_LIST)
        df = self.connection_pool.query_dataframe(query, params)
        
        if df.empty:
            return []
        
        return df['code'].tolist()
    
    @exception_handler(reraise=False, default_return=[])
    def get_stock_list_data_access_interface(self, market: Optional[str] = None) -> List[str]:
        """
        获取股票列表 - 接口兼容方法
        
        Args:
            market: 市场类型, 可选
            
        Returns:
            股票代码列表
        """
        return self.get_stock_list()
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def get_stock_info(self, code: str, level: str = '日线') -> Optional[Dict[str, Any]]:
        """
        获取股票信息
        
        Args:
            code: 股票代码
            level: 数据级别
            
        Returns:
            股票信息字典
        """
        params = {
            'code': code,
            'level': level
        }
        
        query = self.sql_manager.get_query(QueryType.STOCK_INFO)
        df = self.connection_pool.query_dataframe(query, params)
        
        if df.empty:
            return None
        
        return df.iloc[0].to_dict()
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=2.0)
    def get_date_range(self, code: str, level: str = '日线') -> Optional[Dict[str, str]]:
        """
        获取股票数据的日期范围
        
        Args:
            code: 股票代码
            level: 数据级别
            
        Returns:
            日期范围字典 {'start_date': str, 'end_date': str}
        """
        params = {
            'code': code,
            'level': level
        }
        
        query = self.sql_manager.get_query(QueryType.DATE_RANGE)
        df = self.connection_pool.query_dataframe(query, params)
        
        if df.empty:
            return None
        
        return {
            'start_date': str(df.iloc[0]['min_date']),
            'end_date': str(df.iloc[0]['max_date'])
        }
    
    @exception_handler(reraise=False, default_return=0)
    @performance_monitor(threshold_seconds=1.0)
    def get_stock_count(self, level: str = '日线') -> int:
        """
        获取股票数量
        
        Args:
            level: 数据级别
            
        Returns:
            股票数量
        """
        params = {'level': level}
        
        query = self.sql_manager.get_query(QueryType.STOCK_COUNT)
        df = self.connection_pool.query_dataframe(query, params)
        
        if df.empty:
            return 0
        
        return int(df.iloc[0]['stock_count'])
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def get_latest_data(self, code: str, level: str = '日线', limit: int = 1) -> pd.DataFrame:
        """
        获取最新数据
        
        Args:
            code: 股票代码
            level: 数据级别
            limit: 限制条数
            
        Returns:
            最新数据DataFrame
        """
        params = {
            'code': code,
            'level': level,
            'limit': limit
        }
        
        query = self.sql_manager.get_query(QueryType.LATEST_DATA)
        return self.connection_pool.query_dataframe(query, params)

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=2.0)
    def get_stocks_data_batch(self, codes: List[str], start_date: str, end_date: str,
                             columns: Optional[List[str]] = None, level: str = '日线') -> pd.DataFrame:
        """批量获取多只股票数据"""
        if not codes:
            return pd.DataFrame()

        # 构建批量查询
        codes_str = "', '".join(codes)
        columns_str = ', '.join(columns) if columns else '*'

        query = f"""
        SELECT {columns_str}
        FROM stock_info
        WHERE code IN ('{codes_str}')
        AND level = '{level}'
        AND date >= '{start_date}' AND date <= '{end_date}'
        ORDER BY code, date ASC
        """

        return self.connection_pool.query_dataframe(query)

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def get_indicator_data(self, code: str, indicator: str, start_date: str, end_date: str,
                          params: Optional[Dict] = None) -> pd.DataFrame:
        """获取指标数据"""
        # 暂时返回空DataFrame，需要指标数据表
        self.logger.warning(f"指标数据获取功能尚未实现: {indicator}")
        return pd.DataFrame()

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def check_data_exists(self, table: str, conditions: Dict) -> bool:
        """检查数据是否存在"""
        where_clauses = []
        for key, value in conditions.items():
            if isinstance(value, str):
                where_clauses.append(f"{key} = '{value}'")
            else:
                where_clauses.append(f"{key} = {value}")

        where_str = " AND ".join(where_clauses)
        query = f"SELECT COUNT(*) FROM {table} WHERE {where_str}"

        result = self.connection_pool.query_dataframe(query)
        return result.iloc[0, 0] > 0 if not result.empty else False

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def get_latest_data(self, table: str, code: str, columns: Optional[List[str]] = None) -> Optional[Dict]:
        """获取最新数据"""
        columns_str = ', '.join(columns) if columns else '*'

        query = f"""
        SELECT {columns_str}
        FROM {table}
        WHERE code = '{code}'
        ORDER BY date DESC
        LIMIT 1
        """

        result = self.connection_pool.query_dataframe(query)
        return result.iloc[0].to_dict() if not result.empty else None
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=5.0)
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        执行自定义查询
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果DataFrame
        """
        return self.connection_pool.query_dataframe(query, params)
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"DataAccessManager(pool={self.connection_pool})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()
