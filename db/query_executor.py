"""
查询执行器

结合SQL管理器和数据库连接，提供统一的查询执行接口。
支持参数化查询、结果缓存和错误处理。
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
from datetime import datetime
import logging
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType, get_sql_manager
from db.interfaces.data_access_interface import DataAccessInterface
from utils.dependency_injection import get_container

logger = get_logger(__name__)

class QueryExecutor:
    """查询执行器
    
    提供统一的查询执行接口，结合SQL管理器和数据库连接。
    """
    
    def __init__(self, data_access: Optional[DataAccessInterface] = None, 
                 sql_manager: Optional[SQLManager] = None):
        """初始化查询执行器
        
        Args:
            data_access: 数据访问接口，如果为None则从容器获取
            sql_manager: SQL管理器，如果为None则使用默认实例
        """
        self.data_access = data_access or self._get_data_access()
        self.sql_manager = sql_manager or get_sql_manager()
    
    def _get_data_access(self) -> DataAccessInterface:
        """从依赖注入容器获取数据访问接口"""
        try:
            container = get_container()
            return container.get('data_access')
        except Exception as e:
            logger.error(f"无法从容器获取数据访问接口: {e}")
            # 降级处理：使用动态导入
            import importlib
            clickhouse_module = importlib.import_module('db.clickhouse_db')
            ClickhouseDB = getattr(clickhouse_module, 'ClickhouseDB')
            return ClickhouseDB()
    
    def execute_query(self, query_type: QueryType, params: Dict[str, Any]) -> pd.DataFrame:
        """执行查询
        
        Args:
            query_type: 查询类型
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
            
        Raises:
            Exception: 查询执行失败
        """
        try:
            # 构建查询语句
            query = self.sql_manager.build_query(query_type, params)
            
            # 执行查询
            logger.debug(f"执行查询: {query_type.value}")
            result = self.data_access.execute_query(query)
            
            if result is None:
                logger.warning(f"查询返回空结果: {query_type.value}")
                return pd.DataFrame()
            
            return result
        
        except Exception as e:
            logger.error(f"查询执行失败 {query_type.value}: {e}")
            raise
    
    def execute_custom_query(self, query_type: str, params: Dict[str, Any]) -> pd.DataFrame:
        """执行自定义查询
        
        Args:
            query_type: 自定义查询类型
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
        """
        try:
            # 获取自定义查询模板
            query_template = self.sql_manager.get_custom_query(query_type)
            
            # 格式化参数
            formatted_params = self.sql_manager._format_params(params)
            
            # 构建查询语句
            query = query_template % formatted_params
            
            # 执行查询
            logger.debug(f"执行自定义查询: {query_type}")
            result = self.data_access.execute_query(query)
            
            if result is None:
                logger.warning(f"自定义查询返回空结果: {query_type}")
                return pd.DataFrame()
            
            return result
        
        except Exception as e:
            logger.error(f"自定义查询执行失败 {query_type}: {e}")
            raise
    
    def get_stock_data(self, code: str, start_date: str, end_date: str, 
                      level: str = '日线') -> pd.DataFrame:
        """获取股票数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            level: 数据级别
            
        Returns:
            pd.DataFrame: 股票数据
        """
        params = {
            'code': code,
            'start_date': start_date,
            'end_date': end_date,
            'level': level
        }
        return self.execute_query(QueryType.STOCK_DATA, params)
    
    def get_batch_stock_data(self, codes: List[str], start_date: str, 
                           end_date: str, level: str = '日线') -> pd.DataFrame:
        """获取多只股票数据
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            level: 数据级别
            
        Returns:
            pd.DataFrame: 股票数据
        """
        params = {
            'codes': codes,
            'start_date': start_date,
            'end_date': end_date,
            'level': level
        }
        return self.execute_query(QueryType.BATCH_STOCK_DATA, params)
    
    def get_stock_list(self, level: str = '日线') -> pd.DataFrame:
        """获取股票列表
        
        Args:
            level: 数据级别
            
        Returns:
            pd.DataFrame: 股票列表
        """
        params = {'level': level}
        return self.execute_query(QueryType.STOCK_LIST, params)
    
    def get_stock_info(self, code: str, level: str = '日线') -> pd.DataFrame:
        """获取股票信息
        
        Args:
            code: 股票代码
            level: 数据级别
            
        Returns:
            pd.DataFrame: 股票信息
        """
        params = {'code': code, 'level': level}
        return self.execute_query(QueryType.STOCK_INFO, params)
    
    def get_indicator_data(self, table_name: str, code: str, 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """获取指标数据
        
        Args:
            table_name: 表名
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 指标数据
        """
        params = {
            'table_name': table_name,
            'code': code,
            'start_date': start_date,
            'end_date': end_date
        }
        return self.execute_query(QueryType.INDICATOR_DATA, params)
    
    def get_industry_list(self, level: str = '日线') -> pd.DataFrame:
        """获取行业列表
        
        Args:
            level: 数据级别
            
        Returns:
            pd.DataFrame: 行业列表
        """
        params = {'level': level}
        return self.execute_query(QueryType.INDUSTRY_LIST, params)
    
    def get_date_range(self, code: str, level: str = '日线') -> pd.DataFrame:
        """获取股票数据日期范围
        
        Args:
            code: 股票代码
            level: 数据级别
            
        Returns:
            pd.DataFrame: 日期范围
        """
        params = {'code': code, 'level': level}
        return self.execute_query(QueryType.DATE_RANGE, params)
    
    def get_stock_count(self, level: str = '日线') -> int:
        """获取股票总数
        
        Args:
            level: 数据级别
            
        Returns:
            int: 股票总数
        """
        params = {'level': level}
        result = self.execute_query(QueryType.STOCK_COUNT, params)
        
        if result.empty:
            return 0
        
        return int(result.iloc[0]['total_stocks'])
    
    def get_latest_data(self, code: str, limit: int = 1, 
                       level: str = '日线') -> pd.DataFrame:
        """获取最新数据
        
        Args:
            code: 股票代码
            limit: 返回条数
            level: 数据级别
            
        Returns:
            pd.DataFrame: 最新数据
        """
        params = {
            'code': code,
            'limit': limit,
            'level': level
        }
        return self.execute_query(QueryType.LATEST_DATA, params)
    
    def get_performance_data(self, codes: List[str], start_date: str, 
                           end_date: str, level: str = '日线') -> pd.DataFrame:
        """获取性能数据
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            level: 数据级别
            
        Returns:
            pd.DataFrame: 性能数据
        """
        params = {
            'codes': codes,
            'start_date': start_date,
            'end_date': end_date,
            'level': level
        }
        return self.execute_query(QueryType.PERFORMANCE_DATA, params)
    
    def validate_data(self, code: str, date: str, level: str = '日线') -> bool:
        """验证数据存在性
        
        Args:
            code: 股票代码
            date: 日期
            level: 数据级别
            
        Returns:
            bool: 数据是否存在
        """
        params = {
            'code': code,
            'date': date,
            'level': level
        }
        result = self.execute_query(QueryType.VALIDATION_DATA, params)
        return not result.empty

# 全局查询执行器实例
_query_executor = None

def get_query_executor() -> QueryExecutor:
    """获取查询执行器实例
    
    Returns:
        QueryExecutor: 查询执行器实例
    """
    global _query_executor
    if _query_executor is None:
        _query_executor = QueryExecutor()
    return _query_executor 