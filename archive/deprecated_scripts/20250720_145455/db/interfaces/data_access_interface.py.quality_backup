"""
统一数据访问接口层

提供标准的数据访问抽象接口，解决直接依赖数据库实现类的问题
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from datetime import datetime


class DataAccessInterface(ABC):
    """统一数据访问接口"""
    
    @abstractmethod
    def get_stock_data(self, code: str, start_date: str, end_date: str, 
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取股票数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            columns: 需要的列名列表，None表示所有列
            
        Returns:
            股票数据DataFrame
        """
        pass
    
    @abstractmethod
    def get_stocks_data_batch(self, codes: List[str], start_date: str, end_date: str,
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        批量获取多只股票数据
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            columns: 需要的列名列表
            
        Returns:
            股票数据DataFrame
        """
        pass
    
    @abstractmethod
    def get_indicator_data(self, code: str, indicator: str, start_date: str, end_date: str,
                          params: Optional[Dict] = None) -> pd.DataFrame:
        """
        获取指标数据
        
        Args:
            code: 股票代码
            indicator: 指标名称
            start_date: 开始日期
            end_date: 结束日期
            params: 指标参数
            
        Returns:
            指标数据DataFrame
        """
        pass
    
    @abstractmethod
    def get_stock_list(self, industry: Optional[str] = None, 
                      market: Optional[str] = None) -> List[str]:
        """
        获取股票列表
        
        Args:
            industry: 行业筛选
            market: 市场筛选
            
        Returns:
            股票代码列表
        """
        pass
    
    @abstractmethod
    def get_industry_list(self) -> List[str]:
        """
        获取行业列表
        
        Returns:
            行业列表
        """
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        执行查询
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果DataFrame
        """
        pass
    
    @abstractmethod
    def check_data_exists(self, table: str, conditions: Dict) -> bool:
        """
        检查数据是否存在
        
        Args:
            table: 表名
            conditions: 检查条件
            
        Returns:
            数据是否存在
        """
        pass
    
    @abstractmethod
    def get_latest_data(self, table: str, code: str, columns: Optional[List[str]] = None) -> Optional[Dict]:
        """
        获取最新数据
        
        Args:
            table: 表名
            code: 股票代码
            columns: 需要的列名
            
        Returns:
            最新数据字典
        """
        pass


class ClickHouseDataAccess(DataAccessInterface):
    """ClickHouse数据访问实现"""
    
    def __init__(self, connection_pool, sql_manager):
        """
        初始化ClickHouse数据访问
        
        Args:
            connection_pool: 连接池
            sql_manager: SQL管理器
        """
        self.connection_pool = connection_pool
        self.sql_manager = sql_manager
    
class DataAccessError(Exception):
    """数据访问异常"""
    pass


class DataAccessFactory:
    """数据访问工厂类"""
    
    @staticmethod
    def create_clickhouse_access(connection_pool, sql_manager) -> DataAccessInterface:
        """创建ClickHouse数据访问实例"""
        return ClickHouseDataAccess(connection_pool, sql_manager)
    
    @staticmethod
    def create_mock_access() -> DataAccessInterface:
        """创建模拟数据访问实例（用于测试）"""
        try:
            # 使用动态导入避免分层违规
            import importlib
            mock_module = importlib.import_module('tests.mocks.mock_data_access')
            MockDataAccess = mock_module.MockDataAccess
            return MockDataAccess()
        except ImportError:
            # 如果测试模块不可用，返回None
            return None


# 兼容性别名
Data_access_interface = DataAccessInterface
IDataAccess = DataAccessInterface