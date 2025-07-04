"""
数据访问接口定义

定义系统核心数据访问接口，遵循架构分层规则
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd


class IDataAccess(ABC):
    """
    数据访问接口
    
    定义所有数据访问的标准接口，确保分层架构的清晰性
    """
    
    @abstractmethod
    def get_stock_info(self, 
                       stock_code: Union[str, List[str]] = None,
                       level: Union[str, Any] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       filters: Optional[Dict[str, Any]] = None,
                       limit: Optional[int] = None,
                       order_by: str = "date DESC") -> Any:
        """
        获取股票数据
        
        Args:
            stock_code: 股票代码或股票代码列表
            level: K线周期
            start_date: 开始日期
            end_date: 结束日期
            filters: 过滤条件
            limit: 限制返回记录数
            order_by: 排序规则
            
        Returns:
            Any: 股票数据对象
        """
        pass
    
    @abstractmethod
    def get_stock_list(self, 
                       industry: Optional[str] = None,
                       market: Optional[str] = None,
                       filters: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        获取股票代码列表
        
        Args:
            industry: 行业筛选
            market: 市场筛选
            filters: 其他过滤条件
            
        Returns:
            List[str]: 股票代码列表
        """
        pass
    
    @abstractmethod
    def get_industry_list(self) -> pd.DataFrame:
        """
        获取行业列表
        
        Returns:
            pd.DataFrame: 行业列表数据
        """
        pass
    
    @abstractmethod
    def query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        执行SQL查询
        
        Args:
            sql: SQL查询语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
        """
        pass
    
    @abstractmethod
    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        执行SQL语句
        
        Args:
            sql: SQL语句
            params: 执行参数
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        测试数据库连接
        
        Returns:
            bool: 连接状态
        """
        pass
    
    @abstractmethod
    def get_stock_max_date(self) -> datetime:
        """
        获取股票数据最新日期
        
        Returns:
            datetime: 最新日期
        """
        pass
    
    @abstractmethod
    def get_industry_max_date(self) -> datetime:
        """
        获取行业数据最新日期
        
        Returns:
            datetime: 最新日期
        """
        pass
    
    @abstractmethod
    def get_avg_price(self, code: str, start_date: Union[str, datetime]) -> float:
        """
        获取股票平均价格
        
        Args:
            code: 股票代码
            start_date: 开始日期
            
        Returns:
            float: 平均价格
        """
        pass


class IStockDataProvider(ABC):
    """
    股票数据提供者接口
    """
    
    @abstractmethod
    def get_kline_data(self, 
                       code: str, 
                       start_date: str, 
                       end_date: str, 
                       level: str = '日线') -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            level: K线周期
            
        Returns:
            pd.DataFrame: K线数据
        """
        pass
    
    @abstractmethod
    def get_stock_basic_info(self, codes: List[str]) -> pd.DataFrame:
        """
        获取股票基本信息
        
        Args:
            codes: 股票代码列表
            
        Returns:
            pd.DataFrame: 股票基本信息
        """
        pass


class IMarketDataProvider(ABC):
    """
    市场数据提供者接口
    """
    
    @abstractmethod
    def get_market_overview(self, date: str) -> Dict[str, Any]:
        """
        获取市场概览数据
        
        Args:
            date: 查询日期
            
        Returns:
            Dict[str, Any]: 市场概览数据
        """
        pass
    
    @abstractmethod
    def get_industry_performance(self, date: str) -> pd.DataFrame:
        """
        获取行业表现数据
        
        Args:
            date: 查询日期
            
        Returns:
            pd.DataFrame: 行业表现数据
        """
        pass 