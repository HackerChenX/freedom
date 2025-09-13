"""
优化后的数据访问接口设计
移除命名冗余，提供清晰的接口抽象
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
from datetime import datetime


class DataAccessInterface(ABC):
    """
    统一数据访问接口

    提供标准的数据访问抽象，支持股票数据、指标数据、市场数据的统一访问
    所有实现必须保证100%真实数据，禁止任何模拟数据
    """

    @abstractmethod
    def get_stock_data(self, code: str, start_date: str, end_date: str,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取单只股票数据

        Args:
            code: 股票代码
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            columns: 需要的列名列表，None表示获取所有基础列

        Returns:
            股票数据DataFrame，必须包含date, open, high, low, close, volume等基础字段

        Raises:
            DataAccessError: 数据访问失败时抛出
        """
        pass

    @abstractmethod
    def get_stocks_batch(self, codes: List[str], start_date: str, end_date: str,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        批量获取多只股票数据

        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            columns: 需要的列名列表

        Returns:
            多股票数据DataFrame，包含code列用于区分股票

        Raises:
            DataAccessError: 数据访问失败时抛出
        """
        pass

    @abstractmethod
    def get_indicator_data(self, code: str, indicator: str, start_date: str, end_date: str,
                          params: Optional[Dict] = None) -> pd.DataFrame:
        """
        获取技术指标数据

        Args:
            code: 股票代码
            indicator: 指标名称（如'RSI', 'MACD', 'KDJ'等）
            start_date: 开始日期
            end_date: 结束日期
            params: 指标计算参数

        Returns:
            指标数据DataFrame

        Raises:
            DataAccessError: 数据访问失败时抛出
        """
        pass

    @abstractmethod
    def get_stock_list(self, industry: Optional[str] = None,
                      market: Optional[str] = None,
                      filters: Optional[Dict] = None) -> List[str]:
        """
        获取股票代码列表

        Args:
            industry: 行业筛选条件
            market: 市场筛选条件（如'沪市', '深市'）
            filters: 其他筛选条件，如价格范围、市值范围等

        Returns:
            符合条件的股票代码列表

        Raises:
            DataAccessError: 数据访问失败时抛出
        """
        pass

    @abstractmethod
    def get_industry_list(self) -> List[Dict[str, Any]]:
        """
        获取行业信息列表

        Returns:
            行业信息列表，每个元素包含行业代码、名称等信息

        Raises:
            DataAccessError: 数据访问失败时抛出
        """
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        执行自定义查询

        Args:
            query: SQL查询语句
            params: 查询参数

        Returns:
            查询结果DataFrame

        Raises:
            DataAccessError: 数据访问失败时抛出
        """
        pass

    @abstractmethod
    def check_data_exists(self, table: str, conditions: Dict) -> bool:
        """
        检查数据是否存在

        Args:
            table: 表名
            conditions: 检查条件字典

        Returns:
            数据是否存在

        Raises:
            DataAccessError: 数据访问失败时抛出
        """
        pass

    @abstractmethod
    def get_latest_data(self, table: str, code: str,
                       columns: Optional[List[str]] = None) -> Optional[Dict]:
        """
        获取指定股票的最新数据

        Args:
            table: 数据表名
            code: 股票代码
            columns: 需要的列名列表

        Returns:
            最新数据字典，如果没有数据返回None

        Raises:
            DataAccessError: 数据访问失败时抛出
        """
        pass

    @abstractmethod
    def get_market_data(self, date: str, market: Optional[str] = None) -> pd.DataFrame:
        """
        获取市场数据

        Args:
            date: 日期
            market: 市场筛选条件

        Returns:
            市场数据DataFrame

        Raises:
            DataAccessError: 数据访问失败时抛出
        """
        pass

    @abstractmethod
    def get_data_range(self, table: str, code: str = None) -> Tuple[str, str]:
        """
        获取数据的时间范围

        Args:
            table: 表名
            code: 股票代码，None表示所有股票

        Returns:
            (最早日期, 最新日期) 元组

        Raises:
            DataAccessError: 数据访问失败时抛出
        """
        pass


class DataAccessError(Exception):
    """数据访问异常"""

    def __init__(self, message: str, error_code: Optional[str] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_exception = original_exception


class RealDataValidator:
    """
    真实数据验证器
    确保系统中不使用任何模拟数据
    """

    @staticmethod
    def validate_data_source(data_access: DataAccessInterface) -> bool:
        """
        验证数据源是否为真实数据

        Args:
            data_access: 数据访问实例

        Returns:
            是否为真实数据源
        """
        # 检查是否为模拟数据访问
        class_name = data_access.__class__.__name__

        # 禁止的模拟数据类名关键词
        forbidden_keywords = ['mock', 'fake', 'dummy', 'test', 'simulate']

        for keyword in forbidden_keywords:
            if keyword.lower() in class_name.lower():
                return False

        return True

    @staticmethod
    def validate_data_integrity(df: pd.DataFrame) -> bool:
        """
        验证数据完整性

        Args:
            df: 待验证的数据DataFrame

        Returns:
            数据是否完整有效
        """
        if df.empty:
            return True

        # 检查必需字段
        required_columns = ['date']
        for col in required_columns:
            if col not in df.columns:
                return False

        # 检查数据质量
        if df['date'].isna().any():
            return False

        return True


# 兼容性处理
IDataAccess = DataAccessInterface  # 向后兼容别名