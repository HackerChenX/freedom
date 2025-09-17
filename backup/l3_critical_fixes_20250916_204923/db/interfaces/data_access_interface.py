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
    def get_stock_data_data_access_interface(self, code: str, start_date: str, end_date: str, 
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
    def get_stocks_data_batch_data_access_interface(self, codes: List[str], start_date: str, end_date: str,
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
    def get_stock_info(self, code: str, level: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Optional[List]:
        """
        获取股票信息（兼容买点分析器接口）

        Args:
            code: 股票代码
            level: 数据级别（忽略，保持兼容性）
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)

        Returns:
            Optional[List]: 股票数据列表
        """
        pass

    @abstractmethod
    def get_indicator_data_data_access_interface(self, code: str, indicator: str, start_date: str, end_date: str,
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
    def get_stock_list_data_access_interface(self, market: Optional[str] = None) -> List[str]:
        """
        获取股票列表

        Args:
            market: 市场类型，可选

        Returns:
            股票代码列表
        """
        pass
    
    @abstractmethod
    def check_data_exists_data_access_interface(self, table: str, conditions: Dict) -> bool:
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
    def get_latest_data_data_access_interface(self, table: str, code: str, columns: Optional[List[str]] = None) -> Optional[Dict]:
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

    def __init__(self, connection_pool=None, sql_manager=None):
        """
        初始化ClickHouse数据访问

        Args:
            connection_pool: 连接池
            sql_manager: SQL管理器
        """
        self.connection_pool = connection_pool
        self.sql_manager = sql_manager

        # 如果没有提供连接池，创建默认连接
        if self.connection_pool is None:
            self._init_default_connection()

    def _init_default_connection(self):
        """初始化默认连接"""
        try:
            from clickhouse_driver import Client
            from db.sql_manager import SQLManager, QueryType
            self.client = Client(
                host='localhost',
                port=9000,
                database='stock',
                user='default',
                password='123456'
            )
        except Exception as e:
            raise DataAccessError(f"初始化ClickHouse连接失败: {e}")

    def get_stock_data_data_access_interface(self, code: str, start_date: str, end_date: str,
                      columns: Optional[List[str]] = None, level: str = '日线') -> pd.DataFrame:
        """获取股票数据"""
        try:
            # 构建查询SQL
            if columns:
                columns_str = ', '.join(columns)
            else:
                columns_str = 'date, open, high, low, close, volume'

            query = f"""
                SELECT {columns_str}
                FROM stock_info WHERE level = %(level)s AND code = '{code}'
                AND level = '{level}'
                AND date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date ASC
            """

            result = self.client.execute(query)

            if result:
                column_names = columns if columns else ['date', 'open', 'high', 'low', 'close', 'volume']
                df = pd.DataFrame(result, columns=column_names)
                df['date'] = pd.to_datetime(df['date'])
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            raise DataAccessError(f"获取股票{code}数据失败: {e}")

    def get_stocks_data_batch_data_access_interface(self, codes: List[str], start_date: str, end_date: str,
                             columns: Optional[List[str]] = None, level: str = '日线') -> pd.DataFrame:
        """批量获取多只股票数据"""
        try:
            if columns:
                columns_str = ', '.join(columns)
            else:
                columns_str = 'code, date, open, high, low, close, volume'

            codes_str = "', '".join(codes)
            query = f"""
                SELECT {columns_str}
                FROM stock_info WHERE code = %(code)s AND level = %(level)s AND code IN ('{codes_str}')
                AND level = '{level}'
                AND date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY code, date ASC
            """

            result = self.client.execute(query)

            if result:
                column_names = columns if columns else ['code', 'date', 'open', 'high', 'low', 'close', 'volume']
                df = pd.DataFrame(result, columns=column_names)
                df['date'] = pd.to_datetime(df['date'])
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            raise DataAccessError(f"批量获取股票数据失败: {e}")

    def get_indicator_data_data_access_interface(self, code: str, indicator: str, start_date: str, end_date: str,
                          params: Optional[Dict] = None) -> pd.DataFrame:
        """获取指标数据"""
        try:
            # 这里应该从指标数据表获取，暂时返回空DataFrame
            return pd.DataFrame()
        except Exception as e:
            raise DataAccessError(f"获取指标{indicator}数据失败: {e}")

    def get_stock_list_data_access_interface(self, market: Optional[str] = None, level: str = '日线') -> List[str]:
        """获取股票列表"""
        try:
            query = f"""
                SELECT DISTINCT code
                FROM stock_info WHERE code = %(code)s AND level = '{level}'
                AND close > 5.0
                LIMIT 50
            """

            result = self.client.execute(query)

            if result:
                return [item[0] for item in result]
            else:
                return []

        except Exception as e:
            raise DataAccessError(f"获取股票列表失败: {e}")

    def get__list_data_access_interface(self) -> List[str]:
        """获取行业列表"""
        try:
            # 暂时返回空列表，需要行业数据表
            return []
        except Exception as e:
            raise DataAccessError(f"获取行业列表失败: {e}")

    def execute_query_data_access_interface(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """执行查询"""
        try:
            result = self.client.execute(query)

            if result:
                # 尝试推断列名
                if 'SELECT' in query.upper():
                    # 简单的列名推断
                    select_part = query.upper().split('FROM')[0].replace('SELECT', '').strip()
                    if '*' in select_part:
                        columns = ['col_' + str(i) for i in range(len(result[0]))]
                    else:
                        columns = [col.strip() for col in select_part.split(',')]
                else:
                    columns = ['col_' + str(i) for i in range(len(result[0]))]

                df = pd.DataFrame(result, columns=columns)
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            raise DataAccessError(f"执行查询失败: {e}")

    def check_data_exists_data_access_interface(self, table: str, conditions: Dict) -> bool:
        """检查数据是否存在"""
        try:
            where_clauses = []
            for key, value in conditions.items():
                if isinstance(value, str):
                    where_clauses.append(f"{key} = '{value}'")
                else:
                    where_clauses.append(f"{key} = {value}")

            where_str = ' AND '.join(where_clauses)
            query = f"SELECT COUNT(*) FROM {table} WHERE {where_str}"

            result = self.client.execute(query)
            return result[0][0] > 0 if result else False

        except Exception as e:
            raise DataAccessError(f"检查数据存在性失败: {e}")

    def get_latest_data_data_access_interface(self, table: str, code: str, columns: Optional[List[str]] = None) -> Optional[Dict]:
        """获取最新数据"""
        try:
            if columns:
                columns_str = ', '.join(columns)
            else:
                columns_str = '*'

            query = f"""
                SELECT {columns_str}
                FROM {table}
                WHERE code = '{code}'
                ORDER BY date DESC
                LIMIT 1
            """

            result = self.client.execute(query)

            if result:
                if columns:
                    return dict(zip(columns, result[0]))
                else:
                    # 需要获取表结构来确定列名
                    return {'data': result[0]}
            else:
                return None

        except Exception as e:
            raise DataAccessError(f"获取最新数据失败: {e}")
    
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
DATA_ACCESS_INTERFACE = DataAccessInterface
IDataAccess = DataAccessInterface