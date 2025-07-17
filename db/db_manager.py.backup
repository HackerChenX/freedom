from db.query_executor import get_query_executor
from db.sql_manager import QueryType
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据库连接管理模块，重构为支持依赖注入的普通类
"""

from utils.dependency_injection import get_container
from utils.logger import getLogger

class DBManager:
    """数据库管理器 - 提供统一的数据库访问接口"""
    
    def __init__(self):
        """初始化数据库管理器"""
        self.logger = getLogger(__name__)
        
        try:
            # 使用依赖注入容器获取数据访问接口
            container = get_container()
            from db.interfaces.data_access_interface import IDataAccess
            self._data_access = container.resolve(IDataAccess)
            self.logger.info("数据库管理器初始化完成")
        except Exception as e:
            self.logger.error(f"初始化数据库管理器失败: {e}")
            # 降级处理：延迟创建数据访问接口
            self.logger.warning("依赖注入获取数据访问接口失败，将在首次使用时创建")
            self._data_access = None
    
    @property
    def data_access(self):
        """获取数据访问接口"""
        if self._data_access is None:
            self._ensure_data_access()
        return self._data_access
    
    def _ensure_data_access(self):
        """确保数据访问接口已初始化"""
        if self._data_access is not None:
            return
        
        try:
            # 尝试从容器获取
            container = get_container()
            from db.interfaces.data_access_interface import IDataAccess
            self._data_access = container.resolve(IDataAccess)
            self.logger.info("延迟获取数据访问接口成功")
        except Exception as e:
            self.logger.error(f"延迟获取数据访问接口失败: {e}")
            # 最后的降级处理（使用动态导入避免直接依赖）
            try:
                import importlib
                clickhouse_module = importlib.import_module('db.clickhouse_db')
                self._data_access = clickhouse_module.ClickhouseDB()
                self.logger.warning("使用最后的降级ClickhouseDB实现")
            except Exception as fallback_error:
                self.logger.error(f"最后的降级处理也失败: {fallback_error}")
                raise RuntimeError("无法初始化数据访问接口")
    
    def get_connection(self):
        """获取数据库连接 - 兼容性方法"""
        return self._data_access
    
    def query_manager(self, sql: str, params=None):
        """执行查询"""
        self._ensure_data_access()
        return self._data_access.query_manager(sql, params)
    
    def query_dataframe(self, sql: str, params=None):
        """执行查询并返回DataFrame"""
        self._ensure_data_access()
        return self._data_access.query_dataframe(sql, params)
    
    def execute_dbmanager(self, sql: str, params=None):
        """执行SQL语句"""
        self._ensure_data_access()
        return self._data_access.execute_dbmanager(sql, params)
    
    def insert_dataframe(self, table_name: str, df, **kwargs):
        """插入DataFrame数据"""
        self._ensure_data_access()
        return self._data_access.insert_dataframe(table_name, df, **kwargs)
    
    def get_stock_data_manager_db_manager(self, stock_code: str, start_date: str, end_date: str, period: str = '1d'):
        """获取股票数据"""
        self._ensure_data_access()
        return self._data_access.get_stock_data_manager_db_manager(stock_code, start_date, end_date, period)
    
    def get_stock_list_manager_db_manager(self, **kwargs):
        """获取股票列表"""
        self._ensure_data_access()
        return self._data_access.get_stock_list_manager_db_manager(**kwargs)
    
    def get_market_data(self, **kwargs):
        """获取市场数据"""
        self._ensure_data_access()
        return self._data_access.get_market_data(**kwargs)
    
    def get_last_trade_date(self):
        """获取最后交易日期"""
        self._ensure_data_access()
        return self._data_access.get_last_trade_date()
    
    def get_index_stocks(self, index_code: str):
        """获取指数成分股"""
        self._ensure_data_access()
        return self._data_access.get_index_stocks(index_code)
    
    def get_stocks_by_industry(self, industry: str):
        """按行业获取股票"""
        self._ensure_data_access()
        return self._data_access.get_stocks_by_industry(industry)


# ===== 依赖注入和兼容性接口 =====

def get_db_manager() -> DBManager:
    """
    获取数据库管理器实例（依赖注入方式）
    
    Returns:
        DBManager: 数据库管理器实例
    """
    try:
        container = get_container()
        return container.resolve(DBManager)
    except Exception as e:
        logger = getLogger(__name__)
        logger.warning(f"从依赖注入容器获取DBManager失败，创建新实例: {e}")
        return DBManager()


# 注册到依赖注入容器
try:
    container = get_container()
    if not container.is_registered(DBManager):
        container.register_singleton(DBManager, DBManager)
        logger = getLogger(__name__)
        logger.info("DBManager已注册到依赖注入容器")
except Exception as e:
    logger = getLogger(__name__)
    logger.warning(f"注册DBManager到依赖注入容器失败: {e}")