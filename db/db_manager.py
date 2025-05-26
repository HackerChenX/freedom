"""
数据库连接管理模块，使用单例模式管理数据库连接
"""

from db.clickhouse_db import get_clickhouse_db
from config import get_config

class DBManager:
    """
    数据库管理器，单例模式
    """
    _instance = None
    _db = None
    
    @classmethod
    def get_instance(cls):
        """
        获取DBManager实例
        
        Returns:
            DBManager: 单例实例
        """
        if cls._instance is None:
            cls._instance = DBManager()
        return cls._instance
    
    def __init__(self):
        """
        初始化数据库连接
        """
        if DBManager._instance is not None:
            raise Exception("DBManager是单例类，请使用get_instance()方法获取实例")
        
        DBManager._instance = self
        self.init_db()
    
    def init_db(self):
        """
        初始化数据库连接
        """
        db_config = get_config('db')
        self._db = get_clickhouse_db(config=db_config)
    
    @property
    def db(self):
        """
        获取数据库连接
        
        Returns:
            ClickHouseDB: 数据库连接实例
        """
        if self._db is None:
            self.init_db()
        return self._db
    
    def get_stock_info(self, stock_code, level, start, end):
        """
        获取股票信息
        
        Args:
            stock_code: 股票代码
            level: K线周期
            start: 开始日期
            end: 结束日期
            
        Returns:
            list: 股票数据列表
        """
        return self.db.get_stock_info(stock_code, level, start, end)
    
    def get_industry_info(self, symbol, start, end):
        """
        获取行业信息
        
        Args:
            symbol: 行业代码
            start: 开始日期
            end: 结束日期
            
        Returns:
            list: 行业数据列表
        """
        return self.db.get_industry_info(symbol, start, end)
    
    def save_stock_info(self, data, level):
        """
        保存股票信息
        
        Args:
            data: 股票数据
            level: K线周期
        """
        return self.db.save_stock_info(data, level)
    
    def save_industry_info(self, data):
        """
        保存行业信息
        
        Args:
            data: 行业数据
        """
        return self.db.save_industry_info(data)
    
    def get_stock_max_date(self):
        """
        获取股票最大日期
        
        Returns:
            datetime: 最大日期
        """
        return self.db.get_stock_max_date()
    
    def get_industry_max_date(self):
        """
        获取行业最大日期
        
        Returns:
            datetime: 最大日期
        """
        return self.db.get_industry_max_date()
    
    def get_industry_stock(self, industry):
        """
        获取行业股票列表
        
        Args:
            industry: 行业名称
            
        Returns:
            list: 股票代码列表
        """
        return self.db.get_industry_stock(industry)
    
    def get_avg_price(self, code, start):
        """
        获取平均价格
        
        Args:
            code: 股票代码
            start: 开始日期
            
        Returns:
            float: 平均价格
        """
        return self.db.get_avg_price(code, start) 