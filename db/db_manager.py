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
    
    def __new__(cls):
        """
        实现单例模式
        
        Returns:
            DBManager: 单例实例
        """
        if cls._instance is None:
            cls._instance = super(DBManager, cls).__new__(cls)
            cls._instance.init_db()
        return cls._instance
    
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
        初始化方法，由于使用__new__实现单例，这里不需要做额外工作
        """
        pass
    
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
    
    def get_stock_info(self, stock_code, level, start_date, end_date):
        """
        获取股票信息
        
        Args:
            stock_code: 股票代码
            level: K线周期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            list: 股票数据列表
        """
        return self.db.get_stock_info(stock_code, level, start_date, end_date)
    
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
    
    def get_stock_name(self, stock_code):
        """
        获取股票名称
        
        Args:
            stock_code: 股票代码
            
        Returns:
            str: 股票名称，如果未找到则返回股票代码
        """
        try:
            # 获取股票列表
            stocks_df = self.db.get_stock_list()
            # 标准化列名
            if 'stock_code' in stocks_df.columns and 'stock_name' in stocks_df.columns:
                stocks_df = stocks_df.rename(columns={'stock_code': 'code', 'stock_name': 'name'})
            # 过滤出匹配的股票
            matched_stocks = stocks_df[stocks_df['code'] == stock_code]
            # 如果找到匹配的股票，返回名称
            if not matched_stocks.empty:
                return matched_stocks.iloc[0]['name']
            # 未找到匹配的股票，返回股票代码
            return stock_code
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"获取股票 {stock_code} 名称失败: {e}")
            return stock_code
            
    def get_trading_dates(self, stock_code, start_date, end_date):
        """
        获取交易日期列表
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            list: 交易日期列表
        """
        try:
            # 获取日线数据
            df = self.db.get_stock_info(stock_code, 'day', start_date, end_date)
            
            # 如果数据为空，返回空列表
            if df.empty:
                return []
                
            # 返回日期列表
            return df['date'].tolist()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"获取股票 {stock_code} 交易日期失败: {e}")
            return []
            
    def get_stock_list(self):
        """
        获取股票列表
        
        Returns:
            DataFrame: 股票列表
        """
        return self.db.get_stock_list() 