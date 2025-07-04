"""
数据访问管理器实现

实现IDataAccess接口，提供统一的数据访问服务
"""

import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd

from db.interfaces.data_access_interface import IDataAccess, IStockDataProvider, IMarketDataProvider
from db.interfaces.cache_interface import ICacheManager
from db.interfaces.connection_interface import IConnectionManager
from db.container import get_container
from models.stock_info import StockInfo
from enums.period import Period
from utils.logger import get_logger
from utils.decorators import performance_monitor
from utils.exceptions import DataAccessError, DataValidationError

logger = get_logger(__name__)


class DataAccessManager(IDataAccess, IStockDataProvider, IMarketDataProvider):
    """
    数据访问管理器
    
    实现统一的数据访问接口，集成缓存和连接管理
    """
    
    def __init__(self, 
                 connection_manager: Optional[IConnectionManager] = None,
                 cache_manager: Optional[ICacheManager] = None):
        """
        初始化数据访问管理器
        
        Args:
            connection_manager: 连接管理器
            cache_manager: 缓存管理器
        """
        # 使用依赖注入容器获取依赖
        container = get_container()
        
        self.connection_manager = connection_manager or container.resolve(IConnectionManager)
        self.cache_manager = cache_manager or container.resolve(ICacheManager)
        
        logger.info("数据访问管理器初始化完成")
    
    @performance_monitor(threshold=1.0)
    def get_stock_info(self, 
                       stock_code: Union[str, List[str]] = None,
                       level: Union[str, Period] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       filters: Optional[Dict[str, Any]] = None,
                       limit: Optional[int] = None,
                       order_by: str = "date DESC") -> StockInfo:
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
            StockInfo: 股票数据对象
        """
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key('stock_info', {
                'stock_code': stock_code,
                'level': level,
                'start_date': start_date,
                'end_date': end_date,
                'filters': filters,
                'limit': limit,
                'order_by': order_by
            })
            
            # 尝试从缓存获取
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"从缓存获取股票数据: {cache_key}")
                return cached_result
            
            # 构建查询
            query, params = self._build_stock_info_query(
                stock_code, level, start_date, end_date, filters, limit, order_by
            )
            
            # 执行查询
            result_df = self.query(query, params)
            
            # 创建StockInfo对象
            stock_info = StockInfo(result_df)
            
            # 缓存结果
            self.cache_manager.set(cache_key, stock_info, ttl=300)  # 5分钟缓存
            
            return stock_info
            
        except Exception as e:
            logger.error(f"获取股票数据失败: {e}")
            raise DataAccessError(f"获取股票数据失败: {e}")
    
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
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key('stock_list', {
                'industry': industry,
                'market': market,
                'filters': filters
            })
            
            # 尝试从缓存获取
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 构建查询
            query, params = self._build_stock_list_query(industry, market, filters)
            
            # 执行查询
            result_df = self.query(query, params)
            
            # 提取股票代码列表
            stock_list = result_df['code'].unique().tolist() if not result_df.empty else []
            
            # 缓存结果
            self.cache_manager.set(cache_key, stock_list, ttl=600)  # 10分钟缓存
            
            return stock_list
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise DataAccessError(f"获取股票列表失败: {e}")
    
    def get_industry_list(self) -> pd.DataFrame:
        """
        获取行业列表
        
        Returns:
            pd.DataFrame: 行业列表数据
        """
        try:
            cache_key = "industry_list"
            
            # 尝试从缓存获取
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 构建查询
            query = """
            SELECT DISTINCT industry
            FROM stock_info 
            WHERE industry != '' AND industry IS NOT NULL
            ORDER BY industry
            """
            
            # 执行查询
            result_df = self.query(query)
            
            # 缓存结果
            self.cache_manager.set(cache_key, result_df, ttl=3600)  # 1小时缓存
            
            return result_df
            
        except Exception as e:
            logger.error(f"获取行业列表失败: {e}")
            raise DataAccessError(f"获取行业列表失败: {e}")
    
    def query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        执行SQL查询
        
        Args:
            sql: SQL查询语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
        """
        try:
            with self.connection_manager.get_connection() as conn:
                return conn.query(sql, params or {})
        except Exception as e:
            logger.error(f"查询执行失败: {sql}, 错误: {e}")
            raise DataAccessError(f"查询执行失败: {e}")
    
    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        执行SQL语句
        
        Args:
            sql: SQL语句
            params: 执行参数
        """
        try:
            with self.connection_manager.get_connection() as conn:
                conn.execute(sql, params or {})
        except Exception as e:
            logger.error(f"SQL执行失败: {sql}, 错误: {e}")
            raise DataAccessError(f"SQL执行失败: {e}")
    
    def test_connection(self) -> bool:
        """
        测试数据库连接
        
        Returns:
            bool: 连接状态
        """
        return self.connection_manager.test_connection()
    
    def get_stock_max_date(self) -> datetime:
        """
        获取股票数据最新日期
        
        Returns:
            datetime: 最新日期
        """
        try:
            query = "SELECT MAX(date) as max_date FROM stock_info"
            result = self.query(query)
            
            if result.empty or pd.isna(result.iloc[0, 0]):
                return datetime.now()
            
            return result.iloc[0, 0]
            
        except Exception as e:
            logger.error(f"获取股票最新日期失败: {e}")
            raise DataAccessError(f"获取股票最新日期失败: {e}")
    
    def get_industry_max_date(self) -> datetime:
        """
        获取行业数据最新日期
        
        Returns:
            datetime: 最新日期
        """
        try:
            query = """
            SELECT MAX(date) as max_date 
            FROM stock_info 
            WHERE industry != '' AND industry IS NOT NULL
            """
            result = self.query(query)
            
            if result.empty or pd.isna(result.iloc[0, 0]):
                return datetime.now()
            
            return result.iloc[0, 0]
            
        except Exception as e:
            logger.error(f"获取行业最新日期失败: {e}")
            raise DataAccessError(f"获取行业最新日期失败: {e}")
    
    def get_avg_price(self, code: str, start_date: Union[str, datetime]) -> float:
        """
        获取股票平均价格
        
        Args:
            code: 股票代码
            start_date: 开始日期
            
        Returns:
            float: 平均价格
        """
        try:
            # 格式化日期
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            
            query = """
            SELECT AVG(close) as avg_price
            FROM stock_info
            WHERE code = %(code)s AND date >= %(start_date)s
            """
            
            params = {'code': code, 'start_date': start_date}
            result = self.query(query, params)
            
            if result.empty or pd.isna(result.iloc[0, 0]):
                return 0.0
            
            return float(result.iloc[0, 0])
            
        except Exception as e:
            logger.error(f"获取股票平均价格失败: {e}")
            raise DataAccessError(f"获取股票平均价格失败: {e}")
    
    # IStockDataProvider接口实现
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
        stock_info = self.get_stock_info(
            stock_code=code,
            level=level,
            start_date=start_date,
            end_date=end_date,
            order_by="date ASC"
        )
        return stock_info.data
    
    def get_stock_basic_info(self, codes: List[str]) -> pd.DataFrame:
        """
        获取股票基本信息
        
        Args:
            codes: 股票代码列表
            
        Returns:
            pd.DataFrame: 股票基本信息
        """
        try:
            if not codes:
                return pd.DataFrame()
            
            # 构建查询
            code_list = "', '".join(codes)
            query = f"""
            SELECT DISTINCT code, name, industry
            FROM stock_info
            WHERE code IN ('{code_list}')
            """
            
            return self.query(query)
            
        except Exception as e:
            logger.error(f"获取股票基本信息失败: {e}")
            raise DataAccessError(f"获取股票基本信息失败: {e}")
    
    # IMarketDataProvider接口实现
    def get_market_overview(self, date: str) -> Dict[str, Any]:
        """
        获取市场概览数据
        
        Args:
            date: 查询日期
            
        Returns:
            Dict[str, Any]: 市场概览数据
        """
        try:
            query = """
            SELECT 
                COUNT(*) as total_stocks,
                AVG(price_change) as avg_change,
                SUM(CASE WHEN price_change > 0 THEN 1 ELSE 0 END) as rising_count,
                SUM(CASE WHEN price_change < 0 THEN 1 ELSE 0 END) as falling_count
            FROM stock_info
            WHERE date = %(date)s AND level = '日线'
            """
            
            result = self.query(query, {'date': date})
            
            if result.empty:
                return {}
            
            return result.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"获取市场概览失败: {e}")
            raise DataAccessError(f"获取市场概览失败: {e}")
    
    def get_industry_performance(self, date: str) -> pd.DataFrame:
        """
        获取行业表现数据
        
        Args:
            date: 查询日期
            
        Returns:
            pd.DataFrame: 行业表现数据
        """
        try:
            query = """
            SELECT 
                industry,
                COUNT(*) as stock_count,
                AVG(price_change) as avg_change,
                AVG(turnover_rate) as avg_turnover
            FROM stock_info
            WHERE date = %(date)s AND level = '日线' 
            AND industry != '' AND industry IS NOT NULL
            GROUP BY industry
            ORDER BY avg_change DESC
            """
            
            return self.query(query, {'date': date})
            
        except Exception as e:
            logger.error(f"获取行业表现失败: {e}")
            raise DataAccessError(f"获取行业表现失败: {e}")
    
    # 私有辅助方法
    def _generate_cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """
        生成缓存键
        
        Args:
            prefix: 前缀
            params: 参数字典
            
        Returns:
            str: 缓存键
        """
        import hashlib
        import json
        
        # 序列化参数
        params_str = json.dumps(params, sort_keys=True, default=str)
        
        # 生成哈希
        hash_obj = hashlib.md5(params_str.encode())
        
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def _build_stock_info_query(self, 
                               stock_code: Union[str, List[str]],
                               level: Union[str, Period],
                               start_date: Optional[str],
                               end_date: Optional[str],
                               filters: Optional[Dict[str, Any]],
                               limit: Optional[int],
                               order_by: str) -> tuple:
        """
        构建股票信息查询
        
        Returns:
            tuple: (query, params)
        """
        # 基础查询
        query = """
        SELECT code, name, date, level, open, high, low, close, volume, 
               turnover_rate, price_change, price_range, industry
        FROM stock_info
        WHERE 1=1
        """
        
        params = {}
        
        # 股票代码条件
        if stock_code:
            if isinstance(stock_code, str):
                query += " AND code = %(code)s"
                params['code'] = stock_code
            else:
                code_list = "', '".join(stock_code)
                query += f" AND code IN ('{code_list}')"
        
        # K线周期条件
        if level:
            level_str = str(level) if isinstance(level, Period) else level
            query += " AND level = %(level)s"
            params['level'] = level_str
        
        # 日期范围条件
        if start_date:
            query += " AND date >= %(start_date)s"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND date <= %(end_date)s"
            params['end_date'] = end_date
        
        # 其他过滤条件
        if filters:
            for key, value in filters.items():
                if key in ['industry', 'market']:
                    query += f" AND {key} = %({key})s"
                    params[key] = value
        
        # 排序
        query += f" ORDER BY {order_by}"
        
        # 限制条数
        if limit:
            query += f" LIMIT {limit}"
        
        return query, params
    
    def _build_stock_list_query(self, 
                               industry: Optional[str],
                               market: Optional[str],
                               filters: Optional[Dict[str, Any]]) -> tuple:
        """
        构建股票列表查询
        
        Returns:
            tuple: (query, params)
        """
        query = """
        SELECT DISTINCT code
        FROM stock_info
        WHERE 1=1
        """
        
        params = {}
        
        if industry:
            query += " AND industry = %(industry)s"
            params['industry'] = industry
        
        if market:
            query += " AND market = %(market)s"
            params['market'] = market
        
        if filters:
            for key, value in filters.items():
                query += f" AND {key} = %({key})s"
                params[key] = value
        
        query += " ORDER BY code"
        
        return query, params 