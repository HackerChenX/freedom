"""
数据访问管理器实现

实现DataAccessInterface接口，提供统一的数据访问服务
"""

import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import pandas as pd

from config import get_config
from db.interfaces.data_access_interface import DataAccessInterface
from utils.dependency_injection import get_container
from models.stock_info import StockInfo
from enums.period import Period
from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.exceptions import DataAccessError

logger = get_logger(__name__)


class DataAccessManager(DataAccessInterface):
    """
    数据访问管理器
    
    实现统一的数据访问接口，集成缓存和连接管理
    """
    
    def __init__(self, 
                 connection_manager=None,
                 cache_service=None):
        """
        初始化数据访问管理器
        
        Args:
            connection_manager: 连接管理器
            cache_service: 缓存服务
        """
        self.connection_manager = connection_manager
        self.cache_service = cache_service
        
        logger.info("数据访问管理器初始化完成")
    
    @exception_handler(reraise=False, default_return=pd.DataFrame())
    def get_stock_data_data_access_manager(self, code: str, start_date: str, end_date: str, 
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
        try:
            from db.unified_data_manager import get_unified_data_manager
            data_manager = get_unified_data_manager()
            return data_manager.get_stock_daily_data(code, start_date, end_date)
        except Exception as e:
            logger.error(f"获取股票数据失败: {e}")
            return pd.DataFrame()
    
    # 添加标准接口方法 - 这是其他模块期望的方法名
    @exception_handler(reraise=False, default_return=pd.DataFrame())
    def get_stock_data(self, code: str = None, stock_code: str = None, 
                      start_date: str = None, end_date: str = None, 
                      columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """
        获取股票数据 - 标准接口方法
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            columns: 需要的列名列表，None表示所有列
            
        Returns:
            股票数据DataFrame
        """
        return self.get_stock_data_data_access_manager(code, start_date, end_date, columns)
    
    @exception_handler(reraise=False, default_return=pd.DataFrame())
    def get_stocks_data_batch_data_access_manager(self, codes: List[str], start_date: str, end_date: str,
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
        try:
            from db.unified_data_manager import get_unified_data_manager
            data_manager = get_unified_data_manager()
            return data_manager.get_stocks_data_batch(codes, start_date, end_date)
        except Exception as e:
            logger.error(f"批量获取股票数据失败: {e}")
            return pd.DataFrame()
    
    @exception_handler(reraise=False, default_return=pd.DataFrame())
    def get_indicator_data_data_access_manager(self, code: str, indicator: str, start_date: str, end_date: str,
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
        try:
            # 这里可以集成指标计算系统
            logger.info(f"获取指标数据: {indicator} for {code}")
            return pd.DataFrame()  # 暂时返回空DataFrame
        except Exception as e:
            logger.error(f"获取指标数据失败: {e}")
            return pd.DataFrame()
    
    @exception_handler(reraise=False, default_return=[])
    def get_stock_list(self, industry: Optional[str] = None, 
                      market: Optional[str] = None,
                      limit: Optional[int] = None) -> List[str]:
        """
        获取股票列表（标准接口）
        
        Args:
            industry: 行业筛选
            market: 市场筛选
            limit: 限制返回数量
            
        Returns:
            股票代码列表
        """
        result = self.get_stock_list_data_access_manager(industry=industry, market=market)
        if limit and len(result) > limit:
            return result[:limit]
        return result
    
    @exception_handler(reraise=False, default_return=[])
    def get_stock_list_data_access_manager(self, industry: Optional[str] = None, 
                      market: Optional[str] = None) -> List[str]:
        """
        获取股票列表
        
        Args:
            industry: 行业筛选
            market: 市场筛选
            
        Returns:
            股票代码列表
        """
        try:
            from db.unified_data_manager import get_unified_data_manager
            data_manager = get_unified_data_manager()
            return data_manager.get_stock_list()
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    @exception_handler(reraise=False, default_return=[])
    def get_industry_list_data_access_manager(self) -> List[str]:
        """
        获取行业列表
        
        Returns:
            行业列表
        """
        try:
            from db.unified_data_manager import get_unified_data_manager
            data_manager = get_unified_data_manager()
            return data_manager.get_industry_list()
        except Exception as e:
            logger.error(f"获取行业列表失败: {e}")
            return []
    
    @exception_handler(reraise=False, default_return=pd.DataFrame())
    def execute_query_data_access_manager(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        执行查询
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果DataFrame
        """
        try:
            from db.unified_data_manager import get_unified_data_manager
            data_manager = get_unified_data_manager()
            return data_manager.execute_query(query, params)
        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            return pd.DataFrame()
    
    @exception_handler(reraise=False, default_return=False)
    def check_data_exists_data_access_manager(self, table: str, conditions: Dict) -> bool:
        """
        检查数据是否存在
        
        Args:
            table: 表名
            conditions: 检查条件
            
        Returns:
            数据是否存在
        """
        try:
            # 构建查询来检查数据是否存在
            where_clauses = []
            for key, value in conditions.items():
                where_clauses.append(f"{key} = '{value}'")
            
            where_sql = " AND ".join(where_clauses)
            query = f"SELECT COUNT(*) as count FROM {table} WHERE {where_sql} LIMIT 1"
            
            result = self.execute_query(query)
            return len(result) > 0 and result.iloc[0]['count'] > 0
        except Exception as e:
            logger.error(f"检查数据存在性失败: {e}")
            return False
    
    @exception_handler(reraise=False, default_return=None)
    def get_latest_data_data_access_manager(self, table: str, code: str, columns: Optional[List[str]] = None) -> Optional[Dict]:
        """
        获取最新数据
        
        Args:
            table: 表名
            code: 股票代码
            columns: 需要的列名
            
        Returns:
            最新数据字典
        """
        try:
            col_str = "*" if not columns else ", ".join(columns)
            query = f"SELECT {col_str} FROM {table} WHERE code = '{code}' ORDER BY date DESC LIMIT 1"
            
            result = self.execute_query(query)
            if len(result) > 0:
                return result.iloc[0].to_dict()
            return None
        except Exception as e:
            logger.error(f"获取最新数据失败: {e}")
            return None
    
    @performance_monitor(threshold=1.0)
    def get_stock_info_Manager_Data_Access_Manager(self, 
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
            Stock_info: 股票数据对象
        """
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key_Data_Access_Manager('stock_info', {
                'stock_code': stock_code,
                'level': level,
                'start_date': start_date,
                'end_date': end_date,
                'filters': filters,
                'limit': limit,
                'order_by': order_by
            })
            
            # 尝试从缓存获取
            if self.cache_service:
                cached_result = self.cache_service.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"从缓存获取股票数据: {cache_key}")
                    return cached_result
            
            # 构建查询
            query, params = self._build_stock_info_query(
                stock_code, level, start_date, end_date, filters, limit, order_by
            )
            
            # 执行查询
            result_df = self.query_Manager_Data_Access_Manager(query, params)
            
            # 创建Stock_info对象
            stock_info = StockInfo(result_df)
            
            # 缓存结果
            if self.cache_service:
                self.cache_service.set(cache_key, stock_info, ttl=300)  # 5分钟缓存
            
            return stock_info
            
        except Exception as e:
            logger.error(f"获取股票数据失败: {e}")
            raise DataAccessError(f"获取股票数据失败: {e}")
    
    def get_stock_list_Manager_Data_Access_Manager(self, 
                       industry: Optional[str] = None,
                       market: Optional[str] = None,
                       only_active: bool = True) -> List[str]:
        """
        获取股票列表
        
        Args:
            industry: 行业筛选
            market: 市场筛选
            only_active: 是否只返回活跃股票
            
        Returns:
            List[str]: 股票代码列表
        """
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key_Data_Access_Manager('stock_list', {
                'industry': industry,
                'market': market,
                'only_active': only_active
            })
            
            # 尝试从缓存获取
            if self.cache_service:
                cached_result = self.cache_service.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"从缓存获取股票列表: {cache_key}")
                    return cached_result
            
            # 构建查询
            query = """
            SELECT DISTINCT code
            FROM stock_info
            WHERE 1=1
            """
            
            params = {}
            
            if industry:
                query += " AND industry = %(industry)s"
                params['industry'] = industry
            
            if only_active:
                # 只返回最近有交易的股票
                query += " AND date >= '2020-01-01'"
            
            query += " ORDER BY code"
            
            # 执行查询
            result_df = self.query_Manager_Data_Access_Manager(query, params)
            
            # 提取股票代码列表
            stock_list = result_df['code'].tolist() if not result_df.empty else []
            
            # 缓存结果
            if self.cache_service:
                self.cache_service.set(cache_key, stock_list, ttl=600)  # 10分钟缓存
            
            return stock_list
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise DataAccessError(f"获取股票列表失败: {e}")
    
    def get_industry_dataframe(self) -> pd.DataFrame:
        """
        获取行业列表Data_frame
        
        Returns:
            pd.DataFrame: 行业列表数据
        """
        try:
            cache_key = "industry_list"
            
            # 尝试从缓存获取
            if self.cache_service:
                cached_result = self.cache_service.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # 构建查询
            query = """
            SELECT DISTINCT industry
            FROM stock_info WHERE 1=1
            WHERE industry != '' AND industry IS NOT NULL
            ORDER BY industry
            """
            
            # 执行查询
            result_df = self.query_Manager_Data_Access_Manager(query)
            
            # 缓存结果
            if self.cache_service:
                self.cache_service.set(cache_key, result_df, ttl=3600)  # 1小时缓存
            
            return result_df
            
        except Exception as e:
            logger.error(f"获取行业列表失败: {e}")
            raise DataAccessError(f"获取行业列表失败: {e}")
    
    def query_Manager_Data_Access_Manager(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
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
                return conn.query_Manager_Data_Access_Manager(sql, params or {})
        except Exception as e:
            logger.error(f"查询执行失败: {sql}, 错误: {e}")
            raise DataAccessError(f"查询执行失败: {e}")
    
    def query_dataframe_Manager_Data_Access_Manager(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        执行SQL查询并返回Data_frame（query方法的别名）
        
        Args:
            sql: SQL查询语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
        """
        return self.query_Manager_Data_Access_Manager(sql, params)
    
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
    
    def test_connection_Manager_Data_Access_Manager(self) -> bool:
        """
        测试数据库连接
        
        Returns:
            bool: 连接状态
        """
        return self.connection_manager.test_connection_Manager_Data_Access_Manager()
    
    def get_stock_max_date_Manager(self) -> datetime:
        """
        获取股票数据最新日期
        
        Returns:
            datetime: 最新日期
        """
        try:
            query = "SELECT MAX(date) as max_date FROM stock_info WHERE date >= '2020-01-01'"
            result = self.query_Manager_Data_Access_Manager(query)
            
            if result.empty or pd.isna(result.iloc[0, 0]):
                return datetime.now()
            
            return result.iloc[0, 0]
            
        except Exception as e:
            logger.error(f"获取股票最新日期失败: {e}")
            raise DataAccessError(f"获取股票最新日期失败: {e}")
    
    def get_industry_max_date_Manager(self) -> datetime:
        """
        获取行业数据最新日期
        
        Returns:
            datetime: 最新日期
        """
        try:
            query = """
            SELECT MAX(date) as max_date 
            FROM stock_info WHERE 1=1
            WHERE industry != '' AND industry IS NOT NULL
            """
            result = self.query_Manager_Data_Access_Manager(query)
            
            if result.empty or pd.isna(result.iloc[0, 0]):
                return datetime.now()
            
            return result.iloc[0, 0]
            
        except Exception as e:
            logger.error(f"获取行业最新日期失败: {e}")
            raise DataAccessError(f"获取行业最新日期失败: {e}")
    
    def get_avg_price_Manager(self, code: str, start_date: Union[str, datetime]) -> float:
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
            FROM stock_info WHERE 1=1
            WHERE code = %(code)s AND date >= %(start_date)s
            """
            
            params = {'code': code, 'start_date': start_date}
            result = self.query_Manager_Data_Access_Manager(query, params)
            
            if result.empty or pd.isna(result.iloc[0, 0]):
                return 0.0
            
            return float(result.iloc[0, 0])
            
        except Exception as e:
            logger.error(f"获取股票平均价格失败: {e}")
            raise DataAccessError(f"获取股票平均价格失败: {e}")
    
    # IStockDataProvider接口实现
    def get_kline_data_Manager_Data_Access_Manager(self, 
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
        stock_info = self.get_stock_info_Manager_Data_Access_Manager(
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
            FROM stock_info WHERE 1=1
            WHERE code IN ('{code_list}')
            """
            
            return self.query_Manager_Data_Access_Manager(query)
            
        except Exception as e:
            logger.error(f"获取股票基本信息失败: {e}")
            raise DataAccessError(f"获取股票基本信息失败: {e}")
    
    # IMarketDataProvider接口实现
    def get_market_overview_Manager(self, date: str) -> Dict[str, Any]:
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
            FROM stock_info WHERE 1=1
            WHERE date = %(date)s AND level = '日线'
            """
            
            result = self.query_Manager_Data_Access_Manager(query, {'date': date})
            
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
                AVG(turnover) as avg_turnover
            FROM stock_info WHERE 1=1
            WHERE date = %(date)s AND level = '日线' 
            AND industry != '' AND industry IS NOT NULL
            GROUP BY industry
            ORDER BY avg_change DESC
            """
            
            return self.query_Manager_Data_Access_Manager(query, {'date': date})
            
        except Exception as e:
            logger.error(f"获取行业表现失败: {e}")
            raise DataAccessError(f"获取行业表现失败: {e}")
    
    # 私有辅助方法
    def _generate_cache_key_Data_Access_Manager(self, prefix: str, params: Dict[str, Any]) -> str:
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
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               filters: Optional[Dict[str, Any]] = None,
                               limit: Optional[int] = None,
                               order_by: str = "date DESC") -> Tuple[str, Dict[str, Any]]:
        """
        构建股票信息查询语句
        
        Args:
            stock_code: 股票代码或股票代码列表
            level: K线周期
            start_date: 开始日期
            end_date: 结束日期
            filters: 过滤条件
            limit: 限制返回记录数
            order_by: 排序规则
            
        Returns:
            Tuple[str, Dict[str, Any]]: 查询语句和参数
        """
        # 基础查询
        query = """
        SELECT code, name, date, level, open, high, low, close, volume, 
               turnover, price_change, price_range, industry
        FROM stock_info
        WHERE 1=1
        """
        
        params = {}
        
        # 股票代码条件
        if stock_code:
            if isinstance(stock_code, str):
                query += " AND code = %(stock_code)s"
                params['stock_code'] = stock_code
            elif isinstance(stock_code, list):
                placeholders = ", ".join([f"%(code_{i})s" for i in range(len(stock_code))])
                query += f" AND code IN ({placeholders})"
                for i, code in enumerate(stock_code):
                    params[f'code_{i}'] = code
        
        # K线周期条件
        if level:
            if isinstance(level, Period):
                level_str = level.value
            else:
                level_str = str(level)
            query += " AND level = %(level)s"
            params['level'] = level_str
        
        # 日期条件
        if start_date:
            query += " AND date >= %(start_date)s"
            params['start_date'] = start_date
            
        if end_date:
            query += " AND date <= %(end_date)s"
            params['end_date'] = end_date
        
        # 额外过滤条件
        if filters:
            for key, value in filters.items():
                if value is not None:
                    query += f" AND {key} = %({key})s"
                    params[key] = value
        
        # 排序
        query += f" ORDER BY {order_by}"
        
        # 限制
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
        query_executor.get_stock_list()
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
    
    def get_stock_data_Manager_Data_Access_Manager(self, 
                       stock_code: str, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None,
                       period: str = 'daily', 
                       lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        获取股票数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 周期
            lookback_days: 向前获取的天数
            
        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key_Data_Access_Manager('stock_data', {
                'stock_code': stock_code,
                'start_date': start_date,
                'end_date': end_date,
                'period': period,
                'lookback_days': lookback_days
            })
            
            # 尝试从缓存获取
            if self.cache_service:
                cached_result = self.cache_service.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"从缓存获取股票数据: {cache_key}")
                    return cached_result
            
            # 转换周期参数
            level = self._convert_period_to_level_Data_Access_Manager(period)
            
            # 获取股票数据
            stock_info = self.get_stock_info_Manager_Data_Access_Manager(
                stock_code=stock_code,
                level=level,
                start_date=start_date,
                end_date=end_date
            )
            
            # 转换为DataFrame
            result_df = stock_info.to_dataframe()
            
            # 缓存结果
            if self.cache_service:
                self.cache_service.set(cache_key, result_df, ttl=300)  # 5分钟缓存
            
            return result_df
            
        except Exception as e:
            logger.error(f"获取股票数据失败: {e}")
            raise DataAccessError(f"获取股票数据失败: {e}")
    
    def get_market_data_Manager(self, date: str) -> Dict[str, Any]:
        """
        获取市场数据
        
        Args:
            date: 查询日期
            
        Returns:
            Dict[str, Any]: 市场数据
        """
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key_Data_Access_Manager('market_data', {'date': date})
            
            # 尝试从缓存获取
            if self.cache_service:
                cached_result = self.cache_service.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # 获取市场概览数据
            market_overview = self.get_market_overview_Manager(date)
            
            # 缓存结果
            if self.cache_service:
                self.cache_service.set(cache_key, market_overview, ttl=600)  # 10分钟缓存
            
            return market_overview
            
        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            raise DataAccessError(f"获取市场数据失败: {e}")
    
    def get_last_trade_date_Manager(self) -> str:
        """
        获取最后交易日期
        
        Returns:
            str: 最后交易日期
        """
        try:
            # 生成缓存键
            cache_key = "last_trade_date"
            
            # 尝试从缓存获取
            if self.cache_service:
                cached_result = self.cache_service.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # 获取最新股票数据的日期
            max_date = self.get_stock_max_date_Manager()
            last_trade_date = max_date.strftime('%Y-%m-%d')
            
            # 缓存结果
            if self.cache_service:
                self.cache_service.set(cache_key, last_trade_date, ttl=3600)  # 1小时缓存
            
            return last_trade_date
            
        except Exception as e:
            logger.error(f"获取最后交易日期失败: {e}")
            raise DataAccessError(f"获取最后交易日期失败: {e}")
    
    def get_index_stocks_Manager(self, index_code: str) -> List[str]:
        """
        获取指数成分股
        
        Args:
            index_code: 指数代码
            
        Returns:
            List[str]: 成分股列表
        """
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key_Data_Access_Manager('index_stocks', {'index_code': index_code})
            
            # 尝试从缓存获取
            if self.cache_service:
                cached_result = self.cache_service.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # 构建查询SQL
            query = """
            SELECT DISTINCT stock_code
            FROM index_stocks 
            WHERE index_code = %(index_code)s
            ORDER BY stock_code
            """
            
            # 执行查询
            result_df = self.query_Manager_Data_Access_Manager(query, {'index_code': index_code})
            
            # 提取股票代码列表
            stock_list = result_df['stock_code'].tolist() if not result_df.empty else []
            
            # 缓存结果
            if self.cache_service:
                self.cache_service.set(cache_key, stock_list, ttl=3600)  # 1小时缓存
            
            return stock_list
            
        except Exception as e:
            logger.error(f"获取指数成分股失败: {e}")
            raise DataAccessError(f"获取指数成分股失败: {e}")
    
    def get_stocks_by_industry_Manager(self, industry: str) -> List[str]:
        """
        根据行业获取股票列表
        
        Args:
            industry: 行业名称
            
        Returns:
            List[str]: 股票代码列表
        """
        try:
            # 使用get_stock_list方法获取行业股票
            stock_list = self.get_stock_list_Manager_Data_Access_Manager(industry=industry)
            return stock_list
            
        except Exception as e:
            logger.error(f"根据行业获取股票列表失败: {e}")
            raise DataAccessError(f"根据行业获取股票列表失败: {e}")
    
    def _convert_period_to_level_Data_Access_Manager(self, period: str) -> str:
        """
        转换周期参数到level格式
        
        Args:
            period: 周期参数 (daily, weekly, monthly)
            
        Returns:
            str: level格式 (日线, 周线, 月线)
        """
        period_mapping = {
            'daily': '日线',
            'weekly': '周线', 
            'monthly': '月线',
            '1d': '日线',
            '1w': '周线',
            '1m': '月线'
        }
        
        return period_mapping.get(period.lower(), '日线')

    # 实现IDataAccess接口的抽象方法
    
    def get_stocks_data_batch_Manager(self, codes: List[str], start_date: str, 
                             end_date: str, level: str = '日线') -> Dict[str, pd.DataFrame]:
        """批量获取多只股票的K线数据
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期  
            level: K线级别
            
        Returns:
            Dict[str, pd.DataFrame]: 股票代码到数据的映射
        """
        try:
            if not codes:
                return {}
            
            # 生成缓存键
            cache_key = self._generate_cache_key_Data_Access_Manager('stocks_batch', {
                'codes': sorted(codes),  # 排序确保缓存键一致性
                'start_date': start_date,
                'end_date': end_date,
                'level': level
            })
            
            # 尝试从缓存获取
            if self.cache_service:
                cached_result = self.cache_service.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"从缓存获取批量股票数据: {cache_key}")
                    return cached_result
            
            # 构建批量查询
            code_list = "', '".join(codes)
            query = f"""
            SELECT code, date, open, high, low, close, volume, amount, 
                   price_change, price_change_pct, name, industry
            FROM stock_info WHERE 1=1
            WHERE code IN ('{code_list}')
              AND level = %(level)s
              AND date >= %(start_date)s
              AND date <= %(end_date)s
            ORDER BY code, date
            """
            
            params = {
                'level': level,
                'start_date': start_date,
                'end_date': end_date
            }
            
            # 执行查询
            result_df = self.query_Manager_Data_Access_Manager(query, params)
            
            # 按股票代码分组
            result_dict = {}
            if not result_df.empty:
                for code in codes:
                    code_data = result_df[result_df['code'] == code].copy()
                    result_dict[code] = code_data
            else:
                # 如果没有数据，为每个代码创建空DataFrame
                for code in codes:
                    result_dict[code] = pd.DataFrame()
            
            # 缓存结果
            if self.cache_service:
                self.cache_service.set(cache_key, result_dict, ttl=300)  # 5分钟缓存
            
            return result_dict
            
        except Exception as e:
            logger.error(f"批量获取股票数据失败: {e}")
            raise DataAccessError(f"批量获取股票数据失败: {e}")
    
    def execute_query_Manager(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """执行自定义SQL查询
        
        Args:
            query: SQL查询语句，支持参数化
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
            
        Raises:
            DataAccessError: 查询执行失败
        """
        return self.query_Manager_Data_Access_Manager(query, params)
    
    def get_latest_data_Manager(self, code: str, level: str = '日线', limit: int = 100) -> pd.DataFrame:
        """获取最新的股票数据
        
        Args:
            code: 股票代码
            level: K线级别
            limit: 返回记录数
            
        Returns:
            pd.DataFrame: 最新的股票数据
        """
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key_Data_Access_Manager('latest_data', {
                'code': code,
                'level': level,
                'limit': limit
            })
            
            # 尝试从缓存获取
            if self.cache_service:
                cached_result = self.cache_service.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"从缓存获取最新数据: {cache_key}")
                    return cached_result
            
            # 构建查询
            query = """
            SELECT code, date, open, high, low, close, volume, amount,
                   price_change, price_change_pct, name, industry
            FROM stock_info WHERE 1=1
            WHERE code = %(code)s AND level = %(level)s
            ORDER BY date DESC
            LIMIT %(limit)s
            """
            
            params = {
                'code': code,
                'level': level,
                'limit': limit
            }
            
            # 执行查询
            result_df = self.query_Manager_Data_Access_Manager(query, params)
            
            # 缓存结果
            if self.cache_service:
                self.cache_service.set(cache_key, result_df, ttl=60)  # 1分钟缓存
            
            return result_df
            
        except Exception as e:
            logger.error(f"获取最新数据失败: {e}")
            raise DataAccessError(f"获取最新数据失败: {e}")
    
    def check_data_exists_data_access_manager(self, code: str, date: str, level: str = '日线') -> bool:
        """检查指定股票在指定日期的数据是否存在
        
        Args:
            code: 股票代码
            date: 日期
            level: K线级别
            
        Returns:
            bool: 数据是否存在
        """
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key_Data_Access_Manager('data_exists', {
                'code': code,
                'date': date,
                'level': level
            })
            
            # 尝试从缓存获取
            if self.cache_service:
                cached_result = self.cache_service.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # 构建查询
            query = """
            SELECT COUNT(*) as count
            FROM stock_info WHERE 1=1
            WHERE code = %(code)s 
              AND date = %(date)s 
              AND level = %(level)s
            """
            
            params = {
                'code': code,
                'date': date,
                'level': level
            }
            
            # 执行查询
            result_df = self.query_Manager_Data_Access_Manager(query, params)
            
            # 检查是否存在数据
            exists = not result_df.empty and result_df.iloc[0]['count'] > 0
            
            # 缓存结果
            if self.cache_service:
                self.cache_service.set(cache_key, exists, ttl=3600)  # 1小时缓存
            
            return exists
            
        except Exception as e:
            logger.error(f"检查数据存在性失败: {e}")
            return False
    
    def get_industry_list_Manager_Data_Access_Manager(self) -> List[str]:
        """获取所有行业列表
        
        Returns:
            List[str]: 行业名称列表
        """
        try:
            # 调用DataFrame方法获取数据
            industry_df = self.get_industry_dataframe()
            
            # 转换为列表
            if not industry_df.empty and 'industry' in industry_df.columns:
                return industry_df['industry'].tolist()
            else:
                return []
                
        except Exception as e:
            logger.error(f"获取行业列表失败: {e}")
            return []
    
    # 实现DataAccessInterface的所有抽象方法
    def get_stock_data_data_access_interface(self, code: str, start_date: str, end_date: str, 
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """实现抽象接口方法"""
        return self.get_stock_data_data_access_manager(code, start_date, end_date, columns)
    
    def get_stocks_data_batch_data_access_interface(self, codes: List[str], start_date: str, end_date: str,
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """实现抽象接口方法"""
        return self.get_stocks_data_batch_data_access_manager(codes, start_date, end_date, columns)
    
    def get_indicator_data_data_access_interface(self, code: str, indicator: str, start_date: str, end_date: str,
                          params: Optional[Dict] = None) -> pd.DataFrame:
        """实现抽象接口方法"""
        return self.get_indicator_data_data_access_manager(code, indicator, start_date, end_date, params)
    
    def get_stock_list_data_access_interface(self, industry: Optional[str] = None, 
                      market: Optional[str] = None) -> List[str]:
        """实现抽象接口方法"""
        return self.get_stock_list_data_access_manager(industry, market)
    
    def get_industry_list_data_access_interface(self) -> List[str]:
        """实现抽象接口方法"""
        return self.get_industry_list_data_access_manager()
    
    def execute_query_data_access_interface(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """实现抽象接口方法"""
        return self.execute_query_data_access_manager(query, params)
    
    def check_data_exists_data_access_interface(self, table: str, conditions: Dict) -> bool:
        """实现抽象接口方法"""
        return self.check_data_exists_data_access_manager(table, conditions)
    
    def get_latest_data_data_access_interface(self, table: str, code: str, columns: Optional[List[str]] = None) -> Optional[Dict]:
        """实现抽象接口方法"""
        return self.get_latest_data_data_access_manager(table, code, columns)
    
    @exception_handler(reraise=False, default_return=None)
    def get_previous_trade_date(self, date: str, days_back: int) -> str:
        """
        获取指定日期往前N个交易日的日期
        
        Args:
            date: 基准日期 (YYYY-MM-DD格式)
            days_back: 往前推的交易日天数
            
        Returns:
            str: 往前N个交易日的日期
        """
        try:
            # 构建查询SQL，获取往前N个交易日的日期
            query = f"""
            SELECT DISTINCT date 
            FROM stock_info 
            WHERE date <= '{date}'
            AND level = '日线'
            ORDER BY date DESC
            LIMIT 1 OFFSET {days_back - 1}
            """
            
            # 使用统一数据管理器执行查询
            with get_container().get('connection_manager').get_connection() as connection:
                cursor = connection.execute(query)
                results = list(cursor) if hasattr(cursor, '__iter__') else []
                
                if results:
                    return results[0][0]  # 返回日期字符串
                else:
                    # 如果找不到，返回一个合理的默认值（往前推120天）
                    from datetime import datetime, timedelta
                    base_date = datetime.strptime(date, '%Y-%m-%d')
                    previous_date = base_date - timedelta(days=days_back + 30)  # 加一些缓冲
                    return previous_date.strftime('%Y-%m-%d')
                    
        except Exception as e:
            logger.error(f"获取前置交易日期失败: {e}")
            # 返回一个合理的默认值
            from datetime import datetime, timedelta
            try:
                base_date = datetime.strptime(date, '%Y-%m-%d')
                previous_date = base_date - timedelta(days=days_back + 30)
                return previous_date.strftime('%Y-%m-%d')
            except:
                return '2025-01-01'  # 最后的保底日期

 