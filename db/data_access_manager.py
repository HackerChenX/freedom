"""
数据访问管理器

统一的数据访问接口实现，支持多种数据源，使用统一连接池适配器
"""

import time
import threading
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd

from db.interfaces.data_access_interface import DataAccessInterface
from db.enhanced_connection_pool import get_connection_pool
from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.cache import MemoryCache

logger = get_logger(__name__)


class DataAccessManager(DataAccessInterface):
    """
    数据访问管理器

    统一的数据访问接口实现，使用连接池适配器管理数据库连接
    """

    def __init__(self, use_cache: bool = True):
        """
        初始化数据访问管理器 - 任务5简化版本

        Args:
            use_cache: 是否使用缓存
        """
        self.use_cache = use_cache
        self._lock = threading.Lock()

        # 直接使用增强连接池
        try:
            self.connection_pool = get_connection_pool()
            logger.info(f"数据访问管理器初始化完成，使用增强连接池")
        except Exception as e:
            logger.error(f"连接池初始化失败: {e}")
            raise

        # 初始化缓存
        if self.use_cache:
            try:
                self.cache = MemoryCache(max_size=1000, default_ttl=300)  # 5分钟TTL
                logger.info("缓存系统初始化完成")
            except Exception as e:
                logger.warning(f"缓存初始化失败，禁用缓存: {e}")
                self.use_cache = False
                self.cache = None
        else:
            self.cache = None
    
    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=False, default_return=pd.DataFrame())
    def get_stock_data_data_access_manager(self, code: str, start_date: str, end_date: str,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取股票数据

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            columns: 需要的列名列表

        Returns:
            pd.DataFrame: 股票数据
        """
        # 生成缓存键
        cache_key = f"stock_data_{code}_{start_date}_{end_date}_{columns}"

        # 检查缓存
        if self.use_cache and self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"从缓存获取股票数据: {code}")
                return cached_data

        logger.info(f"从数据库获取股票数据: {code}, {start_date} - {end_date}")

        try:
            # 构建查询语句
            if columns:
                columns_str = ', '.join(columns)
            else:
                columns_str = 'code, name, date, open, high, low, close, volume, turnover_rate'

            query = f"""
            SELECT {columns_str}
            FROM stock_info
            WHERE code = '{code}'
            AND level = '日线'
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date ASC
            """

            # 执行查询
            with self.connection_pool.get_connection() as conn:
                result_df = conn.query_dataframe(query)

            # 缓存结果
            if self.use_cache and self.cache and not result_df.empty:
                self.cache.set(cache_key, result_df)

            logger.info(f"获取到 {len(result_df)} 条股票数据: {code}")
            return result_df

        except Exception as e:
            logger.error(f"获取股票数据失败: {code}, 错误: {e}")
            # 返回空DataFrame而不是模拟数据
            return pd.DataFrame()

    
    @performance_monitor(threshold=5.0)
    @exception_handler(reraise=False, default_return=pd.DataFrame())
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
            pd.DataFrame: 股票数据
        """
        logger.info(f"批量获取股票数据: {len(codes)}只股票")

        try:
            # 构建批量查询
            if columns:
                columns_str = ', '.join(columns)
            else:
                columns_str = 'code, name, date, open, high, low, close, volume, turnover_rate'

            codes_str = "', '".join(codes)
            query = f"""
            SELECT {columns_str}
            FROM stock_info
            WHERE code IN ('{codes_str}')
            AND level = '日线'
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY code, date ASC
            """

            # 执行查询
            with self.connection_pool.get_connection() as conn:
                result_df = conn.query_dataframe(query)

            logger.info(f"批量获取到 {len(result_df)} 条股票数据")
            return result_df

        except Exception as e:
            logger.error(f"批量获取股票数据失败: {e}")
            return pd.DataFrame()
    
    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=False, default_return=[])
    def get_stock_info_data_access_manager(self, code: str, level: str, start_date: str, end_date: str) -> List[List]:
        """
        获取股票信息（兼容现有接口）

        Args:
            code: 股票代码
            level: K线级别
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            List[List]: 股票数据行列表
        """
        logger.info(f"获取股票信息: {code}, {level}, {start_date} - {end_date}")

        try:
            query = f"""
            SELECT code, name, date, open, high, low, close, volume, turnover_rate,
                   price_change, price_range, industry, datetime, seq
            FROM stock_info
            WHERE code = '{code}'
            AND level = '{level}'
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date ASC
            """

            with self.connection_pool.get_connection() as conn:
                result = conn.execute(query)

            logger.info(f"获取到 {len(result)} 条股票信息: {code}")
            return result

        except Exception as e:
            logger.error(f"获取股票信息失败: {code}, 错误: {e}")
            return []

    def get_stock_info(self, code: str, level: str, start_date: str, end_date: str) -> List[List]:
        """
        获取股票信息（兼容买点分析器接口）

        Args:
            code: 股票代码
            level: K线级别
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)

        Returns:
            List[List]: 股票数据行列表
        """
        # 转换日期格式从YYYYMMDD到YYYY-MM-DD
        try:
            if len(start_date) == 8:
                start_date_formatted = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
            else:
                start_date_formatted = start_date

            if len(end_date) == 8:
                end_date_formatted = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
            else:
                end_date_formatted = end_date

            return self.get_stock_info_data_access_manager(code, level, start_date_formatted, end_date_formatted)

        except Exception as e:
            logger.error(f"日期格式转换失败: {start_date}, {end_date}, 错误: {e}")
            return []


    @performance_monitor(threshold=2.0)
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
            pd.DataFrame: 指标数据
        """
        logger.info(f"获取指标数据: {code}, {indicator}, {start_date} - {end_date}")

        try:
            # 首先获取基础股票数据
            stock_data = self.get_stock_data_data_access_manager(code, start_date, end_date)

            if stock_data.empty:
                logger.warning(f"无法获取股票数据，无法计算指标: {code}")
                return pd.DataFrame()

            # 这里应该调用指标计算系统
            # 暂时返回基础数据，后续集成指标计算
            logger.info(f"指标数据计算功能待集成: {indicator}")
            return stock_data

        except Exception as e:
            logger.error(f"获取指标数据失败: {code}, {indicator}, 错误: {e}")
            return pd.DataFrame()
    
    @performance_monitor(threshold=3.0)
    @exception_handler(reraise=False, default_return=[])
    def get_stock_list_data_access_manager(self, industry: Optional[str] = None,
                      market: Optional[str] = None) -> List[str]:
        """
        获取股票列表

        Args:
            industry: 行业筛选
            market: 市场筛选

        Returns:
            List[str]: 股票代码列表
        """
        logger.info(f"获取股票列表: industry={industry}, market={market}")

        try:
            # 构建查询条件
            where_conditions = ["level = '日线'"]

            if industry:
                where_conditions.append(f"industry = '{industry}'")

            if market:
                # 根据股票代码前缀判断市场
                if market.upper() == 'SH':
                    where_conditions.append("(code LIKE '60%' OR code LIKE '68%')")
                elif market.upper() == 'SZ':
                    where_conditions.append("(code LIKE '00%' OR code LIKE '30%')")

            where_clause = " AND ".join(where_conditions)

            query = f"""
            SELECT DISTINCT code
            FROM stock_info
            WHERE {where_clause}
            ORDER BY code
            """

            with self.connection_pool.get_connection() as conn:
                result = conn.execute(query)

            # 提取股票代码列表
            stock_codes = [row[0] for row in result]

            logger.info(f"获取到 {len(stock_codes)} 只股票")
            return stock_codes

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []

    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=False, default_return=[])
    def get_industry_list_data_access_manager(self) -> List[str]:
        """
        获取行业列表

        Returns:
            List[str]: 行业列表
        """
        logger.info("获取行业列表")

        try:
            query = """
            SELECT DISTINCT industry
            FROM stock_info
            WHERE industry IS NOT NULL AND industry != ''
            ORDER BY industry
            """

            with self.connection_pool.get_connection() as conn:
                result = conn.execute(query)

            # 提取行业列表
            industries = [row[0] for row in result if row[0]]

            logger.info(f"获取到 {len(industries)} 个行业")
            return industries

        except Exception as e:
            logger.error(f"获取行业列表失败: {e}")
            return []
    
    @performance_monitor(threshold=5.0)
    @exception_handler(reraise=False, default_return=pd.DataFrame())
    def execute_query_data_access_manager(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        执行查询

        Args:
            query: SQL查询语句
            params: 查询参数

        Returns:
            pd.DataFrame: 查询结果
        """
        logger.info(f"执行查询: {query[:100]}...")

        try:
            with self.connection_pool.get_connection() as conn:
                result_df = conn.query_dataframe(query, params)

            logger.info(f"查询执行成功，返回 {len(result_df)} 行数据")
            return result_df

        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            return pd.DataFrame()
    
    @performance_monitor(threshold=1.0)
    @exception_handler(reraise=False, default_return=False)
    def check_data_exists(self, table: str, conditions: Dict) -> bool:
        """
        检查数据是否存在

        Args:
            table: 表名
            conditions: 检查条件

        Returns:
            bool: 数据是否存在
        """
        logger.info(f"检查数据存在: table={table}, conditions={conditions}")

        try:
            # 构建WHERE条件
            where_conditions = []
            for key, value in conditions.items():
                if isinstance(value, str):
                    where_conditions.append(f"{key} = '{value}'")
                else:
                    where_conditions.append(f"{key} = {value}")

            where_clause = " AND ".join(where_conditions)

            query = f"""
            SELECT COUNT(*) as count
            FROM {table}
            WHERE {where_clause}
            """

            with self.connection_pool.get_connection() as conn:
                result = conn.execute(query)

            count = result[0][0] if result else 0
            exists = count > 0

            logger.info(f"数据存在检查结果: {exists} (count: {count})")
            return exists

        except Exception as e:
            logger.error(f"检查数据存在失败: {e}")
            return False
    
    @performance_monitor(threshold=1.0)
    @exception_handler(reraise=False, default_return=None)
    def get_latest_data_data_access_manager(self, table: str, code: str, columns: Optional[List[str]] = None) -> Optional[Dict]:
        """
        获取最新数据

        Args:
            table: 表名
            code: 股票代码
            columns: 需要的列名

        Returns:
            Optional[Dict]: 最新数据字典
        """
        logger.info(f"获取最新数据: table={table}, code={code}")

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

            with self.connection_pool.get_connection() as conn:
                result = conn.execute(query)

            if result:
                # 假设第一行是列名，第二行是数据（简化处理）
                row = result[0]
                if columns:
                    return dict(zip(columns, row))
                else:
                    # 使用标准列名
                    standard_columns = ['code', 'name', 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover_rate']
                    return dict(zip(standard_columns[:len(row)], row))

            return None

        except Exception as e:
            logger.error(f"获取最新数据失败: {e}")
            return None

    def get_connection_statistics(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        try:
            return self.connection_pool.get_stats()
        except Exception as e:
            logger.error(f"获取连接池统计失败: {e}")
            return {}

    def close(self):
        """关闭数据访问管理器"""
        try:
            if self.connection_pool:
                self.connection_pool.close()
            logger.info("数据访问管理器已关闭")
        except Exception as e:
            logger.error(f"关闭数据访问管理器失败: {e}")

    # 实现抽象接口方法（带_data_access_interface后缀）
    def get_stock_data_data_access_interface(self, code: str, start_date: str, end_date: str, 
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """实现抽象接口方法"""
        return self.get_stock_data_data_access_manager(code, start_date, end_date, columns)
    
    def get_stocks_data_batch_data_access_interface(self, codes: List[str], start_date: str, end_date: str,
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """实现抽象接口方法"""
        return self.get_stocks_data_batch(codes, start_date, end_date, columns)
    
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
        return self.check_data_exists(table, conditions)
    
    def get_latest_data_data_access_interface(self, table: str, code: str, columns: Optional[List[str]] = None) -> Optional[Dict]:
        """实现抽象接口方法"""
        return self.get_latest_data_data_access_manager(table, code, columns) 