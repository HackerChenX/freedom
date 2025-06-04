#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
from clickhouse_driver import Client
import datetime
import logging
from typing import Dict, Optional, List, Any, Union
import threading
import time
import os
import atexit
import numpy as np
from enums.period import Period  # 添加 Period 枚举的导入

# 导入KlinePeriod枚举，但使用try-except避免循环导入问题
try:
    from enums.kline_period import KlinePeriod
    HAS_KLINE_PERIOD = True
except ImportError:
    HAS_KLINE_PERIOD = False

# 配置日志
logger = logging.getLogger('clickhouse_db')

# 默认配置
DEFAULT_CONFIG = {
    'host': 'localhost',
    'port': 9000,
    'user': 'default',
    'password': '123456',
    'database': 'stock'
}

def get_default_config() -> Dict[str, Any]:
    """
    获取默认ClickHouse配置
    
    Returns:
        Dict[str, Any]: 配置字典
    """
    return DEFAULT_CONFIG.copy()

class ClickHouseDBManager:
    """
    ClickHouse数据库连接管理器
    使用单例模式确保连接复用
    """
    _instance = None
    _connections: Dict[str, Dict[str, Any]] = {}  # 连接池
    _lock = threading.RLock()  # 添加可重入锁，保护连接池
    _max_idle_time = 300  # 连接最大空闲时间（秒）
    _cleanup_interval = 60  # 清理间隔（秒）
    _cleanup_thread = None  # 清理线程
    _shutting_down = False  # 关闭标志
    
    @classmethod
    def get_instance(cls) -> 'ClickHouseDBManager':
        """
        获取单例实例
        
        Returns:
            ClickHouseDBManager: 管理器实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._start_cleanup_thread()
                    atexit.register(cls._cleanup_all_connections)
        return cls._instance
    
    @classmethod
    def _start_cleanup_thread(cls) -> None:
        """启动连接池清理线程"""
        if cls._cleanup_thread is None:
            cls._cleanup_thread = threading.Thread(
                target=cls._connection_cleanup_task, 
                daemon=True
            )
            cls._cleanup_thread.start()
    
    @classmethod
    def _connection_cleanup_task(cls) -> None:
        """定期清理空闲连接的任务"""
        while not cls._shutting_down:
            time.sleep(cls._cleanup_interval)
            try:
                cls._cleanup_idle_connections()
            except Exception as e:
                logger.error(f"清理空闲连接时出错: {e}")
    
    @classmethod
    def _cleanup_idle_connections(cls) -> None:
        """清理空闲连接"""
        with cls._lock:
            current_time = time.time()
            keys_to_remove = []
            
            for key, conn_info in cls._connections.items():
                if conn_info['in_use']:
                    continue
                    
                if current_time - conn_info['last_used'] > cls._max_idle_time:
                    try:
                        conn_info['client'].disconnect()
                        logger.debug(f"关闭空闲连接: {key}")
                    except Exception as e:
                        logger.warning(f"关闭连接时出错: {key}, 错误: {e}")
                    
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del cls._connections[key]
    
    @classmethod
    def _cleanup_all_connections(cls) -> None:
        """清理所有连接（程序退出时调用）"""
        logger.info("正在关闭所有数据库连接...")
        cls._shutting_down = True
        
        with cls._lock:
            for key, conn_info in cls._connections.items():
                try:
                    conn_info['client'].disconnect()
                    logger.debug(f"关闭连接: {key}")
                except Exception as e:
                    logger.warning(f"关闭连接时出错: {key}, 错误: {e}")
            
            cls._connections.clear()
    
    def get_connection(self, config: Optional[Dict[str, Any]] = None) -> 'ClickHouseDBConnection':
        """
        获取数据库连接
        
        Args:
            config: 连接配置，如果为None则使用默认配置
            
        Returns:
            ClickHouseDBConnection: 数据库连接对象
            
        Raises:
            Exception: 连接失败时抛出
        """
        if config is None:
            config = get_default_config()
        
        # 生成连接键
        conn_key = f"{config['host']}:{config['port']}:{config['database']}:{config['user']}"
        
        with self._lock:
            # 检查连接池中是否有可用连接
            if conn_key in self._connections:
                conn_info = self._connections[conn_key]
                
                # 如果连接正在使用，创建新连接
                if conn_info['in_use']:
                    client = Client(**config)
                    new_conn_key = f"{conn_key}_{id(client)}"
                    self._connections[new_conn_key] = {
                        'client': client,
                        'in_use': True,
                        'last_used': time.time(),
                        'config': config
                    }
                    return ClickHouseDBConnection(self, new_conn_key)
                
                # 标记连接为使用中并返回
                conn_info['in_use'] = True
                conn_info['last_used'] = time.time()
                return ClickHouseDBConnection(self, conn_key)
            
            # 创建新连接
            try:
                client = Client(**config)
                self._connections[conn_key] = {
                    'client': client,
                    'in_use': True,
                    'last_used': time.time(),
                    'config': config
                }
                return ClickHouseDBConnection(self, conn_key)
            except Exception as e:
                logger.error(f"创建ClickHouse连接失败: {e}")
                raise
    
    def release_connection(self, conn_key: str) -> None:
        """
        释放连接（标记为未使用）
        
        Args:
            conn_key: 连接键
        """
        with self._lock:
            if conn_key in self._connections:
                self._connections[conn_key]['in_use'] = False
                self._connections[conn_key]['last_used'] = time.time()
            else:
                logger.warning(f"尝试释放不存在的连接: {conn_key}")

class ClickHouseDBConnection:
    """
    ClickHouse数据库连接包装类，支持上下文管理器模式
    """
    
    def __init__(self, manager: ClickHouseDBManager, conn_key: str):
        """
        初始化连接对象
        
        Args:
            manager: 连接管理器
            conn_key: 连接键
        """
        self.manager = manager
        self.conn_key = conn_key
        self.client = manager._connections[conn_key]['client']
    
    def __enter__(self) -> 'ClickHouseDBConnection':
        """
        上下文管理器入口
        
        Returns:
            ClickHouseDBConnection: 连接对象自身
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        上下文管理器退出，自动释放连接
        """
        self.manager.release_connection(self.conn_key)
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        执行SQL语句
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Raises:
            Exception: 执行失败时抛出
        """
        try:
            self.client.execute(query, params or {})
        except Exception as e:
            logger.error(f"执行SQL失败: {query}, 错误: {e}")
            raise
    
    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        执行查询并返回结果
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
            
        Raises:
            Exception: 执行失败时抛出
        """
        try:
            # 直接执行查询获取结果
            result = self.client.execute(query, params or {})
            if not result:
                return pd.DataFrame()
            
            # 获取列名 - 执行一个只返回结构的查询
            try:
                # 尝试从原始查询中获取列名
                meta_result = self.client.execute(f"SELECT * FROM ({query}) WHERE 0=1", params or {})
                column_names = [col[0] for col in self.client.description]
            except Exception:
                # 如果失败，使用序号作为列名
                column_names = [f"col_{i}" for i in range(len(result[0]))]
            
            # 创建DataFrame
            return pd.DataFrame(result, columns=column_names)
        except Exception as e:
            logger.error(f"执行查询失败: {query}, 错误: {e}")
            raise

class ClickHouseDB:
    """
    ClickHouse数据库操作类，提供SQL执行和数据查询功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化ClickHouse数据库操作对象
        
        Args:
            config: 数据库连接配置，如果为None则使用默认配置
        """
        self.config = config or get_default_config()
        self.manager = ClickHouseDBManager.get_instance()
    
    def __enter__(self) -> 'ClickHouseDB':
        """
        上下文管理器入口
        
        Returns:
            ClickHouseDB: 数据库操作对象自身
        """
        self.connection = self.manager.get_connection(self.config)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        上下文管理器退出，自动释放连接
        """
        # 连接本身具有上下文管理器，会自动释放
        pass
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        执行SQL语句
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Raises:
            Exception: 执行失败时抛出
        """
        with self.manager.get_connection(self.config) as conn:
            conn.execute(query, params)
    
    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        执行查询并返回结果
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
            
        Raises:
            Exception: 执行失败时抛出
        """
        with self.manager.get_connection(self.config) as conn:
            return conn.query(query, params)
    
    def get_kline_data(self, 
                      stock_code: str, 
                      start_date: Union[str, datetime.datetime], 
                      end_date: Union[str, datetime.datetime], 
                      period: Union[str, int] = 'day') -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: K线周期，可以是表名或周期枚举
            
        Returns:
            pd.DataFrame: K线数据
        """
        try:
            # 处理日期
            start_date_str = self._format_date_param(start_date)
            end_date_str = self._format_date_param(end_date)
            
            # 表名直接使用传入的period参数，无需转换
            table = period
            
            # 构建查询参数
            params = {
                "code": stock_code,
                "start_date": start_date_str,
                "end_date": end_date_str
            }
            
            # 构建SQL查询
            query = f"""
            SELECT 
                date, code, open, high, low, close, volume, amount
            FROM 
                {table}
            WHERE 
                code = %(code)s AND
                date >= %(start_date)s AND
                date <= %(end_date)s
            ORDER BY 
                date
            """
            
            # 执行查询
            return self.query(query, params)
            
        except Exception as e:
            logger.error(f"获取K线数据时出错: {e}")
            return pd.DataFrame()
    
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        
        Returns:
            pd.DataFrame: 股票列表数据
            
        Raises:
            Exception: 查询失败时抛出
        """
        query = """
        SELECT DISTINCT
            code as stock_code, 
            name as stock_name
        FROM 
            stock_info
        WHERE 
            level = '日线'
        """
        
        result = self.query(query)
        
        # 确保列名标准化
        if not result.empty and 'col_0' in result.columns:
            column_mapping = {
                'col_0': 'stock_code',
                'col_1': 'stock_name'
            }
            result = result.rename(columns=column_mapping)
        
        return result
    
    def get_stock_info(self, stock_code: str, level: Union[str, Period], start_date: Union[str, datetime.datetime], 
                       end_date: Union[str, datetime.datetime]) -> pd.DataFrame:
        """
        获取股票K线数据

        Args:
            stock_code: 股票代码
            level: K线周期，必须是 Period 枚举值或可转换为 Period 的字符串
            start_date: 开始日期，支持格式：'YYYY-MM-DD' 或 'YYYYMMDD'
            end_date: 结束日期，支持格式：'YYYY-MM-DD' 或 'YYYYMMDD'

        Returns:
            pd.DataFrame: 股票K线数据，包含以下字段：
                - code: 股票代码
                - name: 股票名称
                - date: 日期
                - level: K线周期
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - volume: 成交量
                - turnover_rate: 换手率
                - price_change: 价格变动
                - price_range: 价格区间
                - industry: 行业

        Raises:
            ValueError: 参数无效时抛出
            Exception: 查询失败时抛出
        """
        try:
            # 参数验证
            if not isinstance(stock_code, str) or not stock_code.isdigit():
                raise ValueError(f"无效的股票代码格式: {stock_code}，应为数字字符串")
            
            # 转换周期为 Period 枚举
            if isinstance(level, str):
                try:
                    level = Period.from_string(level)
                except ValueError as e:
                    raise ValueError(f"无效的K线周期: {level}，支持的周期: {', '.join(Period.get_all_period_values())}") from e
            elif not isinstance(level, Period):
                raise ValueError(f"周期参数必须是 Period 枚举或可转换为 Period 的字符串，当前类型: {type(level)}")
            
            normalized_level = level.value
            logger.info(f"查询股票 {stock_code} 的 {normalized_level} 周期数据")
            
            # 转换日期格式
            def parse_date(date_str: str) -> datetime.date:
                try:
                    if len(date_str) == 8:  # 20220101格式
                        return datetime.datetime.strptime(date_str, '%Y%m%d').date()
                    else:  # 其他格式
                        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                except ValueError as e:
                    raise ValueError(f"无效的日期格式: {date_str}，支持格式: YYYY-MM-DD 或 YYYYMMDD") from e
            
            # 转换开始日期
            if isinstance(start_date, str):
                start_date = parse_date(start_date)
            elif isinstance(start_date, datetime.datetime):
                start_date = start_date.date()
            elif not isinstance(start_date, datetime.date):
                raise ValueError(f"无效的开始日期类型: {type(start_date)}")
            
            # 转换结束日期
            if isinstance(end_date, str):
                end_date = parse_date(end_date)
            elif isinstance(end_date, datetime.datetime):
                end_date = end_date.date()
            elif not isinstance(end_date, datetime.date):
                raise ValueError(f"无效的结束日期类型: {type(end_date)}")
            
            # 验证日期范围
            if start_date > end_date:
                raise ValueError(f"开始日期 {start_date} 不能晚于结束日期 {end_date}")
            
            # 预检查数据是否存在
            check_query = """
            SELECT 
                COUNT(*) as count,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM 
                stock_info
            WHERE 
                code = %(stock_code)s AND level = %(level)s
            """
            
            check_params = {
                'stock_code': stock_code,
                'level': normalized_level
            }
            
            check_result = self.query(check_query, check_params)
            
            if check_result.empty or check_result.iloc[0, 0] == 0:
                logger.warning(f"数据库中不存在股票 {stock_code} 的 {normalized_level} 周期数据")
                return pd.DataFrame(columns=['code', 'name', 'date', 'level', 'open', 'high', 'low', 'close', 
                                           'volume', 'turnover_rate', 'price_change', 'price_range', 'industry'])
            
            min_date = check_result.iloc[0, 1]
            max_date = check_result.iloc[0, 2]
            
            logger.info(f"股票 {stock_code} 的 {normalized_level} 周期数据范围: {min_date} 至 {max_date}")
            
            # 调整查询日期范围
            if start_date < min_date:
                logger.warning(f"开始日期 {start_date} 早于数据最早日期 {min_date}，将使用最早日期")
                start_date = min_date
            
            if end_date > max_date:
                logger.warning(f"结束日期 {end_date} 晚于数据最新日期 {max_date}，将使用最新日期")
                end_date = max_date
            
            # 构建查询
            query = """
            SELECT 
                code, name, date, level, open, high, low, close, volume, turnover_rate, price_change, price_range, industry
            FROM 
                stock_info
            WHERE 
                code = %(stock_code)s AND level = %(level)s
                AND date BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY 
                date
            """
            
            params = {
                'stock_code': stock_code,
                'level': normalized_level,
                'start_date': start_date,
                'end_date': end_date
            }
            
            logger.info(f"执行查询: {query} 参数: {params}")
            result = self.query(query, params)
            
            if result.empty:
                logger.warning(f"未找到股票 {stock_code} 在 {start_date} 至 {end_date} 期间的 {normalized_level} 周期数据")
                return pd.DataFrame(columns=['code', 'name', 'date', 'level', 'open', 'high', 'low', 'close', 
                                           'volume', 'turnover_rate', 'price_change', 'price_range', 'industry'])
            
            # 标准化列名
            if 'col_0' in result.columns:
                column_mapping = {
                    'col_0': 'code',
                    'col_1': 'name',
                    'col_2': 'date',
                    'col_3': 'level',
                    'col_4': 'open',
                    'col_5': 'high',
                    'col_6': 'low',
                    'col_7': 'close',
                    'col_8': 'volume',
                    'col_9': 'turnover_rate',
                    'col_10': 'price_change',
                    'col_11': 'price_range',
                    'col_12': 'industry'
                }
                result = result.rename(columns=column_mapping)
            
            logger.info(f"成功获取 {len(result)} 条股票 {stock_code} 的 {normalized_level} 周期数据")
            return result
            
        except ValueError as e:
            logger.error(f"参数验证失败: {e}")
            raise
        except Exception as e:
            logger.error(f"从数据库加载股票 {stock_code} {level} 数据失败: {e}")
            raise
    
    def get_industry_stocks(self, industry: str) -> pd.DataFrame:
        """
        获取指定行业的股票
        
        Args:
            industry: 行业名称
            
        Returns:
            pd.DataFrame: 行业股票列表
            
        Raises:
            Exception: 查询失败时抛出
        """
        query = """
        SELECT 
            code, name, industry, area, list_date
        FROM 
            stock_info
        WHERE 
            is_valid = 1 AND
            industry = %(industry)s
        """
        
        params = {'industry': industry}
        return self.query(query, params)
    
    def get_industry_info(self, symbol: str, start_date: Union[str, datetime.datetime], 
                         end_date: Union[str, datetime.datetime]) -> pd.DataFrame:
        """
        获取行业指数数据
        
        Args:
            symbol: 行业代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 行业指数数据
            
        Raises:
            Exception: 查询失败时抛出
        """
        # 转换日期格式
        if isinstance(start_date, str):
            try:
                # 尝试转换为日期格式
                if len(start_date) == 8:  # 20220101格式
                    start_date = datetime.datetime.strptime(start_date, '%Y%m%d').date()
                else:  # 其他格式
                    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"无法将开始日期 {start_date} 转换为日期格式，使用原始字符串")
        
        if isinstance(end_date, str):
            try:
                # 尝试转换为日期格式
                if len(end_date) == 8:  # 20220101格式
                    end_date = datetime.datetime.strptime(end_date, '%Y%m%d').date()
                else:  # 其他格式
                    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"无法将结束日期 {end_date} 转换为日期格式，使用原始字符串")
        
        try:
            # 构建查询 - 查询两种情况：行业代码匹配或者industry字段匹配
            query = """
            SELECT 
                date, code, name, open, high, low, close, volume, turnover_rate
            FROM 
                stock_info
            WHERE 
                (code = %(symbol)s OR industry = %(symbol)s) AND 
                (level = '行业' OR level = '日线') AND
                date BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY 
                date
            """
            
            params = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date
            }
            
            result = self.query(query, params)
            
            # 使用DataFrame.empty属性判断结果是否为空
            if result.empty:
                logger.warning(f"未找到行业 {symbol} 的数据")
                return pd.DataFrame()
            
            # 重命名列
            column_map = {
                'col_0': 'date',
                'col_1': 'code',
                'col_2': 'name',
                'col_3': 'open',
                'col_4': 'high',
                'col_5': 'low',
                'col_6': 'close',
                'col_7': 'volume',
                'col_8': 'turnover_rate'
            }
            
            result.rename(columns=column_map, inplace=True)
                
            return result
            
        except Exception as e:
            logger.error(f"从数据库加载行业 {symbol} 数据失败: {e}")
            return pd.DataFrame()
    
    def get_industry_stock(self, industry: str) -> List[str]:
        """
        获取行业股票代码列表
        
        Args:
            industry: 行业名称
            
        Returns:
            List[str]: 股票代码列表
            
        Raises:
            Exception: 查询失败时抛出
        """
        df = self.get_industry_stocks(industry)
        if df.empty:
            return []
        return df['code'].tolist()
    
    def get_stock_max_date(self) -> datetime.datetime:
        """
        获取股票数据最新日期
        
        Returns:
            datetime.datetime: 最新日期
            
        Raises:
            Exception: 查询失败时抛出
        """
        query = """
        SELECT 
            MAX(date) as max_date
        FROM 
            stock_info
        """
        
        result = self.query(query)
        if result.empty or pd.isna(result.iloc[0, 0]):
            return datetime.datetime.now()
        return result.iloc[0, 0]
    
    def get_industry_max_date(self) -> datetime.datetime:
        """
        获取行业数据最新日期
        
        Returns:
            datetime.datetime: 最新日期
            
        Raises:
            Exception: 查询失败时抛出
        """
        query = """
        SELECT 
            MAX(date) as max_date
        FROM 
            stock_info
        WHERE
            industry != ''
        """
        
        result = self.query(query)
        if result.empty or pd.isna(result.iloc[0, 0]):
            return datetime.datetime.now()
        return result.iloc[0, 0]
    
    def get_avg_price(self, code: str, start_date: Union[str, datetime.datetime]) -> float:
        """
        获取股票平均价格
        
        Args:
            code: 股票代码
            start_date: 开始日期
            
        Returns:
            float: 平均价格
            
        Raises:
            Exception: 查询失败时抛出
        """
        # 转换日期格式
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
        
        # 构建查询
        query = """
        SELECT 
            AVG(close) as avg_price
        FROM 
            stock_info
        WHERE 
            code = %(code)s AND
            date >= %(start_date)s
        """
        
        params = {
            'code': code,
            'start_date': start_date
        }
        
        result = self.query(query, params)
        if result.empty or pd.isna(result.iloc[0, 0]):
            return 0.0
        return float(result.iloc[0, 0])
    
    def save_stock_info(self, data: pd.DataFrame, level: str) -> None:
        """
        保存股票K线数据
        
        Args:
            data: 股票数据DataFrame
            level: K线周期
            
        Raises:
            Exception: 保存失败时抛出
        """
        if data.empty:
            logger.warning("股票数据为空，不保存")
            return
        
        # 映射周期到level值
        period_map = {
            'day': 'day',
            'week': 'week',
            'month': 'month',
            '60min': '60min',
            '30min': '30min',
            '15min': '15min',
            '5min': '5min',
            # 支持大写格式
            'DAILY': 'day',
            'WEEKLY': 'week',
            'MONTHLY': 'month',
            'MIN60': '60min',
            'MIN30': '30min',
            'MIN15': '15min',
            'MIN5': '5min'
        }
        
        level_value = period_map.get(level, 'day')
        
        # 确保列名小写
        data.columns = [col.lower() for col in data.columns]
        
        # 添加level字段（如果不存在）
        if 'level' not in data.columns:
            data['level'] = level_value
        
        # 构建插入字段和值
        fields = ", ".join(data.columns)
        
        # 批量插入
        with self.manager.get_connection(self.config) as conn:
            for _, row in data.iterrows():
                values = ", ".join([f"%(f{i})s" for i in range(len(row))])
                
                query = f"""
                INSERT INTO stock_info ({fields})
                VALUES ({values})
                ON DUPLICATE KEY UPDATE 
                    open = VALUES(open),
                    high = VALUES(high),
                    low = VALUES(low),
                    close = VALUES(close),
                    volume = VALUES(volume),
                    turnover_rate = VALUES(turnover_rate)
                """
                
                params = {f"f{i}": val for i, val in enumerate(row)}
                try:
                    conn.execute(query, params)
                except Exception as e:
                    logger.error(f"保存股票数据失败: {e}")
        
        logger.info(f"已保存{len(data)}条{level}周期股票数据")
    
    def save_industry_info(self, data: pd.DataFrame) -> None:
        """
        保存行业指数数据
        
        Args:
            data: 行业数据DataFrame
            
        Raises:
            Exception: 保存失败时抛出
        """
        if data.empty:
            logger.warning("行业数据为空，不保存")
            return
        
        # 确保列名小写
        data.columns = [col.lower() for col in data.columns]
        
        # 添加level字段（如果不存在）
        if 'level' not in data.columns:
            data['level'] = 'industry'
        
        # 构建插入字段和值
        fields = ", ".join(data.columns)
        
        # 批量插入
        with self.manager.get_connection(self.config) as conn:
            for _, row in data.iterrows():
                values = ", ".join([f"%(f{i})s" for i in range(len(row))])
                
                query = f"""
                INSERT INTO stock_info ({fields})
                VALUES ({values})
                ON DUPLICATE KEY UPDATE 
                    open = VALUES(open),
                    high = VALUES(high),
                    low = VALUES(low),
                    close = VALUES(close),
                    volume = VALUES(volume)
                """
                
                params = {f"f{i}": val for i, val in enumerate(row)}
                try:
                    conn.execute(query, params)
                except Exception as e:
                    logger.error(f"保存行业数据失败: {e}")
        
        logger.info(f"已保存{len(data)}条行业指数数据")
    
    def save_result(self, 
                   result_type: str, 
                   result_data: pd.DataFrame, 
                   strategy_id: Optional[str] = None) -> None:
        """
        保存结果数据
        
        Args:
            result_type: 结果类型
            result_data: 结果数据
            strategy_id: 策略ID
            
        Raises:
            Exception: 保存失败时抛出
        """
        if result_data.empty:
            logger.warning("结果数据为空，不保存")
            return
        
        # 添加结果时间和策略ID
        result_data['result_time'] = datetime.datetime.now()
        if strategy_id:
            result_data['strategy_id'] = strategy_id
        
        # 确保有结果类型
        result_data['result_type'] = result_type
        
        # 构建插入字段和值
        fields = ", ".join(result_data.columns)
        
        # 批量插入
        with self.manager.get_connection(self.config) as conn:
            for _, row in result_data.iterrows():
                values = ", ".join([f"%(f{i})s" for i in range(len(row))])
                
                query = f"""
                INSERT INTO analysis_results ({fields})
                VALUES ({values})
                """
                
                params = {f"f{i}": val for i, val in enumerate(row)}
                conn.execute(query, params)
        
        logger.info(f"已保存{len(result_data)}条{result_type}结果数据")

    def get_stock_name(self, stock_code: str) -> str:
        """
        获取股票名称
        Args:
            stock_code: 股票代码
        Returns:
            str: 股票名称，如果未找到则返回股票代码
        """
        try:
            query = """
            SELECT code, name FROM stock_info WHERE code = %(stock_code)s AND level = '日线' LIMIT 1
            """
            params = {'stock_code': stock_code}
            result = self.query(query, params)
            # 标准化列名
            if not result.empty:
                if 'col_0' in result.columns:
                    result = result.rename(columns={'col_0': 'code', 'col_1': 'name'})
                elif 'stock_code' in result.columns and 'stock_name' in result.columns:
                    result = result.rename(columns={'stock_code': 'code', 'stock_name': 'name'})
                # 取第一行的name
                return result.iloc[0]['name']
            return stock_code
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 名称失败: {e}")
            return stock_code

    def _format_date_param(self, date_param: Union[str, datetime.datetime, datetime.date]) -> str:
        """
        格式化日期参数为数据库可接受的格式
        
        Args:
            date_param: 日期参数，可以是字符串、datetime.datetime或datetime.date
            
        Returns:
            str: 格式化后的日期字符串
        """
        if date_param is None:
            return datetime.datetime.now().strftime('%Y-%m-%d 00:00:00')
            
        if isinstance(date_param, str):
            # 处理YYYYMMDD格式
            if len(date_param) == 8 and date_param.isdigit():
                return f"{date_param[:4]}-{date_param[4:6]}-{date_param[6:]} 00:00:00"
            
            # 处理YYYY-MM-DD格式
            elif len(date_param) == 10 and date_param[4] == '-' and date_param[7] == '-':
                return f"{date_param} 00:00:00"
                
            # 如果已经包含时间部分，直接返回
            elif len(date_param) > 10 and date_param[4] == '-' and date_param[7] == '-':
                return date_param
                
            # 处理无效日期
            elif date_param == '19700101' or date_param == '1970-01-01':
                return datetime.datetime.now().strftime('%Y-%m-%d 00:00:00')
                
            else:
                logger.warning(f"无法识别的日期格式: {date_param}，使用当前日期")
                return datetime.datetime.now().strftime('%Y-%m-%d 00:00:00')
                
        elif isinstance(date_param, datetime.datetime):
            return date_param.strftime('%Y-%m-%d %H:%M:%S')
            
        elif isinstance(date_param, datetime.date):
            return f"{date_param.strftime('%Y-%m-%d')} 00:00:00"
            
        else:
            logger.warning(f"无法处理的日期类型: {type(date_param)}，使用当前日期")
            return datetime.datetime.now().strftime('%Y-%m-%d 00:00:00')

def get_clickhouse_db(config: Optional[Dict[str, Any]] = None) -> ClickHouseDB:
    """
    获取ClickHouseDB实例
    
    Args:
        config: 数据库连接配置，如果为None则使用默认配置
        
    Returns:
        ClickHouseDB: 数据库操作对象
    """
    return ClickHouseDB(config) 