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
from models.stock_info import StockInfo  # 导入StockInfo类
import json

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
        self._database_created = False # 用于确保数据库只创建一次

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

    def query_dataframe(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        执行查询并返回Pandas DataFrame
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
        """
        try:
            return self.client.query_dataframe(query, params)
        except Exception as e:
            logger.error(f"查询DataFrame失败: {query}, 错误: {e}")
            # 返回一个空的DataFrame以保持类型一致性
            return pd.DataFrame()


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
        self.db_connection = self.manager.get_connection(self.config)

    def __enter__(self) -> 'ClickHouseDB':
        """
        上下文管理器入口
        
        Returns:
            ClickHouseDB: 数据库操作对象自身
        """
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
        执行查询并返回Pandas DataFrame
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
        """
        return self.db_connection.query_dataframe(query, params)

    def get_stock_info(self,
                       stock_code: Union[str, List[str]] = None,
                       level: Union[str, Period] = None,
                       start_date: Union[str, datetime.datetime, None] = None,
                       end_date: Union[str, datetime.datetime, None] = None,
                       filters: Optional[Dict[str, Any]] = None,
                       limit: Optional[int] = None,
                       order_by: str = "date DESC",
                       group_by: Optional[str] = None) -> StockInfo:
        """
        统一的股票数据查询方法，替代 get_kline_data, get_stock_list 和原有的 get_stock_info 方法
        
        Args:
            stock_code: 股票代码或股票代码列表，如果为None则查询所有股票
            level: K线周期，可以是 Period 枚举值或字符串 ('day', 'week', 'month', '15min', '30min', '60min')
            start_date: 开始日期，格式可以是 'YYYY-MM-DD', 'YYYYMMDD' 或 datetime 对象
            end_date: 结束日期，格式可以是 'YYYY-MM-DD', 'YYYYMMDD' 或 datetime 对象
            filters: 过滤条件字典，可包含 price, industry, market 等字段
            limit: 限制返回的记录数量
            order_by: 排序规则
            group_by: 分组字段
            
        Returns:
            StockInfo 对象
            
        Raises:
            ValueError: 参数无效时抛出
            Exception: 查询失败时抛出
        """
        try:
            # 标准化周期
            db_level = None
            if level is not None:
                if isinstance(level, str):
                    # 字符串转换为标准周期格式
                    level_map = {
                        'day': '日线',
                        'daily': '日线',
                        'week': '周线',
                        'weekly': '周线',
                        'month': '月线',
                        'monthly': '月线',
                        '60min': '60分钟',
                        '30min': '30分钟',
                        '15min': '15分钟',
                        '5min': '5分钟'
                    }
                    db_level = level_map.get(level.lower(), level)
                elif isinstance(level, Period):
                    # Period枚举转换为字符串
                    period_map = {
                        Period.DAILY: '日线',
                        Period.WEEKLY: '周线',
                        Period.MONTHLY: '月线',
                        Period.MIN_60: '60分钟',
                        Period.MIN_30: '30分钟',
                        Period.MIN_15: '15分钟',
                        Period.MIN_5: '5分钟'
                    }
                    db_level = period_map.get(level, '日线')

            # 处理日期格式
            formatted_start_date = self._format_date_param(start_date) if start_date else None
            formatted_end_date = self._format_date_param(end_date) if end_date else None

            # 使用固定的完整字段列表
            # 字段直接从 stockInfo 对象字段保持一致，从stockInfo对象中获取
            fields = StockInfo.get_fields()
            # 构建SQL查询
            field_str = ", ".join(fields)

            # 构建WHERE子句
            conditions = []
            params = {}

            # 添加股票代码条件
            if stock_code is not None:
                if isinstance(stock_code, list):
                    if len(stock_code) == 1:
                        conditions.append("code = %(stock_code)s")
                        params['stock_code'] = stock_code[0]
                    elif len(stock_code) > 1:
                        placeholders = ", ".join([f"%(stock_code_{i})s" for i in range(len(stock_code))])
                        conditions.append(f"code IN ({placeholders})")
                        for i, code in enumerate(stock_code):
                            params[f'stock_code_{i}'] = code
                else:
                    conditions.append("code = %(stock_code)s")
                    params['stock_code'] = stock_code

            # 添加级别条件
            if db_level:
                conditions.append("level = %(level)s")
                params['level'] = db_level

            # 添加日期条件
            if formatted_start_date:
                conditions.append("date >= %(start_date)s")
                params['start_date'] = formatted_start_date

            if formatted_end_date:
                conditions.append("date <= %(end_date)s")
                params['end_date'] = formatted_end_date

            # 添加过滤条件
            if filters:
                # 价格过滤
                if 'price' in filters and isinstance(filters['price'], dict):
                    price = filters['price']
                    if 'min' in price and price['min'] > 0:
                        conditions.append("close >= %(price_min)s")
                        params['price_min'] = price['min']
                    if 'max' in price and price['max'] > 0:
                        conditions.append("close <= %(price_max)s")
                        params['price_max'] = price['max']

                # 行业过滤
                if 'industry' in filters and filters['industry']:
                    industries = filters['industry']
                    if isinstance(industries, list) and industries:
                        industry_placeholders = ", ".join([f"%(industry_{i})s" for i in range(len(industries))])
                        conditions.append(f"industry IN ({industry_placeholders})")
                        for i, industry in enumerate(industries):
                            params[f'industry_{i}'] = industry
                    elif isinstance(industries, str):
                        conditions.append("industry = %(industry)s")
                        params['industry'] = industries

                # 市场过滤
                if 'market' in filters and filters['market']:
                    markets = filters['market']
                    if isinstance(markets, list) and markets:
                        market_placeholders = ", ".join([f"%(market_{i})s" for i in range(len(markets))])
                        conditions.append(f"market IN ({market_placeholders})")
                        for i, market in enumerate(markets):
                            params[f'market_{i}'] = market
                    elif isinstance(markets, str):
                        conditions.append("market = %(market)s")
                        params['market'] = markets

            # 调整GROUP BY子句，确保所有非聚合字段都包含在内
            group_by_fields = set()
            if group_by:
                # 分割并添加已有的分组字段
                group_by_fields = set(field.strip() for field in group_by.split(','))

                # 检查fields中是否有非聚合字段需要添加到GROUP BY中
                for field in fields:
                    # 如果字段不是聚合函数(不含min, max, sum, avg, count等)
                    if not any(agg_func in field.lower() for agg_func in ['min(', 'max(', 'sum(', 'avg(', 'count(']):
                        # 提取字段名（处理类似"code as stock_code"的情况）
                        if ' as ' in field.lower():
                            field_name = field.split(' as ')[0].strip()
                        else:
                            field_name = field.strip()
                        # 排除常量字段（如 '0 as market_cap'）
                        if not field_name.startswith("'") and not field_name.startswith(
                                '"') and not field_name.isdigit():
                            group_by_fields.add(field_name)

            # 处理多股票查询的情况 - 确保查询中始终保留原始的code字段
            if isinstance(stock_code, list) and len(
                    stock_code) > 1 and not formatted_start_date and not formatted_end_date:
                # 添加ORDER BY子句，确保每个股票按日期倒序排序
                if not order_by:
                    order_by = "code, date DESC"

                # 添加限制每个股票只返回最新的一条记录
                need_latest_by_code = True
            else:
                need_latest_by_code = False

            # 构建完整查询
            query = f"SELECT {field_str} FROM stock_info"

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            # 添加GROUP BY子句
            if group_by_fields:
                query += f" GROUP BY {', '.join(group_by_fields)}"

            # 如果在GROUP BY查询中，ORDER BY子句只能引用分组字段或聚合字段
            if order_by:
                # 分析ORDER BY中的字段
                order_parts = []
                for part in order_by.split(','):
                    part = part.strip()
                    field_name = part.split()[0].strip()  # 提取字段名（忽略ASC/DESC）

                    # 检查是否是聚合字段或分组字段
                    is_aggregate = any(
                        agg_func in field_name.lower() for agg_func in ['min(', 'max(', 'sum(', 'avg(', 'count('])

                    if is_aggregate or field_name in group_by_fields:
                        order_parts.append(part)
                    else:
                        # 如果不是聚合字段或分组字段，默认按第一个分组字段排序
                        if group_by_fields:
                            default_field = next(iter(group_by_fields))
                            order_parts.append(f"{default_field} DESC")
                            logger.warning(
                                f"ORDER BY引用了非分组字段 '{field_name}'，已替换为分组字段 '{default_field}'")
                        break  # 一旦发现不兼容的排序字段，停止处理其他排序字段

                if order_parts:
                    order_by = ', '.join(order_parts)
                else:
                    order_by = None

            # 添加ORDER BY子句
            if order_by:
                query += f" ORDER BY {order_by}"

            # 添加LIMIT子句
            if limit:
                query += f" LIMIT {limit}"
            # 对于多股票查询，添加额外的查询包装以获取每个股票的最新记录
            elif need_latest_by_code:
                # 使用简单的LIMIT BY子句
                query += " LIMIT 1 BY code"
                logger.debug(f"使用LIMIT BY子句为每个股票获取最新记录: {query}")

            # 执行查询
            logger.debug(f"执行SQL查询: {query}, 参数: {params}")
            result = self.query(query, params)

            # 处理结果
            if result.empty:
                # 返回空StockInfo对象
                empty_stock = StockInfo()
                if isinstance(stock_code, str):
                    empty_stock.code = stock_code
                empty_stock.level = level if isinstance(level, str) else (level.value if level else None)
                return empty_stock

            # 标准化列名（处理col_0, col_1等通用列名）
            if 'col_0' in result.columns:
                column_mapping = {}
                for i, field in enumerate(fields):
                    if f'col_{i}' in result.columns:
                        # 从field中提取列名（处理类似"code as stock_code"的情况）
                        if ' as ' in field:
                            col_name = field.split(' as ')[1].strip()
                        else:
                            col_name = field.strip()
                        column_mapping[f'col_{i}'] = col_name

            if column_mapping:
                result = result.rename(columns=column_mapping)

            # 始终返回StockInfo对象
            return StockInfo(result)

        except Exception as e:
            logger.error(f"查询股票数据失败: {e}")
            # 返回空StockInfo对象
            empty_stock = StockInfo()
            if isinstance(stock_code, str):
                empty_stock.code = stock_code
            empty_stock.level = level if isinstance(level, str) else (level.value if level else None)
            return empty_stock

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

    def get_stock_min_date(self, stock_code: str, level: Optional[Union[str, Period]] = None) -> Optional[str]:
        """
        获取股票在特定周期下的最早日期
        
        Args:
            stock_code: 股票代码
            level: K线周期，可选
            
        Returns:
            str: 最早日期，格式YYYY-MM-DD，如果未找到返回None
        """
        try:
            # 标准化周期
            db_level = None
            if level is not None:
                if isinstance(level, str):
                    # 字符串转换为标准周期格式
                    level_map = {
                        'day': '日线',
                        'daily': '日线',
                        'week': '周线',
                        'weekly': '周线',
                        'month': '月线',
                        'monthly': '月线',
                        '60min': '60分钟线',
                        '30min': '30分钟线',
                        '15min': '15分钟线',
                        '5min': '5分钟线'
                    }
                    db_level = level_map.get(level.lower(), level)
                elif isinstance(level, Period):
                    # Period枚举转换为字符串
                    period_map = {
                        Period.DAILY: '日线',
                        Period.WEEKLY: '周线',
                        Period.MONTHLY: '月线',
                        Period.MIN_60: '60分钟线',
                        Period.MIN_30: '30分钟线',
                        Period.MIN_15: '15分钟线',
                        Period.MIN_5: '5分钟线'
                    }
                    db_level = period_map.get(level, '日线')

            # 构建查询条件
            conditions = ["code = %(stock_code)s"]
            params = {'stock_code': stock_code}

            if db_level:
                conditions.append("level = %(level)s")
                params['level'] = db_level

            # 构建查询
            query = f"""
            SELECT MIN(date) as min_date
            FROM stock_info
            WHERE {" AND ".join(conditions)}
            """

            # 执行查询
            result = self.query(query, params)

            if not result.empty and 'min_date' in result.columns and not pd.isna(result.iloc[0]['min_date']):
                min_date = result.iloc[0]['min_date']
                return min_date.strftime('%Y-%m-%d') if isinstance(min_date, datetime.datetime) else str(min_date)

            return None
        except Exception as e:
            logger.error(f"获取股票最早日期时出错: {e}")
            return None

    def get_stock_max_date(self, stock_code: Optional[str] = None, level: Optional[Union[str, Period]] = None) -> \
    Optional[str]:
        """
        获取股票在特定周期下的最新日期
        
        Args:
            stock_code: 股票代码，如果为None则查询所有股票
            level: K线周期，可选
            
        Returns:
            str: 最新日期，格式YYYY-MM-DD，如果未找到返回None
        """
        try:
            # 标准化周期
            db_level = None
            if level is not None:
                if isinstance(level, str):
                    level_map = {
                        'day': '日线',
                        'daily': '日线',
                        'week': '周线',
                        'weekly': '周线',
                        'month': '月线',
                        'monthly': '月线',
                        '60min': '60分钟线',
                        '30min': '30分钟线',
                        '15min': '15分钟线',
                        '5min': '5分钟线'
                    }
                    db_level = level_map.get(level.lower(), level)
                elif isinstance(level, Period):
                    period_map = {
                        Period.DAILY: '日线',
                        Period.WEEKLY: '周线',
                        Period.MONTHLY: '月线',
                        Period.MIN_60: '60分钟线',
                        Period.MIN_30: '30分钟线',
                        Period.MIN_15: '15分钟线',
                        Period.MIN_5: '5分钟线'
                    }
                    db_level = period_map.get(level, '日线')

            # 构建查询条件
            conditions = []
            params = {}

            if stock_code:
                conditions.append("code = %(stock_code)s")
                params['stock_code'] = stock_code

            if db_level:
                conditions.append("level = %(level)s")
                params['level'] = db_level

            # 构建查询
            query = "SELECT MAX(date) as max_date FROM stock_info"
            if conditions:
                query += f" WHERE {' AND '.join(conditions)}"

            # 执行查询
            result = self.query(query, params)

            if not result.empty and 'max_date' in result.columns and not pd.isna(result.iloc[0]['max_date']):
                max_date = result.iloc[0]['max_date']
                return max_date.strftime('%Y-%m-%d') if isinstance(max_date, datetime.datetime) else str(max_date)

            return None
        except Exception as e:
            logger.error(f"获取股票最新日期时出错: {e}")
            return None

    def get_industry_list(self) -> pd.DataFrame:
        """
        获取行业列表
        
        Returns:
            pd.DataFrame: 行业列表，包含行业名称和代码
        """
        try:
            # 构建查询
            query = """
            SELECT DISTINCT industry as name, '' as code
            FROM stock_info
            WHERE industry != ''
            ORDER BY industry
            """

            # 执行查询
            result = self.query(query)

            # 处理列名
            if 'col_0' in result.columns:
                result = result.rename(columns={'col_0': 'name', 'col_1': 'code'})

            return result
        except Exception as e:
            logger.error(f"获取行业列表时出错: {e}")
            return pd.DataFrame(columns=['name', 'code'])

    def save_selection_result(self, result: pd.DataFrame, strategy_id: str,
                              selection_date: Optional[str] = None) -> bool:
        """
        保存选股结果
        
        Args:
            result: 选股结果DataFrame
            strategy_id: 策略ID
            selection_date: 选股日期，默认为当前日期
            
        Returns:
            bool: 保存成功返回True，否则返回False
        """
        if result is None or result.empty:
            logger.warning("选股结果为空，不保存")
            return True

        # 处理日期参数
        if selection_date is None:
            selection_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # 添加策略ID和日期
        result_copy = result.copy()
        result_copy['strategy_id'] = strategy_id
        result_copy['selection_date'] = selection_date

        # 将复杂类型转为JSON字符串
        for col in result_copy.columns:
            if isinstance(result_copy[col].iloc[0], (dict, list)) or result_copy[col].dtype == 'object':
                result_copy[col] = result_copy[col].apply(
                    lambda x: json.dumps(x) if x is not None and not isinstance(x, str) else x
                )

        try:
            # 构建插入字段和值
            fields = ", ".join(result_copy.columns)
            values = ", ".join([f"%(f{i})s" for i in range(len(result_copy.columns))])

            # 构建SQL语句
            query = f"""
            INSERT INTO stock_selection_result ({fields})
            VALUES ({values})
            """

            # 逐行插入数据
            for _, row in result_copy.iterrows():
                params = {f"f{i}": val for i, val in enumerate(row)}
                self.execute(query, params)

            return True
        except Exception as e:
            logger.error(f"保存选股结果时出错: {e}")
            return False

    def get_selection_history(self, strategy_id: Optional[str] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              limit: int = 100) -> pd.DataFrame:
        """
        获取选股历史记录
        
        Args:
            strategy_id: 策略ID，None表示所有策略
            start_date: 开始日期，None表示不限制
            end_date: 结束日期，None表示当前日期
            limit: 返回记录数限制
            
        Returns:
            pd.DataFrame: 选股历史记录
        """
        try:
            # 构建条件
            conditions = []
            params = {}

            if strategy_id:
                conditions.append("strategy_id = %(strategy_id)s")
                params['strategy_id'] = strategy_id

            if start_date:
                conditions.append("selection_date >= %(start_date)s")
                params['start_date'] = start_date

            if end_date:
                conditions.append("selection_date <= %(end_date)s")
                params['end_date'] = end_date
            elif not end_date:
                end_date = datetime.datetime.now().strftime("%Y-%m-%d")
                conditions.append("selection_date <= %(end_date)s")
                params['end_date'] = end_date

            # 构建查询
            query = """
            SELECT strategy_id, selection_date, COUNT(*) as stock_count
            FROM stock_selection_result
            """

            if conditions:
                query += f" WHERE {' AND '.join(conditions)}"

            query += """
            GROUP BY strategy_id, selection_date
            ORDER BY selection_date DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            # 执行查询
            result = self.query(query, params)

            return result
        except Exception as e:
            logger.error(f"获取选股历史记录时出错: {e}")
            return pd.DataFrame(columns=['strategy_id', 'selection_date', 'stock_count'])

    def get_selection_result(self, strategy_id: str, selection_date: str) -> pd.DataFrame:
        """
        获取指定日期的选股结果
        
        Args:
            strategy_id: 策略ID
            selection_date: 选股日期
            
        Returns:
            pd.DataFrame: 选股结果
        """
        try:
            # 构建查询
            query = """
            SELECT *
            FROM stock_selection_result
            WHERE strategy_id = %(strategy_id)s
              AND selection_date = %(selection_date)s
            ORDER BY signal_strength DESC
            """

            params = {
                'strategy_id': strategy_id,
                'selection_date': selection_date
            }

            # 执行查询
            result = self.query(query, params)

            # 处理JSON字段
            if not result.empty:
                for col in result.columns:
                    if result[col].dtype == 'object' and col not in ['stock_code', 'stock_name', 'strategy_id',
                                                                     'selection_date']:
                        try:
                            result[col] = result[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                        except:
                            pass

            return result
        except Exception as e:
            logger.error(f"获取选股结果时出错: {e}")
            return pd.DataFrame()

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

    def _format_date_param(self, date_param: Union[str, datetime.datetime, None]) -> Optional[str]:
        """
        格式化日期参数为数据库可接受的格式
        
        Args:
            date_param: 日期参数，可以是字符串、datetime对象或None
            
        Returns:
            str: 格式化后的日期字符串 (YYYY-MM-DD)，如果输入为None则返回None
        """
        if date_param is None:
            return None

        if isinstance(date_param, datetime.datetime) or isinstance(date_param, datetime.date):
            return date_param.strftime('%Y-%m-%d')

        if isinstance(date_param, str):
            # 尝试处理不同格式的日期字符串
            if len(date_param) == 8 and date_param.isdigit():
                # 20220101 格式
                return f"{date_param[:4]}-{date_param[4:6]}-{date_param[6:8]}"
            elif len(date_param) == 10 and date_param[4] == '-' and date_param[7] == '-':
                # 2022-01-01 格式，已正确
                return date_param
            else:
                # 尝试解析其他格式
                try:
                    dt = pd.to_datetime(date_param)
                    return dt.strftime('%Y-%m-%d')
                except:
                    logger.warning(f"无法解析日期格式: {date_param}，使用原值")
                    return date_param

        # 其他类型，尝试转换为字符串
        return str(date_param)

    def init_database(self):
        """
        初始化数据库和表
        """
        logger.info("开始数据库初始化...")
        try:
            self._create_database()
            self._run_ddl_scripts()
            logger.info("数据库初始化成功完成。")
            return True
        except Exception as e:
            logger.error(f"数据库初始化过程中发生错误: {e}", exc_info=True)
            return False

    def _create_database(self):
        """
        如果数据库不存在，则创建它
        """
        database_name = self.db_connection.client.database
        if not database_name:
            logger.warning("未配置数据库名称，跳过数据库创建。")
            return
            
        try:
            logger.info(f"检查并创建数据库: {database_name}")
            self.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
            logger.info(f"数据库 {database_name} 已存在或已成功创建。")
        except Exception as e:
            logger.error(f"创建数据库 {database_name} 失败: {e}")
            raise

    def _execute_sql_from_file(self, file_path: str):
        """
        执行单个SQL文件中的所有语句
        
        Args:
            file_path: SQL文件的路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            
            # ClickHouse客户端可以一次执行多个语句
            if sql_script.strip():
                logger.info(f"正在执行SQL文件: {os.path.basename(file_path)}")
                self.execute(sql_script)
                logger.info(f"成功执行SQL文件: {os.path.basename(file_path)}")
        except FileNotFoundError:
            logger.error(f"SQL文件未找到: {file_path}")
            raise
        except Exception as e:
            logger.error(f"执行SQL文件 {file_path} 失败: {e}")
            raise

    def _run_ddl_scripts(self):
        """
        运行 `sql/ddl` 目录中的所有SQL脚本
        """
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ddl_dir = os.path.join(project_root, 'sql', 'ddl')
        
        if not os.path.isdir(ddl_dir):
            logger.warning(f"DDL目录不存在，跳过脚本执行: {ddl_dir}")
            return
            
        logger.info(f"开始从目录执行DDL脚本: {ddl_dir}")
        
        # 获取所有.sql文件并排序
        try:
            sql_files = sorted([f for f in os.listdir(ddl_dir) if f.endswith('.sql')])
        except FileNotFoundError:
            logger.warning(f"DDL目录不存在或无法访问: {ddl_dir}")
            return
            
        if not sql_files:
            logger.info("在DDL目录中未找到可执行的SQL脚本。")
            return
            
        for sql_file in sql_files:
            file_path = os.path.join(ddl_dir, sql_file)
            self._execute_sql_from_file(file_path)


def get_clickhouse_db(config: Optional[Dict[str, Any]] = None) -> ClickHouseDB:
    """
    获取ClickHouseDB实例
    
    Args:
        config: 数据库连接配置，如果为None则使用默认配置
        
    Returns:
        ClickHouseDB: 数据库操作对象
    """
    return ClickHouseDB(config)
