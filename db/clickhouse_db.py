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
    'password': '',
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
            result = self.client.execute(query, params or {})
            if not result:
                return pd.DataFrame()
            
            # 获取列名
            description = self.client.execute(f"SELECT * FROM ({query}) LIMIT 0", params or {})
            column_names = [item[0] for item in self.client.description]
            
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
            period: K线周期，可以是字符串或者枚举值
            
        Returns:
            pd.DataFrame: K线数据
            
        Raises:
            ValueError: 参数无效时抛出
            Exception: 查询失败时抛出
        """
        # 转换日期格式
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        # 转换周期格式
        period_str = period
        if HAS_KLINE_PERIOD and isinstance(period, KlinePeriod):
            period_str = period.value
        
        # 根据周期确定表名
        table_name = f"kline_{period_str}"
        
        # 构建查询
        query = f"""
        SELECT 
            date, code, open, high, low, close, volume, amount
        FROM 
            {table_name}
        WHERE 
            code = %(code)s AND
            date >= %(start_date)s AND
            date <= %(end_date)s
        ORDER BY 
            date
        """
        
        params = {
            'code': stock_code,
            'start_date': start_date,
            'end_date': end_date
        }
        
        return self.query(query, params)
    
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        
        Returns:
            pd.DataFrame: 股票列表数据
            
        Raises:
            Exception: 查询失败时抛出
        """
        query = """
        SELECT 
            code, name, industry, area, list_date
        FROM 
            stock_basic
        WHERE 
            is_valid = 1
        """
        
        return self.query(query)
    
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
            stock_basic
        WHERE 
            is_valid = 1 AND
            industry = %(industry)s
        """
        
        params = {'industry': industry}
        return self.query(query, params)
    
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

def get_clickhouse_db(config: Optional[Dict[str, Any]] = None) -> ClickHouseDB:
    """
    获取ClickHouseDB实例
    
    Args:
        config: 数据库连接配置，如果为None则使用默认配置
        
    Returns:
        ClickHouseDB: 数据库操作对象
    """
    return ClickHouseDB(config) 