"""
连接管理器实现

实现IConnectionManager接口，提供数据库连接管理功能
"""

import time
import threading
from typing import Dict, List, Optional, Any, ContextManager
from datetime import datetime
from contextlib import contextmanager
import pandas as pd

from db.interfaces.connection_interface import IConnectionManager, IConnectionPool, IConnection, ITransactionManager, IHealthChecker
from db.enhanced_connection_pool import get_connection_pool, initialize_connection_pool
from config.config import get_config
from utils.logger import get_logger
from utils.decorators import performance_monitor
from utils.exceptions import DataAccessError

logger = get_logger(__name__)


class ConnectionWrapper(IConnection):
    """
    连接包装器
    
    封装原始连接对象，提供统一接口
    """
    
    def __init__(self, raw_connection, pool_ref=None):
        """
        初始化连接包装器
        
        Args:
            raw_connection: 原始连接对象
            pool_ref: 连接池引用
        """
        self.raw_connection = raw_connection
        self.pool_ref = pool_ref
        self.is_closed = False
        self.last_activity = time.time()
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        执行SQL语句
        
        Args:
            query: SQL语句
            params: 查询参数
        """
        try:
            self.last_activity = time.time()
            self.raw_connection.execute(query, params or {})
        except Exception as e:
            logger.error(f"执行SQL失败: {query}, 错误: {e}")
            raise DataAccessError(f"执行SQL失败: {e}")
    
    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        执行查询并返回结果
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
        """
        try:
            self.last_activity = time.time()
            return self.raw_connection.query_dataframe(query, params or {})
        except Exception as e:
            logger.error(f"查询执行失败: {query}, 错误: {e}")
            raise DataAccessError(f"查询执行失败: {e}")
    
    def query_dataframe(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        执行查询并返回DataFrame
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
        """
        return self.query(query, params)
    
    def is_alive(self) -> bool:
        """
        检查连接是否有效
        
        Returns:
            bool: 连接有效返回True
        """
        if self.is_closed:
            return False
        
        try:
            # 执行简单查询测试连接
            self.query("SELECT 1")
            return True
        except Exception:
            return False
    
    def close(self) -> None:
        """关闭连接"""
        if not self.is_closed:
            self.is_closed = True
            if hasattr(self.raw_connection, 'close'):
                self.raw_connection.close()


class ConnectionManager(IConnectionManager):
    """
    连接管理器实现
    
    管理数据库连接的生命周期和连接池
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化连接管理器
        
        Args:
            config: 连接配置
        """
        self.config = config or self._get_default_config()
        self.connection_pool = None
        self.connections: Dict[str, ConnectionWrapper] = {}
        self.connection_counter = 0
        self.lock = threading.RLock()
        
        # 初始化连接池
        self._initialize_pool()
        
        # 健康检查器
        self.health_checker = HealthChecker(self)
        
        logger.info("连接管理器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        try:
            config = get_config()
            return {
                'host': config.get('clickhouse.host', 'localhost'),
                'port': config.get('clickhouse.port', 9000),
                'database': config.get('clickhouse.database', 'stock'),
                'user': config.get('clickhouse.user', 'default'),
                'password': config.get('clickhouse.password', ''),
                'max_connections': config.get('clickhouse.max_connections', 20),
                'min_connections': config.get('clickhouse.min_connections', 5)
            }
        except Exception as e:
            logger.warning(f"获取配置失败，使用默认配置: {e}")
            return {
                'host': 'localhost',
                'port': 9000,
                'database': 'stock',
                'user': 'default',
                'password': '',
                'max_connections': 20,
                'min_connections': 5
            }
    
    def _initialize_pool(self) -> None:
        """初始化连接池"""
        try:
            self.connection_pool = initialize_connection_pool(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 9000),
                database=self.config.get('database', 'stock'),
                user=self.config.get('user', 'default'),
                password=self.config.get('password', ''),
                max_connections=self.config.get('max_connections', 20),
                min_connections=self.config.get('min_connections', 5)
            )
            logger.info("连接池初始化成功")
        except Exception as e:
            logger.error(f"连接池初始化失败: {e}")
            raise DataAccessError(f"连接池初始化失败: {e}")
    
    @contextmanager
    def get_connection(self, config: Optional[Dict[str, Any]] = None) -> ContextManager[IConnection]:
        """
        获取数据库连接
        
        Args:
            config: 连接配置
            
        Returns:
            ContextManager: 连接上下文管理器
        """
        connection = None
        connection_id = None
        
        try:
            with self.lock:
                self.connection_counter += 1
                connection_id = f"conn_{self.connection_counter}_{int(time.time())}"
            
            # 从连接池获取连接
            raw_connection = self.connection_pool.get_connection()
            
            # 包装连接
            connection = ConnectionWrapper(raw_connection.__enter__(), self.connection_pool)
            
            # 记录连接
            with self.lock:
                self.connections[connection_id] = connection
            
            logger.debug(f"获取连接: {connection_id}")
            
            yield connection
            
        except Exception as e:
            logger.error(f"获取连接失败: {e}")
            raise DataAccessError(f"获取连接失败: {e}")
        finally:
            # 清理连接
            if connection_id and connection:
                self.release_connection(connection_id)
                if hasattr(raw_connection, '__exit__'):
                    raw_connection.__exit__(None, None, None)
    
    def release_connection(self, connection_id: str) -> None:
        """
        释放数据库连接
        
        Args:
            connection_id: 连接ID
        """
        with self.lock:
            if connection_id in self.connections:
                connection = self.connections.pop(connection_id)
                connection.close()
                logger.debug(f"释放连接: {connection_id}")
    
    def test_connection(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        测试数据库连接
        
        Args:
            config: 连接配置
            
        Returns:
            bool: 连接成功返回True
        """
        try:
            with self.get_connection(config) as conn:
                result = conn.query("SELECT 1 as test")
                return not result.empty
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        获取连接统计信息
        
        Returns:
            Dict[str, Any]: 连接统计信息
        """
        with self.lock:
            active_connections = len(self.connections)
            
            # 从连接池获取统计信息
            pool_stats = {}
            if self.connection_pool and hasattr(self.connection_pool, 'get_stats'):
                try:
                    pool_stats = self.connection_pool.get_stats()
                except Exception as e:
                    logger.warning(f"获取连接池统计失败: {e}")
            
            return {
                'active_connections': active_connections,
                'total_created': self.connection_counter,
                'pool_stats': pool_stats,
                'health_status': self.health_checker.is_healthy()
            }
    
    def close_all_connections(self) -> None:
        """关闭所有连接"""
        with self.lock:
            connection_ids = list(self.connections.keys())
            for connection_id in connection_ids:
                self.release_connection(connection_id)
            
            logger.info(f"关闭所有连接: {len(connection_ids)}个")


class HealthChecker(IHealthChecker):
    """
    健康检查器实现
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        初始化健康检查器
        
        Args:
            connection_manager: 连接管理器
        """
        self.connection_manager = connection_manager
        self.last_check_time = datetime.now()
        self.is_healthy_status = True
        self.lock = threading.Lock()
        
        # 启动定期健康检查
        self._start_health_check_thread()
    
    def check_health(self) -> Dict[str, Any]:
        """
        检查数据库健康状态
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        health_info = {
            'is_healthy': False,
            'check_time': datetime.now(),
            'connection_test': False,
            'response_time': None,
            'error_message': None
        }
        
        try:
            start_time = time.time()
            
            # 测试连接
            connection_test = self.connection_manager.test_connection()
            
            end_time = time.time()
            response_time = end_time - start_time
            
            health_info.update({
                'is_healthy': connection_test,
                'connection_test': connection_test,
                'response_time': response_time
            })
            
            with self.lock:
                self.last_check_time = health_info['check_time']
                self.is_healthy_status = connection_test
            
        except Exception as e:
            health_info['error_message'] = str(e)
            logger.error(f"健康检查失败: {e}")
            
            with self.lock:
                self.is_healthy_status = False
        
        return health_info
    
    def is_healthy(self) -> bool:
        """
        检查是否健康
        
        Returns:
            bool: 健康返回True
        """
        with self.lock:
            return self.is_healthy_status
    
    def get_last_check_time(self) -> datetime:
        """
        获取最后检查时间
        
        Returns:
            datetime: 最后检查时间
        """
        with self.lock:
            return self.last_check_time
    
    def _start_health_check_thread(self) -> None:
        """启动健康检查线程"""
        def health_check_worker():
            while True:
                try:
                    time.sleep(60)  # 每分钟检查一次
                    self.check_health()
                except Exception as e:
                    logger.error(f"健康检查线程错误: {e}")
        
        health_thread = threading.Thread(target=health_check_worker, daemon=True)
        health_thread.start()
        logger.info("健康检查线程已启动")


class TransactionManager(ITransactionManager):
    """
    事务管理器实现
    """
    
    def __init__(self, connection: IConnection):
        """
        初始化事务管理器
        
        Args:
            connection: 数据库连接
        """
        self.connection = connection
        self.in_transaction_flag = False
        self.lock = threading.Lock()
    
    @contextmanager
    def begin_transaction(self) -> ContextManager:
        """
        开始事务
        
        Returns:
            ContextManager: 事务上下文管理器
        """
        with self.lock:
            if self.in_transaction_flag:
                raise DataAccessError("事务已经开始")
            
            try:
                self.connection.execute("BEGIN")
                self.in_transaction_flag = True
                logger.debug("事务开始")
                
                yield self
                
                # 如果没有异常，提交事务
                self.commit()
                
            except Exception as e:
                # 发生异常，回滚事务
                self.rollback()
                logger.error(f"事务执行失败，已回滚: {e}")
                raise
            finally:
                self.in_transaction_flag = False
    
    def commit(self) -> None:
        """提交事务"""
        if self.in_transaction_flag:
            try:
                self.connection.execute("COMMIT")
                logger.debug("事务提交")
            except Exception as e:
                logger.error(f"事务提交失败: {e}")
                raise DataAccessError(f"事务提交失败: {e}")
    
    def rollback(self) -> None:
        """回滚事务"""
        if self.in_transaction_flag:
            try:
                self.connection.execute("ROLLBACK")
                logger.debug("事务回滚")
            except Exception as e:
                logger.error(f"事务回滚失败: {e}")
                # 回滚失败不抛出异常，避免掩盖原始异常
    
    def in_transaction(self) -> bool:
        """
        检查是否在事务中
        
        Returns:
            bool: 在事务中返回True
        """
        return self.in_transaction_flag 