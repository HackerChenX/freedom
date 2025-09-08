#!/usr/bin/python
# -*- coding: UTF-8 -*-

from config import get_config
"""
增强的Click_house连接池管理器

解决并发查询问题，为每个线程提供独立的数据库连接实例
"""

import threading
import time
import queue
import logging
from typing import Dict, Optional, Any, List
from contextlib import contextmanager
import atexit
from clickhouse_driver import Client
import pandas as pd

from utils.logger import getLogger

logger = getLogger(__name__)


class ClickHouseConnectionPool:
    """
    增强的Click_house连接池
    
    特性：
    - 支持并发查询（每个线程独立连接）
    - 连接复用和自动清理
    - 连接健康检查
    - 性能监控
    """
    
    def __init__(self,
                 host: str = 'localhost',
                 port: int = 9000,
                 database: str = 'stock',
                 user: str = 'default',
                 password: str = '',  # 🔧 Ultra Think修复：使用空密码作为默认值
                 max_connections: int = 20,
                 min_connections: int = 5,
                 max_idle_time: int = 300,
                 health_check_interval: int = 60):
        """
        初始化连接池
        
        Args:
            host: Click_house主机地址
            port: Click_house端口
            database: 数据库名
            user: 用户名
            password: 密码
            max_connections: 最大连接数
            min_connections: 最小连接数
            max_idle_time: 最大空闲时间（秒）
            health_check_interval: 健康检查间隔（秒）
        """
        self.config = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        
        # 连接池队列
        self.available_connections = queue.Queue(maxsize=max_connections)
        self.all_connections = {}  # 所有连接的跟踪
        self.connection_stats = {}  # 连接统计信息
        
        # 线程安全锁
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'total_created': 0,
            'total_destroyed': 0,
            'current_active': 0,
            'current_idle': 0,
            'total_requests': 0,
            'total_errors': 0,
            'avg_response_time': 0.0
        }
        
        # 控制标志
        self.is_closed = False
        
        # 初始化最小连接数
        self._initialize_pool()
        
        # 启动健康检查线程
        self._start_health_check_thread()
        
        # 注册清理函数
        atexit.register(self.close_Pool)
        
        logger.info(f"ClickHouse连接池初始化完成，配置: {self.config}, "
                   f"连接数范围: {min_connections}-{max_connections}")
    
    def _initialize_pool(self):
        """初始化连接池，创建最小连接数"""
        for _ in range(self.min_connections):
            try:
                conn = self._create_connection()
                self.available_connections.put(conn)
            except Exception as e:
                logger.error(f"初始化连接池时创建连接失败: {e}")
    
    def _create_connection(self) -> 'PooledConnection':
        """创建新的数据库连接"""
        try:
            client = Client(**self.config)
            
            # 测试连接
            client.execute("SELECT 1")
            
            conn_id = f"conn_{int(time.time() * 1000)}_{threading.current_thread().ident}"
            
            with self.lock:
                self.stats['total_created'] += 1
                
                pooled_conn = PooledConnection(
                    client=client,
                    pool=self,
                    connection_id=conn_id
                )
                
                self.all_connections[conn_id] = {
                    'connection': pooled_conn,
                    'created_time': time.time(),
                    'last_used': time.time(),
                    'use_count': 0,
                    'is_healthy': True
                }
                
                logger.debug(f"创建新连接: {conn_id}")
                return pooled_conn
                
        except Exception as e:
            with self.lock:
                self.stats['total_errors'] += 1
            logger.error(f"创建数据库连接失败: {e}")
            raise
    
    def _start_health_check_thread(self):
        """启动健康检查线程"""
        def health_check_task():
            while not self.is_closed:
                try:
                    time.sleep(self.health_check_interval)
                    self._perform_health_check()
                    self._cleanup_idle_connections()
                except Exception as e:
                    logger.error(f"健康检查任务出错: {e}")
        
        health_thread = threading.Thread(target=health_check_task, daemon=True)
        health_thread.start()
        logger.debug("健康检查线程已启动")
    
    def _perform_health_check(self):
        """执行连接健康检查"""
        with self.lock:
            unhealthy_connections = []
            
            for conn_id, conn_info in self.all_connections.items():
                try:
                    # 简单的健康检查查询
                    conn_info['connection'].client.execute("SELECT 1")
                    conn_info['is_healthy'] = True
                except Exception as e:
                    logger.warning(f"连接 {conn_id} 健康检查失败: {e}")
                    conn_info['is_healthy'] = False
                    unhealthy_connections.append(conn_id)
            
            # 移除不健康的连接
            for conn_id in unhealthy_connections:
                self._destroy_connection(conn_id)
    
    def _cleanup_idle_connections(self):
        """清理空闲连接"""
        current_time = time.time()
        
        with self.lock:
            idle_connections = []
            
            for conn_id, conn_info in self.all_connections.items():
                if (current_time - conn_info['last_used'] > self.max_idle_time and
                    len(self.all_connections) > self.min_connections):
                    idle_connections.append(conn_id)
            
            for conn_id in idle_connections:
                self._destroy_connection(conn_id)
                logger.debug(f"清理空闲连接: {conn_id}")
    
    def _destroy_connection(self, conn_id: str):
        """销毁连接"""
        if conn_id in self.all_connections:
            try:
                conn_info = self.all_connections[conn_id]
                conn_info['connection'].client.disconnect()
            except Exception as e:
                logger.warning(f"关闭连接时出错: {e}")
            finally:
                del self.all_connections[conn_id]
                self.stats['total_destroyed'] += 1
    
    @contextmanager
    def get_connection(self):
        """
        获取连接的上下文管理器
        
        Returns:
            PooledConnection: 池化连接对象
        """
        if self.is_closed:
            raise RuntimeError("连接池已关闭")
        
        connection = None
        start_time = time.time()
        
        try:
            with self.lock:
                self.stats['total_requests'] += 1
            
            # 尝试从队列获取可用连接
            try:
                connection = self.available_connections.get_nowait()
                logger.debug(f"从池中获取连接: {connection.connection_id}")
            except queue.Empty:
                # 队列为空，创建新连接
                if len(self.all_connections) < self.max_connections:
                    connection = self._create_connection()
                    logger.debug(f"创建新连接: {connection.connection_id}")
                else:
                    # 达到最大连接数，等待可用连接
                    timeout = get_config('performance.timeout') or 30  # 🔧 Ultra Think修复：添加默认30秒超时
                    connection = self.available_connections.get(timeout)
                    logger.debug(f"等待获取连接: {connection.connection_id}")
            
            # 更新连接统计
            with self.lock:
                if connection.connection_id in self.all_connections:
                    conn_info = self.all_connections[connection.connection_id]
                    conn_info['last_used'] = time.time()
                    conn_info['use_count'] += 1
                    self.stats['current_active'] += 1
            
            yield connection
            
        except Exception as e:
            with self.lock:
                self.stats['total_errors'] += 1
            logger.error(f"获取连接时出错: {e}")
            raise
        finally:
            # 归还连接到池中
            if connection and not self.is_closed:
                try:
                    self.available_connections.put_nowait(connection)
                    with self.lock:
                        self.stats['current_active'] -= 1
                        # 更新平均响应时间
                        response_time = time.time() - start_time
                        self.stats['avg_response_time'] = (
                            (self.stats['avg_response_time'] * (self.stats['total_requests'] - 1) + response_time) /
                            self.stats['total_requests']
                        )
                    logger.debug(f"归还连接到池: {connection.connection_id}")
                except queue.Full:
                    # 队列已满，销毁连接
                    self._destroy_connection(connection.connection_id)
                    logger.debug(f"队列已满，销毁连接: {connection.connection_id}")
    
    def get_stats_Pool(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        with self.lock:
            self.stats['current_idle'] = self.available_connections.qsize()
            self.stats['total_connections'] = len(self.all_connections)
            return self.stats.copy()

    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息（标准接口）"""
        return self.get_stats_Pool()
    
    def close_Pool(self):
        """关闭连接池"""
        if self.is_closed:
            return
        
        logger.info("正在关闭ClickHouse连接池...")
        self.is_closed = True
        
        with self.lock:
            # 关闭所有连接
            for conn_id in list(self.all_connections.keys()):
                self._destroy_connection(conn_id)
            
            # 清空队列
            while not self.available_connections.empty():
                try:
                    self.available_connections.get_nowait()
                except queue.Empty:
                    break
        
        logger.info("ClickHouse连接池已关闭")


class PooledConnection:
    """池化连接包装器"""
    
    def __init__(self, client, pool, connection_id):
        """
        初始化池化连接
        
        Args:
            client: ClickHouse客户端
            pool: 连接池实例
            connection_id: 连接ID
        """
        self.client = client
        self.pool = pool
        self.connection_id = connection_id
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """执行SQL语句"""
        try:
            return self.client.execute(query, params or {})
        except Exception as e:
            logger.error(f"执行SQL失败 [{self.connection_id}]: {query}, 错误: {e}")
            raise
    
    def query_dataframe(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """执行查询并返回DataFrame (标准接口)"""
        try:
            # 使用正确的clickhouse_driver方法
            result = self.client.execute(query, params or {}, with_column_types=True)
            
            if not result:
                return pd.DataFrame()
            
            data, columns = result
            column_names = [col[0] for col in columns]
            
            return pd.DataFrame(data, columns=column_names)
        except Exception as e:
            logger.error(f"查询DataFrame失败 [{self.connection_id}]: {query}, 错误: {e}")
            return pd.DataFrame()

    def query_dataframe_Pool(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """执行查询并返回DataFrame"""
        return self.query_dataframe(query, params)
    
    def query_dataframe_enhanced_connection_pool(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """执行查询并返回DataFrame (向后兼容)"""
        return self.query_dataframe(query, params)


# 全局连接池实例
_connection_pool = None
_pool_lock = threading.Lock()


def get_connection_pool() -> ClickHouseConnectionPool:
    """获取全局连接池实例"""
    global _connection_pool
    
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                _connection_pool = ClickHouseConnectionPool()
    
    return _connection_pool


def initialize_connection_pool(**kwargs) -> ClickHouseConnectionPool:
    """
    初始化连接池
    
    Args:
        **kwargs: 连接池配置参数
        
    Returns:
        ClickHouseConnectionPool: 连接池实例
    """
    global _connection_pool
    
    with _pool_lock:
        if _connection_pool is not None:
            _connection_pool.close_Pool()
        
        _connection_pool = ClickHouseConnectionPool(**kwargs)
        logger.info("全局ClickHouse连接池已初始化")
    
    return _connection_pool


def close_connection_pool():
    """关闭全局连接池"""
    global _connection_pool
    
    with _pool_lock:
        if _connection_pool is not None:
            _connection_pool.close_Pool()
            _connection_pool = None
            logger.info("全局ClickHouse连接池已关闭")
