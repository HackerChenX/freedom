#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
修复版本的ClickHouse数据库操作类
解决以下问题：
1. 并发连接问题
2. Stock_info对象属性问题
3. 连接池管理问题
"""

import pandas as pd
from clickhouse_driver import Client
import datetime
import logging
from typing import Dict, Optional, List, Any, Union
import threading
import time
import os
from models.stock_info import Stock_info
from config import get_config

# 配置日志
logger = logging.getLogger('clickhouse_db_final_fixed')

class ClickHouseDatabaseFixed:
    """
    最终修复版的ClickHouse数据库操作类
    """

    def __init__(self, config=None):
        """初始化数据库连接"""
        self.config = config or self._get_default_config()
        self._connection_lock = threading.RLock()
        self._connections = {}  # 线程本地连接

        # 验证配置
        self._validate_config()
        logger.info(f"ClickHouse数据库初始化完成: {self.config['host']}:{self.config['port']}")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'host': get_config('database.host', 'localhost'),
            'port': 9000,  # ClickHouse原生协议端口
            'user': get_config('database.user', 'default'),
            'password': get_config('database.password', ''),
            'database': get_config('database.name', 'stock')
        }

    def _validate_config(self):
        """验证配置"""
        required_fields = ['host', 'port', 'database', 'user']
        for field in required_fields:
            if not self.config.get(field):
                raise ValueError(f"缺少必需的配置项: {field}")

    def _get_thread_connection(self) -> Client:
        """获取线程本地连接"""
        thread_id = threading.current_thread().ident

        with self._connection_lock:
            if thread_id not in self._connections:
                try:
                    client = Client(**self.config)
                    # 测试连接
                    client.execute("SELECT 1")
                    self._connections[thread_id] = client
                    logger.debug(f"为线程 {thread_id} 创建新连接")
                except Exception as e:
                    logger.error(f"创建数据库连接失败: {e}")
                    raise

            return self._connections[thread_id]

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None):
        """执行SQL语句"""
        try:
            client = self._get_thread_connection()
            return client.execute(query, params or {})
        except Exception as e:
            logger.error(f"执行SQL失败: {query[:100]}..., 错误: {e}")
            # 清理连接以便重试
            thread_id = threading.current_thread().ident
            with self._connection_lock:
                if thread_id in self._connections:
                    try:
                        self._connections[thread_id].disconnect()
                    except:
                        pass
                    del self._connections[thread_id]
            raise

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """执行查询并返回DataFrame"""
        try:
            result = self.execute(query, params)
            if not result:
                return pd.DataFrame()

            # 从查询中提取列名
            column_names = self._extract_column_names_from_query(query)
            if not column_names and result:
                column_names = [f"col_{i}" for i in range(len(result[0]))]

            df = pd.DataFrame(result, columns=column_names or [])
            return df

        except Exception as e:
            logger.error(f"查询执行失败: {query[:100]}..., 错误: {e}")
            return pd.DataFrame()

    def _extract_column_names_from_query(self, query: str) -> List[str]:
        """从查询语句中提取列名"""
        try:
            import re
            clean_query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
            clean_query = re.sub(r'--.*', '', clean_query)
            clean_query = ' '.join(clean_query.split())

            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', clean_query, re.IGNORECASE | re.DOTALL)
            if not select_match:
                return []

            select_part = select_match.group(1).strip()
            if select_part.strip() == '*':
                return []

            fields = [field.strip() for field in select_part.split(',')]
            column_names = []

            for field in fields:
                if ' AS ' in field.upper():
                    alias = field.upper().split(' AS ')[-1].strip()
                    column_names.append(alias.strip('"').strip("'"))
                else:
                    clean_field = field.split('.')[-1].strip()
                    column_names.append(clean_field)

            return column_names
        except Exception:
            return []

    def get_stock_info(self,
                       stock_code: Union[str, List[str]] = None,
                       level: str = None,
                       start_date: str = None,
                       end_date: str = None,
                       limit: Optional[int] = None) -> Stock_info:
        """
        获取股票信息 - 修复版本
        """
        try:
            # 构建查询条件
            conditions = ["date >= '2020-01-01'"]
            params = {}

            # 股票代码条件
            if stock_code:
                if isinstance(stock_code, str):
                    conditions.append("code = %(stock_code)s")
                    params['stock_code'] = stock_code
                elif isinstance(stock_code, list):
                    if len(stock_code) == 1:
                        conditions.append("code = %(stock_code)s")
                        params['stock_code'] = stock_code[0]
                    else:
                        placeholders = ", ".join([f"%(stock_code_{i})s" for i in range(len(stock_code))])
                        conditions.append(f"code IN ({placeholders})")
                        for i, code in enumerate(stock_code):
                            params[f'stock_code_{i}'] = code

            # K线周期条件
            if level:
                conditions.append("level = %(level)s")
                params['level'] = level

            # 日期条件
            if start_date:
                conditions.append("date >= %(start_date)s")
                params['start_date'] = start_date

            if end_date:
                conditions.append("date <= %(end_date)s")
                params['end_date'] = end_date

            # 构建完整查询 - 修复SQL语法
            query = f"""
            SELECT date, code, name, open, high, low, close, volume,
                   turnover_rate, price_change, price_range, level
            FROM stock.stock_info
            WHERE {' AND '.join(conditions)}
            ORDER BY date DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            # 执行查询
            result = self.query(query, params)

            # 创建Stock_info对象
            if result.empty:
                logger.warning(f"未找到股票 {stock_code} 的数据")
                return Stock_info()

            logger.debug(f"获取到 {len(result)} 条股票数据")
            return Stock_info(result)

        except Exception as e:
            logger.error(f"获取股票信息失败: {e}")
            return Stock_info()

    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            result = self.execute("SELECT 1")
            is_connected = result == [(1,)]
            if is_connected:
                logger.info("ClickHouse连接测试成功")
            return is_connected
        except Exception as e:
            logger.error(f"ClickHouse连接测试失败: {e}")
            return False

    def close(self):
        """关闭所有连接"""
        with self._connection_lock:
            for thread_id, client in list(self._connections.items()):
                try:
                    client.disconnect()
                except:
                    pass
            self._connections.clear()
            logger.info("所有数据库连接已关闭")


def get_clickhouse_db_final_fixed(config=None) -> ClickHouseDatabaseFixed:
    """
    获取最终修复版的ClickHouse数据库实例
    """
    return ClickHouseDatabaseFixed(config)