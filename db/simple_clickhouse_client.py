#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化的ClickHouse客户端

专门为测试环境设计，避免连接泄漏问题。

Author: AI Assistant
Date: 2025-07-19
"""

import time
import logging
import pandas as pd
from typing import Optional, Dict, Any
import clickhouse_connect
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SimpleClickHouseClient:
    """简化的ClickHouse客户端，专注于稳定性"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化客户端"""
        self.config = config or {
            'host': 'localhost',
            'port': 9000,
            'user': 'default',
            'password': '123456',
            'database': 'stock'
        }
        self._connection_count = 0
        
    @contextmanager
    def get_connection(self):
        """获取连接的上下文管理器"""
        client = None
        try:
            self._connection_count += 1
            logger.debug(f"创建连接 #{self._connection_count}")
            
            client = clickhouse_connect.get_client(
                host=self.config['host'],
                port=self.config['port'],
                username=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                connect_timeout=5,
                send_receive_timeout=15,
                pool_mgr=False  # 禁用内部连接池
            )
            
            # 测试连接
            client.ping()
            yield client
            
        except Exception as e:
            logger.error(f"连接失败: {e}")
            raise
        finally:
            if client:
                try:
                    client.disconnect()
                    logger.debug(f"关闭连接 #{self._connection_count}")
                except Exception as e:
                    logger.warning(f"关闭连接失败: {e}")
    
    def query(self, sql: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """执行查询"""
        for attempt in range(max_retries):
            try:
                with self.get_connection() as client:
                    result = client.query_df(sql)
                    logger.debug(f"查询成功，返回 {len(result)} 行数据")
                    return result
                    
            except Exception as e:
                logger.warning(f"查询失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"查询最终失败: {sql[:100]}...")
                    return None
                time.sleep(0.5)  # 短暂等待后重试
        
        return None
    
    def execute(self, sql: str, max_retries: int = 3) -> bool:
        """执行SQL语句"""
        for attempt in range(max_retries):
            try:
                with self.get_connection() as client:
                    client.command(sql)
                    logger.debug("SQL执行成功")
                    return True
                    
            except Exception as e:
                logger.warning(f"SQL执行失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"SQL执行最终失败: {sql[:100]}...")
                    return False
                time.sleep(0.5)
        
        return False


# 全局简化客户端实例
_simple_client = None


def get_simple_clickhouse_client() -> SimpleClickHouseClient:
    """获取全局简化客户端实例"""
    global _simple_client
    if _simple_client is None:
        _simple_client = SimpleClickHouseClient()
    return _simple_client


def test_simple_client():
    """测试简化客户端"""
    print("🔍 测试简化ClickHouse客户端...")
    
    client = get_simple_clickhouse_client()
    
    # 测试基本查询
    result = client.query("SELECT COUNT(*) as count FROM stock.stock_info")
    if result is not None and not result.empty:
        count = result.iloc[0, 0]
        print(f"✅ 连接测试成功，数据库中有 {count} 条记录")
        return True
    else:
        print("❌ 连接测试失败")
        return False


if __name__ == "__main__":
    test_simple_client()
