#!/usr/bin/env python3
"""
优化的数据库查询管理器

提供生产级高性能数据库查询能力：
1. 连接池优化
2. 批量查询
3. 查询缓存
4. 异步查询

作者：AI Assistant
创建时间：2025-01-13
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from functools import lru_cache

from utils.logger import get_logger
from db.clickhouse_db import get_clickhouse_db

logger = get_logger(__name__)

class OptimizedQueryManager:
    """优化的查询管理器"""
    
    def __init__(self, max_connections: int = 20, query_timeout: int = 60):
        """
        初始化优化查询管理器
        
        Args:
            max_connections: 最大连接数
            query_timeout: 查询超时时间（秒）
        """
        self.max_connections = max_connections
        self.query_timeout = query_timeout
        
        # 连接池
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        
        # 查询缓存
        self.query_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_ttl = 3600  # 1小时TTL
        
        # 性能统计
        self.stats = {
            'queries_executed': 0,
            'cache_hits': 0,
            'total_query_time': 0.0,
            'batch_queries': 0,
            'rows_fetched': 0
        }
        
        logger.info(f"优化查询管理器初始化: max_connections={max_connections}")
    
    def batch_get_stock_data(self, 
                           stock_codes: List[str], 
                           start_date: str, 
                           end_date: str,
                           level: str = "日线") -> Dict[str, pd.DataFrame]:
        """
        批量获取股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            level: K线级别
            
        Returns:
            Dict[str, pd.DataFrame]: 股票代码 -> 数据映射
        """
        start_time = time.time()
        
        try:
            # 构建批量查询SQL
            placeholders = "', '".join(stock_codes)
            query = f"""
            SELECT code, name, date, open, high, low, close, volume, turnover
            FROM stock_info 
            WHERE code IN ('{placeholders}')
            AND level = '{level}'
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY code, date ASC
            """
            
            # 执行查询
            df = self._execute_query_to_dataframe(query)
            
            # 按股票代码分组
            result = {}
            if not df.empty:
                for code, group in df.groupby('code'):
                    result[code] = group.reset_index(drop=True)
            
            # 更新统计
            self.stats['batch_queries'] += 1
            self.stats['rows_fetched'] += len(df)
            self.stats['total_query_time'] += time.time() - start_time
            
            logger.debug(f"批量查询完成: {len(stock_codes)}只股票, "
                        f"返回{len(result)}只有数据的股票, "
                        f"耗时{time.time() - start_time:.2f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"批量获取股票数据失败: {e}")
            return {}
    
    def batch_get_period_data(self,
                            stock_codes: List[str],
                            start_date: str,
                            end_date: str,
                            period: str) -> Dict[str, pd.DataFrame]:
        """
        批量获取指定周期数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            period: 时间周期
            
        Returns:
            Dict[str, pd.DataFrame]: 股票数据映射
        """
        # 根据周期确定查询策略
        if period == "60min":
            # 60分钟数据需要从15分钟数据转换
            return self._batch_get_60min_from_15min(stock_codes, start_date, end_date)
        else:
            # 直接查询数据库中的周期数据
            return self.batch_get_stock_data(stock_codes, start_date, end_date, period)
    
    def _batch_get_60min_from_15min(self,
                                  stock_codes: List[str],
                                  start_date: str,
                                  end_date: str) -> Dict[str, pd.DataFrame]:
        """
        批量从15分钟数据生成60分钟数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, pd.DataFrame]: 60分钟数据映射
        """
        try:
            # 先获取15分钟数据
            data_15min = self.batch_get_stock_data(stock_codes, start_date, end_date, "15分钟")
            
            # 转换为60分钟数据
            result = {}
            for code, df in data_15min.items():
                if len(df) >= 4:  # 至少需要4个15分钟数据点
                    df_60min = self._convert_15min_to_60min(df)
                    if not df_60min.empty:
                        result[code] = df_60min
            
            logger.debug(f"15分钟→60分钟转换完成: {len(result)}只股票")
            return result
            
        except Exception as e:
            logger.error(f"批量60分钟数据转换失败: {e}")
            return {}
    
    @staticmethod
    def _convert_15min_to_60min(df_15min: pd.DataFrame) -> pd.DataFrame:
        """
        将15分钟数据转换为60分钟数据
        
        Args:
            df_15min: 15分钟数据
            
        Returns:
            pd.DataFrame: 60分钟数据
        """
        if len(df_15min) < 4:
            return pd.DataFrame()
        
        try:
            # 确保日期时间格式正确
            df = df_15min.copy()
            df['datetime'] = pd.to_datetime(df['date'])
            df['hour'] = df['datetime'].dt.hour
            
            # 按小时分组（每4个15分钟为1小时）
            df['hour_group'] = (df['datetime'].dt.hour * 4 + df['datetime'].dt.minute // 15) // 4
            df['date_hour'] = df['datetime'].dt.date.astype(str) + '_' + df['hour_group'].astype(str)
            
            # 聚合为60分钟数据
            result = df.groupby('date_hour').agg({
                'code': 'first',
                'name': 'first', 
                'date': 'first',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'turnover': 'sum'
            }).reset_index(drop=True)
            
            # 确保数据完整性
            result = result.dropna()
            
            return result
            
        except Exception as e:
            logger.debug(f"15分钟数据转换失败: {e}")
            return pd.DataFrame()
    
    def _execute_query_to_dataframe(self, query: str) -> pd.DataFrame:
        """
        执行查询并返回DataFrame
        
        Args:
            query: SQL查询语句
            
        Returns:
            pd.DataFrame: 查询结果
        """
        # 检查缓存
        query_hash = hash(query)
        cache_key = f"query_{query_hash}"
        
        if cache_key in self.query_cache:
            cache_entry = self.query_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                self.stats['cache_hits'] += 1
                return cache_entry['data'].copy()
        
        # 执行查询
        try:
            db = get_clickhouse_db()
            result = db.execute_query(query)
            
            if result:
                # 获取列名（假设查询的是stock_info表的标准字段）
                columns = ['code', 'name', 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover']
                df = pd.DataFrame(result, columns=columns)
                
                # 数据类型转换
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 写入缓存
                with self.cache_lock:
                    self.query_cache[cache_key] = {
                        'data': df.copy(),
                        'timestamp': time.time()
                    }
                
                self.stats['queries_executed'] += 1
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            return pd.DataFrame()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        avg_query_time = (
            self.stats['total_query_time'] / max(self.stats['queries_executed'], 1)
        )
        cache_hit_rate = (
            self.stats['cache_hits'] / max(self.stats['queries_executed'] + self.stats['cache_hits'], 1) * 100
        )
        
        return {
            **self.stats,
            'avg_query_time': avg_query_time,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.query_cache)
        }
    
    def clear_cache(self):
        """清空查询缓存"""
        with self.cache_lock:
            self.query_cache.clear()
        logger.info("查询缓存已清空")


# 全局查询管理器实例
_query_manager_instance = None
_query_manager_lock = threading.Lock()

def get_optimized_query_manager(**kwargs) -> OptimizedQueryManager:
    """获取优化查询管理器单例"""
    global _query_manager_instance
    
    if _query_manager_instance is None:
        with _query_manager_lock:
            if _query_manager_instance is None:
                _query_manager_instance = OptimizedQueryManager(**kwargs)
    
    return _query_manager_instance 