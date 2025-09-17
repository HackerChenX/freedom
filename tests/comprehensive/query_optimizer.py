#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClickHouse查询优化器

提供针对ClickHouse的查询优化，支持高性能的股票数据查询
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import pandas as pd
import json
import re

from utils.logger import getLogger
from db.interfaces.data_access_interface import DataAccessInterface
from db.sql_manager import SQLManager, QueryType
from .cache_manager import get_cache_manager

logger = getLogger(__name__)


@dataclass
class QueryStats:
    """查询统计信息"""
    query_count: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    rows_fetched: int = 0
    bytes_fetched: int = 0


class ClickHouseOptimizer:
    """ClickHouse查询优化器"""
    
    def __init__(self, data_access: DataAccessInterface):
        """
        初始化ClickHouse查询优化器
        
        Args:
            data_access: 数据访问接口
        """
        self.data_access = data_access
        self.cache_manager = get_cache_manager()
        self.connection_pool_size = 50
        self.batch_size = 1000
        self.query_timeout = 30
        self.stats = QueryStats()
        self.lock = threading.RLock()
        
        logger.info("ClickHouse查询优化器初始化完成")
    
    def optimize_stock_universe_query(self, 
                                    date_range: Tuple[str, str],
                                    conditions: Dict[str, Any]) -> str:
        """
        优化股票池查询
        
        Args:
            date_range: 日期范围 (start_date, end_date)
            conditions: 查询条件
            
        Returns:
            str: 优化后的SQL查询
        """
        start_date, end_date = date_range
        
        # 构建基础查询
        base_query = """
        SELECT DISTINCT code
        FROM stock_daily
        PREWHERE date BETWEEN '{start_date}' AND '{end_date}'
        """
        
        # 添加WHERE条件
        where_clauses = []
        
        if 'min_volume' in conditions and conditions['min_volume'] > 0:
            where_clauses.append(f"volume >= {conditions['min_volume']}")
        
        if 'min_price' in conditions and conditions['min_price'] > 0:
            where_clauses.append(f"close >= {conditions['min_price']}")
        
        if 'max_price' in conditions and conditions['max_price'] < float('inf'):
            where_clauses.append(f"close <= {conditions['max_price']}")
        
        if 'exclude_st' in conditions and conditions['exclude_st']:
            where_clauses.append("NOT match(name, '.*ST.*')")
        
        if 'exclude_suspended' in conditions and conditions['exclude_suspended']:
            where_clauses.append("volume > 0")
        
        # 添加WHERE子句
        if where_clauses:
            where_clause = " AND ".join(where_clauses)
            base_query += f"\nWHERE {where_clause}"
        
        # 添加优化提示
        base_query += """
        SETTINGS max_threads = 8,
                 max_memory_usage = 10000000000,
                 use_uncompressed_cache = 1,
                 max_execution_time = 30
        """
        
        # 格式化查询
        optimized_query = base_query.format(start_date=start_date, end_date=end_date)
        
        return optimized_query
    
    def optimize_stock_data_query(self, 
                                stock_codes: List[str],
                                date_range: Tuple[str, str],
                                columns: List[str] = None) -> str:
        """
        优化股票数据查询
        
        Args:
            stock_codes: 股票代码列表
            date_range: 日期范围 (start_date, end_date)
            columns: 要查询的列，None表示所有列
            
        Returns:
            str: 优化后的SQL查询
        """
        start_date, end_date = date_range
        
        # 默认列
        if columns is None:
            columns = [
                'code', 'name', 'date', 'level', 'open', 'close', 'high', 'low',
                'volume', 'turnover_rate', 'price_change', 'price_range', 'industry'
            ]
        
        # 构建列列表
        columns_str = ', '.join(columns)
        
        # 构建股票代码IN子句
        # 对于大量股票，使用字典表优化
        if len(stock_codes) > 100:
            # 创建临时字典表
            temp_table = f"tmp_stock_codes_{int(time.time())}"
            stock_codes_str = "', '".join(stock_codes)
            
            dict_query = f"""
            CREATE TEMPORARY TABLE {temp_table} (code String)
            ENGINE = Memory AS
            SELECT code FROM VALUES('{stock_codes_str}')
            """
            
            # 主查询使用JOIN
            base_query = f"""
            SELECT {columns_str}
            FROM stock_daily AS s
            INNER JOIN {temp_table} AS t ON s.code = t.code
            PREWHERE date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY code, date
            """
            
            # 添加优化提示
            base_query += """
            SETTINGS max_threads = 8,
                     max_memory_usage = 10000000000,
                     use_uncompressed_cache = 1,
                     max_execution_time = 30
            """
            
            return dict_query + ";\n" + base_query
        else:
            # 对于少量股票，直接使用IN
            stock_codes_str = "', '".join(stock_codes)
            
            base_query = f"""
            SELECT {columns_str}
            FROM stock_daily
            PREWHERE code IN ('{stock_codes_str}')
                AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY code, date
            """
            
            # 添加优化提示
            base_query += """
            SETTINGS max_threads = 8,
                     max_memory_usage = 10000000000,
                     use_uncompressed_cache = 1,
                     max_execution_time = 30
            """
            
            return base_query
    
    def optimize_pattern_query(self, 
                             pattern_id: str,
                             date_range: Tuple[str, str],
                             conditions: Dict[str, Any] = None) -> str:
        """
        优化形态查询
        
        Args:
            pattern_id: 形态ID
            date_range: 日期范围 (start_date, end_date)
            conditions: 查询条件
            
        Returns:
            str: 优化后的SQL查询
        """
        start_date, end_date = date_range
        conditions = conditions or {}
        
        # 解析形态ID，提取指标名称
        indicator_name = self._extract_indicator_from_pattern(pattern_id)
        
        # 构建基础查询
        base_query = f"""
        SELECT s.code, s.name, s.date, s.close, s.volume, s.industry
        FROM stock_daily AS s
        INNER JOIN indicator_results AS i ON s.code = i.code AND s.date = i.date
        PREWHERE s.date BETWEEN '{start_date}' AND '{end_date}'
            AND i.indicator = '{indicator_name}'
        """
        
        # 添加形态条件
        pattern_condition = self._get_pattern_condition(pattern_id)
        if pattern_condition:
            base_query += f"AND {pattern_condition}\n"
        
        # 添加其他条件
        where_clauses = []
        
        if 'min_volume' in conditions and conditions['min_volume'] > 0:
            where_clauses.append(f"s.volume >= {conditions['min_volume']}")
        
        if 'min_price' in conditions and conditions['min_price'] > 0:
            where_clauses.append(f"s.close >= {conditions['min_price']}")
        
        if 'exclude_st' in conditions and conditions['exclude_st']:
            where_clauses.append("NOT match(s.name, '.*ST.*')")
        
        # 添加WHERE子句
        if where_clauses:
            where_clause = " AND ".join(where_clauses)
            base_query += f"WHERE {where_clause}\n"
        
        # 添加排序和限制
        base_query += """
        ORDER BY s.date DESC
        LIMIT 1000
        """
        
        # 添加优化提示
        base_query += """
        SETTINGS max_threads = 8,
                 max_memory_usage = 10000000000,
                 use_uncompressed_cache = 1,
                 max_execution_time = 30
        """
        
        return base_query
    
    def _extract_indicator_from_pattern(self, pattern_id: str) -> str:
        """
        从形态ID提取指标名称
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            str: 指标名称
        """
        # 形态ID通常格式为 INDICATOR_PATTERN
        parts = pattern_id.split('_', 1)
        return parts[0] if parts else pattern_id
    
    def _get_pattern_condition(self, pattern_id: str) -> str:
        """
        获取形态条件
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            str: 形态条件SQL
        """
        pattern_upper = pattern_id.upper()
        
        # 常见形态条件映射
        pattern_conditions = {
            'BULLISH': "i.value->>'signal' = 'bullish'",
            'BEARISH': "i.value->>'signal' = 'bearish'",
            'GOLDEN_CROSS': "i.value->>'cross_type' = 'golden'",
            'DEATH_CROSS': "i.value->>'cross_type' = 'death'",
            'OVERSOLD': "i.value->>'condition' = 'oversold'",
            'OVERBOUGHT': "i.value->>'condition' = 'overbought'"
        }
        
        # 检查形态ID是否包含已知的形态条件
        for pattern, condition in pattern_conditions.items():
            if pattern in pattern_upper:
                return condition
        
        # 默认条件
        return "i.value->>'pattern' = '{pattern_id}'"
    
    def execute_optimized_query(self, query: str) -> pd.DataFrame:
        """
        执行优化后的查询
        
        Args:
            query: SQL查询
            
        Returns:
            pd.DataFrame: 查询结果
        """
        start_time = time.time()
        
        try:
            # 检查缓存
            cache_key = self._get_query_cache_key(query)
            cached_result = self.cache_manager.get('query_results', cache_key)
            
            if cached_result is not None:
                # 缓存命中
                with self.lock:
                    self.stats.query_count += 1
                    self.stats.cache_hits += 1
                    execution_time = time.time() - start_time
                    self.stats.total_execution_time += execution_time
                
                logger.debug(f"查询缓存命中，执行时间: {execution_time:.3f}秒")
                return cached_result
            
            # 缓存未命中，执行查询
            # 这里简化处理，实际应该调用数据访问接口执行查询
            result = self._execute_query(query)
            
            # 更新统计信息
            with self.lock:
                self.stats.query_count += 1
                self.stats.cache_misses += 1
                execution_time = time.time() - start_time
                self.stats.total_execution_time += execution_time
                
                if execution_time > self.stats.max_execution_time:
                    self.stats.max_execution_time = execution_time
                
                if self.stats.min_execution_time == 0 or execution_time < self.stats.min_execution_time:
                    self.stats.min_execution_time = execution_time
                
                self.stats.avg_execution_time = self.stats.total_execution_time / self.stats.query_count
                
                if isinstance(result, pd.DataFrame):
                    self.stats.rows_fetched += len(result)
                    self.stats.bytes_fetched += result.memory_usage(deep=True).sum()
            
            # 缓存结果
            self.cache_manager.put('query_results', cache_key, result, persist=False)
            
            logger.debug(f"查询执行完成，执行时间: {execution_time:.3f}秒，返回 {len(result)} 行")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"查询执行失败: {e}，执行时间: {execution_time:.3f}秒")
            
            # 更新统计信息
            with self.lock:
                self.stats.query_count += 1
                self.stats.total_execution_time += execution_time
            
            # 返回空DataFrame
            return pd.DataFrame()
    
    def _execute_query(self, query: str) -> pd.DataFrame:
        """
        执行查询（实际实现）
        
        Args:
            query: SQL查询
            
        Returns:
            pd.DataFrame: 查询结果
        """
        # 这里应该调用数据访问接口执行查询
        # 简化实现，返回空DataFrame
        return pd.DataFrame()
    
    def _get_query_cache_key(self, query: str) -> str:
        """
        获取查询缓存键
        
        Args:
            query: SQL查询
            
        Returns:
            str: 缓存键
        """
        # 规范化查询
        normalized_query = self._normalize_query(query)
        
        # 使用规范化查询的哈希作为缓存键
        import hashlib
        return hashlib.md5(normalized_query.encode()).hexdigest()
    
    def _normalize_query(self, query: str) -> str:
        """
        规范化查询
        
        Args:
            query: SQL查询
            
        Returns:
            str: 规范化后的查询
        """
        # 移除注释
        query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
        
        # 移除多余的空白字符
        query = re.sub(r'\s+', ' ', query)
        
        # 移除SETTINGS子句
        query = re.sub(r'SETTINGS.*$', '', query)
        
        return query.strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取查询统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            return {
                'query_count': self.stats.query_count,
                'total_execution_time': self.stats.total_execution_time,
                'avg_execution_time': self.stats.avg_execution_time,
                'max_execution_time': self.stats.max_execution_time,
                'min_execution_time': self.stats.min_execution_time,
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'cache_hit_rate': self.stats.cache_hits / max(1, self.stats.query_count),
                'rows_fetched': self.stats.rows_fetched,
                'bytes_fetched': self.stats.bytes_fetched,
                'bytes_fetched_mb': self.stats.bytes_fetched / (1024 * 1024)
            }
    
    def clear_stats(self) -> None:
        """清空统计信息"""
        with self.lock:
            self.stats = QueryStats()


def main():
    """测试ClickHouse查询优化器"""
    from db.interfaces.data_access_interface import DataAccessInterface
from db.sql_manager import SQLManager, QueryType
    
    # 创建数据访问接口
    data_access = None  # 实际应用中应该获取真实的数据访问接口
    
    # 创建查询优化器
    optimizer = ClickHouseOptimizer(data_access)
    
    # 测试股票池查询优化
    date_range = ('20240101', '20241231')
    conditions = {
        'min_volume': 1000000,
        'min_price': 10.0,
        'exclude_st': True
    }
    
    optimized_query = optimizer.optimize_stock_universe_query(date_range, conditions)
    print("优化后的股票池查询:")
    print(optimized_query)
    
    # 测试股票数据查询优化
    stock_codes = ['000001', '000002', '000063', '600036', '600519']
    optimized_query = optimizer.optimize_stock_data_query(stock_codes, date_range)
    print("\n优化后的股票数据查询:")
    print(optimized_query)
    
    # 测试形态查询优化
    pattern_id = 'MACD_GOLDEN_CROSS'
    optimized_query = optimizer.optimize_pattern_query(pattern_id, date_range, conditions)
    print("\n优化后的形态查询:")
    print(optimized_query)


if __name__ == "__main__":
    main()