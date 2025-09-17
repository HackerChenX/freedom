"""
优化的数据访问管理器
集成连接池优化、查询优化和多层缓存的统一数据访问接口
"""

import time
import threading
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager
import pandas as pd
import logging

from db.enhanced_connection_pool import ClickHouseConnectionPool, get_connection_pool
from db.query_optimizer import QueryOptimizer, get_query_optimizer
from db.multi_layer_cache import MultiLayerCache, get_multi_cache
from db.interfaces.data_access_interface import DataAccessInterface
from utils.enhanced_performance_monitor import performance_monitor
from utils.enhanced_exception_handler import exception_handler, ErrorSeverity, ErrorCategory
from config.unified_config_manager import get_config
from db.sql_manager import SQLManager, QueryType

logger = logging.getLogger(__name__)


class OptimizedDataAccessManager(DataAccessInterface):
    """
    优化的数据访问管理器
    
    集成功能：
    - 优化的连接池管理
    - 智能查询优化
    - 多层缓存机制
    - 性能监控和统计
    - 自动重试和错误恢复
    """
    
    def __init__(self,
                 connection_pool: Optional[ClickHouseConnectionPool] = None,
                 query_optimizer: Optional[QueryOptimizer] = None,
                 cache_manager: Optional[MultiLayerCache] = None,
                 enable_cache: bool = True,
                 enable_query_optimization: bool = True,
                 enable_monitoring: bool = True):
        """
        初始化优化数据访问管理器
        
        Args:
            connection_pool: 连接池实例
            query_optimizer: 查询优化器实例
            cache_manager: 缓存管理器实例
            enable_cache: 是否启用缓存
            enable_query_optimization: 是否启用查询优化
            enable_monitoring: 是否启用监控
        """
        self.connection_pool = connection_pool or get_connection_pool()
        self.query_optimizer = query_optimizer or get_query_optimizer()
        self.cache_manager = cache_manager or get_multi_cache()
        
        self.enable_cache = enable_cache
        self.enable_query_optimization = enable_query_optimization
        self.enable_monitoring = enable_monitoring
        
        # 统计信息
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'query_optimizations': 0,
            'errors': 0,
            'total_execution_time': 0.0
        }
        self.stats_lock = threading.RLock()
        
        logger.info(f"优化数据访问管理器初始化完成 - "
                   f"缓存: {enable_cache}, 优化: {enable_query_optimization}, 监控: {enable_monitoring}")
    
    def _generate_cache_key(self, query: str, params: Optional[Dict] = None) -> str:
        """生成缓存键"""
        import hashlib
        content = query + str(sorted((params or {}).items()))
        return f"query:{hashlib.md5(content.encode()).hexdigest()}"
    
    def _update_stats(self, execution_time: float, cache_hit: bool = False, error: bool = False):
        """更新统计信息"""
        with self.stats_lock:
            self.stats['total_queries'] += 1
            self.stats['total_execution_time'] += execution_time
            
            if cache_hit:
                self.stats['cache_hits'] += 1
            else:
                self.stats['cache_misses'] += 1
            
            if error:
                self.stats['errors'] += 1
    
    @performance_monitor(threshold_seconds=2.0)
    @exception_handler(severity=ErrorSeverity.MEDIUM, category=ErrorCategory.DATABASE)
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Any]:
        """
        执行查询语句
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            List[Any]: 查询结果
        """
        start_time = time.time()
        cache_hit = False
        
        try:
            # 1. 检查缓存
            if self.enable_cache:
                cache_key = self._generate_cache_key(query, params)
                cached_result = self.cache_manager.get(cache_key)
                
                if cached_result is not None:
                    cache_hit = True
                    execution_time = time.time() - start_time
                    self._update_stats(execution_time, cache_hit=True)
                    
                    if self.enable_monitoring:
                        self.query_optimizer.record_query_execution(query, execution_time, True)
                    
                    logger.debug(f"缓存命中: {cache_key[:16]}...")
                    return cached_result
            
            # 2. 查询优化
            optimized_query = query
            optimized_params = params or {}
            
            if self.enable_query_optimization:
                try:
                    optimized_query, optimized_params = self.query_optimizer.optimize_query(query, params)
                    with self.stats_lock:
                        self.stats['query_optimizations'] += 1
                except Exception as e:
                    logger.debug(f"查询优化失败，使用原始查询: {e}")
            
            # 3. 执行查询
            with self.connection_pool.get_connection() as conn:
                result = conn.execute(optimized_query, optimized_params)
            
            # 4. 缓存结果
            if self.enable_cache and result:
                try:
                    cache_ttl = get_config('database.cache_ttl', 300)
                    self.cache_manager.set(cache_key, result, cache_ttl)
                except Exception as e:
                    logger.debug(f"缓存设置失败: {e}")
            
            # 5. 更新统计
            execution_time = time.time() - start_time
            self._update_stats(execution_time, cache_hit=False)
            
            if self.enable_monitoring:
                self.query_optimizer.record_query_execution(optimized_query, execution_time, True)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_stats(execution_time, cache_hit=cache_hit, error=True)
            
            if self.enable_monitoring:
                self.query_optimizer.record_query_execution(query, execution_time, False)
            
            logger.error(f"查询执行失败: {e}")
            raise
    
    @performance_monitor(threshold_seconds=2.0)
    def query_dataframe(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        执行查询并返回DataFrame
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果DataFrame
        """
        start_time = time.time()
        cache_hit = False
        
        try:
            # 1. 检查缓存
            if self.enable_cache:
                cache_key = self._generate_cache_key(f"df:{query}", params)
                cached_result = self.cache_manager.get(cache_key)
                
                if cached_result is not None:
                    cache_hit = True
                    execution_time = time.time() - start_time
                    self._update_stats(execution_time, cache_hit=True)
                    
                    logger.debug(f"DataFrame缓存命中: {cache_key[:16]}...")
                    return cached_result
            
            # 2. 查询优化
            optimized_query = query
            optimized_params = params or {}
            
            if self.enable_query_optimization:
                try:
                    optimized_query, optimized_params = self.query_optimizer.optimize_query(query, params)
                except Exception as e:
                    logger.debug(f"查询优化失败，使用原始查询: {e}")
            
            # 3. 执行查询
            with self.connection_pool.get_connection() as conn:
                result_df = conn.query_dataframe(optimized_query, optimized_params)
            
            # 4. 缓存结果
            if self.enable_cache and not result_df.empty:
                try:
                    cache_ttl = get_config('database.cache_ttl', 300)
                    self.cache_manager.set(cache_key, result_df, cache_ttl)
                except Exception as e:
                    logger.debug(f"DataFrame缓存设置失败: {e}")
            
            # 5. 更新统计
            execution_time = time.time() - start_time
            self._update_stats(execution_time, cache_hit=False)
            
            if self.enable_monitoring:
                self.query_optimizer.record_query_execution(optimized_query, execution_time, True)
            
            return result_df
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_stats(execution_time, cache_hit=cache_hit, error=True)
            
            if self.enable_monitoring:
                self.query_optimizer.record_query_execution(query, execution_time, False)
            
            logger.error(f"DataFrame查询执行失败: {e}")
            return pd.DataFrame()
    
    def get_stock_data(self, code: str, start_date: str, end_date: str, level: str = '日线') -> pd.DataFrame:
        """
        获取股票数据（优化版本）
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            level: 数据级别
            
        Returns:
            pd.DataFrame: 股票数据
        """
        query = """
        SELECT code, name, date, open, high, low, close, volume, turnover_rate
        FROM stock_info WHERE level = %(level)s AND code = %(code)s
        AND level = %(level)s
        AND date >= %(start_date)s AND date <= %(end_date)s
        ORDER BY date ASC
        """
        
        params = {
            'code': code,
            'level': level,
            'start_date': start_date,
            'end_date': end_date
        }
        
        return self.query_dataframe(query, params)
    
    def get_batch_stock_data(self, codes: List[str], start_date: str, end_date: str, level: str = '日线') -> pd.DataFrame:
        """
        批量获取股票数据（优化版本）
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            level: 数据级别
            
        Returns:
            pd.DataFrame: 批量股票数据
        """
        # 使用优化的批量查询模板
        query = """
        SELECT code, name, date, open, high, low, close, volume, turnover_rate
        FROM stock_info WHERE code = %(code)s AND level = %(level)s AND code IN %(codes)s
        AND level = %(level)s
        AND date >= %(start_date)s AND date <= %(end_date)s
        ORDER BY code, date ASC
        """
        
        params = {
            'codes': tuple(codes),
            'level': level,
            'start_date': start_date,
            'end_date': end_date
        }
        
        return self.query_dataframe(query, params)
    
    def clear_cache(self):
        """清空缓存"""
        if self.enable_cache:
            self.cache_manager.clear()
            logger.info("数据访问缓存已清空")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        with self.stats_lock:
            # 计算缓存命中率
            total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
            cache_hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0.0
            
            # 计算平均执行时间
            avg_execution_time = (self.stats['total_execution_time'] / self.stats['total_queries']) if self.stats['total_queries'] > 0 else 0.0
            
            # 获取连接池统计
            pool_stats = self.connection_pool.get_statistics()
            
            # 获取查询优化器统计
            optimizer_stats = self.query_optimizer.get_query_statistics()
            
            # 获取缓存统计
            cache_stats = self.cache_manager.get_statistics()
            
            return {
                'data_access_stats': {
                    'total_queries': self.stats['total_queries'],
                    'cache_hits': self.stats['cache_hits'],
                    'cache_misses': self.stats['cache_misses'],
                    'cache_hit_rate': cache_hit_rate,
                    'query_optimizations': self.stats['query_optimizations'],
                    'errors': self.stats['errors'],
                    'avg_execution_time': avg_execution_time
                },
                'connection_pool_stats': pool_stats,
                'query_optimizer_stats': optimizer_stats,
                'cache_stats': cache_stats
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        try:
            # 测试连接池
            with self.connection_pool.get_connection() as conn:
                conn.execute("SELECT 1")

            pool_healthy = True
        except Exception as e:
            pool_healthy = False
            logger.error(f"连接池健康检查失败: {e}")

        # 获取统计信息
        stats = self.get_performance_stats()

        # 计算健康分数
        health_score = 100
        if stats['data_access_stats']['errors'] > 0:
            error_rate = stats['data_access_stats']['errors'] / stats['data_access_stats']['total_queries']
            health_score -= min(50, error_rate * 100)

        if not pool_healthy:
            health_score -= 30

        return {
            'healthy': pool_healthy and health_score > 70,
            'health_score': health_score,
            'connection_pool_healthy': pool_healthy,
            'cache_enabled': self.enable_cache,
            'optimization_enabled': self.enable_query_optimization,
            'monitoring_enabled': self.enable_monitoring,
            'stats_summary': stats['data_access_stats']
        }

    # 实现DataAccessInterface的抽象方法
    def get_stock_data_data_access_interface(self, code: str, start_date: str, end_date: str,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """获取股票数据（接口实现）"""
        return self.get_stock_data(code, start_date, end_date)

    def get_stocks_data_batch_data_access_interface(self, codes: List[str], start_date: str, end_date: str,
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """批量获取股票数据（接口实现）"""
        return self.get_batch_stock_data(codes, start_date, end_date)

    def get_indicator_data_data_access_interface(self, code: str, indicator: str, start_date: str, end_date: str,
                          params: Optional[Dict] = None) -> pd.DataFrame:
        """获取指标数据（接口实现）"""
        # 基础实现，可以根据需要扩展
        query = """
        SELECT code, date, close, volume
        FROM stock_info WHERE level = %(level)s AND code = %(code)s
        AND date >= %(start_date)s AND date <= %(end_date)s
        ORDER BY date ASC
        """

        query_params = {
            'code': code,
            'start_date': start_date,
            'end_date': end_date
        }

        return self.query_dataframe(query, query_params)

    def get_stock_list_data_access_interface(self, market: Optional[str] = None) -> List[str]:
        """获取股票列表（接口实现）"""
        query = "SELECT DISTINCT code FROM stock_info"
        conditions = []
        params = {}

        if params)
        return [row[0] for row in result] if result else []

    def get__list_data_access_interface(self) -> List[str]:
        """获取行业列表（接口实现）"""
        query = "SELECT DISTINCT FROM stock_info WHERE code = %(code)s AND level = %(level)s AND IS NOT NULL ORDER BY result = self.execute_query(query)
        return [row[0] for row in result] if result else []

    def execute_query_data_access_interface(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """执行查询（接口实现）"""
        return self.query_dataframe(query, params)

    def check_data_exists_data_access_interface(self, table: str, conditions: Dict) -> bool:
        """检查数据是否存在（接口实现）"""
        where_clauses = []
        params = {}

        for key, value in conditions.items():
            where_clauses.append(f"{key} = %({key})s")
            params[key] = value

        query = f"SELECT 1 FROM {table} WHERE {' AND '.join(where_clauses)} LIMIT 1"
        result = self.execute_query(query, params)
        return len(result) > 0

    def get_latest_data_data_access_interface(self, table: str, code: str, columns: Optional[List[str]] = None) -> Optional[Dict]:
        """获取最新数据（接口实现）"""
        cols = ', '.join(columns) if columns else '*'
        query = f"""
        SELECT {cols}
        FROM {table}
        WHERE code = %(code)s
        ORDER BY date DESC
        LIMIT 1
        """

        params = {'code': code}
        df = self.query_dataframe(query, params)

        if df.empty:
            return None

        return df.iloc[0].to_dict()


# 全局优化数据访问管理器实例
_optimized_manager = None
_manager_lock = threading.Lock()


def get_optimized_data_access_manager() -> OptimizedDataAccessManager:
    """获取全局优化数据访问管理器实例"""
    global _optimized_manager
    
    if _optimized_manager is None:
        with _manager_lock:
            if _optimized_manager is None:
                _optimized_manager = OptimizedDataAccessManager()
    
    return _optimized_manager


# 导出主要类和函数
__all__ = [
    'OptimizedDataAccessManager',
    'get_optimized_data_access_manager'
]
