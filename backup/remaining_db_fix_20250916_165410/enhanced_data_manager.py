from db.query_executor import get_query_executor
from db.sql_manager import QueryType
#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
增强的数据管理器

使用连接池解决并发查询问题，提供查询缓存和性能优化
"""

import time
import threading
import hashlib
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd

from db.enhanced_connection_pool import get_connection_pool
from utils.logger import getLogger
from utils.decorators import performance_monitor
from utils.exceptions import DataAccessError, DataValidationError
from enums.period import Period
from models.stock_info import StockInfo

logger = getLogger(__name__)


class EnhanceddatamanagerManager:
    """
    增强的数据管理器
    
    特性：
    - 使用连接池支持并发查询
    - 智能查询缓存
    - 查询性能优化
    - 自动重试机制
    """
    
    def __init__(self, 
                 cache_enabled: bool = True,
                 max_cache_size: int = 2000,
                 default_ttl: int = 1800,
                 enable_query_optimization: bool = True):
        """
        初始化增强数据管理器
        
        Args:
            cache_enabled: 是否启用缓存
            max_cache_size: 最大缓存条目数
            default_ttl: 默认缓存有效期（秒）
            enable_query_optimization: 是否启用查询优化
        """
        self.connection_pool = get_connection_pool()
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        self.enable_query_optimization = enable_query_optimization
        
        # 查询缓存
        self.query_cache = {}
        self.cache_timestamps = {}
        self.cache_access_count = {}
        self.cache_lock = threading.RLock()
        
        # 性能统计
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_query_time': 0.0,
            'avg_query_time': 0.0,
            'concurrent_queries': 0,
            'max_concurrent_queries': 0,
            'query_errors': 0
        }
        self.stats_lock = threading.Lock()
        
        # 查询优化配置
        self.optimization_config = {
            'use_limit_for_large_queries': True,
            'default_limit': 100000,
            'enable_query_hints': True,
            'parallel_query_threshold': 1000
        }
        
        logger.info(f"增强数据管理器初始化完成，缓存: {'启用' if cache_enabled else '禁用'}, "
                   f"查询优化: {'启用' if enable_query_optimization else '禁用'}")
    
    @performance_monitor(threshold=1.0)
    def get_stock_info_Manager_Enhanced_Data_Manager(self, 
                       stock_code: Union[str, List[str]] = None,
                       level: Union[str, Period] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       filters: Optional[Dict[str, Any]] = None,
                       limit: Optional[int] = None,
                       order_by: str = "date DESC",
                       cache_ttl: Optional[int] = None) -> StockInfo:
        """
        获取股票数据（支持并发查询）
        
        Args:
            stock_code: 股票代码或股票代码列表
            level: K线周期
            start_date: 开始日期
            end_date: 结束日期
            filters: 过滤条件
            limit: 限制返回记录数
            order_by: 排序规则
            cache_ttl: 缓存有效期
            
        Returns:
            Stock_info: 股票数据对象
        """
        query_start_time = time.time()
        
        try:
            with self.stats_lock:
                self.stats['total_queries'] += 1
                self.stats['concurrent_queries'] += 1
                self.stats['max_concurrent_queries'] = max(
                    self.stats['max_concurrent_queries'],
                    self.stats['concurrent_queries']
                )
            
            # 构建缓存键
            cache_key = self._build_cache_key_Enhanced_Data_Manager({
                'stock_code': stock_code,
                'level': level,
                'start_date': start_date,
                'end_date': end_date,
                'filters': filters,
                'limit': limit,
                'order_by': order_by
            })
            
            # 检查缓存
            ttl = cache_ttl if cache_ttl is not None else self.default_ttl
            cached_result = self._get_from_cache_Enhanced_Data_Manager(cache_key, ttl)
            if cached_result is not None:
                return cached_result
            
            # 构建优化的查询
            query, params = self._build_optimized_query_Enhanced_Data_Manager(
                stock_code, level, start_date, end_date, filters, limit, order_by
            )
            
            # 执行查询
            result_df = self._execute_query_with_retry_Enhanced_Data_Manager(query, params)
            
            # 创建StockInfo对象
            stock_info = StockInfo(result_df)
            
            # 缓存结果
            self._set_cache_Enhanced_Data_Manager(cache_key, stock_info, ttl)
            
            return stock_info
            
        except Exception as e:
            with self.stats_lock:
                self.stats['query_errors'] += 1
            logger.error(f"获取股票数据失败: {e}")
            raise DataAccessError(f"获取股票数据失败: {e}")
        finally:
            query_time = time.time() - query_start_time
            with self.stats_lock:
                self.stats['concurrent_queries'] -= 1
                self.stats['total_query_time'] += query_time
                self.stats['avg_query_time'] = (
                    self.stats['total_query_time'] / self.stats['total_queries']
                )
    
    def _build_optimized_query_Enhanced_Data_Manager(self, 
                              stock_code: Union[str, List[str]] = None,
                              level: Union[str, Period] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              filters: Optional[Dict[str, Any]] = None,
                              limit: Optional[int] = None,
                              order_by: str = "date DESC") -> tuple:
        """构建优化的查询语句"""
        
        # 获取字段列表
        fields = StockInfo.get_fields()
        field_str = ", ".join(fields)
        
        # 构建WHERE条件
        conditions = []
        params = {}
        
        # 股票代码条件
        if stock_code is not None:
            if isinstance(stock_code, list):
                if len(stock_code) == 1:
                    conditions.append("code = %(stock_code)s")
                    params['stock_code'] = stock_code[0]
                elif len(stock_code) > 1:
                    # 优化：使用IN查询
                    placeholders = ", ".join([f"%(stock_code_{i})s" for i in range(len(stock_code))])
                    conditions.append(f"code IN ({placeholders})")
                    for i, code in enumerate(stock_code):
                        params[f'stock_code_{i}'] = code
            else:
                conditions.append("code = %(stock_code)s")
                params['stock_code'] = stock_code
        
        # 级别条件
        if level:
            db_level = self._normalize_level_Enhanced_Data_Manager(level)
            if db_level:
                conditions.append("level = %(level)s")
                params['level'] = db_level
        
        # 日期条件（优化：使用索引友好的格式）
        if start_date:
            conditions.append("date >= %(start_date)s")
            params['start_date'] = start_date
        
        if end_date:
            conditions.append("date <= %(end_date)s")
            params['end_date'] = end_date
        
        # 过滤条件
        if filters:
            self._add_filter_conditions_Enhanced_Data_Manager(conditions, params, filters)
        
        # 构建完整查询
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # 查询优化
        query_hints = ""
        if self.enable_query_optimization and self.optimization_config['enable_query_hints']:
            query_hints = "/* SETTINGS max_threads = 4 */"
        
        # 构建查询语句
        query = f"""
        {query_hints}
        SELECT {field_str}
        FROM stock_info WHERE 1=1
        WHERE {where_clause}
        ORDER BY {order_by}
        """
        
        # 添加LIMIT（查询优化）
        if limit:
            query += f" LIMIT {limit}"
        elif (self.enable_query_optimization and 
              self.optimization_config['use_limit_for_large_queries'] and
              not limit):
            # 对于大查询自动添加限制
            query += f" LIMIT {self.optimization_config['default_limit']}"
        
        return query.strip(), params
    
    def _normalize_level_Enhanced_Data_Manager(self, level: Union[str, Period]) -> Optional[str]:
        """标准化周期参数"""
        if isinstance(level, str):
            level_map = {
                'day': '日线', 'daily': '日线',
                'week': '周线', 'weekly': '周线',
                'month': '月线', 'monthly': '月线',
                '60min': '60分钟', '30min': '30分钟', '15min': '15分钟'
            }
            return level_map.get(level.lower(), level)
        elif isinstance(level, Period):
            period_map = {
                Period.DAILY: '日线',
                Period.WEEKLY: '周线',
                Period.MONTHLY: '月线',
                Period.MIN_60: '60分钟',
                Period.MIN_30: '30分钟',
                Period.MIN_15: '15分钟'
            }
            return period_map.get(level, '日线')
        return None
    
    def _add_filter_conditions_Enhanced_Data_Manager(self, conditions: List[str], params: Dict[str, Any], filters: Dict[str, Any]):
        """添加过滤条件"""
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
        
        # 成交量过滤
        if 'volume' in filters and isinstance(filters['volume'], dict):
            volume = filters['volume']
            if 'min' in volume and volume['min'] > 0:
                conditions.append("volume >= %(volume_min)s")
                params['volume_min'] = volume['min']
    
    def _execute_query_with_retry_Enhanced_Data_Manager(self, query: str, params: Dict[str, Any], max_retries: int = 3) -> pd.DataFrame:
        """执行查询并支持重试"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                with self.connection_pool.get_connection() as conn:
                    result = conn.query_dataframe(query, params)
                    logger.debug(f"查询成功，返回 {len(result)} 条记录")
                    return result
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"查询失败（尝试 {attempt + 1}/{max_retries}）: {e}")
                
                if attempt < max_retries - 1:
                    # 等待后重试
                    time.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    break
        
        # 所有重试都失败
        raise DataAccessError(f"查询失败，已重试 {max_retries} 次: {last_exception}")
    
    def _build_cache_key_Enhanced_Data_Manager(self, params: Dict[str, Any]) -> str:
        """构建缓存键"""
        cache_str = json.dumps(params, sort_keys=True, default=str)
        return f"stock_info_{hashlib.md5(cache_str.encode()).hexdigest()}"
    
    def _get_from_cache_Enhanced_Data_Manager(self, key: str, ttl: int) -> Optional[StockInfo]:
        """从缓存获取数据"""
        if not self.cache_enabled:
            return None
        
        with self.cache_lock:
            if key not in self.query_cache:
                with self.stats_lock:
                    self.stats['cache_misses'] += 1
                return None
            
            # 检查是否过期
            if time.time() - self.cache_timestamps.get(key, 0) > ttl:
                self._remove_from_cache_Enhanced_Data_Manager(key)
                with self.stats_lock:
                    self.stats['cache_misses'] += 1
                return None
            
            # 更新访问统计
            self.cache_access_count[key] = self.cache_access_count.get(key, 0) + 1
            with self.stats_lock:
                self.stats['cache_hits'] += 1
            
            return self.query_cache[key]
    
    def _set_cache_Enhanced_Data_Manager(self, key: str, value: StockInfo, ttl: int):
        """设置缓存"""
        if not self.cache_enabled:
            return
        
        with self.cache_lock:
            # 检查缓存大小
            if len(self.query_cache) >= self.max_cache_size:
                self._evict_cache_item_Enhanced_Data_Manager()
            
            self.query_cache[key] = value
            self.cache_timestamps[key] = time.time()
            self.cache_access_count[key] = 1
    
    def _remove_from_cache_Enhanced_Data_Manager(self, key: str):
        """从缓存中移除项目"""
        self.query_cache.pop(key, None)
        self.cache_timestamps.pop(key, None)
        self.cache_access_count.pop(key, None)
    
    def _evict_cache_item_Enhanced_Data_Manager(self):
        """驱逐最少使用的缓存项"""
        if not self.query_cache:
            return
        
        # 找到访问次数最少的项
        min_key = min(self.cache_access_count.items(), key=lambda x: x[1])[0]
        self._remove_from_cache_Enhanced_Data_Manager(min_key)
        logger.debug(f"缓存驱逐: {min_key}")
    
    def clear_cache_Manager_Enhanced_Data_Manager(self, pattern: Optional[str] = None):
        """清除缓存"""
        with self.cache_lock:
            if pattern is None:
                old_size = len(self.query_cache)
                self.query_cache.clear()
                self.cache_timestamps.clear()
                self.cache_access_count.clear()
                logger.info(f"已清除所有缓存，共 {old_size} 项")
            else:
                keys_to_remove = [k for k in self.query_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    self._remove_from_cache_Enhanced_Data_Manager(key)
                logger.info(f"已清除匹配 '{pattern}' 的缓存，共 {len(keys_to_remove)} 项")
    
    def get_stats_Manager_Enhanced_Data_Manager(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        # 添加缓存统计
        with self.cache_lock:
            stats.update({
                'cache_size': len(self.query_cache),
                'cache_hit_rate': (
                    self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                    if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
                )
            })
        
        # 添加连接池统计
        pool_stats = self.connection_pool.get_stats_Manager_Enhanced_Data_Manager()
        stats['connection_pool'] = pool_stats
        
        return stats
    
    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        return self.connection_pool.get_stats_Manager_Enhanced_Data_Manager()


# 全局实例
_enhanced_data_manager = None
_manager_lock = threading.Lock()


def get_enhanced_data_manager_Manager() -> Enhanced_data_manager:
    """获取全局增强数据管理器实例"""
    global _enhanced_data_manager
    
    if _enhanced_data_manager is None:
        with _manager_lock:
            if _enhanced_data_manager is None:
                _enhanced_data_manager = Enhanced_data_manager_Manager()
    
    return _enhanced_data_manager
