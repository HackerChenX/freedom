"""
查询性能优化模块
提供SQL查询优化、缓存机制和慢查询检测功能
"""

import time
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pandas as pd
import logging

from utils.enhanced_performance_monitor import performance_monitor
from utils.enhanced_exception_handler import exception_handler, ErrorSeverity, ErrorCategory
from config.unified_config_manager import get_config
from db.sql_manager import SQLManager, QueryType

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """查询指标数据"""
    query_hash: str
    query_template: str
    execution_count: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    error_count: int = 0
    last_executed: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def update_execution(self, execution_time: float, success: bool = True):
        """更新执行统计"""
        self.execution_count += 1
        self.total_execution_time += execution_time
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)
        self.avg_execution_time = self.total_execution_time / self.execution_count
        self.last_executed = time.time()
        
        if not success:
            self.error_count += 1


@dataclass
class CacheEntry:
    """缓存条目"""
    data: Any
    created_time: float
    last_accessed: float
    access_count: int = 0
    ttl: float = 300.0  # 默认5分钟TTL
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.created_time > self.ttl
    
    def access(self):
        """记录访问"""
        self.access_count += 1
        self.last_accessed = time.time()


class QueryOptimizer:
    """
    查询优化器
    
    功能：
    - SQL查询模板优化
    - 查询结果缓存
    - 慢查询检测和分析
    - 批量查询优化
    - 分页查询优化
    """
    
    def __init__(self,
                 cache_size: int = 1000,
                 default_ttl: float = 300.0,
                 slow_query_threshold: float = 2.0,
                 enable_cache: bool = True,
                 enable_monitoring: bool = True):
        """
        初始化查询优化器
        
        Args:
            cache_size: 缓存大小
            default_ttl: 默认缓存TTL（秒）
            slow_query_threshold: 慢查询阈值（秒）
            enable_cache: 是否启用缓存
            enable_monitoring: 是否启用监控
        """
        self.cache_size = cache_size
        self.default_ttl = default_ttl
        self.slow_query_threshold = slow_query_threshold
        self.enable_cache = enable_cache
        self.enable_monitoring = enable_monitoring
        
        # 缓存存储
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.RLock()
        
        # 查询统计
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.metrics_lock = threading.RLock()
        
        # 慢查询记录
        self.slow_queries = deque(maxlen=100)
        
        # 查询模板
        self.optimized_templates = {
            'stock_basic_data': self._get_stock_basic_template(),
            'stock_range_data': self._get_stock_range_template(),
            'stock_batch_data': self._get_stock_batch_template(),
            'stock_indicator_data': self._get_stock_indicator_template()
        }
        
        logger.info(f"查询优化器初始化完成 - 缓存: {enable_cache}, 监控: {enable_monitoring}")
    
    def _get_stock_basic_template(self) -> str:
        """获取股票基础数据查询模板"""
        return """
        SELECT code, name, date, open, high, low, close, volume, turnover_rate
        FROM stock_info WHERE level = %(level)s AND code = %(code)s
        AND level = %(level)s
        AND date >= %(start_date)s AND date <= %(end_date)s
        ORDER BY date ASC
        """
    
    def _get_stock_range_template(self) -> str:
        """获取股票范围数据查询模板"""
        return """
        SELECT code, name, date, open, high, low, close, volume, turnover_rate FROM stock_info WHERE level = %(level)s AND code = %(code)s
        AND level = %(level)s
        AND date >= %(start_date)s AND date <= %(end_date)s
        ORDER BY date ASC
        LIMIT %(limit)s OFFSET %(offset)s
        """
    
    def _get_stock_batch_template(self) -> str:
        """获取批量股票数据查询模板"""
        return """
        SELECT code, name, date, open, high, low, close, volume, turnover_rate
        FROM stock_info WHERE code = %(code)s AND level = %(level)s AND code IN %(codes)s
        AND level = %(level)s
        AND date >= %(start_date)s AND date <= %(end_date)s
        ORDER BY code, date ASC
        """
    
    def _get_stock_indicator_template(self) -> str:
        """获取股票指标数据查询模板"""
        return """
        SELECT code, date, close, volume,
               LAG(close, 1) OVER (PARTITION BY code ORDER BY date) as prev_close,
               AVG(close) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN %(period)s PRECEDING AND CURRENT ROW) as ma
        FROM stock_info WHERE level = %(level)s AND code = %(code)s
        AND level = %(level)s
        AND date >= %(start_date)s AND date <= %(end_date)s
        ORDER BY date ASC
        """
    
    def _generate_cache_key(self, query: str, params: Optional[Dict] = None) -> str:
        """生成缓存键"""
        content = query + str(sorted((params or {}).items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _normalize_query(self, query: str) -> str:
        """标准化查询语句"""
        # 移除多余空白字符
        normalized = ' '.join(query.split())
        # 转换为小写（保留参数占位符）
        return normalized.strip()
    
    def _get_query_hash(self, query: str) -> str:
        """获取查询哈希"""
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    @performance_monitor(threshold_seconds=0.1)
    def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if not self.enable_cache:
            return None
        
        with self.cache_lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                if entry.is_expired():
                    del self.cache[cache_key]
                    return None
                
                entry.access()
                return entry.data
        
        return None
    
    @performance_monitor(threshold_seconds=0.1)
    def put_to_cache(self, cache_key: str, data: Any, ttl: Optional[float] = None):
        """将数据放入缓存"""
        if not self.enable_cache:
            return
        
        with self.cache_lock:
            # 检查缓存大小，如果超过限制则清理
            if len(self.cache) >= self.cache_size:
                self._cleanup_cache()
            
            entry = CacheEntry(
                data=data,
                created_time=time.time(),
                last_accessed=time.time(),
                ttl=ttl or self.default_ttl
            )
            
            self.cache[cache_key] = entry
    
    def _cleanup_cache(self):
        """清理缓存"""
        current_time = time.time()
        
        # 移除过期条目
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        # 如果还是太多，移除最少使用的条目
        if len(self.cache) >= self.cache_size:
            # 按访问次数和最后访问时间排序
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: (x[1].access_count, x[1].last_accessed)
            )
            
            # 移除最少使用的25%
            remove_count = len(sorted_items) // 4
            for key, _ in sorted_items[:remove_count]:
                del self.cache[key]
    
    @exception_handler(severity=ErrorSeverity.MEDIUM, category=ErrorCategory.DATABASE)
    def optimize_query(self, query: str, params: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        优化查询语句
        
        Args:
            query: 原始查询语句
            params: 查询参数
            
        Returns:
            Tuple[str, Dict]: 优化后的查询和参数
        """
        # 标准化查询
        normalized_query = self._normalize_query(query)
        optimized_params = params or {}
        
        # 检查是否有预定义的优化模板
        for template_name, template in self.optimized_templates.items():
            if self._is_query_match_template(normalized_query, template_name):
                return template, optimized_params
        
        # 应用通用优化规则
        optimized_query = self._apply_optimization_rules(normalized_query)
        
        return optimized_query, optimized_params
    
    def _is_query_match_template(self, query: str, template_name: str) -> bool:
        """检查查询是否匹配模板"""
        # 简单的模式匹配逻辑
        if template_name == 'stock_basic_data':
            return 'stock_info' in query and 'code' in query and 'date' in query
        elif template_name == 'stock_batch_data':
            return 'stock_info' in query and 'IN' in query.upper()
        
        return False
    
    def _apply_optimization_rules(self, query: str) -> str:
        """应用查询优化规则"""
        optimized = query
        
        # 规则1: 确保有ORDER BY子句
        if 'ORDER BY' not in optimized.upper():
            if 'stock_info' in optimized and 'date' in optimized:
                optimized += ' ORDER BY date ASC'
        
        # 规则2: 添加LIMIT子句防止大结果集
        if 'LIMIT' not in optimized.upper() and 'COUNT' not in optimized.upper():
            optimized += ' LIMIT 10000'
        
        return optimized
    
    def record_query_execution(self, query: str, execution_time: float, success: bool = True):
        """记录查询执行统计"""
        if not self.enable_monitoring:
            return
        
        query_hash = self._get_query_hash(query)
        
        with self.metrics_lock:
            if query_hash not in self.query_metrics:
                self.query_metrics[query_hash] = QueryMetrics(
                    query_hash=query_hash,
                    query_template=query[:200] + '...' if len(query) > 200 else query
                )
            
            self.query_metrics[query_hash].update_execution(execution_time, success)
            
            # 检查是否为慢查询
            if execution_time > self.slow_query_threshold:
                slow_query_info = {
                    'timestamp': time.time(),
                    'query': query[:500] + '...' if len(query) > 500 else query,
                    'execution_time': execution_time,
                    'query_hash': query_hash
                }
                self.slow_queries.append(slow_query_info)
                logger.warning(f"慢查询检测: {execution_time:.3f}s - {query[:100]}...")
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """获取查询统计信息"""
        with self.metrics_lock:
            total_queries = sum(m.execution_count for m in self.query_metrics.values())
            total_errors = sum(m.error_count for m in self.query_metrics.values())
            
            # 最慢的查询
            slowest_queries = sorted(
                self.query_metrics.values(),
                key=lambda x: x.max_execution_time,
                reverse=True
            )[:10]
            
            # 最频繁的查询
            frequent_queries = sorted(
                self.query_metrics.values(),
                key=lambda x: x.execution_count,
                reverse=True
            )[:10]
            
            return {
                'total_queries': total_queries,
                'total_errors': total_errors,
                'unique_queries': len(self.query_metrics),
                'slow_queries_count': len(self.slow_queries),
                'cache_size': len(self.cache),
                'cache_hit_rate': self._calculate_cache_hit_rate(),
                'slowest_queries': [
                    {
                        'query_hash': q.query_hash,
                        'template': q.query_template,
                        'max_time': q.max_execution_time,
                        'avg_time': q.avg_execution_time,
                        'count': q.execution_count
                    }
                    for q in slowest_queries
                ],
                'frequent_queries': [
                    {
                        'query_hash': q.query_hash,
                        'template': q.query_template,
                        'count': q.execution_count,
                        'avg_time': q.avg_execution_time,
                        'error_rate': q.error_count / q.execution_count if q.execution_count > 0 else 0
                    }
                    for q in frequent_queries
                ]
            }
    
    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        with self.metrics_lock:
            total_hits = sum(m.cache_hits for m in self.query_metrics.values())
            total_misses = sum(m.cache_misses for m in self.query_metrics.values())
            
            if total_hits + total_misses == 0:
                return 0.0
            
            return total_hits / (total_hits + total_misses) * 100
    
    def clear_cache(self):
        """清空缓存"""
        with self.cache_lock:
            self.cache.clear()
        logger.info("查询缓存已清空")
    
    def get_slow_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取慢查询列表"""
        return list(self.slow_queries)[-limit:]


# 全局查询优化器实例
_query_optimizer = None
_optimizer_lock = threading.Lock()


def get_query_optimizer() -> QueryOptimizer:
    """获取全局查询优化器实例"""
    global _query_optimizer
    
    if _query_optimizer is None:
        with _optimizer_lock:
            if _query_optimizer is None:
                _query_optimizer = QueryOptimizer()
    
    return _query_optimizer


# 导出主要类和函数
__all__ = [
    'QueryOptimizer',
    'QueryMetrics',
    'CacheEntry',
    'get_query_optimizer'
]
