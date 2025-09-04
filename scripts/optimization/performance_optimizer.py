#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能优化器

专门用于优化股票选股系统的性能，确保4000+只个股的测试能在5分钟内完成。

核心优化策略：
1. 数据库查询优化
2. 并行处理优化
3. 内存使用优化
4. 缓存机制优化
5. 早停机制实现

Author: AI Assistant
Date: 2025-07-19
"""

import os
import sys
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from db.unified_data_manager import get_unified_data_manager

logger = get_logger(__name__)


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化性能优化器
        
        Args:
            config: 优化配置
        """
        self.config = config or self._get_default_config()
        self.data_manager = get_unified_data_manager()
        
        # 性能监控
        self.performance_stats = {
            'query_times': [],
            'processing_times': [],
            'memory_usage': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 缓存系统
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        logger.info("⚡ 性能优化器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'database': {
                'connection_pool_size': 20,
                'query_timeout': 30,
                'batch_size': 100,
                'parallel_queries': True
            },
            'processing': {
                'max_workers': 8,
                'chunk_size': 50,
                'use_multiprocessing': False,
                'memory_limit_mb': 2048
            },
            'cache': {
                'enabled': True,
                'max_size': 1000,
                'ttl_seconds': 3600
            },
            'optimization': {
                'early_stop_enabled': True,
                'vectorization_enabled': True,
                'lazy_loading': True,
                'data_compression': True
            }
        }
    
    @performance_monitor(threshold=1.0)
    def optimize_data_loading(self, stock_codes: List[str], 
                            date_range: tuple) -> Dict[str, pd.DataFrame]:
        """
        优化数据加载
        
        Args:
            stock_codes: 股票代码列表
            date_range: 日期范围 (start_date, end_date)
            
        Returns:
            Dict[str, pd.DataFrame]: 股票数据字典
        """
        try:
            logger.info(f"⚡ 开始优化数据加载: {len(stock_codes)}只股票")
            start_time = time.time()
            
            # 检查缓存
            cache_key = self._generate_cache_key(stock_codes, date_range)
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info("✅ 使用缓存数据")
                return cached_data
            
            # 批量并行加载
            if self.config['database']['parallel_queries']:
                data = self._parallel_load_data(stock_codes, date_range)
            else:
                data = self._sequential_load_data(stock_codes, date_range)
            
            # 缓存结果
            if self.config['cache']['enabled']:
                self._save_to_cache(cache_key, data)
            
            loading_time = time.time() - start_time
            self.performance_stats['query_times'].append(loading_time)
            
            logger.info(f"✅ 数据加载完成: {len(data)}只股票, 耗时{loading_time:.2f}秒")
            return data
            
        except Exception as e:
            logger.error(f"❌ 数据加载优化失败: {e}")
            raise
    
    def _parallel_load_data(self, stock_codes: List[str], 
                          date_range: tuple) -> Dict[str, pd.DataFrame]:
        """并行加载数据"""
        try:
            data = {}
            batch_size = self.config['database']['batch_size']
            max_workers = self.config['processing']['max_workers']
            
            # 分批处理
            batches = [stock_codes[i:i + batch_size] 
                      for i in range(0, len(stock_codes), batch_size)]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for batch in batches:
                    future = executor.submit(
                        self._load_batch_data, batch, date_range
                    )
                    futures.append(future)
                
                # 收集结果
                for future in futures:
                    batch_data = future.result()
                    data.update(batch_data)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ 并行数据加载失败: {e}")
            raise
    
    def _load_batch_data(self, stock_codes: List[str], 
                        date_range: tuple) -> Dict[str, pd.DataFrame]:
        """加载批量数据"""
        try:
            batch_data = {}
            start_date, end_date = date_range
            
            for code in stock_codes:
                try:
                    data = self.data_manager.get_stock_data(
                        code=code,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if not data.empty:
                        # 数据压缩优化
                        if self.config['optimization']['data_compression']:
                            data = self._compress_dataframe(data)
                        batch_data[code] = data
                        
                except Exception as e:
                    logger.warning(f"⚠️ 加载股票 {code} 数据失败: {e}")
                    continue
            
            return batch_data
            
        except Exception as e:
            logger.error(f"❌ 批量数据加载失败: {e}")
            return {}
    
    def _sequential_load_data(self, stock_codes: List[str], 
                            date_range: tuple) -> Dict[str, pd.DataFrame]:
        """顺序加载数据"""
        try:
            data = {}
            start_date, end_date = date_range
            
            for i, code in enumerate(stock_codes):
                try:
                    stock_data = self.data_manager.get_stock_data(
                        code=code,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if not stock_data.empty:
                        if self.config['optimization']['data_compression']:
                            stock_data = self._compress_dataframe(stock_data)
                        data[code] = stock_data
                    
                    # 进度报告
                    if (i + 1) % 100 == 0:
                        logger.info(f"📊 数据加载进度: {i + 1}/{len(stock_codes)}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ 加载股票 {code} 数据失败: {e}")
                    continue
            
            return data
            
        except Exception as e:
            logger.error(f"❌ 顺序数据加载失败: {e}")
            return {}
    
    def _compress_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """压缩DataFrame以节省内存"""
        try:
            # 数值列压缩
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
                elif df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
            
            # 字符串列压缩
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype('category')
            
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ DataFrame压缩失败: {e}")
            return df
    
    @performance_monitor(threshold=2.0)
    def optimize_strategy_execution(self, strategy_func: Callable,
                                  stock_data: Dict[str, pd.DataFrame],
                                  strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化策略执行
        
        Args:
            strategy_func: 策略执行函数
            stock_data: 股票数据
            strategy_params: 策略参数
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        try:
            logger.info("⚡ 开始优化策略执行")
            start_time = time.time()
            
            # 检查早停条件
            if self.config['optimization']['early_stop_enabled']:
                early_stop_checker = self._create_early_stop_checker()
            else:
                early_stop_checker = None
            
            # 选择执行方式
            if self.config['processing']['use_multiprocessing']:
                results = self._multiprocess_execution(
                    strategy_func, stock_data, strategy_params, early_stop_checker
                )
            else:
                results = self._threaded_execution(
                    strategy_func, stock_data, strategy_params, early_stop_checker
                )
            
            execution_time = time.time() - start_time
            self.performance_stats['processing_times'].append(execution_time)
            
            logger.info(f"✅ 策略执行优化完成，耗时{execution_time:.2f}秒")
            return results
            
        except Exception as e:
            logger.error(f"❌ 策略执行优化失败: {e}")
            raise
    
    def _threaded_execution(self, strategy_func: Callable,
                          stock_data: Dict[str, pd.DataFrame],
                          strategy_params: Dict[str, Any],
                          early_stop_checker: Optional[Callable] = None) -> Dict[str, Any]:
        """多线程执行"""
        try:
            results = {}
            max_workers = self.config['processing']['max_workers']
            chunk_size = self.config['processing']['chunk_size']
            
            # 分块处理
            stock_items = list(stock_data.items())
            chunks = [stock_items[i:i + chunk_size] 
                     for i in range(0, len(stock_items), chunk_size)]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for chunk in chunks:
                    future = executor.submit(
                        self._process_chunk, strategy_func, chunk, 
                        strategy_params, early_stop_checker
                    )
                    futures.append(future)
                
                # 收集结果
                for future in futures:
                    chunk_results = future.result()
                    results.update(chunk_results)
                    
                    # 检查早停
                    if early_stop_checker and early_stop_checker():
                        logger.warning("⏰ 触发早停机制")
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 多线程执行失败: {e}")
            return {}
    
    def _process_chunk(self, strategy_func: Callable,
                      chunk: List[tuple],
                      strategy_params: Dict[str, Any],
                      early_stop_checker: Optional[Callable] = None) -> Dict[str, Any]:
        """处理数据块"""
        try:
            chunk_results = {}
            
            for stock_code, stock_data in chunk:
                try:
                    # 检查早停
                    if early_stop_checker and early_stop_checker():
                        break
                    
                    # 执行策略
                    result = strategy_func(stock_code, stock_data, strategy_params)
                    chunk_results[stock_code] = result
                    
                except Exception as e:
                    logger.warning(f"⚠️ 处理股票 {stock_code} 失败: {e}")
                    continue
            
            return chunk_results
            
        except Exception as e:
            logger.error(f"❌ 数据块处理失败: {e}")
            return {}
    
    def _create_early_stop_checker(self) -> Callable:
        """创建早停检查器"""
        start_time = time.time()
        max_time = 240  # 4分钟早停
        
        def check_early_stop():
            return (time.time() - start_time) > max_time
        
        return check_early_stop
    
    def _generate_cache_key(self, stock_codes: List[str], 
                          date_range: tuple) -> str:
        """生成缓存键"""
        codes_hash = hash(tuple(sorted(stock_codes)))
        date_hash = hash(date_range)
        return f"data_{codes_hash}_{date_hash}"
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if not self.config['cache']['enabled']:
            return None
        
        with self._cache_lock:
            if key in self._cache:
                data, timestamp = self._cache[key]
                
                # 检查TTL
                if time.time() - timestamp < self.config['cache']['ttl_seconds']:
                    self.performance_stats['cache_hits'] += 1
                    return data
                else:
                    del self._cache[key]
        
        self.performance_stats['cache_misses'] += 1
        return None
    
    def _save_to_cache(self, key: str, data: Any):
        """保存数据到缓存"""
        if not self.config['cache']['enabled']:
            return
        
        with self._cache_lock:
            # 检查缓存大小
            if len(self._cache) >= self.config['cache']['max_size']:
                # 删除最旧的条目
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            
            self._cache[key] = (data, time.time())
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        try:
            stats = self.performance_stats
            
            return {
                'query_performance': {
                    'total_queries': len(stats['query_times']),
                    'avg_query_time': np.mean(stats['query_times']) if stats['query_times'] else 0,
                    'max_query_time': max(stats['query_times']) if stats['query_times'] else 0,
                    'min_query_time': min(stats['query_times']) if stats['query_times'] else 0
                },
                'processing_performance': {
                    'total_processes': len(stats['processing_times']),
                    'avg_processing_time': np.mean(stats['processing_times']) if stats['processing_times'] else 0,
                    'max_processing_time': max(stats['processing_times']) if stats['processing_times'] else 0
                },
                'cache_performance': {
                    'cache_hits': stats['cache_hits'],
                    'cache_misses': stats['cache_misses'],
                    'hit_rate': stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100 
                              if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
                },
                'optimization_status': {
                    'parallel_queries_enabled': self.config['database']['parallel_queries'],
                    'vectorization_enabled': self.config['optimization']['vectorization_enabled'],
                    'cache_enabled': self.config['cache']['enabled'],
                    'early_stop_enabled': self.config['optimization']['early_stop_enabled']
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 生成性能报告失败: {e}")
            return {}
