#!/usr/bin/env python3
"""
高性能策略执行器

专为生产级大规模选股设计的高性能执行引擎，支持：
1. 批量数据库查询
2. 多进程并行计算
3. 智能缓存策略
4. 向量化指标计算

作者：AI Assistant
创建时间：2025-01-13
"""

import sys
import time
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from functools import lru_cache

# 系统导入
from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from db.unified_data_manager import get_unified_data_manager
from strategy.strategy_parser import StrategyParser
from indicators.zxm.buy_point_indicators import ZXMBSAbsorb

logger = get_logger(__name__)

class HighPerformanceExecutor:
    """高性能策略执行器"""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 batch_size: int = 100,
                 enable_cache: bool = True,
                 cache_ttl: int = 3600):
        """
        初始化高性能执行器
        
        Args:
            max_workers: 最大工作进程数，默认为CPU核心数
            batch_size: 批处理大小
            enable_cache: 是否启用缓存
            cache_ttl: 缓存TTL（秒）
        """
        self.max_workers = max_workers or min(cpu_count(), 8)  # 限制最大进程数
        self.batch_size = batch_size
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        
        # 数据管理器
        self.data_manager = get_unified_data_manager()
        
        # 性能统计
        self.stats = {
            'total_stocks': 0,
            'processed_stocks': 0,
            'selected_stocks': 0,
            'batch_queries': 0,
            'cache_hits': 0,
            'processing_time': 0.0,
            'start_time': None
        }
        
        # 缓存
        self._cache = {} if enable_cache else None
        self._cache_lock = threading.Lock()
        
        logger.info(f"高性能执行器初始化: workers={self.max_workers}, batch_size={batch_size}")
    
    @performance_monitor(threshold=10.0)
    def execute_strategy(self, 
                        strategy_config: Dict[str, Any], 
                        stock_codes: List[str],
                        target_date: str,
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        执行策略选股（高性能版本）
        
        Args:
            strategy_config: 策略配置
            stock_codes: 股票代码列表
            target_date: 目标日期
            limit: 结果限制数量
            
        Returns:
            List[Dict[str, Any]]: 选股结果
        """
        self.stats['start_time'] = time.time()
        self.stats['total_stocks'] = len(stock_codes)
        
        logger.info(f"🚀 开始高性能选股执行: {len(stock_codes)}只股票, 目标日期: {target_date}")
        
        try:
            # 1. 批量获取股票数据
            all_stock_data = self._batch_load_stock_data(stock_codes, target_date)
            logger.info(f"📊 批量加载完成: {len(all_stock_data)}只股票数据")
            
            # 2. 并行计算指标
            calculation_results = self._parallel_calculate_indicators(
                all_stock_data, strategy_config, target_date
            )
            logger.info(f"🧮 并行计算完成: {len(calculation_results)}个结果")
            
            # 3. 过滤和排序
            selected_stocks = self._filter_and_rank_results(
                calculation_results, strategy_config, limit
            )
            
            self.stats['selected_stocks'] = len(selected_stocks)
            self.stats['processing_time'] = time.time() - self.stats['start_time']
            
            logger.info(f"✅ 高性能选股完成: 选出{len(selected_stocks)}只股票, "
                       f"耗时{self.stats['processing_time']:.2f}秒")
            
            return selected_stocks
            
        except Exception as e:
            logger.error(f"❌ 高性能选股执行失败: {e}")
            raise
    
    def _batch_load_stock_data(self, stock_codes: List[str], target_date: str) -> Dict[str, pd.DataFrame]:
        """
        批量加载股票数据
        
        Args:
            stock_codes: 股票代码列表
            target_date: 目标日期
            
        Returns:
            Dict[str, pd.DataFrame]: 股票代码 -> 数据映射
        """
        all_data = {}
        
        # 分批处理以避免单次查询过大
        for i in range(0, len(stock_codes), self.batch_size):
            batch_codes = stock_codes[i:i + self.batch_size]
            
            try:
                # 批量查询60分钟数据
                batch_data = self._batch_query_period_data(
                    batch_codes, target_date, "60min"
                )
                all_data.update(batch_data)
                
                self.stats['batch_queries'] += 1
                logger.debug(f"批次 {i//self.batch_size + 1}: 加载了{len(batch_data)}只股票数据")
                
            except Exception as e:
                logger.warning(f"批次 {i//self.batch_size + 1} 加载失败: {e}")
                continue
        
        return all_data
    
    def _batch_query_period_data(self, 
                                stock_codes: List[str], 
                                target_date: str, 
                                period: str) -> Dict[str, pd.DataFrame]:
        """
        批量查询指定周期数据
        
        Args:
            stock_codes: 股票代码列表
            target_date: 目标日期
            period: 时间周期
            
        Returns:
            Dict[str, pd.DataFrame]: 股票数据映射
        """
        result = {}
        
        # 计算查询日期范围（需要足够的历史数据计算指标）
        from datetime import datetime, timedelta
        end_date = datetime.strptime(target_date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=120)  # 4个月历史数据
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = target_date
        
        # 使用线程池并行查询每只股票
        with ThreadPoolExecutor(max_workers=min(10, len(stock_codes))) as executor:
            future_to_code = {
                executor.submit(
                    self._query_single_stock_data, 
                    code, start_date_str, end_date_str, period
                ): code 
                for code in stock_codes
            }
            
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    data = future.result(timeout=30)  # 30秒超时
                    if not data.empty:
                        result[code] = data
                except Exception as e:
                    logger.debug(f"股票{code}数据查询失败: {e}")
                    continue
        
        return result
    
    def _query_single_stock_data(self, 
                                code: str, 
                                start_date: str, 
                                end_date: str, 
                                period: str) -> pd.DataFrame:
        """
        查询单只股票数据（带缓存）
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 时间周期
            
        Returns:
            pd.DataFrame: 股票数据
        """
        # 检查缓存
        cache_key = f"{code}_{start_date}_{end_date}_{period}"
        if self.enable_cache and cache_key in self._cache:
            self.stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        try:
            # 查询数据
            data = self.data_manager.get_period_data(
                stock_code=code,
                start_date=start_date,
                end_date=end_date,
                period=period
            )
            
            # 写入缓存
            if self.enable_cache and not data.empty:
                with self._cache_lock:
                    self._cache[cache_key] = data
            
            return data
            
        except Exception as e:
            logger.debug(f"查询股票{code}数据失败: {e}")
            return pd.DataFrame()
    
    def _parallel_calculate_indicators(self, 
                                     stock_data: Dict[str, pd.DataFrame],
                                     strategy_config: Dict[str, Any],
                                     target_date: str) -> List[Dict[str, Any]]:
        """
        并行计算指标
        
        Args:
            stock_data: 股票数据映射
            strategy_config: 策略配置
            target_date: 目标日期
            
        Returns:
            List[Dict[str, Any]]: 计算结果列表
        """
        results = []
        stock_items = list(stock_data.items())
        
        # 分批并行处理
        batch_size = max(1, len(stock_items) // self.max_workers)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i in range(0, len(stock_items), batch_size):
                batch = stock_items[i:i + batch_size]
                future = executor.submit(
                    _calculate_batch_indicators,
                    batch, strategy_config, target_date
                )
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    batch_results = future.result(timeout=120)  # 2分钟超时
                    results.extend(batch_results)
                    self.stats['processed_stocks'] += len(batch_results)
                except Exception as e:
                    logger.warning(f"批次指标计算失败: {e}")
                    continue
        
        return results
    
    def _filter_and_rank_results(self, 
                                results: List[Dict[str, Any]],
                                strategy_config: Dict[str, Any],
                                limit: Optional[int]) -> List[Dict[str, Any]]:
        """
        过滤和排序结果
        
        Args:
            results: 计算结果列表
            strategy_config: 策略配置
            limit: 结果限制数量
            
        Returns:
            List[Dict[str, Any]]: 过滤排序后的结果
        """
        # 过滤满足条件的股票
        filtered_results = [
            result for result in results 
            if result.get('buy_signal', False)
        ]
        
        # 按ZXM强度排序
        filtered_results.sort(
            key=lambda x: x.get('xg_value', 0), 
            reverse=True
        )
        
        # 限制结果数量
        if limit and len(filtered_results) > limit:
            filtered_results = filtered_results[:limit]
        
        return filtered_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if self.stats['start_time']:
            self.stats['processing_time'] = time.time() - self.stats['start_time']
        
        return {
            **self.stats,
            'stocks_per_second': (
                self.stats['processed_stocks'] / max(self.stats['processing_time'], 0.1)
            ),
            'cache_hit_rate': (
                self.stats['cache_hits'] / max(self.stats['processed_stocks'], 1) * 100
            ),
            'selection_rate': (
                self.stats['selected_stocks'] / max(self.stats['total_stocks'], 1) * 100
            )
        }


def _calculate_batch_indicators(batch_data: List[Tuple[str, pd.DataFrame]], 
                               strategy_config: Dict[str, Any],
                               target_date: str) -> List[Dict[str, Any]]:
    """
    批量计算指标（在子进程中执行）
    
    Args:
        batch_data: 批量股票数据
        strategy_config: 策略配置
        target_date: 目标日期
        
    Returns:
        List[Dict[str, Any]]: 计算结果
    """
    results = []
    indicator = ZXMBSAbsorb()
    
    for code, data in batch_data:
        try:
            if data.empty or len(data) < 70:  # 需要足够的历史数据
                continue
            
            # 计算ZXM指标
            result = indicator._calculate(data)
            
            # 获取目标日期的结果
            target_row = result[result['date'] == target_date]
            if target_row.empty:
                # 使用最新日期
                target_row = result.iloc[[-1]]
            
            if not target_row.empty:
                row = target_row.iloc[0]
                results.append({
                    'stock_code': code,
                    'date': target_date,
                    'buy_signal': bool(row['buy_signal']),
                    'xg_value': int(row['XG']),
                    'v11_ema': float(row['EMA_V11_3']),
                    'v12': float(row['V12']),
                    'aa_signal': bool(row['AA']),
                    'bb_signal': bool(row['BB'])
                })
                
        except Exception as e:
            # 静默处理错误，避免打印大量错误信息
            continue
    
    return results


# 全局高性能执行器实例
_executor_instance = None
_executor_lock = threading.Lock()

def get_high_performance_executor(**kwargs) -> HighPerformanceExecutor:
    """获取高性能执行器单例"""
    global _executor_instance
    
    if _executor_instance is None:
        with _executor_lock:
            if _executor_instance is None:
                _executor_instance = HighPerformanceExecutor(**kwargs)
    
    return _executor_instance 