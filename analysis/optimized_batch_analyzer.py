#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化的批量分析器

实现并行处理、缓存优化和内存管理的高性能批量分析器
"""

import os
import sys
import time
import multiprocessing
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import gc
import threading
from functools import lru_cache

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer
from analysis.buypoints.period_data_processor import PeriodDataProcessor
from analysis.buypoints.auto_indicator_analyzer import AutoIndicatorAnalyzer
from indicators.indicator_registry import IndicatorRegistry
from indicators.factory import IndicatorFactory

logger = get_logger(__name__)


class OptimizedIndicatorCalculator:
    """优化的指标计算器"""
    
    def __init__(self):
        self.indicator_cache = {}
        self.calculation_cache = {}
        self._lock = threading.Lock()
        
    @lru_cache(maxsize=128)
    def get_indicator_instance(self, indicator_name: str):
        """获取指标实例（带缓存）"""
        try:
            registry = IndicatorRegistry()
            indicator = registry.create_indicator(indicator_name)
            if indicator is None:
                factory = IndicatorFactory()
                indicator = factory.create_indicator(indicator_name)
            return indicator
        except Exception as e:
            logger.warning(f"创建指标 {indicator_name} 失败: {e}")
            return None
    
    def calculate_indicator_batch(self, 
                                indicator_name: str,
                                stock_data_list: List[Dict[str, pd.DataFrame]]) -> List[Dict]:
        """
        批量计算指标
        
        Args:
            indicator_name: 指标名称
            stock_data_list: 股票数据列表
            
        Returns:
            List[Dict]: 计算结果列表
        """
        indicator = self.get_indicator_instance(indicator_name)
        if indicator is None:
            return []
            
        results = []
        for stock_data in stock_data_list:
            try:
                # 对每个周期计算指标
                period_results = {}
                for period, df in stock_data.items():
                    if df is None or df.empty:
                        continue
                        
                    # 使用数据副本避免修改原始数据
                    df_copy = df.copy()
                    indicator_df = indicator.calculate(df_copy)
                    
                    if indicator_df is not None and not indicator_df.empty:
                        # 获取形态信息
                        patterns = indicator.get_patterns(indicator_df)
                        period_results[period] = patterns
                        
                results.append(period_results)
                
            except Exception as e:
                logger.warning(f"计算指标 {indicator_name} 时出错: {e}")
                results.append({})
                
        return results


class ParallelDataProcessor:
    """并行数据处理器"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.data_cache = {}
        
    def load_stock_data_parallel(self, 
                                stock_buypoint_pairs: List[Tuple[str, str]]) -> List[Dict[str, pd.DataFrame]]:
        """
        并行加载股票数据
        
        Args:
            stock_buypoint_pairs: (股票代码, 买点日期) 对列表
            
        Returns:
            List[Dict[str, pd.DataFrame]]: 股票数据列表
        """
        logger.info(f"开始并行加载 {len(stock_buypoint_pairs)} 个股票数据，使用 {self.max_workers} 个进程")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_pair = {
                executor.submit(self._load_single_stock_data, stock_code, buypoint_date): (stock_code, buypoint_date)
                for stock_code, buypoint_date in stock_buypoint_pairs
            }
            
            results = []
            for future in concurrent.futures.as_completed(future_to_pair):
                stock_code, buypoint_date = future_to_pair[future]
                try:
                    stock_data = future.result()
                    results.append(stock_data)
                except Exception as e:
                    logger.error(f"加载股票 {stock_code} 数据失败: {e}")
                    results.append({})
                    
        logger.info(f"并行数据加载完成，成功加载 {len([r for r in results if r])} 个股票")
        return results
    
    @staticmethod
    def _load_single_stock_data(stock_code: str, buypoint_date: str) -> Dict[str, pd.DataFrame]:
        """
        加载单个股票数据（静态方法，用于多进程）
        
        Args:
            stock_code: 股票代码
            buypoint_date: 买点日期
            
        Returns:
            Dict[str, pd.DataFrame]: 股票数据
        """
        try:
            data_processor = PeriodDataProcessor()
            return data_processor.get_multi_period_data(
                stock_code=stock_code,
                end_date=buypoint_date
            )
        except Exception as e:
            logger.error(f"加载股票 {stock_code} 数据时出错: {e}")
            return {}


class OptimizedBatchAnalyzer:
    """优化的批量分析器"""
    
    def __init__(self, max_workers: Optional[int] = None, enable_caching: bool = True):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.enable_caching = enable_caching
        self.data_processor = ParallelDataProcessor(max_workers)
        self.indicator_calculator = OptimizedIndicatorCalculator()
        self.base_analyzer = BuyPointBatchAnalyzer()
        
        logger.info(f"优化批量分析器初始化完成，最大工作进程: {self.max_workers}")
        
    def analyze_batch_buypoints_optimized(self, 
                                        buypoints_df: pd.DataFrame,
                                        chunk_size: int = 4) -> List[Dict[str, Any]]:
        """
        优化的批量买点分析
        
        Args:
            buypoints_df: 买点数据DataFrame
            chunk_size: 分块大小
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        logger.info(f"开始优化批量分析，总买点数: {len(buypoints_df)}, 分块大小: {chunk_size}")
        
        # 准备股票买点对
        stock_buypoint_pairs = [
            (row['stock_code'], row['buypoint_date'])
            for _, row in buypoints_df.iterrows()
        ]
        
        # 并行加载所有股票数据
        start_time = time.time()
        all_stock_data = self.data_processor.load_stock_data_parallel(stock_buypoint_pairs)
        data_load_time = time.time() - start_time
        logger.info(f"数据加载完成，耗时: {data_load_time:.2f}秒")
        
        # 分块处理分析
        results = []
        total_chunks = (len(buypoints_df) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(buypoints_df))
            
            chunk_buypoints = buypoints_df.iloc[start_idx:end_idx]
            chunk_stock_data = all_stock_data[start_idx:end_idx]
            
            logger.info(f"处理分块 {chunk_idx + 1}/{total_chunks}, 买点数: {len(chunk_buypoints)}")
            
            # 分析当前分块
            chunk_results = self._analyze_chunk_parallel(chunk_buypoints, chunk_stock_data)
            results.extend(chunk_results)
            
            # 内存清理
            if chunk_idx % 2 == 0:  # 每处理2个分块清理一次内存
                gc.collect()
                
        logger.info(f"优化批量分析完成，成功分析: {len(results)}/{len(buypoints_df)}")
        return results
    
    def _analyze_chunk_parallel(self, 
                              chunk_buypoints: pd.DataFrame,
                              chunk_stock_data: List[Dict[str, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """
        并行分析分块数据
        
        Args:
            chunk_buypoints: 分块买点数据
            chunk_stock_data: 分块股票数据
            
        Returns:
            List[Dict[str, Any]]: 分析结果
        """
        results = []
        
        # 使用线程池并行分析指标
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 为每个买点提交分析任务
            future_to_index = {}
            
            for idx, (buypoint_row, stock_data) in enumerate(zip(chunk_buypoints.itertuples(), chunk_stock_data)):
                if not stock_data:  # 跳过空数据
                    continue
                    
                future = executor.submit(
                    self._analyze_single_buypoint_optimized,
                    buypoint_row.stock_code,
                    buypoint_row.buypoint_date,
                    stock_data
                )
                future_to_index[future] = idx
            
            # 收集结果
            chunk_results = [None] * len(chunk_buypoints)
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    if result:
                        chunk_results[idx] = result
                except Exception as e:
                    logger.error(f"分析买点时出错: {e}")
            
            # 过滤有效结果
            results = [r for r in chunk_results if r is not None]
            
        return results
    
    def _analyze_single_buypoint_optimized(self, 
                                         stock_code: str,
                                         buypoint_date: str,
                                         stock_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        优化的单个买点分析
        
        Args:
            stock_code: 股票代码
            buypoint_date: 买点日期
            stock_data: 股票数据
            
        Returns:
            Optional[Dict[str, Any]]: 分析结果
        """
        try:
            # 使用基础分析器的指标分析逻辑，但优化数据处理
            indicator_analyzer = AutoIndicatorAnalyzer()
            
            # 定位目标行
            target_rows = {}
            for period, df in stock_data.items():
                if df is not None and not df.empty:
                    target_rows[period] = len(df) - 1
            
            # 分析指标
            indicator_results = indicator_analyzer.analyze_all_indicators(
                stock_data, target_rows
            )
            
            if not indicator_results:
                return None
                
            return {
                'stock_code': stock_code,
                'buypoint_date': buypoint_date,
                'indicator_results': indicator_results
            }
            
        except Exception as e:
            logger.error(f"分析买点 {stock_code} {buypoint_date} 时出错: {e}")
            return None
