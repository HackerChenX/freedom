#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化后的买点分析器

集成向量化计算和智能缓存的高性能买点分析器
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer
from analysis.intelligent_cache_system import IntelligentCacheSystem, CachedIndicatorCalculator
from analysis.vectorized_indicator_optimizer import VectorizedIndicatorOptimizer

logger = get_logger(__name__)


class OptimizedBuyPointAnalyzer(BuyPointBatchAnalyzer):
    """优化后的买点分析器"""
    
    def __init__(self, enable_cache: bool = True, enable_vectorization: bool = True):
        super().__init__()
        
        # 初始化优化组件
        self.enable_cache = enable_cache
        self.enable_vectorization = enable_vectorization
        
        if self.enable_cache:
            self.cache_system = IntelligentCacheSystem(
                max_memory_cache_size=500,
                enable_disk_cache=True,
                cache_dir="data/cache/indicators"
            )
            self.cached_calculator = CachedIndicatorCalculator(self.cache_system)
        
        if self.enable_vectorization:
            self.vectorized_optimizer = VectorizedIndicatorOptimizer()
        
        # 性能统计
        self.performance_stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'vectorized_calculations': 0,
            'total_time_saved': 0.0
        }
    
    def _calculate_indicator_optimized(self, indicator_name: str, stock_data: Dict[str, pd.DataFrame], 
                                     target_rows: Dict[str, int], stock_code: str, 
                                     end_date: str) -> Any:
        """
        优化的指标计算方法
        
        Args:
            indicator_name: 指标名称
            stock_data: 股票数据
            target_rows: 目标行数
            stock_code: 股票代码
            end_date: 结束日期
            
        Returns:
            Any: 指标计算结果
        """
        self.performance_stats['total_calculations'] += 1
        
        # 如果启用缓存，先尝试从缓存获取
        if self.enable_cache:
            cache_params = {
                'target_rows': target_rows,
                'data_hash': self._generate_data_hash(stock_data)
            }
            
            cached_result = self.cache_system.get_cached_indicator(
                stock_code, indicator_name, end_date, cache_params
            )
            
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                logger.debug(f"缓存命中: {indicator_name} for {stock_code}")
                return cached_result
        
        # 缓存未命中，执行计算
        start_time = time.time()
        
        # 检查是否可以使用向量化优化
        if self.enable_vectorization and self._can_vectorize(indicator_name):
            result = self._calculate_vectorized_indicator(indicator_name, stock_data, target_rows)
            self.performance_stats['vectorized_calculations'] += 1
        else:
            # 使用原始方法计算
            result = self._calculate_original_indicator(indicator_name, stock_data, target_rows)
        
        calculation_time = time.time() - start_time
        
        # 如果启用缓存，保存结果
        if self.enable_cache:
            cache_params = {
                'target_rows': target_rows,
                'data_hash': self._generate_data_hash(stock_data)
            }
            self.cache_system.cache_indicator_result(
                stock_code, indicator_name, end_date, result, cache_params
            )
        
        logger.debug(f"计算指标 {indicator_name}: {calculation_time:.4f}s")
        return result
    
    def _generate_data_hash(self, stock_data: Dict[str, pd.DataFrame]) -> str:
        """生成股票数据的哈希值"""
        try:
            # 简化的哈希生成，基于数据的形状和最后几行
            hash_parts = []
            for period, df in stock_data.items():
                if df is not None and not df.empty:
                    hash_parts.append(f"{period}_{len(df)}_{df.iloc[-1]['close']:.2f}")
            return "_".join(hash_parts)
        except:
            return "unknown_hash"
    
    def _can_vectorize(self, indicator_name: str) -> bool:
        """检查指标是否可以向量化"""
        vectorizable_indicators = {
            'MA', 'EMA', 'RSI', 'MACD', 'BOLL', 'KDJ', 'ATR', 'ADX',
            'VOLUME_RATIO', 'OBV', 'BIAS', 'ROC', 'MOMENTUM'
        }
        return indicator_name.upper() in vectorizable_indicators
    
    def _calculate_vectorized_indicator(self, indicator_name: str, 
                                      stock_data: Dict[str, pd.DataFrame], 
                                      target_rows: Dict[str, int]) -> Any:
        """使用向量化方法计算指标"""
        # 使用日线数据进行向量化计算
        if 'daily' not in stock_data or stock_data['daily'] is None:
            return self._calculate_original_indicator(indicator_name, stock_data, target_rows)
        
        df = stock_data['daily']
        
        try:
            if indicator_name.upper() in ['MA', 'EMA']:
                return self.vectorized_optimizer.optimize_moving_average_calculations(df, [5, 10, 20, 60])
            elif indicator_name.upper() == 'RSI':
                return self.vectorized_optimizer.optimize_rsi_calculation(df)
            elif indicator_name.upper() == 'MACD':
                return self.vectorized_optimizer.optimize_macd_calculation(df)
            elif indicator_name.upper() == 'BOLL':
                return self.vectorized_optimizer.optimize_bollinger_bands_calculation(df)
            elif indicator_name.upper() == 'KDJ':
                return self.vectorized_optimizer.optimize_kdj_calculation(df)
            elif indicator_name.upper() in ['OBV', 'VOLUME_RATIO']:
                return self.vectorized_optimizer.optimize_volume_indicators(df)
            elif indicator_name.upper() in ['ATR', 'ADX']:
                return self.vectorized_optimizer.optimize_trend_indicators(df)
            else:
                # 回退到原始方法
                return self._calculate_original_indicator(indicator_name, stock_data, target_rows)
        except Exception as e:
            logger.warning(f"向量化计算失败 {indicator_name}: {e}")
            return self._calculate_original_indicator(indicator_name, stock_data, target_rows)
    
    def _calculate_original_indicator(self, indicator_name: str, 
                                    stock_data: Dict[str, pd.DataFrame], 
                                    target_rows: Dict[str, int]) -> Any:
        """使用原始方法计算指标"""
        try:
            # 创建指标实例
            indicator = self.indicator_analyzer.indicator_registry.create_indicator(indicator_name)
            if indicator is None:
                indicator = self.indicator_analyzer.indicator_factory.create_indicator(indicator_name)
            
            if indicator is None:
                logger.warning(f"无法创建指标: {indicator_name}")
                return None
            
            # 计算指标（使用日线数据）
            if 'daily' in stock_data and stock_data['daily'] is not None:
                df_copy = stock_data['daily'].copy()
                return indicator.calculate(df_copy)
            
            return None
        except Exception as e:
            logger.error(f"原始指标计算失败 {indicator_name}: {e}")
            return None
    
    def analyze_single_buypoint_optimized(self, stock_code: str, buypoint_date: str) -> Optional[Dict[str, Any]]:
        """
        优化的单个买点分析
        
        Args:
            stock_code: 股票代码
            buypoint_date: 买点日期
            
        Returns:
            Optional[Dict[str, Any]]: 分析结果
        """
        start_time = time.time()
        
        try:
            # 获取股票数据
            stock_data = self.data_processor.get_multi_period_data(
                stock_code=stock_code,
                end_date=buypoint_date
            )
            
            if not stock_data:
                logger.warning(f"无法获取股票数据: {stock_code} {buypoint_date}")
                return None
            
            # 计算目标行数
            target_rows = {period: len(df) - 1 for period, df in stock_data.items() 
                          if df is not None and not df.empty}
            
            # 分析所有指标（使用优化方法）
            indicator_results = {}
            for indicator_name in self.indicator_analyzer.all_indicators:
                try:
                    result = self._calculate_indicator_optimized(
                        indicator_name, stock_data, target_rows, stock_code, buypoint_date
                    )
                    if result is not None:
                        indicator_results[indicator_name] = result
                except Exception as e:
                    logger.warning(f"指标计算失败 {indicator_name}: {e}")
            
            analysis_time = time.time() - start_time
            
            return {
                'stock_code': stock_code,
                'buypoint_date': buypoint_date,
                'analysis_time': analysis_time,
                'indicator_count': len(indicator_results),
                'indicator_results': indicator_results,
                'optimization_stats': self.performance_stats.copy()
            }
            
        except Exception as e:
            logger.error(f"买点分析失败 {stock_code} {buypoint_date}: {e}")
            return None
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        stats = self.performance_stats.copy()
        
        if self.enable_cache:
            cache_stats = self.cache_system.get_cache_stats()
            stats.update(cache_stats)
        
        # 计算优化效率
        if stats['total_calculations'] > 0:
            stats['cache_hit_rate'] = (stats['cache_hits'] / stats['total_calculations']) * 100
            stats['vectorization_rate'] = (stats['vectorized_calculations'] / stats['total_calculations']) * 100
        
        return stats


def performance_comparison_test(sample_size: int = 3):
    """性能对比测试"""
    print("="*60)
    print("优化后买点分析器性能对比测试")
    print("="*60)
    
    # 加载测试数据
    base_analyzer = BuyPointBatchAnalyzer()
    buypoints_df = base_analyzer.load_buypoints_from_csv("data/buypoints.csv")
    
    if buypoints_df.empty:
        print("❌ 无法加载买点数据")
        return
    
    test_df = buypoints_df.head(sample_size)
    
    # 测试原始分析器
    print(f"1. 测试原始分析器（{sample_size}个买点）...")
    start_time = time.time()
    original_results = []
    for _, row in test_df.iterrows():
        result = base_analyzer.analyze_single_buypoint(row['stock_code'], row['buypoint_date'])
        if result:
            original_results.append(result)
    original_time = time.time() - start_time
    
    # 测试优化分析器
    print(f"2. 测试优化分析器（{sample_size}个买点）...")
    optimized_analyzer = OptimizedBuyPointAnalyzer(enable_cache=True, enable_vectorization=True)
    start_time = time.time()
    optimized_results = []
    for _, row in test_df.iterrows():
        result = optimized_analyzer.analyze_single_buypoint_optimized(row['stock_code'], row['buypoint_date'])
        if result:
            optimized_results.append(result)
    optimized_time = time.time() - start_time
    
    # 第二轮测试（测试缓存效果）
    print(f"3. 第二轮优化测试（测试缓存效果）...")
    start_time = time.time()
    cached_results = []
    for _, row in test_df.iterrows():
        result = optimized_analyzer.analyze_single_buypoint_optimized(row['stock_code'], row['buypoint_date'])
        if result:
            cached_results.append(result)
    cached_time = time.time() - start_time
    
    # 显示结果
    print(f"\n性能对比结果:")
    print(f"原始分析器时间: {original_time:.4f}s")
    print(f"优化分析器时间: {optimized_time:.4f}s")
    print(f"缓存测试时间: {cached_time:.4f}s")
    
    if original_time > 0:
        improvement1 = (original_time - optimized_time) / original_time * 100
        improvement2 = (original_time - cached_time) / original_time * 100
        print(f"首次优化提升: {improvement1:.1f}%")
        print(f"缓存优化提升: {improvement2:.1f}%")
    
    # 优化统计
    opt_stats = optimized_analyzer.get_optimization_stats()
    print(f"\n优化统计:")
    print(f"总计算次数: {opt_stats['total_calculations']}")
    print(f"缓存命中次数: {opt_stats['cache_hits']}")
    print(f"向量化计算次数: {opt_stats['vectorized_calculations']}")
    if 'cache_hit_rate' in opt_stats:
        print(f"缓存命中率: {opt_stats['cache_hit_rate']:.1f}%")
    if 'vectorization_rate' in opt_stats:
        print(f"向量化使用率: {opt_stats['vectorization_rate']:.1f}%")
    
    # 保存结果
    output_dir = "data/result/optimized_performance"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_info': {
            'sample_size': sample_size,
            'original_results': len(original_results),
            'optimized_results': len(optimized_results)
        },
        'performance': {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'cached_time': cached_time,
            'improvement_percentage': improvement1 if original_time > 0 else 0,
            'cache_improvement_percentage': improvement2 if original_time > 0 else 0
        },
        'optimization_stats': opt_stats
    }
    
    results_path = os.path.join(output_dir, 'optimized_performance_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存到: {results_path}")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    performance_comparison_test(sample_size)
