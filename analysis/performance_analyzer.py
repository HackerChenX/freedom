#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票分析系统性能分析器

分析系统性能瓶颈，提供优化建议
"""

import os
import sys
import time
import psutil
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import cProfile
import pstats
from io import StringIO

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


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.start_time = None
        self.end_time = None
        self.memory_usage = []
        self.cpu_usage = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_profiling(self):
        """开始性能分析"""
        self.start_time = time.time()
        self.profiler.enable()
        self._start_monitoring()
        
    def stop_profiling(self):
        """停止性能分析"""
        self.end_time = time.time()
        self.profiler.disable()
        self._stop_monitoring()
        
    def _start_monitoring(self):
        """开始监控系统资源"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def _stop_monitoring(self):
        """停止监控系统资源"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            
    def _monitor_resources(self):
        """监控系统资源使用情况"""
        process = psutil.Process()
        while self.monitoring:
            try:
                # 获取内存使用情况
                memory_info = process.memory_info()
                self.memory_usage.append({
                    'timestamp': time.time(),
                    'rss': memory_info.rss / 1024 / 1024,  # MB
                    'vms': memory_info.vms / 1024 / 1024   # MB
                })
                
                # 获取CPU使用情况
                cpu_percent = process.cpu_percent()
                self.cpu_usage.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent
                })
                
                time.sleep(0.5)  # 每0.5秒采样一次
            except Exception as e:
                logger.warning(f"监控资源使用时出错: {e}")
                break
                
    def get_profile_stats(self) -> Dict[str, Any]:
        """获取性能分析统计信息"""
        if not self.start_time or not self.end_time:
            return {}
            
        # 获取函数调用统计
        s = StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(50)  # 显示前50个最耗时的函数
        
        # 计算总执行时间
        total_time = self.end_time - self.start_time
        
        # 计算平均内存使用
        avg_memory = 0
        max_memory = 0
        if self.memory_usage:
            avg_memory = sum(m['rss'] for m in self.memory_usage) / len(self.memory_usage)
            max_memory = max(m['rss'] for m in self.memory_usage)
            
        # 计算平均CPU使用
        avg_cpu = 0
        max_cpu = 0
        if self.cpu_usage:
            avg_cpu = sum(c['cpu_percent'] for c in self.cpu_usage) / len(self.cpu_usage)
            max_cpu = max(c['cpu_percent'] for c in self.cpu_usage)
            
        return {
            'total_time': total_time,
            'avg_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'avg_cpu_percent': avg_cpu,
            'max_cpu_percent': max_cpu,
            'profile_output': s.getvalue(),
            'memory_timeline': self.memory_usage,
            'cpu_timeline': self.cpu_usage
        }


class PerformanceAnalyzer:
    """股票分析系统性能分析器"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.results = {}
        
    def analyze_batch_processing_performance(self, 
                                           buypoints_csv: str,
                                           sample_size: int = 5) -> Dict[str, Any]:
        """
        分析批量处理性能
        
        Args:
            buypoints_csv: 买点数据CSV文件路径
            sample_size: 采样大小，用于性能测试
            
        Returns:
            Dict[str, Any]: 性能分析结果
        """
        logger.info(f"开始分析批量处理性能，采样大小: {sample_size}")
        
        # 加载买点数据
        analyzer = BuyPointBatchAnalyzer()
        buypoints_df = analyzer.load_buypoints_from_csv(buypoints_csv)
        
        if buypoints_df.empty:
            logger.error("无法加载买点数据")
            return {}
            
        # 限制采样大小
        if len(buypoints_df) > sample_size:
            buypoints_df = buypoints_df.head(sample_size)
            
        logger.info(f"实际分析 {len(buypoints_df)} 个买点")
        
        # 开始性能分析
        self.profiler.start_profiling()
        
        # 执行批量分析
        start_time = time.time()
        results = analyzer.analyze_batch_buypoints(buypoints_df)
        end_time = time.time()
        
        # 停止性能分析
        self.profiler.stop_profiling()
        
        # 获取性能统计
        perf_stats = self.profiler.get_profile_stats()
        
        # 计算性能指标
        total_time = end_time - start_time
        avg_time_per_stock = total_time / len(buypoints_df) if len(buypoints_df) > 0 else 0
        
        analysis_result = {
            'sample_size': len(buypoints_df),
            'total_time': total_time,
            'avg_time_per_stock': avg_time_per_stock,
            'successful_analyses': len(results),
            'success_rate': len(results) / len(buypoints_df) if len(buypoints_df) > 0 else 0,
            'performance_stats': perf_stats
        }
        
        logger.info(f"批量处理性能分析完成: 总时间 {total_time:.2f}s, 平均每股 {avg_time_per_stock:.2f}s")

        return analysis_result

    def analyze_indicator_calculation_performance(self,
                                                stock_data: Dict[str, pd.DataFrame],
                                                target_rows: Dict[str, int]) -> Dict[str, Any]:
        """
        分析指标计算性能

        Args:
            stock_data: 股票数据
            target_rows: 目标行索引

        Returns:
            Dict[str, Any]: 指标计算性能分析结果
        """
        logger.info("开始分析指标计算性能")

        # 创建指标分析器
        indicator_analyzer = AutoIndicatorAnalyzer()

        # 获取所有指标
        all_indicators = indicator_analyzer.all_indicators
        logger.info(f"将分析 {len(all_indicators)} 个指标的性能")

        # 分析每个指标的性能
        indicator_performance = {}

        for indicator_name in all_indicators:
            try:
                # 开始计时
                start_time = time.time()

                # 创建指标实例
                indicator = indicator_analyzer.indicator_registry.create_indicator(indicator_name)
                if indicator is None:
                    indicator = indicator_analyzer.indicator_factory.create_indicator(indicator_name)

                if indicator is None:
                    continue

                # 对每个周期的数据计算指标
                period_times = {}
                total_calculation_time = 0

                for period, df in stock_data.items():
                    if df is None or df.empty:
                        continue

                    period_start = time.time()

                    # 计算指标
                    df_copy = df.copy()
                    indicator_df = indicator.calculate(df_copy)

                    period_end = time.time()
                    period_time = period_end - period_start
                    period_times[period] = period_time
                    total_calculation_time += period_time

                end_time = time.time()
                total_time = end_time - start_time

                indicator_performance[indicator_name] = {
                    'total_time': total_time,
                    'calculation_time': total_calculation_time,
                    'overhead_time': total_time - total_calculation_time,
                    'period_times': period_times,
                    'avg_period_time': total_calculation_time / len(period_times) if period_times else 0
                }

            except Exception as e:
                logger.warning(f"分析指标 {indicator_name} 性能时出错: {e}")
                indicator_performance[indicator_name] = {
                    'error': str(e),
                    'total_time': 0
                }

        # 按耗时排序
        sorted_indicators = sorted(
            indicator_performance.items(),
            key=lambda x: x[1].get('total_time', 0),
            reverse=True
        )

        # 计算统计信息
        total_indicators = len(indicator_performance)
        successful_indicators = len([p for p in indicator_performance.values() if 'error' not in p])
        total_time = sum(p.get('total_time', 0) for p in indicator_performance.values())
        avg_time_per_indicator = total_time / total_indicators if total_indicators > 0 else 0

        result = {
            'total_indicators': total_indicators,
            'successful_indicators': successful_indicators,
            'success_rate': successful_indicators / total_indicators if total_indicators > 0 else 0,
            'total_time': total_time,
            'avg_time_per_indicator': avg_time_per_indicator,
            'indicator_performance': dict(sorted_indicators),
            'top_10_slowest': dict(sorted_indicators[:10]),
            'performance_summary': self._generate_performance_summary(dict(sorted_indicators))
        }

        logger.info(f"指标计算性能分析完成: 总时间 {total_time:.2f}s, 平均每指标 {avg_time_per_indicator:.3f}s")

        return result

    def analyze_data_loading_performance(self,
                                       stock_codes: List[str],
                                       end_dates: List[str]) -> Dict[str, Any]:
        """
        分析数据加载性能

        Args:
            stock_codes: 股票代码列表
            end_dates: 结束日期列表

        Returns:
            Dict[str, Any]: 数据加载性能分析结果
        """
        logger.info(f"开始分析数据加载性能，股票数量: {len(stock_codes)}")

        # 创建数据处理器
        data_processor = PeriodDataProcessor()

        loading_times = []
        cache_hits = 0
        cache_misses = 0

        for i, (stock_code, end_date) in enumerate(zip(stock_codes, end_dates)):
            try:
                start_time = time.time()

                # 检查缓存
                cache_key = f"{stock_code}_{end_date}"
                is_cache_hit = cache_key in data_processor.data_cache

                if is_cache_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1

                # 获取多周期数据
                stock_data = data_processor.get_multi_period_data(
                    stock_code=stock_code,
                    end_date=end_date
                )

                end_time = time.time()
                load_time = end_time - start_time

                loading_times.append({
                    'stock_code': stock_code,
                    'end_date': end_date,
                    'load_time': load_time,
                    'cache_hit': is_cache_hit,
                    'data_size': sum(len(df) for df in stock_data.values() if df is not None)
                })

            except Exception as e:
                logger.warning(f"加载股票 {stock_code} 数据时出错: {e}")
                loading_times.append({
                    'stock_code': stock_code,
                    'end_date': end_date,
                    'load_time': 0,
                    'cache_hit': False,
                    'error': str(e)
                })

        # 计算统计信息
        successful_loads = [t for t in loading_times if 'error' not in t]
        total_time = sum(t['load_time'] for t in successful_loads)
        avg_time = total_time / len(successful_loads) if successful_loads else 0

        cache_hit_times = [t['load_time'] for t in successful_loads if t['cache_hit']]
        cache_miss_times = [t['load_time'] for t in successful_loads if not t['cache_hit']]

        avg_cache_hit_time = sum(cache_hit_times) / len(cache_hit_times) if cache_hit_times else 0
        avg_cache_miss_time = sum(cache_miss_times) / len(cache_miss_times) if cache_miss_times else 0

        result = {
            'total_requests': len(stock_codes),
            'successful_loads': len(successful_loads),
            'success_rate': len(successful_loads) / len(stock_codes) if stock_codes else 0,
            'total_time': total_time,
            'avg_time_per_load': avg_time,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_rate': cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0,
            'avg_cache_hit_time': avg_cache_hit_time,
            'avg_cache_miss_time': avg_cache_miss_time,
            'cache_efficiency': avg_cache_miss_time / avg_cache_hit_time if avg_cache_hit_time > 0 else 0,
            'loading_details': loading_times
        }

        logger.info(f"数据加载性能分析完成: 总时间 {total_time:.2f}s, 缓存命中率 {result['cache_hit_rate']:.2%}")

        return result

    def _generate_performance_summary(self, indicator_performance: Dict[str, Dict]) -> Dict[str, Any]:
        """
        生成性能摘要

        Args:
            indicator_performance: 指标性能数据

        Returns:
            Dict[str, Any]: 性能摘要
        """
        times = [p.get('total_time', 0) for p in indicator_performance.values() if 'error' not in p]

        if not times:
            return {}

        return {
            'min_time': min(times),
            'max_time': max(times),
            'median_time': np.median(times),
            'std_time': np.std(times),
            'total_time': sum(times),
            'slow_indicators_count': len([t for t in times if t > np.median(times) * 2]),
            'fast_indicators_count': len([t for t in times if t < np.median(times) * 0.5])
        }

    def analyze_memory_usage_patterns(self,
                                    buypoints_csv: str,
                                    sample_size: int = 3) -> Dict[str, Any]:
        """
        分析内存使用模式

        Args:
            buypoints_csv: 买点数据CSV文件路径
            sample_size: 采样大小

        Returns:
            Dict[str, Any]: 内存使用分析结果
        """
        logger.info("开始分析内存使用模式")

        # 获取初始内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 加载买点数据
        analyzer = BuyPointBatchAnalyzer()
        buypoints_df = analyzer.load_buypoints_from_csv(buypoints_csv)

        if buypoints_df.empty:
            return {}

        # 限制采样大小
        if len(buypoints_df) > sample_size:
            buypoints_df = buypoints_df.head(sample_size)

        memory_snapshots = []

        # 逐个分析买点，记录内存使用
        for idx, row in buypoints_df.iterrows():
            stock_code = row['stock_code']
            buypoint_date = row['buypoint_date']

            # 记录分析前内存
            before_memory = process.memory_info().rss / 1024 / 1024

            # 分析单个买点
            result = analyzer.analyze_single_buypoint(stock_code, buypoint_date)

            # 记录分析后内存
            after_memory = process.memory_info().rss / 1024 / 1024

            memory_snapshots.append({
                'stock_code': stock_code,
                'buypoint_date': buypoint_date,
                'before_memory_mb': before_memory,
                'after_memory_mb': after_memory,
                'memory_increase_mb': after_memory - before_memory,
                'has_result': bool(result)
            })

            logger.info(f"分析 {stock_code}: 内存从 {before_memory:.1f}MB 增加到 {after_memory:.1f}MB")

        # 计算内存使用统计
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory - initial_memory

        memory_increases = [s['memory_increase_mb'] for s in memory_snapshots]
        avg_memory_increase = sum(memory_increases) / len(memory_increases) if memory_increases else 0
        max_memory_increase = max(memory_increases) if memory_increases else 0

        result = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'total_memory_increase_mb': total_memory_increase,
            'avg_memory_increase_per_stock_mb': avg_memory_increase,
            'max_memory_increase_mb': max_memory_increase,
            'memory_snapshots': memory_snapshots,
            'memory_efficiency': total_memory_increase / len(buypoints_df) if len(buypoints_df) > 0 else 0
        }

        logger.info(f"内存使用分析完成: 总增长 {total_memory_increase:.1f}MB, 平均每股 {avg_memory_increase:.1f}MB")

        return result

    def generate_optimization_recommendations(self,
                                            performance_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成优化建议

        Args:
            performance_results: 性能分析结果

        Returns:
            List[Dict[str, Any]]: 优化建议列表
        """
        recommendations = []

        # 分析批量处理性能
        if 'batch_processing' in performance_results:
            batch_perf = performance_results['batch_processing']
            avg_time = batch_perf.get('avg_time_per_stock', 0)

            if avg_time > 60:  # 超过1分钟每股
                recommendations.append({
                    'category': '批量处理优化',
                    'priority': 'HIGH',
                    'issue': f'平均每股分析时间过长: {avg_time:.1f}秒',
                    'recommendation': '实施并行处理，使用多进程或多线程',
                    'expected_improvement': '50-80%性能提升',
                    'implementation': 'multiprocessing.Pool或concurrent.futures'
                })
            elif avg_time > 30:  # 超过30秒每股
                recommendations.append({
                    'category': '批量处理优化',
                    'priority': 'MEDIUM',
                    'issue': f'平均每股分析时间较长: {avg_time:.1f}秒',
                    'recommendation': '优化指标计算算法，实施缓存机制',
                    'expected_improvement': '20-40%性能提升',
                    'implementation': '算法优化和智能缓存'
                })

        # 分析指标计算性能
        if 'indicator_calculation' in performance_results:
            indicator_perf = performance_results['indicator_calculation']
            top_slow = indicator_perf.get('top_10_slowest', {})

            for indicator_name, perf_data in list(top_slow.items())[:3]:
                if perf_data.get('total_time', 0) > 1:  # 超过1秒
                    recommendations.append({
                        'category': '指标计算优化',
                        'priority': 'HIGH',
                        'issue': f'指标 {indicator_name} 计算耗时: {perf_data.get("total_time", 0):.2f}秒',
                        'recommendation': '优化算法实现，使用向量化计算',
                        'expected_improvement': '30-60%性能提升',
                        'implementation': 'numpy向量化操作，避免循环计算'
                    })

        # 分析数据加载性能
        if 'data_loading' in performance_results:
            data_perf = performance_results['data_loading']
            cache_hit_rate = data_perf.get('cache_hit_rate', 0)

            if cache_hit_rate < 0.5:  # 缓存命中率低于50%
                recommendations.append({
                    'category': '缓存优化',
                    'priority': 'MEDIUM',
                    'issue': f'缓存命中率过低: {cache_hit_rate:.1%}',
                    'recommendation': '改进缓存策略，增加缓存容量',
                    'expected_improvement': '20-40%数据加载性能提升',
                    'implementation': 'LRU缓存，预加载策略'
                })

        # 分析内存使用
        if 'memory_usage' in performance_results:
            memory_perf = performance_results['memory_usage']
            avg_increase = memory_perf.get('avg_memory_increase_per_stock_mb', 0)

            if avg_increase > 100:  # 每股超过100MB内存增长
                recommendations.append({
                    'category': '内存优化',
                    'priority': 'HIGH',
                    'issue': f'内存使用过高: 平均每股 {avg_increase:.1f}MB',
                    'recommendation': '实施内存管理策略，及时释放不需要的数据',
                    'expected_improvement': '50-70%内存使用减少',
                    'implementation': '数据分块处理，垃圾回收优化'
                })

        # 系统级优化建议
        cpu_count = multiprocessing.cpu_count()
        recommendations.append({
            'category': '系统优化',
            'priority': 'MEDIUM',
            'issue': f'系统有 {cpu_count} 个CPU核心，可以利用并行处理',
            'recommendation': '实施多进程并行处理架构',
            'expected_improvement': f'理论上可获得 {cpu_count}x 性能提升',
            'implementation': f'使用 {cpu_count} 个进程的进程池'
        })

        return recommendations
