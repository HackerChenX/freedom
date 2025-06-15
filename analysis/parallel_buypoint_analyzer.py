#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
并行买点分析器

实现多进程并行处理的高性能买点分析器
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
import json

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer
from analysis.buypoints.period_data_processor import PeriodDataProcessor
from analysis.buypoints.auto_indicator_analyzer import AutoIndicatorAnalyzer

logger = get_logger(__name__)


def analyze_single_buypoint_worker(args):
    """
    单个买点分析工作函数（用于多进程）
    
    Args:
        args: (stock_code, buypoint_date, worker_id)
        
    Returns:
        Dict: 分析结果
    """
    stock_code, buypoint_date, worker_id = args
    
    try:
        # 在每个进程中创建独立的分析器实例
        analyzer = BuyPointBatchAnalyzer()
        
        # 分析单个买点
        result = analyzer.analyze_single_buypoint(stock_code, buypoint_date)
        
        return {
            'stock_code': stock_code,
            'buypoint_date': buypoint_date,
            'worker_id': worker_id,
            'success': bool(result),
            'result': result
        }
        
    except Exception as e:
        return {
            'stock_code': stock_code,
            'buypoint_date': buypoint_date,
            'worker_id': worker_id,
            'success': False,
            'error': str(e)
        }


class ParallelBuyPointAnalyzer:
    """并行买点分析器"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        logger.info(f"并行分析器初始化，最大工作进程: {self.max_workers}")
        
    def analyze_batch_buypoints_parallel(self, 
                                       buypoints_df: pd.DataFrame,
                                       chunk_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        并行批量分析买点
        
        Args:
            buypoints_df: 买点数据DataFrame
            chunk_size: 分块大小，默认为进程数
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        if chunk_size is None:
            chunk_size = self.max_workers
            
        logger.info(f"开始并行批量分析，总买点数: {len(buypoints_df)}, 进程数: {self.max_workers}")
        
        # 准备工作任务
        tasks = []
        for idx, row in buypoints_df.iterrows():
            worker_id = idx % self.max_workers
            tasks.append((row['stock_code'], row['buypoint_date'], worker_id))
        
        # 使用进程池并行处理
        results = []
        start_time = time.time()
        
        with multiprocessing.Pool(processes=self.max_workers) as pool:
            # 提交所有任务
            future_results = pool.map(analyze_single_buypoint_worker, tasks)
            
            # 收集结果
            for result in future_results:
                if result['success']:
                    results.append(result['result'])
                else:
                    logger.warning(f"分析失败 {result['stock_code']} {result['buypoint_date']}: {result.get('error', '未知错误')}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"并行分析完成: 总时间 {total_time:.2f}s, 成功 {len(results)}/{len(buypoints_df)}")
        
        return results
    
    def analyze_batch_buypoints_concurrent(self, 
                                         buypoints_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        使用concurrent.futures的并行分析
        
        Args:
            buypoints_df: 买点数据DataFrame
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        logger.info(f"开始concurrent.futures并行分析，总买点数: {len(buypoints_df)}")
        
        results = []
        start_time = time.time()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_buypoint = {}
            
            for idx, row in buypoints_df.iterrows():
                worker_id = idx % self.max_workers
                future = executor.submit(
                    analyze_single_buypoint_worker,
                    (row['stock_code'], row['buypoint_date'], worker_id)
                )
                future_to_buypoint[future] = (row['stock_code'], row['buypoint_date'])
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_buypoint):
                stock_code, buypoint_date = future_to_buypoint[future]
                try:
                    result = future.result()
                    if result['success']:
                        results.append(result['result'])
                    else:
                        logger.warning(f"分析失败 {stock_code} {buypoint_date}: {result.get('error', '未知错误')}")
                except Exception as e:
                    logger.error(f"获取结果时出错 {stock_code} {buypoint_date}: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"concurrent.futures分析完成: 总时间 {total_time:.2f}s, 成功 {len(results)}/{len(buypoints_df)}")
        
        return results


def compare_performance(buypoints_csv: str, sample_size: int = 5):
    """
    性能对比测试
    
    Args:
        buypoints_csv: 买点数据CSV文件路径
        sample_size: 测试样本大小
    """
    print("="*60)
    print("并行处理性能对比测试")
    print("="*60)
    
    # 加载测试数据
    base_analyzer = BuyPointBatchAnalyzer()
    buypoints_df = base_analyzer.load_buypoints_from_csv(buypoints_csv)
    
    if buypoints_df.empty:
        print("❌ 无法加载买点数据")
        return
    
    # 限制测试样本
    test_df = buypoints_df.head(sample_size)
    print(f"测试样本: {len(test_df)} 个买点")
    
    results = {}
    
    # 1. 串行处理测试
    print("\n1. 串行处理测试...")
    start_time = time.time()
    serial_results = base_analyzer.analyze_batch_buypoints(test_df)
    serial_time = time.time() - start_time
    
    results['serial'] = {
        'time': serial_time,
        'success_count': len(serial_results),
        'avg_time_per_stock': serial_time / len(test_df)
    }
    
    print(f"   串行处理: {serial_time:.2f}s, 平均每股: {serial_time/len(test_df):.2f}s")
    
    # 2. 多进程并行测试
    print("\n2. 多进程并行测试...")
    parallel_analyzer = ParallelBuyPointAnalyzer()
    start_time = time.time()
    parallel_results = parallel_analyzer.analyze_batch_buypoints_parallel(test_df)
    parallel_time = time.time() - start_time
    
    results['parallel'] = {
        'time': parallel_time,
        'success_count': len(parallel_results),
        'avg_time_per_stock': parallel_time / len(test_df)
    }
    
    print(f"   并行处理: {parallel_time:.2f}s, 平均每股: {parallel_time/len(test_df):.2f}s")
    
    # 3. concurrent.futures测试
    print("\n3. concurrent.futures测试...")
    start_time = time.time()
    concurrent_results = parallel_analyzer.analyze_batch_buypoints_concurrent(test_df)
    concurrent_time = time.time() - start_time
    
    results['concurrent'] = {
        'time': concurrent_time,
        'success_count': len(concurrent_results),
        'avg_time_per_stock': concurrent_time / len(test_df)
    }
    
    print(f"   concurrent.futures: {concurrent_time:.2f}s, 平均每股: {concurrent_time/len(test_df):.2f}s")
    
    # 性能对比分析
    print("\n4. 性能对比分析")
    if serial_time > 0:
        parallel_speedup = serial_time / parallel_time if parallel_time > 0 else 0
        concurrent_speedup = serial_time / concurrent_time if concurrent_time > 0 else 0
        
        print(f"   多进程加速比: {parallel_speedup:.2f}x")
        print(f"   concurrent.futures加速比: {concurrent_speedup:.2f}x")
        print(f"   多进程性能提升: {(serial_time - parallel_time) / serial_time * 100:.1f}%")
        print(f"   concurrent.futures性能提升: {(serial_time - concurrent_time) / serial_time * 100:.1f}%")
    
    # 保存结果
    output_dir = "data/result/parallel_performance_test"
    os.makedirs(output_dir, exist_ok=True)
    
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'test_info': {
            'sample_size': len(test_df),
            'max_workers': parallel_analyzer.max_workers
        },
        'performance_comparison': results,
        'speedup_analysis': {
            'parallel_speedup': parallel_speedup if serial_time > 0 else 0,
            'concurrent_speedup': concurrent_speedup if serial_time > 0 else 0
        }
    }
    
    results_path = os.path.join(output_dir, 'parallel_performance_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n5. 测试完成，结果已保存到: {results_path}")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python parallel_buypoint_analyzer.py <buypoints.csv> [sample_size]")
        sys.exit(1)
    
    buypoints_csv = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    compare_performance(buypoints_csv, sample_size)
