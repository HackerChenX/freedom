#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
内存优化测试脚本

测试大规模股票选股时的内存使用和GC压力优化效果。
验证内存优化器在4000+股票场景下的性能表现。

Author: System
Date: 2025-01-15
"""

import os
import sys
import time
import psutil
import gc
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
import tracemalloc

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from strategy.large_scale_memory_optimizer import LargeScaleMemoryOptimizer, LargeScaleMemoryConfig
from strategy.strategy_executor import UnifiedStrategyExecutor as UnifiedStrategyExecutor
from utils.logger import get_logger

logger = get_logger(__name__)


class MemoryOptimizationTester:
    """内存优化测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.baseline_results = {}
        
    def run_comprehensive_memory_test(self, stock_count: int = 4000) -> Dict[str, Any]:
        """
        运行综合内存测试
        
        Args:
            stock_count: 测试股票数量
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        logger.info(f"🧪 开始综合内存优化测试: {stock_count}只股票")
        
        # 创建测试股票列表
        stock_codes = self._generate_test_stocks(stock_count)
        
        # 测试1: 基准测试（无优化）
        logger.info("📊 执行基准测试（无优化）...")
        baseline_result = self._run_baseline_test(stock_codes)
        
        # 测试2: 内存优化测试
        logger.info("🚀 执行内存优化测试...")
        optimized_result = self._run_optimized_test(stock_codes)
        
        # 测试3: 内存压力测试
        logger.info("💪 执行内存压力测试...")
        pressure_result = self._run_memory_pressure_test(stock_codes)
        
        # 对比分析
        comparison = self._compare_results(baseline_result, optimized_result, pressure_result)
        
        # 生成测试报告
        test_report = self._generate_test_report(
            stock_count, baseline_result, optimized_result, pressure_result, comparison
        )
        
        logger.info("✅ 综合内存优化测试完成")
        return test_report
    
    def _generate_test_stocks(self, count: int) -> List[str]:
        """
        生成测试股票列表
        
        Args:
            count: 股票数量
            
        Returns:
            List[str]: 股票代码列表
        """
        stocks = []
        for i in range(1, count + 1):
            if i <= 2000:
                stocks.append(f"{i:06d}.SH")  # 上海市场
            else:
                stocks.append(f"{i-2000:06d}.SZ")  # 深圳市场
        return stocks
    
    def _run_baseline_test(self, stock_codes: List[str]) -> Dict[str, Any]:
        """
        运行基准测试（无优化）
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            Dict[str, Any]: 基准测试结果
        """
        # 开始内存跟踪
        tracemalloc.start()
        start_time = time.time()
        initial_memory = psutil.virtual_memory().percent
        
        # 模拟传统的股票处理方式（一次性加载大量数据）
        results = []
        processed_count = 0
        gc_count = 0
        memory_peak = initial_memory
        
        try:
            # 批量处理，模拟传统方式
            batch_size = 500  # 大批次
            for i in range(0, len(stock_codes), batch_size):
                batch = stock_codes[i:i+batch_size]
                
                # 模拟数据加载和处理
                batch_data = self._simulate_stock_processing(batch)
                results.extend(batch_data)
                processed_count += len(batch)
                
                # 检查内存使用
                current_memory = psutil.virtual_memory().percent
                memory_peak = max(memory_peak, current_memory)
                
                # 偶尔触发GC
                if i % 2000 == 0:
                    gc.collect()
                    gc_count += 1
                
                # 如果内存过高，强制清理
                if current_memory > 85:
                    gc.collect()
                    gc_count += 1
        
        except Exception as e:
            logger.error(f"基准测试失败: {e}")
        
        end_time = time.time()
        final_memory = psutil.virtual_memory().percent
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'test_type': 'baseline',
            'processed_count': processed_count,
            'processing_time': end_time - start_time,
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_peak': memory_peak,
            'memory_allocated_mb': peak / 1024 / 1024,
            'gc_count': gc_count,
            'results_count': len(results),
            'processing_speed': processed_count / (end_time - start_time)
        }
    
    def _run_optimized_test(self, stock_codes: List[str]) -> Dict[str, Any]:
        """
        运行内存优化测试
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            Dict[str, Any]: 优化测试结果
        """
        # 创建内存优化配置
        config = LargeScaleMemoryConfig(
            max_memory_usage_percent=75.0,
            initial_batch_size=100,
            min_batch_size=20,
            max_batch_size=300,
            auto_batch_adjustment=True,
            enable_streaming=True,
            memory_pressure_relief=True
        )
        
        optimizer = LargeScaleMemoryOptimizer(config)
        
        # 开始内存跟踪
        tracemalloc.start()
        start_time = time.time()
        initial_memory = psutil.virtual_memory().percent
        
        results = []
        
        def processing_func(stock_code: str) -> Dict[str, Any]:
            """模拟处理函数"""
            return self._simulate_single_stock_processing(stock_code)
        
        try:
            # 使用内存优化器处理
            for result in optimizer.process_large_stock_selection(stock_codes, processing_func):
                results.append(result)
        
        except Exception as e:
            logger.error(f"内存优化测试失败: {e}")
        
        end_time = time.time()
        final_memory = psutil.virtual_memory().percent
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 获取优化器报告
        optimizer_report = optimizer.get_memory_optimization_report()
        
        return {
            'test_type': 'optimized',
            'processed_count': len(stock_codes),
            'processing_time': end_time - start_time,
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_peak': optimizer_report['performance_stats']['memory_peak'],
            'memory_allocated_mb': peak / 1024 / 1024,
            'gc_count': optimizer_report['performance_stats']['gc_triggered'],
            'results_count': len(results),
            'processing_speed': len(stock_codes) / (end_time - start_time),
            'batch_adjustments': optimizer_report['performance_stats']['batch_adjustments'],
            'memory_pressure_events': optimizer_report['performance_stats']['memory_pressure_relief_count'],
            'optimizer_report': optimizer_report
        }
    
    def _run_memory_pressure_test(self, stock_codes: List[str]) -> Dict[str, Any]:
        """
        运行内存压力测试
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            Dict[str, Any]: 压力测试结果
        """
        # 创建高压力配置
        config = LargeScaleMemoryConfig(
            max_memory_usage_percent=60.0,  # 更严格的内存限制
            initial_batch_size=50,          # 更小的初始批次
            min_batch_size=10,              # 更小的最小批次
            max_batch_size=100,             # 更小的最大批次
            auto_batch_adjustment=True,
            memory_pressure_relief=True,
            force_gc_threshold=65.0         # 更低的GC阈值
        )
        
        optimizer = LargeScaleMemoryOptimizer(config)
        
        # 模拟内存压力（预先分配一些内存）
        pressure_data = [np.random.randn(100000) for _ in range(10)]
        
        start_time = time.time()
        initial_memory = psutil.virtual_memory().percent
        
        results = []
        
        def processing_func(stock_code: str) -> Dict[str, Any]:
            """模拟处理函数（更耗内存）"""
            # 模拟更复杂的计算
            temp_data = np.random.randn(1000)
            result = self._simulate_single_stock_processing(stock_code)
            del temp_data  # 手动清理
            return result
        
        try:
            # 使用内存优化器处理
            for result in optimizer.process_large_stock_selection(stock_codes, processing_func):
                results.append(result)
        
        except Exception as e:
            logger.error(f"内存压力测试失败: {e}")
        
        finally:
            # 清理压力数据
            del pressure_data
            gc.collect()
        
        end_time = time.time()
        final_memory = psutil.virtual_memory().percent
        
        # 获取优化器报告
        optimizer_report = optimizer.get_memory_optimization_report()
        
        return {
            'test_type': 'pressure',
            'processed_count': len(stock_codes),
            'processing_time': end_time - start_time,
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_peak': optimizer_report['performance_stats']['memory_peak'],
            'gc_count': optimizer_report['performance_stats']['gc_triggered'],
            'results_count': len(results),
            'processing_speed': len(stock_codes) / (end_time - start_time),
            'batch_adjustments': optimizer_report['performance_stats']['batch_adjustments'],
            'memory_pressure_events': optimizer_report['performance_stats']['memory_pressure_relief_count'],
            'optimizer_report': optimizer_report
        }
    
    def _simulate_stock_processing(self, stock_codes: List[str]) -> List[Dict[str, Any]]:
        """
        模拟股票批量处理
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            List[Dict[str, Any]]: 处理结果
        """
        results = []
        
        # 模拟加载大量数据
        for code in stock_codes:
            # 模拟股票数据（消耗内存）
            stock_data = {
                'close': np.random.randn(252) + 100,  # 一年的价格数据
                'volume': np.random.randint(1000000, 10000000, 252),
                'high': np.random.randn(252) + 105,
                'low': np.random.randn(252) + 95
            }
            
            # 模拟指标计算
            ma5 = pd.Series(stock_data['close']).rolling(5).mean()
            ma20 = pd.Series(stock_data['close']).rolling(20).mean()
            
            # 模拟评分
            score = np.random.random()
            
            if score > 0.7:  # 30%的匹配率
                results.append({
                    'stock_code': code,
                    'score': score,
                    'data_size': len(stock_data['close'])
                })
        
        return results
    
    def _simulate_single_stock_processing(self, stock_code: str) -> Dict[str, Any]:
        """
        模拟单只股票处理
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        # 模拟轻量级处理
        score = np.random.random()
        
        if score > 0.7:  # 30%的匹配率
            return {
                'stock_code': stock_code,
                'score': score,
                'processed_at': time.time()
            }
        
        return None
    
    def _compare_results(
        self, 
        baseline: Dict[str, Any], 
        optimized: Dict[str, Any], 
        pressure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        对比测试结果
        
        Args:
            baseline: 基准测试结果
            optimized: 优化测试结果  
            pressure: 压力测试结果
            
        Returns:
            Dict[str, Any]: 对比结果
        """
        return {
            'performance_improvement': {
                'speed_improvement_percent': (
                    (optimized['processing_speed'] / baseline['processing_speed'] - 1) * 100
                ),
                'time_reduction_percent': (
                    (baseline['processing_time'] - optimized['processing_time']) / baseline['processing_time'] * 100
                ),
                'memory_peak_reduction_percent': (
                    (baseline['memory_peak'] - optimized['memory_peak']) / baseline['memory_peak'] * 100
                ),
                'gc_count_change_percent': (
                    (optimized['gc_count'] - baseline['gc_count']) / max(baseline['gc_count'], 1) * 100
                )
            },
            'pressure_test_performance': {
                'pressure_vs_baseline_speed': pressure['processing_speed'] / baseline['processing_speed'],
                'pressure_vs_optimized_speed': pressure['processing_speed'] / optimized['processing_speed'],
                'memory_pressure_events': pressure['memory_pressure_events'],
                'batch_adjustments': pressure['batch_adjustments']
            },
            'memory_optimization_effectiveness': {
                'baseline_memory_efficiency': baseline['processed_count'] / baseline['memory_peak'],
                'optimized_memory_efficiency': optimized['processed_count'] / optimized['memory_peak'],
                'efficiency_improvement_ratio': (
                    (optimized['processed_count'] / optimized['memory_peak']) / 
                    (baseline['processed_count'] / baseline['memory_peak'])
                )
            }
        }
    
    def _generate_test_report(
        self,
        stock_count: int,
        baseline: Dict[str, Any],
        optimized: Dict[str, Any], 
        pressure: Dict[str, Any],
        comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成测试报告
        
        Args:
            stock_count: 测试股票数量
            baseline: 基准测试结果
            optimized: 优化测试结果
            pressure: 压力测试结果
            comparison: 对比结果
            
        Returns:
            Dict[str, Any]: 测试报告
        """
        return {
            'test_summary': {
                'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'stock_count': stock_count,
                'test_types': ['baseline', 'optimized', 'pressure']
            },
            'results': {
                'baseline': baseline,
                'optimized': optimized,
                'pressure': pressure
            },
            'comparison': comparison,
            'conclusions': {
                'memory_optimization_effective': comparison['performance_improvement']['memory_peak_reduction_percent'] > 10,
                'speed_improvement_achieved': comparison['performance_improvement']['speed_improvement_percent'] > 0,
                'pressure_handling_good': pressure['memory_pressure_events'] > 0 and pressure['processing_speed'] > 0,
                'overall_rating': self._calculate_overall_rating(comparison)
            },
            'recommendations': self._generate_recommendations(comparison)
        }
    
    def _calculate_overall_rating(self, comparison: Dict[str, Any]) -> str:
        """
        计算总体评级
        
        Args:
            comparison: 对比结果
            
        Returns:
            str: 评级
        """
        perf = comparison['performance_improvement']
        
        score = 0
        if perf['memory_peak_reduction_percent'] > 20:
            score += 3
        elif perf['memory_peak_reduction_percent'] > 10:
            score += 2
        elif perf['memory_peak_reduction_percent'] > 0:
            score += 1
        
        if perf['speed_improvement_percent'] > 50:
            score += 3
        elif perf['speed_improvement_percent'] > 20:
            score += 2
        elif perf['speed_improvement_percent'] > 0:
            score += 1
        
        if perf['time_reduction_percent'] > 30:
            score += 2
        elif perf['time_reduction_percent'] > 10:
            score += 1
        
        if score >= 7:
            return "优秀"
        elif score >= 5:
            return "良好"
        elif score >= 3:
            return "一般"
        else:
            return "需要改进"
    
    def _generate_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """
        生成改进建议
        
        Args:
            comparison: 对比结果
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        perf = comparison['performance_improvement']
        
        if perf['memory_peak_reduction_percent'] < 10:
            recommendations.append("建议进一步优化内存使用，考虑更小的批次大小或更频繁的垃圾回收")
        
        if perf['speed_improvement_percent'] < 0:
            recommendations.append("处理速度有所下降，建议优化算法效率或调整并发参数")
        
        if comparison['pressure_test_performance']['memory_pressure_events'] == 0:
            recommendations.append("内存压力处理机制未触发，建议调整压力阈值")
        
        if perf['gc_count_change_percent'] > 100:
            recommendations.append("GC触发次数显著增加，建议优化内存分配策略")
        
        if not recommendations:
            recommendations.append("内存优化效果良好，建议在生产环境中部署")
        
        return recommendations


def main_test_memory_optimization():
    """主函数"""
    tester = MemoryOptimizationTester()
    
    # 运行不同规模的测试
    test_sizes = [1000, 2000, 4000]
    
    for size in test_sizes:
        logger.info(f"🧪 开始测试 {size} 只股票的内存优化效果...")
        
        try:
            report = tester.run_comprehensive_memory_test(size)
            
            # 输出关键结果
            print(f"\n{'='*60}")
            print(f"内存优化测试报告 - {size}只股票")
            print(f"{'='*60}")
            
            baseline = report['results']['baseline']
            optimized = report['results']['optimized']
            comparison = report['comparison']['performance_improvement']
            
            print(f"基准测试:")
            print(f"  处理时间: {baseline['processing_time']:.2f}秒")
            print(f"  内存峰值: {baseline['memory_peak']:.1f}%")
            print(f"  处理速度: {baseline['processing_speed']:.1f}股/秒")
            
            print(f"\n优化测试:")
            print(f"  处理时间: {optimized['processing_time']:.2f}秒")
            print(f"  内存峰值: {optimized['memory_peak']:.1f}%")
            print(f"  处理速度: {optimized['processing_speed']:.1f}股/秒")
            
            print(f"\n性能改进:")
            print(f"  速度提升: {comparison['speed_improvement_percent']:.1f}%")
            print(f"  时间减少: {comparison['time_reduction_percent']:.1f}%")
            print(f"  内存峰值降低: {comparison['memory_peak_reduction_percent']:.1f}%")
            
            print(f"\n总体评级: {report['conclusions']['overall_rating']}")
            
            # 保存报告
            report_file = f"memory_optimization_test_report_{size}_stocks.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"测试报告已保存: {report_file}")
            
        except Exception as e:
            logger.error(f"测试 {size} 只股票失败: {e}")
            continue


if __name__ == "__main__":
    main_test_memory_optimization() 