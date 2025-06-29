#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
测试优化后的股票选股性能

对比优化前后的性能差异
"""

import time
import pandas as pd
import psutil
import os
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到Python路径
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from strategy.strategy_executor import StrategyExecutor
from strategy.optimized_strategy_executor import OptimizedStrategyExecutor
from strategy.strategy_manager import StrategyManager
from utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceComparator:
    """性能对比器"""
    
    def __init__(self):
        self.strategy_manager = StrategyManager()
        
        # 创建测试策略
        self.test_strategy = {
            "strategy_id": "PERFORMANCE_TEST_STRATEGY",
            "name": "性能测试策略",
            "description": "用于测试选股性能的策略",
            "conditions": [
                {
                    "type": "price",
                    "field": "close",
                    "operator": ">",
                    "value": 5.0,
                    "description": "股价大于5元"
                },
                {
                    "type": "volume",
                    "field": "volume",
                    "operator": ">",
                    "value": 1000000,
                    "description": "成交量大于100万"
                },
                {
                    "type": "indicator",
                    "indicator": "rsi",
                    "operator": "between",
                    "value": [30, 70],
                    "description": "RSI在30-70之间"
                }
            ],
            "filters": {
                "market_cap": {"min": 1000000000},  # 市值大于10亿
                "price": {"min": 3.0, "max": 100.0}  # 价格在3-100元之间
            },
            "result_filters": {
                "max_results": 50,
                "min_score": 50
            }
        }
    
    def get_stock_sample(self, sample_size: int = 1000) -> pd.DataFrame:
        """
        获取股票样本
        
        Args:
            sample_size: 样本大小
            
        Returns:
            pd.DataFrame: 股票列表
        """
        try:
            # 模拟股票列表
            stock_codes = []
            
            # 生成沪深股票代码
            for i in range(1, sample_size // 2 + 1):
                # 沪市股票
                sh_code = f"{i:06d}.SH"
                stock_codes.append(sh_code)
                
                # 深市股票
                sz_code = f"{i:06d}.SZ"
                stock_codes.append(sz_code)
            
            # 限制到指定数量
            stock_codes = stock_codes[:sample_size]
            
            return pd.DataFrame({
                'stock_code': stock_codes,
                'stock_name': [f'股票{i}' for i in range(len(stock_codes))]
            })
            
        except Exception as e:
            logger.error(f"获取股票样本失败: {e}")
            return pd.DataFrame()
    
    def test_original_executor(self, stock_sample: pd.DataFrame) -> Dict[str, Any]:
        """
        测试原始执行器性能
        
        Args:
            stock_sample: 股票样本
            
        Returns:
            Dict[str, Any]: 性能结果
        """
        logger.info("开始测试原始策略执行器")
        
        # 记录初始状态
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        try:
            # 创建原始执行器
            executor = StrategyExecutor(max_workers=16, cache_enabled=True)
            
            # 保存测试策略
            strategy_id = "test_original_strategy"
            self.strategy_manager.save_strategy(strategy_id, self.test_strategy)
            
            # 执行策略
            def progress_callback(progress, message):
                print(f"原始执行器进度: {progress:.1%} - {message}")
            
            results = executor.execute_strategy_by_id(
                strategy_id=strategy_id,
                strategy_manager=self.strategy_manager,
                end_date=datetime.now().strftime("%Y-%m-%d"),
                progress_callback=progress_callback
            )
            
            # 记录结束状态
            end_time = time.time()
            end_memory = psutil.virtual_memory().percent
            
            # 清理
            self.strategy_manager.delete_strategy(strategy_id)
            
            return {
                'executor_type': 'original',
                'total_time': end_time - start_time,
                'result_count': len(results) if results is not None else 0,
                'memory_start': start_memory,
                'memory_end': end_memory,
                'memory_peak': max(start_memory, end_memory),
                'avg_time_per_stock': (end_time - start_time) / len(stock_sample) if len(stock_sample) > 0 else 0,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"原始执行器测试失败: {e}")
            return {
                'executor_type': 'original',
                'total_time': time.time() - start_time,
                'result_count': 0,
                'memory_start': start_memory,
                'memory_end': psutil.virtual_memory().percent,
                'success': False,
                'error': str(e)
            }
    
    def test_optimized_executor(self, stock_sample: pd.DataFrame) -> Dict[str, Any]:
        """
        测试优化执行器性能
        
        Args:
            stock_sample: 股票样本
            
        Returns:
            Dict[str, Any]: 性能结果
        """
        logger.info("开始测试优化策略执行器")
        
        # 记录初始状态
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        try:
            # 创建优化执行器
            executor = OptimizedStrategyExecutor(
                max_workers=32,  # 增加并发数
                cache_enabled=True,
                batch_size=150,  # 使用较大的批次
                enable_memory_monitoring=True
            )
            
            # 执行优化策略
            def progress_callback(progress, message):
                print(f"优化执行器进度: {progress:.1%} - {message}")
            
            results = executor.execute_strategy_optimized(
                strategy_plan=self.test_strategy,
                end_date=datetime.now().strftime("%Y-%m-%d"),
                progress_callback=progress_callback,
                enable_early_stop=False,  # 不启用早停，完整测试
                max_results=50
            )
            
            # 记录结束状态
            end_time = time.time()
            end_memory = psutil.virtual_memory().percent
            
            # 获取性能报告
            performance_report = executor.get_performance_report()
            
            return {
                'executor_type': 'optimized',
                'total_time': end_time - start_time,
                'result_count': len(results) if results is not None else 0,
                'memory_start': start_memory,
                'memory_end': end_memory,
                'memory_peak': performance_report.get('memory_peak', end_memory),
                'avg_time_per_stock': (end_time - start_time) / len(stock_sample) if len(stock_sample) > 0 else 0,
                'performance_report': performance_report,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"优化执行器测试失败: {e}")
            return {
                'executor_type': 'optimized',
                'total_time': time.time() - start_time,
                'result_count': 0,
                'memory_start': start_memory,
                'memory_end': psutil.virtual_memory().percent,
                'success': False,
                'error': str(e)
            }
    
    def run_performance_comparison(self, sample_sizes: list = [500, 1000, 2000]) -> Dict[str, Any]:
        """
        运行性能对比测试
        
        Args:
            sample_sizes: 测试样本大小列表
            
        Returns:
            Dict[str, Any]: 对比结果
        """
        logger.info("开始性能对比测试")
        
        results = {
            'test_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'system_info': {
                'cpu_count': os.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version
            },
            'tests': []
        }
        
        for sample_size in sample_sizes:
            logger.info(f"测试样本大小: {sample_size}")
            
            # 获取股票样本
            stock_sample = self.get_stock_sample(sample_size)
            
            if stock_sample.empty:
                logger.warning(f"无法获取样本大小为 {sample_size} 的股票数据")
                continue
            
            test_result = {
                'sample_size': sample_size,
                'actual_sample_size': len(stock_sample)
            }
            
            # 测试原始执行器
            print(f"\n=== 测试原始执行器 (样本: {sample_size}) ===")
            original_result = self.test_original_executor(stock_sample)
            test_result['original'] = original_result
            
            # 等待一段时间，让系统恢复
            time.sleep(2)
            
            # 测试优化执行器
            print(f"\n=== 测试优化执行器 (样本: {sample_size}) ===")
            optimized_result = self.test_optimized_executor(stock_sample)
            test_result['optimized'] = optimized_result
            
            # 计算性能提升
            if (original_result.get('success') and optimized_result.get('success') and 
                original_result.get('total_time', 0) > 0):
                
                time_improvement = (
                    (original_result['total_time'] - optimized_result['total_time']) /
                    original_result['total_time'] * 100
                )
                test_result['performance_improvement'] = {
                    'time_reduction_percent': time_improvement,
                    'speed_multiplier': original_result['total_time'] / optimized_result['total_time']
                }
            
            results['tests'].append(test_result)
            
            # 等待系统恢复
            time.sleep(3)
        
        return results
    
    def print_comparison_report(self, results: Dict[str, Any]):
        """
        打印对比报告
        
        Args:
            results: 对比结果
        """
        print("\n" + "="*80)
        print("🚀 股票选股性能优化对比报告")
        print("="*80)
        
        print(f"测试时间: {results['test_time']}")
        print(f"系统信息: CPU核心数={results['system_info']['cpu_count']}, "
              f"内存={results['system_info']['memory_total_gb']:.1f}GB")
        
        print("\n📊 性能对比结果:")
        print("-"*80)
        
        for test in results['tests']:
            sample_size = test['sample_size']
            print(f"\n🔍 样本大小: {sample_size} 只股票")
            
            original = test.get('original', {})
            optimized = test.get('optimized', {})
            
            if original.get('success') and optimized.get('success'):
                print(f"  原始执行器:")
                print(f"    ⏱️  总耗时: {original['total_time']:.2f}秒")
                print(f"    📈 平均每股: {original['avg_time_per_stock']*1000:.2f}毫秒")
                print(f"    🧠 内存使用: {original['memory_start']:.1f}% → {original['memory_end']:.1f}%")
                print(f"    📋 结果数量: {original['result_count']}")
                
                print(f"  优化执行器:")
                print(f"    ⏱️  总耗时: {optimized['total_time']:.2f}秒")
                print(f"    📈 平均每股: {optimized['avg_time_per_stock']*1000:.2f}毫秒")
                print(f"    🧠 内存使用: {optimized['memory_start']:.1f}% → {optimized['memory_end']:.1f}%")
                print(f"    📋 结果数量: {optimized['result_count']}")
                
                if 'performance_improvement' in test:
                    improvement = test['performance_improvement']
                    print(f"  🎯 性能提升:")
                    print(f"    ⚡ 时间减少: {improvement['time_reduction_percent']:.1f}%")
                    print(f"    🚀 速度提升: {improvement['speed_multiplier']:.1f}倍")
            else:
                print(f"  ❌ 测试失败")
                if not original.get('success'):
                    print(f"    原始执行器错误: {original.get('error', '未知错误')}")
                if not optimized.get('success'):
                    print(f"    优化执行器错误: {optimized.get('error', '未知错误')}")
        
        print("\n" + "="*80)


def main():
    """主函数"""
    print("🚀 开始股票选股性能优化测试")
    
    # 创建性能对比器
    comparator = PerformanceComparator()
    
    # 运行性能对比测试
    # 从小样本开始测试，避免系统压力过大
    sample_sizes = [100, 500, 1000]  # 可以根据需要调整
    
    try:
        results = comparator.run_performance_comparison(sample_sizes)
        
        # 打印对比报告
        comparator.print_comparison_report(results)
        
        # 保存结果到文件
        import json
        result_file = f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 详细结果已保存到: {result_file}")
        
    except Exception as e:
        logger.error(f"性能测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 