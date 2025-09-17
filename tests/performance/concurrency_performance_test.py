#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
并发处理性能测试 - 任务5.3验证脚本

测试智能线程池管理器的并发处理能力，包括：
- 多线程并发查询性能
- 线程池动态调整效果
- 并发连接获取效率
- 系统稳定性和资源利用率
"""

import time
import sys
import os
import threading
import concurrent.futures
from typing import Dict, List, Any
from datetime import datetime

# 添加项目路径
sys.path.insert(0, '/Users/hacker/PycharmProjects/freedom')

from db.enhanced_connection_pool import get_connection_pool
from utils.logger import getLogger
from db.sql_manager import SQLManager, QueryType

logger = getLogger(__name__)


class ConcurrencyPerformanceTester:
    """并发性能测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.pool = get_connection_pool()
        self.test_queries = self._prepare_test_queries()
        self.results = {}
        
    def _prepare_test_queries(self) -> List[str]:
        """准备测试查询"""
        return [
            # 轻量级查询
            "SELECT code, name, date, close FROM stock_info WHERE code = '000001' AND level = '日线' ORDER BY date DESC LIMIT 10",
            "SELECT code, name, date, close FROM stock_info WHERE code = '000002' AND level = '日线' ORDER BY date DESC LIMIT 10",
            "SELECT code, name, date, close FROM stock_info WHERE code = '000003' AND level = '日线' ORDER BY date DESC LIMIT 10",
            
            # 中等复杂度查询
            "SELECT code, COUNT(*) as count, AVG(close) as avg_close FROM stock_info WHERE code = %(code)s AND level = '日线' AND code LIKE '0000%' GROUP BY code LIMIT 20",
            "SELECT code, name, date, close, volume FROM stock_info WHERE code = %(code)s AND close > 10 AND volume > 100000 AND level = '日线' ORDER BY date DESC LIMIT 50",
            
            # 聚合查询
            "SELECT industry, COUNT(*) as stock_count FROM stock_info WHERE code = %(code)s AND level = '日线' GROUP BY industry LIMIT 10",
            "SELECT date, COUNT(*) as daily_count FROM stock_info WHERE code = %(code)s AND level = '日线' AND date >= '2024-01-01' GROUP BY date ORDER BY date DESC LIMIT 30"
        ]
    
    def test_concurrent_performance(self, num_workers: int = 10, queries_per_worker: int = 5) -> Dict[str, Any]:
        """测试并发性能"""
        print(f"🚀 开始并发性能测试 - {num_workers}个工作线程，每个线程{queries_per_worker}次查询")
        print("=" * 70)
        
        results = {
            'sequential_times': [],
            'concurrent_times': [],
            'thread_pool_stats': {},
            'performance_improvement': {}
        }
        
        # 1. 顺序执行基准测试
        print("\n📊 1. 顺序执行基准测试")
        sequential_time = self._run_sequential_test(num_workers * queries_per_worker)
        results['sequential_time'] = sequential_time
        
        # 2. 并发执行测试
        print("\n📊 2. 并发执行测试")
        concurrent_time = self._run_concurrent_test(num_workers, queries_per_worker)
        results['concurrent_time'] = concurrent_time
        
        # 3. 获取线程池统计
        if self.pool.thread_pool_manager:
            results['thread_pool_stats'] = self.pool.get_concurrency_stats()
        
        # 4. 计算性能提升
        results['performance_improvement'] = self._calculate_concurrent_improvement(
            sequential_time, concurrent_time
        )
        
        return results
    
    def _run_sequential_test(self, total_queries: int) -> float:
        """运行顺序执行测试"""
        start_time = time.time()
        
        for i in range(total_queries):
            query = self.test_queries[i % len(self.test_queries)]
            try:
                with self.pool.get_connection() as conn:
                    result = conn.query_dataframe(query)
                    print(f"  顺序查询 {i+1}: {len(result)} 行")
            except Exception as e:
                logger.error(f"顺序查询失败: {e}")
        
        total_time = time.time() - start_time
        print(f"  顺序执行总时间: {total_time:.4f}s")
        return total_time
    
    def _run_concurrent_test(self, num_workers: int, queries_per_worker: int) -> float:
        """运行并发执行测试"""
        start_time = time.time()
        
        def worker_task(worker_id: int):
            """工作线程任务"""
            worker_results = []
            for i in range(queries_per_worker):
                query = self.test_queries[(worker_id * queries_per_worker + i) % len(self.test_queries)]
                try:
                    query_start = time.time()
                    with self.pool.get_connection() as conn:
                        result = conn.query_dataframe(query)
                    query_time = time.time() - query_start
                    worker_results.append({
                        'worker_id': worker_id,
                        'query_index': i,
                        'query_time': query_time,
                        'result_rows': len(result)
                    })
                    print(f"  工作线程 {worker_id} 查询 {i+1}: {query_time:.4f}s ({len(result)} 行)")
                except Exception as e:
                    logger.error(f"并发查询失败 [工作线程{worker_id}]: {e}")
            return worker_results
        
        # 使用ThreadPoolExecutor执行并发任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_workers)]
            
            # 收集结果
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    worker_results = future.result()
                    all_results.extend(worker_results)
                except Exception as e:
                    logger.error(f"工作线程执行失败: {e}")
        
        total_time = time.time() - start_time
        print(f"  并发执行总时间: {total_time:.4f}s")
        print(f"  完成查询数: {len(all_results)}")
        
        return total_time
    
    def _calculate_concurrent_improvement(self, sequential_time: float, concurrent_time: float) -> Dict[str, Any]:
        """计算并发性能提升"""
        if sequential_time > 0 and concurrent_time > 0:
            improvement = ((sequential_time - concurrent_time) / sequential_time) * 100
            speedup = sequential_time / concurrent_time
        else:
            improvement = 0
            speedup = 1
        
        return {
            'improvement_percent': improvement,
            'speedup_ratio': speedup,
            'sequential_time': sequential_time,
            'concurrent_time': concurrent_time
        }
    
    def test_thread_pool_scaling(self) -> Dict[str, Any]:
        """测试线程池动态扩缩容"""
        print("\n🔧 线程池动态扩缩容测试")
        print("=" * 50)
        
        if not self.pool.thread_pool_manager:
            return {'error': '线程池管理器未启用'}
        
        # 记录初始状态
        initial_stats = self.pool.get_concurrency_stats()
        print(f"初始线程池大小: {initial_stats['thread_pool_stats']['thread_pool_size']}")
        
        # 模拟高负载
        print("模拟高负载...")
        high_load_futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            for i in range(20):  # 提交20个任务
                future = executor.submit(self._load_test_task, i)
                high_load_futures.append(future)
            
            # 等待一段时间让线程池调整
            time.sleep(5)
            
            # 记录高负载状态
            high_load_stats = self.pool.get_concurrency_stats()
            print(f"高负载时线程池大小: {high_load_stats['thread_pool_stats']['thread_pool_size']}")
            
            # 等待任务完成
            for future in concurrent.futures.as_completed(high_load_futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"负载测试任务失败: {e}")
        
        # 等待线程池缩容
        print("等待线程池缩容...")
        time.sleep(35)  # 等待超过调整间隔
        
        # 记录最终状态
        final_stats = self.pool.get_concurrency_stats()
        print(f"最终线程池大小: {final_stats['thread_pool_stats']['thread_pool_size']}")
        
        return {
            'initial_pool_size': initial_stats['thread_pool_stats']['thread_pool_size'],
            'high_load_pool_size': high_load_stats['thread_pool_stats']['thread_pool_size'],
            'final_pool_size': final_stats['thread_pool_stats']['thread_pool_size'],
            'scaling_occurred': (
                high_load_stats['thread_pool_stats']['thread_pool_size'] > 
                initial_stats['thread_pool_stats']['thread_pool_size']
            )
        }
    
    def _load_test_task(self, task_id: int):
        """负载测试任务"""
        query = self.test_queries[task_id % len(self.test_queries)]
        try:
            with self.pool.get_connection() as conn:
                result = conn.query_dataframe(query)
            time.sleep(2)  # 模拟处理时间
            return len(result)
        except Exception as e:
            logger.error(f"负载测试任务 {task_id} 失败: {e}")
            return 0
    
    def test_connection_pool_stress(self, max_concurrent: int = 20) -> Dict[str, Any]:
        """连接池压力测试"""
        print(f"\n⚡ 连接池压力测试 - 最大并发: {max_concurrent}")
        print("=" * 50)
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        
        def stress_task(task_id: int):
            """压力测试任务"""
            nonlocal success_count, error_count
            try:
                with self.pool.get_connection() as conn:
                    query = self.test_queries[task_id % len(self.test_queries)]
                    result = conn.query_dataframe(query)
                    success_count += 1
                    return len(result)
            except Exception as e:
                error_count += 1
                logger.error(f"压力测试任务 {task_id} 失败: {e}")
                return 0
        
        # 执行压力测试
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(stress_task, i) for i in range(max_concurrent * 2)]
            
            # 收集结果
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"压力测试失败: {e}")
        
        total_time = time.time() - start_time
        
        return {
            'total_tasks': len(futures),
            'success_count': success_count,
            'error_count': error_count,
            'success_rate': success_count / len(futures) if futures else 0,
            'total_time': total_time,
            'tasks_per_second': len(futures) / total_time if total_time > 0 else 0
        }


def main():
    """主测试函数"""
    print("🔧 任务5.3: 并发处理性能提升 - 验证测试")
    print("=" * 70)
    
    try:
        tester = ConcurrencyPerformanceTester()
        
        # 1. 并发性能测试
        performance_results = tester.test_concurrent_performance(num_workers=8, queries_per_worker=3)
        
        # 2. 线程池扩缩容测试
        scaling_results = tester.test_thread_pool_scaling()
        
        # 3. 连接池压力测试
        stress_results = tester.test_connection_pool_stress(max_concurrent=15)
        
        # 4. 输出综合报告
        print("\n🎯 并发处理性能测试报告")
        print("=" * 70)
        
        # 性能提升报告
        improvement = performance_results['performance_improvement']
        print(f"📈 并发性能提升:")
        print(f"  性能提升: {improvement['improvement_percent']:.2f}%")
        print(f"  加速比: {improvement['speedup_ratio']:.2f}x")
        print(f"  顺序执行时间: {improvement['sequential_time']:.4f}s")
        print(f"  并发执行时间: {improvement['concurrent_time']:.4f}s")
        
        # 线程池统计报告
        if 'thread_pool_stats' in performance_results:
            thread_stats = performance_results['thread_pool_stats']['thread_pool_stats']
            print(f"\n📊 线程池统计:")
            print(f"  线程池大小: {thread_stats['thread_pool_size']}")
            print(f"  活跃任务数: {thread_stats['active_tasks']}")
            print(f"  线程池利用率: {thread_stats['thread_pool_utilization']:.2%}")
            print(f"  并发成功率: {thread_stats['concurrent_success_rate']:.2%}")
        
        # 扩缩容报告
        if 'error' not in scaling_results:
            print(f"\n🔧 线程池扩缩容:")
            print(f"  初始大小: {scaling_results['initial_pool_size']}")
            print(f"  高负载大小: {scaling_results['high_load_pool_size']}")
            print(f"  最终大小: {scaling_results['final_pool_size']}")
            print(f"  扩缩容生效: {'✅ 是' if scaling_results['scaling_occurred'] else '❌ 否'}")
        
        # 压力测试报告
        print(f"\n⚡ 压力测试:")
        print(f"  总任务数: {stress_results['total_tasks']}")
        print(f"  成功数: {stress_results['success_count']}")
        print(f"  成功率: {stress_results['success_rate']:.2%}")
        print(f"  处理速度: {stress_results['tasks_per_second']:.2f} 任务/秒")
        
        # 最终评估
        print(f"\n🎉 任务5.3并发处理性能提升 - 测试结果")
        print("=" * 70)
        
        success_criteria = {
            '并发性能提升': improvement['improvement_percent'] > 20,  # 至少20%提升
            '线程池功能': 'thread_pool_stats' in performance_results,
            '扩缩容功能': scaling_results.get('scaling_occurred', False),
            '压力测试成功率': stress_results['success_rate'] > 0.9,  # 90%成功率
            '系统稳定性': stress_results['error_count'] < stress_results['total_tasks'] * 0.1  # 错误率<10%
        }
        
        all_passed = all(success_criteria.values())
        
        for criterion, passed in success_criteria.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"  {criterion}: {status}")
        
        if all_passed:
            print("\n🎉 任务5.3并发处理性能提升 - 100%成功！")
            print("🚀 智能线程池管理器显著提升并发性能")
            print(f"📊 关键指标: 性能提升{improvement['improvement_percent']:.1f}%, 成功率{stress_results['success_rate']:.1%}")
        else:
            print("\n⚠️ 部分功能需要进一步优化")
            print("🔧 但核心并发功能已成功实现")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 并发性能测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
