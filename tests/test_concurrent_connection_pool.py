#!/usr/bin/python
# -*- coding: UTF-8 -*-

from config import get_config
"""
并发连接池测试脚本

测试增强连接池在高并发场景下的性能和稳定性
"""

import time
import threading
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from db.enhanced_connection_pool import get_connection_pool, initialize_connection_pool
from utils.logger import getLogger

logger = getLogger(__name__)


class ConcurrentConnectionPoolTester:
    """并发连接池测试器"""
    
    def __init__(self):
        self.pool = None
        self.test_results = []
        self.error_count = 0
        self.success_count = 0
        
    def setup_test_environment(self):
        """设置测试环境"""
        logger.info("设置测试环境...")
        
        # 初始化连接池
        self.pool = initialize_connection_pool(
            host=get_config('database.host', 'localhost'),
            port=get_config('database.port', 9000),
            database=get_config('database.name', 'stock'),
            user=get_config('database.user', 'default'),
            password=get_config('database.password', ''),
            max_connections=get_config('performance.max_connections'),
            min_connections=5
        )
        
        logger.info("测试环境设置完成")
    
    def single_query_test(self, worker_id: int, query_count: int = 10) -> Dict[str, Any]:
        """单个工作线程的查询测试"""
        worker_results = {
            'worker_id': worker_id,
            'success_count': 0,
            'error_count': 0,
            'total_time': 0.0,
            'query_times': [],
            'errors': []
        }
        
        start_time = time.time()
        
        for i in range(query_count):
            query_start = time.time()
            
            try:
                with self.pool.get_connection() as conn:
                    # 执行简单的测试查询
                    result = conn.query_dataframe(
                        "SELECT code, COUNT(*) as count FROM stock_info WHERE level = '日线' GROUP BY code LIMIT 10"
                    )
                    
                    if not result.empty:
                        worker_results['success_count'] += 1
                    else:
                        worker_results['error_count'] += 1
                        worker_results['errors'].append(f"Query {i}: Empty result")
                        
                query_time = time.time() - query_start
                worker_results['query_times'].append(query_time)
                
            except Exception as e:
                worker_results['error_count'] += 1
                worker_results['errors'].append(f"Query {i}: {str(e)}")
                logger.error(f"Worker {worker_id} Query {i} failed: {e}")
        
        worker_results['total_time'] = time.time() - start_time
        
        return worker_results
    
    def run_concurrent_test(self, num_workers: int = 15, queries_per_worker: int = 10) -> Dict[str, Any]:
        """运行并发测试"""
        logger.info(f"开始并发测试 - {num_workers} 个工作线程，每个线程 {queries_per_worker} 次查询")
        
        start_time = time.time()
        
        # 使用线程池执行并发查询
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.single_query_test, worker_id, queries_per_worker)
                for worker_id in range(num_workers)
            ]
            
            # 收集结果
            worker_results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    worker_results.append(result)
                except Exception as e:
                    logger.error(f"工作线程执行失败: {e}")
        
        total_time = time.time() - start_time
        
        # 统计测试结果
        test_summary = self._analyze_test_results(worker_results, total_time, num_workers, queries_per_worker)
        
        return test_summary
    
    def _analyze_test_results(self, worker_results: List[Dict[str, Any]], 
                             total_time: float, num_workers: int, 
                             queries_per_worker: int) -> Dict[str, Any]:
        """分析测试结果"""
        
        total_queries = num_workers * queries_per_worker
        total_success = sum(r['success_count'] for r in worker_results)
        total_errors = sum(r['error_count'] for r in worker_results)
        
        all_query_times = []
        for result in worker_results:
            all_query_times.extend(result['query_times'])
        
        # 计算统计指标
        if all_query_times:
            avg_query_time = statistics.mean(all_query_times)
            median_query_time = statistics.median(all_query_times)
            max_query_time = max(all_query_times)
            min_query_time = min(all_query_times)
        else:
            avg_query_time = median_query_time = max_query_time = min_query_time = 0.0
        
        # 成功率
        success_rate = (total_success / total_queries * 100) if total_queries > 0 else 0
        
        # 吞吐量
        throughput = total_success / total_time if total_time > 0 else 0
        
        # 获取连接池统计
        pool_stats = self.pool.get_stats_Pool()
        
        test_summary = {
            'test_config': {
                'num_workers': num_workers,
                'queries_per_worker': queries_per_worker,
                'total_queries': total_queries
            },
            'performance_metrics': {
                'total_time': f"{total_time:.2f}s",
                'success_rate': f"{success_rate:.1f}%",
                'throughput': f"{throughput:.1f} queries/sec",
                'avg_query_time': f"{avg_query_time:.3f}s",
                'median_query_time': f"{median_query_time:.3f}s",
                'min_query_time': f"{min_query_time:.3f}s",
                'max_query_time': f"{max_query_time:.3f}s"
            },
            'result_summary': {
                'total_success': total_success,
                'total_errors': total_errors,
                'error_rate': f"{(total_errors / total_queries * 100) if total_queries > 0 else 0:.1f}%"
            },
            'connection_pool_stats': pool_stats,
            'worker_details': worker_results
        }
        
        return test_summary
    
    def run_stress_test(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """运行压力测试"""
        logger.info(f"开始压力测试 - 持续 {duration_seconds} 秒")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        stress_results = {
            'start_time': start_time,
            'duration': duration_seconds,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'peak_concurrent_connections': 0,
            'errors': []
        }
        
        def stress_worker():
            """压力测试工作函数"""
            while time.time() < end_time:
                try:
                    request_start = time.time()
                    
                    with self.pool.get_connection() as conn:
                        result = conn.query_dataframe("SELECT COUNT(*) as count FROM stock_info LIMIT 1")
                        
                        stress_results['total_requests'] += 1
                        
                        if not result.empty:
                            stress_results['successful_requests'] += 1
                        else:
                            stress_results['failed_requests'] += 1
                    
                    response_time = time.time() - request_start
                    
                    # 更新平均响应时间
                    if stress_results['total_requests'] > 0:
                        stress_results['avg_response_time'] = (
                            (stress_results['avg_response_time'] * (stress_results['total_requests'] - 1) + response_time) /
                            stress_results['total_requests']
                        )
                    
                except Exception as e:
                    stress_results['failed_requests'] += 1
                    stress_results['errors'].append(str(e))
                    
                time.sleep(0.01)  # 短暂休息，避免过于密集
        
        # 启动多个压力测试线程
        stress_threads = []
        for i in range(10):  # 10个并发压力测试线程
            thread = threading.Thread(target=stress_worker)
            thread.start()
            stress_threads.append(thread)
        
        # 等待所有线程完成
        for thread in stress_threads:
            thread.join()
        
        actual_duration = time.time() - start_time
        stress_results['actual_duration'] = actual_duration
        stress_results['requests_per_second'] = stress_results['total_requests'] / actual_duration
        stress_results['success_rate'] = (
            stress_results['successful_requests'] / stress_results['total_requests'] * 100
            if stress_results['total_requests'] > 0 else 0
        )
        
        return stress_results
    
    def print_test_report(self, test_results: Dict[str, Any]):
        """打印测试报告"""
        print("\n" + "="*80)
        print("并发连接池测试报告")
        print("="*80)
        
        if 'test_config' in test_results:
            config = test_results['test_config']
            print(f"\n📋 测试配置:")
            print(f"   工作线程数: {config['num_workers']}")
            print(f"   每线程查询数: {config['queries_per_worker']}")
            print(f"   总查询数: {config['total_queries']}")
        
        if 'performance_metrics' in test_results:
            perf = test_results['performance_metrics']
            print(f"\n📊 性能指标:")
            print(f"   总执行时间: {perf['total_time']}")
            print(f"   成功率: {perf['success_rate']}")
            print(f"   吞吐量: {perf['throughput']}")
            print(f"   平均查询时间: {perf['avg_query_time']}")
            print(f"   查询时间中位数: {perf['median_query_time']}")
            print(f"   最快查询时间: {perf['min_query_time']}")
            print(f"   最慢查询时间: {perf['max_query_time']}")
        
        if 'result_summary' in test_results:
            summary = test_results['result_summary']
            print(f"\n📈 结果统计:")
            print(f"   成功查询: {summary['total_success']}")
            print(f"   失败查询: {summary['total_errors']}")
            print(f"   错误率: {summary['error_rate']}")
        
        if 'connection_pool_stats' in test_results:
            pool_stats = test_results['connection_pool_stats']
            print(f"\n🔗 连接池统计:")
            print(f"   总创建连接数: {pool_stats.get('total_created', 0)}")
            print(f"   总销毁连接数: {pool_stats.get('total_destroyed', 0)}")
            print(f"   当前活跃连接: {pool_stats.get('current_active', 0)}")
            print(f"   当前空闲连接: {pool_stats.get('current_idle', 0)}")
            print(f"   总连接数: {pool_stats.get('total_connections', 0)}")
            print(f"   平均响应时间: {pool_stats.get('avg_response_time', 0):.3f}s")
            print(f"   总请求数: {pool_stats.get('total_requests', 0)}")
            print(f"   错误数: {pool_stats.get('total_errors', 0)}")
        
        # 压力测试特有指标
        if 'requests_per_second' in test_results:
            print(f"\n🚀 压力测试指标:")
            print(f"   持续时间: {test_results.get('actual_duration', 0):.1f}s")
            print(f"   请求/秒: {test_results['requests_per_second']:.1f}")
            print(f"   总请求数: {test_results['total_requests']}")
            print(f"   成功请求: {test_results['successful_requests']}")
            print(f"   失败请求: {test_results['failed_requests']}")
            print(f"   成功率: {test_results['success_rate']:.1f}%")
            print(f"   平均响应时间: {test_results['avg_response_time']:.3f}s")
        
        print("\n" + "="*80)


def main_test_concurrent_connection_pool():
    """主测试函数"""
    tester = ConcurrentConnectionPoolTester()
    
    try:
        # 设置测试环境
        tester.setup_test_environment()
        
        # 运行并发测试
        print("开始并发连接池测试...")
        concurrent_results = tester.run_concurrent_test(num_workers=15, queries_per_worker=10)
        tester.print_test_report(concurrent_results)
        
        # 运行压力测试
        print("\n开始压力测试...")
        stress_results = tester.run_stress_test(duration_seconds=30)
        tester.print_test_report(stress_results)
        
        print("\n✅ 所有测试完成！")
        
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        print(f"\n❌ 测试失败: {e}")
    
    finally:
        # 清理资源
        if tester.pool:
            tester.pool.close_Pool()


if __name__ == "__main__":
    main_test_concurrent_connection_pool() 