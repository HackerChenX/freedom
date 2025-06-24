#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
并发优化性能测试

验证连接池和查询缓存优化的效果
"""

import sys
import os
import time
import threading
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from db.enhanced_connection_pool import initialize_connection_pool, get_connection_pool
from db.unified_data_manager import get_unified_data_manager
from db.query_cache import get_query_cache
from utils.logger import get_logger

logger = get_logger(__name__)


class ConcurrentOptimizationTest:
    """并发优化性能测试器"""
    
    def __init__(self):
        """初始化测试器"""
        # 初始化连接池
        self.connection_pool = initialize_connection_pool(
            host='localhost',
            port=9000,
            database='stock',
            max_connections=20,
            min_connections=5
        )
        
        # 初始化增强数据管理器
        self.data_manager = get_unified_data_manager()
        
        # 初始化查询缓存
        self.query_cache = get_query_cache()
        
        # 测试结果
        self.test_results = {
            'concurrent_tests': [],
            'cache_tests': [],
            'optimization_comparison': {},
            'performance_metrics': {}
        }
        
        logger.info("并发优化性能测试器初始化完成")
    
    def test_concurrent_queries(self, concurrent_count: int = 10, queries_per_thread: int = 5) -> Dict[str, Any]:
        """测试并发查询性能"""
        logger.info(f"开始并发查询测试，并发数: {concurrent_count}, 每线程查询数: {queries_per_thread}")
        
        # 准备测试查询
        test_queries = self._prepare_test_queries(queries_per_thread)
        
        # 执行并发测试
        start_time = time.time()
        results = []
        errors = []
        
        def execute_queries(thread_id: int) -> Dict[str, Any]:
            """执行查询的线程函数"""
            thread_results = {
                'thread_id': thread_id,
                'queries_executed': 0,
                'queries_successful': 0,
                'total_time': 0,
                'errors': []
            }
            
            thread_start = time.time()
            
            for i, query_params in enumerate(test_queries):
                try:
                    query_start = time.time()
                    
                    # 执行查询
                    result = self.data_manager.get_stock_info(**query_params)
                    
                    query_time = time.time() - query_start
                    thread_results['queries_executed'] += 1
                    
                    if result and hasattr(result, 'data') and not result.data.empty:
                        thread_results['queries_successful'] += 1
                    
                    logger.debug(f"线程 {thread_id} 查询 {i+1} 完成，耗时: {query_time:.3f}秒")
                    
                except Exception as e:
                    thread_results['errors'].append(str(e))
                    logger.error(f"线程 {thread_id} 查询 {i+1} 失败: {e}")
            
            thread_results['total_time'] = time.time() - thread_start
            return thread_results
        
        # 使用线程池执行并发查询
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_count) as executor:
            futures = [executor.submit(execute_queries, i) for i in range(concurrent_count)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append(str(e))
        
        total_time = time.time() - start_time
        
        # 分析结果
        analysis = self._analyze_concurrent_results(results, total_time, concurrent_count, queries_per_thread)
        
        logger.info(f"并发查询测试完成，总耗时: {total_time:.3f}秒，成功率: {analysis['success_rate']:.2%}")
        
        return analysis
    
    def test_cache_performance(self, cache_test_rounds: int = 3) -> Dict[str, Any]:
        """测试缓存性能"""
        logger.info(f"开始缓存性能测试，测试轮数: {cache_test_rounds}")
        
        # 准备测试查询
        test_queries = self._prepare_test_queries(10)
        
        cache_results = {
            'rounds': [],
            'cache_hit_improvement': 0,
            'avg_response_time_improvement': 0
        }
        
        for round_num in range(cache_test_rounds):
            logger.info(f"执行缓存测试轮次 {round_num + 1}/{cache_test_rounds}")
            
            round_start = time.time()
            round_results = {
                'round': round_num + 1,
                'queries': [],
                'total_time': 0,
                'cache_hits': 0,
                'cache_misses': 0
            }
            
            for i, query_params in enumerate(test_queries):
                query_start = time.time()
                
                try:
                    result = self.data_manager.get_stock_info(**query_params)
                    query_time = time.time() - query_start
                    
                    round_results['queries'].append({
                        'query_id': i,
                        'response_time': query_time,
                        'success': True
                    })
                    
                except Exception as e:
                    query_time = time.time() - query_start
                    round_results['queries'].append({
                        'query_id': i,
                        'response_time': query_time,
                        'success': False,
                        'error': str(e)
                    })
            
            round_results['total_time'] = time.time() - round_start
            cache_results['rounds'].append(round_results)
        
        # 分析缓存效果
        cache_analysis = self._analyze_cache_results(cache_results)
        
        logger.info(f"缓存性能测试完成，缓存命中率提升: {cache_analysis.get('hit_rate_improvement', 0):.2%}")
        
        return cache_analysis
    
    def test_optimization_comparison(self) -> Dict[str, Any]:
        """对比优化前后的性能"""
        logger.info("开始优化效果对比测试")
        
        # 获取当前统计信息
        current_stats = {
            'data_manager': self.data_manager.get_stats(),
            'connection_pool': self.connection_pool.get_stats(),
            'query_cache': self.query_cache.get_stats()
        }
        
        # 执行基准测试
        benchmark_results = self._run_benchmark_test()
        
        comparison = {
            'current_stats': current_stats,
            'benchmark_results': benchmark_results,
            'optimization_metrics': self._calculate_optimization_metrics(current_stats, benchmark_results)
        }
        
        logger.info("优化效果对比测试完成")
        return comparison
    
    def _prepare_test_queries(self, count: int) -> List[Dict[str, Any]]:
        """准备测试查询"""
        queries = []
        
        # 不同类型的查询
        query_templates = [
            # 单股票查询
            {'stock_code': '000001', 'level': 'DAILY', 'limit': 100},
            {'stock_code': '000002', 'level': 'DAILY', 'limit': 100},
            {'stock_code': '600000', 'level': 'DAILY', 'limit': 100},
            
            # 多股票查询
            {'stock_code': ['000001', '000002', '600000'], 'level': 'DAILY', 'limit': 300},
            {'stock_code': ['600036', '600519', '000858'], 'level': 'DAILY', 'limit': 300},
            
            # 日期范围查询
            {'stock_code': '000001', 'start_date': '2024-06-01', 'end_date': '2024-06-20', 'level': 'DAILY'},
            {'stock_code': '000002', 'start_date': '2024-05-01', 'end_date': '2024-05-31', 'level': 'DAILY'},
            
            # 过滤查询
            {'level': 'DAILY', 'filters': {'price': {'min': 10, 'max': 100}}, 'limit': 50},
            {'level': 'DAILY', 'filters': {'industry': ['电子', '计算机']}, 'limit': 50},
            
            # 大数据量查询
            {'level': 'DAILY', 'start_date': '2024-01-01', 'limit': 1000}
        ]
        
        # 循环生成指定数量的查询
        for i in range(count):
            queries.append(query_templates[i % len(query_templates)])
        
        return queries
    
    def _analyze_concurrent_results(self, results: List[Dict[str, Any]], 
                                  total_time: float, 
                                  concurrent_count: int, 
                                  queries_per_thread: int) -> Dict[str, Any]:
        """分析并发测试结果"""
        total_queries = sum(r['queries_executed'] for r in results)
        successful_queries = sum(r['queries_successful'] for r in results)
        total_errors = sum(len(r['errors']) for r in results)
        
        avg_thread_time = sum(r['total_time'] for r in results) / len(results) if results else 0
        
        analysis = {
            'concurrent_count': concurrent_count,
            'queries_per_thread': queries_per_thread,
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'total_errors': total_errors,
            'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
            'total_execution_time': total_time,
            'avg_thread_time': avg_thread_time,
            'queries_per_second': total_queries / total_time if total_time > 0 else 0,
            'concurrent_efficiency': (avg_thread_time * concurrent_count) / total_time if total_time > 0 else 0,
            'thread_results': results
        }
        
        return analysis
    
    def _analyze_cache_results(self, cache_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析缓存测试结果"""
        rounds = cache_results['rounds']
        
        if len(rounds) < 2:
            return {'error': '需要至少2轮测试来分析缓存效果'}
        
        # 计算第一轮和最后一轮的性能差异
        first_round = rounds[0]
        last_round = rounds[-1]
        
        first_avg_time = sum(q['response_time'] for q in first_round['queries']) / len(first_round['queries'])
        last_avg_time = sum(q['response_time'] for q in last_round['queries']) / len(last_round['queries'])
        
        improvement = (first_avg_time - last_avg_time) / first_avg_time if first_avg_time > 0 else 0
        
        analysis = {
            'total_rounds': len(rounds),
            'first_round_avg_time': first_avg_time,
            'last_round_avg_time': last_avg_time,
            'response_time_improvement': improvement,
            'cache_effectiveness': improvement > 0.1,  # 10%以上改善认为有效
            'rounds_detail': rounds
        }
        
        return analysis
    
    def _run_benchmark_test(self) -> Dict[str, Any]:
        """运行基准测试"""
        logger.info("执行基准性能测试")
        
        # 简单的基准测试
        test_queries = self._prepare_test_queries(5)
        
        start_time = time.time()
        successful_queries = 0
        
        for query_params in test_queries:
            try:
                result = self.data_manager.get_stock_info(**query_params)
                if result and hasattr(result, 'data') and not result.data.empty:
                    successful_queries += 1
            except Exception as e:
                logger.warning(f"基准测试查询失败: {e}")
        
        total_time = time.time() - start_time
        
        return {
            'total_queries': len(test_queries),
            'successful_queries': successful_queries,
            'total_time': total_time,
            'avg_query_time': total_time / len(test_queries),
            'success_rate': successful_queries / len(test_queries)
        }
    
    def _calculate_optimization_metrics(self, current_stats: Dict[str, Any], 
                                      benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算优化指标"""
        data_manager_stats = current_stats.get('data_manager', {})
        pool_stats = current_stats.get('connection_pool', {})
        cache_stats = current_stats.get('query_cache', {})
        
        metrics = {
            'avg_query_time': data_manager_stats.get('avg_query_time', 0),
            'cache_hit_rate': data_manager_stats.get('cache_hit_rate', 0),
            'concurrent_efficiency': pool_stats.get('total_requests', 0) / pool_stats.get('total_created', 1),
            'connection_reuse_rate': 1 - (pool_stats.get('total_created', 0) / max(pool_stats.get('total_requests', 1), 1)),
            'query_cache_hit_rate': cache_stats.get('cache_hit_rate', 0),
            'memory_cache_efficiency': cache_stats.get('memory_hit_rate', 0),
            'benchmark_performance': benchmark_results.get('avg_query_time', 0)
        }
        
        return metrics
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行全面的优化测试"""
        logger.info("=" * 80)
        logger.info("开始并发优化全面性能测试")
        logger.info("=" * 80)
        
        test_results = {
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'concurrent_test': {},
            'cache_test': {},
            'optimization_comparison': {},
            'overall_assessment': {}
        }
        
        try:
            # 1. 并发查询测试
            logger.info("步骤 1: 并发查询性能测试")
            test_results['concurrent_test'] = self.test_concurrent_queries(
                concurrent_count=10, queries_per_thread=5
            )
            
            # 2. 缓存性能测试
            logger.info("步骤 2: 缓存性能测试")
            test_results['cache_test'] = self.test_cache_performance(cache_test_rounds=3)
            
            # 3. 优化效果对比
            logger.info("步骤 3: 优化效果对比测试")
            test_results['optimization_comparison'] = self.test_optimization_comparison()
            
            # 4. 整体评估
            test_results['overall_assessment'] = self._generate_overall_assessment(test_results)
            
            logger.info("并发优化全面性能测试完成")
            
        except Exception as e:
            logger.error(f"测试过程中发生错误: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    def _generate_overall_assessment(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成整体评估"""
        concurrent_test = test_results.get('concurrent_test', {})
        cache_test = test_results.get('cache_test', {})
        optimization = test_results.get('optimization_comparison', {})
        
        # 评估并发性能
        concurrent_success_rate = concurrent_test.get('success_rate', 0)
        concurrent_grade = 'A' if concurrent_success_rate >= 0.95 else 'B' if concurrent_success_rate >= 0.8 else 'C'
        
        # 评估缓存效果
        cache_improvement = cache_test.get('response_time_improvement', 0)
        cache_grade = 'A' if cache_improvement >= 0.3 else 'B' if cache_improvement >= 0.1 else 'C'
        
        # 评估整体优化效果
        metrics = optimization.get('optimization_metrics', {})
        cache_hit_rate = metrics.get('cache_hit_rate', 0)
        connection_reuse_rate = metrics.get('connection_reuse_rate', 0)
        
        overall_grade = 'A' if (concurrent_grade == 'A' and cache_grade in ['A', 'B']) else 'B'
        
        assessment = {
            'concurrent_performance_grade': concurrent_grade,
            'cache_performance_grade': cache_grade,
            'overall_grade': overall_grade,
            'concurrent_success_rate': concurrent_success_rate,
            'cache_improvement': cache_improvement,
            'cache_hit_rate': cache_hit_rate,
            'connection_reuse_rate': connection_reuse_rate,
            'optimization_successful': overall_grade in ['A', 'B'],
            'recommendations': self._generate_recommendations(concurrent_grade, cache_grade, metrics)
        }
        
        return assessment
    
    def _generate_recommendations(self, concurrent_grade: str, cache_grade: str, 
                                metrics: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if concurrent_grade == 'C':
            recommendations.append("并发性能需要进一步优化，建议增加连接池大小或优化查询语句")
        
        if cache_grade == 'C':
            recommendations.append("缓存效果不明显，建议调整缓存策略或增加缓存大小")
        
        if metrics.get('connection_reuse_rate', 0) < 0.8:
            recommendations.append("连接复用率较低，建议优化连接池配置")
        
        if metrics.get('cache_hit_rate', 0) < 0.5:
            recommendations.append("缓存命中率较低，建议优化缓存键策略或增加缓存时间")
        
        if not recommendations:
            recommendations.append("系统优化效果良好，建议继续监控性能指标")
        
        return recommendations


def main():
    """主函数"""
    print("=" * 80)
    print("并发优化性能测试")
    print("验证连接池和查询缓存优化效果")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 创建测试实例
        test_framework = ConcurrentOptimizationTest()
        
        # 运行全面测试
        results = test_framework.run_comprehensive_test()
        
        # 显示结果摘要
        print("=" * 80)
        print("测试结果摘要")
        print("=" * 80)
        
        if 'error' in results:
            print(f"❌ 测试失败: {results['error']}")
            return 1
        
        # 并发测试结果
        concurrent_test = results.get('concurrent_test', {})
        print(f"🔄 并发测试成功率: {concurrent_test.get('success_rate', 0):.2%}")
        print(f"⚡ 查询吞吐量: {concurrent_test.get('queries_per_second', 0):.1f} 查询/秒")
        print(f"🎯 并发效率: {concurrent_test.get('concurrent_efficiency', 0):.2f}")
        
        # 缓存测试结果
        cache_test = results.get('cache_test', {})
        print(f"📈 缓存性能提升: {cache_test.get('response_time_improvement', 0):.2%}")
        
        # 整体评估
        assessment = results.get('overall_assessment', {})
        print(f"\n🎯 整体评级: {assessment.get('overall_grade', 'N/A')}")
        print(f"✅ 优化成功: {'是' if assessment.get('optimization_successful', False) else '否'}")
        
        # 优化建议
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            print(f"\n💡 优化建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"test_reports/concurrent_optimization_test_{timestamp}.json"
        
        import json
        os.makedirs("test_reports", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📄 详细结果已保存: {output_file}")
        
        # 判断测试结果
        if assessment.get('optimization_successful', False):
            print("\n🎉 并发优化测试通过！系统性能已显著提升。")
            return 0
        else:
            print("\n⚠️ 并发优化效果有限，建议进一步调整优化策略。")
            return 1
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
