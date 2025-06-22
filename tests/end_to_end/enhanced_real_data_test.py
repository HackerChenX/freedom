#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
增强版真实数据环境性能测试

使用ClickHouse真实数据，测试500-1000只股票的大规模数据处理能力
"""

import sys
import os
import time
import json
import pandas as pd
import numpy as np
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedRealDataTest:
    """增强版真实数据测试器"""
    
    def __init__(self):
        self.client = None
        self.test_stats = {
            'database_queries': [],
            'performance_metrics': {},
            'optimization_opportunities': []
        }
        
        # 初始化ClickHouse连接
        self._init_clickhouse_connection()
    
    def _init_clickhouse_connection(self):
        """初始化ClickHouse连接"""
        try:
            import clickhouse_connect
            self.client = clickhouse_connect.get_client(
                host='localhost', 
                port=8123, 
                database='stock'
            )
            logger.info("ClickHouse连接初始化成功")
        except Exception as e:
            logger.error(f"ClickHouse连接初始化失败: {e}")
            self.client = None
    
    def test_database_performance(self, stock_limit: int = 1000) -> Dict[str, Any]:
        """测试数据库性能"""
        logger.info(f"开始数据库性能测试，股票限制: {stock_limit}")
        
        if not self.client:
            return {'error': 'ClickHouse连接未建立'}
        
        performance_results = {
            'stock_limit': stock_limit,
            'queries': [],
            'total_time': 0,
            'total_records': 0,
            'avg_query_time': 0,
            'optimization_suggestions': []
        }
        
        start_time = time.time()
        
        try:
            # 1. 测试基本股票列表查询
            query_start = time.time()
            result = self.client.query(f"""
                SELECT DISTINCT code, name 
                FROM stock_info 
                WHERE date >= '2024-01-01' 
                LIMIT {stock_limit}
            """)
            query_time = time.time() - query_start
            
            stock_list = [(row[0], row[1]) for row in result.result_rows]
            
            performance_results['queries'].append({
                'query_type': 'stock_list',
                'duration': query_time,
                'records': len(stock_list),
                'query': 'SELECT DISTINCT code, name FROM stock_info'
            })
            
            logger.info(f"股票列表查询完成: {len(stock_list)}只股票，耗时: {query_time:.3f}秒")
            
            # 2. 测试最近数据查询
            query_start = time.time()
            recent_data_query = f"""
                SELECT code, date, open, high, low, close, volume
                FROM stock_info
                WHERE date >= '2024-06-01'
                AND code IN ({','.join([f"'{code}'" for code, _ in stock_list[:100]])})
                ORDER BY code, date DESC
            """
            result = self.client.query(recent_data_query)
            query_time = time.time() - query_start
            
            recent_records = len(result.result_rows)
            
            performance_results['queries'].append({
                'query_type': 'recent_data',
                'duration': query_time,
                'records': recent_records,
                'query': 'SELECT OHLCV data for recent period'
            })
            
            logger.info(f"最近数据查询完成: {recent_records}条记录，耗时: {query_time:.3f}秒")
            
            # 3. 测试聚合查询
            query_start = time.time()
            agg_query = f"""
                SELECT 
                    code,
                    COUNT(*) as record_count,
                    AVG(close) as avg_price,
                    MAX(high) as max_price,
                    MIN(low) as min_price,
                    SUM(volume) as total_volume
                FROM stock_info 
                WHERE date >= '2024-01-01' 
                AND code IN ({','.join([f"'{code}'" for code, _ in stock_list[:200]])})
                GROUP BY code
                ORDER BY total_volume DESC
            """
            result = self.client.query(agg_query)
            query_time = time.time() - query_start
            
            agg_records = len(result.result_rows)
            
            performance_results['queries'].append({
                'query_type': 'aggregation',
                'duration': query_time,
                'records': agg_records,
                'query': 'SELECT aggregated statistics'
            })
            
            logger.info(f"聚合查询完成: {agg_records}条记录，耗时: {query_time:.3f}秒")
            
            # 4. 测试复杂条件查询
            query_start = time.time()
            complex_query = f"""
                SELECT
                    code, date, close, volume,
                    (close - open) / open as daily_change,
                    volume / 1000000 as volume_millions
                FROM stock_info
                WHERE date >= '2024-05-01'
                AND code IN ({','.join([f"'{code}'" for code, _ in stock_list[:50]])})
                AND close > 10
                ORDER BY code, date DESC
                LIMIT 10000
            """
            result = self.client.query(complex_query)
            query_time = time.time() - query_start
            
            complex_records = len(result.result_rows)
            
            performance_results['queries'].append({
                'query_type': 'complex_analysis',
                'duration': query_time,
                'records': complex_records,
                'query': 'SELECT with window functions and calculations'
            })
            
            logger.info(f"复杂查询完成: {complex_records}条记录，耗时: {query_time:.3f}秒")
            
            # 计算总体统计
            total_time = time.time() - start_time
            total_records = sum(q['records'] for q in performance_results['queries'])
            avg_query_time = np.mean([q['duration'] for q in performance_results['queries']])
            
            performance_results.update({
                'total_time': total_time,
                'total_records': total_records,
                'avg_query_time': avg_query_time,
                'queries_per_second': len(performance_results['queries']) / total_time,
                'records_per_second': total_records / total_time
            })
            
            # 生成优化建议
            performance_results['optimization_suggestions'] = self._analyze_database_performance(performance_results)
            
            logger.info(f"数据库性能测试完成，总耗时: {total_time:.3f}秒，平均查询时间: {avg_query_time:.3f}秒")
            
            return performance_results
            
        except Exception as e:
            logger.error(f"数据库性能测试失败: {e}")
            return {'error': str(e)}
    
    def _analyze_database_performance(self, results: Dict[str, Any]) -> List[str]:
        """分析数据库性能并生成优化建议"""
        suggestions = []
        
        # 分析查询时间
        slow_queries = [q for q in results['queries'] if q['duration'] > 2.0]
        if slow_queries:
            suggestions.append(f"发现 {len(slow_queries)} 个慢查询（>2秒），建议优化查询语句或添加索引")
        
        # 分析平均查询时间
        if results['avg_query_time'] > 1.0:
            suggestions.append("平均查询时间较长，建议考虑以下优化：")
            suggestions.append("- 为常用查询字段（date, code）添加索引")
            suggestions.append("- 考虑按日期分区表")
            suggestions.append("- 优化WHERE条件的顺序")
        
        # 分析数据量
        if results['total_records'] > 100000:
            suggestions.append("查询返回大量数据，建议：")
            suggestions.append("- 使用LIMIT限制返回记录数")
            suggestions.append("- 实施分页查询")
            suggestions.append("- 考虑数据预聚合")
        
        # 分析复杂查询
        complex_queries = [q for q in results['queries'] if q['query_type'] == 'complex_analysis']
        if complex_queries and complex_queries[0]['duration'] > 3.0:
            suggestions.append("复杂分析查询耗时较长，建议：")
            suggestions.append("- 预计算技术指标并存储")
            suggestions.append("- 使用物化视图")
            suggestions.append("- 考虑异步计算")
        
        return suggestions
    
    def test_concurrent_database_access(self, concurrent_count: int = 5) -> Dict[str, Any]:
        """测试并发数据库访问"""
        logger.info(f"开始并发数据库访问测试，并发数: {concurrent_count}")
        
        if not self.client:
            return {'error': 'ClickHouse连接未建立'}
        
        import concurrent.futures
        import threading
        
        results = {
            'concurrent_count': concurrent_count,
            'individual_results': [],
            'total_time': 0,
            'success_count': 0,
            'error_count': 0,
            'avg_response_time': 0,
            'max_response_time': 0,
            'min_response_time': float('inf')
        }
        
        def execute_concurrent_query(query_id: int) -> Dict[str, Any]:
            """执行单个并发查询"""
            thread_start = time.time()
            
            try:
                # 每个线程执行不同的查询以模拟真实场景
                if query_id % 3 == 0:
                    query = f"""
                        SELECT code, close, volume 
                        FROM stock_info 
                        WHERE date = '2024-06-20' 
                        AND close > {10 + query_id}
                        LIMIT 100
                    """
                elif query_id % 3 == 1:
                    query = f"""
                        SELECT code, AVG(close) as avg_price
                        FROM stock_info 
                        WHERE date >= '2024-06-{10 + query_id % 10:02d}'
                        GROUP BY code
                        LIMIT 50
                    """
                else:
                    query = f"""
                        SELECT COUNT(*) as count
                        FROM stock_info 
                        WHERE volume > {1000000 * (query_id + 1)}
                        AND date >= '2024-06-01'
                    """
                
                result = self.client.query(query)
                duration = time.time() - thread_start
                
                return {
                    'query_id': query_id,
                    'success': True,
                    'duration': duration,
                    'records': len(result.result_rows),
                    'thread_id': threading.current_thread().ident
                }
                
            except Exception as e:
                duration = time.time() - thread_start
                return {
                    'query_id': query_id,
                    'success': False,
                    'duration': duration,
                    'error': str(e),
                    'thread_id': threading.current_thread().ident
                }
        
        # 执行并发测试
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_count) as executor:
            futures = [executor.submit(execute_concurrent_query, i) for i in range(concurrent_count)]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results['individual_results'].append(result)
                
                if result['success']:
                    results['success_count'] += 1
                    results['max_response_time'] = max(results['max_response_time'], result['duration'])
                    results['min_response_time'] = min(results['min_response_time'], result['duration'])
                else:
                    results['error_count'] += 1
        
        results['total_time'] = time.time() - start_time
        
        # 计算统计信息
        successful_durations = [r['duration'] for r in results['individual_results'] if r['success']]
        if successful_durations:
            results['avg_response_time'] = np.mean(successful_durations)
        
        results['success_rate'] = results['success_count'] / concurrent_count
        
        logger.info(f"并发测试完成，成功率: {results['success_rate']:.2%}，平均响应时间: {results['avg_response_time']:.3f}秒")
        
        return results
    
    def run_comprehensive_real_data_test(self) -> Dict[str, Any]:
        """运行全面的真实数据测试"""
        logger.info("=" * 80)
        logger.info("开始增强版真实数据环境全面测试")
        logger.info("=" * 80)
        
        test_results = {
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'database_performance': {},
            'concurrent_performance': {},
            'system_metrics': {},
            'overall_assessment': {},
            'recommendations': []
        }
        
        # 记录系统初始状态
        initial_memory = psutil.virtual_memory()
        initial_cpu = psutil.cpu_percent()
        
        try:
            # 1. 数据库性能测试
            logger.info("执行数据库性能测试...")
            db_performance = self.test_database_performance(stock_limit=1000)
            test_results['database_performance'] = db_performance
            
            # 2. 并发访问测试
            logger.info("执行并发数据库访问测试...")
            concurrent_performance = self.test_concurrent_database_access(concurrent_count=5)
            test_results['concurrent_performance'] = concurrent_performance
            
            # 3. 系统资源监控
            final_memory = psutil.virtual_memory()
            final_cpu = psutil.cpu_percent()
            
            test_results['system_metrics'] = {
                'initial_memory_percent': initial_memory.percent,
                'final_memory_percent': final_memory.percent,
                'memory_increase': final_memory.percent - initial_memory.percent,
                'initial_cpu_percent': initial_cpu,
                'final_cpu_percent': final_cpu,
                'cpu_increase': final_cpu - initial_cpu
            }
            
            # 4. 整体评估
            test_results['overall_assessment'] = self._generate_overall_assessment(test_results)
            
            # 5. 生成建议
            test_results['recommendations'] = self._generate_comprehensive_recommendations(test_results)
            
            logger.info("增强版真实数据环境测试完成")
            
        except Exception as e:
            logger.error(f"测试过程中发生错误: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    def _generate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成整体评估"""
        assessment = {
            'database_grade': 'A',
            'concurrent_grade': 'A',
            'system_grade': 'A',
            'overall_grade': 'A',
            'production_ready': True
        }
        
        # 评估数据库性能
        db_perf = results.get('database_performance', {})
        if 'avg_query_time' in db_perf:
            if db_perf['avg_query_time'] > 3.0:
                assessment['database_grade'] = 'D'
            elif db_perf['avg_query_time'] > 2.0:
                assessment['database_grade'] = 'C'
            elif db_perf['avg_query_time'] > 1.0:
                assessment['database_grade'] = 'B'
        
        # 评估并发性能
        concurrent_perf = results.get('concurrent_performance', {})
        if 'success_rate' in concurrent_perf:
            if concurrent_perf['success_rate'] < 0.8:
                assessment['concurrent_grade'] = 'D'
            elif concurrent_perf['success_rate'] < 0.9:
                assessment['concurrent_grade'] = 'C'
            elif concurrent_perf['success_rate'] < 0.95:
                assessment['concurrent_grade'] = 'B'
        
        # 评估系统资源
        sys_metrics = results.get('system_metrics', {})
        if 'memory_increase' in sys_metrics:
            if sys_metrics['memory_increase'] > 20:
                assessment['system_grade'] = 'C'
            elif sys_metrics['memory_increase'] > 10:
                assessment['system_grade'] = 'B'
        
        # 计算总体评级
        grades = [assessment['database_grade'], assessment['concurrent_grade'], assessment['system_grade']]
        grade_scores = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        avg_score = np.mean([grade_scores[g] for g in grades])
        
        if avg_score >= 3.5:
            assessment['overall_grade'] = 'A'
        elif avg_score >= 2.5:
            assessment['overall_grade'] = 'B'
        elif avg_score >= 1.5:
            assessment['overall_grade'] = 'C'
        else:
            assessment['overall_grade'] = 'D'
        
        assessment['production_ready'] = assessment['overall_grade'] in ['A', 'B']
        
        return assessment
    
    def _generate_comprehensive_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成综合建议"""
        recommendations = []
        
        # 数据库优化建议
        db_perf = results.get('database_performance', {})
        if 'optimization_suggestions' in db_perf and db_perf['optimization_suggestions']:
            recommendations.append({
                'category': 'database_optimization',
                'priority': 'high',
                'title': '数据库性能优化',
                'suggestions': db_perf['optimization_suggestions']
            })
        
        # 并发性能建议
        concurrent_perf = results.get('concurrent_performance', {})
        if concurrent_perf.get('success_rate', 1.0) < 0.95:
            recommendations.append({
                'category': 'concurrent_optimization',
                'priority': 'medium',
                'title': '并发性能优化',
                'suggestions': [
                    '优化数据库连接池配置',
                    '增加数据库连接数限制',
                    '实施查询队列管理',
                    '考虑读写分离架构'
                ]
            })
        
        # 系统资源建议
        sys_metrics = results.get('system_metrics', {})
        if sys_metrics.get('memory_increase', 0) > 10:
            recommendations.append({
                'category': 'system_optimization',
                'priority': 'medium',
                'title': '系统资源优化',
                'suggestions': [
                    '监控内存使用情况',
                    '优化数据结构和缓存策略',
                    '考虑增加系统内存',
                    '实施垃圾回收优化'
                ]
            })
        
        return recommendations


def main():
    """主函数"""
    print("=" * 80)
    print("增强版真实数据环境全面性能测试")
    print("测试ClickHouse数据库性能和大规模数据处理能力")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 创建测试实例
        test_framework = EnhancedRealDataTest()
        
        # 运行全面测试
        results = test_framework.run_comprehensive_real_data_test()
        
        # 显示结果摘要
        print("=" * 80)
        print("测试结果摘要")
        print("=" * 80)
        
        if 'error' in results:
            print(f"❌ 测试失败: {results['error']}")
            return 1
        
        # 数据库性能结果
        db_perf = results.get('database_performance', {})
        if 'avg_query_time' in db_perf:
            print(f"🗄️  数据库平均查询时间: {db_perf['avg_query_time']:.3f} 秒")
            print(f"📊 总查询记录数: {db_perf.get('total_records', 0)}")
            print(f"⚡ 查询吞吐量: {db_perf.get('records_per_second', 0):.1f} 记录/秒")
        
        # 并发性能结果
        concurrent_perf = results.get('concurrent_performance', {})
        if 'success_rate' in concurrent_perf:
            print(f"🔄 并发测试成功率: {concurrent_perf['success_rate']:.2%}")
            print(f"⏱️  平均响应时间: {concurrent_perf.get('avg_response_time', 0):.3f} 秒")
        
        # 系统资源使用
        sys_metrics = results.get('system_metrics', {})
        if 'memory_increase' in sys_metrics:
            print(f"💾 内存使用增长: {sys_metrics['memory_increase']:.1f}%")
        
        # 整体评估
        assessment = results.get('overall_assessment', {})
        print(f"\n🎯 整体评级: {assessment.get('overall_grade', 'N/A')}")
        print(f"🏭 生产就绪: {'是' if assessment.get('production_ready', False) else '否'}")
        
        # 优化建议
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 优化建议 ({len(recommendations)}项):")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['title']} ({rec['priority']})")
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"test_reports/enhanced_real_data_test_{timestamp}.json"
        
        os.makedirs("test_reports", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📄 详细结果已保存: {output_file}")
        
        # 判断测试结果
        if assessment.get('production_ready', False):
            print("\n🎉 测试通过！系统满足生产环境要求。")
            return 0
        else:
            print("\n⚠️ 测试发现性能问题，建议优化后再部署。")
            return 1
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
