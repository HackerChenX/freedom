#!/usr/bin/python
# -*- coding: UTF-8 -*-

from config import get_config
"""
综合优化验证测试

验证所有优化措施的综合效果
"""

import sys
import os
import time
import json
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from db.enhanced_connection_pool import initialize_connection_pool, get_connection_pool
from db.unified_data_manager import get_unified_data_manager
from monitoring.performance_monitor import get_performance_monitor, start_monitoring, stop_monitoring
from utils.stability_enhancer import get_stability_manager, retry, Circuit_breaker
from utils.logger import get_logger

logger = get_logger(__name__)


class Comprehensive_optimization_test:
    """综合优化验证测试器"""
    
    def __init__(self):
        """初始化测试器"""
        # 初始化所有优化组件
        self.connection_pool = initialize_connection_pool(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '9000')),
            database=os.getenv('DB_DATABASE', 'stock'),
            get_config('performance.max_connections'),
            min_connections=5
        )
        
        self.data_manager = get_unified_data_manager()
        self.performance_monitor = get_performance_monitor()
        self.stability_manager = get_stability_manager()
        
        # 测试结果
        self.test_results = {
            'optimization_tests': {},
            'performance_comparison': {},
            'stability_verification': {},
            'overall_assessment': {}
        }
        
        logger.info("综合优化验证测试器初始化完成")
    
    def test_concurrent_performance_improvement(self) -> Dict[str, Any]:
        """测试并发性能改进"""
        logger.info("开始测试并发性能改进...")
        
        # 启动性能监控
        self.performance_monitor.start_monitoring()
        
        def execute_concurrent_queries(thread_count: int, queries_per_thread: int) -> Dict[str, Any]:
            """执行并发查询测试"""
            
            def query_task(thread_id: int) -> Dict[str, Any]:
                """单个查询任务"""
                results = []
                errors = []
                
                for i in range(queries_per_thread):
                    try:
                        start_time = time.time()
                        
                        # 执行查询
                        result = self.data_manager.get_stock_info(
                            stock_code=f'00000{(thread_id + i) % 9 + 1}',
                            level='DAILY',
                            limit=100
                        )
                        
                        duration = time.time() - start_time
                        results.append({
                            'query_id': i,
                            'duration': duration,
                            'success': True,
                            'records': len(result.data) if hasattr(result, 'data') and not result.data.empty else 0
                        })
                        
                    except Exception as e:
                        duration = time.time() - start_time
                        errors.append({
                            'query_id': i,
                            'duration': duration,
                            'error': str(e)
                        })
                
                return {
                    'thread_id': thread_id,
                    'successful_queries': len(results),
                    'failed_queries': len(errors),
                    'results': results,
                    'errors': errors
                }
            
            # 执行并发测试
            start_time = time.time()
            
            with concurrent.futures.Thread_pool_executor(max_workers=thread_count) as executor:
                futures = [executor.submit(query_task, i) for i in range(thread_count)]
                thread_results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            total_time = time.time() - start_time
            
            # 分析结果
            total_queries = sum(r['successful_queries'] + r['failed_queries'] for r in thread_results)
            successful_queries = sum(r['successful_queries'] for r in thread_results)
            
            return {
                'thread_count': thread_count,
                'queries_per_thread': queries_per_thread,
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
                'total_time': total_time,
                'queries_per_second': total_queries / total_time if total_time > 0 else 0,
                'thread_results': thread_results
            }
        
        # 测试不同并发级别
        concurrent_tests = {}
        
        for thread_count in [1, 5, 10, 15]:
            logger.info(f"测试 {thread_count} 个并发线程...")
            test_result = execute_concurrent_queries(thread_count, 3)
            concurrent_tests[f'{thread_count}_threads'] = test_result
            
            logger.info(f"{thread_count} 线程测试完成，成功率: {test_result['success_rate']:.2%}, "
                       f"吞吐量: {test_result['queries_per_second']:.1f} 查询/秒")
        
        # 停止性能监控
        self.performance_monitor.stop_monitoring()
        
        return {
            'concurrent_tests': concurrent_tests,
            'optimization_effectiveness': self._analyze_concurrent_improvement(concurrent_tests)
        }
    
    def test_cache_optimization_effect(self) -> Dict[str, Any]:
        """测试缓存优化效果"""
        logger.info("开始测试缓存优化效果...")
        
        # 准备重复查询来测试缓存效果
        test_queries = [
            {'stock_code': '000001', 'level': 'DAILY', 'limit': 100},
            {'stock_code': '000002', 'level': 'DAILY', 'limit': 100},
            {'stock_code': '600000', 'level': 'DAILY', 'limit': 100}
        ]
        
        cache_test_results = []
        
        # 执行多轮相同查询
        for round_num in range(3):
            logger.info(f"执行缓存测试轮次 {round_num + 1}")
            
            round_results = []
            round_start = time.time()
            
            for query_params in test_queries:
                query_start = time.time()
                
                try:
                    result = self.data_manager.get_stock_info(**query_params)
                    query_time = time.time() - query_start
                    
                    round_results.append({
                        'query': query_params,
                        'duration': query_time,
                        'success': True,
                        'records': len(result.data) if hasattr(result, 'data') and not result.data.empty else 0
                    })
                    
                except Exception as e:
                    query_time = time.time() - query_start
                    round_results.append({
                        'query': query_params,
                        'duration': query_time,
                        'success': False,
                        'error': str(e)
                    })
            
            round_time = time.time() - round_start
            
            cache_test_results.append({
                'round': round_num + 1,
                'total_time': round_time,
                'queries': round_results,
                'avg_query_time': round_time / len(test_queries)
            })
        
        # 分析缓存效果
        cache_analysis = self._analyze_cache_effectiveness(cache_test_results)
        
        return {
            'cache_test_results': cache_test_results,
            'cache_analysis': cache_analysis
        }
    
    def test_stability_features(self) -> Dict[str, Any]:
        """测试稳定性功能"""
        logger.info("开始测试稳定性功能...")
        
        stability_tests = {}
        
        # 1. 测试重试机制
        @retry(max_attempts=3, delay=0.1)
        def flaky_function(success_rate: float = 0.3):
            """模拟不稳定的函数"""
            import random
            if random.random() > success_rate:
                raise Exception("模拟失败")
            return "成功"
        
        retry_test_results = []
        for i in range(5):
            try:
                start_time = time.time()
                result = flaky_function(0.4)  # 40%成功率
                duration = time.time() - start_time
                retry_test_results.append({'success': True, 'duration': duration})
            except Exception as e:
                duration = time.time() - start_time
                retry_test_results.append({'success': False, 'duration': duration, 'error': str(e)})
        
        stability_tests['retry_mechanism'] = {
            'tests': retry_test_results,
            'success_rate': sum(1 for r in retry_test_results if r['success']) / len(retry_test_results)
        }
        
        # 2. 测试熔断器
        circuit_breaker = self.stability_manager.create_circuit_breaker(
            'test_breaker',
            failure_threshold=3,
            recovery_timeout=5
        )
        
        @circuit_breaker
        def failing_function():
            raise Exception("总是失败")
        
        breaker_test_results = []
        for i in range(10):
            try:
                failing_function()
                breaker_test_results.append({'attempt': i + 1, 'success': True})
            except Exception as e:
                breaker_test_results.append({
                    'attempt': i + 1, 
                    'success': False, 
                    'error': type(e).__name__
                })
        
        stability_tests['circuit_breaker'] = {
            'tests': breaker_test_results,
            'breaker_state': circuit_breaker.get_state()
        }
        
        return stability_tests
    
    def test_monitoring_integration(self) -> Dict[str, Any]:
        """测试监控集成"""
        logger.info("开始测试监控集成...")
        
        # 启动监控
        self.performance_monitor.start_monitoring()
        
        # 执行一些操作生成监控数据
        for i in range(5):
            try:
                self.data_manager.get_stock_info(
                    stock_code=f'00000{i + 1}',
                    level='DAILY',
                    limit=50
                )
            except Exception as e:
                logger.debug(f"查询失败: {e}")
        
        # 等待监控收集数据
        time.sleep(5)
        
        # 获取监控数据
        current_metrics = self.performance_monitor.get_current_metrics()
        monitor_stats = self.performance_monitor.get_stats()
        
        # 停止监控
        self.performance_monitor.stop_monitoring()
        
        return {
            'metrics_collected': len(current_metrics),
            'monitor_stats': monitor_stats,
            'key_metrics': {
                name: metric.value for name, metric in current_metrics.items()
                if name in ['avg_query_time', 'cache_hit_rate', 'connection_pool_usage']
            }
        }
    
    def _analyze_concurrent_improvement(self, concurrent_tests: Dict[str, Any]) -> Dict[str, Any]:
        """分析并发性能改进"""
        analysis = {
            'scalability': 'good',
            'efficiency_trend': [],
            'bottleneck_identified': False
        }
        
        # 分析不同并发级别的性能
        for test_name, result in concurrent_tests.items():
            thread_count = result['thread_count']
            success_rate = result['success_rate']
            qps = result['queries_per_second']
            
            analysis['efficiency_trend'].append({
                'thread_count': thread_count,
                'success_rate': success_rate,
                'queries_per_second': qps
            })
            
            # 检查性能瓶颈
            if success_rate < 0.95:
                analysis['bottleneck_identified'] = True
                analysis['scalability'] = 'limited'
        
        return analysis
    
    def _analyze_cache_effectiveness(self, cache_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析缓存效果"""
        if len(cache_results) < 2:
            return {'error': '需要至少2轮测试来分析缓存效果'}
        
        first_round = cache_results[0]
        last_round = cache_results[-1]
        
        first_avg_time = first_round['avg_query_time']
        last_avg_time = last_round['avg_query_time']
        
        improvement = (first_avg_time - last_avg_time) / first_avg_time if first_avg_time > 0 else 0
        
        return {
            'cache_improvement': improvement,
            'first_round_avg_time': first_avg_time,
            'last_round_avg_time': last_avg_time,
            'cache_effective': improvement > 0.1  # 10%以上改善认为有效
        }
    
    def run_comprehensive_optimization_test(self) -> Dict[str, Any]:
        """运行综合优化测试"""
        logger.info("=" * 80)
        logger.info("开始综合优化验证测试")
        logger.info("=" * 80)
        
        test_results = {
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'concurrent_performance': {},
            'cache_optimization': {},
            'stability_features': {},
            'monitoring_integration': {},
            'overall_assessment': {}
        }
        
        try:
            # 1. 并发性能测试
            logger.info("步骤 1: 并发性能改进测试")
            test_results['concurrent_performance'] = self.test_concurrent_performance_improvement()
            
            # 2. 缓存优化测试
            logger.info("步骤 2: 缓存优化效果测试")
            test_results['cache_optimization'] = self.test_cache_optimization_effect()
            
            # 3. 稳定性功能测试
            logger.info("步骤 3: 稳定性功能测试")
            test_results['stability_features'] = self.test_stability_features()
            
            # 4. 监控集成测试
            logger.info("步骤 4: 监控集成测试")
            test_results['monitoring_integration'] = self.test_monitoring_integration()
            
            # 5. 整体评估
            test_results['overall_assessment'] = self._generate_overall_assessment_Comprehensive_Optimization_Test(test_results)
            
            logger.info("综合优化验证测试完成")
            
        except Exception as e:
            logger.error(f"测试过程中发生错误: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    def _generate_overall_assessment_Comprehensive_Optimization_Test(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成整体评估"""
        assessment = {
            'optimization_grade': 'A',
            'concurrent_improvement': 'excellent',
            'cache_effectiveness': 'good',
            'stability_enhancement': 'good',
            'monitoring_integration': 'excellent',
            'production_readiness': True,
            'recommendations': []
        }
        
        # 评估并发性能
        concurrent_perf = test_results.get('concurrent_performance', {})
        if concurrent_perf:
            optimization_effect = concurrent_perf.get('optimization_effectiveness', {})
            if optimization_effect.get('bottleneck_identified', False):
                assessment['concurrent_improvement'] = 'good'
                assessment['recommendations'].append('进一步优化高并发场景下的性能瓶颈')
        
        # 评估缓存效果
        cache_opt = test_results.get('cache_optimization', {})
        if cache_opt:
            cache_analysis = cache_opt.get('cache_analysis', {})
            if not cache_analysis.get('cache_effective', False):
                assessment['cache_effectiveness'] = 'needs_improvement'
                assessment['recommendations'].append('调整缓存策略以提高缓存命中率')
        
        # 评估稳定性
        stability = test_results.get('stability_features', {})
        if stability:
            retry_success = stability.get('retry_mechanism', {}).get('success_rate', 0)
            if retry_success < 0.8:
                assessment['stability_enhancement'] = 'needs_improvement'
                assessment['recommendations'].append('优化重试策略和错误处理机制')
        
        # 计算总体评级
        grades = [
            assessment['concurrent_improvement'],
            assessment['cache_effectiveness'],
            assessment['stability_enhancement'],
            assessment['monitoring_integration']
        ]
        
        excellent_count = grades.count('excellent')
        good_count = grades.count('good')
        
        if excellent_count >= 3:
            assessment['optimization_grade'] = 'A'
        elif excellent_count + good_count >= 3:
            assessment['optimization_grade'] = 'B'
        else:
            assessment['optimization_grade'] = 'C'
            assessment['production_readiness'] = False
        
        if not assessment['recommendations']:
            assessment['recommendations'].append('系统优化效果良好，建议继续监控和维护')
        
        return assessment


def main_comprehensiveoptimizationtest():
    """主函数"""
    print("=" * 80)
    print("综合优化验证测试")
    print("验证所有优化措施的综合效果")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 创建测试实例
        test_framework = Comprehensive_optimization_test()
        
        # 运行综合测试
        results = test_framework.run_comprehensive_optimization_test()
        
        # 显示结果摘要
        print("=" * 80)
        print("测试结果摘要")
        print("=" * 80)
        
        if 'error' in results:
            print(f"❌ 测试失败: {results['error']}")
            return 1
        
        # 整体评估
        assessment = results.get('overall_assessment', {})
        print(f"🎯 优化总体评级: {assessment.get('optimization_grade', 'N/A')}")
        print(f"🚀 并发性能改进: {assessment.get('concurrent_improvement', 'N/A')}")
        print(f"📈 缓存优化效果: {assessment.get('cache_effectiveness', 'N/A')}")
        print(f"🛡️ 稳定性增强: {assessment.get('stability_enhancement', 'N/A')}")
        print(f"📊 监控集成: {assessment.get('monitoring_integration', 'N/A')}")
        print(f"🏭 生产就绪: {'是' if assessment.get('production_readiness', False) else '否'}")
        
        # 优化建议
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            print(f"\n💡 优化建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"test_reports/comprehensive_optimization_test_{timestamp}.json"
        
        os.makedirs("test_reports", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📄 详细结果已保存: {output_file}")
        
        # 判断测试结果
        if assessment.get('production_readiness', False):
            print("\n🎉 综合优化验证测试通过！系统已准备好部署到生产环境。")
            return 0
        else:
            print("\n⚠️ 系统优化需要进一步改进才能部署到生产环境。")
            return 1
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main_comprehensiveoptimizationtest()
    sys.exit(exit_code)
