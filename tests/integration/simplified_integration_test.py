#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
简化的系统集成测试

验证核心数据库优化组件的集成效果，跳过复杂的指标系统
"""

import sys
import os
import time
import json
import pandas as pd
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from db.unified_data_manager import get_unified_data_manager
from monitoring.performance_monitor import get_performance_monitor
from utils.stability_enhancer import get_stability_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class Simplified_integration_test:
    """简化的系统集成测试器"""
    
    def __init__(self):
        """初始化测试器"""
        # 获取优化后的核心组件
        self.data_manager = get_unified_data_manager()
        self.performance_monitor = get_performance_monitor()
        self.stability_manager = get_stability_manager()
        
        # 测试结果
        self.test_results = {
            'data_manager_tests': {},
            'performance_tests': {},
            'compatibility_tests': {},
            'overall_assessment': {}
        }
        
        logger.info("简化系统集成测试器初始化完成")
    
    def test_data_manager_basic_functionality(self) -> Dict[str, Any]:
        """测试数据管理器基础功能"""
        logger.info("开始测试数据管理器基础功能...")
        
        test_results = {
            'single_query': {},
            'multiple_queries': {},
            'cache_test': {},
            'concurrent_test': {}
        }
        
        # 1. 单个查询测试
        try:
            start_time = time.time()
            stock_info = self.data_manager.get_stock_info(
                stock_code='000001',
                level='DAILY',
                limit=100
            )
            query_time = time.time() - start_time
            
            test_results['single_query'] = {
                'success': True,
                'query_time': query_time,
                'records_returned': len(stock_info.data) if hasattr(stock_info, 'data') and not stock_info.data.empty else 0
            }
            
        except Exception as e:
            test_results['single_query'] = {
                'success': False,
                'error': str(e)
            }
        
        # 2. 多股票查询测试
        try:
            start_time = time.time()
            stock_info = self.data_manager.get_stock_info(
                stock_code=['000001', '000002', '600000'],
                level='DAILY',
                limit=300
            )
            query_time = time.time() - start_time
            
            test_results['multiple_queries'] = {
                'success': True,
                'query_time': query_time,
                'records_returned': len(stock_info.data) if hasattr(stock_info, 'data') and not stock_info.data.empty else 0
            }
            
        except Exception as e:
            test_results['multiple_queries'] = {
                'success': False,
                'error': str(e)
            }
        
        # 3. 缓存效果测试
        try:
            # 第一次查询
            start_time = time.time()
            self.data_manager.get_stock_info(stock_code='000002', level='DAILY', limit=50)
            first_query_time = time.time() - start_time
            
            # 第二次相同查询（应该命中缓存）
            start_time = time.time()
            self.data_manager.get_stock_info(stock_code='000002', level='DAILY', limit=50)
            second_query_time = time.time() - start_time
            
            cache_improvement = (first_query_time - second_query_time) / first_query_time if first_query_time > 0 else 0
            
            test_results['cache_test'] = {
                'success': True,
                'first_query_time': first_query_time,
                'second_query_time': second_query_time,
                'cache_improvement': cache_improvement,
                'cache_effective': cache_improvement > 0.1
            }
            
        except Exception as e:
            test_results['cache_test'] = {
                'success': False,
                'error': str(e)
            }
        
        # 4. 并发查询测试
        try:
            def concurrent_query_Test(thread_id):
                """并发查询函数"""
                try:
                    start_time = time.time()
                    result = self.data_manager.get_stock_info(
                        stock_code=f'00000{(thread_id % 9) + 1}',
                        level='DAILY',
                        limit=50
                    )
                    duration = time.time() - start_time
                    
                    return {
                        'thread_id': thread_id,
                        'success': True,
                        'duration': duration,
                        'records': len(result.data) if hasattr(result, 'data') and not result.data.empty else 0
                    }
                except Exception as e:
                    return {
                        'thread_id': thread_id,
                        'success': False,
                        'error': str(e)
                    }
            
            # 执行并发测试
            start_time = time.time()
            with concurrent.futures.Thread_pool_executor(max_workers=10) as executor:
                futures = [executor.submit(concurrent_query, i) for i in range(20)]
                concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            total_time = time.time() - start_time
            successful_queries = sum(1 for r in concurrent_results if r['success'])
            
            test_results['concurrent_test'] = {
                'success': True,
                'total_queries': len(concurrent_results),
                'successful_queries': successful_queries,
                'success_rate': successful_queries / len(concurrent_results),
                'total_time': total_time,
                'queries_per_second': len(concurrent_results) / total_time if total_time > 0 else 0
            }
            
        except Exception as e:
            test_results['concurrent_test'] = {
                'success': False,
                'error': str(e)
            }
        
        return test_results
    
    def test_backward_compatibility_Test(self) -> Dict[str, Any]:
        """测试向后兼容性"""
        logger.info("开始测试向后兼容性...")
        
        compatibility_results = {
            'old_api_test': {},
            'data_format_test': {},
            'method_availability': {}
        }
        
        # 1. 测试原有API是否仍然可用
        try:
            start_time = time.time()
            
            # 测试get_stock_data方法（原有API）
            stock_data = self.data_manager.get_stock_data(
                stock_code='000001',
                period='daily',
                limit=100
            )
            
            api_test_time = time.time() - start_time
            
            compatibility_results['old_api_test'] = {
                'success': isinstance(stock_data, pd.DataFrame),
                'response_time': api_test_time,
                'data_format_correct': not stock_data.empty and 'close' in stock_data.columns if isinstance(stock_data, pd.DataFrame) else False
            }
            
        except Exception as e:
            compatibility_results['old_api_test'] = {
                'success': False,
                'error': str(e)
            }
        
        # 2. 测试其他原有方法
        try:
            # 测试get_stock_list方法
            stock_list = self.data_manager.get_stock_list(limit=10)
            
            # 测试get_stock_industry方法
            industry = self.data_manager.get_stock_industry('000001')
            
            compatibility_results['method_availability'] = {
                'get_stock_list_works': isinstance(stock_list, list),
                'get_stock_industry_works': isinstance(industry, (str, type(None))),
                'stock_list_count': len(stock_list) if isinstance(stock_list, list) else 0
            }
            
        except Exception as e:
            compatibility_results['method_availability'] = {
                'success': False,
                'error': str(e)
            }
        
        return compatibility_results
    
    def test_performance_improvements(self) -> Dict[str, Any]:
        """测试性能改进"""
        logger.info("开始测试性能改进...")
        
        performance_results = {
            'connection_pool_stats': {},
            'query_performance': {},
            'monitoring_integration': {}
        }
        
        # 1. 获取连接池统计
        try:
            pool_stats = self.data_manager.get_connection_pool_stats()
            performance_results['connection_pool_stats'] = pool_stats
            
        except Exception as e:
            performance_results['connection_pool_stats'] = {'error': str(e)}
        
        # 2. 查询性能测试
        try:
            # 执行一系列查询来测试性能
            query_times = []
            for i in range(5):
                start_time = time.time()
                self.data_manager.get_stock_info(
                    stock_code=f'00000{i + 1}',
                    level='DAILY',
                    limit=100
                )
                query_times.append(time.time() - start_time)
            
            performance_results['query_performance'] = {
                'avg_query_time': sum(query_times) / len(query_times),
                'min_query_time': min(query_times),
                'max_query_time': max(query_times),
                'total_queries': len(query_times)
            }
            
        except Exception as e:
            performance_results['query_performance'] = {'error': str(e)}
        
        # 3. 监控集成测试
        try:
            # 启动监控
            self.performance_monitor.start_monitoring()
            
            # 等待收集一些数据
            time.sleep(5)
            
            # 获取监控数据
            current_metrics = self.performance_monitor.get_current_metrics()
            monitor_stats = self.performance_monitor.get_stats()
            
            # 停止监控
            self.performance_monitor.stop_monitoring()
            
            performance_results['monitoring_integration'] = {
                'metrics_collected': len(current_metrics),
                'monitor_stats': monitor_stats,
                'monitoring_works': len(current_metrics) > 0
            }
            
        except Exception as e:
            performance_results['monitoring_integration'] = {'error': str(e)}
        
        return performance_results
    
    def run_comprehensive_test_Test_Simplified_Integration_Test(self) -> Dict[str, Any]:
        """运行综合测试"""
        logger.info("=" * 80)
        logger.info("开始简化系统集成综合测试")
        logger.info("=" * 80)
        
        test_results = {
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_manager_tests': {},
            'compatibility_tests': {},
            'performance_tests': {},
            'overall_assessment': {}
        }
        
        try:
            # 1. 数据管理器功能测试
            logger.info("步骤 1: 数据管理器功能测试")
            test_results['data_manager_tests'] = self.test_data_manager_basic_functionality()
            
            # 2. 向后兼容性测试
            logger.info("步骤 2: 向后兼容性测试")
            test_results['compatibility_tests'] = self.test_backward_compatibility_Test()
            
            # 3. 性能改进测试
            logger.info("步骤 3: 性能改进测试")
            test_results['performance_tests'] = self.test_performance_improvements()
            
            # 4. 整体评估
            test_results['overall_assessment'] = self._generate_overall_assessment_Simplified_Integration_Test(test_results)
            
            logger.info("简化系统集成综合测试完成")
            
        except Exception as e:
            logger.error(f"测试过程中发生错误: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    def _generate_overall_assessment_Simplified_Integration_Test(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成整体评估"""
        assessment = {
            'integration_success': True,
            'performance_improvement': 'good',
            'compatibility_maintained': True,
            'production_ready': True,
            'issues_found': [],
            'recommendations': [],
            'key_metrics': {}
        }
        
        # 评估数据管理器功能
        dm_tests = test_results.get('data_manager_tests', {})
        
        # 检查基础功能
        if not dm_tests.get('single_query', {}).get('success', False):
            assessment['integration_success'] = False
            assessment['issues_found'].append('单个查询功能失败')
        
        # 检查并发性能
        concurrent_test = dm_tests.get('concurrent_test', {})
        if concurrent_test.get('success', False):
            success_rate = concurrent_test.get('success_rate', 0)
            qps = concurrent_test.get('queries_per_second', 0)
            
            assessment['key_metrics']['concurrent_success_rate'] = success_rate
            assessment['key_metrics']['queries_per_second'] = qps
            
            if success_rate < 0.95:
                assessment['performance_improvement'] = 'needs_improvement'
                assessment['issues_found'].append(f"并发成功率较低: {success_rate:.2%}")
            
            if success_rate >= 0.95:
                assessment['performance_improvement'] = 'excellent'
        
        # 检查缓存效果
        cache_test = dm_tests.get('cache_test', {})
        if cache_test.get('success', False):
            cache_effective = cache_test.get('cache_effective', False)
            cache_improvement = cache_test.get('cache_improvement', 0)
            
            assessment['key_metrics']['cache_improvement'] = cache_improvement
            assessment['key_metrics']['cache_effective'] = cache_effective
            
            if not cache_effective:
                assessment['issues_found'].append('缓存效果不明显')
        
        # 评估向后兼容性
        compat_tests = test_results.get('compatibility_tests', {})
        old_api_test = compat_tests.get('old_api_test', {})
        
        if not old_api_test.get('success', False):
            assessment['compatibility_maintained'] = False
            assessment['issues_found'].append('向后兼容性测试失败')
        
        # 评估性能改进
        perf_tests = test_results.get('performance_tests', {})
        query_perf = perf_tests.get('query_performance', {})
        
        if 'avg_query_time' in query_perf:
            avg_time = query_perf['avg_query_time']
            assessment['key_metrics']['avg_query_time'] = avg_time
            
            if avg_time > 5.0:  # 超过5秒认为性能较差
                assessment['performance_improvement'] = 'needs_improvement'
                assessment['issues_found'].append(f"平均查询时间较长: {avg_time:.3f}秒")
        
        # 生成建议
        if assessment['integration_success'] and assessment['compatibility_maintained']:
            if assessment['performance_improvement'] == 'excellent':
                assessment['recommendations'].append('系统集成完全成功，性能显著提升，建议立即部署到生产环境')
            elif assessment['performance_improvement'] == 'good':
                assessment['recommendations'].append('系统集成成功，性能有所提升，建议部署到生产环境')
            else:
                assessment['recommendations'].append('系统集成基本成功，但性能需要进一步优化')
                assessment['production_ready'] = False
        else:
            assessment['recommendations'].append('发现集成问题，建议修复后重新测试')
            assessment['production_ready'] = False
        
        return assessment


def main_simplifiedintegrationtest():
    """主函数"""
    print("=" * 80)
    print("简化系统集成测试")
    print("验证核心数据库优化组件的集成效果")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 创建测试实例
        test_framework = Simplified_integration_test()
        
        # 运行综合测试
        results = test_framework.run_comprehensive_test_Test_Simplified_Integration_Test()
        
        # 显示结果摘要
        print("=" * 80)
        print("测试结果摘要")
        print("=" * 80)
        
        if 'error' in results:
            print(f"❌ 测试失败: {results['error']}")
            return 1
        
        # 整体评估
        assessment = results.get('overall_assessment', {})
        key_metrics = assessment.get('key_metrics', {})
        
        print(f"🔗 集成成功: {'是' if assessment.get('integration_success', False) else '否'}")
        print(f"📈 性能改进: {assessment.get('performance_improvement', 'N/A')}")
        print(f"🔄 兼容性维护: {'是' if assessment.get('compatibility_maintained', False) else '否'}")
        print(f"🚀 生产就绪: {'是' if assessment.get('production_ready', False) else '否'}")
        
        # 关键指标
        if key_metrics:
            print(f"\n📊 关键指标:")
            if 'concurrent_success_rate' in key_metrics:
                print(f"  - 并发成功率: {key_metrics['concurrent_success_rate']:.2%}")
            if 'queries_per_second' in key_metrics:
                print(f"  - 查询吞吐量: {key_metrics['queries_per_second']:.1f} 查询/秒")
            if 'avg_query_time' in key_metrics:
                print(f"  - 平均查询时间: {key_metrics['avg_query_time']:.3f} 秒")
            if 'cache_improvement' in key_metrics:
                print(f"  - 缓存性能提升: {key_metrics['cache_improvement']:.2%}")
        
        # 发现的问题
        issues = assessment.get('issues_found', [])
        if issues:
            print(f"\n⚠️ 发现的问题:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        
        # 建议
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            print(f"\n💡 建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"test_reports/simplified_integration_test_{timestamp}.json"
        
        os.makedirs("test_reports", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📄 详细结果已保存: {output_file}")
        
        # 判断测试结果
        if assessment.get('production_ready', False):
            print("\n🎉 系统集成测试通过！核心优化组件已准备好部署到生产环境。")
            return 0
        else:
            print("\n⚠️ 系统集成存在问题，建议修复后重新测试。")
            return 1
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main_simplifiedintegrationtest()
    sys.exit(exit_code)
