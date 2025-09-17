#!/usr/bin/env python3
"""
Day 6: 性能压力和生产级验证测试
基于生产级测试计划的第三阶段测试
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from strategy.configurable_strategy_engine import ConfigurableStrategyEngine
from db.enhanced_connection_pool import ClickHouseConnectionPool
from utils.performance_monitor import performance_monitor
from utils.exception_handler import exception_handler
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

class PerformanceStressTest:
    """性能压力和生产级验证测试"""
    
    def __init__(self):
        self.strategy_engine = ConfigurableStrategyEngine()
        self.connection_pool = ClickHouseConnectionPool()
        self.results = {
            'test_start_time': datetime.now().isoformat(),
            'tests': {},
            'summary': {}
        }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=300.0)
    def test_concurrent_strategy_execution(self) -> Dict[str, Any]:
        """测试并发策略执行"""
        logger.info("🔍 测试并发策略执行")
        
        try:
            # 获取股票代码用于测试
            query = "SELECT DISTINCT code FROM stock_info WHERE code = %(code)s AND level = '日线' LIMIT 100"
            with self.connection_pool.get_connection() as conn:
                stock_data = conn.query_dataframe(query)
            stock_codes = stock_data['code'].tolist()
            
            strategies_to_test = [
                "technical_momentum_combo",
                "trend_reversal_combo",
                "high_success_combo"
            ]
            
            # 并发执行策略
            start_time = time.time()
            results = {}
            
            def execute_strategy(strategy_name):
                try:
                    strategy_start = time.time()
                    result = self.strategy_engine.execute_combination(
                        combination_id=strategy_name,
                        stock_codes=stock_codes,
                        target_date="2024-12-30"
                    )
                    strategy_end = time.time()
                    
                    return {
                        'strategy': strategy_name,
                        'success': result is not None and len(result) > 0,
                        'execution_time': strategy_end - strategy_start,
                        'stock_count': len(result) if result else 0,
                        'result': result
                    }
                except Exception as e:
                    return {
                        'strategy': strategy_name,
                        'success': False,
                        'error': str(e),
                        'execution_time': 0,
                        'stock_count': 0
                    }
            
            # 使用线程池并发执行
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_strategy = {
                    executor.submit(execute_strategy, strategy): strategy 
                    for strategy in strategies_to_test
                }
                
                for future in as_completed(future_to_strategy):
                    strategy_result = future.result()
                    results[strategy_result['strategy']] = strategy_result
                    logger.info(f"  策略 {strategy_result['strategy']}: {'✅ 成功' if strategy_result['success'] else '❌ 失败'}")
            
            end_time = time.time()
            total_execution_time = end_time - start_time
            
            # 计算性能指标
            successful_strategies = sum(1 for r in results.values() if r.get('success', False))
            success_rate = (successful_strategies / len(strategies_to_test)) * 100
            avg_execution_time = sum(r.get('execution_time', 0) for r in results.values()) / len(results)
            
            # 性能标准：并发执行时间≤180秒，成功率≥80%
            performance_ok = total_execution_time <= 180 and success_rate >= 80
            
            test_result = {
                'test_name': 'concurrent_strategy_execution',
                'success': performance_ok,
                'total_execution_time': total_execution_time,
                'average_execution_time': avg_execution_time,
                'success_rate': success_rate,
                'successful_strategies': successful_strategies,
                'total_strategies': len(strategies_to_test),
                'strategy_results': results,
                'performance_standard': '≤180秒, ≥80%成功率'
            }
            
            if performance_ok:
                logger.info(f"✅ 并发策略执行测试通过: {total_execution_time:.2f}秒, {success_rate:.1f}%成功率")
            else:
                logger.warning(f"⚠️ 并发策略执行测试未达标: {total_execution_time:.2f}秒, {success_rate:.1f}%成功率")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 并发策略执行测试失败: {e}")
            return {
                'test_name': 'concurrent_strategy_execution',
                'success': False,
                'error': str(e)
            }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=120.0)
    def test_large_dataset_processing(self) -> Dict[str, Any]:
        """测试大数据集处理"""
        logger.info("🔍 测试大数据集处理")
        
        try:
            # 获取大量股票代码进行测试
            query = "SELECT DISTINCT code FROM stock_info WHERE code = %(code)s AND level = '日线' LIMIT 500"
            with self.connection_pool.get_connection() as conn:
                stock_data = conn.query_dataframe(query)
            stock_codes = stock_data['code'].tolist()
            
            start_time = time.time()
            
            # 使用简单策略处理大数据集
            result = self.strategy_engine.execute_combination(
                combination_id="high_success_combo",
                stock_codes=stock_codes,
                target_date="2024-12-30"
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            success = result is not None
            stock_count = len(result) if result else 0
            processing_rate = len(stock_codes) / execution_time if execution_time > 0 else 0
            
            # 性能标准：处理时间≤120秒，处理速度≥4股票/秒
            performance_ok = execution_time <= 120 and processing_rate >= 4
            
            test_result = {
                'test_name': 'large_dataset_processing',
                'success': performance_ok and success,
                'execution_time': execution_time,
                'input_stock_count': len(stock_codes),
                'output_stock_count': stock_count,
                'processing_rate': processing_rate,
                'performance_standard': '≤120秒, ≥4股票/秒',
                'details': f"处理{len(stock_codes)}只股票，耗时{execution_time:.2f}秒，速度{processing_rate:.2f}股票/秒"
            }
            
            if performance_ok and success:
                logger.info(f"✅ 大数据集处理测试通过: {execution_time:.2f}秒, {processing_rate:.2f}股票/秒")
            else:
                logger.warning(f"⚠️ 大数据集处理测试未达标: {execution_time:.2f}秒, {processing_rate:.2f}股票/秒")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 大数据集处理测试失败: {e}")
            return {
                'test_name': 'large_dataset_processing',
                'success': False,
                'error': str(e)
            }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=60.0)
    def test_database_connection_stress(self) -> Dict[str, Any]:
        """测试数据库连接压力"""
        logger.info("🔍 测试数据库连接压力")
        
        try:
            # 并发数据库查询测试
            def execute_query(query_id):
                try:
                    start_time = time.time()
                    query = f"""
                    SELECT COUNT(*) as count, 
                           AVG(close) as avg_price,
                           MAX(high) as max_high,
                           MIN(low) as min_low
                    FROM stock_info WHERE code = %(code)s AND level = '日线' 
                    AND date >= '2024-12-01' 
                    AND date <= '2024-12-30'
                    LIMIT 1000
                    """
                    
                    with self.connection_pool.get_connection() as conn:
                        result = conn.query_dataframe(query)
                    
                    end_time = time.time()
                    
                    return {
                        'query_id': query_id,
                        'success': not result.empty,
                        'execution_time': end_time - start_time,
                        'record_count': result.iloc[0]['count'] if not result.empty else 0
                    }
                except Exception as e:
                    return {
                        'query_id': query_id,
                        'success': False,
                        'error': str(e),
                        'execution_time': 0
                    }
            
            # 并发执行多个查询
            start_time = time.time()
            query_results = []
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(execute_query, i) for i in range(20)]
                for future in as_completed(futures):
                    query_results.append(future.result())
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 计算性能指标
            successful_queries = sum(1 for r in query_results if r.get('success', False))
            success_rate = (successful_queries / len(query_results)) * 100
            avg_query_time = sum(r.get('execution_time', 0) for r in query_results) / len(query_results)
            
            # 性能标准：总时间≤60秒，成功率≥95%，平均查询时间≤3秒
            performance_ok = total_time <= 60 and success_rate >= 95 and avg_query_time <= 3
            
            test_result = {
                'test_name': 'database_connection_stress',
                'success': performance_ok,
                'total_execution_time': total_time,
                'average_query_time': avg_query_time,
                'success_rate': success_rate,
                'successful_queries': successful_queries,
                'total_queries': len(query_results),
                'query_results': query_results,
                'performance_standard': '≤60秒, ≥95%成功率, ≤3秒/查询'
            }
            
            if performance_ok:
                logger.info(f"✅ 数据库连接压力测试通过: {total_time:.2f}秒, {success_rate:.1f}%成功率, {avg_query_time:.2f}秒/查询")
            else:
                logger.warning(f"⚠️ 数据库连接压力测试未达标: {total_time:.2f}秒, {success_rate:.1f}%成功率, {avg_query_time:.2f}秒/查询")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 数据库连接压力测试失败: {e}")
            return {
                'test_name': 'database_connection_stress',
                'success': False,
                'error': str(e)
            }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=180.0)
    def test_system_resource_usage(self) -> Dict[str, Any]:
        """测试系统资源使用"""
        logger.info("🔍 测试系统资源使用")
        
        try:
            # 获取股票代码用于测试
            query = "SELECT DISTINCT code FROM stock_info WHERE code = %(code)s AND level = '日线' LIMIT 200"
            with self.connection_pool.get_connection() as conn:
                stock_data = conn.query_dataframe(query)
            stock_codes = stock_data['code'].tolist()
            
            # 执行资源密集型操作
            start_time = time.time()
            
            strategies = ["technical_momentum_combo", "trend_reversal_combo"]
            results = []
            
            for strategy in strategies:
                for i in range(3):  # 每个策略执行3次
                    try:
                        result = self.strategy_engine.execute_combination(
                            combination_id=strategy,
                            stock_codes=stock_codes,
                            target_date="2024-12-30"
                        )
                        results.append({
                            'strategy': strategy,
                            'run': i + 1,
                            'success': result is not None and len(result) > 0,
                            'stock_count': len(result) if result else 0
                        })
                    except Exception as e:
                        results.append({
                            'strategy': strategy,
                            'run': i + 1,
                            'success': False,
                            'error': str(e)
                        })
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 计算性能指标
            successful_runs = sum(1 for r in results if r.get('success', False))
            success_rate = (successful_runs / len(results)) * 100
            
            # 性能标准：执行时间≤180秒，成功率≥80%
            performance_ok = execution_time <= 180 and success_rate >= 80
            
            test_result = {
                'test_name': 'system_resource_usage',
                'success': performance_ok,
                'execution_time': execution_time,
                'success_rate': success_rate,
                'successful_runs': successful_runs,
                'total_runs': len(results),
                'run_results': results,
                'performance_standard': '≤180秒, ≥80%成功率'
            }
            
            if performance_ok:
                logger.info(f"✅ 系统资源使用测试通过: {execution_time:.2f}秒, {success_rate:.1f}%成功率")
            else:
                logger.warning(f"⚠️ 系统资源使用测试未达标: {execution_time:.2f}秒, {success_rate:.1f}%成功率")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 系统资源使用测试失败: {e}")
            return {
                'test_name': 'system_resource_usage',
                'success': False,
                'error': str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🚀 开始Day 6性能压力和生产级验证测试")
        
        # 执行所有测试
        tests = [
            self.test_concurrent_strategy_execution,
            self.test_large_dataset_processing,
            self.test_database_connection_stress,
            self.test_system_resource_usage
        ]
        
        for test_func in tests:
            try:
                test_result = test_func()
                self.results['tests'][test_result['test_name']] = test_result
            except Exception as e:
                logger.error(f"测试执行失败: {e}")
                self.results['tests'][test_func.__name__] = {
                    'success': False,
                    'error': str(e)
                }
        
        # 计算总体结果
        total_tests = len(self.results['tests'])
        successful_tests = sum(1 for test in self.results['tests'].values() if test.get('success', False))
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        self.results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'success_rate': success_rate,
            'overall_success': success_rate >= 85,  # 要求85%以上成功率
            'test_end_time': datetime.now().isoformat()
        }
        
        # 输出结果
        logger.info(f"\n📊 Day 6性能压力和生产级验证测试结果:")
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"成功数: {successful_tests}")
        logger.info(f"失败数: {total_tests - successful_tests}")
        logger.info(f"成功率: {success_rate:.1f}%")
        logger.info(f"整体状态: {'✅ PASSED' if self.results['summary']['overall_success'] else '❌ FAILED'}")
        
        return self.results

def main():
    """主函数"""
    try:
        # 创建结果目录
        os.makedirs('results/day6', exist_ok=True)
        
        # 运行测试
        test_runner = PerformanceStressTest()
        results = test_runner.run_all_tests()
        
        # 保存结果（处理JSON序列化问题）
        results_file = 'results/day6/day6_performance_stress_test_results.json'
        
        # 转换不可序列化的对象
        def convert_for_json(obj):
            if isinstance(obj, bool):
                return obj
            elif hasattr(obj, 'isoformat'):  # datetime对象
                return obj.isoformat()
            elif isinstance(obj, (int, float, str, list, dict, type(None))):
                return obj
            else:
                return str(obj)
        
        # 递归处理结果字典
        def clean_results(data):
            if isinstance(data, dict):
                return {k: clean_results(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_results(item) for item in data]
            else:
                return convert_for_json(data)
        
        cleaned_results = clean_results(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n📄 详细结果已保存到: {results_file}")
        
        # 返回退出码
        return 0 if results['summary']['overall_success'] else 1
        
    except Exception as e:
        logger.error(f"❌ Day 6测试执行失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
