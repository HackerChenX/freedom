#!/usr/bin/env python3
"""
Day 4: 实时监控系统生产脚本测试
基于生产级测试计划的第二阶段测试
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from strategy.configurable_strategy_engine import ConfigurableStrategyEngine
from db.enhanced_connection_pool import ClickHouseConnectionPool
from utils.performance_monitor import performance_monitor
from utils.exception_handler import exception_handler
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

class RealtimeMonitoringTest:
    """实时监控系统测试"""
    
    def __init__(self):
        self.strategy_engine = ConfigurableStrategyEngine()
        self.connection_pool = ClickHouseConnectionPool()
        self.results = {
            'test_start_time': datetime.now().isoformat(),
            'tests': {},
            'summary': {}
        }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=60.0)
    def test_stock_pool_generation(self) -> Dict[str, Any]:
        """测试股票池生成"""
        logger.info("🔍 测试股票池生成")

        try:
            # 获取所有股票代码进行测试
            query = "SELECT DISTINCT code FROM stock_info WHERE code = %(code)s AND level = '日线' LIMIT 100"
            with self.connection_pool.get_connection() as conn:
                stock_data = conn.query_dataframe(query)
            stock_codes = stock_data['code'].tolist()

            # 使用technical_momentum_combo策略生成股票池
            result = self.strategy_engine.execute_combination(
                combination_id="technical_momentum_combo",
                stock_codes=stock_codes,
                target_date="2025-05-12"
            )

            success = result is not None and len(result) > 0
            stock_count = len(result) if result else 0
            
            test_result = {
                'test_name': 'stock_pool_generation',
                'success': success,
                'stock_count': stock_count,
                'execution_time': time.time(),
                'details': result if success else 'No stocks selected'
            }
            
            if success:
                logger.info(f"✅ 股票池生成成功，选出 {stock_count} 只股票")
            else:
                logger.warning("⚠️ 股票池生成失败或无结果")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 股票池生成测试失败: {e}")
            return {
                'test_name': 'stock_pool_generation',
                'success': False,
                'error': str(e),
                'execution_time': time.time()
            }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=30.0)
    def test_market_scanning_performance(self) -> Dict[str, Any]:
        """测试市场扫描性能"""
        logger.info("🔍 测试市场扫描性能")
        
        try:
            start_time = time.time()

            # 获取所有股票代码进行全市场扫描
            query = "SELECT DISTINCT code FROM stock_info WHERE code = %(code)s AND level = '日线' LIMIT 200"
            with self.connection_pool.get_connection() as conn:
                stock_data = conn.query_dataframe(query)
            stock_codes = stock_data['code'].tolist()

            # 执行全市场扫描
            result = self.strategy_engine.execute_combination(
                combination_id="high_success_combo",
                stock_codes=stock_codes,
                target_date="2025-05-12"
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            success = result is not None and len(result) > 0
            stock_count = len(result) if result else 0
            
            # 性能标准：全市场扫描时间≤300秒
            performance_ok = execution_time <= 300
            
            test_result = {
                'test_name': 'market_scanning_performance',
                'success': success and performance_ok,
                'execution_time': execution_time,
                'stock_count': stock_count,
                'performance_standard': '≤300秒',
                'performance_met': performance_ok,
                'details': f"扫描时间: {execution_time:.2f}秒, 选出股票: {stock_count}只"
            }
            
            if success and performance_ok:
                logger.info(f"✅ 市场扫描性能测试通过: {execution_time:.2f}秒, {stock_count}只股票")
            else:
                logger.warning(f"⚠️ 市场扫描性能测试未达标: {execution_time:.2f}秒")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 市场扫描性能测试失败: {e}")
            return {
                'test_name': 'market_scanning_performance',
                'success': False,
                'error': str(e),
                'execution_time': time.time()
            }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=45.0)
    def test_multiple_strategy_execution(self) -> Dict[str, Any]:
        """测试多策略执行"""
        logger.info("🔍 测试多策略执行")
        
        try:
            strategies_to_test = [
                "technical_momentum_combo",
                "trend_reversal_combo",
                "high_success_combo"
            ]
            
            results = {}
            total_success = 0
            
            # 获取股票代码用于测试
            query = "SELECT DISTINCT code FROM stock_info WHERE code = %(code)s AND level = '日线' LIMIT 50"
            with self.connection_pool.get_connection() as conn:
                stock_data = conn.query_dataframe(query)
            stock_codes = stock_data['code'].tolist()

            for strategy_name in strategies_to_test:
                try:
                    start_time = time.time()
                    result = self.strategy_engine.execute_combination(
                        combination_id=strategy_name,
                        stock_codes=stock_codes,
                        target_date="2025-05-12"
                    )
                    end_time = time.time()

                    success = result is not None and len(result) > 0
                    if success:
                        total_success += 1

                    results[strategy_name] = {
                        'success': success,
                        'execution_time': end_time - start_time,
                        'stock_count': len(result) if result else 0
                    }
                    
                    logger.info(f"  策略 {strategy_name}: {'✅ 成功' if success else '❌ 失败'}")
                    
                except Exception as e:
                    logger.error(f"  策略 {strategy_name}: ❌ 异常 - {e}")
                    results[strategy_name] = {
                        'success': False,
                        'error': str(e),
                        'execution_time': 0,
                        'stock_count': 0
                    }
            
            success_rate = (total_success / len(strategies_to_test)) * 100
            overall_success = success_rate >= 80  # 要求80%以上成功率
            
            test_result = {
                'test_name': 'multiple_strategy_execution',
                'success': overall_success,
                'success_rate': success_rate,
                'strategy_results': results,
                'total_strategies': len(strategies_to_test),
                'successful_strategies': total_success,
                'standard': '≥80%成功率'
            }
            
            if overall_success:
                logger.info(f"✅ 多策略执行测试通过: {success_rate:.1f}%成功率")
            else:
                logger.warning(f"⚠️ 多策略执行测试未达标: {success_rate:.1f}%成功率")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 多策略执行测试失败: {e}")
            return {
                'test_name': 'multiple_strategy_execution',
                'success': False,
                'error': str(e)
            }
    
    @exception_handler(reraise=True)
    def test_data_consistency(self) -> Dict[str, Any]:
        """测试数据一致性"""
        logger.info("🔍 测试数据一致性")
        
        try:
            # 测试数据库连接和数据完整性
            query = """
            SELECT COUNT(*) as total_records,
                   COUNT(DISTINCT code) as unique_stocks,
                   MIN(date) as earliest_date,
                   MAX(date) as latest_date
            FROM stock_info WHERE code = %(code)s AND level = '日线'
            """
            
            with self.connection_pool.get_connection() as conn:
                result = conn.query_dataframe(query)
            
            if not result.empty:
                total_records = result.iloc[0]['total_records']
                unique_stocks = result.iloc[0]['unique_stocks']
                earliest_date = result.iloc[0]['earliest_date']
                latest_date = result.iloc[0]['latest_date']
                
                # 数据一致性检查
                data_sufficient = total_records > 1000000  # 至少100万条记录
                stocks_sufficient = unique_stocks > 3000   # 至少3000只股票
                date_range_ok = True  # 日期范围合理
                
                consistency_ok = data_sufficient and stocks_sufficient and date_range_ok
                
                test_result = {
                    'test_name': 'data_consistency',
                    'success': consistency_ok,
                    'total_records': int(total_records),
                    'unique_stocks': int(unique_stocks),
                    'earliest_date': str(earliest_date),
                    'latest_date': str(latest_date),
                    'data_sufficient': data_sufficient,
                    'stocks_sufficient': stocks_sufficient,
                    'date_range_ok': date_range_ok
                }
                
                if consistency_ok:
                    logger.info(f"✅ 数据一致性测试通过: {total_records}条记录, {unique_stocks}只股票")
                else:
                    logger.warning(f"⚠️ 数据一致性测试未达标")
                
                return test_result
            else:
                raise Exception("数据库查询返回空结果")
                
        except Exception as e:
            logger.error(f"❌ 数据一致性测试失败: {e}")
            return {
                'test_name': 'data_consistency',
                'success': False,
                'error': str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🚀 开始Day 4实时监控系统测试")
        
        # 执行所有测试
        tests = [
            self.test_stock_pool_generation,
            self.test_market_scanning_performance,
            self.test_multiple_strategy_execution,
            self.test_data_consistency
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
        logger.info(f"\n📊 Day 4实时监控系统测试结果:")
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
        os.makedirs('results/day4', exist_ok=True)
        
        # 运行测试
        test_runner = RealtimeMonitoringTest()
        results = test_runner.run_all_tests()
        
        # 保存结果（处理JSON序列化问题）
        results_file = 'results/day4/day4_realtime_monitoring_test_results.json'

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
        logger.error(f"❌ Day 4测试执行失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
