#!/usr/bin/env python3
"""
Day 5: 历史回测系统生产脚本测试
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

class HistoricalBacktestTest:
    """历史回测系统测试"""
    
    def __init__(self):
        self.strategy_engine = ConfigurableStrategyEngine()
        self.connection_pool = ClickHouseConnectionPool()
        self.results = {
            'test_start_time': datetime.now().isoformat(),
            'tests': {},
            'summary': {}
        }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=120.0)
    def test_multi_date_consistency(self) -> Dict[str, Any]:
        """测试多日期选股一致性"""
        logger.info("🔍 测试多日期选股一致性")
        
        try:
            test_dates = [
                "2024-12-30",
                "2024-12-27", 
                "2024-12-26",
                "2024-12-25",
                "2024-12-24"
            ]
            
            strategy_name = "technical_momentum_combo"
            date_results = {}
            consistent_stocks = set()
            first_iteration = True
            
            for test_date in test_dates:
                try:
                    # 获取股票代码用于测试
                    query = "SELECT DISTINCT code FROM stock_info WHERE code = %(code)s AND level = '日线' LIMIT 50"
                    with self.connection_pool.get_connection() as conn:
                        stock_data = conn.query_dataframe(query)
                    stock_codes = stock_data['code'].tolist()

                    start_time = time.time()
                    result = self.strategy_engine.execute_combination(
                        combination_id=strategy_name,
                        stock_codes=stock_codes,
                        target_date=test_date
                    )
                    end_time = time.time()
                    
                    if result and len(result) > 0:
                        selected_codes = set([stock.get('code', '') for stock in result])
                        date_results[test_date] = {
                            'success': True,
                            'stock_count': len(selected_codes),
                            'execution_time': end_time - start_time,
                            'selected_codes': list(selected_codes)
                        }
                        
                        if first_iteration:
                            consistent_stocks = selected_codes.copy()
                            first_iteration = False
                        else:
                            consistent_stocks = consistent_stocks.intersection(selected_codes)
                        
                        logger.info(f"  {test_date}: ✅ 成功选出 {len(selected_codes)} 只股票")
                    else:
                        date_results[test_date] = {
                            'success': False,
                            'stock_count': 0,
                            'execution_time': end_time - start_time,
                            'error': '无选股结果'
                        }
                        logger.warning(f"  {test_date}: ⚠️ 无选股结果")
                        
                except Exception as e:
                    logger.error(f"  {test_date}: ❌ 异常 - {e}")
                    date_results[test_date] = {
                        'success': False,
                        'error': str(e),
                        'execution_time': 0,
                        'stock_count': 0
                    }
            
            # 计算一致性指标
            successful_dates = sum(1 for result in date_results.values() if result.get('success', False))
            success_rate = (successful_dates / len(test_dates)) * 100
            consistency_rate = len(consistent_stocks) / max(1, max([r.get('stock_count', 0) for r in date_results.values()])) * 100
            
            # 一致性标准：成功率≥80%，一致性≥30%
            consistency_ok = success_rate >= 80 and consistency_rate >= 30
            
            test_result = {
                'test_name': 'multi_date_consistency',
                'success': consistency_ok,
                'success_rate': success_rate,
                'consistency_rate': consistency_rate,
                'consistent_stocks_count': len(consistent_stocks),
                'consistent_stocks': list(consistent_stocks),
                'date_results': date_results,
                'total_dates': len(test_dates),
                'successful_dates': successful_dates
            }
            
            if consistency_ok:
                logger.info(f"✅ 多日期一致性测试通过: {success_rate:.1f}%成功率, {consistency_rate:.1f}%一致性")
            else:
                logger.warning(f"⚠️ 多日期一致性测试未达标: {success_rate:.1f}%成功率, {consistency_rate:.1f}%一致性")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 多日期一致性测试失败: {e}")
            return {
                'test_name': 'multi_date_consistency',
                'success': False,
                'error': str(e)
            }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=90.0)
    def test_strategy_stability(self) -> Dict[str, Any]:
        """测试策略稳定性"""
        logger.info("🔍 测试策略稳定性")
        
        try:
            strategies_to_test = [
                "technical_momentum_combo",
                "trend_reversal_combo"
            ]
            
            test_date = "2024-12-30"
            stability_results = {}
            
            # 获取股票代码用于测试
            query = "SELECT DISTINCT code FROM stock_info WHERE code = %(code)s AND level = '日线' LIMIT 30"
            with self.connection_pool.get_connection() as conn:
                stock_data = conn.query_dataframe(query)
            stock_codes = stock_data['code'].tolist()

            for strategy_name in strategies_to_test:
                strategy_results = []

                # 对每个策略执行3次，测试结果稳定性
                for run_num in range(3):
                    try:
                        start_time = time.time()
                        result = self.strategy_engine.execute_combination(
                            combination_id=strategy_name,
                            stock_codes=stock_codes,
                            target_date=test_date
                        )
                        end_time = time.time()

                        if result and len(result) > 0:
                            selected_codes = [stock.get('code', '') for stock in result]
                            strategy_results.append({
                                'run': run_num + 1,
                                'success': True,
                                'stock_count': len(selected_codes),
                                'execution_time': end_time - start_time,
                                'selected_codes': selected_codes
                            })
                        else:
                            strategy_results.append({
                                'run': run_num + 1,
                                'success': False,
                                'stock_count': 0,
                                'execution_time': end_time - start_time
                            })
                            
                    except Exception as e:
                        strategy_results.append({
                            'run': run_num + 1,
                            'success': False,
                            'error': str(e),
                            'execution_time': 0
                        })
                
                # 分析稳定性
                successful_runs = sum(1 for r in strategy_results if r.get('success', False))
                stability_rate = (successful_runs / len(strategy_results)) * 100
                
                # 检查结果一致性
                if successful_runs >= 2:
                    all_codes = [set(r.get('selected_codes', [])) for r in strategy_results if r.get('success', False)]
                    if len(all_codes) >= 2:
                        intersection = set.intersection(*all_codes)
                        union = set.union(*all_codes)
                        result_consistency = len(intersection) / len(union) * 100 if union else 0
                    else:
                        result_consistency = 0
                else:
                    result_consistency = 0
                
                stability_results[strategy_name] = {
                    'stability_rate': stability_rate,
                    'result_consistency': result_consistency,
                    'successful_runs': successful_runs,
                    'total_runs': len(strategy_results),
                    'run_details': strategy_results
                }
                
                logger.info(f"  策略 {strategy_name}: {stability_rate:.1f}%稳定性, {result_consistency:.1f}%一致性")
            
            # 计算整体稳定性
            avg_stability = sum(r['stability_rate'] for r in stability_results.values()) / len(stability_results)
            avg_consistency = sum(r['result_consistency'] for r in stability_results.values()) / len(stability_results)
            
            # 稳定性标准：平均稳定性≥80%，平均一致性≥60%
            stability_ok = avg_stability >= 80 and avg_consistency >= 60
            
            test_result = {
                'test_name': 'strategy_stability',
                'success': stability_ok,
                'average_stability': avg_stability,
                'average_consistency': avg_consistency,
                'strategy_results': stability_results,
                'standard': '≥80%稳定性, ≥60%一致性'
            }
            
            if stability_ok:
                logger.info(f"✅ 策略稳定性测试通过: {avg_stability:.1f}%稳定性, {avg_consistency:.1f}%一致性")
            else:
                logger.warning(f"⚠️ 策略稳定性测试未达标: {avg_stability:.1f}%稳定性, {avg_consistency:.1f}%一致性")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 策略稳定性测试失败: {e}")
            return {
                'test_name': 'strategy_stability',
                'success': False,
                'error': str(e)
            }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=60.0)
    def test_historical_data_coverage(self) -> Dict[str, Any]:
        """测试历史数据覆盖度"""
        logger.info("🔍 测试历史数据覆盖度")
        
        try:
            # 检查不同时间段的数据可用性
            test_periods = [
                ("2024-12-30", "2024年末"),
                ("2024-06-30", "2024年中"),
                ("2024-01-31", "2024年初"),
                ("2023-12-29", "2023年末")
            ]
            
            coverage_results = {}
            
            for test_date, period_name in test_periods:
                try:
                    # 检查该日期的数据可用性
                    query = f"""
                    SELECT COUNT(DISTINCT code) as stock_count,
                           COUNT(*) as record_count
                    FROM stock_info WHERE code = %(code)s AND level = %(level)s AND date = '{test_date}'
                    AND level = '日线'
                    """
                    
                    with self.connection_pool.get_connection() as conn:
                        result = conn.query_dataframe(query)
                    
                    if not result.empty:
                        stock_count = result.iloc[0]['stock_count']
                        record_count = result.iloc[0]['record_count']
                        
                        # 数据充足性检查
                        data_sufficient = stock_count >= 3000 and record_count >= 3000
                        
                        coverage_results[test_date] = {
                            'period_name': period_name,
                            'stock_count': int(stock_count),
                            'record_count': int(record_count),
                            'data_sufficient': data_sufficient,
                            'success': True
                        }
                        
                        logger.info(f"  {period_name} ({test_date}): {stock_count}只股票, {record_count}条记录")
                    else:
                        coverage_results[test_date] = {
                            'period_name': period_name,
                            'success': False,
                            'error': '无数据'
                        }
                        logger.warning(f"  {period_name} ({test_date}): ⚠️ 无数据")
                        
                except Exception as e:
                    coverage_results[test_date] = {
                        'period_name': period_name,
                        'success': False,
                        'error': str(e)
                    }
                    logger.error(f"  {period_name} ({test_date}): ❌ 异常 - {e}")
            
            # 计算覆盖度
            successful_periods = sum(1 for r in coverage_results.values() if r.get('success', False))
            sufficient_periods = sum(1 for r in coverage_results.values() if r.get('data_sufficient', False))
            
            coverage_rate = (successful_periods / len(test_periods)) * 100
            sufficiency_rate = (sufficient_periods / len(test_periods)) * 100
            
            # 覆盖度标准：≥75%的时间段有数据，≥50%的时间段数据充足
            coverage_ok = coverage_rate >= 75 and sufficiency_rate >= 50
            
            test_result = {
                'test_name': 'historical_data_coverage',
                'success': coverage_ok,
                'coverage_rate': coverage_rate,
                'sufficiency_rate': sufficiency_rate,
                'successful_periods': successful_periods,
                'sufficient_periods': sufficient_periods,
                'total_periods': len(test_periods),
                'period_results': coverage_results
            }
            
            if coverage_ok:
                logger.info(f"✅ 历史数据覆盖度测试通过: {coverage_rate:.1f}%覆盖率, {sufficiency_rate:.1f}%充足率")
            else:
                logger.warning(f"⚠️ 历史数据覆盖度测试未达标: {coverage_rate:.1f}%覆盖率, {sufficiency_rate:.1f}%充足率")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ 历史数据覆盖度测试失败: {e}")
            return {
                'test_name': 'historical_data_coverage',
                'success': False,
                'error': str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🚀 开始Day 5历史回测系统测试")
        
        # 执行所有测试
        tests = [
            self.test_multi_date_consistency,
            self.test_strategy_stability,
            self.test_historical_data_coverage
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
            'overall_success': success_rate >= 80,  # 要求80%以上成功率
            'test_end_time': datetime.now().isoformat()
        }
        
        # 输出结果
        logger.info(f"\n📊 Day 5历史回测系统测试结果:")
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
        os.makedirs('results/day5', exist_ok=True)
        
        # 运行测试
        test_runner = HistoricalBacktestTest()
        results = test_runner.run_all_tests()
        
        # 保存结果（处理JSON序列化问题）
        results_file = 'results/day5/day5_historical_backtest_test_results.json'

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
        logger.error(f"❌ Day 5测试执行失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
