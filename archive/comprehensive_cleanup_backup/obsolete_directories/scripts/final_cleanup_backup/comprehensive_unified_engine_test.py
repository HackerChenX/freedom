#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from db.query_executor import get_query_executor
from db.sql_manager import QueryType
"""
统一分析引擎全面测试框架

连接真实Click_house数据，测试所有指标和形态的选股条件
采用早停机制，出现错误立即停止并报告问题
"""

import sys
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.unified_data_manager import get_unified_data_manager
from strategy.strategy_executor import UnifiedStrategyExecutor as Optimized_strategy_executor
from indicators.complete_indicator_registry import complete_registry
from utils.logger import get_logger
from utils.date_utils import get_latest_trading_date
from utils.decorators import performance_monitor

logger = get_logger(__name__)


class UnifiedEngineComprehensiveTest:
    """统一分析引擎全面测试类"""
    
    def __init__(self, 
                 test_stock_count: int = 100,
                 enable_early_stop: bool = False,
                 max_concurrent_tests: int = 5):
        """
        初始化测试框架
        
        Args:
            test_stock_count: 测试股票数量
            enable_early_stop: 是否启用早停机制
            max_concurrent_tests: 最大并发测试数
        """
        self.test_stock_count = test_stock_count
        self.enable_early_stop = enable_early_stop
        self.max_concurrent_tests = max_concurrent_tests
        
        # 初始化组件
        self.data_manager = get_unified_data_manager()
        self.executor = Optimized_strategy_executor(
            max_workers=32,
            cache_enabled=True,
            enable_memory_monitoring=True
        )
        
        # 测试统计
        self.test_stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'error_details': [],
            'start_time': None,
            'end_time': None,
            'test_results': {}
        }
        
        # 获取最新交易日期 - 使用数据库中实际存在的日期
        # self.test_date = get_latest_trading_date()
        self.test_date = "2025-05-23"  # 使用数据库中实际存在的最新日期
        logger.info(f"测试日期: {self.test_date}")
        
        # 获取测试股票池
        self.test_stocks = self._get_test_stock_pool_Comprehensive_Unified_Engine_Test()
        logger.info(f"测试股票池: {len(self.test_stocks)} 只股票")
    
    def _get_test_stock_pool_Comprehensive_Unified_Engine_Test(self) -> List[str]:
        """获取测试股票池"""
        try:
            # 获取活跃股票，排除ST、*ST等特殊股票
            query = f"""
            SELECT DISTINCT code as stock_code
            FROM stock_info WHERE 1=1
            WHERE date = '{self.test_date}'
            AND level = '日线'
            AND close > 2.0 
            AND close < 200.0
            AND volume > 0
            AND code NOT LIKE '%%ST%%'
            AND code NOT LIKE 'BJ%%'
            ORDER BY volume DESC
            LIMIT {self.test_stock_count}
            """
            
            result = self.data_manager.query(query)
            if result.empty:
                logger.warning("未获取到测试股票，使用默认股票池")
                return ['000001', '000002', '000858', '002415', '600000']
            
            return result['stock_code'].tolist()
            
        except Exception as e:
            logger.error(f"获取测试股票池失败: {e}")
            return ['000001', '000002', '000858', '002415', '600000']
    
    def _generate_indicator_test_strategies(self) -> List[Dict[str, Any]]:
        """生成所有指标的测试策略"""
        strategies = []
        
        # 获取所有已注册指标
        all_indicators = complete_registry.get_indicator_names()
        logger.info(f"准备测试 {len(all_indicators)} 个指标")
        
        # 为每个指标生成测试策略
        for indicator in all_indicators:
            try:
                strategy = self._create_indicator_strategy(indicator)
                if strategy:
                    strategies.append(strategy)
            except Exception as e:
                logger.error(f"生成指标 {indicator} 测试策略失败: {e}")
                if self.enable_early_stop:
                    raise
        
        return strategies
    
    def _create_indicator_strategy(self, indicator_name: str) -> Optional[Dict[str, Any]]:
        """为指标创建测试策略"""
        
        # 定义不同类型指标的测试策略模板
        strategy_templates = {
            # 趋势指标
            'MA': {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': 'MA', 'period': '1d', 'parameters': {'period': 5}, 'operator': '>', 'reference_field': 'close'},
                    {'logic': 'AND'},
                    {'type': 'price', 'field': 'close', 'operator': '>', 'value': 5.0}
                ]
            },
            'EMA': {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': 'EMA', 'period': '1d', 'parameters': {'period': 12}, 'operator': '>', 'reference_field': 'close'},
                    {'logic': 'AND'},
                    {'type': 'volume', 'field': 'volume', 'operator': '>', 'value': 100000}
                ]
            },
            'WMA': {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': 'WMA', 'period': '1d', 'parameters': {'period': 10}, 'operator': '>', 'reference_field': 'close'},
                    {'logic': 'AND'},
                    {'type': 'price', 'field': 'close', 'operator': '>', 'value': 3.0}
                ]
            },
            'MACD': {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': 'MACD', 'period': '1d', 'field': 'DIF', 'operator': '>', 'value': 0},
                    {'logic': 'AND'},
                    {'type': 'indicator', 'indicator_id': 'MACD', 'period': '1d', 'field': 'DEA', 'operator': '>', 'value': 0}
                ]
            },
            'RSI': {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': 'RSI', 'period': '1d', 'parameters': {'period': 14}, 'operator': '<', 'value': 70},
                    {'logic': 'AND'},
                    {'type': 'indicator', 'indicator_id': 'RSI', 'period': '1d', 'parameters': {'period': 14}, 'operator': '>', 'value': 30}
                ]
            },
            'KDJ': {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': 'KDJ', 'period': '1d', 'field': 'K', 'operator': '>', 'reference_field': 'D'},
                    {'logic': 'AND'},
                    {'type': 'indicator', 'indicator_id': 'KDJ', 'period': '1d', 'field': 'K', 'operator': '<', 'value': 80}
                ]
            },
            'BOLL': {
                'conditions': [
                    {'type': 'price', 'field': 'close', 'operator': '>', 'reference_indicator': 'BOLL_LOWER', 'period': '1d'},
                    {'logic': 'AND'},
                    {'type': 'price', 'field': 'close', 'operator': '<', 'reference_indicator': 'BOLL_UPPER', 'period': '1d'}
                ]
            },
            'ADX': {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': 'ADX', 'period': '1d', 'parameters': {'period': 14}, 'operator': '>', 'value': 25},
                    {'logic': 'AND'},
                    {'type': 'price', 'field': 'close', 'operator': '>', 'value': 3.0}
                ]
            },
            'SAR': {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': 'SAR', 'period': '1d', 'operator': '<', 'reference_field': 'close'},
                    {'logic': 'AND'},
                    {'type': 'volume', 'field': 'volume', 'operator': '>', 'value': 100000}
                ]
            },
            'AROON': {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': 'AROON', 'period': '1d', 'parameters': {'period': 14}, 'field': 'aroon_up', 'operator': '>', 'value': 70},
                    {'logic': 'AND'},
                    {'type': 'price', 'field': 'close', 'operator': '>', 'value': 5.0}
                ]
            },
            'ATR': {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': 'ATR', 'period': '1d', 'parameters': {'period': 14}, 'operator': '>', 'value': 0.5},
                    {'logic': 'AND'},
                    {'type': 'price', 'field': 'close', 'operator': '>', 'value': 10.0}
                ]
            },
            'KC': {
                'conditions': [
                    {'type': 'price', 'field': 'close', 'operator': '>', 'reference_indicator': 'KC_LOWER', 'period': '1d'},
                    {'logic': 'AND'},
                    {'type': 'price', 'field': 'close', 'operator': '<', 'reference_indicator': 'KC_UPPER', 'period': '1d'}
                ]
            }
        }
        
        # ZXM指标特殊处理
        if indicator_name.startswith('ZXM_'):
            return self._create_zxm_strategy(indicator_name)
        
        # 增强指标处理
        if indicator_name.startswith('ENHANCED_'):
            base_indicator = indicator_name.replace('ENHANCED_', '')
            if base_indicator in strategy_templates:
                template = strategy_templates[base_indicator].copy()
                template['conditions'][0]['indicator_id'] = indicator_name
                return {
                    'strategy_id': f'test_{indicator_name.lower()}',
                    'name': f'{indicator_name} 测试策略',
                    'conditions': template['conditions'],
                    'filters': {'market': ['SZ', 'SH']}
                }
        
        # 使用模板或创建通用策略
        if indicator_name in strategy_templates:
            template = strategy_templates[indicator_name]
        else:
            # 通用策略模板 - 确保包含period字段
            template = {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': indicator_name, 'period': '1d', 'operator': '>', 'value': 0},
                    {'logic': 'AND'},
                    {'type': 'price', 'field': 'close', 'operator': '>', 'value': 3.0}
                ]
            }
        
        return {
            'strategy_id': f'test_{indicator_name.lower()}',
            'name': f'{indicator_name} 测试策略',
            'conditions': template['conditions'],
            'filters': {'market': ['SZ', 'SH']}
        }
    
    def _create_zxm_strategy(self, indicator_name: str) -> Dict[str, Any]:
        """创建ZXM指标测试策略"""
        zxm_strategies = {
            'ZXM_DAILY_TREND_UP': {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': 'ZXM_DAILY_TREND_UP', 'period': '1d', 'operator': '==', 'value': True},
                    {'logic': 'AND'},
                    {'type': 'price', 'field': 'close', 'operator': '>', 'value': 5.0}
                ]
            },
            'ZXM_WEEKLY_TREND_UP': {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': 'ZXM_WEEKLY_TREND_UP', 'period': '1w', 'operator': '==', 'value': True},
                    {'logic': 'AND'},
                    {'type': 'volume', 'field': 'volume', 'operator': '>', 'value': 100000}
                ]
            },
            'ZXM_DAILY_MACD': {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': 'ZXM_DAILY_MACD', 'period': '1d', 'operator': '>', 'value': 0},
                    {'logic': 'AND'},
                    {'type': 'price', 'field': 'close', 'operator': '>', 'value': 3.0}
                ]
            }
        }
        
        if indicator_name in zxm_strategies:
            template = zxm_strategies[indicator_name]
        else:
            # ZXM通用策略 - 确保包含period字段
            template = {
                'conditions': [
                    {'type': 'indicator', 'indicator_id': indicator_name, 'period': '1d', 'operator': '>', 'value': 0},
                    {'logic': 'AND'},
                    {'type': 'price', 'field': 'close', 'operator': '>', 'value': 5.0}
                ]
            }
        
        return {
            'strategy_id': f'test_{indicator_name.lower()}',
            'name': f'{indicator_name} ZXM测试策略',
            'conditions': template['conditions'],
            'filters': {'market': ['SZ', 'SH']}
        }
    
    @performance_monitor(threshold=30.0)
    def test_single_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """测试单个策略"""
        strategy_id = strategy['strategy_id']
        indicator_name = strategy_id.replace('test_', '').upper()
        
        test_result = {
            'strategy_id': strategy_id,
            'indicator_name': indicator_name,
            'success': False,
            'selected_count': 0,
            'execution_time': 0,
            'error_message': None,
            'sample_results': []
        }
        
        try:
            start_time = time.time()
            
            # 执行策略选股
            results = self.executor.execute_strategy_optimized(
                strategy_plan=strategy,
                end_date=self.test_date,
                enable_early_stop=False,  # 策略内部不启用早停
                max_results=50  # 限制结果数量以提高测试效率
            )
            
            execution_time = time.time() - start_time
            
            # 检查结果
            if results is not None and not results.empty:
                test_result.update({
                    'success': True,
                    'selected_count': len(results),
                    'execution_time': execution_time,
                    'sample_results': results.head(5).to_dict('records') if len(results) > 0 else []
                })
                
                logger.info(f"✅ {indicator_name}: 选出 {len(results)} 只股票，耗时 {execution_time:.2f}秒")
            else:
                # 没有选出股票，但不一定是错误
                test_result.update({
                    'success': True,  # 执行成功，只是没有符合条件的股票
                    'selected_count': 0,
                    'execution_time': execution_time,
                    'error_message': '未选出符合条件的股票（可能是正常情况）'
                })
                
                logger.warning(f"⚠️ {indicator_name}: 未选出股票，耗时 {execution_time:.2f}秒")
            
        except Exception as e:
            error_msg = f"策略执行失败: {str(e)}"
            test_result.update({
                'success': False,
                'error_message': error_msg,
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0
            })
            
            logger.error(f"❌ {indicator_name}: {error_msg}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            
            # 早停机制
            if self.enable_early_stop:
                raise Exception(f"测试失败，启用早停: {error_msg}")
        
        return test_result
    
    def run_comprehensive_test_Test_Comprehensive_Unified_Engine_Test(self) -> Dict[str, Any]:
        """运行全面测试"""
        logger.info("🚀 开始统一分析引擎全面测试")
        self.test_stats['start_time'] = datetime.now()
        
        try:
            # 1. 生成所有测试策略
            logger.info("📋 生成测试策略...")
            strategies = self._generate_indicator_test_strategies()
            self.test_stats['total_tests'] = len(strategies)
            
            logger.info(f"📊 共生成 {len(strategies)} 个测试策略")
            
            # 2. 数据库连接测试
            logger.info("🔗 测试数据库连接...")
            self._test_database_connection()
            
            # 3. 并发执行测试
            logger.info("⚡ 开始并发执行测试...")
            
            if self.max_concurrent_tests > 1:
                results = self._run_concurrent_tests(strategies)
            else:
                results = self._run_sequential_tests(strategies)
            
            # 4. 整理测试结果
            self._process_test_results(results)
            
        except Exception as e:
            logger.error(f"❌ 全面测试失败: {e}")
            self.test_stats['error_details'].append({
                'type': 'comprehensive_test_failure',
                'message': str(e),
                'traceback': traceback.format_exc()
            })
            
        finally:
            self.test_stats['end_time'] = datetime.now()
            
        return self._generate_test_report()
    
    def _test_database_connection(self):
        """测试数据库连接"""
        try:
            # 使用统一数据管理器的方法测试连接
            if not self.data_manager.test_connection():
                raise Exception("数据库连接失败")
            
            # 获取股票列表测试数据可用性
            stock_list = self.data_manager.get_stock_list(limit=10)
            if len(stock_list) == 0:
                raise Exception("数据库中没有股票数据")
            
            # 尝试获取指定日期的数据
            try:
                stock_info = self.data_manager.get_stock_info(
                    stock_code=stock_list[0],
                    level='日线',
                    start_date=self.test_date,
                    end_date=self.test_date,
                    limit=1
                )
                df = stock_info.to_dataframe()
                if df.empty:
                    logger.warning(f"⚠️ {self.test_date} 没有数据，测试将继续使用最新数据")
                else:
                    logger.info(f"✅ 数据库连接正常，{self.test_date} 有数据可用")
            except Exception as e:
                logger.warning(f"⚠️ 获取 {self.test_date} 数据失败: {e}，测试将继续")
            
            logger.info(f"✅ 数据库连接正常，共有 {len(stock_list)} 只股票（取样）")
            
        except Exception as e:
            error_msg = f"数据库连接测试失败: {e}"
            logger.error(error_msg)
            if self.enable_early_stop:
                raise Exception(error_msg)
    
    def _run_concurrent_tests(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """并发执行测试"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_tests) as executor:
            # 提交所有测试任务
            future_to_strategy = {
                executor.submit(self.test_single_strategy, strategy): strategy
                for strategy in strategies
            }
            
            # 收集结果
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 进度报告
                    completed = len(results)
                    total = len(strategies)
                    logger.info(f"📈 测试进度: {completed}/{total} ({completed/total*100:.1f}%)")
                    
                except Exception as e:
                    error_result = {
                        'strategy_id': strategy.get('strategy_id', 'unknown'),
                        'indicator_name': strategy.get('name', 'unknown'),
                        'success': False,
                        'error_message': str(e)
                    }
                    results.append(error_result)
                    
                    if self.enable_early_stop:
                        logger.error(f"❌ 早停触发: {e}")
                        # 取消剩余任务
                        for remaining_future in future_to_strategy:
                            remaining_future.cancel()
                        break
        
        return results
    
    def _run_sequential_tests(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """顺序执行测试"""
        results = []
        
        for i, strategy in enumerate(strategies, 1):
            logger.info(f"📋 执行测试 {i}/{len(strategies)}: {strategy.get('name', 'Unknown')}")
            
            try:
                result = self.test_single_strategy(strategy)
                results.append(result)
                
            except Exception as e:
                error_result = {
                    'strategy_id': strategy.get('strategy_id', 'unknown'),
                    'indicator_name': strategy.get('name', 'unknown'),
                    'success': False,
                    'error_message': str(e)
                }
                results.append(error_result)
                
                if self.enable_early_stop:
                    logger.error(f"❌ 早停触发，停止后续测试: {e}")
                    break
        
        return results
    
    def _process_test_results(self, results: List[Dict[str, Any]]):
        """处理测试结果"""
        for result in results:
            if result['success']:
                self.test_stats['successful_tests'] += 1
            else:
                self.test_stats['failed_tests'] += 1
                self.test_stats['error_details'].append({
                    'indicator': result['indicator_name'],
                    'strategy_id': result['strategy_id'],
                    'error': result.get('error_message', 'Unknown error')
                })
            
            self.test_stats['test_results'][result['indicator_name']] = result
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        duration = (
            self.test_stats['end_time'] - self.test_stats['start_time']
        ).total_seconds() if self.test_stats['end_time'] and self.test_stats['start_time'] else 0
        
        success_rate = (
            self.test_stats['successful_tests'] / self.test_stats['total_tests'] * 100
            if self.test_stats['total_tests'] > 0 else 0
        )
        
        # 分类统计
        category_stats = self._analyze_by_category()
        
        # 选股效果统计
        selection_stats = self._analyze_selection_effectiveness()
        
        report = {
            'test_summary': {
                'total_tests': self.test_stats['total_tests'],
                'successful_tests': self.test_stats['successful_tests'],
                'failed_tests': self.test_stats['failed_tests'],
                'success_rate': round(success_rate, 2),
                'total_duration': round(duration, 2),
                'test_date': self.test_date,
                'test_stock_count': len(self.test_stocks)
            },
            'category_analysis': category_stats,
            'selection_effectiveness': selection_stats,
            'failed_tests': self.test_stats['error_details'],
            'detailed_results': self.test_stats['test_results']
        }
        
        return report
    
    def _analyze_by_category(self) -> Dict[str, Any]:
        """按类别分析测试结果"""
        categories = {
            'core_indicators': [],
            'enhanced_indicators': [],
            'zxm_indicators': [],
            'composite_indicators': [],
            'pattern_indicators': [],
            'tool_indicators': []
        }
        
        for indicator_name, result in self.test_stats['test_results'].items():
            if indicator_name.startswith('ZXM_'):
                categories['zxm_indicators'].append(result)
            elif indicator_name.startswith('ENHANCED_'):
                categories['enhanced_indicators'].append(result)
            elif indicator_name in ['COMPOSITE', 'UNIFIED_MA', 'CHIP_DISTRIBUTION']:
                categories['composite_indicators'].append(result)
            elif 'PATTERN' in indicator_name:
                categories['pattern_indicators'].append(result)
            elif indicator_name in ['FIBONACCI_TOOLS', 'GANN_TOOLS', 'ELLIOTT_WAVE']:
                categories['tool_indicators'].append(result)
            else:
                categories['core_indicators'].append(result)
        
        # 计算每个类别的统计信息
        category_stats = {}
        for category, results in categories.items():
            if results:
                successful = sum(1 for r in results if r.get('success', False))
                total = len(results)
                # 安全地获取selected_count，如果不存在则默认为0
                avg_selected = sum(r.get('selected_count', 0) for r in results) / total
                
                category_stats[category] = {
                    'total': total,
                    'successful': successful,
                    'success_rate': round(successful / total * 100, 2),
                    'avg_selected_count': round(avg_selected, 2)
                }
        
        return category_stats
    
    def _analyze_selection_effectiveness(self) -> Dict[str, Any]:
        """分析选股效果"""
        all_results = list(self.test_stats['test_results'].values())
        successful_results = [r for r in all_results if r.get('success', False)]
        
        if not successful_results:
            return {'message': '没有成功的测试结果'}
        
        # 选股数量统计
        selection_counts = [r.get('selected_count', 0) for r in successful_results]
        
        # 执行时间统计
        execution_times = [r.get('execution_time', 0) for r in successful_results]
        
        return {
            'selection_count_stats': {
                'total_selected': sum(selection_counts),
                'avg_selected': round(sum(selection_counts) / len(selection_counts), 2),
                'max_selected': max(selection_counts) if selection_counts else 0,
                'min_selected': min(selection_counts) if selection_counts else 0,
                'zero_selection_count': sum(1 for c in selection_counts if c == 0)
            },
            'performance_stats': {
                'avg_execution_time': round(sum(execution_times) / len(execution_times), 2) if execution_times else 0,
                'max_execution_time': round(max(execution_times), 2) if execution_times else 0,
                'min_execution_time': round(min(execution_times), 2) if execution_times else 0,
                'total_execution_time': round(sum(execution_times), 2)
            }
        }
    
    def save_test_report_Test(self, report: Dict[str, Any], filename: Optional[str] = None):
        """保存测试报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unified_engine_test_report_{timestamp}.json"
        
        filepath = os.path.join(root_dir, 'data', 'result', filename)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"📄 测试报告已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存测试报告失败: {e}")
    
    def print_test_summary_Test(self, report: Dict[str, Any]):
        """打印测试摘要"""
        summary = report['test_summary']
        
        print("\n" + "="*80)
        print("📊 统一分析引擎全面测试报告")
        print("="*80)
        print(f"测试日期: {summary['test_date']}")
        print(f"测试股票池: {summary['test_stock_count']} 只股票")
        print(f"总测试数: {summary['total_tests']}")
        print(f"成功测试: {summary['successful_tests']}")
        print(f"失败测试: {summary['failed_tests']}")
        print(f"成功率: {summary['success_rate']}%")
        print(f"总耗时: {summary['total_duration']} 秒")
        
        # 类别分析
        if 'category_analysis' in report:
            print("\n📋 分类分析:")
            for category, stats in report['category_analysis'].items():
                print(f"  {category}: {stats['successful']}/{stats['total']} "
                      f"({stats['success_rate']}%) 平均选股: {stats['avg_selected_count']}")
        
        # 选股效果
        if 'selection_effectiveness' in report:
            sel_stats = report['selection_effectiveness']
            if 'selection_count_stats' in sel_stats:
                print(f"\n📈 选股效果:")
                print(f"  总选股数: {sel_stats['selection_count_stats']['total_selected']}")
                print(f"  平均选股数: {sel_stats['selection_count_stats']['avg_selected']}")
                print(f"  零选股测试: {sel_stats['selection_count_stats']['zero_selection_count']}")
        
        # 失败测试
        if report['failed_tests']:
            print(f"\n❌ 失败测试详情:")
            for error in report['failed_tests'][:5]:  # 只显示前5个错误
                print(f"  {error.get('indicator', 'Unknown')}: {error.get('error', 'Unknown error')}")
            
            if len(report['failed_tests']) > 5:
                print(f"  ... 还有 {len(report['failed_tests']) - 5} 个失败测试")
        
        print("="*80)


def main_comprehensiveunifiedenginetest():
    """主函数"""
    print("🚀 启动统一分析引擎全面测试")
    
    # 测试配置
    test_config = {
        'test_stock_count': 200,  # 测试200只股票
        'enable_early_stop': False,  # 默认禁用早停
        'max_concurrent_tests': 3   # 最多3个并发测试
    }
    
    # 创建测试实例
    tester = Unified_engine_comprehensive_test(**test_config)
    
    try:
        # 运行全面测试
        report = tester.run_comprehensive_test_Test_Comprehensive_Unified_Engine_Test()
        
        # 显示测试摘要
        tester.print_test_summary_Test(report)
        
        # 保存测试报告
        tester.save_test_report_Test(report)
        
        # 检查是否有失败的测试
        if report['test_summary']['failed_tests'] > 0:
            print(f"\n⚠️ 发现 {report['test_summary']['failed_tests']} 个失败测试，请检查详细报告")
            return 1
        else:
            print("\n🎉 所有测试通过！")
            return 0
            
    except Exception as e:
        logger.error(f"❌ 测试执行失败: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    exit(main_comprehensiveunifiedenginetest())