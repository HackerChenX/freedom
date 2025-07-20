"""
统一策略执行器模块

负责执行策略，对股票列表进行筛选和评分
支持统一策略配置格式，整合多个执行器的优化功能
遵循六层架构规范，提供闭环验证支持
"""

import concurrent.futures
import pandas as pd
import numpy as np
import time
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime

from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from strategy.strategy_manager import StrategyManager
from indicators.complete_indicator_registry import complete_registry
from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, safe_run, cache_result
from utils.strategy_validator import UnifiedStrategyConfigValidator
from utils.cache import get_unified_cache, cache_with_unified_layer
from analysis.enhanced_closed_loop_validator import EnhancedClosedLoopValidator
from strategy.condition_parser import ConditionParser
from utils.exceptions import (
    StrategyExecutionError,
    StrategyValidationError,
    DataAccessError,
    IndicatorExecutionError
)

logger = get_logger(__name__)


class UnifiedStrategyExecutor:
    """
    统一策略执行器，负责执行策略，对股票列表进行筛选和评分

    整合了多个执行器的优化功能：
    - 内存优化处理
    - 批量数据获取
    - 并行计算优化
    - 统一配置格式支持
    - 闭环验证机制
    """

    def __init__(self, max_workers: int = None, cache_enabled: bool = True,
                 enable_memory_optimization: bool = True,
                 enable_unified_config: bool = True):
        """
        初始化统一策略执行器

        Args:
            max_workers: 最大线程数，None表示使用默认值
            cache_enabled: 是否启用结果缓存
            enable_memory_optimization: 是否启用内存优化
            enable_unified_config: 是否启用统一配置格式支持
        """
        self.data_access = get_service(DataAccessInterface)
        self.max_workers = max_workers or min(50, os.cpu_count() * 8)
        self.cache_enabled = cache_enabled
        self.cache = {}

        # 统一配置支持
        self.enable_unified_config = enable_unified_config
        if self.enable_unified_config:
            self.config_validator = UnifiedStrategyConfigValidator()

        # 内存优化支持
        self.enable_memory_optimization = enable_memory_optimization
        self.memory_threshold = 75.0  # 内存使用率阈值

        # 初始化完整指标注册系统
        self.indicator_registry = complete_registry

        # 统一缓存层
        self.unified_cache = get_unified_cache()

        # 增强闭环验证器
        self.enhanced_validator = EnhancedClosedLoopValidator(
            enable_entry_point_analysis=True
        )

        # 条件解析器
        self.condition_parser = ConditionParser()

        # 性能统计
        self.performance_stats = {
            'total_executions': 0,
            'total_stocks_processed': 0,
            'total_time': 0,
            'avg_time_per_execution': 0,
            'cache_hit_rate': 0,
            'memory_peak_usage': 0,
            'unified_config_validations': 0
        }

        logger.info(f"统一策略执行器初始化完成，最大线程数: {self.max_workers}, "
                   f"缓存: {'启用' if cache_enabled else '禁用'}, "
                   f"内存优化: {'启用' if enable_memory_optimization else '禁用'}, "
                   f"统一配置: {'启用' if enable_unified_config else '禁用'}")

    @performance_monitor(threshold=5.0)
    def execute_unified_strategy(
        self,
        strategy_config: Dict[str, Any],
        enable_validation: bool = True,
        enable_closed_loop: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        执行统一格式的策略配置

        Args:
            strategy_config: 统一格式的策略配置
            enable_validation: 是否启用配置验证
            enable_closed_loop: 是否启用闭环验证
            progress_callback: 进度回调函数

        Returns:
            Dict[str, Any]: 执行结果，包含选股结果和验证信息
        """
        start_time = time.time()
        execution_result = {
            'strategy_id': strategy_config.get('strategy', {}).get('id', 'unknown'),
            'execution_time': 0,
            'selection_result': None,
            'validation_result': None,
            'performance_stats': {},
            'errors': [],
            'warnings': []
        }

        try:
            # 1. 配置验证
            if enable_validation and self.enable_unified_config:
                if progress_callback:
                    progress_callback(0.1, "验证策略配置")

                validation_result = self.config_validator.validate_strategy_config(strategy_config)
                if not validation_result['is_valid']:
                    execution_result['errors'].extend(validation_result['errors'])
                    raise StrategyValidationError(f"策略配置验证失败: {validation_result['errors']}")

                execution_result['warnings'].extend(validation_result.get('warnings', []))
                self.performance_stats['unified_config_validations'] += 1

            # 2. 策略解析和执行
            if progress_callback:
                progress_callback(0.2, "解析策略配置")

            parsed_strategy = self._parse_unified_strategy_config(strategy_config)

            if progress_callback:
                progress_callback(0.3, "开始执行选股")

            # 3. 执行选股（使用优化的方法）
            selection_result = self._execute_optimized_selection(
                parsed_strategy,
                progress_callback=lambda p, msg: progress_callback(0.3 + p * 0.5, msg) if progress_callback else None
            )

            execution_result['selection_result'] = selection_result

            # 4. 闭环验证（使用增强验证器）
            if enable_closed_loop and len(selection_result) > 0:
                if progress_callback:
                    progress_callback(0.8, "执行增强闭环验证")

                validation_config = strategy_config.get('validation', {})
                validation_method = validation_config.get('validation_method', 'entry_point_analysis')

                validation_result = self.enhanced_validator.validate_strategy_selection(
                    selection_results=selection_result,
                    strategy_config=strategy_config,
                    validation_method=validation_method
                )
                execution_result['validation_result'] = validation_result

            # 5. 更新性能统计
            execution_time = time.time() - start_time
            execution_result['execution_time'] = execution_time
            self._update_performance_stats(len(selection_result), execution_time)
            execution_result['performance_stats'] = self.get_performance_stats()

            if progress_callback:
                progress_callback(1.0, "策略执行完成")

            logger.info(f"统一策略执行完成: {execution_result['strategy_id']}, "
                       f"选股数量: {len(selection_result)}, 耗时: {execution_time:.2f}秒")

            return execution_result

        except Exception as e:
            execution_result['errors'].append(str(e))
            execution_result['execution_time'] = time.time() - start_time
            logger.error(f"统一策略执行失败: {e}")
            raise StrategyExecutionError(f"统一策略执行失败: {e}")

    def _parse_unified_strategy_config(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """解析统一格式的策略配置"""
        try:
            # 提取策略信息
            strategy_info = strategy_config.get('strategy', {})

            # 提取技术指标配置
            tech_indicators = strategy_config.get('technical_indicators', {})
            primary_indicators = tech_indicators.get('primary_indicators', [])

            # 提取时间条件
            time_criteria = strategy_config.get('time_criteria', {})
            time_frames = time_criteria.get('time_frames', [])

            # 提取过滤条件
            filters = strategy_config.get('filters', {})

            # 提取选股参数
            selection_params = strategy_config.get('selection_parameters', {})

            # 转换为内部执行格式
            parsed_strategy = {
                'strategy_id': strategy_info.get('id'),
                'strategy_name': strategy_info.get('name'),
                'indicators': self._convert_indicators_config(primary_indicators),
                'time_frames': [frame.get('level') for frame in time_frames],
                'filters': filters,
                'max_selections': selection_params.get('max_selections', 50),
                'min_score': selection_params.get('min_score', 0.6),
                'ranking_method': selection_params.get('ranking_method', 'score'),
                'target_date': time_criteria.get('date_range', {}).get('target_date',
                                                datetime.now().strftime('%Y-%m-%d'))
            }

            return parsed_strategy

        except Exception as e:
            logger.error(f"解析统一策略配置失败: {e}")
            raise StrategyValidationError(f"解析统一策略配置失败: {e}")

    def _convert_indicators_config(self, indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换指标配置为内部格式"""
        converted_indicators = []

        for indicator in indicators:
            indicator_id = indicator.get('indicator_id')
            parameters = indicator.get('parameters', {})
            conditions = indicator.get('conditions', [])

            converted_indicator = {
                'type': indicator_id,
                'params': parameters,
                'conditions': []
            }

            # 转换条件格式
            for condition in conditions:
                converted_condition = {
                    'field': condition.get('field'),
                    'operator': condition.get('operator'),
                    'value': condition.get('value'),
                    'lookback_days': condition.get('lookback_days', 1)
                }
                converted_indicator['conditions'].append(converted_condition)

            converted_indicators.append(converted_indicator)

        return converted_indicators

    def _execute_optimized_selection(
        self,
        parsed_strategy: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """执行优化的选股逻辑"""
        try:
            # 1. 获取股票列表
            if progress_callback:
                progress_callback(0.1, "获取股票列表")

            stock_codes = self._get_filtered_stock_list(parsed_strategy.get('filters', {}))
            total_stocks = len(stock_codes)

            if total_stocks == 0:
                logger.warning("过滤后的股票列表为空")
                return []

            logger.info(f"开始处理 {total_stocks} 只股票")

            # 2. 批量获取股票数据（优化）
            if progress_callback:
                progress_callback(0.2, f"批量获取 {total_stocks} 只股票数据")

            stocks_data = self._batch_load_stock_data(
                stock_codes, parsed_strategy.get('target_date')
            )

            # 3. 并行计算指标和评估条件
            if progress_callback:
                progress_callback(0.4, "并行计算指标")

            evaluation_results = self._parallel_evaluate_stocks(
                stocks_data,
                parsed_strategy,
                progress_callback=lambda p, msg: progress_callback(0.4 + p * 0.4, msg) if progress_callback else None
            )

            # 4. 过滤和排序结果
            if progress_callback:
                progress_callback(0.8, "过滤和排序结果")

            filtered_results = self._filter_and_rank_results(
                evaluation_results, parsed_strategy
            )

            return filtered_results

        except Exception as e:
            logger.error(f"优化选股执行失败: {e}")
            raise StrategyExecutionError(f"优化选股执行失败: {e}")

    def _batch_load_stock_data(self, stock_codes: List[str], target_date: str) -> Dict[str, Any]:
        """批量加载股票数据（优化方法）"""
        try:
            # 使用批量查询优化数据获取
            batch_size = min(100, len(stock_codes))  # 动态调整批次大小
            stocks_data = {}

            for i in range(0, len(stock_codes), batch_size):
                batch_codes = stock_codes[i:i + batch_size]

                # 批量获取股票基础数据
                batch_data = self.data_access.get_stocks_data_batch(
                    stock_codes=batch_codes,
                    start_date=target_date,
                    end_date=target_date
                )

                stocks_data.update(batch_data)

            logger.debug(f"批量加载完成: {len(stocks_data)} 只股票数据")
            return stocks_data

        except Exception as e:
            logger.error(f"批量加载股票数据失败: {e}")
            # 回退到单个获取
            return self._fallback_load_stock_data(stock_codes, target_date)

    def _fallback_load_stock_data(self, stock_codes: List[str], target_date: str) -> Dict[str, Any]:
        """回退的股票数据加载方法"""
        stocks_data = {}
        for code in stock_codes:
            try:
                data = self.data_access.get_stock_data(code, target_date, target_date)
                if data is not None and not data.empty:
                    stocks_data[code] = data
            except Exception as e:
                logger.warning(f"获取股票数据失败: {code}, 错误: {e}")
                continue

        return stocks_data

    def _parallel_evaluate_stocks(
        self,
        stocks_data: Dict[str, Any],
        strategy: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """并行评估股票"""
        try:
            results = []
            stock_items = list(stocks_data.items())
            total_stocks = len(stock_items)

            if total_stocks == 0:
                return results

            # 动态调整批次大小
            batch_size = max(1, min(50, total_stocks // self.max_workers))
            processed_count = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_stock = {}
                for stock_code, stock_data in stock_items:
                    future = executor.submit(
                        self._evaluate_single_stock_unified,
                        stock_code, stock_data, strategy
                    )
                    future_to_stock[future] = stock_code

                # 收集结果
                for future in concurrent.futures.as_completed(future_to_stock):
                    stock_code = future_to_stock[future]
                    try:
                        result = future.result(timeout=30)  # 30秒超时
                        if result:
                            results.append(result)

                        processed_count += 1
                        if progress_callback and processed_count % 100 == 0:
                            progress = processed_count / total_stocks
                            progress_callback(progress, f"已处理 {processed_count}/{total_stocks} 只股票")

                    except Exception as e:
                        logger.warning(f"评估股票失败: {stock_code}, 错误: {e}")
                        continue

            logger.info(f"并行评估完成: {len(results)}/{total_stocks} 只股票通过筛选")
            return results

        except Exception as e:
            logger.error(f"并行评估股票失败: {e}")
            raise StrategyExecutionError(f"并行评估股票失败: {e}")

    def _evaluate_single_stock_unified(
        self,
        stock_code: str,
        stock_data: Any,
        strategy: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """评估单只股票（统一格式）"""
        try:
            # 计算技术指标
            indicators_data = {}
            for indicator_config in strategy.get('indicators', []):
                indicator_type = indicator_config.get('type')
                params = indicator_config.get('params', {})

                # 使用指标注册系统计算指标
                if hasattr(self.indicator_registry, 'calculate_indicator'):
                    indicator_result = self.indicator_registry.calculate_indicator(
                        indicator_type, stock_data, **params
                    )
                    indicators_data[indicator_type] = indicator_result

            # 评估条件（使用新的条件解析器）
            total_score = 0
            total_weight = 0
            all_condition_results = {}

            for indicator_config in strategy.get('indicators', []):
                indicator_type = indicator_config.get('type')
                conditions = indicator_config.get('conditions', [])
                indicator_data = indicators_data.get(indicator_type)

                if indicator_data is None:
                    continue

                # 使用条件解析器评估条件
                evaluation_result = self.condition_parser.evaluate_conditions(
                    conditions, indicator_data
                )

                if evaluation_result.get('success', False):
                    condition_score = evaluation_result.get('score', 0)
                    condition_weight = evaluation_result.get('total_weight', 1)

                    total_score += condition_score * condition_weight
                    total_weight += condition_weight

                    # 记录详细结果
                    all_condition_results[indicator_type] = evaluation_result

            # 计算最终评分
            final_score = total_score / total_weight if total_weight > 0 else 0
            min_score = strategy.get('min_score', 0.6)

            if final_score >= min_score:
                return {
                    'stock_code': stock_code,
                    'score': final_score,
                    'total_score': total_score,
                    'total_weight': total_weight,
                    'condition_results': all_condition_results,
                    'indicators_data': indicators_data,
                    'evaluation_method': 'complex_conditions'
                }

            return None

        except Exception as e:
            logger.warning(f"评估股票失败: {stock_code}, 错误: {e}")
            return None

    def _evaluate_condition(self, indicator_data: Any, condition: Dict[str, Any]) -> bool:
        """评估单个条件"""
        try:
            if indicator_data is None:
                return False

            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')

            # 从指标数据中获取字段值
            if isinstance(indicator_data, dict):
                field_value = indicator_data.get(field)
            elif hasattr(indicator_data, field):
                field_value = getattr(indicator_data, field)
            else:
                return False

            if field_value is None:
                return False

            # 执行比较操作
            if operator == '>':
                return field_value > value
            elif operator == '<':
                return field_value < value
            elif operator == '>=':
                return field_value >= value
            elif operator == '<=':
                return field_value <= value
            elif operator == '=':
                return field_value == value
            elif operator == '!=':
                return field_value != value
            elif operator == 'between':
                if isinstance(value, list) and len(value) == 2:
                    return value[0] <= field_value <= value[1]
            elif operator == 'cross_up':
                # 简化的金叉判断
                return field_value > value
            elif operator == 'cross_down':
                # 简化的死叉判断
                return field_value < value

            return False

        except Exception as e:
            logger.warning(f"评估条件失败: {condition}, 错误: {e}")
            return False

    def _filter_and_rank_results(
        self,
        results: List[Dict[str, Any]],
        strategy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """过滤和排序结果"""
        try:
            if not results:
                return []

            # 按评分排序
            ranking_method = strategy.get('ranking_method', 'score')

            if ranking_method == 'score':
                sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            elif ranking_method == 'indicator_strength':
                sorted_results = sorted(results, key=lambda x: x.get('conditions_met', 0), reverse=True)
            else:  # composite
                sorted_results = sorted(
                    results,
                    key=lambda x: (x.get('score', 0) * 0.7 + x.get('conditions_met', 0) * 0.3),
                    reverse=True
                )

            # 限制结果数量
            max_selections = strategy.get('max_selections', 50)
            final_results = sorted_results[:max_selections]

            logger.info(f"结果过滤完成: {len(final_results)}/{len(results)} 只股票入选")
            return final_results

        except Exception as e:
            logger.error(f"过滤和排序结果失败: {e}")
            return results[:strategy.get('max_selections', 50)]

    def _get_filtered_stock_list(self, filters: Dict[str, Any]) -> List[str]:
        """获取过滤后的股票列表"""
        try:
            # 获取所有股票列表
            all_stocks = self.data_access.get_all_stock_codes()

            if not filters:
                return all_stocks

            filtered_stocks = []
            market_filters = filters.get('market_filters', {})
            financial_filters = filters.get('financial_filters', {})

            # 市场过滤
            allowed_markets = market_filters.get('markets', [])
            exclude_st = market_filters.get('exclude_st', True)

            for stock_code in all_stocks:
                # ST股票过滤
                if exclude_st and ('ST' in stock_code or 'st' in stock_code):
                    continue

                # 市场过滤（简化实现）
                if allowed_markets:
                    market_matched = False
                    for market in allowed_markets:
                        if market == '主板' and (stock_code.endswith('.SH') or stock_code.endswith('.SZ')):
                            market_matched = True
                            break
                        elif market == '创业板' and stock_code.startswith('3'):
                            market_matched = True
                            break
                        elif market == '科创板' and stock_code.startswith('688'):
                            market_matched = True
                            break

                    if not market_matched:
                        continue

                filtered_stocks.append(stock_code)

            logger.info(f"股票过滤完成: {len(filtered_stocks)}/{len(all_stocks)} 只股票通过过滤")
            return filtered_stocks

        except Exception as e:
            logger.error(f"获取过滤股票列表失败: {e}")
            # 回退到获取所有股票
            try:
                return self.data_access.get_all_stock_codes()
            except:
                return []

    def _perform_closed_loop_validation(
        self,
        selection_results: List[Dict[str, Any]],
        strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行闭环验证"""
        try:
            validation_config = strategy_config.get('validation', {})
            validation_method = validation_config.get('validation_method', 'entry_point_analysis')
            sample_size = min(validation_config.get('sample_size', 10), len(selection_results))

            if sample_size == 0:
                return {
                    'validation_method': validation_method,
                    'sample_size': 0,
                    'consistency_rate': 0.0,
                    'validation_passed': False,
                    'details': []
                }

            # 选择验证样本
            validation_sample = selection_results[:sample_size]
            validation_details = []
            consistent_count = 0

            for result in validation_sample:
                stock_code = result.get('stock_code')
                original_score = result.get('score', 0)

                # 重新验证该股票是否满足条件
                revalidation_result = self._revalidate_stock_selection(
                    stock_code, strategy_config
                )

                is_consistent = abs(revalidation_result.get('score', 0) - original_score) < 0.1
                if is_consistent:
                    consistent_count += 1

                validation_details.append({
                    'stock_code': stock_code,
                    'original_score': original_score,
                    'revalidation_score': revalidation_result.get('score', 0),
                    'is_consistent': is_consistent,
                    'revalidation_details': revalidation_result
                })

            consistency_rate = consistent_count / sample_size
            validation_threshold = validation_config.get('validation_threshold', 0.8)
            validation_passed = consistency_rate >= validation_threshold

            validation_result = {
                'validation_method': validation_method,
                'sample_size': sample_size,
                'consistent_count': consistent_count,
                'consistency_rate': consistency_rate,
                'validation_threshold': validation_threshold,
                'validation_passed': validation_passed,
                'details': validation_details
            }

            logger.info(f"闭环验证完成: 一致性率 {consistency_rate:.2%}, "
                       f"验证{'通过' if validation_passed else '失败'}")

            return validation_result

        except Exception as e:
            logger.error(f"闭环验证失败: {e}")
            return {
                'validation_method': 'error',
                'error': str(e),
                'validation_passed': False
            }

    def _revalidate_stock_selection(
        self,
        stock_code: str,
        strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """重新验证单只股票的选股结果"""
        try:
            # 重新获取股票数据
            target_date = strategy_config.get('time_criteria', {}).get('date_range', {}).get('target_date')
            if not target_date:
                target_date = datetime.now().strftime('%Y-%m-%d')

            stock_data = self.data_access.get_stock_data(stock_code, target_date, target_date)
            if stock_data is None or stock_data.empty:
                return {'score': 0, 'error': 'No data available'}

            # 解析策略配置
            parsed_strategy = self._parse_unified_strategy_config(strategy_config)

            # 重新评估
            result = self._evaluate_single_stock_unified(stock_code, stock_data, parsed_strategy)

            if result:
                return {
                    'score': result.get('score', 0),
                    'conditions_met': result.get('conditions_met', 0),
                    'total_conditions': result.get('total_conditions', 0)
                }
            else:
                return {'score': 0, 'conditions_met': 0}

        except Exception as e:
            logger.warning(f"重新验证股票失败: {stock_code}, 错误: {e}")
            return {'score': 0, 'error': str(e)}

    def _update_performance_stats(self, stocks_processed: int, execution_time: float):
        """更新性能统计"""
        self.performance_stats['total_executions'] += 1
        self.performance_stats['total_stocks_processed'] += stocks_processed
        self.performance_stats['total_time'] += execution_time

        if self.performance_stats['total_executions'] > 0:
            self.performance_stats['avg_time_per_execution'] = (
                self.performance_stats['total_time'] / self.performance_stats['total_executions']
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return self.performance_stats.copy()


# 保持向后兼容性
class StrategyExecutor(UnifiedStrategyExecutor):
    """
    策略执行器，负责执行策略，对股票列表进行筛选和评分
    向后兼容的别名类
    """

    def __init__(self, max_workers: int = None, cache_enabled: bool = True):
        """
        初始化策略执行器（向后兼容）

        Args:
            max_workers: 最大线程数，None表示使用默认值（CPU核心数 * 5）
            cache_enabled: 是否启用结果缓存
        """
        super().__init__(
            max_workers=max_workers,
            cache_enabled=cache_enabled,
            enable_memory_optimization=False,  # 向后兼容，默认不启用新功能
            enable_unified_config=False
        )
        try:
            self.indicator_registry.register_all_indicators()
            logger.info("✅ 策略执行器已集成CompleteIndicatorRegistry")
        except Exception as e:
            logger.error(f"❌ 初始化指标注册系统失败: {e}")

        # 初始化条件评估器，确保所有线程共享同一个实例
        from strategy.strategy_condition_evaluator import StrategyConditionEvaluator
        self._evaluator = StrategyConditionEvaluator()

        logger.info(f"策略执行器初始化完成，最大线程数: {self.max_workers}, 缓存{'启用' if cache_enabled else '禁用'}")
    
    @performance_monitor(threshold=1.0)
    def execute_strategy_by_id(
        self, 
        strategy_id: str, 
        strategy_manager: StrategyManager,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> pd.DataFrame:
        """
        通过策略ID执行选股策略
        
        Args:
            strategy_id: 策略ID
            strategy_manager: 策略管理器实例
            start_date: 开始日期，默认为None
            end_date: 结束日期，默认为当前日期
            progress_callback: 进度回调函数，参数为进度百分比和状态消息
            
        Returns:
            选股结果Data_frame
            
        Raises:
            StrategyExecutionError: 策略执行错误
        """
        try:
            # 检查缓存
            cache_key = f"strategy_result_{strategy_id}_{start_date}_{end_date}"
            if self.cache_enabled and cache_key in self.cache:
                logger.info(f"使用缓存的策略执行结果: {strategy_id}")
                return self.cache[cache_key]
            
            # 获取策略配置
            if progress_callback:
                progress_callback(0.1, f"正在加载策略: {strategy_id}")
                
            strategy_config = strategy_manager.get_strategy(strategy_id)
            if not strategy_config:
                raise StrategyValidationError(f"未找到策略: {strategy_id}")
                
            # 解析策略
            if progress_callback:
                progress_callback(0.2, "正在解析策略配置")
                
            from strategy.strategy_parser import Strategy_parser
            parser = Strategy_parser()
            strategy_plan = parser.parse_strategy(strategy_config)
            
            # 执行策略
            if progress_callback:
                progress_callback(0.3, "开始执行策略")
                
            result = self.execute_strategy(
                strategy_plan=strategy_plan,
                start_date=start_date,
                end_date=end_date,
                progress_callback=lambda p, m: progress_callback(0.3 + p * 0.7, m) if progress_callback else None
            )
            
            # 缓存结果
            if self.cache_enabled:
                self.cache[cache_key] = result
                
            return result
        except Exception as e:
            logger.error(f"执行策略 {strategy_id} 失败: {e}")
            raise StrategyExecutionError(f"执行策略失败: {str(e)}")
    
    @performance_monitor(threshold=1.0)
    def execute_strategy(
        self, 
        strategy_plan: Dict[str, Any],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        enable_early_stop: bool = False  # 新增参数：是否启用早停
    ) -> pd.DataFrame:
        """
        执行选股策略
        
        Args:
            strategy_plan: 策略执行计划
            start_date: 开始日期，默认为None
            end_date: 结束日期，默认为当前日期
            progress_callback: 进度回调函数，参数为进度百分比和状态消息
            
        Returns:
            选股结果Data_frame
            
        Raises:
            StrategyExecutionError: 策略执行错误
        """
        try:
            # 1. 验证策略计划
            self._validate_strategy_plan(strategy_plan)
            
            # 2. 处理日期参数
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            # 3. 获取股票列表
            if progress_callback:
                progress_callback(0.1, "正在获取股票列表")
                
            stock_list = self._get_filtered_stock_list(strategy_plan.get('filters', {}))
            
            if stock_list.empty:
                logger.warning("过滤后的股票列表为空")
                return pd.DataFrame()
                
            # 4. 获取条件配置
            conditions = strategy_plan.get('conditions', [])
            
            # 5. 开始处理
            if progress_callback:
                progress_callback(0.2, f"开始处理 {len(stock_list)} 只股票")
            
            # 性能优化：使用动态分批处理机制
            total_stocks = len(stock_list)
            
            # 动态调整批次大小，基于总股票数量（优化版本）
            if total_stocks > 5000:
                batch_size = 200  # 超大规模股票，使用更大的批次
            elif total_stocks > 2000:
                batch_size = 150  # 大规模股票
            elif total_stocks > 1000:
                batch_size = 100  # 中等规模股票
            elif total_stocks > 500:
                batch_size = 75   # 小规模股票
            else:
                batch_size = 50   # 默认批次大小
                
            processed_stocks = 0
            results = []
            
            # 预先加载通用缓存数据
            if progress_callback:
                progress_callback(0.22, "预加载通用数据")
            self._preload_common_data(end_date)
            
            # 创建共享的线程池，而不是每个批次单独创建
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 批次处理循环
                for batch_idx in range(0, total_stocks, batch_size):
                    batch_end = min(batch_idx + batch_size, total_stocks)
                    batch = stock_list.iloc[batch_idx:batch_end]
                    
                    if progress_callback:
                        progress_callback(0.25 + 0.7 * batch_idx / total_stocks, 
                                        f"处理批次 {batch_idx//batch_size + 1}/{(total_stocks+batch_size-1)//batch_size}")
                    
                    # 提交本批次的所有任务
                    future_to_stock = {}
                    for _, row in batch.iterrows():
                        stock_code = row['stock_code']
                        # 尝试获取股票名称，如果没有则使用股票代码
                        stock_name = row.get('stock_name', row.get('name', stock_code))

                        future = executor.submit(
                            self._process_stock,
                            stock_code=stock_code,
                            stock_name=stock_name,
                            conditions=conditions,
                            end_date=end_date
                        )
                        future_to_stock[future] = stock_code
                    
                    # 并行处理任务
                    early_stop = False
                    for future in concurrent.futures.as_completed(future_to_stock):
                        stock_code = future_to_stock[future]
                        
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                                # 早停功能：只有在启用早停时才触发
                                if enable_early_stop:
                                    logger.info(f"✅ 找到匹配股票: {stock_code}，触发早停")
                                    early_stop = True
                                    
                                    # 取消剩余的任务
                                    for remaining_future in future_to_stock:
                                        if remaining_future != future and not remaining_future.done():
                                            remaining_future.cancel()
                                    break
                                else:
                                    # 不启用早停时，继续处理但记录找到的股票
                                    logger.info(f"✅ 找到匹配股票: {stock_code} (继续处理)")
                        except Exception as e:
                            logger.error(f"处理股票 {stock_code} 时出错: {e}")
                            # 只有在启用早停时才因错误停止
                            if enable_early_stop:
                                logger.warning(f"⚠️ 遇到错误，触发早停: {e}")
                                early_stop = True
                                
                                # 取消剩余的任务
                                for remaining_future in future_to_stock:
                                    if remaining_future != future and not remaining_future.done():
                                        remaining_future.cancel()
                                break
                            else:
                                # 不启用早停时，记录错误但继续处理
                                logger.warning(f"⚠️ 处理股票 {stock_code} 时出错，继续处理其他股票: {e}")
                        
                        # 更新进度
                        processed_stocks += 1
                        if progress_callback and processed_stocks % 10 == 0:  # 每10只股票更新一次进度，避免频繁更新
                            progress_callback(0.25 + 0.7 * processed_stocks / total_stocks, 
                                            f"已处理 {processed_stocks}/{total_stocks} 只股票")
                    
                    # 如果触发了早停，跳出批次循环
                    if early_stop:
                        logger.info("早停触发，结束处理")
                        break
                    
                    # 批次处理完毕后清理不需要的缓存，减少内存占用
                    self._clean_batch_cache()
            
            if progress_callback:
                progress_callback(0.95, "正在整理结果")
                
            # 6. 转换结果为DataFrame
            if not results:
                logger.info("没有股票满足条件")
                return pd.DataFrame()
                
            result_df = pd.DataFrame(results)
            
            # 7. 排序和处理结果
            if 'score' in result_df.columns:
                result_df = result_df.sort_values(by='score', ascending=False)
            
            # 8. 应用结果过滤
            # 如果策略配置了结果限制，应用这些限制
            if 'result_filters' in strategy_plan:
                result_filters = strategy_plan['result_filters']
                
                # 限制结果数量
                if 'max_results' in result_filters and isinstance(result_filters['max_results'], int):
                    max_results = result_filters['max_results']
                    if len(result_df) > max_results:
                        result_df = result_df.head(max_results)
                
                # 最小评分限制
                if 'min_score' in result_filters and isinstance(result_filters['min_score'], (int, float)):
                    min_score = result_filters['min_score']
                    if 'score' in result_df.columns:
                        result_df = result_df[result_df['score'] >= min_score]
            
            if progress_callback:
                progress_callback(1.0, "策略执行完成")
            
            logger.info(f"选股结果: 共 {len(result_df)} 只股票满足条件")
            return result_df
            
        except Exception as e:
            logger.error(f"执行策略时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise StrategyExecutionError(f"执行策略失败: {str(e)}")
    
    def _preload_common_data(self, date: str) -> None:
        """
        预加载通用数据，提高处理效率
        
        Args:
            date: 日期字符串
        """
        try:
            # 预加载行业数据
            self.data_access.preload_industry_data()
            
            # 预加载指数数据
            index_codes = ['000001.SH', '399001.SZ', '399006.SZ']  # 上证指数、深证成指、创业板指
            for index_code in index_codes:
                self.data_access.get_index_data(index_code, date)
                
            # 预加载市场整体状态
            self.data_access.get_market_status(date)
            
            logger.info("预加载通用数据完成")
        except Exception as e:
            logger.warning(f"预加载通用数据时出错: {e}")
    
    def _clean_batch_cache(self) -> None:
        """
        清理批次处理后的临时缓存
        """
        # 仅保留最常用的缓存项目，清理其他缓存
        if hasattr(self, 'cache') and isinstance(self.cache, dict):
            # 计算缓存大小
            cache_size = len(self.cache)
            if cache_size > 500:  # 降低清理阈值，更频繁地清理缓存
                logger.info(f"缓存项过多 ({cache_size})，进行清理")
                
                # 保留策略结果缓存和最近的股票数据缓存
                keys_to_keep = [k for k in self.cache.keys() if k.startswith('strategy_result_')]
                
                # 添加最近的200个股票数据缓存
                stock_data_keys = [k for k in self.cache.keys() if k.startswith('stock_data_')]
                if len(stock_data_keys) > 200:
                    # 只保留最近的200个
                    keys_to_keep.extend(stock_data_keys[-200:])
                else:
                    keys_to_keep.extend(stock_data_keys)
                
                # 创建新缓存
                new_cache = {k: self.cache[k] for k in keys_to_keep if k in self.cache}
                self.cache = new_cache
                
                # 触发Python垃圾回收
                import gc
                gc.collect()
                
                logger.info(f"缓存清理完成，保留 {len(self.cache)} 项")
    
    @safe_run(error_logger=logger)
    def _process_stock(
        self, 
        stock_code: str, 
        stock_name: str,
        conditions: List[Dict[str, Any]],
        end_date: str
    ) -> Optional[Dict[str, Any]]:
        """
        处理单个股票
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            conditions: 条件列表
            end_date: 结束日期
            
        Returns:
            Optional[Dict[str, Any]]: 处理结果，如果不满足条件则返回None
        """
        try:
            # 1. 缓存键，用于存储和获取数据
            cache_key = f"stock_data_{stock_code}_{end_date}"
            
            # 2. 从缓存获取数据或从数据库加载
            if self.cache_enabled and cache_key in self.cache:
                data = self.cache[cache_key]
                logger.debug(f"使用缓存的 {stock_code} 数据")
            else:
                # 获取股票的最近数据
                try:
                    # 计算开始日期（往前120个交易日）
                    start_date = self.data_access.get_previous_trade_date(end_date, 120)
                    
                    # 获取K线数据 - 明确指定日线数据
                    data = self.data_access.get_stock_data(code=stock_code,
                        start_date=start_date,
                        end_date=end_date,
                        period='daily'  # 明确指定获取日线数据
                    )
                    
                    # 缓存数据
                    if self.cache_enabled:
                        self.cache[cache_key] = data
                        
                except DataAccessError as e:
                    logger.warning(f"获取股票 {stock_code} 数据失败: {e}")
                    return None
                
                except Exception as e:
                    logger.error(f"处理股票 {stock_code} 数据时出错: {e}")
                    return None
            
            # 如果没有数据或数据不足，则跳过
            if data is None or len(data) < 30:
                logger.warning(f"股票 {stock_code} 数据不足，跳过")
                return None
                
            # 3. 评估条件
            # 使用预初始化的evaluator实例，确保所有线程共享同一个指标注册表
            evaluator = self._evaluator
            
            # 创建条件评估追踪对象，用于记录详细评估结果
            condition_details = {
                "passing_indicators": [],
                "failing_indicators": [],
                "condition_results": {}
            }
            
            # 逐条评估并收集结果
            evaluation_results = []
            current_logic = "and"  # 默认使用AND逻辑
            
            for i, condition in enumerate(conditions):
                condition_id = f"condition_{i+1}"
                
                # 检查是否是逻辑运算符
                if condition.get("type") == "logic":
                    current_logic = condition.get("value", "").lower()
                    condition_details["condition_results"][condition_id] = {
                        "type": "logic",
                        "value": current_logic,
                        "result": None  # 逻辑运算符本身没有结果
                    }
                    continue
                
                # 评估当前条件
                try:
                    condition_result = evaluator.evaluate_condition(condition, data, end_date)
                    evaluation_results.append(condition_result)
                    
                    # 记录详细结果
                    condition_type = condition.get("type", "unknown")
                    indicator_id = condition.get("indicator_id", "") or condition.get("indicator", "")
                    
                    condition_details["condition_results"][condition_id] = {
                        "type": condition_type,
                        "indicator_id": indicator_id,
                        "result": condition_result
                    }
                    
                    # 追踪通过和失败的指标
                    if condition_type == "indicator" and indicator_id:
                        if condition_result:
                            condition_details["passing_indicators"].append(indicator_id)
                        else:
                            condition_details["failing_indicators"].append(indicator_id)
                            
                except Exception as e:
                    logger.error(f"评估条件 {i+1} 时出错: {e}")
                    evaluation_results.append(False)  # 出错视为不满足条件
                    condition_details["condition_results"][condition_id] = {
                        "type": condition.get("type", "unknown"),
                        "error": str(e),
                        "result": False
                    }
            
            # 根据逻辑关系计算最终结果
            if current_logic == "and":
                final_result = all(evaluation_results)
            elif current_logic == "or":
                final_result = any(evaluation_results)
            elif current_logic == "not" and evaluation_results:
                # NOT逻辑只考虑第一个条件结果取反
                final_result = not evaluation_results[0]
            else:
                logger.warning(f"未知的逻辑操作符: {current_logic}，默认使用AND")
                final_result = all(evaluation_results)
            
            # 如果不满足条件，直接返回None
            if not final_result:
                return None
            
            # 4. 获取股票的最新价格和基本信息
            try:
                latest_data = data.iloc[-1]
                price = latest_data['close']
                change_pct = latest_data['pct_chg'] if 'pct_chg' in latest_data else 0.0
                
                # 计算评分
                score = self._calculate_stock_score_Strategy_Executor(stock_code, data, condition_details)
                
                # 获取行业信息
                industry = self.data_access.get_stock_industry(stock_code)
                
                return {
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'industry': industry,
                    'price': price,
                    'change_pct': change_pct,
                    'score': score,
                    'match_details': condition_details,
                    'selection_date': end_date
                }
                
            except Exception as e:
                logger.error(f"获取股票 {stock_code} 的价格和基本信息时出错: {e}")
                return None
        
        except Exception as e:
            logger.error(f"处理股票 {stock_code} 时出错: {e}")
            return None
    
    def _calculate_stock_score_Strategy_Executor(self, stock_code: str, data: pd.DataFrame, 
                              details: Dict[str, Any]) -> float:
        """
        计算股票评分
        
        Args:
            stock_code: 股票代码
            data: 股票数据
            details: 条件评估详情
            
        Returns:
            float: 评分（0-100）
        """
        try:
            # 基础分数
            base_score = 60.0
            
            # 如果数据不足，返回基础分数
            if len(data) < 30:
                logger.warning(f"股票 {stock_code} 数据不足，使用基础评分")
                return base_score
                
            # 计算技术指标得分 (30%)
            tech_score = self._calculate_technical_score_Strategy_Executor(data, details)
            
            # 计算趋势得分 (25%)
            trend_score = self._calculate_trend_score_Strategy_Executor(data)
            
            # 计算动量得分 (15%)
            momentum_score = self._calculate_momentum_score_Strategy_Executor(data)
            
            # 计算成交量得分 (15%)
            volume_score = self._calculate_volume_score_Strategy_Executor(data)
            
            # 计算波动性得分 (10%)
            volatility_score = self._calculate_volatility_score_Strategy_Executor(data)
            
            # 计算市场环境得分 (5%)
            market_score = self._calculate_market_score(stock_code, data)
            
            # 综合得分（加权平均）
            final_score = (
                tech_score * 0.3 + 
                trend_score * 0.25 + 
                momentum_score * 0.15 + 
                volume_score * 0.15 + 
                volatility_score * 0.1 + 
                market_score * 0.05
            )
            
            # 确保分数在0-100范围内
            final_score = max(0, min(100, final_score))
            
            return round(final_score, 1)
            
        except Exception as e:
            logger.error(f"计算股票 {stock_code} 评分时出错: {e}")
            return base_score  # 返回默认评分
    
    def _calculate_technical_score_Strategy_Executor(self, data: pd.DataFrame,
                                 details: Dict[str, Any]) -> float:
        """
        计算技术指标评分（使用Complete_indicator_registry动态计算）

        Args:
            data: 股票数据
            details: 条件评估详情

        Returns:
            float: 技术指标评分（0-100）
        """
        try:
            # 优先使用通过的指标信息
            passing_indicators = details.get("passing_indicators", [])

            # 如果有通过的指标，使用动态评分
            if passing_indicators:
                return self._calculate_dynamic_technical_score(data, passing_indicators)

            # 如果没有通过的指标信息，使用CompleteIndicatorRegistry计算核心指标
            return self._calculate_core_indicators_score(data)
            
        except Exception as e:
            logger.error(f"计算技术指标评分时出错: {e}")
            return 60.0

    def _calculate_dynamic_technical_score(self, data: pd.DataFrame, passing_indicators: List[str]) -> float:
        """
        使用Complete_indicator_registry动态计算技术指标评分

        Args:
            data: 股票数据
            passing_indicators: 通过的指标列表

        Returns:
            float: 技术指标评分（0-100）
        """
        try:
            tech_score = 0.0
            tech_weight = 0.0

            # 指标权重配置（重要指标权重更高）
            indicator_weights = {
                "MACD": 1.0, "KDJ": 0.9, "RSI": 0.8, "BOLL": 0.9, "MA": 0.7,
                "VOL": 0.7, "DMI": 0.8, "CCI": 0.6, "WR": 0.6, "OBV": 0.7,
                "PSY": 0.5, "BIAS": 0.6, "ROC": 0.7, "EMV": 0.5, "SAR": 0.8,
                "DMA": 0.6, "MTM": 0.6, "TRIX": 0.7, "STOCHRSI": 0.8
            }

            # 根据通过的指标计算技术分数
            for indicator_id in passing_indicators:
                # 确定指标类型
                indicator_type = None
                for key in indicator_weights.keys():
                    if key in indicator_id.upper():
                        indicator_type = key
                        break

                if indicator_type:
                    weight = indicator_weights.get(indicator_type, 0.5)
                    # 使用CompleteIndicatorRegistry动态获取指标评分
                    score = self._get_indicator_dynamic_score(indicator_type, data)
                else:
                    weight = 0.5
                    score = 60.0

                tech_score += score * weight
                tech_weight += weight

            # 如果没有技术指标通过，返回基础评分
            if tech_weight <= 0:
                return 60.0

            # 计算加权平均技术分数
            tech_score = tech_score / tech_weight

            # 如果多个重要技术指标同时满足，额外加分
            important_indicators = [i for i in passing_indicators if any(k in i.upper() for k in ["MACD", "KDJ", "RSI", "BOLL"])]
            if len(important_indicators) >= 2:
                tech_score += 10.0

            return max(0, min(100, tech_score))

        except Exception as e:
            logger.error(f"动态计算技术指标评分时出错: {e}")
            return 60.0

    def _calculate_core_indicators_score(self, data: pd.DataFrame) -> float:
        """
        使用Complete_indicator_registry计算核心指标评分

        Args:
            data: 股票数据

        Returns:
            float: 核心指标评分（0-100）
        """
        try:
            if not hasattr(self, 'indicator_registry') or self.indicator_registry is None:
                return 60.0

            core_indicators = ['MACD', 'KDJ', 'RSI', 'BOLL', 'MA']
            total_score = 0.0
            valid_count = 0

            for indicator_name in core_indicators:
                try:
                    indicator = self.indicator_registry.create_indicator(indicator_name)
                    if indicator:
                        score = indicator.calculate_raw_score(data)
                        if score is not None and len(score) > 0:
                            total_score += float(score.iloc[-1])
                            valid_count += 1
                except Exception as e:
                    logger.debug(f"计算指标 {indicator_name} 评分失败: {e}")
                    continue

            if valid_count > 0:
                return total_score / valid_count
            else:
                return 60.0

        except Exception as e:
            logger.error(f"计算核心指标评分时出错: {e}")
            return 60.0

    def _get_indicator_dynamic_score(self, indicator_type: str, data: pd.DataFrame) -> float:
        """
        动态获取指标评分

        Args:
            indicator_type: 指标类型
            data: 股票数据

        Returns:
            float: 指标评分
        """
        try:
            # 静态评分作为默认值
            static_scores = {
                "MACD": 75.0, "KDJ": 70.0, "RSI": 65.0, "BOLL": 70.0, "MA": 60.0,
                "VOL": 65.0, "DMI": 70.0, "CCI": 65.0, "WR": 60.0, "OBV": 65.0,
                "PSY": 55.0, "BIAS": 60.0, "ROC": 65.0, "EMV": 55.0, "SAR": 70.0,
                "DMA": 60.0, "MTM": 65.0, "TRIX": 65.0, "STOCHRSI": 70.0
            }

            # 尝试使用CompleteIndicatorRegistry动态计算
            if hasattr(self, 'indicator_registry') and self.indicator_registry is not None:
                try:
                    indicator = self.indicator_registry.create_indicator(indicator_type)
                    if indicator and hasattr(indicator, 'calculate_raw_score'):
                        score = indicator.calculate_raw_score(data)
                        if score is not None and len(score) > 0:
                            return float(score.iloc[-1])
                except Exception as e:
                    logger.debug(f"动态计算指标 {indicator_type} 评分失败: {e}")

            # 回退到静态评分
            return static_scores.get(indicator_type, 60.0)

        except Exception as e:
            logger.debug(f"动态获取指标 {indicator_type} 评分失败: {e}")
            return 60.0
    
    def _calculate_trend_score_Strategy_Executor(self, data: pd.DataFrame) -> float:
        """
        计算趋势强度评分
        
        Args:
            data: 股票数据
            
        Returns:
            float: 趋势评分（0-100）
        """
        try:
            if len(data) < 30 or 'close' not in data.columns:
                return 50.0
                
            # 提取收盘价
            close = data['close'].values
            
            # 计算多个周期的均线
            ma5 = np.convolve(close, np.ones(5)/5, mode='valid')
            ma10 = np.convolve(close, np.ones(10)/10, mode='valid')
            ma20 = np.convolve(close, np.ones(20)/20, mode='valid')
            ma30 = np.convolve(close, np.ones(30)/30, mode='valid')
            
            # 确保所有均线都有足够的数据点
            min_len = min(len(ma5), len(ma10), len(ma20), len(ma30))
            if min_len < 3:
                return 50.0
                
            # 截取最近的数据点进行比较
            ma5 = ma5[-min_len:]
            ma10 = ma10[-min_len:]
            ma20 = ma20[-min_len:]
            ma30 = ma30[-min_len:]
            
            # 计算均线排列得分（多头排列：ma5 > ma10 > ma20 > ma30，满分40分）
            ma_alignment_score = 0
            
            # 检查最近的均线排列情况
            if ma5[-1] > ma10[-1]: ma_alignment_score += 10
            if ma10[-1] > ma20[-1]: ma_alignment_score += 10
            if ma20[-1] > ma30[-1]: ma_alignment_score += 10
            
            # 检查均线斜率（上升趋势），每个均线斜率为正加5分，最多20分
            slope_score = 0
            
            if ma5[-1] > ma5[-3]: slope_score += 5
            if ma10[-1] > ma10[-3]: slope_score += 5
            if ma20[-1] > ma20[-3]: slope_score += 5
            if ma30[-1] > ma30[-3]: slope_score += 5
            
            # 计算价格与均线的关系得分，价格站上均线为正面信号（最多20分）
            price_ma_score = 0
            latest_price = close[-1]
            
            if latest_price > ma5[-1]: price_ma_score += 5
            if latest_price > ma10[-1]: price_ma_score += 5
            if latest_price > ma20[-1]: price_ma_score += 5
            if latest_price > ma30[-1]: price_ma_score += 5
            
            # 计算趋势持续性得分（均线交叉情况，最多20分）
            # 如果短期均线刚刚向上穿越长期均线，这是看涨信号
            continuity_score = 0
            
            # 检查MA5是否刚刚上穿MA10
            if ma5[-1] > ma10[-1] and ma5[-2] <= ma10[-2]: continuity_score += 7
            
            # 检查MA10是否刚刚上穿MA20
            if ma10[-1] > ma20[-1] and ma10[-2] <= ma20[-2]: continuity_score += 7
            
            # 检查MA20是否刚刚上穿MA30
            if ma20[-1] > ma30[-1] and ma20[-2] <= ma30[-2]: continuity_score += 6
            
            # 综合得分（总分100分）
            trend_score = ma_alignment_score + slope_score + price_ma_score + continuity_score
            
            # 对于极强的上升趋势（所有指标都满足），给予额外奖励分数
            if trend_score > 80:
                trend_score += (100 - trend_score) * 0.5
                
            # 确保分数在0-100范围内
            trend_score = max(0, min(100, trend_score))
            
            return trend_score
            
        except Exception as e:
            logger.error(f"计算趋势评分时出错: {e}")
            return 50.0
    
    def _calculate_momentum_score_Strategy_Executor(self, data: pd.DataFrame) -> float:
        """
        计算动量评分
        
        Args:
            data: 股票数据
            
        Returns:
            float: 动量评分（0-100）
        """
        try:
            if len(data) < 30 or 'close' not in data.columns:
                return 50.0
                
            # 计算不同周期的涨幅
            close = data['close']
            change_3d = (close.iloc[-1] / close.iloc[-4] - 1) * 100 if len(close) >= 4 else 0
            change_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
            change_10d = (close.iloc[-1] / close.iloc[-11] - 1) * 100 if len(close) >= 11 else 0
            change_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
            change_30d = (close.iloc[-1] / close.iloc[-31] - 1) * 100 if len(close) >= 31 else 0
            
            # 根据涨幅计算得分
            score_3d = min(100, max(0, 50 + change_3d * 12))  # 每1%对应12分
            score_5d = min(100, max(0, 50 + change_5d * 10))  # 每1%对应10分
            score_10d = min(100, max(0, 50 + change_10d * 5))  # 每1%对应5分
            score_20d = min(100, max(0, 50 + change_20d * 2.5))  # 每1%对应2.5分
            score_30d = min(100, max(0, 50 + change_30d * 2))  # 每1%对应2分
            
            # 加权平均（近期权重高）
            momentum_score = (
                score_3d * 0.25 + 
                score_5d * 0.25 + 
                score_10d * 0.2 + 
                score_20d * 0.15 + 
                score_30d * 0.15
            )
            
            # 连续上涨天数加分
            up_days = 0
            for i in range(1, min(11, len(close))):
                if close.iloc[-i] > close.iloc[-i-1]:
                    up_days += 1
                else:
                    break
            
            # 根据连续上涨天数加分（最多加10分）
            if up_days >= 3:
                bonus = min(10, up_days * 2)
                momentum_score += bonus
                
            # 确保分数在0-100范围内
            momentum_score = max(0, min(100, momentum_score))
            
            return momentum_score
            
        except Exception as e:
            logger.error(f"计算动量评分时出错: {e}")
            return 50.0
    
    def _calculate_volume_score_Strategy_Executor(self, data: pd.DataFrame) -> float:
        """
        计算成交量评分
        
        Args:
            data: 股票数据
            
        Returns:
            float: 成交量评分（0-100）
        """
        try:
            if len(data) < 30 or 'volume' not in data.columns:
                return 50.0
                
            # 计算近期成交量变化
            volume = data['volume']
            recent_vol = volume.iloc[-5:].mean()
            prev_vol = volume.iloc[-10:-5].mean()
            long_term_vol = volume.iloc[-30:].mean()
            
            # 成交量变化率
            vol_change_ratio = recent_vol / prev_vol if prev_vol > 0 else 1.0
            vol_vs_long_term = recent_vol / long_term_vol if long_term_vol > 0 else 1.0
            
            # 计算基础分数（根据近期成交量变化）
            if vol_change_ratio >= 2.0:  # 成交量翻倍
                base_vol_score = 90.0
            elif vol_change_ratio >= 1.5:  # 成交量增加50%
                base_vol_score = 80.0
            elif vol_change_ratio >= 1.2:  # 成交量增加20%
                base_vol_score = 70.0
            elif vol_change_ratio >= 1.0:  # 成交量基本持平
                base_vol_score = 60.0
            elif vol_change_ratio >= 0.8:  # 成交量下降不多
                base_vol_score = 50.0
            elif vol_change_ratio >= 0.5:  # 成交量明显下降
                base_vol_score = 40.0
            else:  # 成交量大幅下降
                base_vol_score = 30.0
                
            # 计算成交量突增评分（近期单日放量）
            vol_spike_score = 0
            
            # 检查近5天内是否有单日成交量是20日平均的2倍以上
            for i in range(1, min(6, len(volume))):
                daily_vol = volume.iloc[-i]
                avg_20d_vol = volume.iloc[-20-i:-i].mean() if i < len(volume) - 20 else volume.iloc[:-i].mean()
                
                if daily_vol >= 2 * avg_20d_vol:
                    vol_spike_score = 20
                    break
                elif daily_vol >= 1.5 * avg_20d_vol:
                    vol_spike_score = 15
                    break
                elif daily_vol >= 1.3 * avg_20d_vol:
                    vol_spike_score = 10
                    break
            
            # 检查放量配合价格变化
            price_volume_score = 0
            if 'close' in data.columns:
                close = data['close']
                price_change = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
                
                # 放量上涨加分，放量下跌减分
                if vol_change_ratio > 1.2:
                    if price_change > 3:  # 明显上涨
                        price_volume_score = 15
                    elif price_change > 0:  # 轻微上涨
                        price_volume_score = 10
                    elif price_change < -3:  # 明显下跌
                        price_volume_score = -15
                    elif price_change < 0:  # 轻微下跌
                        price_volume_score = -10
            
            # 综合成交量评分
            volume_score = base_vol_score + vol_spike_score + price_volume_score
                    
            # 确保分数在0-100范围内
            volume_score = max(0, min(100, volume_score))
            
            return volume_score
            
        except Exception as e:
            logger.error(f"计算成交量评分时出错: {e}")
            return 50.0
    
    def _calculate_volatility_score_Strategy_Executor(self, data: pd.DataFrame) -> float:
        """
        计算波动性评分（较低的波动性得高分，但不能太低）
        
        Args:
            data: 股票数据
            
        Returns:
            float: 波动性评分（0-100）
        """
        try:
            if len(data) < 30 or 'close' not in data.columns:
                return 50.0
                
            # 计算近期收益率的标准差（波动率）
            close = data['close']
            returns = close.pct_change().dropna()
            
            # 计算不同周期的波动率
            volatility_10d = returns.iloc[-10:].std() * 100 if len(returns) >= 10 else 0
            volatility_20d = returns.iloc[-20:].std() * 100 if len(returns) >= 20 else 0
            volatility_30d = returns.iloc[-30:].std() * 100 if len(returns) >= 30 else 0
            
            # 波动率评分（适中的波动率得高分）
            # 过低的波动率（小于1%）表示交易不活跃，得分适中
            # 适中的波动率（1-2%）得分最高
            # 较高的波动率（2-3%）得分中等
            # 过高的波动率（>3%）得分低
            
            def volatility_to_score(vol):
                if vol < 0.5:  # 极低波动率
                    return 40.0
                elif vol < 1.0:  # 低波动率
                    return 60.0
                elif vol < 1.5:  # 适中低波动率
                    return 90.0
                elif vol < 2.0:  # 适中波动率
                    return 80.0
                elif vol < 2.5:  # 适中高波动率
                    return 70.0
                elif vol < 3.0:  # 高波动率
                    return 50.0
                else:  # 极高波动率
                    return 30.0
            
            # 计算不同周期波动率的得分
            score_10d = volatility_to_score(volatility_10d)
            score_20d = volatility_to_score(volatility_20d)
            score_30d = volatility_to_score(volatility_30d)
            
            # 加权平均（短期波动率权重高）
            volatility_score = score_10d * 0.5 + score_20d * 0.3 + score_30d * 0.2
            
            # 额外检查：如果价格近期出现剧烈波动（日K线实体较大），适当减分
            high = data['high'] if 'high' in data.columns else None
            low = data['low'] if 'low' in data.columns else None
            
            if high is not None and low is not None:
                # 计算最近5天的日振幅
                amplitude = ((high.iloc[-5:] - low.iloc[-5:]) / low.iloc[-5:]).mean() * 100
                
                # 如果振幅过大（>5%），减分
                if amplitude > 7:
                    volatility_score -= 15
                elif amplitude > 5:
                    volatility_score -= 10
                elif amplitude > 3:
                    volatility_score -= 5
            
            # 确保分数在0-100范围内
            volatility_score = max(0, min(100, volatility_score))
            
            return volatility_score
            
        except Exception as e:
            logger.error(f"计算波动性评分时出错: {e}")
            return 50.0
    
    def _calculate_market_score(self, stock_code: str, data: pd.DataFrame) -> float:
        """
        计算市场环境相关评分
        
        Args:
            stock_code: 股票代码
            data: 股票数据
            
        Returns:
            float: 市场环境评分（0-100）
        """
        try:
            # 获取指数数据（上证指数作为基准）
            index_code = "000001.SH"  # 上证指数
            end_date = data.index[-1].strftime("%Y-%m-%d") if isinstance(data.index[-1], pd.Timestamp) else data.index[-1]
            start_date = data.index[0].strftime("%Y-%m-%d") if isinstance(data.index[0], pd.Timestamp) else data.index[0]
            
            try:
                index_data = self.data_access.get_index_data(index_code, start_date=start_date, end_date=end_date)
                if index_data is None or len(index_data) < 10:
                    return 50.0  # 无法获取指数数据，返回中性分数
            except:
                return 50.0  # 获取指数数据出错，返回中性分数
            
            # 计算大盘趋势分数
            market_trend_score = 0
            
            # 计算指数短期趋势
            index_close = index_data['close']
            index_ma5 = index_close.rolling(5).mean()
            index_ma10 = index_close.rolling(10).mean()
            index_ma20 = index_close.rolling(20).mean()
            
            # 检查大盘均线排列
            if len(index_ma20.dropna()) > 0:
                last_idx = -1
                if index_close.iloc[last_idx] > index_ma5.iloc[last_idx]: market_trend_score += 10
                if index_ma5.iloc[last_idx] > index_ma10.iloc[last_idx]: market_trend_score += 10
                if index_ma10.iloc[last_idx] > index_ma20.iloc[last_idx]: market_trend_score += 10
                if index_close.iloc[last_idx] > index_close.iloc[last_idx-5]: market_trend_score += 10
            
            # 计算个股相对强度得分
            relative_strength_score = 0
            
            # 计算个股和大盘的近期表现对比
            stock_close = data['close']
            if len(stock_close) > 0 and len(index_close) > 0:
                # 计算5日、10日涨幅对比
                stock_change_5d = (stock_close.iloc[-1] / stock_close.iloc[-6]) - 1 if len(stock_close) >= 6 else 0
                index_change_5d = (index_close.iloc[-1] / index_close.iloc[-6]) - 1 if len(index_close) >= 6 else 0
                
                stock_change_10d = (stock_close.iloc[-1] / stock_close.iloc[-11]) - 1 if len(stock_close) >= 11 else 0
                index_change_10d = (index_close.iloc[-1] / index_close.iloc[-11]) - 1 if len(index_close) >= 11 else 0
                
                # 计算相对强度（个股表现减去大盘表现）
                rs_5d = stock_change_5d - index_change_5d
                rs_10d = stock_change_10d - index_change_10d
                
                # 根据相对强度评分
                if rs_5d > 0.05: relative_strength_score += 30  # 强于大盘5%以上
                elif rs_5d > 0.03: relative_strength_score += 25
                elif rs_5d > 0.01: relative_strength_score += 20
                elif rs_5d > 0: relative_strength_score += 15
                else: relative_strength_score += 5
                
                if rs_10d > 0.08: relative_strength_score += 30  # 强于大盘8%以上
                elif rs_10d > 0.05: relative_strength_score += 25
                elif rs_10d > 0.02: relative_strength_score += 20
                elif rs_10d > 0: relative_strength_score += 15
                else: relative_strength_score += 5
            
            # 计算最终市场评分（大盘趋势占40%，相对强度占60%）
            market_score = market_trend_score * 0.4 + relative_strength_score * 0.6
            
            # 确保分数在0-100范围内
            market_score = max(0, min(100, market_score))
            
            return market_score
            
        except Exception as e:
            logger.error(f"计算市场环境评分时出错: {e}")
            return 50.0
    
    def _get_filtered_stock_list(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        获取经过过滤的股票列表

        Args:
            filters: 过滤条件

        Returns:
            股票列表Data_frame
        """
        # 如果指定了股票代码列表，直接使用
        if 'stock_codes' in filters and filters['stock_codes']:
            stock_codes = filters['stock_codes']
            if isinstance(stock_codes, list):
                return pd.DataFrame({'stock_code': stock_codes})
            else:
                return pd.DataFrame({'stock_code': [stock_codes]})
        
        # 转换filters参数为DataManagerAdapter支持的参数
        market = filters.get('market')
        industry = filters.get('industry')
        limit = filters.get('limit')

        # 调用DataManagerAdapter的get_stock_list方法
        stock_list = self.data_access.get_stock_list(
            market=market,
            industry=industry,
            limit=limit
        )

        # 如果返回的是列表，转换为DataFrame
        if isinstance(stock_list, list):
            if stock_list and isinstance(stock_list[0], str):
                # 如果是股票代码列表
                return pd.DataFrame({'stock_code': stock_list})
            else:
                # 如果是字典列表
                return pd.DataFrame(stock_list)
        elif isinstance(stock_list, pd.DataFrame):
            return stock_list
        else:
            # 备用方案：获取所有股票
            all_stocks = self.data_access.get_all_stock_list()
            return pd.DataFrame(all_stocks)
    
    def _validate_strategy_plan(self, strategy_plan: Dict[str, Any]) -> bool:
        """
        验证策略执行计划
        
        Args:
            strategy_plan: 策略执行计划
            
        Returns:
            验证通过返回True
            
        Raises:
            StrategyValidationError: 策略验证错误
        """
        # 检查必要字段
        required_fields = ['strategy_id', 'name', 'conditions']
        for field in required_fields:
            if field not in strategy_plan:
                raise StrategyValidationError(f"策略执行计划缺少必要字段: {field}")
        
        # 检查条件列表
        conditions = strategy_plan.get('conditions', [])
        if not conditions:
            raise StrategyValidationError("策略执行计划缺少条件")
        
        # 检查条件列表的有效性
        for condition in conditions:
            if 'logic' in condition:
                # 逻辑运算符
                if condition['logic'].upper() not in ['AND', 'OR']:
                    raise StrategyValidationError(f"不支持的逻辑运算符: {condition['logic']}")
            elif 'type' in condition and condition['type'] == 'logic':
                # 新格式的逻辑运算符
                if condition.get('value', '').upper() not in ['AND', 'OR']:
                    raise StrategyValidationError(f"不支持的逻辑运算符: {condition.get('value')}")
            elif 'type' in condition and condition['type'] == 'indicator':
                # 新格式的指标条件
                if 'indicator_id' not in condition:
                    raise StrategyValidationError(f"条件缺少必要字段: indicator_id")
                if 'period' not in condition:
                    raise StrategyValidationError(f"条件缺少必要字段: period")
            elif 'type' in condition and condition['type'] == 'basic':
                # 基础条件（价格、成交量等），不需要indicator_id和period
                if 'field' not in condition:
                    raise StrategyValidationError(f"基础条件缺少必要字段: field")
                if 'operator' not in condition:
                    raise StrategyValidationError(f"基础条件缺少必要字段: operator")
                if 'value' not in condition:
                    raise StrategyValidationError(f"基础条件缺少必要字段: value")
            elif 'indicator_id' in condition:
                # 旧格式的指标条件，需要验证indicator_id和period
                if 'period' not in condition:
                    raise StrategyValidationError(f"条件缺少必要字段: period")
            else:
                # 其他类型的条件，暂时跳过验证
                pass
        
        return True
    
    def clear_cache_Executor(self):
        """清除结果缓存"""
        old_size = len(self.cache)
        self.cache.clear()
        logger.info(f"已清除策略执行器缓存，共 {old_size} 项")
    
    def get_cache_stats_Executor(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        return {
            'enabled': self.cache_enabled,
            'size': len(self.cache),
            'keys': list(self.cache.keys())
        } 