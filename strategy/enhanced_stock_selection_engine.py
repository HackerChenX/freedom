from strategy.unified_base_strategy import UnifiedBaseStrategy
"""
增强选股执行引擎

高效的股票筛选和评分系统，支持多维度评估和实时选股
遵循六层架构规范，提供高性能的选股执行能力
"""

import os
import asyncio
import concurrent.futures
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import time

from utils.logger import get_logger, get_service
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container
from db.interfaces.data_access_interface import DataAccessInterface
from indicators.complete_indicator_registry import complete_registry
from strategy.enhanced_strategy_config_engine import StrategyConfig, StrategyRule, StrategyCondition
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


@dataclass
class StockSelectionResult:
    """选股结果数据类"""
    stock_code: str
    stock_name: str
    score: float
    rank: int
    matched_conditions: List[str]
    analysis_details: Dict[str, Any]
    selection_time: str
    recommendation: str


@dataclass
class SelectionMetrics:
    """选股指标数据类"""
    total_stocks: int
    analyzed_stocks: int
    selected_stocks: int
    execution_time: float
    average_score: float
    success_rate: float


class EnhancedStockSelectionEngine:
    """
    增强选股执行引擎
    
    提供高效的股票筛选和评分功能
    """
    
    def __init__(self, max_workers: int = None):
        """初始化选股执行引擎"""
        self.logger = logger
        self.data_access = get_service(DataAccessInterface)
        self.indicator_registry = complete_registry
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 4)
        
        # 缓存配置
        self.cache_enabled = True
        self.cache = {}
        self.cache_ttl = 300  # 5分钟缓存
        
        # 性能统计
        self.performance_stats = {
            'total_selections': 0,
            'total_stocks_analyzed': 0,
            'total_execution_time': 0.0,
            'average_selection_time': 0.0,
            'cache_hit_rate': 0.0
        }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=10.0)
    def execute_selection(self, strategy_config: StrategyConfig, 
                         stock_pool: Optional[List[str]] = None,
                         selection_date: Optional[str] = None) -> Dict[str, Any]:
        """
        执行选股策略
        
        Args:
            strategy_config: 策略配置
            stock_pool: 股票池，None表示使用全市场
            selection_date: 选股日期，None表示使用最新日期
            
        Returns:
            Dict[str, Any]: 选股结果
        """
        start_time = time.time()
        self.logger.info(f"开始执行选股策略: {strategy_config.name}")
        
        # 准备股票池
        if stock_pool is None:
            stock_pool = self._get_default_stock_pool()
        
        # 准备选股日期
        if selection_date is None:
            selection_date = self._get_latest_trading_date()
        
        # 执行选股
        selection_results = []
        analysis_details = {}
        
        # 并行处理股票
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_stock = {
                executor.submit(self._analyze_single_stock, stock_code, strategy_config, selection_date): stock_code
                for stock_code in stock_pool
            }
            
            for future in concurrent.futures.as_completed(future_to_stock):
                stock_code = future_to_stock[future]
                try:
                    result = future.result()
                    if result and result['is_selected']:
                        selection_results.append(result)
                    analysis_details[stock_code] = result
                except Exception as e:
                    self.logger.error(f"分析股票 {stock_code} 时出错: {e}")
                    analysis_details[stock_code] = {
                        'stock_code': stock_code,
                        'error': str(e),
                        'is_selected': False
                    }
        
        # 排序和排名
        selection_results.sort(key=lambda x: x['score'], reverse=True)
        for i, result in enumerate(selection_results):
            result['rank'] = i + 1
        
        # 计算指标
        execution_time = time.time() - start_time
        metrics = self._calculate_selection_metrics(
            len(stock_pool), len(analysis_details), len(selection_results), execution_time
        )
        
        # 更新性能统计
        self._update_performance_stats(metrics)
        
        result = {
            'strategy_id': strategy_config.strategy_id,
            'strategy_name': strategy_config.name,
            'selection_date': selection_date,
            'selection_time': datetime.now().isoformat(),
            'selected_stocks': selection_results,
            'analysis_details': analysis_details,
            'metrics': metrics,
            'summary': self._generate_selection_summary(selection_results, metrics)
        }
        
        self.logger.info(f"选股完成，选中 {len(selection_results)} 只股票，耗时 {execution_time:.2f}秒")
        return result
    
    def _analyze_single_stock(self, stock_code: str, strategy_config: StrategyConfig, 
                             selection_date: str) -> Dict[str, Any]:
        """分析单只股票"""
        try:
            # 检查缓存
            cache_key = f"{stock_code}_{strategy_config.strategy_id}_{selection_date}"
            if self.cache_enabled and cache_key in self.cache:
                cache_data = self.cache[cache_key]
                if time.time() - cache_data['timestamp'] < self.cache_ttl:
                    return cache_data['result']
            
            # 获取股票数据
            stock_data = self._get_stock_data(stock_code, selection_date)
            if stock_data is None or stock_data.empty:
                return {
                    'stock_code': stock_code,
                    'error': '无法获取股票数据',
                    'is_selected': False,
                    'score': 0.0
                }
            
            # 计算技术指标
            indicators = self._calculate_indicators(stock_data)
            
            # 评估策略规则
            rule_results = []
            total_score = 0.0
            matched_conditions = []
            
            for rule in strategy_config.rules:
                rule_result = self._evaluate_rule(rule, indicators, stock_data)
                rule_results.append(rule_result)
                
                if rule_result['is_matched']:
                    total_score += rule_result['score']
                    matched_conditions.extend(rule_result['matched_conditions'])
            
            # 计算最终评分
            final_score = total_score / len(strategy_config.rules) if strategy_config.rules else 0.0
            
            # 判断是否选中
            min_score = strategy_config.global_settings.get('min_score', 60.0)
            is_selected = final_score >= min_score and len(matched_conditions) > 0
            
            result = {
                'stock_code': stock_code,
                'stock_name': stock_data['name'].iloc[0] if 'name' in stock_data.columns else '',
                'score': final_score,
                'is_selected': is_selected,
                'matched_conditions': matched_conditions,
                'rule_results': rule_results,
                'indicators': indicators,
                'analysis_time': datetime.now().isoformat(),
                'recommendation': self._generate_recommendation(final_score, is_selected)
            }
            
            # 缓存结果
            if self.cache_enabled:
                self.cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"分析股票 {stock_code} 时出错: {e}")
            return {
                'stock_code': stock_code,
                'error': str(e),
                'is_selected': False,
                'score': 0.0
            }
    
    def _get_stock_data(self, stock_code: str, selection_date: str) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            # 计算数据范围（需要足够的历史数据计算指标）
            end_date = datetime.strptime(selection_date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=120)  # 4个月历史数据
            
            return self.data_access.get_stock_data(
                stock_code=stock_code,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=selection_date,
                level='日线'
            )
        except Exception as e:
            self.logger.error(f"获取股票 {stock_code} 数据时出错: {e}")
            return None
    
    def _calculate_indicators(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """计算技术指标"""
        indicators = {}
        
        try:
            # 计算常用指标
            for indicator_name in ['MACD', 'RSI', 'KDJ', 'BOLL', 'MA', 'VOL']:
                try:
                    if indicator_name in self.indicator_registry:
                        indicator_class = self.indicator_registry[indicator_name]
                        indicator = indicator_class()
                        result = indicator.calculate(stock_data)
                        indicators[indicator_name] = result
                except Exception as e:
                    self.logger.warning(f"计算指标 {indicator_name} 时出错: {e}")
                    continue
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"计算指标时出错: {e}")
            return {}
    
    def _evaluate_rule(self, rule: StrategyRule, indicators: Dict[str, Any], 
                      stock_data: pd.DataFrame) -> Dict[str, Any]:
        """评估策略规则"""
        try:
            matched_conditions = []
            condition_scores = []
            
            for condition in rule.conditions:
                condition_result = self._evaluate_condition(condition, indicators, stock_data)
                
                if condition_result['is_matched']:
                    matched_conditions.append(condition_result['description'])
                    condition_scores.append(condition_result['score'] * condition.weight)
                elif condition.required:
                    # 必需条件不满足，规则失败
                    return {
                        'rule_id': rule.rule_id,
                        'rule_name': rule.name,
                        'is_matched': False,
                        'score': 0.0,
                        'matched_conditions': [],
                        'reason': f"必需条件不满足: {condition.pattern}"
                    }
            
            # 计算规则得分
            if condition_scores:
                rule_score = sum(condition_scores) / len(condition_scores)
            else:
                rule_score = 0.0
            
            # 判断规则是否匹配
            is_matched = rule_score >= rule.min_score and len(matched_conditions) > 0
            
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.name,
                'is_matched': is_matched,
                'score': rule_score,
                'matched_conditions': matched_conditions,
                'condition_count': len(matched_conditions),
                'total_conditions': len(rule.conditions)
            }
            
        except Exception as e:
            self.logger.error(f"评估规则 {rule.name} 时出错: {e}")
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.name,
                'is_matched': False,
                'score': 0.0,
                'matched_conditions': [],
                'error': str(e)
            }

    def _evaluate_condition(self, condition: StrategyCondition, indicators: Dict[str, Any],
                           stock_data: pd.DataFrame) -> Dict[str, Any]:
        """评估策略条件"""
        try:
            # 获取指标数据
            if condition.indicator not in indicators:
                return {
                    'is_matched': False,
                    'score': 0.0,
                    'description': f"{condition.indicator} 指标数据不可用",
                    'error': f"指标 {condition.indicator} 未计算"
                }

            indicator_data = indicators[condition.indicator]

            # 检查形态
            pattern_matched = self._check_pattern(condition, indicator_data, stock_data)

            if pattern_matched['is_matched']:
                return {
                    'is_matched': True,
                    'score': pattern_matched['score'],
                    'description': f"{condition.indicator}.{condition.pattern} 形态匹配",
                    'details': pattern_matched['details']
                }
            else:
                return {
                    'is_matched': False,
                    'score': 0.0,
                    'description': f"{condition.indicator}.{condition.pattern} 形态不匹配",
                    'reason': pattern_matched.get('reason', '未知原因')
                }

        except Exception as e:
            self.logger.error(f"评估条件时出错: {e}")
            return {
                'is_matched': False,
                'score': 0.0,
                'description': f"条件评估失败: {str(e)}",
                'error': str(e)
            }

    def _check_pattern(self, condition: StrategyCondition, indicator_data: Any,
                      stock_data: pd.DataFrame) -> Dict[str, Any]:
        """检查技术形态"""
        try:
            # 简化的形态检查逻辑
            # 实际实现中应该调用具体指标的形态识别方法

            if isinstance(indicator_data, pd.DataFrame) and not indicator_data.empty:
                # 获取最新数据点
                latest_data = indicator_data.iloc[-1]

                # 基于形态名称进行简单匹配
                pattern_score = self._calculate_pattern_score(condition.pattern, latest_data, indicator_data)

                if pattern_score > 0.6:  # 阈值可配置
                    return {
                        'is_matched': True,
                        'score': pattern_score * 100,
                        'details': {
                            'pattern': condition.pattern,
                            'score': pattern_score,
                            'latest_value': latest_data.to_dict() if hasattr(latest_data, 'to_dict') else str(latest_data)
                        }
                    }

            return {
                'is_matched': False,
                'score': 0.0,
                'reason': '形态条件不满足'
            }

        except Exception as e:
            self.logger.error(f"检查形态时出错: {e}")
            return {
                'is_matched': False,
                'score': 0.0,
                'reason': f"形态检查失败: {str(e)}"
            }

    def _calculate_pattern_score(self, pattern_name: str, latest_data: Any,
                                historical_data: pd.DataFrame) -> float:
        """计算形态得分"""
        try:
            # 简化的形态评分逻辑
            # 实际实现中应该根据具体形态进行详细计算

            base_score = 0.5  # 基础分数

            # 根据形态名称调整分数
            if 'golden_cross' in pattern_name.lower():
                # 金叉形态检查
                if isinstance(latest_data, dict) and 'dif' in latest_data and 'dea' in latest_data:
                    if latest_data['dif'] > latest_data['dea']:
                        base_score += 0.3
            elif 'oversold' in pattern_name.lower():
                # 超卖形态检查
                if isinstance(latest_data, (int, float)) and latest_data < 30:
                    base_score += 0.3
            elif 'volume_surge' in pattern_name.lower():
                # 放量形态检查
                if len(historical_data) >= 5:
                    recent_volume = historical_data['volume'].iloc[-1] if 'volume' in historical_data.columns else 0
                    avg_volume = historical_data['volume'].iloc[-5:].mean() if 'volume' in historical_data.columns else 0
                    if recent_volume > avg_volume * 1.5:
                        base_score += 0.3

            return min(1.0, base_score)

        except Exception as e:
            self.logger.error(f"计算形态得分时出错: {e}")
            return 0.0

    def _get_default_stock_pool(self) -> List[str]:
        """获取默认股票池"""
        try:
            # 获取主板股票池（简化实现）
            return ['000001', '000002', '000858', '600519', '000858']  # 示例股票池
        except Exception as e:
            self.logger.error(f"获取默认股票池时出错: {e}")
            return []

    def _get_latest_trading_date(self) -> str:
        """获取最新交易日期"""
        try:
            # 简化实现，返回当前日期
            return datetime.now().strftime('%Y-%m-%d')
        except Exception as e:
            self.logger.error(f"获取最新交易日期时出错: {e}")
            return datetime.now().strftime('%Y-%m-%d')

    def _calculate_selection_metrics(self, total_stocks: int, analyzed_stocks: int,
                                   selected_stocks: int, execution_time: float) -> SelectionMetrics:
        """计算选股指标"""
        return SelectionMetrics(
            total_stocks=total_stocks,
            analyzed_stocks=analyzed_stocks,
            selected_stocks=selected_stocks,
            execution_time=execution_time,
            average_score=0.0,  # 需要从结果中计算
            success_rate=selected_stocks / analyzed_stocks if analyzed_stocks > 0 else 0.0
        )

    def _update_performance_stats(self, metrics: SelectionMetrics):
        """更新性能统计"""
        self.performance_stats['total_selections'] += 1
        self.performance_stats['total_stocks_analyzed'] += metrics.analyzed_stocks
        self.performance_stats['total_execution_time'] += metrics.execution_time

        if self.performance_stats['total_selections'] > 0:
            self.performance_stats['average_selection_time'] = (
                self.performance_stats['total_execution_time'] /
                self.performance_stats['total_selections']
            )

    def _generate_selection_summary(self, selection_results: List[Dict[str, Any]],
                                  metrics: SelectionMetrics) -> Dict[str, Any]:
        """生成选股摘要"""
        if not selection_results:
            return {
                'total_selected': 0,
                'average_score': 0.0,
                'top_stocks': [],
                'performance': asdict(metrics)
            }

        scores = [result['score'] for result in selection_results]

        return {
            'total_selected': len(selection_results),
            'average_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'top_stocks': selection_results[:10],  # 前10只股票
            'performance': asdict(metrics),
            'selection_distribution': self._calculate_score_distribution(scores)
        }

    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """计算评分分布"""
        distribution = {
            '90-100': 0,
            '80-89': 0,
            '70-79': 0,
            '60-69': 0,
            '50-59': 0,
            '<50': 0
        }

        for score in scores:
            if score >= 90:
                distribution['90-100'] += 1
            elif score >= 80:
                distribution['80-89'] += 1
            elif score >= 70:
                distribution['70-79'] += 1
            elif score >= 60:
                distribution['60-69'] += 1
            elif score >= 50:
                distribution['50-59'] += 1
            else:
                distribution['<50'] += 1

        return distribution

    def _generate_recommendation(self, score: float, is_selected: bool) -> str:
        """生成投资建议"""
        if not is_selected:
            return "不推荐"

        if score >= 90:
            return "强烈推荐"
        elif score >= 80:
            return "推荐"
        elif score >= 70:
            return "谨慎推荐"
        elif score >= 60:
            return "观望"
        else:
            return "不推荐"

    @exception_handler(reraise=False, default_return={})
    @performance_monitor(threshold=1.0)
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'performance_stats': self.performance_stats.copy(),
            'cache_stats': {
                'cache_size': len(self.cache),
                'cache_enabled': self.cache_enabled,
                'cache_ttl': self.cache_ttl
            },
            'system_stats': {
                'max_workers': self.max_workers,
                'indicators_available': getattr(self.indicator_registry, '__len__', lambda: 0)() if hasattr(self.indicator_registry, '__len__') else len(getattr(self.indicator_registry, 'registry', {}))
            }
        }
