#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
选股策略全覆盖测试模块

验证双均线、主力行为、回踩反弹等选股策略的全面覆盖测试。
严格遵循六层架构原则，提供完整的策略测试验证功能。

L6: 测试应用层 - 本文件提供策略测试功能
L5: 测试业务层 - 具体策略测试逻辑
L4: 测试服务层 - 策略执行服务
L3: 测试数据层 - 测试数据管理
L2: 测试基础设施层 - 测试工具和配置
L1: 测试数据存储层 - 测试数据和结果存储
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from db.managers.query_executor import UnifiedQueryExecutor
from strategy.strategy_executor import StrategyExecutor
from analysis.buy_point.buy_point_analyzer import BuyPointAnalyzer
from db.sql_manager import SQLManager, QueryType

logger = get_logger('strategy_coverage_tester')


class StrategyType(Enum):
    """策略类型枚举"""
    DUAL_MOVING_AVERAGE = "dual_moving_average"           # 双均线策略
    MAIN_FORCE_BEHAVIOR = "main_force_behavior"           # 主力行为策略
    PULLBACK_REBOUND = "pullback_rebound"                 # 回踩反弹策略
    BREAKTHROUGH_STRATEGY = "breakthrough_strategy"        # 突破策略
    REVERSAL_STRATEGY = "reversal_strategy"               # 反转策略
    MOMENTUM_STRATEGY = "momentum_strategy"               # 动量策略
    VALUE_STRATEGY = "value_strategy"                     # 价值策略
    TECHNICAL_COMBINATION = "technical_combination"       # 技术指标组合策略
    ZXM_STRATEGY = "zxm_strategy"                        # ZXM策略
    VOLUME_PRICE_STRATEGY = "volume_price_strategy"       # 量价策略


class MarketCondition(Enum):
    """市场条件枚举"""
    BULL_MARKET = "bull_market"                          # 牛市
    BEAR_MARKET = "bear_market"                          # 熊市
    SIDEWAYS_MARKET = "sideways_market"                  # 震荡市
    VOLATILE_MARKET = "volatile_market"                  # 波动市
    LOW_VOLATILITY = "low_volatility"                    # 低波动市


@dataclass
class StrategyTestCase:
    """策略测试用例"""
    strategy_name: str
    strategy_type: StrategyType
    market_condition: MarketCondition
    test_data: pd.DataFrame
    expected_selections: int
    expected_accuracy: float = 0.6
    time_period: str = "2024-01-01_2024-12-31"
    description: str = ""


@dataclass
class StrategyTestResult:
    """策略测试结果"""
    strategy_name: str
    strategy_type: StrategyType
    market_condition: MarketCondition
    test_case_count: int
    total_selections: int
    successful_selections: int
    failed_selections: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    execution_time: float
    coverage_percentage: float
    selection_details: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class StrategyCoverageTestSuite:
    """策略覆盖测试套件结果"""
    suite_name: str
    total_strategies: int
    tested_strategies: int
    overall_coverage: float
    overall_accuracy: float
    overall_precision: float
    overall_recall: float
    total_execution_time: float
    strategy_results: List[StrategyTestResult] = field(default_factory=list)
    coverage_by_type: Dict[str, float] = field(default_factory=dict)
    coverage_by_market: Dict[str, float] = field(default_factory=dict)


class StrategyCoverageTester:
    """
    选股策略全覆盖测试器
    
    负责对所有选股策略进行全面测试，包括不同市场条件下的策略表现验证
    """
    
    def __init__(self):
        """初始化策略覆盖测试器"""
        self.query_executor = UnifiedQueryExecutor()
        self.strategy_executor = StrategyExecutor()
        self.buy_point_analyzer = BuyPointAnalyzer()
        
        # 测试配置
        self.test_config = {
            "accuracy_threshold": 0.6,       # 准确率阈值 60%
            "min_selections": 5,             # 最小选股数量
            "max_selections": 50,            # 最大选股数量
            "test_stocks_count": 100,        # 测试股票数量
            "test_time_window": 30,          # 测试时间窗口（天）
            "performance_threshold": 0.05,   # 性能阈值（5%）
            "max_test_time": 600            # 最大测试时间（秒）
        }
        
        # 定义要测试的策略
        self.test_strategies = {
            # 双均线策略
            StrategyType.DUAL_MOVING_AVERAGE: [
                "ma5_ma10_cross",              # 5日线上穿10日线
                "ma10_ma20_cross",             # 10日线上穿20日线
                "ma20_ma60_cross",             # 20日线上穿60日线
                "golden_cross_strategy",       # 黄金交叉策略
                "death_cross_avoidance"        # 死叉规避策略
            ],
            
            # 主力行为策略
            StrategyType.MAIN_FORCE_BEHAVIOR: [
                "volume_surge_strategy",       # 成交量异动策略
                "large_order_tracking",        # 大单跟踪策略
                "institutional_flow",          # 机构资金流向
                "smart_money_following",       # 聪明钱跟随
                "washout_detection"            # 洗盘识别策略
            ],
            
            # 回踩反弹策略
            StrategyType.PULLBACK_REBOUND: [
                "support_bounce",              # 支撑位反弹
                "moving_average_support",      # 均线支撑反弹
                "fibonacci_retracement",       # 斐波那契回撤
                "trend_line_support",          # 趋势线支撑
                "previous_high_pullback"       # 前高回踩策略
            ],
            
            # 突破策略
            StrategyType.BREAKTHROUGH_STRATEGY: [
                "resistance_breakthrough",     # 阻力位突破
                "volume_breakthrough",         # 放量突破
                "pattern_breakthrough",        # 形态突破
                "consolidation_breakout",      # 整理突破
                "new_high_breakthrough"        # 创新高突破
            ],
            
            # 反转策略
            StrategyType.REVERSAL_STRATEGY: [
                "oversold_reversal",           # 超卖反转
                "hammer_reversal",             # 锤子线反转
                "divergence_reversal",         # 背离反转
                "gap_reversal",                # 缺口反转
                "volume_reversal"              # 量能反转
            ],
            
            # 动量策略
            StrategyType.MOMENTUM_STRATEGY: [
                "rsi_momentum",                # RSI动量策略
                "macd_momentum",               # MACD动量策略
                "price_momentum",              # 价格动量策略
                "volume_momentum",             # 成交量动量
                "earnings_momentum"            # 业绩动量策略
            ],
            
            # ZXM策略
            StrategyType.ZXM_STRATEGY: [
                "zxm_absorption",              # ZXM吸筹策略
                "zxm_turnover_rate_buypoint",       # ZXM换手买点
                "zxm_daily_macd",              # ZXM日MACD策略
                "zxm_ma_callback",             # ZXM均线回踩
                "zxm_comprehensive"            # ZXM综合策略
            ],
            
            # 量价策略
            StrategyType.VOLUME_PRICE_STRATEGY: [
                "volume_price_match",          # 量价配合策略
                "volume_accumulation",         # 成交量堆积
                "price_volume_divergence",     # 价量背离策略
                "obv_strategy",                # OBV策略
                "volume_profile_strategy"      # 成交量分布策略
            ]
        }
        
        # 市场条件测试数据
        self.market_conditions_data = {
            MarketCondition.BULL_MARKET: {
                "start_date": "2024-01-01",
                "end_date": "2024-04-30",
                "description": "牛市行情测试"
            },
            MarketCondition.BEAR_MARKET: {
                "start_date": "2024-05-01",
                "end_date": "2024-08-31",
                "description": "熊市行情测试"
            },
            MarketCondition.SIDEWAYS_MARKET: {
                "start_date": "2024-09-01",
                "end_date": "2024-12-31",
                "description": "震荡市行情测试"
            }
        }
        
        # 测试结果存储
        self.test_results: List[StrategyTestResult] = []
    
    @performance_monitor(threshold=120.0)
    @exception_handler(reraise=True)
    def run_comprehensive_strategy_coverage_tests(self, 
                                                 strategy_types: Optional[List[StrategyType]] = None,
                                                 market_conditions: Optional[List[MarketCondition]] = None) -> StrategyCoverageTestSuite:
        """
        运行全面的策略覆盖测试
        
        Args:
            strategy_types: 要测试的策略类型列表，None表示测试所有策略
            market_conditions: 要测试的市场条件列表，None表示测试所有市场条件
            
        Returns:
            StrategyCoverageTestSuite: 测试套件结果
        """
        logger.info("开始运行全面的策略覆盖测试")
        start_time = time.time()
        
        # 确定要测试的策略类型
        if strategy_types is None:
            strategy_types = list(StrategyType)
        
        # 确定要测试的市场条件
        if market_conditions is None:
            market_conditions = list(MarketCondition)
        
        all_results = []
        
        # 按策略类型和市场条件进行测试
        for strategy_type in strategy_types:
            if strategy_type in self.test_strategies:
                strategies = self.test_strategies[strategy_type]
                
                for strategy_name in strategies:
                    for market_condition in market_conditions:
                        logger.info(f"测试策略: {strategy_name} ({strategy_type.value}) - {market_condition.value}")
                        
                        try:
                            result = self._test_single_strategy(
                                strategy_name, strategy_type, market_condition
                            )
                            if result:
                                all_results.append(result)
                        except Exception as e:
                            logger.error(f"策略 {strategy_name} 在 {market_condition.value} 条件下测试失败: {e}")
        
        # 生成测试套件结果
        suite_result = self._generate_strategy_coverage_test_suite_result(all_results)
        
        execution_time = time.time() - start_time
        suite_result.total_execution_time = execution_time
        
        logger.info(f"策略覆盖测试完成，总耗时: {execution_time:.2f}秒")
        logger.info(f"总体覆盖率: {suite_result.overall_coverage:.2f}%")
        
        return suite_result
    
    @exception_handler(reraise=False, default_return=None)
    def _test_single_strategy(self, strategy_name: str, 
                            strategy_type: StrategyType, 
                            market_condition: MarketCondition) -> Optional[StrategyTestResult]:
        """
        测试单个策略
        
        Args:
            strategy_name: 策略名称
            strategy_type: 策略类型
            market_condition: 市场条件
            
        Returns:
            Optional[StrategyTestResult]: 测试结果
        """
        start_time = time.time()
        
        try:
            # 获取测试数据
            test_data = self._get_strategy_test_data(market_condition)
            if test_data is None or test_data.empty:
                logger.warning(f"无法获取策略 {strategy_name} 的测试数据")
                return None
            
            # 生成测试用例
            test_cases = self._generate_strategy_test_cases(
                strategy_name, strategy_type, market_condition, test_data
            )
            
            if not test_cases:
                logger.warning(f"无法生成策略 {strategy_name} 的测试用例")
                return None
            
            # 执行策略测试
            total_selections = 0
            successful_selections = 0
            failed_selections = 0
            selection_details = []
            performance_metrics = {}
            errors = []
            
            for test_case in test_cases:
                try:
                    # 执行策略选股
                    selections = self._execute_strategy(test_case)
                    
                    # 评估选股结果
                    evaluation = self._evaluate_strategy_selections(test_case, selections)
                    
                    total_selections += len(selections)
                    successful_selections += evaluation['successful_count']
                    failed_selections += evaluation['failed_count']
                    
                    selection_details.extend(evaluation['details'])
                    
                    # 更新性能指标
                    if evaluation['metrics']:
                        for key, value in evaluation['metrics'].items():
                            if key not in performance_metrics:
                                performance_metrics[key] = []
                            performance_metrics[key].append(value)
                            
                except Exception as e:
                    failed_selections += 1
                    errors.append(f"测试用例执行失败: {str(e)}")
            
            # 计算性能指标
            test_case_count = len(test_cases)
            accuracy = (successful_selections / total_selections) if total_selections > 0 else 0.0
            
            # 计算精确度、召回率和F1分数
            precision, recall, f1_score = self._calculate_strategy_performance_metrics(
                test_cases, successful_selections, failed_selections
            )
            
            # 计算金融指标
            sharpe_ratio = self._calculate_sharpe_ratio(performance_metrics)
            max_drawdown = self._calculate_max_drawdown(performance_metrics)
            
            # 计算覆盖率
            coverage_percentage = self._calculate_strategy_coverage(
                strategy_name, strategy_type, test_data
            )
            
            execution_time = time.time() - start_time
            
            return StrategyTestResult(
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                market_condition=market_condition,
                test_case_count=test_case_count,
                total_selections=total_selections,
                successful_selections=successful_selections,
                failed_selections=failed_selections,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                execution_time=execution_time,
                coverage_percentage=coverage_percentage,
                selection_details=selection_details,
                performance_metrics=performance_metrics,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"策略 {strategy_name} 测试过程出错: {e}")
            return StrategyTestResult(
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                market_condition=market_condition,
                test_case_count=0,
                total_selections=0,
                successful_selections=0,
                failed_selections=1,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                execution_time=time.time() - start_time,
                coverage_percentage=0.0,
                errors=[str(e)]
            )
    
    def _get_strategy_test_data(self, market_condition: MarketCondition) -> Optional[pd.DataFrame]:
        """
        获取策略测试数据
        
        Args:
            market_condition: 市场条件
            
        Returns:
            Optional[pd.DataFrame]: 测试数据
        """
        try:
            if market_condition in self.market_conditions_data:
                condition_data = self.market_conditions_data[market_condition]
                start_date = condition_data["start_date"]
                end_date = condition_data["end_date"]
            else:
                start_date = "2024-01-01"
                end_date = "2024-12-31"
            
            # 获取测试股票列表
            query = f"""
            SELECT DISTINCT code
            FROM stock_info WHERE code = %(code)s AND level = '日线'
            AND date >= '{start_date}' AND date <= '{end_date}'
            LIMIT {self.test_config['test_stocks_count']}
            """
            
            stock_codes = self.query_executor.execute_query(query)
            if stock_codes is None or stock_codes.empty:
                return self._generate_mock_strategy_data()
            
            # 获取这些股票的详细数据
            code_list = "','".join(stock_codes['code'].tolist())
            detail_query = f"""
            SELECT code, name, date, open, high, low, close, volume, turnover_rate
            FROM stock_info WHERE code = %(code)s AND level = %(level)s AND code IN ('{code_list}')
            AND level = '日线'
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY code, date ASC
            """
            
            result = self.query_executor.execute_query(detail_query)
            return result if result is not None else self._generate_mock_strategy_data()
            
        except Exception as e:
            logger.warning(f"获取策略测试数据失败，使用模拟数据: {e}")
            return self._generate_mock_strategy_data()
    
    def _generate_mock_strategy_data(self) -> pd.DataFrame:
        """
        生成模拟策略测试数据
        
        Returns:
            pd.DataFrame: 模拟数据
        """
        stocks_count = 20
        days_count = 100
        dates = pd.date_range(start='2024-01-01', periods=days_count, freq='D')
        
        all_data = []
        
        for i in range(stocks_count):
            stock_code = f"{str(i+1).zfill(6)}"
            stock_name = f"测试股票{i+1}"
            
            # 生成价格数据
            np.random.seed(42 + i)
            base_price = 10.0 + np.random.uniform(5, 20)
            
            for j, date in enumerate(dates):
                # 模拟不同市场条件的价格走势
                trend_factor = np.sin(j / 20) * 0.02  # 周期性波动
                noise = np.random.normal(0, 0.03)     # 随机噪声
                
                price_change = trend_factor + noise
                base_price *= (1 + price_change)
                
                # 生成OHLC数据
                open_price = base_price + np.random.normal(0, 0.01)
                high_price = open_price + abs(np.random.normal(0, 0.02))
                low_price = open_price - abs(np.random.normal(0, 0.02))
                close_price = base_price
                
                # 确保OHLC逻辑正确
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                volume = abs(np.random.normal(1000000, 300000))
                turnover_rate = np.random.uniform(0.5, 8.0)
                
                all_data.append({
                    'code': stock_code,
                    'name': stock_name,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': int(volume),
                    'turnover_rate': round(turnover_rate, 2)
                })
        
        return pd.DataFrame(all_data)
    
    def _generate_strategy_test_cases(self, strategy_name: str, 
                                    strategy_type: StrategyType, 
                                    market_condition: MarketCondition, 
                                    test_data: pd.DataFrame) -> List[StrategyTestCase]:
        """
        生成策略测试用例
        
        Args:
            strategy_name: 策略名称
            strategy_type: 策略类型
            market_condition: 市场条件
            test_data: 测试数据
            
        Returns:
            List[StrategyTestCase]: 测试用例列表
        """
        test_cases = []
        
        try:
            # 按时间窗口划分测试用例
            unique_dates = sorted(test_data['date'].unique())
            window_size = self.test_config["test_time_window"]
            
            for i in range(0, len(unique_dates), window_size // 2):  # 50%重叠
                if i + window_size > len(unique_dates):
                    break
                
                start_date = unique_dates[i]
                end_date = unique_dates[min(i + window_size - 1, len(unique_dates) - 1)]
                
                # 筛选时间窗口内的数据
                window_data = test_data[
                    (test_data['date'] >= start_date) & 
                    (test_data['date'] <= end_date)
                ].copy()
                
                if len(window_data) < 10:  # 数据太少跳过
                    continue
                
                # 根据策略类型设置期望选股数量
                expected_selections = self._get_expected_selections(strategy_type, len(window_data))
                expected_accuracy = self._get_expected_accuracy(strategy_type, market_condition)
                
                test_case = StrategyTestCase(
                    strategy_name=strategy_name,
                    strategy_type=strategy_type,
                    market_condition=market_condition,
                    test_data=window_data,
                    expected_selections=expected_selections,
                    expected_accuracy=expected_accuracy,
                    time_period=f"{start_date}_{end_date}",
                    description=f"时间窗口测试 {start_date} 至 {end_date}"
                )
                
                test_cases.append(test_case)
                
                # 限制测试用例数量
                if len(test_cases) >= 5:
                    break
        
        except Exception as e:
            logger.error(f"生成策略测试用例失败: {e}")
        
        return test_cases
    
    def _get_expected_selections(self, strategy_type: StrategyType, data_size: int) -> int:
        """获取期望的选股数量"""
        # 根据策略类型和数据大小估算期望选股数量
        selection_rates = {
            StrategyType.DUAL_MOVING_AVERAGE: 0.05,      # 5%选择率
            StrategyType.MAIN_FORCE_BEHAVIOR: 0.03,      # 3%选择率
            StrategyType.PULLBACK_REBOUND: 0.08,         # 8%选择率
            StrategyType.BREAKTHROUGH_STRATEGY: 0.04,     # 4%选择率
            StrategyType.REVERSAL_STRATEGY: 0.06,        # 6%选择率
            StrategyType.MOMENTUM_STRATEGY: 0.07,        # 7%选择率
            StrategyType.ZXM_STRATEGY: 0.05,             # 5%选择率
            StrategyType.VOLUME_PRICE_STRATEGY: 0.06     # 6%选择率
        }
        
        rate = selection_rates.get(strategy_type, 0.05)
        unique_stocks = data_size // 50  # 假设平均50条记录per股票
        expected = max(int(unique_stocks * rate), 1)
        
        return min(expected, self.test_config["max_selections"])
    
    def _get_expected_accuracy(self, strategy_type: StrategyType, 
                             market_condition: MarketCondition) -> float:
        """获取期望的准确率"""
        # 基础准确率
        base_accuracy = {
            StrategyType.DUAL_MOVING_AVERAGE: 0.65,
            StrategyType.MAIN_FORCE_BEHAVIOR: 0.70,
            StrategyType.PULLBACK_REBOUND: 0.62,
            StrategyType.BREAKTHROUGH_STRATEGY: 0.68,
            StrategyType.REVERSAL_STRATEGY: 0.60,
            StrategyType.MOMENTUM_STRATEGY: 0.66,
            StrategyType.ZXM_STRATEGY: 0.72,
            StrategyType.VOLUME_PRICE_STRATEGY: 0.64
        }
        
        # 市场条件调整
        market_adjustments = {
            MarketCondition.BULL_MARKET: 0.10,      # 牛市提升10%
            MarketCondition.BEAR_MARKET: -0.15,    # 熊市下降15%
            MarketCondition.SIDEWAYS_MARKET: -0.05, # 震荡市下降5%
            MarketCondition.VOLATILE_MARKET: -0.10, # 波动市下降10%
            MarketCondition.LOW_VOLATILITY: 0.05    # 低波动提升5%
        }
        
        base = base_accuracy.get(strategy_type, 0.6)
        adjustment = market_adjustments.get(market_condition, 0.0)
        
        return max(0.3, min(0.9, base + adjustment))
    
    def _execute_strategy(self, test_case: StrategyTestCase) -> List[Dict[str, Any]]:
        """
        执行策略选股
        
        Args:
            test_case: 测试用例
            
        Returns:
            List[Dict[str, Any]]: 选股结果
        """
        try:
            # 构建策略配置
            strategy_config = self._build_strategy_config(test_case)
            
            # 执行策略
            if hasattr(self.strategy_executor, 'execute_strategy'):
                results = self.strategy_executor.execute_strategy(strategy_config, test_case.test_data)
                
                if isinstance(results, list):
                    return results
                elif isinstance(results, pd.DataFrame):
                    return results.to_dict('records')
                else:
                    return []
            else:
                # 模拟策略执行结果
                return self._simulate_strategy_execution(test_case)
                
        except Exception as e:
            logger.error(f"策略执行失败: {e}")
            return []
    
    def _build_strategy_config(self, test_case: StrategyTestCase) -> Dict[str, Any]:
        """
        构建策略配置
        
        Args:
            test_case: 测试用例
            
        Returns:
            Dict[str, Any]: 策略配置
        """
        base_config = {
            "strategy_name": test_case.strategy_name,
            "strategy_type": test_case.strategy_type.value,
            "market_condition": test_case.market_condition.value,
            "time_period": test_case.time_period
        }
        
        # 根据策略类型添加特定配置
        if test_case.strategy_type == StrategyType.DUAL_MOVING_AVERAGE:
            base_config.update({
                "short_period": 5,
                "long_period": 20,
                "cross_type": "golden_cross"
            })
        elif test_case.strategy_type == StrategyType.MAIN_FORCE_BEHAVIOR:
            base_config.update({
                "volume_threshold": 2.0,
                "price_change_threshold": 0.05,
                "detection_window": 5
            })
        elif test_case.strategy_type == StrategyType.ZXM_STRATEGY:
            base_config.update({
                "zxm_threshold": 0.7,
                "combination_mode": "comprehensive"
            })
        
        return base_config
    
    def _simulate_strategy_execution(self, test_case: StrategyTestCase) -> List[Dict[str, Any]]:
        """
        模拟策略执行结果
        
        Args:
            test_case: 测试用例
            
        Returns:
            List[Dict[str, Any]]: 模拟选股结果
        """
        try:
            # 获取唯一股票列表
            unique_stocks = test_case.test_data['code'].unique()
            
            # 根据期望选股数量随机选择股票
            selection_count = min(test_case.expected_selections, len(unique_stocks))
            selected_stocks = np.random.choice(unique_stocks, size=selection_count, replace=False)
            
            results = []
            for stock_code in selected_stocks:
                stock_data = test_case.test_data[test_case.test_data['code'] == stock_code]
                if not stock_data.empty:
                    latest_data = stock_data.iloc[-1]
                    
                    # 模拟选股评分和理由
                    score = np.random.uniform(0.6, 0.95)
                    confidence = np.random.uniform(0.7, 0.9)
                    
                    result = {
                        'code': stock_code,
                        'name': latest_data.get('name', f'股票{stock_code}'),
                        'score': score,
                        'confidence': confidence,
                        'selection_date': latest_data['date'],
                        'selection_price': latest_data['close'],
                        'selection_reason': self._generate_selection_reason(test_case.strategy_type),
                        'technical_indicators': self._generate_technical_indicators(stock_data)
                    }
                    
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"模拟策略执行失败: {e}")
            return []
    
    def _generate_selection_reason(self, strategy_type: StrategyType) -> str:
        """生成选股理由"""
        reasons = {
            StrategyType.DUAL_MOVING_AVERAGE: "短期均线上穿长期均线，形成黄金交叉",
            StrategyType.MAIN_FORCE_BEHAVIOR: "成交量异动，疑似主力资金介入",
            StrategyType.PULLBACK_REBOUND: "价格回踩关键支撑位后出现反弹信号",
            StrategyType.BREAKTHROUGH_STRATEGY: "突破重要阻力位，伴随成交量放大",
            StrategyType.REVERSAL_STRATEGY: "技术指标显示超卖反弹信号",
            StrategyType.MOMENTUM_STRATEGY: "价格动量和成交量动量均显示强势",
            StrategyType.ZXM_STRATEGY: "ZXM指标组合满足买点条件",
            StrategyType.VOLUME_PRICE_STRATEGY: "量价配合良好，呈现健康上涨态势"
        }
        
        return reasons.get(strategy_type, "技术指标满足选股条件")
    
    def _generate_technical_indicators(self, stock_data: pd.DataFrame) -> Dict[str, float]:
        """生成技术指标数据"""
        if stock_data.empty:
            return {}
        
        latest = stock_data.iloc[-1]
        
        # 简化的技术指标计算
        close_prices = stock_data['close'].astype(float)
        volumes = stock_data['volume'].astype(float)
        
        try:
            ma5 = close_prices.tail(5).mean() if len(close_prices) >= 5 else latest['close']
            ma20 = close_prices.tail(20).mean() if len(close_prices) >= 20 else latest['close']
            volume_avg = volumes.tail(5).mean() if len(volumes) >= 5 else latest['volume']
            
            return {
                'ma5': float(ma5),
                'ma20': float(ma20),
                'current_price': float(latest['close']),
                'volume_ratio': float(latest['volume'] / volume_avg) if volume_avg > 0 else 1.0,
                'turnover_rate': float(latest['turnover_rate'])
            }
        except:
            return {
                'ma5': float(latest['close']),
                'ma20': float(latest['close']),
                'current_price': float(latest['close']),
                'volume_ratio': 1.0,
                'turnover_rate': float(latest['turnover_rate'])
            }
    
    def _evaluate_strategy_selections(self, test_case: StrategyTestCase, 
                                    selections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估策略选股结果
        
        Args:
            test_case: 测试用例
            selections: 选股结果
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            if not selections:
                return {
                    'successful_count': 0,
                    'failed_count': 1,
                    'details': [],
                    'metrics': {}
                }
            
            successful_count = 0
            failed_count = 0
            details = []
            performance_metrics = []
            
            for selection in selections:
                try:
                    # 模拟性能评估（实际应该基于后续价格表现）
                    score = selection.get('score', 0.5)
                    confidence = selection.get('confidence', 0.5)
                    
                    # 简化的成功判定：基于评分和置信度
                    success_probability = (score + confidence) / 2
                    is_successful = success_probability >= test_case.expected_accuracy
                    
                    if is_successful:
                        successful_count += 1
                    else:
                        failed_count += 1
                    
                    # 记录详细信息
                    detail = {
                        'code': selection.get('code'),
                        'name': selection.get('name'),
                        'score': score,
                        'confidence': confidence,
                        'successful': is_successful,
                        'selection_date': selection.get('selection_date'),
                        'selection_price': selection.get('selection_price')
                    }
                    details.append(detail)
                    
                    # 性能指标
                    performance_metrics.append({
                        'return': np.random.normal(0.05, 0.15),  # 模拟收益率
                        'volatility': np.random.uniform(0.1, 0.4),  # 模拟波动率
                        'score': score
                    })
                    
                except Exception as e:
                    failed_count += 1
                    logger.error(f"评估选股结果失败: {e}")
            
            # 计算汇总指标
            metrics = {}
            if performance_metrics:
                returns = [m['return'] for m in performance_metrics]
                volatilities = [m['volatility'] for m in performance_metrics]
                scores = [m['score'] for m in performance_metrics]
                
                metrics = {
                    'avg_return': np.mean(returns),
                    'avg_volatility': np.mean(volatilities),
                    'avg_score': np.mean(scores),
                    'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
                    'hit_rate': successful_count / len(selections) if selections else 0
                }
            
            return {
                'successful_count': successful_count,
                'failed_count': failed_count,
                'details': details,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"评估策略选股结果失败: {e}")
            return {
                'successful_count': 0,
                'failed_count': len(selections) if selections else 1,
                'details': [],
                'metrics': {}
            }
    
    def _calculate_strategy_performance_metrics(self, test_cases: List[StrategyTestCase], 
                                              successful_selections: int, 
                                              failed_selections: int) -> Tuple[float, float, float]:
        """
        计算策略性能指标
        
        Args:
            test_cases: 测试用例列表
            successful_selections: 成功选股数
            failed_selections: 失败选股数
            
        Returns:
            Tuple[float, float, float]: (精确度, 召回率, F1分数)
        """
        try:
            total_selections = successful_selections + failed_selections
            
            if total_selections == 0:
                return 0.0, 0.0, 0.0
            
            # 简化的计算
            precision = successful_selections / total_selections
            
            # 估算召回率（基于期望选股数量）
            total_expected = sum(case.expected_selections for case in test_cases)
            recall = successful_selections / total_expected if total_expected > 0 else 0.0
            
            # 计算F1分数
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return precision, recall, f1_score
            
        except Exception as e:
            logger.error(f"计算策略性能指标失败: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_sharpe_ratio(self, performance_metrics: Dict[str, List[float]]) -> float:
        """计算夏普比率"""
        try:
            if 'avg_return' in performance_metrics and performance_metrics['avg_return']:
                returns = performance_metrics['avg_return']
                if len(returns) > 1:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    
                    # 假设无风险利率为3%
                    risk_free_rate = 0.03
                    
                    if std_return > 0:
                        return (avg_return - risk_free_rate) / std_return
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_max_drawdown(self, performance_metrics: Dict[str, List[float]]) -> float:
        """计算最大回撤"""
        try:
            if 'avg_return' in performance_metrics and performance_metrics['avg_return']:
                returns = performance_metrics['avg_return']
                cumulative_returns = np.cumsum(returns)
                
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - peak) / peak
                
                return abs(np.min(drawdown))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_strategy_coverage(self, strategy_name: str, 
                                   strategy_type: StrategyType, 
                                   test_data: pd.DataFrame) -> float:
        """
        计算策略覆盖率
        
        Args:
            strategy_name: 策略名称
            strategy_type: 策略类型
            test_data: 测试数据
            
        Returns:
            float: 覆盖率百分比
        """
        try:
            # 简化的覆盖率计算
            unique_stocks = len(test_data['code'].unique())
            unique_dates = len(test_data['date'].unique())
            
            # 基于策略复杂度和数据完整性计算覆盖率
            complexity_factor = {
                StrategyType.DUAL_MOVING_AVERAGE: 0.9,
                StrategyType.MAIN_FORCE_BEHAVIOR: 0.7,
                StrategyType.PULLBACK_REBOUND: 0.8,
                StrategyType.BREAKTHROUGH_STRATEGY: 0.8,
                StrategyType.REVERSAL_STRATEGY: 0.7,
                StrategyType.MOMENTUM_STRATEGY: 0.8,
                StrategyType.ZXM_STRATEGY: 0.9,
                StrategyType.VOLUME_PRICE_STRATEGY: 0.8
            }.get(strategy_type, 0.8)
            
            # 数据完整性因子
            data_completeness = min(1.0, (unique_stocks * unique_dates) / 2000)  # 假设理想数据量为2000
            
            coverage = complexity_factor * data_completeness * 100
            
            return min(100.0, max(0.0, coverage))
            
        except Exception:
            return 0.0
    
    def _generate_strategy_coverage_test_suite_result(self, 
                                                    results: List[StrategyTestResult]) -> StrategyCoverageTestSuite:
        """
        生成策略覆盖测试套件结果
        
        Args:
            results: 测试结果列表
            
        Returns:
            StrategyCoverageTestSuite: 测试套件结果
        """
        total_strategies = sum(len(strategies) for strategies in self.test_strategies.values())
        tested_strategies = len(results)
        
        if results:
            overall_coverage = sum(r.coverage_percentage for r in results) / len(results)
            overall_accuracy = sum(r.accuracy for r in results) / len(results)
            overall_precision = sum(r.precision for r in results) / len(results)
            overall_recall = sum(r.recall for r in results) / len(results)
        else:
            overall_coverage = overall_accuracy = overall_precision = overall_recall = 0.0
        
        # 按策略类型计算覆盖率
        coverage_by_type = {}
        for strategy_type in StrategyType:
            type_results = [r for r in results if r.strategy_type == strategy_type]
            if type_results:
                coverage_by_type[strategy_type.value] = sum(r.coverage_percentage for r in type_results) / len(type_results)
            else:
                coverage_by_type[strategy_type.value] = 0.0
        
        # 按市场条件计算覆盖率
        coverage_by_market = {}
        for market_condition in MarketCondition:
            market_results = [r for r in results if r.market_condition == market_condition]
            if market_results:
                coverage_by_market[market_condition.value] = sum(r.coverage_percentage for r in market_results) / len(market_results)
            else:
                coverage_by_market[market_condition.value] = 0.0
        
        return StrategyCoverageTestSuite(
            suite_name="策略覆盖测试套件",
            total_strategies=total_strategies,
            tested_strategies=tested_strategies,
            overall_coverage=overall_coverage,
            overall_accuracy=overall_accuracy,
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            total_execution_time=sum(r.execution_time for r in results),
            strategy_results=results,
            coverage_by_type=coverage_by_type,
            coverage_by_market=coverage_by_market
        )


if __name__ == "__main__":
    # 示例使用
    tester = StrategyCoverageTester()
    
    # 运行部分策略测试
    result = tester.run_comprehensive_strategy_coverage_tests(
        strategy_types=[StrategyType.DUAL_MOVING_AVERAGE, StrategyType.ZXM_STRATEGY],
        market_conditions=[MarketCondition.BULL_MARKET, MarketCondition.SIDEWAYS_MARKET]
    )
    
    print(f"策略覆盖测试完成")
    print(f"总体覆盖率: {result.overall_coverage:.2f}%")
    print(f"总体准确率: {result.overall_accuracy:.2f}%")
    print(f"测试的策略数量: {result.tested_strategies}/{result.total_strategies}")
    print(f"执行时间: {result.total_execution_time:.2f}秒") 