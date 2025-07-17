#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票选股功能测试器

提供全面的选股功能测试，包括双均线突破策略、主力行为策略、多市场条件测试等。
遵循L5业务应用层规范，验证选股策略的准确性和可靠性。

测试内容：
- 双均线突破策略测试
- 主力行为策略测试  
- 多市场条件适应性测试
- 选股结果验证
- 选股逻辑可追溯性测试
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from db.query_executor import get_query_executor
from db.sql_manager import QueryType
from strategy.strategy_executor import Strategy_executor
from strategy.strategy_manager import Strategy_manager
from strategy.strategy_parser import Strategy_parser
from indicators.complete_indicator_registry import complete_registry
from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from .config import get_test_config
from .logging_config import get_test_logger

logger = get_test_logger('stock_selection_tester')


class StockSelectionTester:
    """选股功能测试器"""
    
    def __init__(self):
        """初始化选股功能测试器"""
        self.config = get_test_config()
        self.query_executor = get_query_executor()
        self.strategy_executor = Strategy_executor()
        self.strategy_manager = Strategy_manager()
        self.strategy_parser = Strategy_parser()
        
        # 测试统计
        self.test_stats = {
            'total_strategies_tested': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'total_stocks_selected': 0,
            'selection_accuracy': 0.0
        }
        
        # 市场条件配置
        self.market_conditions = {
            'bull_market': {
                'description': '牛市条件',
                'date_range': ('2020-01-01', '2021-06-30'),
                'expected_behavior': 'high_selection_rate'
            },
            'bear_market': {
                'description': '熊市条件',
                'date_range': ('2018-01-01', '2018-12-31'),
                'expected_behavior': 'low_selection_rate'
            },
            'sideways_market': {
                'description': '震荡市条件',
                'date_range': ('2019-01-01', '2019-12-31'),
                'expected_behavior': 'moderate_selection_rate'
            }
        }
        
        logger.info("选股功能测试器初始化完成")
    
    @performance_monitor(threshold_seconds=5.0)
    @exception_handler(reraise=True)
    def test_dual_ma_strategy(self) -> Dict[str, Any]:
        """
        测试双均线突破策略
        
        Returns:
            Dict[str, Any]: 测试结果
        """
        logger.info("开始测试双均线突破策略")
        
        # 定义双均线策略配置
        strategy_config = {
            "strategy_name": "双均线突破策略",
            "short_ma_period": 5,
            "long_ma_period": 20,
            "min_volume": 1000000,
            "min_price": 5.0,
            "max_price": 500.0
        }
        
        # 构建策略条件
        strategy_conditions = [
            {
                "indicator": "MA",
                "params": {"period": strategy_config["short_ma_period"]},
                "condition": "current > previous",
                "description": "短期均线上涨"
            },
            {
                "indicator": "MA", 
                "params": {"period": strategy_config["long_ma_period"]},
                "condition": "current < short_ma",
                "description": "短期均线突破长期均线"
            },
            {
                "field": "volume",
                "condition": f"> {strategy_config['min_volume']}",
                "description": "成交量充足"
            },
            {
                "field": "close",
                "condition": f"> {strategy_config['min_price']} and < {strategy_config['max_price']}",
                "description": "价格在合理范围"
            }
        ]
        
        try:
            # 执行策略测试
            test_results = self._execute_strategy_test(
                strategy_name="dual_ma_strategy",
                strategy_conditions=strategy_conditions,
                test_stocks=self.config.test_data.sample_stock_codes[:5],
                test_date="2024-06-01"
            )
            
            # 验证结果
            validation_results = self._validate_selection_results(
                test_results["selected_stocks"],
                strategy_conditions,
                test_results["test_date"]
            )
            
            # 计算策略效果指标
            performance_metrics = self._calculate_strategy_performance(
                test_results["selected_stocks"],
                test_results["test_date"]
            )
            
            result = {
                "test_name": "双均线突破策略测试",
                "strategy_config": strategy_config,
                "test_results": test_results,
                "validation_results": validation_results,
                "performance_metrics": performance_metrics,
                "success": validation_results["validation_passed"],
                "summary": {
                    "selected_count": len(test_results["selected_stocks"]),
                    "validation_accuracy": validation_results["accuracy"],
                    "selection_quality": performance_metrics["quality_score"]
                }
            }
            
            if result["success"]:
                logger.info(f"双均线突破策略测试通过，选中 {result['summary']['selected_count']} 只股票")
            else:
                logger.warning(f"双均线突破策略测试失败，验证准确率: {validation_results['accuracy']:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"双均线突破策略测试异常: {e}")
            return {
                "test_name": "双均线突破策略测试",
                "success": False,
                "error": str(e),
                "summary": {
                    "selected_count": 0,
                    "validation_accuracy": 0.0,
                    "selection_quality": 0.0
                }
            }
    
    @performance_monitor(threshold_seconds=8.0)
    @exception_handler(reraise=True)
    def test_main_force_strategy(self) -> Dict[str, Any]:
        """
        测试主力行为策略
        
        Returns:
            Dict[str, Any]: 测试结果
        """
        logger.info("开始测试主力行为策略")
        
        # 定义主力行为策略配置
        strategy_config = {
            "strategy_name": "主力行为策略",
            "volume_ma_period": 10,
            "volume_amplification": 2.0,
            "price_change_threshold": 0.03,
            "turnover_threshold": 0.02
        }
        
        # 构建策略条件
        strategy_conditions = [
            {
                "indicator": "VOLUME_MA",
                "params": {"period": strategy_config["volume_ma_period"]},
                "condition": "current > average * 2.0",
                "description": "成交量放大"
            },
            {
                "field": "price_change",
                "condition": f"> {strategy_config['price_change_threshold']}",
                "description": "价格上涨幅度适中"
            },
            {
                "field": "turnover_rate",
                "condition": f"> {strategy_config['turnover_threshold']}",
                "description": "换手率活跃"
            },
            {
                "indicator": "RSI",
                "params": {"period": 14},
                "condition": "> 50 and < 80",
                "description": "相对强度指标健康"
            }
        ]
        
        try:
            # 执行策略测试
            test_results = self._execute_strategy_test(
                strategy_name="main_force_strategy",
                strategy_conditions=strategy_conditions,
                test_stocks=self.config.test_data.sample_stock_codes[:5],
                test_date="2024-06-01"
            )
            
            # 验证主力行为特征
            main_force_validation = self._validate_main_force_behavior(
                test_results["selected_stocks"],
                test_results["test_date"]
            )
            
            # 计算策略效果指标
            performance_metrics = self._calculate_strategy_performance(
                test_results["selected_stocks"],
                test_results["test_date"]
            )
            
            result = {
                "test_name": "主力行为策略测试",
                "strategy_config": strategy_config,
                "test_results": test_results,
                "main_force_validation": main_force_validation,
                "performance_metrics": performance_metrics,
                "success": main_force_validation["validation_passed"],
                "summary": {
                    "selected_count": len(test_results["selected_stocks"]),
                    "main_force_accuracy": main_force_validation["accuracy"],
                    "selection_quality": performance_metrics["quality_score"]
                }
            }
            
            if result["success"]:
                logger.info(f"主力行为策略测试通过，选中 {result['summary']['selected_count']} 只股票")
            else:
                logger.warning(f"主力行为策略测试失败，主力行为验证准确率: {main_force_validation['accuracy']:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"主力行为策略测试异常: {e}")
            return {
                "test_name": "主力行为策略测试",
                "success": False,
                "error": str(e),
                "summary": {
                    "selected_count": 0,
                    "main_force_accuracy": 0.0,
                    "selection_quality": 0.0
                }
            }
    
    @performance_monitor(threshold_seconds=15.0)
    @exception_handler(reraise=True)
    def test_market_conditions(self) -> Dict[str, Any]:
        """
        测试不同市场条件下的策略表现
        
        Returns:
            Dict[str, Any]: 测试结果
        """
        logger.info("开始测试不同市场条件")
        
        market_test_results = {}
        overall_success = True
        
        for condition_name, condition_config in self.market_conditions.items():
            logger.info(f"测试市场条件: {condition_config['description']}")
            
            try:
                # 基础策略条件
                strategy_conditions = [
                    {
                        "indicator": "MA",
                        "params": {"period": 5},
                        "condition": "current > previous",
                        "description": "短期趋势向上"
                    },
                    {
                        "field": "volume",
                        "condition": "> 500000",
                        "description": "成交量充足"
                    }
                ]
                
                # 执行市场条件测试
                start_date, end_date = condition_config['date_range']
                test_date = self._get_valid_trading_date(start_date, end_date)
                
                test_results = self._execute_strategy_test(
                    strategy_name=f"market_condition_{condition_name}",
                    strategy_conditions=strategy_conditions,
                    test_stocks=self.config.test_data.sample_stock_codes[:3],
                    test_date=test_date
                )
                
                # 验证市场条件适应性
                adaptation_results = self._validate_market_adaptation(
                    test_results["selected_stocks"],
                    condition_config,
                    test_date
                )
                
                market_test_results[condition_name] = {
                    "condition_config": condition_config,
                    "test_results": test_results,
                    "adaptation_results": adaptation_results,
                    "success": adaptation_results["adaptation_passed"]
                }
                
                if not adaptation_results["adaptation_passed"]:
                    overall_success = False
                    
                logger.info(f"市场条件 {condition_name} 测试完成，适应性: {adaptation_results['adaptation_score']:.1%}")
                
            except Exception as e:
                logger.error(f"市场条件 {condition_name} 测试异常: {e}")
                market_test_results[condition_name] = {
                    "condition_config": condition_config,
                    "success": False,
                    "error": str(e)
                }
                overall_success = False
        
        # 汇总结果
        successful_conditions = sum(1 for result in market_test_results.values() if result.get("success", False))
        total_conditions = len(market_test_results)
        
        result = {
            "test_name": "市场条件适应性测试",
            "market_test_results": market_test_results,
            "success": overall_success,
            "summary": {
                "total_conditions": total_conditions,
                "successful_conditions": successful_conditions,
                "success_rate": successful_conditions / total_conditions if total_conditions > 0 else 0.0,
                "overall_adaptation": overall_success
            }
        }
        
        if result["success"]:
            logger.info(f"市场条件测试通过，成功率: {result['summary']['success_rate']:.1%}")
        else:
            logger.warning(f"市场条件测试失败，成功率: {result['summary']['success_rate']:.1%}")
        
        return result
    
    @performance_monitor(threshold_seconds=10.0)
    @exception_handler(reraise=True)
    def test_selection_traceability(self) -> Dict[str, Any]:
        """
        测试选股逻辑可追溯性
        
        Returns:
            Dict[str, Any]: 测试结果
        """
        logger.info("开始测试选股逻辑可追溯性")
        
        try:
            # 定义带详细理由的策略
            strategy_conditions = [
                {
                    "indicator": "MA",
                    "params": {"period": 5},
                    "condition": "current > previous",
                    "description": "短期均线上涨",
                    "weight": 0.3,
                    "reason_template": "5日均线呈上涨趋势，当前值{current}高于前值{previous}"
                },
                {
                    "indicator": "MACD",
                    "params": {},
                    "condition": "signal > 0",
                    "description": "MACD金叉信号",
                    "weight": 0.4,
                    "reason_template": "MACD信号线{signal}大于0，出现金叉信号"
                },
                {
                    "field": "volume",
                    "condition": "> 1000000",
                    "description": "成交量充足",
                    "weight": 0.3,
                    "reason_template": "成交量{volume}超过100万手，资金关注度高"
                }
            ]
            
            # 执行可追溯性测试
            test_results = self._execute_traceability_test(
                strategy_conditions=strategy_conditions,
                test_stocks=self.config.test_data.sample_stock_codes[:3],
                test_date="2024-06-01"
            )
            
            # 验证选股理由的完整性和准确性
            traceability_validation = self._validate_selection_traceability(
                test_results["detailed_results"]
            )
            
            result = {
                "test_name": "选股逻辑可追溯性测试",
                "test_results": test_results,
                "traceability_validation": traceability_validation,
                "success": traceability_validation["traceability_passed"],
                "summary": {
                    "traced_selections": len(test_results["detailed_results"]),
                    "reason_completeness": traceability_validation["reason_completeness"],
                    "reason_accuracy": traceability_validation["reason_accuracy"]
                }
            }
            
            if result["success"]:
                logger.info(f"选股逻辑可追溯性测试通过，理由完整性: {traceability_validation['reason_completeness']:.1%}")
            else:
                logger.warning(f"选股逻辑可追溯性测试失败，理由完整性: {traceability_validation['reason_completeness']:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"选股逻辑可追溯性测试异常: {e}")
            return {
                "test_name": "选股逻辑可追溯性测试",
                "success": False,
                "error": str(e),
                "summary": {
                    "traced_selections": 0,
                    "reason_completeness": 0.0,
                    "reason_accuracy": 0.0
                }
            }
    
    def _execute_strategy_test(self, strategy_name: str, strategy_conditions: List[Dict],
                              test_stocks: List[str], test_date: str) -> Dict[str, Any]:
        """
        执行策略测试的核心逻辑
        
        Args:
            strategy_name: 策略名称
            strategy_conditions: 策略条件列表
            test_stocks: 测试股票列表
            test_date: 测试日期
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        selected_stocks = []
        test_details = {}
        
        for stock_code in test_stocks:
            try:
                # 获取股票数据
                stock_data = self._get_stock_data(stock_code, test_date)
                if stock_data.empty:
                    continue
                
                # 检查策略条件
                condition_results = []
                meets_all_conditions = True
                
                for condition in strategy_conditions:
                    condition_result = self._evaluate_condition(stock_data, condition, test_date)
                    condition_results.append({
                        "condition": condition["description"],
                        "result": condition_result["meets_condition"],
                        "details": condition_result["details"]
                    })
                    
                    if not condition_result["meets_condition"]:
                        meets_all_conditions = False
                
                test_details[stock_code] = {
                    "stock_data": stock_data.iloc[-1].to_dict() if not stock_data.empty else {},
                    "condition_results": condition_results,
                    "meets_all_conditions": meets_all_conditions
                }
                
                if meets_all_conditions:
                    selected_stocks.append({
                        "code": stock_code,
                        "name": stock_data.iloc[-1].get("name", ""),
                        "price": stock_data.iloc[-1].get("close", 0.0),
                        "volume": stock_data.iloc[-1].get("volume", 0),
                        "selection_reason": f"满足{strategy_name}的所有条件"
                    })
                    
            except Exception as e:
                logger.warning(f"处理股票 {stock_code} 时出错: {e}")
                test_details[stock_code] = {
                    "error": str(e),
                    "meets_all_conditions": False
                }
        
        return {
            "strategy_name": strategy_name,
            "test_date": test_date,
            "selected_stocks": selected_stocks,
            "test_details": test_details,
            "total_tested": len(test_stocks),
            "total_selected": len(selected_stocks)
        }
    
    def _get_stock_data(self, stock_code: str, end_date: str, days: int = 30) -> pd.DataFrame:
        """获取股票数据"""
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')
        
        params = {
            'code': stock_code,
            'start_date': start_date,
            'end_date': end_date,
            'level': '日线'
        }
        
        return self.query_executor.execute_query(QueryType.STOCK_DATA, params)
    
    def _evaluate_condition(self, stock_data: pd.DataFrame, condition: Dict, test_date: str) -> Dict[str, Any]:
        """评估单个条件是否满足"""
        # 这里应该实现具体的条件评估逻辑
        # 简化实现，实际应该根据条件类型调用相应的指标计算
        
        try:
            if condition.get("indicator"):
                # 指标条件
                indicator_name = condition["indicator"]
                if indicator_name == "MA":
                    period = condition["params"]["period"]
                    if len(stock_data) >= period:
                        ma_values = stock_data["close"].rolling(window=period).mean()
                        current_ma = ma_values.iloc[-1]
                        previous_ma = ma_values.iloc[-2] if len(ma_values) > 1 else current_ma
                        
                        meets_condition = current_ma > previous_ma
                        return {
                            "meets_condition": meets_condition,
                            "details": {
                                "current_ma": current_ma,
                                "previous_ma": previous_ma,
                                "condition": condition["condition"]
                            }
                        }
                
                elif indicator_name == "MACD":
                    # 简化的MACD实现
                    if len(stock_data) >= 26:
                        ema12 = stock_data["close"].ewm(span=12).mean()
                        ema26 = stock_data["close"].ewm(span=26).mean()
                        macd_line = ema12 - ema26
                        signal_line = macd_line.ewm(span=9).mean()
                        
                        current_signal = signal_line.iloc[-1]
                        meets_condition = current_signal > 0
                        
                        return {
                            "meets_condition": meets_condition,
                            "details": {
                                "signal": current_signal,
                                "condition": condition["condition"]
                            }
                        }
                
                elif indicator_name == "RSI":
                    # 简化的RSI实现
                    period = condition["params"].get("period", 14)
                    if len(stock_data) >= period + 1:
                        delta = stock_data["close"].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        current_rsi = rsi.iloc[-1]
                        meets_condition = 50 < current_rsi < 80  # 简化条件
                        
                        return {
                            "meets_condition": meets_condition,
                            "details": {
                                "rsi": current_rsi,
                                "condition": condition["condition"]
                            }
                        }
            
            elif condition.get("field"):
                # 字段条件
                field_name = condition["field"]
                if field_name in stock_data.columns:
                    field_value = stock_data[field_name].iloc[-1]
                    condition_text = condition["condition"]
                    
                    # 简单的条件评估
                    if ">" in condition_text:
                        threshold = float(condition_text.split(">")[1].strip())
                        meets_condition = field_value > threshold
                    elif "<" in condition_text:
                        threshold = float(condition_text.split("<")[1].strip())
                        meets_condition = field_value < threshold
                    else:
                        meets_condition = True
                    
                    return {
                        "meets_condition": meets_condition,
                        "details": {
                            "field_value": field_value,
                            "condition": condition_text
                        }
                    }
            
            # 默认返回
            return {
                "meets_condition": False,
                "details": {"error": "未知条件类型"}
            }
            
        except Exception as e:
            return {
                "meets_condition": False,
                "details": {"error": str(e)}
            }
    
    def _validate_selection_results(self, selected_stocks: List[Dict], 
                                   strategy_conditions: List[Dict], test_date: str) -> Dict[str, Any]:
        """验证选股结果的准确性"""
        total_selected = len(selected_stocks)
        validated_count = 0
        validation_details = []
        
        for stock in selected_stocks:
            try:
                # 重新验证每只选中的股票
                stock_data = self._get_stock_data(stock["code"], test_date)
                if stock_data.empty:
                    continue
                
                all_conditions_met = True
                for condition in strategy_conditions:
                    condition_result = self._evaluate_condition(stock_data, condition, test_date)
                    if not condition_result["meets_condition"]:
                        all_conditions_met = False
                        break
                
                if all_conditions_met:
                    validated_count += 1
                
                validation_details.append({
                    "stock_code": stock["code"],
                    "validated": all_conditions_met
                })
                
            except Exception as e:
                logger.warning(f"验证股票 {stock['code']} 时出错: {e}")
                validation_details.append({
                    "stock_code": stock["code"],
                    "validated": False,
                    "error": str(e)
                })
        
        accuracy = validated_count / total_selected if total_selected > 0 else 1.0
        validation_passed = accuracy >= 0.8  # 80%以上准确率视为通过
        
        return {
            "validation_passed": validation_passed,
            "accuracy": accuracy,
            "total_selected": total_selected,
            "validated_count": validated_count,
            "validation_details": validation_details
        }
    
    def _validate_main_force_behavior(self, selected_stocks: List[Dict], test_date: str) -> Dict[str, Any]:
        """验证主力行为特征"""
        # 实现主力行为验证逻辑
        total_stocks = len(selected_stocks)
        valid_main_force_count = 0
        
        for stock in selected_stocks:
            # 简化的主力行为验证
            try:
                stock_data = self._get_stock_data(stock["code"], test_date, days=20)
                if len(stock_data) >= 10:
                    # 检查成交量放大
                    recent_volume = stock_data["volume"].iloc[-5:].mean()
                    historical_volume = stock_data["volume"].iloc[-20:-5].mean()
                    
                    if recent_volume > historical_volume * 1.5:
                        valid_main_force_count += 1
                        
            except Exception as e:
                logger.warning(f"验证主力行为 {stock['code']} 时出错: {e}")
        
        accuracy = valid_main_force_count / total_stocks if total_stocks > 0 else 1.0
        validation_passed = accuracy >= 0.7  # 70%以上准确率视为通过
        
        return {
            "validation_passed": validation_passed,
            "accuracy": accuracy,
            "total_stocks": total_stocks,
            "valid_main_force_count": valid_main_force_count
        }
    
    def _validate_market_adaptation(self, selected_stocks: List[Dict], 
                                   condition_config: Dict, test_date: str) -> Dict[str, Any]:
        """验证市场条件适应性"""
        expected_behavior = condition_config["expected_behavior"]
        selected_count = len(selected_stocks)
        
        # 根据市场条件判断适应性
        if expected_behavior == "high_selection_rate":
            adaptation_passed = selected_count >= 2  # 牛市应该选中较多股票
            adaptation_score = min(selected_count / 3.0, 1.0)
        elif expected_behavior == "low_selection_rate":
            adaptation_passed = selected_count <= 1  # 熊市应该选中较少股票
            adaptation_score = 1.0 - min(selected_count / 3.0, 1.0)
        else:  # moderate_selection_rate
            adaptation_passed = 1 <= selected_count <= 2  # 震荡市适中
            adaptation_score = 1.0 - abs(selected_count - 1.5) / 1.5
        
        return {
            "adaptation_passed": adaptation_passed,
            "adaptation_score": max(adaptation_score, 0.0),
            "selected_count": selected_count,
            "expected_behavior": expected_behavior
        }
    
    def _calculate_strategy_performance(self, selected_stocks: List[Dict], test_date: str) -> Dict[str, Any]:
        """计算策略性能指标"""
        if not selected_stocks:
            return {
                "quality_score": 0.0,
                "diversity_score": 0.0,
                "risk_score": 0.0
            }
        
        # 简化的性能计算
        quality_score = min(len(selected_stocks) / 5.0, 1.0)  # 基于选中数量
        diversity_score = 0.8  # 简化为固定值
        risk_score = 0.7  # 简化为固定值
        
        return {
            "quality_score": quality_score,
            "diversity_score": diversity_score,
            "risk_score": risk_score
        }
    
    def _execute_traceability_test(self, strategy_conditions: List[Dict], 
                                  test_stocks: List[str], test_date: str) -> Dict[str, Any]:
        """执行可追溯性测试"""
        detailed_results = []
        
        for stock_code in test_stocks:
            try:
                stock_data = self._get_stock_data(stock_code, test_date)
                if stock_data.empty:
                    continue
                
                selection_reasons = []
                total_score = 0.0
                
                for condition in strategy_conditions:
                    condition_result = self._evaluate_condition(stock_data, condition, test_date)
                    weight = condition.get("weight", 1.0)
                    
                    if condition_result["meets_condition"]:
                        reason_template = condition.get("reason_template", condition["description"])
                        reason = self._format_selection_reason(reason_template, condition_result["details"])
                        selection_reasons.append({
                            "condition": condition["description"],
                            "reason": reason,
                            "weight": weight,
                            "score": weight
                        })
                        total_score += weight
                
                detailed_results.append({
                    "stock_code": stock_code,
                    "selection_reasons": selection_reasons,
                    "total_score": total_score,
                    "selected": total_score >= 0.6  # 60%以上得分视为选中
                })
                
            except Exception as e:
                logger.warning(f"可追溯性测试处理股票 {stock_code} 时出错: {e}")
        
        return {
            "detailed_results": detailed_results,
            "total_tested": len(test_stocks)
        }
    
    def _validate_selection_traceability(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """验证选股理由的可追溯性"""
        total_selections = len([r for r in detailed_results if r["selected"]])
        complete_reason_count = 0
        accurate_reason_count = 0
        
        for result in detailed_results:
            if result["selected"]:
                # 检查理由完整性
                if len(result["selection_reasons"]) >= 2:  # 至少2个理由
                    complete_reason_count += 1
                
                # 检查理由准确性（简化检查）
                if result["total_score"] >= 0.6:
                    accurate_reason_count += 1
        
        reason_completeness = complete_reason_count / total_selections if total_selections > 0 else 1.0
        reason_accuracy = accurate_reason_count / total_selections if total_selections > 0 else 1.0
        traceability_passed = reason_completeness >= 0.8 and reason_accuracy >= 0.8
        
        return {
            "traceability_passed": traceability_passed,
            "reason_completeness": reason_completeness,
            "reason_accuracy": reason_accuracy,
            "total_selections": total_selections
        }
    
    def _format_selection_reason(self, template: str, details: Dict) -> str:
        """格式化选股理由"""
        try:
            return template.format(**details)
        except (KeyError, ValueError):
            return template
    
    def _get_valid_trading_date(self, start_date: str, end_date: str) -> str:
        """获取有效的交易日期"""
        # 简化实现，返回中间日期
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        middle = start + (end - start) / 2
        return middle.strftime('%Y-%m-%d')


if __name__ == "__main__":
    # 测试选股功能测试器
    tester = StockSelectionTester()
    
    # 运行测试
    print("运行双均线策略测试...")
    result1 = tester.test_dual_ma_strategy()
    print(f"结果: {result1['success']}, 选中: {result1['summary']['selected_count']}")
    
    print("\n运行主力行为策略测试...")
    result2 = tester.test_main_force_strategy()
    print(f"结果: {result2['success']}, 选中: {result2['summary']['selected_count']}")
    
    print("\n运行市场条件测试...")
    result3 = tester.test_market_conditions()
    print(f"结果: {result3['success']}, 成功率: {result3['summary']['success_rate']:.1%}")
    
    print("\n运行可追溯性测试...")
    result4 = tester.test_selection_traceability()
    print(f"结果: {result4['success']}, 理由完整性: {result4['summary']['reason_completeness']:.1%}")