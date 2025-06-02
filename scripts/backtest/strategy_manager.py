#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测策略管理模块

负责执行和管理回测策略
"""

import os
import sys
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.decorators import performance_monitor, time_it
from utils.period_data_structure import (
    IndicatorPeriodResult,
    PeriodAnalysisResult,
    MultiPeriodAnalysisResult
)
from scripts.backtest.data_manager import BacktestDataManager
from scripts.backtest.pattern_analyzer import PatternAnalyzer

# 获取日志记录器
logger = get_logger(__name__)


class BacktestStrategy:
    """
    回测策略基类
    
    所有回测策略都应该继承此类，并实现必要的方法
    """

    def __init__(self, name: str = "Base Strategy", description: str = ""):
        """
        初始化回测策略
        
        Args:
            name: 策略名称
            description: 策略描述
        """
        self.name = name
        self.description = description
        self.data_manager = BacktestDataManager()
        self.pattern_analyzer = PatternAnalyzer()
        
        # 策略参数
        self.params = {}
        
        # 策略结果
        self.results = []
        
        logger.info(f"初始化策略: {self.name}")
    
    def set_params(self, **kwargs):
        """
        设置策略参数
        
        Args:
            **kwargs: 策略参数
        """
        for key, value in kwargs.items():
            self.params[key] = value
        
        logger.info(f"设置策略参数: {kwargs}")
    
    def run(self, stock_code: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        运行策略
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Any]: 策略结果
        """
        logger.info(f"运行策略 {self.name} 于股票 {stock_code}, 时间段: {start_date} - {end_date}")
        
        # 获取数据
        data = self.data_manager.get_stock_data(
            stock_code=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            logger.warning(f"股票 {stock_code} 在时间段 {start_date} - {end_date} 内没有数据")
            return {
                "stock_code": stock_code,
                "success": False,
                "error": "无数据"
            }
        
        # 执行策略
        try:
            result = self._execute_strategy(stock_code, data)
            result.update({
                "stock_code": stock_code,
                "strategy_name": self.name,
                "start_date": start_date,
                "end_date": end_date,
                "success": True
            })
            
            # 保存结果
            self.results.append(result)
            
            return result
        except Exception as e:
            logger.error(f"执行策略 {self.name} 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "stock_code": stock_code,
                "strategy_name": self.name,
                "start_date": start_date,
                "end_date": end_date,
                "success": False,
                "error": str(e)
            }
    
    def _execute_strategy(self, stock_code: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        执行策略逻辑
        
        子类必须实现此方法
        
        Args:
            stock_code: 股票代码
            data: 股票数据
            
        Returns:
            Dict[str, Any]: 策略结果
        """
        raise NotImplementedError("子类必须实现_execute_strategy方法")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        获取策略结果
        
        Returns:
            List[Dict[str, Any]]: 策略结果列表
        """
        return self.results
    
    def clear_results(self):
        """清除策略结果"""
        self.results = []
        logger.info(f"清除策略 {self.name} 的结果")


class PatternRecognitionStrategy(BacktestStrategy):
    """
    形态识别策略
    
    基于技术形态识别的回测策略
    """

    def __init__(self, name: str = "Pattern Recognition Strategy", 
                description: str = "基于技术形态识别的回测策略"):
        """
        初始化形态识别策略
        
        Args:
            name: 策略名称
            description: 策略描述
        """
        super().__init__(name, description)
        
        # 形态列表
        self.pattern_ids = []
        
        # 最小形态强度
        self.min_pattern_strength = 0.6
    
    def set_patterns(self, pattern_ids: List[str]):
        """
        设置要识别的形态
        
        Args:
            pattern_ids: 形态ID列表
        """
        self.pattern_ids = pattern_ids
        logger.info(f"设置形态列表: {pattern_ids}")
    
    def set_min_strength(self, min_strength: float):
        """
        设置最小形态强度
        
        Args:
            min_strength: 最小形态强度 (0.0-1.0)
        """
        self.min_pattern_strength = min_strength
        logger.info(f"设置最小形态强度: {min_strength}")
    
    def _execute_strategy(self, stock_code: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        执行形态识别策略
        
        Args:
            stock_code: 股票代码
            data: 股票数据
            
        Returns:
            Dict[str, Any]: 策略结果
        """
        # 分析形态
        if self.pattern_ids:
            # 分析指定形态
            patterns = self.pattern_analyzer.analyze_patterns(
                data=data,
                pattern_ids=self.pattern_ids,
                min_strength=self.min_pattern_strength
            )
        else:
            # 分析所有形态
            patterns = self.pattern_analyzer.analyze_all_patterns(
                data=data,
                min_strength=self.min_pattern_strength
            )
        
        # 返回结果
        result = {
            "patterns": patterns,
            "pattern_count": len(patterns),
            "has_patterns": len(patterns) > 0
        }
        
        # 如果有形态，添加最强形态
        if patterns:
            strongest_pattern = max(patterns, key=lambda p: p.get("strength", 0))
            result["strongest_pattern"] = strongest_pattern
            
            # 添加最后一个交易日的收盘价
            result["last_close"] = data["close"].iloc[-1]
            
            # 添加股票名称
            result["stock_name"] = self.data_manager.get_stock_name(stock_code)
        
        return result


class MultiPeriodPatternStrategy(BacktestStrategy):
    """
    多周期形态策略
    
    分析多个周期的技术形态
    """

    def __init__(self, name: str = "Multi-Period Pattern Strategy", 
                description: str = "多周期技术形态分析策略"):
        """
        初始化多周期形态策略
        
        Args:
            name: 策略名称
            description: 策略描述
        """
        super().__init__(name, description)
        
        # 周期列表
        self.periods = ["daily", "weekly", "monthly"]
        
        # 形态组合
        self.pattern_combinations = []
        
        # 最小形态强度
        self.min_pattern_strength = 60  # 百分比
    
    def set_periods(self, periods: List[str]):
        """
        设置要分析的周期
        
        Args:
            periods: 周期列表
        """
        self.periods = periods
        logger.info(f"设置周期列表: {periods}")
    
    def set_pattern_combinations(self, combinations: List[Dict[str, Any]]):
        """
        设置形态组合
        
        Args:
            combinations: 形态组合列表
        """
        self.pattern_combinations = combinations
        logger.info(f"设置形态组合: {combinations}")
    
    def _execute_strategy(self, stock_code: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        执行多周期形态策略
        
        Args:
            stock_code: 股票代码
            data: 股票数据
            
        Returns:
            Dict[str, Any]: 策略结果
        """
        # 获取周期数据
        period_data = {}
        for period in self.periods:
            period_df = self.data_manager.get_period_data(
                stock_code=stock_code,
                period=period,
                start_date=data["date"].min().strftime("%Y-%m-%d"),
                end_date=data["date"].max().strftime("%Y-%m-%d")
            )
            
            if not period_df.empty:
                period_data[period] = period_df
        
        # 分析各周期形态
        period_patterns = {}
        for period, period_df in period_data.items():
            patterns = self.pattern_analyzer.analyze_all_patterns(
                data=period_df,
                min_strength=self.min_pattern_strength / 100  # 转换为0-1范围
            )
            period_patterns[period] = patterns
        
        # 检查是否匹配形态组合
        if self.pattern_combinations:
            # TODO: 实现形态组合匹配逻辑
            matched = False
            matched_patterns = []
        else:
            matched = any(len(patterns) > 0 for patterns in period_patterns.values())
            matched_patterns = []
            for period, patterns in period_patterns.items():
                for pattern in patterns:
                    matched_patterns.append({
                        "period": period,
                        "pattern_id": pattern.get("pattern_id", ""),
                        "pattern_name": pattern.get("pattern_name", ""),
                        "strength": pattern.get("strength", 0) * 100,  # 转换为百分比
                        "description": pattern.get("description", "")
                    })
        
        # 返回结果
        result = {
            "period_patterns": period_patterns,
            "matched": matched,
            "matched_patterns": matched_patterns,
            "pattern_count": sum(len(patterns) for patterns in period_patterns.values())
        }
        
        # 添加最后一个交易日的收盘价
        result["last_close"] = data["close"].iloc[-1]
        
        # 添加股票名称
        result["stock_name"] = self.data_manager.get_stock_name(stock_code)
        
        return result


class ZXMPatternStrategy(BacktestStrategy):
    """
    ZXM形态策略
    
    基于ZXM体系的形态分析策略
    """

    def __init__(self, name: str = "ZXM Pattern Strategy", 
                description: str = "基于ZXM体系的形态分析策略"):
        """
        初始化ZXM形态策略
        
        Args:
            name: 策略名称
            description: 策略描述
        """
        super().__init__(name, description)
        
        # 周期列表
        self.periods = ["daily", "weekly", "monthly"]
        
        # 最小形态强度
        self.min_pattern_strength = 60  # 百分比
        
        # 买入信号分数阈值
        self.buy_score_threshold = 60  # 百分比
    
    def set_periods(self, periods: List[str]):
        """
        设置要分析的周期
        
        Args:
            periods: 周期列表
        """
        self.periods = periods
        logger.info(f"设置周期列表: {periods}")
    
    def _execute_strategy(self, stock_code: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        执行ZXM形态策略
        
        Args:
            stock_code: 股票代码
            data: 股票数据
            
        Returns:
            Dict[str, Any]: 策略结果
        """
        from utils.period_data_structure import MultiPeriodAnalysisResult
        
        # 创建多周期分析结果对象
        analysis_result = MultiPeriodAnalysisResult(stock_code=stock_code)
        analysis_result.buy_date = data["date"].iloc[-1]
        
        # 分析各周期指标
        for period in self.periods:
            period_df = self.data_manager.get_period_data(
                stock_code=stock_code,
                period=period,
                start_date=data["date"].min().strftime("%Y-%m-%d"),
                end_date=data["date"].max().strftime("%Y-%m-%d")
            )
            
            if period_df.empty:
                continue
            
            # 分析ZXM指标
            from indicators.zxm.zxm_factory import ZXMIndicatorFactory
            
            factory = ZXMIndicatorFactory()
            period_result = factory.analyze_all(
                data=period_df,
                period=period
            )
            
            analysis_result.add_period_result(period, period_result)
        
        # 分析ZXM体系指标
        zxm_analysis = self.pattern_analyzer.analyze_zxm_indicators(
            result=analysis_result,
            title=f"股票 {stock_code} ZXM体系指标分析"
        )
        
        # 格式化ZXM分析结果
        zxm_text = self.pattern_analyzer.format_zxm_analysis(zxm_analysis)
        
        # 返回结果
        result = {
            "analysis_result": analysis_result,
            "zxm_analysis": zxm_analysis,
            "zxm_text": zxm_text,
            "composite_score": zxm_analysis.get("composite_score", 0),
            "composite_rating": zxm_analysis.get("composite_rating", ""),
            "trend_consistency": zxm_analysis.get("trend_consistency", {})
        }
        
        # 添加操作建议
        advice = zxm_analysis.get("trend_consistency", {}).get("advice", "")
        result["advice"] = advice
        
        # 判断是否有买入信号
        result["has_buy_signal"] = result["composite_score"] >= self.buy_score_threshold
        
        # 添加最后一个交易日的收盘价
        result["last_close"] = data["close"].iloc[-1]
        
        # 添加股票名称
        result["stock_name"] = self.data_manager.get_stock_name(stock_code)
        
        return result


# 策略管理器
class StrategyManager:
    """
    策略管理器
    
    负责管理和执行各种回测策略
    """

    def __init__(self):
        """初始化策略管理器"""
        self.strategies = {}
        self.data_manager = BacktestDataManager()
        
        # 初始化策略工厂
        self._init_strategies()
        
        logger.info("策略管理器初始化完成")
    
    def _init_strategies(self):
        """初始化内置策略"""
        # 形态识别策略
        self.register_strategy("pattern", PatternRecognitionStrategy())
        
        # 多周期形态策略
        self.register_strategy("multi_period", MultiPeriodPatternStrategy())
        
        # ZXM形态策略
        self.register_strategy("zxm", ZXMPatternStrategy())
    
    def register_strategy(self, strategy_id: str, strategy: BacktestStrategy):
        """
        注册策略
        
        Args:
            strategy_id: 策略ID
            strategy: 策略对象
        """
        self.strategies[strategy_id] = strategy
        logger.info(f"注册策略: {strategy_id} - {strategy.name}")
    
    def get_strategy(self, strategy_id: str) -> Optional[BacktestStrategy]:
        """
        获取策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            Optional[BacktestStrategy]: 策略对象
        """
        return self.strategies.get(strategy_id)
    
    def list_strategies(self) -> Dict[str, str]:
        """
        列出所有策略
        
        Returns:
            Dict[str, str]: 策略ID和名称的映射
        """
        return {
            strategy_id: strategy.name
            for strategy_id, strategy in self.strategies.items()
        }
    
    def run_strategy(self, strategy_id: str, stock_code: str, 
                    start_date: str, end_date: str, **kwargs) -> Dict[str, Any]:
        """
        运行策略
        
        Args:
            strategy_id: 策略ID
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 策略参数
            
        Returns:
            Dict[str, Any]: 策略结果
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            logger.error(f"未找到策略: {strategy_id}")
            return {"success": False, "error": f"未找到策略: {strategy_id}"}
        
        # 设置策略参数
        if kwargs:
            strategy.set_params(**kwargs)
        
        # 运行策略
        return strategy.run(stock_code, start_date, end_date)
    
    def batch_run(self, strategy_id: str, stock_codes: List[str], 
                 start_date: str, end_date: str, **kwargs) -> List[Dict[str, Any]]:
        """
        批量运行策略
        
        Args:
            strategy_id: 策略ID
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 策略参数
            
        Returns:
            List[Dict[str, Any]]: 策略结果列表
        """
        results = []
        
        for stock_code in stock_codes:
            result = self.run_strategy(
                strategy_id=strategy_id,
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )
            
            results.append(result)
        
        return results


# 测试代码
if __name__ == "__main__":
    # 初始化策略管理器
    manager = StrategyManager()
    
    # 获取策略列表
    strategies = manager.list_strategies()
    print(f"可用策略: {strategies}")
    
    # 测试形态识别策略
    result = manager.run_strategy(
        strategy_id="pattern",
        stock_code="000001",
        start_date="20220101",
        end_date="20220131"
    )
    
    print(f"形态识别策略结果: {result}")
    
    # 测试ZXM形态策略
    result = manager.run_strategy(
        strategy_id="zxm",
        stock_code="000001",
        start_date="20220101",
        end_date="20220131"
    )
    
    print(f"ZXM形态策略结果: {result}") 