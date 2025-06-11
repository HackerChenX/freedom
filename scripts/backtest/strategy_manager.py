#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术形态分析系统 - 形态匹配器

负责匹配和分析技术形态
"""

import os
import sys
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.decorators import performance_monitor, time_it
from scripts.backtest.pattern_analyzer import PatternAnalyzer
from scripts.backtest.data_manager import BacktestDataManager

# 获取日志记录器
logger = get_logger(__name__)


class PatternMatcher:
    """
    形态匹配器基类
    
    所有形态匹配器的基类，定义了形态匹配的基本接口
    """

    def __init__(self, matcher_id: str):
        """
        初始化形态匹配器
        
        Args:
            matcher_id: 匹配器ID
        """
        self.matcher_id = matcher_id
        self.data_manager = BacktestDataManager()
        self.pattern_analyzer = PatternAnalyzer()
        
        # 匹配参数
        self.min_pattern_strength = 0.6  # 最小形态强度
        self.patterns = []  # 形态列表
        self.periods = ["daily"]  # 周期列表
        
        # 分析结果
        self.results = {}
        
        logger.info(f"形态匹配器 {matcher_id} 初始化完成")
    
    def set_min_strength(self, min_strength: float):
        """
        设置最小形态强度
        
        Args:
            min_strength: 最小形态强度
        """
        self.min_pattern_strength = min_strength
        logger.info(f"设置最小形态强度: {min_strength}")
    
    def set_patterns(self, pattern_ids: List[str]):
        """
        设置形态列表
        
        Args:
            pattern_ids: 形态ID列表
        """
        self.patterns = pattern_ids
        logger.info(f"设置形态列表: {pattern_ids}")
    
    def set_periods(self, periods: List[str]):
        """
        设置周期列表
        
        Args:
            periods: 周期列表
        """
        self.periods = periods
        logger.info(f"设置周期列表: {periods}")
    
    def _execute_analysis(self, stock_code: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        执行形态分析
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        raise NotImplementedError("子类必须实现_execute_analysis方法")
    
    def analyze(self, stock_code: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        运行形态分析
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            # 获取股票信息
            stock_name = self.data_manager.get_stock_name(stock_code)
            if not stock_name:
                return {
                    "success": False,
                    "error": f"未找到股票 {stock_code} 的信息"
                }
            
            # 执行分析
            result = self._execute_analysis(stock_code, start_date, end_date)
            
            # 添加基本信息
            result.update({
                "success": True,
                "stock_code": stock_code,
                "stock_name": stock_name,
                "start_date": start_date,
                "end_date": end_date,
                "analysis_type": self.matcher_id,
                "last_close": self.data_manager.get_last_close(stock_code, end_date)
            })
            
            # 保存结果
            result_key = f"{stock_code}_{start_date}_{end_date}"
            self.results[result_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"形态分析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def clear_results(self):
        """清除所有分析结果"""
        self.results.clear()
        logger.info("所有分析结果已清除")


class BasicPatternMatcher(PatternMatcher):
    """
    基础形态匹配器
    
    用于识别单个形态
    """

    def __init__(self):
        """初始化基础形态匹配器"""
        super().__init__("pattern")
    
    def _execute_analysis(self, stock_code: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        执行形态分析
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 获取日线数据
        df = self.data_manager.get_stock_data(stock_code, "daily", start_date, end_date)
        if df is None or df.empty:
            return {
                "success": False,
                "error": "获取股票数据失败"
            }
        
        # 分析形态
        patterns = self.pattern_analyzer.analyze_all_patterns(df)
        
        # 过滤形态
        if self.patterns:
            patterns = [p for p in patterns if p["pattern_id"] in self.patterns]
        
        # 过滤强度
        patterns = [p for p in patterns if p["strength"] >= self.min_pattern_strength]
        
        # 按强度排序
        patterns.sort(key=lambda x: x["strength"], reverse=True)
        
        return {
            "patterns": patterns,
            "pattern_count": len(patterns)
        }


class MultiPeriodPatternMatcher(PatternMatcher):
    """
    多周期形态匹配器
    
    用于分析多个周期的形态组合
    """

    def __init__(self):
        """初始化多周期形态匹配器"""
        super().__init__("multi_period")
        self.pattern_combinations = []  # 形态组合
        self.combination_mode = "all"  # 组合模式：all-全部匹配，any-任意匹配
        self.min_pattern_count = 1  # 最小匹配数量
    
    def set_pattern_combinations(self, combinations: List[Dict[str, Any]]):
        """
        设置形态组合
        
        Args:
            combinations: 形态组合列表
        """
        self.pattern_combinations = combinations
        logger.info(f"设置形态组合: {combinations}")
    
    def set_combination_mode(self, mode: str):
        """
        设置组合模式
        
        Args:
            mode: 组合模式，可选值：all-全部匹配，any-任意匹配
        """
        if mode not in ["all", "any"]:
            raise ValueError("组合模式必须是 'all' 或 'any'")
        self.combination_mode = mode
        logger.info(f"设置组合模式: {mode}")
    
    def set_min_pattern_count(self, count: int):
        """
        设置最小匹配数量
        
        Args:
            count: 最小匹配数量
        """
        self.min_pattern_count = count
        logger.info(f"设置最小匹配数量: {count}")
    
    def _calculate_combination_score(self, matched_patterns: List[Dict[str, Any]]) -> float:
        """
        计算组合得分
        
        Args:
            matched_patterns: 匹配的形态列表
            
        Returns:
            float: 组合得分
        """
        if not matched_patterns:
            return 0.0
        
        # 计算各项得分
        strength_score = sum(p["strength"] for p in matched_patterns) / len(matched_patterns)
        count_score = min(len(matched_patterns) / len(self.pattern_combinations), 1.0)
        
        # 计算周期多样性得分
        period_patterns = {}
        for pattern in matched_patterns:
            period = pattern.get("period", "daily")
            if period not in period_patterns:
                period_patterns[period] = []
            period_patterns[period].append(pattern)
        
        diversity_score = len(period_patterns) / len(self.periods)
        
        # 计算最终得分
        weights = {
            "strength": 0.4,
            "count": 0.3,
            "diversity": 0.3
        }
        
        final_score = (
            strength_score * weights["strength"] +
            count_score * weights["count"] +
            diversity_score * weights["diversity"]
        ) * 100
        
        return round(final_score, 2)
    
    def _match_pattern_combinations(self, period_patterns: Dict[str, List[Dict[str, Any]]]) -> Tuple[bool, List[Dict[str, Any]], float]:
        """
        匹配形态组合
        
        Args:
            period_patterns: 各周期的形态列表
            
        Returns:
            Tuple[bool, List[Dict[str, Any]], float]: (是否匹配, 匹配的形态列表, 组合得分)
        """
        if not self.pattern_combinations:
            return False, [], 0.0
        
        matched_patterns = []
        
        # 遍历形态组合
        for combination in self.pattern_combinations:
            required_patterns = combination.get("patterns", [])
            min_strength = combination.get("min_strength", self.min_pattern_strength)
            min_count = combination.get("min_count", 1)
            
            # 检查每个周期
            period_matches = []
            for period in self.periods:
                patterns = period_patterns.get(period, [])
                
                # 检查形态
                for pattern in patterns:
                    if pattern["pattern_id"] in required_patterns:
                        if pattern["strength"] >= min_strength:
                            period_matches.append(pattern)
            
            # 根据组合模式判断是否匹配
            if self.combination_mode == "all":
                # 全部匹配模式：所有形态都必须匹配
                if len(period_matches) >= len(required_patterns):
                    matched_patterns.extend(period_matches)
            else:
                # 任意匹配模式：达到最小匹配数量即可
                if len(period_matches) >= min_count:
                    matched_patterns.extend(period_matches)
        
        # 计算组合得分
        score = self._calculate_combination_score(matched_patterns)
        
        # 判断是否满足最小匹配数量
        is_matched = len(matched_patterns) >= self.min_pattern_count
        
        return is_matched, matched_patterns, score
    
    def _execute_analysis(self, stock_code: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        执行形态分析
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 获取各周期数据
        period_data = {}
        period_patterns = {}
        
        for period in self.periods:
            # 获取数据
            df = self.data_manager.get_stock_data(stock_code, period, start_date, end_date)
            if df is None or df.empty:
                continue
            
            period_data[period] = df
            
            # 分析形态
            patterns = self.pattern_analyzer.analyze_all_patterns(df)
            
            # 过滤形态
            if self.patterns:
                patterns = [p for p in patterns if p["pattern_id"] in self.patterns]
            
            # 过滤强度
            patterns = [p for p in patterns if p["strength"] >= self.min_pattern_strength]
            
            # 添加周期信息
            for pattern in patterns:
                pattern["period"] = period
            
            period_patterns[period] = patterns
        
        # 匹配形态组合
        is_matched, matched_patterns, score = self._match_pattern_combinations(period_patterns)
        
        # 按强度排序
        matched_patterns.sort(key=lambda x: x["strength"], reverse=True)
        
        return {
            "is_matched": is_matched,
            "matched_patterns": matched_patterns,
            "combination_score": score,
            "combination_mode": self.combination_mode,
            "min_pattern_count": self.min_pattern_count,
            "period_patterns": period_patterns
        }


class ZXMPatternMatcher(PatternMatcher):
    """
    ZXM形态匹配器
    
    用于分析ZXM形态系统
    """

    def __init__(self):
        """初始化ZXM形态匹配器"""
        super().__init__("zxm")
        self.score_threshold = 60.0  # 分数阈值
    
    def _calculate_composite_score(self, period_patterns: Dict[str, List[Dict[str, Any]]]) -> float:
        """
        计算综合得分
        
        Args:
            period_patterns: 各周期的形态列表
            
        Returns:
            float: 综合得分
        """
        if not period_patterns:
            return 0.0
        
        # 计算各周期得分
        period_scores = {}
        for period, patterns in period_patterns.items():
            if not patterns:
                period_scores[period] = 0.0
                continue
            
            # 计算周期得分
            strength_sum = sum(p["strength"] for p in patterns)
            pattern_count = len(patterns)
            period_scores[period] = (strength_sum / pattern_count) * 100
        
        # 计算综合得分
        weights = {
            "daily": 0.5,
            "weekly": 0.3,
            "monthly": 0.2
        }
        
        final_score = sum(
            period_scores.get(period, 0.0) * weights.get(period, 0.0)
            for period in self.periods
        )
        
        return round(final_score, 2)
    
    def _get_composite_rating(self, score: float) -> str:
        """
        获取综合评级
        
        Args:
            score: 综合得分
            
        Returns:
            str: 综合评级
        """
        if score >= 80:
            return "强势"
        elif score >= 60:
            return "中性"
        else:
            return "弱势"
    
    def _get_advice(self, score: float, rating: str) -> str:
        """
        获取操作建议
        
        Args:
            score: 综合得分
            rating: 综合评级
            
        Returns:
            str: 操作建议
        """
        if rating == "强势":
            return "形态强势，建议关注"
        elif rating == "中性":
            return "形态中性，建议观察"
        else:
            return "形态弱势，建议回避"
    
    def _execute_analysis(self, stock_code: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        执行形态分析
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 获取各周期数据
        period_data = {}
        period_patterns = {}
        
        for period in self.periods:
            # 获取数据
            df = self.data_manager.get_stock_data(stock_code, period, start_date, end_date)
            if df is None or df.empty:
                continue
            
            period_data[period] = df
            
            # 分析形态
            patterns = self.pattern_analyzer.analyze_all_patterns(df)
            
            # 过滤形态
            if self.patterns:
                patterns = [p for p in patterns if p["pattern_id"] in self.patterns]
            
            # 过滤强度
            patterns = [p for p in patterns if p["strength"] >= self.min_pattern_strength]
            
            # 添加周期信息
            for pattern in patterns:
                pattern["period"] = period
            
            period_patterns[period] = patterns
        
        # 计算综合得分
        composite_score = self._calculate_composite_score(period_patterns)
        
        # 获取综合评级
        composite_rating = self._get_composite_rating(composite_score)
        
        # 获取操作建议
        advice = self._get_advice(composite_score, composite_rating)
        
        # 判断是否有形态信号
        has_pattern = composite_score >= self.score_threshold
        
        # 生成分析文本
        zxm_text = f"ZXM形态分析结果:\n"
        zxm_text += f"综合得分: {composite_score}\n"
        zxm_text += f"综合评级: {composite_rating}\n"
        zxm_text += f"操作建议: {advice}\n"
        zxm_text += f"\n各周期形态:\n"
        
        for period in self.periods:
            patterns = period_patterns.get(period, [])
            zxm_text += f"\n{period}周期:\n"
            if patterns:
                for pattern in patterns:
                    zxm_text += f"- {pattern['pattern_name']}: 强度 {pattern['strength']:.2f}\n"
            else:
                zxm_text += "未发现形态\n"
        
        return {
            "composite_score": composite_score,
            "composite_rating": composite_rating,
            "advice": advice,
            "has_pattern": has_pattern,
            "period_patterns": period_patterns,
            "zxm_text": zxm_text
        }


class StrategyManager:
    def __init__(self):
        pass


class PatternManager:
    """
    形态管理器
    
    负责管理和运行形态匹配器
    """

    def __init__(self):
        """初始化形态管理器"""
        self.matchers = {
            "pattern": BasicPatternMatcher(),
            "multi_period": MultiPeriodPatternMatcher(),
            "zxm": ZXMPatternMatcher()
        }
        logger.info("形态管理器初始化完成")
    
    def get_matcher(self, matcher_id: str) -> PatternMatcher:
        """
        获取形态匹配器
        
        Args:
            matcher_id: 匹配器ID
            
        Returns:
            PatternMatcher: 形态匹配器实例
        """
        matcher = self.matchers.get(matcher_id)
        if not matcher:
            raise ValueError(f"未找到匹配器: {matcher_id}")
        return matcher
    
    def run_analysis(self, matcher_id: str, stock_code: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        运行形态分析
        
        Args:
            matcher_id: 匹配器ID
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        matcher = self.get_matcher(matcher_id)
        return matcher.analyze(stock_code, start_date, end_date)
    
    def batch_analyze(self, matcher_id: str, stock_codes: List[str], start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        批量运行形态分析
        
        Args:
            matcher_id: 匹配器ID
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        matcher = self.get_matcher(matcher_id)
        results = []
        
        for stock_code in stock_codes:
            result = matcher.analyze(stock_code, start_date, end_date)
            results.append(result)
        
        return results
    
    def clear_results(self):
        """清除所有分析结果"""
        for matcher in self.matchers.values():
            matcher.clear_results()
        logger.info("所有分析结果已清除") 