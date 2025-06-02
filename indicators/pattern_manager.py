#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pattern Manager - 形态管理器

管理和组织各种技术指标形态的中心模块，支持跨周期、跨指标的形态识别和分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass
from datetime import datetime

from indicators.pattern_registry import PatternRegistry
from indicators.base_indicator import PatternResult
from enums.kline_period import KlinePeriod
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PatternOccurrence:
    """形态出现记录"""
    pattern_id: str
    indicator_id: str
    period: str
    date: str
    strength: float
    display_name: str
    details: Dict[str, Any] = None


class PatternManager:
    """
    形态管理器
    
    管理和组织各种技术指标形态，支持：
    1. 形态的注册和检索
    2. 跨周期形态分析
    3. 多指标形态组合分析
    4. 形态统计和评分
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'PatternManager':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = PatternManager()
        return cls._instance
    
    def __init__(self):
        """初始化形态管理器"""
        self.registry = PatternRegistry
        self.pattern_occurrences = []  # 存储形态出现记录
        self.period_weights = {  # 不同周期的权重配置
            "1min": 0.3,
            "5min": 0.4,
            "15min": 0.5,
            "30min": 0.6,
            "60min": 0.7,
            "DAILY": 1.0,
            "WEEKLY": 1.2,
            "MONTHLY": 1.5
        }
    
    def register_pattern_occurrence(self, 
                                  pattern_id: str,
                                  indicator_id: str,
                                  period: str,
                                  date: str,
                                  strength: float,
                                  display_name: str = None,
                                  details: Dict[str, Any] = None) -> None:
        """
        注册形态出现记录
        
        Args:
            pattern_id: 形态ID
            indicator_id: 指标ID
            period: 周期
            date: 日期
            strength: 强度
            display_name: 显示名称
            details: 详细信息
        """
        if display_name is None:
            display_name = self.registry.get_display_name(pattern_id)
            
        occurrence = PatternOccurrence(
            pattern_id=pattern_id,
            indicator_id=indicator_id,
            period=period,
            date=date,
            strength=strength,
            display_name=display_name,
            details=details or {}
        )
        
        self.pattern_occurrences.append(occurrence)
        logger.debug(f"注册形态出现: {indicator_id}:{pattern_id} 在 {period} 周期的 {date}")
    
    def register_multiple_patterns(self, 
                                 patterns: List[Dict[str, Any]],
                                 indicator_id: str,
                                 period: str,
                                 date: str) -> None:
        """
        批量注册多个形态出现记录
        
        Args:
            patterns: 形态列表，每个形态是包含pattern_id、strength等属性的字典
            indicator_id: 指标ID
            period: 周期
            date: 日期
        """
        for pattern in patterns:
            self.register_pattern_occurrence(
                pattern_id=pattern.get('pattern_id'),
                indicator_id=indicator_id,
                period=period,
                date=date,
                strength=pattern.get('strength', 50.0),
                display_name=pattern.get('display_name'),
                details=pattern.get('details', {})
            )
    
    def clear_occurrences(self) -> None:
        """清除所有形态出现记录"""
        self.pattern_occurrences = []
        logger.debug("已清除所有形态出现记录")
    
    def get_patterns_by_date(self, date: str) -> List[PatternOccurrence]:
        """
        获取指定日期的所有形态
        
        Args:
            date: 日期
            
        Returns:
            List[PatternOccurrence]: 形态出现记录列表
        """
        return [p for p in self.pattern_occurrences if p.date == date]
    
    def get_patterns_by_period(self, period: str) -> List[PatternOccurrence]:
        """
        获取指定周期的所有形态
        
        Args:
            period: 周期
            
        Returns:
            List[PatternOccurrence]: 形态出现记录列表
        """
        return [p for p in self.pattern_occurrences if p.period == period]
    
    def get_patterns_by_indicator(self, indicator_id: str) -> List[PatternOccurrence]:
        """
        获取指定指标的所有形态
        
        Args:
            indicator_id: 指标ID
            
        Returns:
            List[PatternOccurrence]: 形态出现记录列表
        """
        return [p for p in self.pattern_occurrences if p.indicator_id == indicator_id]
    
    def get_patterns_by_type(self, pattern_type: str) -> List[PatternOccurrence]:
        """
        获取指定类型的所有形态
        
        Args:
            pattern_type: 形态类型（如"bullish"、"bearish"）
            
        Returns:
            List[PatternOccurrence]: 形态出现记录列表
        """
        result = []
        for p in self.pattern_occurrences:
            signal_type = self.registry.get_signal_type(p.pattern_id)
            if signal_type == pattern_type:
                result.append(p)
        return result
    
    def get_multi_period_patterns(self, 
                                indicator_id: str, 
                                pattern_id: str,
                                periods: List[str] = None) -> Dict[str, List[PatternOccurrence]]:
        """
        获取指定指标和形态在多个周期上的出现情况
        
        Args:
            indicator_id: 指标ID
            pattern_id: 形态ID
            periods: 周期列表，如果为None则查询所有周期
            
        Returns:
            Dict[str, List[PatternOccurrence]]: 按周期分组的形态出现记录
        """
        result = {}
        
        # 如果未指定周期，则获取所有存在的周期
        if periods is None:
            periods = list(set(p.period for p in self.pattern_occurrences))
        
        for period in periods:
            period_patterns = [
                p for p in self.pattern_occurrences 
                if p.indicator_id == indicator_id and 
                p.pattern_id == pattern_id and 
                p.period == period
            ]
            if period_patterns:
                result[period] = period_patterns
        
        return result
    
    def calculate_pattern_score(self, 
                              date: str,
                              weight_by_period: bool = True) -> Dict[str, float]:
        """
        计算指定日期所有形态的综合评分
        
        Args:
            date: 日期
            weight_by_period: 是否按周期加权
            
        Returns:
            Dict[str, float]: 形态评分，包含总分和细分评分
        """
        date_patterns = self.get_patterns_by_date(date)
        
        if not date_patterns:
            return {
                "total_score": 50.0,  # 中性评分
                "bullish_score": 0.0,
                "bearish_score": 0.0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0
            }
        
        bullish_score = 0.0
        bearish_score = 0.0
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for pattern in date_patterns:
            # 获取形态信号类型
            signal_type = self.registry.get_signal_type(pattern.pattern_id)
            
            # 获取形态评分影响
            score_impact = self.registry.get_score_impact(pattern.pattern_id)
            
            # 应用周期权重
            if weight_by_period:
                period_weight = self.period_weights.get(pattern.period, 1.0)
                weighted_impact = score_impact * period_weight
            else:
                weighted_impact = score_impact
            
            # 根据信号类型分类
            if signal_type == "bullish" or weighted_impact > 0:
                bullish_score += abs(weighted_impact)
                bullish_count += 1
            elif signal_type == "bearish" or weighted_impact < 0:
                bearish_score += abs(weighted_impact)
                bearish_count += 1
            else:
                neutral_count += 1
        
        # 计算总评分
        total_patterns = bullish_count + bearish_count + neutral_count
        
        if total_patterns > 0:
            # 看涨得分与看跌得分的差值决定最终倾向
            if bullish_score > bearish_score:
                # 看涨倾向
                score_diff = bullish_score - bearish_score
                total_score = 50.0 + min(50.0, score_diff)
            elif bearish_score > bullish_score:
                # 看跌倾向
                score_diff = bearish_score - bullish_score
                total_score = 50.0 - min(50.0, score_diff)
            else:
                # 中性
                total_score = 50.0
        else:
            total_score = 50.0  # 默认中性
        
        return {
            "total_score": total_score,
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count
        }
    
    def get_common_patterns(self, 
                          periods: List[str] = None,
                          min_occurrence: int = 2) -> List[Dict[str, Any]]:
        """
        获取跨周期共同出现的形态
        
        Args:
            periods: 要分析的周期列表，如果为None则分析所有周期
            min_occurrence: 最小出现次数
            
        Returns:
            List[Dict[str, Any]]: 共同形态列表
        """
        if periods is None:
            periods = list(set(p.period for p in self.pattern_occurrences))
            
        # 获取所有形态ID和指标ID的组合
        pattern_indicator_pairs = set(
            (p.pattern_id, p.indicator_id) for p in self.pattern_occurrences
        )
        
        common_patterns = []
        
        for pattern_id, indicator_id in pattern_indicator_pairs:
            # 统计该形态在各周期中的出现情况
            period_occurrences = {}
            
            for period in periods:
                period_patterns = [
                    p for p in self.pattern_occurrences 
                    if p.pattern_id == pattern_id and 
                    p.indicator_id == indicator_id and 
                    p.period == period
                ]
                
                if period_patterns:
                    period_occurrences[period] = period_patterns
            
            # 如果形态在足够多的周期中出现
            if len(period_occurrences) >= min_occurrence:
                # 计算形态的平均强度
                all_patterns = [p for patterns in period_occurrences.values() for p in patterns]
                avg_strength = sum(p.strength for p in all_patterns) / len(all_patterns)
                
                # 创建共同形态记录
                common_pattern = {
                    "pattern_id": pattern_id,
                    "indicator_id": indicator_id,
                    "display_name": all_patterns[0].display_name,
                    "periods": list(period_occurrences.keys()),
                    "avg_strength": avg_strength,
                    "occurrence_count": len(period_occurrences),
                    "details": {
                        period: [p.details for p in patterns] 
                        for period, patterns in period_occurrences.items()
                    }
                }
                
                common_patterns.append(common_pattern)
        
        # 按出现次数和强度排序
        common_patterns.sort(key=lambda x: (x["occurrence_count"], x["avg_strength"]), reverse=True)
        
        return common_patterns
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """
        获取形态统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        if not self.pattern_occurrences:
            return {
                "total_patterns": 0,
                "by_indicator": {},
                "by_period": {},
                "by_signal_type": {},
                "most_common": []
            }
        
        # 按指标统计
        by_indicator = {}
        for p in self.pattern_occurrences:
            if p.indicator_id not in by_indicator:
                by_indicator[p.indicator_id] = 0
            by_indicator[p.indicator_id] += 1
        
        # 按周期统计
        by_period = {}
        for p in self.pattern_occurrences:
            if p.period not in by_period:
                by_period[p.period] = 0
            by_period[p.period] += 1
        
        # 按信号类型统计
        by_signal_type = {"bullish": 0, "bearish": 0, "neutral": 0}
        for p in self.pattern_occurrences:
            signal_type = self.registry.get_signal_type(p.pattern_id)
            if signal_type:
                if signal_type not in by_signal_type:
                    by_signal_type[signal_type] = 0
                by_signal_type[signal_type] += 1
            else:
                by_signal_type["neutral"] += 1
        
        # 最常见的形态
        pattern_counts = {}
        for p in self.pattern_occurrences:
            key = (p.indicator_id, p.pattern_id)
            if key not in pattern_counts:
                pattern_counts[key] = 0
            pattern_counts[key] += 1
        
        most_common = []
        for (indicator_id, pattern_id), count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            display_name = next((p.display_name for p in self.pattern_occurrences 
                              if p.indicator_id == indicator_id and p.pattern_id == pattern_id), pattern_id)
            most_common.append({
                "indicator_id": indicator_id,
                "pattern_id": pattern_id,
                "display_name": display_name,
                "count": count
            })
        
        return {
            "total_patterns": len(self.pattern_occurrences),
            "by_indicator": by_indicator,
            "by_period": by_period,
            "by_signal_type": by_signal_type,
            "most_common": most_common
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        将形态出现记录转换为DataFrame
        
        Returns:
            pd.DataFrame: 形态数据框
        """
        if not self.pattern_occurrences:
            return pd.DataFrame(columns=[
                'pattern_id', 'indicator_id', 'period', 'date', 
                'strength', 'display_name', 'signal_type'
            ])
        
        data = []
        for p in self.pattern_occurrences:
            signal_type = self.registry.get_signal_type(p.pattern_id)
            data.append({
                'pattern_id': p.pattern_id,
                'indicator_id': p.indicator_id,
                'period': p.period,
                'date': p.date,
                'strength': p.strength,
                'display_name': p.display_name,
                'signal_type': signal_type
            })
        
        return pd.DataFrame(data) 