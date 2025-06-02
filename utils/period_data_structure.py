#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
周期分离数据结构模块

提供用于存储和管理按周期组织的分析结果的数据结构
"""

import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple, Set
from collections import defaultdict
import json

from enums.period import Period
from utils.logger import get_logger

logger = get_logger(__name__)


class IndicatorPeriodResult:
    """
    指标在特定周期下的分析结果
    
    存储单个指标在特定周期下的计算结果、形态和评分
    """
    
    def __init__(self, indicator_name: str, period: Period):
        """
        初始化指标周期结果
        
        Args:
            indicator_name: 指标名称
            period: 周期类型
        """
        self.indicator_name = indicator_name
        self.period = period
        self.data = None  # 计算结果数据
        self.patterns = []  # 识别出的形态列表
        self.score = None  # 指标评分
        self.signals = {}  # 指标信号
        self.metadata = {}  # 其他元数据
        
    def set_data(self, data: pd.DataFrame):
        """设置计算结果数据"""
        self.data = data
        
    def add_pattern(self, pattern: Dict[str, Any]):
        """添加识别出的形态"""
        self.patterns.append(pattern)
        
    def set_patterns(self, patterns: List[Dict[str, Any]]):
        """设置形态列表"""
        self.patterns = patterns
        
    def set_score(self, score: Union[float, pd.Series]):
        """设置指标评分"""
        self.score = score
        
    def set_signals(self, signals: Dict[str, pd.Series]):
        """设置指标信号"""
        self.signals = signals
        
    def add_metadata(self, key: str, value: Any):
        """添加元数据"""
        self.metadata[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        result = {
            'indicator_name': self.indicator_name,
            'period': self.period.value,
            'patterns': self.patterns,
            'metadata': self.metadata
        }
        
        # 处理pandas对象
        if self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                # 只保留最后几行数据
                last_n = min(10, len(self.data))
                result['data'] = self.data.tail(last_n).to_dict(orient='records')
            else:
                result['data'] = self.data
                
        if self.score is not None:
            if isinstance(self.score, pd.Series):
                result['score'] = float(self.score.iloc[-1])
            else:
                result['score'] = float(self.score)
                
        # 处理信号
        if self.signals:
            signal_dict = {}
            for signal_name, signal_series in self.signals.items():
                if isinstance(signal_series, pd.Series):
                    # 只保留最后几个信号
                    last_n = min(5, len(signal_series))
                    signal_dict[signal_name] = signal_series.tail(last_n).to_dict()
                else:
                    signal_dict[signal_name] = signal_series
            result['signals'] = signal_dict
            
        return result


class PeriodAnalysisResult:
    """
    周期分析结果容器
    
    存储特定周期下所有指标的分析结果
    """
    
    def __init__(self, period: Period):
        """
        初始化周期分析结果
        
        Args:
            period: 周期类型
        """
        self.period = period
        self.indicators = {}  # 指标名称 -> IndicatorPeriodResult
        self.metadata = {}  # 周期相关元数据
        
    def add_indicator_result(self, indicator_result: IndicatorPeriodResult):
        """添加指标结果"""
        self.indicators[indicator_result.indicator_name] = indicator_result
        
    def get_indicator_result(self, indicator_name: str) -> Optional[IndicatorPeriodResult]:
        """获取指标结果"""
        return self.indicators.get(indicator_name)
        
    def get_all_patterns(self) -> List[Dict[str, Any]]:
        """获取所有指标的形态"""
        all_patterns = []
        for indicator_result in self.indicators.values():
            all_patterns.extend(indicator_result.patterns)
        return all_patterns
        
    def add_metadata(self, key: str, value: Any):
        """添加元数据"""
        self.metadata[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            'period': self.period.value,
            'indicators': {name: result.to_dict() for name, result in self.indicators.items()},
            'metadata': self.metadata,
            'all_patterns': self.get_all_patterns()
        }


class MultiPeriodAnalysisResult:
    """
    多周期分析结果
    
    存储所有周期的分析结果，并提供跨周期分析功能
    """
    
    def __init__(self, stock_code: str, buy_date: Optional[str] = None):
        """
        初始化多周期分析结果
        
        Args:
            stock_code: 股票代码
            buy_date: 买点日期，可选
        """
        self.stock_code = stock_code
        self.buy_date = buy_date
        self.periods = {}  # 周期 -> PeriodAnalysisResult
        self.metadata = {
            'stock_code': stock_code,
            'buy_date': buy_date
        }
        
    def add_period_result(self, period_result: PeriodAnalysisResult):
        """添加周期结果"""
        self.periods[period_result.period] = period_result
        
    def get_period_result(self, period: Period) -> Optional[PeriodAnalysisResult]:
        """获取周期结果"""
        return self.periods.get(period)
        
    def get_indicator_result(self, indicator_name: str, period: Period) -> Optional[IndicatorPeriodResult]:
        """获取指定周期和指标的结果"""
        period_result = self.get_period_result(period)
        if period_result:
            return period_result.get_indicator_result(indicator_name)
        return None
        
    def extract_cross_period_patterns(self) -> Dict[str, List[Tuple[Period, Dict[str, Any]]]]:
        """
        提取跨周期形态，按指标分组
        
        Returns:
            Dict[str, List[Tuple[Period, Dict[str, Any]]]]: 指标名称 -> [(周期, 形态), ...]
        """
        cross_period_patterns = defaultdict(list)
        
        for period, period_result in self.periods.items():
            for indicator_name, indicator_result in period_result.indicators.items():
                for pattern in indicator_result.patterns:
                    cross_period_patterns[indicator_name].append((period, pattern))
                    
        return dict(cross_period_patterns)
        
    def get_common_patterns(self) -> List[Dict[str, Any]]:
        """
        获取在多个周期中共同出现的形态
        
        Returns:
            List[Dict[str, Any]]: 共同形态列表
        """
        # 按pattern_id分组形态
        pattern_groups = defaultdict(list)
        
        for period, period_result in self.periods.items():
            for indicator_name, indicator_result in period_result.indicators.items():
                for pattern in indicator_result.patterns:
                    if 'pattern_id' in pattern:
                        pattern_groups[pattern['pattern_id']].append((period, pattern))
        
        # 找出在多个周期中出现的形态
        common_patterns = []
        for pattern_id, occurrences in pattern_groups.items():
            if len(occurrences) > 1:  # 在多个周期中出现
                periods = [period.value for period, _ in occurrences]
                pattern = occurrences[0][1].copy()  # 使用第一个出现的形态作为基础
                pattern['periods'] = periods
                common_patterns.append(pattern)
                
        return common_patterns
        
    def get_pattern_distribution(self) -> Dict[str, Dict[Period, int]]:
        """
        获取形态在各周期的分布
        
        Returns:
            Dict[str, Dict[Period, int]]: 形态ID -> {周期 -> 出现次数}
        """
        distribution = defaultdict(lambda: defaultdict(int))
        
        for period, period_result in self.periods.items():
            for indicator_name, indicator_result in period_result.indicators.items():
                for pattern in indicator_result.patterns:
                    if 'pattern_id' in pattern:
                        distribution[pattern['pattern_id']][period] += 1
                        
        return dict(distribution)
    
    def add_metadata(self, key: str, value: Any):
        """添加元数据"""
        self.metadata[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        result = {
            'stock_code': self.stock_code,
            'buy_date': self.buy_date,
            'periods': {period.value: period_result.to_dict() 
                       for period, period_result in self.periods.items()},
            'metadata': self.metadata,
            'common_patterns': self.get_common_patterns()
        }
        
        return result
        
    def to_json(self, file_path: Optional[str] = None) -> Optional[str]:
        """
        转换为JSON字符串或保存到文件
        
        Args:
            file_path: 保存路径，如不指定则返回JSON字符串
            
        Returns:
            Optional[str]: 如未指定保存路径，则返回JSON字符串
        """
        # 使用自定义JSON编码器处理特殊类型
        class ResultEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (pd.DataFrame, pd.Series)):
                    return obj.to_dict()
                if isinstance(obj, Period):
                    return obj.value
                return super(ResultEncoder, self).default(obj)
        
        result_dict = self.to_dict()
        json_str = json.dumps(result_dict, cls=ResultEncoder, indent=2, ensure_ascii=False)
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                logger.info(f"保存分析结果到文件: {file_path}")
                return None
            except Exception as e:
                logger.error(f"保存分析结果到文件时出错: {e}")
                return json_str
        else:
            return json_str 