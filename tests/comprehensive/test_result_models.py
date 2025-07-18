#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试结果模型

提供详细的测试结果数据结构，支持完整的测试结果表示和序列化
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import json
import pandas as pd

from utils.logger import getLogger

logger = getLogger(__name__)


@dataclass
class StockData:
    """股票数据"""
    code: str
    name: str
    date: str
    industry: str = ""
    price: float = 0.0
    volume: float = 0.0
    turnover_rate: float = 0.0
    market_cap: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StockSelection:
    """股票选择结果"""
    stock_code: str
    stock_name: str
    date: str
    pattern_id: str
    indicator_name: str
    confidence_score: float
    selection_details: Dict[str, Any] = field(default_factory=dict)
    technical_values: Dict[str, float] = field(default_factory=dict)
    stock_data: Optional[StockData] = None


@dataclass
class VerificationResult:
    """验证结果"""
    stock_code: str
    date: str
    expected_pattern: str
    detected_patterns: List[str]
    pattern_match: bool
    confidence_score: float
    verification_details: Dict[str, Any] = field(default_factory=dict)
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'stock_code': self.stock_code,
            'date': self.date,
            'expected_pattern': self.expected_pattern,
            'detected_patterns': self.detected_patterns,
            'pattern_match': self.pattern_match,
            'confidence_score': self.confidence_score,
            'verification_details': self.verification_details,
            'technical_indicators': self.technical_indicators
        }


@dataclass
class PatternTestResult:
    """形态测试结果"""
    pattern_id: str
    pattern_name: str
    indicator_name: str
    stocks_selected: int
    verifications_attempted: int
    verifications_successful: int
    success_rate: float
    execution_time: float = 0.0
    selected_stocks: List[StockSelection] = field(default_factory=list)
    verification_results: List[VerificationResult] = field(default_factory=list)
    pattern_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'indicator_name': self.indicator_name,
            'stocks_selected': self.stocks_selected,
            'verifications_attempted': self.verifications_attempted,
            'verifications_successful': self.verifications_successful,
            'success_rate': self.success_rate,
            'execution_time': self.execution_time,
            'pattern_metadata': self.pattern_metadata,
            'selected_stocks_count': len(self.selected_stocks),
            'verification_results_count': len(self.verification_results)
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        data = []
        for stock in self.selected_stocks:
            # 查找对应的验证结果
            verification = next(
                (v for v in self.verification_results 
                 if v.stock_code == stock.stock_code and v.date == stock.date),
                None
            )
            
            row = {
                'pattern_id': self.pattern_id,
                'indicator_name': self.indicator_name,
                'stock_code': stock.stock_code,
                'stock_name': stock.stock_name,
                'date': stock.date,
                'selection_confidence': stock.confidence_score
            }
            
            if verification:
                row.update({
                    'verification_match': verification.pattern_match,
                    'verification_confidence': verification.confidence_score,
                    'detected_patterns': ','.join(verification.detected_patterns)
                })
            
            data.append(row)
        
        return pd.DataFrame(data)


@dataclass
class IndicatorTestResult:
    """指标测试结果"""
    indicator_name: str
    total_patterns: int
    patterns_tested: int
    patterns_with_selections: int
    total_stocks_selected: int
    verification_success_rate: float
    execution_time: float = 0.0
    pattern_results: Dict[str, PatternTestResult] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    indicator_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'indicator_name': self.indicator_name,
            'total_patterns': self.total_patterns,
            'patterns_tested': self.patterns_tested,
            'patterns_with_selections': self.patterns_with_selections,
            'total_stocks_selected': self.total_stocks_selected,
            'verification_success_rate': self.verification_success_rate,
            'execution_time': self.execution_time,
            'performance_metrics': self.performance_metrics,
            'indicator_metadata': self.indicator_metadata,
            'pattern_results_count': len(self.pattern_results)
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        data = []
        for pattern_id, pattern_result in self.pattern_results.items():
            data.append({
                'indicator_name': self.indicator_name,
                'pattern_id': pattern_id,
                'pattern_name': pattern_result.pattern_name,
                'stocks_selected': pattern_result.stocks_selected,
                'verifications_attempted': pattern_result.verifications_attempted,
                'verifications_successful': pattern_result.verifications_successful,
                'success_rate': pattern_result.success_rate,
                'execution_time': pattern_result.execution_time
            })
        
        return pd.DataFrame(data)
    
    def get_successful_patterns(self) -> List[str]:
        """获取成功的形态列表"""
        return [
            pattern_id for pattern_id, result in self.pattern_results.items()
            if result.success_rate >= 0.7 and result.stocks_selected > 0
        ]
    
    def get_failed_patterns(self) -> List[str]:
        """获取失败的形态列表"""
        return [
            pattern_id for pattern_id, result in self.pattern_results.items()
            if result.success_rate < 0.5 or result.stocks_selected == 0
        ]


@dataclass
class TestResultSummary:
    """测试结果摘要"""
    total_indicators: int
    total_patterns: int
    total_stocks_selected: int
    total_verifications: int
    successful_verifications: int
    overall_success_rate: float
    execution_time: float
    indicators_with_stocks: int
    patterns_with_stocks: int
    unique_stocks: int
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_indicators': self.total_indicators,
            'total_patterns': self.total_patterns,
            'total_stocks_selected': self.total_stocks_selected,
            'total_verifications': self.total_verifications,
            'successful_verifications': self.successful_verifications,
            'overall_success_rate': self.overall_success_rate,
            'execution_time': self.execution_time,
            'indicators_with_stocks': self.indicators_with_stocks,
            'patterns_with_stocks': self.patterns_with_stocks,
            'unique_stocks': self.unique_stocks,
            'performance_metrics': self.performance_metrics
        }


@dataclass
class TestResults:
    """综合测试结果"""
    test_id: str
    start_time: datetime
    end_time: datetime
    total_indicators_tested: int
    total_patterns_tested: int
    total_stocks_selected: int
    total_verifications_performed: int
    overall_success_rate: float
    indicator_results: Dict[str, IndicatorTestResult] = field(default_factory=dict)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'test_id': self.test_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_indicators_tested': self.total_indicators_tested,
            'total_patterns_tested': self.total_patterns_tested,
            'total_stocks_selected': self.total_stocks_selected,
            'total_verifications_performed': self.total_verifications_performed,
            'overall_success_rate': self.overall_success_rate,
            'summary_statistics': self.summary_statistics,
            'indicator_results_count': len(self.indicator_results)
        }
    
    def to_json(self, file_path: str) -> None:
        """保存为JSON文件"""
        result_dict = self.to_dict()
        
        # 添加指标结果摘要
        result_dict['indicator_summaries'] = {
            name: result.to_dict()
            for name, result in self.indicator_results.items()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试结果已保存到: {file_path}")
    
    def get_summary(self) -> TestResultSummary:
        """获取测试结果摘要"""
        # 计算唯一股票数
        unique_stocks = set()
        indicators_with_stocks = 0
        patterns_with_stocks = 0
        
        for indicator_name, indicator_result in self.indicator_results.items():
            if indicator_result.total_stocks_selected > 0:
                indicators_with_stocks += 1
            
            for pattern_id, pattern_result in indicator_result.pattern_results.items():
                if pattern_result.stocks_selected > 0:
                    patterns_with_stocks += 1
                
                for stock in pattern_result.selected_stocks:
                    unique_stocks.add(stock.stock_code)
        
        return TestResultSummary(
            total_indicators=self.total_indicators_tested,
            total_patterns=self.total_patterns_tested,
            total_stocks_selected=self.total_stocks_selected,
            total_verifications=self.total_verifications_performed,
            successful_verifications=int(self.total_verifications_performed * self.overall_success_rate),
            overall_success_rate=self.overall_success_rate,
            execution_time=(self.end_time - self.start_time).total_seconds(),
            indicators_with_stocks=indicators_with_stocks,
            patterns_with_stocks=patterns_with_stocks,
            unique_stocks=len(unique_stocks),
            performance_metrics=self.summary_statistics.get('performance_metrics', {})
        )
    
    def get_top_patterns(self, limit: int = 10) -> List[Tuple[str, PatternTestResult]]:
        """获取最成功的形态"""
        all_patterns = []
        
        for indicator_name, indicator_result in self.indicator_results.items():
            for pattern_id, pattern_result in indicator_result.pattern_results.items():
                if pattern_result.stocks_selected > 0 and pattern_result.success_rate > 0:
                    all_patterns.append((pattern_id, pattern_result))
        
        # 按成功率排序
        return sorted(
            all_patterns,
            key=lambda x: (x[1].success_rate, x[1].stocks_selected),
            reverse=True
        )[:limit]
    
    def get_top_stocks(self, limit: int = 10) -> List[Tuple[str, str, int]]:
        """获取最常被选中的股票"""
        stock_counts = {}
        stock_names = {}
        
        for indicator_result in self.indicator_results.values():
            for pattern_result in indicator_result.pattern_results.values():
                for stock in pattern_result.selected_stocks:
                    if stock.stock_code not in stock_counts:
                        stock_counts[stock.stock_code] = 0
                        stock_names[stock.stock_code] = stock.stock_name
                    
                    stock_counts[stock.stock_code] += 1
        
        # 按选中次数排序
        top_stocks = sorted(
            stock_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [(code, stock_names[code], count) for code, count in top_stocks]


def merge_test_results(results_list: List[TestResults]) -> TestResults:
    """
    合并多个测试结果
    
    Args:
        results_list: 测试结果列表
        
    Returns:
        TestResults: 合并后的测试结果
    """
    if not results_list:
        return None
    
    # 使用第一个结果作为基础
    base_result = results_list[0]
    
    # 创建新的合并结果
    merged_result = TestResults(
        test_id=f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        start_time=min(r.start_time for r in results_list),
        end_time=max(r.end_time for r in results_list),
        total_indicators_tested=sum(r.total_indicators_tested for r in results_list),
        total_patterns_tested=sum(r.total_patterns_tested for r in results_list),
        total_stocks_selected=sum(r.total_stocks_selected for r in results_list),
        total_verifications_performed=sum(r.total_verifications_performed for r in results_list),
        overall_success_rate=0.0  # 稍后计算
    )
    
    # 合并指标结果
    all_indicator_results = {}
    for result in results_list:
        for indicator_name, indicator_result in result.indicator_results.items():
            if indicator_name not in all_indicator_results:
                all_indicator_results[indicator_name] = indicator_result
            else:
                # 合并同一指标的结果
                existing = all_indicator_results[indicator_name]
                existing.total_patterns += indicator_result.total_patterns
                existing.patterns_tested += indicator_result.patterns_tested
                existing.patterns_with_selections += indicator_result.patterns_with_selections
                existing.total_stocks_selected += indicator_result.total_stocks_selected
                
                # 合并形态结果
                for pattern_id, pattern_result in indicator_result.pattern_results.items():
                    if pattern_id not in existing.pattern_results:
                        existing.pattern_results[pattern_id] = pattern_result
                    else:
                        # 合并同一形态的结果
                        existing_pattern = existing.pattern_results[pattern_id]
                        existing_pattern.stocks_selected += pattern_result.stocks_selected
                        existing_pattern.verifications_attempted += pattern_result.verifications_attempted
                        existing_pattern.verifications_successful += pattern_result.verifications_successful
                        
                        # 更新成功率
                        if existing_pattern.verifications_attempted > 0:
                            existing_pattern.success_rate = (
                                existing_pattern.verifications_successful / 
                                existing_pattern.verifications_attempted
                            )
                        
                        # 合并选股结果和验证结果
                        existing_pattern.selected_stocks.extend(pattern_result.selected_stocks)
                        existing_pattern.verification_results.extend(pattern_result.verification_results)
                
                # 更新指标成功率
                total_verifications = sum(
                    p.verifications_attempted for p in existing.pattern_results.values()
                )
                successful_verifications = sum(
                    p.verifications_successful for p in existing.pattern_results.values()
                )
                
                if total_verifications > 0:
                    existing.verification_success_rate = successful_verifications / total_verifications
    
    # 设置合并后的指标结果
    merged_result.indicator_results = all_indicator_results
    
    # 计算总体成功率
    total_verifications = merged_result.total_verifications_performed
    successful_verifications = sum(
        sum(p.verifications_successful for p in ind.pattern_results.values())
        for ind in all_indicator_results.values()
    )
    
    if total_verifications > 0:
        merged_result.overall_success_rate = successful_verifications / total_verifications
    
    # 合并摘要统计
    merged_summary = {}
    for result in results_list:
        for key, value in result.summary_statistics.items():
            if key not in merged_summary:
                merged_summary[key] = value
            elif isinstance(value, (int, float)):
                merged_summary[key] = merged_summary.get(key, 0) + value
            elif isinstance(value, dict):
                if key not in merged_summary:
                    merged_summary[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        merged_summary[key][sub_key] = merged_summary[key].get(sub_key, 0) + sub_value
    
    merged_result.summary_statistics = merged_summary
    
    return merged_result