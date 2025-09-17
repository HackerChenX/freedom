#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试结果验证器

基于现有验证器组件，专门验证综合选股测试的结果
确保每个形态至少选出一只股票，验证测试成功标准和失败分析
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from utils.logger import getLogger
from .validators import RealDataValidator, DataQualityChecker
from .stock_selection_tester import TestResults, IndicatorTestResult, PatternTestResult, VerificationResult
from .buypoint_verification_engine import BatchVerificationResult
from db.sql_manager import SQLManager, QueryType

logger = getLogger(__name__)


@dataclass
class PatternValidationResult:
    """形态验证结果"""
    pattern_id: str
    indicator_name: str
    has_selections: bool
    selection_count: int
    verification_success_rate: float
    confidence_score: float
    validation_passed: bool
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class IndicatorValidationResult:
    """指标验证结果"""
    indicator_name: str
    total_patterns: int
    patterns_with_selections: int
    patterns_validation_passed: int
    overall_success_rate: float
    validation_passed: bool
    pattern_results: List[PatternValidationResult] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


@dataclass
class ComprehensiveValidationResult:
    """综合验证结果"""
    test_id: str
    validation_time: datetime
    total_indicators: int
    total_patterns: int
    patterns_with_selections: int
    patterns_validation_passed: int
    overall_validation_passed: bool
    validation_score: float
    indicator_results: List[IndicatorValidationResult] = field(default_factory=list)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class TestResultValidator:
    """测试结果验证器"""
    
    def __init__(self):
        """初始化测试结果验证器"""
        # 复用现有验证组件
        self.data_validator = RealDataValidator()
        self.quality_checker = DataQualityChecker()
        
        # 验证标准
        self.validation_criteria = {
            'min_selection_per_pattern': 1,        # 每个形态至少选出1只股票
            'min_verification_success_rate': 0.5,  # 最低验证成功率50%
            'min_confidence_score': 0.3,           # 最低置信度30%
            'min_pattern_success_rate': 0.7,       # 指标中至少70%的形态成功
            'min_overall_success_rate': 0.6        # 总体成功率至少60%
        }
        
        logger.info("测试结果验证器初始化完成")
    
    def validate_test_results(self, test_results: TestResults) -> ComprehensiveValidationResult:
        """
        验证综合测试结果
        
        Args:
            test_results: 综合测试结果
            
        Returns:
            ComprehensiveValidationResult: 综合验证结果
        """
        logger.info(f"开始验证测试结果: {test_results.test_id}")
        
        validation_result = ComprehensiveValidationResult(
            test_id=test_results.test_id,
            validation_time=datetime.now(),
            total_indicators=test_results.total_indicators_tested,
            total_patterns=test_results.total_patterns_tested,
            patterns_with_selections=0,
            patterns_validation_passed=0,
            overall_validation_passed=False,
            validation_score=0.0
        )
        
        try:
            # 验证每个指标
            indicator_validations = []
            total_patterns_with_selections = 0
            total_patterns_validation_passed = 0
            
            for indicator_name, indicator_result in test_results.indicator_results.items():
                indicator_validation = self._validate_indicator_result(
                    indicator_name, indicator_result
                )
                indicator_validations.append(indicator_validation)
                
                total_patterns_with_selections += indicator_validation.patterns_with_selections
                total_patterns_validation_passed += indicator_validation.patterns_validation_passed
            
            validation_result.indicator_results = indicator_validations
            validation_result.patterns_with_selections = total_patterns_with_selections
            validation_result.patterns_validation_passed = total_patterns_validation_passed
            
            # 计算总体验证结果
            validation_result.overall_validation_passed = self._check_overall_validation(
                validation_result, test_results
            )
            
            # 计算验证评分
            validation_result.validation_score = self._calculate_validation_score(
                validation_result, test_results
            )
            
            # 生成汇总统计
            validation_result.summary_statistics = self._generate_validation_summary(
                validation_result, test_results
            )
            
            # 生成建议
            validation_result.recommendations = self._generate_recommendations(
                validation_result, test_results
            )
            
            logger.info(f"测试结果验证完成: 验证通过={validation_result.overall_validation_passed}, 评分={validation_result.validation_score:.2f}")
            
        except Exception as e:
            logger.error(f"验证测试结果失败: {e}")
            validation_result.recommendations.append(f"验证过程出错: {str(e)}")
        
        return validation_result
    
    def _validate_indicator_result(self, 
                                 indicator_name: str, 
                                 indicator_result: IndicatorTestResult) -> IndicatorValidationResult:
        """验证指标结果"""
        logger.debug(f"验证指标结果: {indicator_name}")
        
        validation_result = IndicatorValidationResult(
            indicator_name=indicator_name,
            total_patterns=indicator_result.total_patterns,
            patterns_with_selections=0,
            patterns_validation_passed=0,
            overall_success_rate=indicator_result.verification_success_rate,
            validation_passed=False
        )
        
        # 验证每个形态
        pattern_validations = []
        patterns_with_selections = 0
        patterns_validation_passed = 0
        
        for pattern_id, pattern_result in indicator_result.pattern_results.items():
            pattern_validation = self._validate_pattern_result(
                pattern_id, indicator_name, pattern_result
            )
            pattern_validations.append(pattern_validation)
            
            if pattern_validation.has_selections:
                patterns_with_selections += 1
            
            if pattern_validation.validation_passed:
                patterns_validation_passed += 1
        
        validation_result.pattern_results = pattern_validations
        validation_result.patterns_with_selections = patterns_with_selections
        validation_result.patterns_validation_passed = patterns_validation_passed
        
        # 检查指标级别的验证标准
        if validation_result.total_patterns > 0:
            pattern_success_rate = patterns_validation_passed / validation_result.total_patterns
            validation_result.validation_passed = (
                pattern_success_rate >= self.validation_criteria['min_pattern_success_rate']
            )
            
            if not validation_result.validation_passed:
                validation_result.issues.append(
                    f"形态成功率 {pattern_success_rate:.1%} 低于标准 {self.validation_criteria['min_pattern_success_rate']:.1%}"
                )
        
        return validation_result
    
    def _validate_pattern_result(self, 
                               pattern_id: str,
                               indicator_name: str,
                               pattern_result: PatternTestResult) -> PatternValidationResult:
        """验证形态结果"""
        logger.debug(f"验证形态结果: {pattern_id}")
        
        validation_result = PatternValidationResult(
            pattern_id=pattern_id,
            indicator_name=indicator_name,
            has_selections=pattern_result.stocks_selected > 0,
            selection_count=pattern_result.stocks_selected,
            verification_success_rate=pattern_result.success_rate,
            confidence_score=self._calculate_pattern_confidence(pattern_result),
            validation_passed=False
        )
        
        # 检查选股数量标准
        if pattern_result.stocks_selected < self.validation_criteria['min_selection_per_pattern']:
            validation_result.issues.append(
                f"选股数量 {pattern_result.stocks_selected} 低于最低标准 {self.validation_criteria['min_selection_per_pattern']}"
            )
        
        # 检查验证成功率标准
        if pattern_result.success_rate < self.validation_criteria['min_verification_success_rate']:
            validation_result.issues.append(
                f"验证成功率 {pattern_result.success_rate:.1%} 低于标准 {self.validation_criteria['min_verification_success_rate']:.1%}"
            )
        
        # 检查置信度标准
        if validation_result.confidence_score < self.validation_criteria['min_confidence_score']:
            validation_result.issues.append(
                f"置信度 {validation_result.confidence_score:.2f} 低于标准 {self.validation_criteria['min_confidence_score']:.2f}"
            )
        
        # 综合判断验证是否通过
        validation_result.validation_passed = (
            validation_result.has_selections and
            pattern_result.success_rate >= self.validation_criteria['min_verification_success_rate'] and
            validation_result.confidence_score >= self.validation_criteria['min_confidence_score']
        )
        
        # 生成建议
        if not validation_result.validation_passed:
            if not validation_result.has_selections:
                validation_result.recommendations.append("需要调整选股策略以确保能选出股票")
            if pattern_result.success_rate < self.validation_criteria['min_verification_success_rate']:
                validation_result.recommendations.append("需要改进形态匹配算法提高验证成功率")
            if validation_result.confidence_score < self.validation_criteria['min_confidence_score']:
                validation_result.recommendations.append("需要优化置信度计算方法")
        
        return validation_result
    
    def _calculate_pattern_confidence(self, pattern_result: PatternTestResult) -> float:
        """计算形态置信度"""
        if not pattern_result.verification_results:
            return 0.0
        
        # 基于验证结果计算平均置信度
        confidence_scores = [
            result.confidence_score for result in pattern_result.verification_results
            if result.confidence_score > 0
        ]
        
        if not confidence_scores:
            return 0.0
        
        return sum(confidence_scores) / len(confidence_scores)
    
    def _check_overall_validation(self, 
                                validation_result: ComprehensiveValidationResult,
                                test_results: TestResults) -> bool:
        """检查总体验证是否通过"""
        # 检查总体成功率
        if test_results.overall_success_rate < self.validation_criteria['min_overall_success_rate']:
            return False
        
        # 检查形态覆盖率
        if validation_result.total_patterns > 0:
            pattern_coverage = validation_result.patterns_with_selections / validation_result.total_patterns
            if pattern_coverage < 0.5:  # 至少50%的形态有选股
                return False
        
        # 检查指标覆盖率
        if validation_result.total_indicators > 0:
            successful_indicators = sum(
                1 for indicator in validation_result.indicator_results 
                if indicator.validation_passed
            )
            indicator_coverage = successful_indicators / validation_result.total_indicators
            if indicator_coverage < 0.5:  # 至少50%的指标验证通过
                return False
        
        return True
    
    def _calculate_validation_score(self, 
                                  validation_result: ComprehensiveValidationResult,
                                  test_results: TestResults) -> float:
        """计算验证评分"""
        scores = []
        
        # 总体成功率评分
        overall_score = min(1.0, test_results.overall_success_rate / self.validation_criteria['min_overall_success_rate'])
        scores.append(overall_score)
        
        # 形态覆盖率评分
        if validation_result.total_patterns > 0:
            pattern_coverage = validation_result.patterns_with_selections / validation_result.total_patterns
            coverage_score = min(1.0, pattern_coverage / 0.7)  # 目标70%覆盖率
            scores.append(coverage_score)
        
        # 验证通过率评分
        if validation_result.total_patterns > 0:
            validation_pass_rate = validation_result.patterns_validation_passed / validation_result.total_patterns
            pass_rate_score = min(1.0, validation_pass_rate / 0.6)  # 目标60%通过率
            scores.append(pass_rate_score)
        
        # 指标成功率评分
        if validation_result.total_indicators > 0:
            successful_indicators = sum(
                1 for indicator in validation_result.indicator_results 
                if indicator.validation_passed
            )
            indicator_success_rate = successful_indicators / validation_result.total_indicators
            indicator_score = min(1.0, indicator_success_rate / 0.5)  # 目标50%指标成功
            scores.append(indicator_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _generate_validation_summary(self, 
                                   validation_result: ComprehensiveValidationResult,
                                   test_results: TestResults) -> Dict[str, Any]:
        """生成验证汇总统计"""
        return {
            'validation_overview': {
                'total_indicators_tested': validation_result.total_indicators,
                'total_patterns_tested': validation_result.total_patterns,
                'patterns_with_selections': validation_result.patterns_with_selections,
                'patterns_validation_passed': validation_result.patterns_validation_passed,
                'overall_validation_passed': validation_result.overall_validation_passed,
                'validation_score': validation_result.validation_score
            },
            'coverage_statistics': {
                'pattern_coverage_rate': (
                    validation_result.patterns_with_selections / validation_result.total_patterns
                    if validation_result.total_patterns > 0 else 0
                ),
                'pattern_success_rate': (
                    validation_result.patterns_validation_passed / validation_result.total_patterns
                    if validation_result.total_patterns > 0 else 0
                ),
                'indicator_success_rate': (
                    sum(1 for ind in validation_result.indicator_results if ind.validation_passed) / 
                    validation_result.total_indicators
                    if validation_result.total_indicators > 0 else 0
                )
            },
            'test_execution_stats': {
                'total_stocks_selected': test_results.total_stocks_selected,
                'total_verifications_performed': test_results.total_verifications_performed,
                'overall_success_rate': test_results.overall_success_rate,
                'execution_time': (
                    (test_results.end_time - test_results.start_time).total_seconds()
                    if test_results.end_time > test_results.start_time else 0
                )
            },
            'validation_criteria': self.validation_criteria
        }
    
    def _generate_recommendations(self, 
                                validation_result: ComprehensiveValidationResult,
                                test_results: TestResults) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于总体验证结果的建议
        if not validation_result.overall_validation_passed:
            recommendations.append("总体验证未通过，需要系统性改进")
        
        # 基于覆盖率的建议
        if validation_result.total_patterns > 0:
            pattern_coverage = validation_result.patterns_with_selections / validation_result.total_patterns
            if pattern_coverage < 0.5:
                recommendations.append(f"形态覆盖率过低 ({pattern_coverage:.1%})，建议检查选股算法")
        
        # 基于成功率的建议
        if test_results.overall_success_rate < self.validation_criteria['min_overall_success_rate']:
            recommendations.append(f"总体成功率 ({test_results.overall_success_rate:.1%}) 低于标准，建议优化验证逻辑")
        
        # 基于指标级别的建议
        failed_indicators = [
            ind for ind in validation_result.indicator_results 
            if not ind.validation_passed
        ]
        
        if len(failed_indicators) > validation_result.total_indicators * 0.5:
            recommendations.append("超过一半的指标验证失败，建议全面检查指标实现")
        
        # 基于形态级别的建议
        patterns_without_selections = validation_result.total_patterns - validation_result.patterns_with_selections
        if patterns_without_selections > 0:
            recommendations.append(f"有 {patterns_without_selections} 个形态未选出股票，建议调整选股条件")
        
        # 性能相关建议
        execution_time = (test_results.end_time - test_results.start_time).total_seconds()
        if execution_time > 300:  # 超过5分钟
            recommendations.append("执行时间超过5分钟限制，建议优化性能")
        
        # 数据质量建议
        if test_results.total_stocks_selected == 0:
            recommendations.append("未选出任何股票，建议检查数据源和选股逻辑")
        
        # 验证质量建议
        if test_results.total_verifications_performed == 0:
            recommendations.append("未执行任何验证，建议检查验证流程")
        
        return recommendations
    
    def validate_pattern_selection_quality(self, 
                                         pattern_results: List[PatternTestResult]) -> Dict[str, Any]:
        """验证形态选股质量"""
        logger.info("验证形态选股质量...")
        
        quality_report = {
            'total_patterns': len(pattern_results),
            'patterns_with_selections': 0,
            'total_stocks_selected': 0,
            'selection_distribution': {},
            'quality_issues': [],
            'quality_score': 0.0
        }
        
        try:
            selection_counts = []
            
            for pattern_result in pattern_results:
                if pattern_result.stocks_selected > 0:
                    quality_report['patterns_with_selections'] += 1
                    quality_report['total_stocks_selected'] += pattern_result.stocks_selected
                    selection_counts.append(pattern_result.stocks_selected)
                
                # 检查选股质量
                if pattern_result.stocks_selected == 0:
                    quality_report['quality_issues'].append(
                        f"形态 {pattern_result.pattern_id} 未选出任何股票"
                    )
                elif pattern_result.success_rate < 0.3:
                    quality_report['quality_issues'].append(
                        f"形态 {pattern_result.pattern_id} 验证成功率过低: {pattern_result.success_rate:.1%}"
                    )
            
            # 选股分布统计
            if selection_counts:
                quality_report['selection_distribution'] = {
                    'min_selections': min(selection_counts),
                    'max_selections': max(selection_counts),
                    'avg_selections': sum(selection_counts) / len(selection_counts),
                    'median_selections': np.median(selection_counts)
                }
            
            # 计算质量评分
            if quality_report['total_patterns'] > 0:
                coverage_score = quality_report['patterns_with_selections'] / quality_report['total_patterns']
                issue_penalty = min(0.5, len(quality_report['quality_issues']) / quality_report['total_patterns'])
                quality_report['quality_score'] = max(0.0, coverage_score - issue_penalty)
            
            logger.info(f"形态选股质量验证完成，评分: {quality_report['quality_score']:.2f}")
            
        except Exception as e:
            logger.error(f"验证形态选股质量失败: {e}")
            quality_report['quality_issues'].append(f"质量验证过程出错: {str(e)}")
        
        return quality_report
    
    def validate_verification_consistency(self, 
                                        verification_results: List[VerificationResult]) -> Dict[str, Any]:
        """验证闭环验证一致性"""
        logger.info("验证闭环验证一致性...")
        
        consistency_report = {
            'total_verifications': len(verification_results),
            'successful_verifications': 0,
            'pattern_matches': 0,
            'consistency_score': 0.0,
            'consistency_issues': []
        }
        
        try:
            pattern_match_count = 0
            confidence_scores = []
            
            for result in verification_results:
                if result.pattern_match:
                    pattern_match_count += 1
                    consistency_report['successful_verifications'] += 1
                
                if result.confidence_score > 0:
                    confidence_scores.append(result.confidence_score)
                
                # 检查一致性问题
                if result.expected_pattern not in result.detected_patterns and result.pattern_match:
                    consistency_report['consistency_issues'].append(
                        f"股票 {result.stock_code} 验证结果不一致：期望 {result.expected_pattern}，检测到 {result.detected_patterns}"
                    )
            
            consistency_report['pattern_matches'] = pattern_match_count
            
            # 计算一致性评分
            if consistency_report['total_verifications'] > 0:
                match_rate = pattern_match_count / consistency_report['total_verifications']
                issue_penalty = min(0.3, len(consistency_report['consistency_issues']) / consistency_report['total_verifications'])
                consistency_report['consistency_score'] = max(0.0, match_rate - issue_penalty)
            
            # 置信度统计
            if confidence_scores:
                consistency_report['confidence_statistics'] = {
                    'avg_confidence': sum(confidence_scores) / len(confidence_scores),
                    'min_confidence': min(confidence_scores),
                    'max_confidence': max(confidence_scores)
                }
            
            logger.info(f"闭环验证一致性验证完成，评分: {consistency_report['consistency_score']:.2f}")
            
        except Exception as e:
            logger.error(f"验证闭环验证一致性失败: {e}")
            consistency_report['consistency_issues'].append(f"一致性验证过程出错: {str(e)}")
        
        return consistency_report


# 全局测试结果验证器实例
_test_result_validator = None


def get_test_result_validator() -> TestResultValidator:
    """
    获取全局测试结果验证器实例
    
    Returns:
        TestResultValidator: 测试结果验证器实例
    """
    global _test_result_validator
    if _test_result_validator is None:
        _test_result_validator = TestResultValidator()
    return _test_result_validator


def main():
    """测试结果验证器测试"""
    print("测试结果验证器功能测试...")
    
    validator = TestResultValidator()
    print(f"验证标准: {validator.validation_criteria}")
    
    # 这里可以添加更多测试逻辑
    print("测试结果验证器初始化成功")


if __name__ == "__main__":
    main()