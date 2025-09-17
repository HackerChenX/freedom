#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试结果验证器单元测试
"""

import unittest
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from tests.comprehensive.test_result_validator import TestResultValidator, PatternValidationResult, IndicatorValidationResult, ComprehensiveValidationResult
from db.sql_manager import SQLManager, QueryType


# 模拟测试结果类
@dataclass
class MockVerificationResult:
    stock_code: str
    date: str
    expected_pattern: str
    detected_patterns: List[str]
    pattern_match: bool
    confidence_score: float
    verification_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockPatternTestResult:
    pattern_id: str
    pattern_name: str
    stocks_selected: int
    verifications_attempted: int
    verifications_successful: int
    success_rate: float
    selected_stocks: List[Any] = field(default_factory=list)
    verification_results: List[MockVerificationResult] = field(default_factory=list)


@dataclass
class MockIndicatorTestResult:
    indicator_name: str
    total_patterns: int
    patterns_tested: int
    patterns_with_selections: int
    total_stocks_selected: int
    verification_success_rate: float
    pattern_results: Dict[str, MockPatternTestResult] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockTestResults:
    test_id: str
    start_time: datetime
    end_time: datetime
    total_indicators_tested: int
    total_patterns_tested: int
    total_stocks_selected: int
    total_verifications_performed: int
    overall_success_rate: float
    indicator_results: Dict[str, MockIndicatorTestResult] = field(default_factory=dict)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)


class TestResultValidatorTest(unittest.TestCase):
    """测试结果验证器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.validator = TestResultValidator()
        
        # 创建模拟测试结果
        self.test_results = self._create_mock_test_results()
    
    def _create_mock_test_results(self) -> MockTestResults:
        """创建模拟测试结果"""
        # 创建验证结果
        verification_results = [
            MockVerificationResult(
                stock_code="000001",
                date="20240101",
                expected_pattern="MA_GOLDEN_CROSS",
                detected_patterns=["MA_GOLDEN_CROSS"],
                pattern_match=True,
                confidence_score=0.8
            ),
            MockVerificationResult(
                stock_code="000002",
                date="20240102",
                expected_pattern="MA_GOLDEN_CROSS",
                detected_patterns=["MA_DEATH_CROSS"],
                pattern_match=False,
                confidence_score=0.2
            )
        ]
        
        # 创建形态测试结果
        pattern_results = {
            "MA_GOLDEN_CROSS": MockPatternTestResult(
                pattern_id="MA_GOLDEN_CROSS",
                pattern_name="MA金叉",
                stocks_selected=2,
                verifications_attempted=2,
                verifications_successful=1,
                success_rate=0.5,
                verification_results=verification_results
            )
        }
        
        # 创建指标测试结果
        indicator_results = {
            "MA": MockIndicatorTestResult(
                indicator_name="MA",
                total_patterns=1,
                patterns_tested=1,
                patterns_with_selections=1,
                total_stocks_selected=2,
                verification_success_rate=0.5,
                pattern_results=pattern_results
            )
        }
        
        # 创建测试结果
        return MockTestResults(
            test_id="test_001",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_indicators_tested=1,
            total_patterns_tested=1,
            total_stocks_selected=2,
            total_verifications_performed=2,
            overall_success_rate=0.5,
            indicator_results=indicator_results
        )
    
    def test_validate_test_results(self):
        """测试验证测试结果"""
        validation_result = self.validator.validate_test_results(self.test_results)
        
        self.assertIsNotNone(validation_result)
        self.assertEqual(validation_result.test_id, "test_001")
        self.assertEqual(validation_result.total_indicators, 1)
        self.assertEqual(validation_result.total_patterns, 1)
        self.assertEqual(validation_result.patterns_with_selections, 1)
    
    def test_validate_pattern_selection_quality(self):
        """测试验证形态选股质量"""
        pattern_results = list(self.test_results.indicator_results["MA"].pattern_results.values())
        quality_report = self.validator.validate_pattern_selection_quality(pattern_results)
        
        self.assertIsNotNone(quality_report)
        self.assertEqual(quality_report['total_patterns'], 1)
        self.assertEqual(quality_report['patterns_with_selections'], 1)
        self.assertEqual(quality_report['total_stocks_selected'], 2)
    
    def test_validate_verification_consistency(self):
        """测试验证闭环验证一致性"""
        verification_results = self.test_results.indicator_results["MA"].pattern_results["MA_GOLDEN_CROSS"].verification_results
        consistency_report = self.validator.validate_verification_consistency(verification_results)
        
        self.assertIsNotNone(consistency_report)
        self.assertEqual(consistency_report['total_verifications'], 2)
        self.assertEqual(consistency_report['successful_verifications'], 1)
        self.assertEqual(consistency_report['pattern_matches'], 1)


if __name__ == '__main__':
    unittest.main()