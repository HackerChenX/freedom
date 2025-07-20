#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
买点分析测试运行器

执行买点分析功能的综合测试套件
"""

import unittest
import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class BuyPointTestRunner:
    """买点分析测试运行器"""
    
    def __init__(self):
        """初始化测试运行器"""
        self.test_suite = None
        self.test_results = None
        
    def discover_tests(self, test_pattern: str = "test_*.py") -> unittest.TestSuite:
        """发现测试用例"""
        logger.info("发现买点分析测试用例...")
        
        test_dir = Path(__file__).parent
        loader = unittest.TestLoader()
        
        # 发现所有测试
        suite = loader.discover(
            start_dir=str(test_dir),
            pattern=test_pattern,
            top_level_dir=str(project_root)
        )
        
        test_count = suite.countTestCases()
        logger.info(f"发现 {test_count} 个测试用例")
        
        return suite
    
    def run_specific_tests(self, test_names: List[str]) -> unittest.TestResult:
        """运行指定的测试"""
        logger.info(f"运行指定测试: {test_names}")
        
        from tests.buypoint_analysis.test_buypoint_comprehensive import BuyPointAnalysisTestSuite
        
        suite = unittest.TestSuite()
        
        for test_name in test_names:
            if hasattr(BuyPointAnalysisTestSuite, test_name):
                suite.addTest(BuyPointAnalysisTestSuite(test_name))
            else:
                logger.warning(f"测试方法不存在: {test_name}")
        
        return self._execute_test_suite(suite)
    
    def run_category_tests(self, category: str) -> unittest.TestResult:
        """运行特定类别的测试"""
        logger.info(f"运行类别测试: {category}")
        
        from tests.buypoint_analysis.test_buypoint_comprehensive import BuyPointAnalysisTestSuite
        
        # 定义类别到测试方法的映射
        category_mappings = {
            'trend': [
                'test_ma_golden_cross', 'test_ma_death_cross', 'test_ma_bullish_alignment',
                'test_ema_golden_cross', 'test_dmi_golden_cross'
            ],
            'oscillator': [
                'test_rsi_overbought', 'test_rsi_oversold', 'test_rsi_golden_cross',
                'test_kdj_golden_cross', 'test_kdj_death_cross', 'test_kdj_overbought', 'test_kdj_oversold'
            ],
            'momentum': [
                'test_macd_golden_cross', 'test_macd_death_cross', 
                'test_macd_above_zero_golden', 'test_macd_histogram_divergence'
            ],
            'volume': [
                'test_obv_golden_cross', 'test_pvt_golden_cross', 'test_mfi_overbought'
            ],
            'volatility': [
                'test_boll_upper_breakout', 'test_boll_lower_breakout', 
                'test_boll_squeeze', 'test_boll_expansion'
            ],
            'candlestick': [
                'test_doji_pattern', 'test_hammer_pattern', 'test_shooting_star_pattern',
                'test_engulfing_pattern', 'test_morning_star_pattern', 'test_evening_star_pattern',
                'test_three_white_soldiers_pattern', 'test_three_black_crows_pattern'
            ],
            'integration': ['test_multiple_patterns_integration'],
            'negative': ['test_negative_patterns'],
            'performance': ['test_performance_benchmark']
        }
        
        if category not in category_mappings:
            logger.error(f"未知的测试类别: {category}")
            logger.info(f"可用类别: {list(category_mappings.keys())}")
            return None
        
        return self.run_specific_tests(category_mappings[category])
    
    def run_all_tests(self) -> unittest.TestResult:
        """运行所有测试"""
        logger.info("运行所有买点分析测试...")
        
        suite = self.discover_tests()
        return self._execute_test_suite(suite)
    
    def run_quick_tests(self) -> unittest.TestResult:
        """运行快速测试（核心形态）"""
        logger.info("运行快速测试...")
        
        quick_tests = [
            'test_macd_golden_cross',
            'test_rsi_overbought', 
            'test_kdj_golden_cross',
            'test_boll_upper_breakout',
            'test_doji_pattern'
        ]
        
        return self.run_specific_tests(quick_tests)
    
    def _execute_test_suite(self, suite: unittest.TestSuite) -> unittest.TestResult:
        """执行测试套件"""
        logger.info(f"开始执行 {suite.countTestCases()} 个测试...")
        
        start_time = time.time()
        
        # 创建测试运行器
        runner = unittest.TextTestRunner(
            verbosity=2,
            stream=sys.stdout,
            buffer=True
        )
        
        # 执行测试
        result = runner.run(suite)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 输出结果摘要
        self._print_test_summary(result, execution_time)
        
        return result
    
    def _print_test_summary(self, result: unittest.TestResult, execution_time: float):
        """打印测试摘要"""
        print("\n" + "=" * 80)
        print("买点分析测试结果摘要")
        print("=" * 80)
        
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        successful = total_tests - failures - errors - skipped
        
        print(f"总测试数: {total_tests}")
        print(f"成功: {successful}")
        print(f"失败: {failures}")
        print(f"错误: {errors}")
        print(f"跳过: {skipped}")
        print(f"成功率: {successful/total_tests:.1%}" if total_tests > 0 else "成功率: 0%")
        print(f"执行时间: {execution_time:.2f}秒")
        
        if result.failures:
            print(f"\n失败的测试:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print(f"\n错误的测试:")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        print("=" * 80)
        
        # 判断整体结果
        if failures == 0 and errors == 0:
            print("🎉 所有测试通过！买点分析系统工作正常。")
        elif successful / total_tests >= 0.8:
            print("✅ 大部分测试通过，系统基本正常。")
        else:
            print("⚠️ 多个测试失败，需要检查买点分析系统。")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="买点分析功能测试运行器")
    
    parser.add_argument(
        '--mode', 
        choices=['all', 'quick', 'category', 'specific'],
        default='all',
        help='测试模式'
    )
    
    parser.add_argument(
        '--category',
        choices=['trend', 'oscillator', 'momentum', 'volume', 'volatility', 'candlestick', 'integration', 'negative', 'performance'],
        help='测试类别（当mode=category时使用）'
    )
    
    parser.add_argument(
        '--tests',
        nargs='+',
        help='指定的测试方法名（当mode=specific时使用）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 80)
    print("买点分析功能测试套件")
    print("=" * 80)
    
    runner = BuyPointTestRunner()
    result = None
    
    try:
        if args.mode == 'all':
            result = runner.run_all_tests()
        elif args.mode == 'quick':
            result = runner.run_quick_tests()
        elif args.mode == 'category':
            if not args.category:
                print("错误: 类别模式需要指定 --category 参数")
                return 1
            result = runner.run_category_tests(args.category)
        elif args.mode == 'specific':
            if not args.tests:
                print("错误: 指定模式需要提供 --tests 参数")
                return 1
            result = runner.run_specific_tests(args.tests)
        
        if result is None:
            return 1
        
        # 返回适当的退出码
        if result.failures or result.errors:
            return 1
        else:
            return 0
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        return 1
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
