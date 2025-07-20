#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行更新后的综合测试套件

整合反向验证、买点分析、形态识别等测试，验证重构后系统的准确性
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tests.reverse_validation.reverse_validation_framework import Reverse_validation_framework
from tests.comprehensive.stock_selection_tester import ComprehensiveStockSelectionTester
from utils.logger import getLogger

logger = getLogger(__name__)


class UpdatedComprehensiveTestSuite:
    """更新后的综合测试套件"""
    
    def __init__(self):
        """初始化测试套件"""
        self.reverse_validation = Reverse_validation_framework()
        self.comprehensive_tester = ComprehensiveStockSelectionTester()
        self.test_results = {}
        self.start_time = None
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("=" * 100)
        print("股票选股系统 - 综合测试套件 (重构后版本)")
        print("=" * 100)
        
        self.start_time = datetime.now()
        overall_results = {
            'test_suite_info': {
                'version': '2.0_refactored',
                'start_time': self.start_time.isoformat(),
                'test_types': ['system_validation', 'reverse_validation', 'pattern_recognition', 'buypoint_analysis']
            },
            'results': {}
        }
        
        try:
            # 1. 系统验证测试
            print("\n🔍 第1阶段: 系统验证测试")
            print("-" * 50)
            system_validation = await self._run_system_validation()
            overall_results['results']['system_validation'] = system_validation
            
            if not system_validation.get('overall_status', False):
                print("❌ 系统验证失败，停止后续测试")
                return overall_results
            
            # 2. 反向验证测试
            print("\n🔄 第2阶段: 反向验证测试")
            print("-" * 50)
            reverse_validation = await self._run_reverse_validation()
            overall_results['results']['reverse_validation'] = reverse_validation
            
            # 3. 形态识别测试
            print("\n📊 第3阶段: 形态识别测试")
            print("-" * 50)
            pattern_recognition = await self._run_pattern_recognition_tests()
            overall_results['results']['pattern_recognition'] = pattern_recognition
            
            # 4. 买点分析测试
            print("\n🎯 第4阶段: 买点分析测试")
            print("-" * 50)
            buypoint_analysis = await self._run_buypoint_analysis_tests()
            overall_results['results']['buypoint_analysis'] = buypoint_analysis
            
            # 5. 性能基准测试
            print("\n⚡ 第5阶段: 性能基准测试")
            print("-" * 50)
            performance_benchmark = await self._run_performance_benchmark()
            overall_results['results']['performance_benchmark'] = performance_benchmark
            
            # 计算总体结果
            overall_results['summary'] = self._calculate_overall_summary(overall_results['results'])
            
        except Exception as e:
            logger.error(f"测试套件执行失败: {e}")
            overall_results['error'] = str(e)
            
        finally:
            end_time = datetime.now()
            overall_results['test_suite_info']['end_time'] = end_time.isoformat()
            overall_results['test_suite_info']['total_duration'] = (end_time - self.start_time).total_seconds()
            
        return overall_results
    
    async def _run_system_validation(self) -> Dict[str, Any]:
        """运行系统验证测试"""
        print("验证重构后的系统组件...")
        
        # 验证反向验证框架
        rv_validation = self.reverse_validation.validate_refactored_system()
        print(f"反向验证框架: {'✓' if rv_validation['overall_status'] else '✗'}")
        
        # 验证综合测试器
        ct_validation = self.comprehensive_tester._validate_refactored_components()
        ct_overall = all(ct_validation.values()) if ct_validation else False
        print(f"综合测试器: {'✓' if ct_overall else '✗'}")
        
        # 测试核心功能
        core_functions = await self._test_core_functions()
        print(f"核心功能: {'✓' if core_functions['all_passed'] else '✗'}")
        
        overall_status = (
            rv_validation['overall_status'] and 
            ct_overall and 
            core_functions['all_passed']
        )
        
        return {
            'reverse_validation_framework': rv_validation,
            'comprehensive_tester': ct_validation,
            'core_functions': core_functions,
            'overall_status': overall_status
        }
    
    async def _test_core_functions(self) -> Dict[str, Any]:
        """测试核心功能"""
        tests = {
            'pattern_generation': False,
            'buypoint_analysis': False,
            'pattern_registry': False,
            'data_access': False
        }
        
        try:
            # 测试形态生成
            test_data = self.reverse_validation.pattern_generator.generate_pattern_data(
                "MACD_GOLDEN_CROSS", 30, "CORE_TEST"
            )
            tests['pattern_generation'] = test_data is not None and not test_data.empty
            
            # 测试买点分析
            analysis_result = self.reverse_validation.buypoint_analyzer.analyze_stock(
                "000001", "20240101", "核心测试"
            )
            tests['buypoint_analysis'] = analysis_result is not None
            
            # 测试形态注册表
            patterns = self.reverse_validation.pattern_registry.get_all_patterns()
            tests['pattern_registry'] = len(patterns) > 0
            
            # 测试数据访问
            tests['data_access'] = self.reverse_validation.data_access is not None
            
        except Exception as e:
            logger.error(f"核心功能测试失败: {e}")
        
        tests['all_passed'] = all(tests.values())
        return tests
    
    async def _run_reverse_validation(self) -> Dict[str, Any]:
        """运行反向验证测试"""
        print("执行反向验证测试...")
        
        # 测试核心指标
        core_indicators = ['MACD', 'RSI', 'KDJ', 'BOLL']
        result = await self.reverse_validation.run_comprehensive_validation_async(core_indicators)
        
        summary = result['test_summary']
        print(f"测试完成: {summary['successful_tests']}/{summary['total_tests']} 成功")
        print(f"成功率: {summary['success_rate']:.1%}")
        print(f"耗时: {summary['duration_seconds']:.2f}秒")
        
        return result
    
    async def _run_pattern_recognition_tests(self) -> Dict[str, Any]:
        """运行形态识别测试"""
        print("执行形态识别测试...")
        
        # 获取所有已注册的形态
        all_patterns = self.reverse_validation.pattern_registry.get_all_patterns()
        
        # 选择一些代表性形态进行测试
        test_patterns = [
            'MACD_GOLDEN_CROSS', 'RSI_OVERBOUGHT', 'KDJ_GOLDEN_CROSS',
            'BOLL_UPPER_BREAKOUT', 'MA_GOLDEN_CROSS', 'DMI_GOLDEN_CROSS'
        ]
        
        pattern_results = {}
        successful_recognitions = 0
        total_recognitions = 0
        
        for pattern_id in test_patterns:
            if pattern_id in all_patterns or pattern_id in self.reverse_validation.expected_patterns:
                try:
                    # 生成测试数据
                    test_data = self.reverse_validation.pattern_generator.generate_pattern_data(
                        pattern_id, 60, f"PATTERN_TEST_{pattern_id}"
                    )
                    
                    if test_data is not None and not test_data.empty:
                        # 运行识别测试
                        result = self.reverse_validation.run_single_pattern_validation(
                            pattern_id.split('_')[0], pattern_id, test_data
                        )
                        
                        pattern_results[pattern_id] = result
                        total_recognitions += 1
                        
                        if result['is_successful']:
                            successful_recognitions += 1
                            
                        print(f"  {pattern_id}: {'✓' if result['is_successful'] else '✗'} "
                              f"(匹配度: {result['match_score']:.2f})")
                    
                except Exception as e:
                    logger.error(f"形态识别测试失败 {pattern_id}: {e}")
                    pattern_results[pattern_id] = {'error': str(e), 'is_successful': False}
        
        recognition_rate = successful_recognitions / total_recognitions if total_recognitions > 0 else 0
        print(f"形态识别成功率: {recognition_rate:.1%}")
        
        return {
            'pattern_results': pattern_results,
            'summary': {
                'total_patterns_tested': total_recognitions,
                'successful_recognitions': successful_recognitions,
                'recognition_rate': recognition_rate,
                'tested_patterns': test_patterns
            }
        }
    
    async def _run_buypoint_analysis_tests(self) -> Dict[str, Any]:
        """运行买点分析测试"""
        print("执行买点分析测试...")
        
        # 测试不同类型的买点
        test_stocks = [
            ("000001", "20240101", "平安银行"),
            ("000002", "20240101", "万科A"),
            ("000858", "20240101", "五粮液")
        ]
        
        analysis_results = {}
        successful_analyses = 0
        
        for stock_code, date, name in test_stocks:
            try:
                result = self.reverse_validation.buypoint_analyzer.analyze_stock(
                    stock_code, date, name
                )
                
                analysis_results[stock_code] = {
                    'success': result is not None,
                    'result': result,
                    'indicators_count': len(result) if result else 0
                }
                
                if result is not None:
                    successful_analyses += 1
                    print(f"  {stock_code} ({name}): ✓ 分析成功")
                else:
                    print(f"  {stock_code} ({name}): ✗ 分析失败")
                    
            except Exception as e:
                logger.error(f"买点分析失败 {stock_code}: {e}")
                analysis_results[stock_code] = {'success': False, 'error': str(e)}
        
        analysis_rate = successful_analyses / len(test_stocks)
        print(f"买点分析成功率: {analysis_rate:.1%}")
        
        return {
            'analysis_results': analysis_results,
            'summary': {
                'total_stocks_tested': len(test_stocks),
                'successful_analyses': successful_analyses,
                'analysis_success_rate': analysis_rate
            }
        }
    
    async def _run_performance_benchmark(self) -> Dict[str, Any]:
        """运行性能基准测试"""
        print("执行性能基准测试...")
        
        # 测试批量处理性能
        start_time = time.time()
        
        # 生成多个测试数据
        test_count = 10
        successful_tests = 0
        
        for i in range(test_count):
            try:
                test_data = self.reverse_validation.pattern_generator.generate_pattern_data(
                    "MACD_GOLDEN_CROSS", 30, f"PERF_TEST_{i}"
                )
                
                if test_data is not None:
                    result = self.reverse_validation.run_single_pattern_validation(
                        "MACD", "MACD_GOLDEN_CROSS", test_data
                    )
                    if result['is_successful']:
                        successful_tests += 1
                        
            except Exception as e:
                logger.error(f"性能测试失败 {i}: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        performance_metrics = {
            'total_tests': test_count,
            'successful_tests': successful_tests,
            'duration_seconds': duration,
            'tests_per_second': test_count / duration if duration > 0 else 0,
            'average_test_time': duration / test_count if test_count > 0 else 0,
            'success_rate': successful_tests / test_count if test_count > 0 else 0
        }
        
        print(f"性能指标: {performance_metrics['tests_per_second']:.1f} 测试/秒")
        print(f"平均耗时: {performance_metrics['average_test_time']:.3f} 秒/测试")
        
        return performance_metrics
    
    def _calculate_overall_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算总体测试摘要"""
        summary = {
            'overall_success': True,
            'component_status': {},
            'performance_summary': {},
            'recommendations': []
        }
        
        # 检查各组件状态
        if 'system_validation' in results:
            summary['component_status']['system_validation'] = results['system_validation']['overall_status']
        
        if 'reverse_validation' in results:
            rv_success_rate = results['reverse_validation']['test_summary']['success_rate']
            summary['component_status']['reverse_validation'] = rv_success_rate > 0.7
        
        if 'pattern_recognition' in results:
            pr_success_rate = results['pattern_recognition']['summary']['recognition_rate']
            summary['component_status']['pattern_recognition'] = pr_success_rate > 0.7
        
        if 'buypoint_analysis' in results:
            ba_success_rate = results['buypoint_analysis']['summary']['analysis_success_rate']
            summary['component_status']['buypoint_analysis'] = ba_success_rate > 0.7
        
        # 计算总体成功状态
        summary['overall_success'] = all(summary['component_status'].values())
        
        # 性能摘要
        if 'performance_benchmark' in results:
            perf = results['performance_benchmark']
            summary['performance_summary'] = {
                'tests_per_second': perf['tests_per_second'],
                'average_test_time': perf['average_test_time'],
                'meets_performance_target': perf['tests_per_second'] > 1.0  # 目标: >1测试/秒
            }
        
        # 生成建议
        if not summary['overall_success']:
            summary['recommendations'].append("系统存在问题，需要进一步调试")
        
        if summary['performance_summary'].get('tests_per_second', 0) < 1.0:
            summary['recommendations'].append("性能需要优化，建议检查数据访问和计算效率")
        
        return summary


async def main():
    """主函数"""
    test_suite = UpdatedComprehensiveTestSuite()
    
    try:
        # 运行所有测试
        results = await test_suite.run_all_tests()
        
        # 保存结果
        output_dir = Path("tests/data/result")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"comprehensive_test_results_{timestamp}.json"
        
        import json
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 显示总结
        print("\n" + "=" * 100)
        print("综合测试完成总结:")
        print("=" * 100)
        
        summary = results.get('summary', {})
        print(f"✓ 总体状态: {'通过' if summary.get('overall_success', False) else '失败'}")
        
        for component, status in summary.get('component_status', {}).items():
            print(f"✓ {component}: {'通过' if status else '失败'}")
        
        perf = summary.get('performance_summary', {})
        if perf:
            print(f"✓ 性能指标: {perf.get('tests_per_second', 0):.1f} 测试/秒")
            print(f"✓ 平均耗时: {perf.get('average_test_time', 0):.3f} 秒/测试")
        
        if summary.get('recommendations'):
            print("\n建议:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        print(f"\n详细结果已保存到: {result_file}")
        print("=" * 100)
        
        return 0 if summary.get('overall_success', False) else 1
        
    except Exception as e:
        logger.error(f"测试套件执行失败: {e}")
        print(f"❌ 测试套件执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
