#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
买点分析快速验证脚本

快速验证买点分析系统的核心功能是否正常工作
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer
from tests.buypoint_analysis.enhanced_test_data_generator import EnhancedTestDataGenerator
from tests.buypoint_analysis.mock_data_interface import setup_test_data, cleanup_test_data
from utils.logger import get_logger

logger = get_logger(__name__)


class QuickValidation:
    """买点分析快速验证"""
    
    def __init__(self):
        """初始化验证器"""
        self.buypoint_analyzer = BuyPointAnalyzer()
        self.data_generator = EnhancedTestDataGenerator()
        self.validation_results = {}
        
    def validate_core_patterns(self) -> Dict[str, Any]:
        """验证核心技术形态"""
        print("🔍 验证核心技术形态...")
        
        # 核心形态列表
        core_patterns = [
            'MACD_GOLDEN_CROSS',
            'RSI_OVERBOUGHT',
            'KDJ_GOLDEN_CROSS', 
            'BOLL_UPPER_BREAKOUT',
            'DOJI'
        ]
        
        results = {}
        
        for pattern in core_patterns:
            print(f"  测试 {pattern}...")
            
            try:
                # 生成测试数据
                test_data = self.data_generator.generate_pattern_data(
                    pattern_type=pattern,
                    data_points=60,
                    stock_code=f"QUICK_{pattern}"
                )
                
                if test_data is None:
                    results[pattern] = {
                        'success': False,
                        'error': '无法生成测试数据'
                    }
                    print(f"    ❌ 数据生成失败")
                    continue
                
                # 设置模拟数据
                stock_code = test_data['code'].iloc[0]
                mock_manager = setup_test_data({stock_code: test_data})

                try:
                    # 执行买点分析
                    start_time = time.time()

                    # 使用固定的买点日期，确保与数据生成器一致
                    buy_date = "20250801"  # 2025年8月1日

                    analysis_result = self.buypoint_analyzer.analyze_stock(
                        stock_code=stock_code,
                        buy_date=buy_date,
                        stock_name=test_data['name'].iloc[0]
                    )

                    execution_time = time.time() - start_time
                finally:
                    # 清理模拟数据
                    cleanup_test_data()
                
                # 评估结果
                if analysis_result:
                    pattern_found = self._check_pattern_in_result(pattern, analysis_result)
                    results[pattern] = {
                        'success': True,
                        'pattern_found': pattern_found,
                        'execution_time': execution_time,
                        'result_count': len(analysis_result) if isinstance(analysis_result, dict) else 1
                    }
                    
                    status = "✓" if pattern_found else "?"
                    print(f"    {status} 分析完成 ({execution_time:.3f}s)")
                else:
                    results[pattern] = {
                        'success': False,
                        'error': '分析返回空结果'
                    }
                    print(f"    ❌ 分析失败")
                    
            except Exception as e:
                results[pattern] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"    ❌ 异常: {e}")
        
        return results
    
    def validate_data_generation(self) -> Dict[str, Any]:
        """验证数据生成功能"""
        print("\n🔍 验证数据生成功能...")
        
        test_patterns = ['MACD_GOLDEN_CROSS', 'RSI_OVERSOLD', 'BOLL_SQUEEZE']
        results = {}
        
        for pattern in test_patterns:
            print(f"  生成 {pattern} 数据...")
            
            try:
                test_data = self.data_generator.generate_pattern_data(
                    pattern_type=pattern,
                    data_points=60,
                    stock_code=f"DATA_{pattern}"
                )
                
                if test_data is not None:
                    # 验证数据质量
                    quality_check = self._validate_data_quality(test_data)
                    results[pattern] = {
                        'success': True,
                        'data_points': len(test_data),
                        'quality_score': quality_check['score'],
                        'issues': quality_check['issues']
                    }
                    
                    print(f"    ✓ 生成成功 ({len(test_data)} 行, 质量: {quality_check['score']:.1%})")
                else:
                    results[pattern] = {
                        'success': False,
                        'error': '数据生成失败'
                    }
                    print(f"    ❌ 生成失败")
                    
            except Exception as e:
                results[pattern] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"    ❌ 异常: {e}")
        
        return results
    
    def validate_system_integration(self) -> Dict[str, Any]:
        """验证系统集成"""
        print("\n🔍 验证系统集成...")
        
        try:
            # 测试买点分析器初始化
            analyzer_init = self._test_analyzer_initialization()
            print(f"  买点分析器初始化: {'✓' if analyzer_init['success'] else '❌'}")
            
            # 测试数据生成器初始化
            generator_init = self._test_generator_initialization()
            print(f"  数据生成器初始化: {'✓' if generator_init['success'] else '❌'}")
            
            # 测试依赖注入
            dependency_test = self._test_dependency_injection()
            print(f"  依赖注入系统: {'✓' if dependency_test['success'] else '❌'}")
            
            # 测试日志系统
            logging_test = self._test_logging_system()
            print(f"  日志系统: {'✓' if logging_test['success'] else '❌'}")
            
            return {
                'analyzer_initialization': analyzer_init,
                'generator_initialization': generator_init,
                'dependency_injection': dependency_test,
                'logging_system': logging_test,
                'overall_success': all([
                    analyzer_init['success'],
                    generator_init['success'], 
                    dependency_test['success'],
                    logging_test['success']
                ])
            }
            
        except Exception as e:
            return {
                'overall_success': False,
                'error': str(e)
            }
    
    def _check_pattern_in_result(self, pattern: str, result: Any) -> bool:
        """检查结果中是否包含期望的形态"""
        if not result:
            return False
        
        # 转换为字符串进行模糊匹配
        result_str = str(result).lower()
        pattern_keywords = pattern.lower().split('_')
        
        # 检查是否包含形态关键词
        return any(keyword in result_str for keyword in pattern_keywords)
    
    def _validate_data_quality(self, data) -> Dict[str, Any]:
        """验证数据质量"""
        issues = []
        score = 1.0
        
        try:
            # 检查必需列
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'code', 'name']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                issues.append(f"缺少列: {missing_columns}")
                score -= 0.3
            
            # 检查数据完整性
            if data.isnull().any().any():
                issues.append("存在空值")
                score -= 0.2
            
            # 检查价格逻辑
            if not ((data['low'] <= data['open']) & (data['low'] <= data['close']) & 
                   (data['high'] >= data['open']) & (data['high'] >= data['close'])).all():
                issues.append("价格逻辑错误")
                score -= 0.3
            
            # 检查数据范围
            if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
                issues.append("存在非正数价格")
                score -= 0.2
            
            score = max(0.0, score)
            
        except Exception as e:
            issues.append(f"验证异常: {e}")
            score = 0.0
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _test_analyzer_initialization(self) -> Dict[str, Any]:
        """测试分析器初始化"""
        try:
            analyzer = BuyPointAnalyzer()
            return {
                'success': True,
                'analyzer_type': type(analyzer).__name__
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_generator_initialization(self) -> Dict[str, Any]:
        """测试生成器初始化"""
        try:
            generator = EnhancedTestDataGenerator()
            return {
                'success': True,
                'generator_type': type(generator).__name__,
                'pattern_count': len(generator.pattern_configs)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_dependency_injection(self) -> Dict[str, Any]:
        """测试依赖注入"""
        try:
            from utils.dependency_injection import get_service, get_container
            
            container = get_container()
            return {
                'success': True,
                'container_type': type(container).__name__
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_logging_system(self) -> Dict[str, Any]:
        """测试日志系统"""
        try:
            logger.info("测试日志消息")
            return {
                'success': True,
                'logger_name': logger.name
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_quick_validation(self) -> Dict[str, Any]:
        """运行快速验证"""
        print("=" * 80)
        print("买点分析系统 - 快速验证")
        print("=" * 80)
        
        start_time = time.time()
        
        # 运行各项验证
        core_patterns_result = self.validate_core_patterns()
        data_generation_result = self.validate_data_generation()
        system_integration_result = self.validate_system_integration()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算总体结果
        core_success_rate = sum(1 for r in core_patterns_result.values() if r.get('success', False)) / len(core_patterns_result)
        data_success_rate = sum(1 for r in data_generation_result.values() if r.get('success', False)) / len(data_generation_result)
        integration_success = system_integration_result.get('overall_success', False)
        
        overall_success = (core_success_rate >= 0.8 and data_success_rate >= 0.8 and integration_success)
        
        # 生成报告
        report = {
            'validation_time': total_time,
            'core_patterns': {
                'results': core_patterns_result,
                'success_rate': core_success_rate
            },
            'data_generation': {
                'results': data_generation_result,
                'success_rate': data_success_rate
            },
            'system_integration': system_integration_result,
            'overall_success': overall_success,
            'summary': {
                'total_tests': len(core_patterns_result) + len(data_generation_result) + 4,
                'successful_tests': int(core_success_rate * len(core_patterns_result)) + 
                                  int(data_success_rate * len(data_generation_result)) + 
                                  (4 if integration_success else 0)
            }
        }
        
        # 打印总结
        self._print_validation_summary(report)
        
        return report
    
    def _print_validation_summary(self, report: Dict[str, Any]):
        """打印验证总结"""
        print("\n" + "=" * 80)
        print("快速验证结果总结")
        print("=" * 80)
        
        print(f"验证时间: {report['validation_time']:.2f}秒")
        print(f"核心形态测试成功率: {report['core_patterns']['success_rate']:.1%}")
        print(f"数据生成测试成功率: {report['data_generation']['success_rate']:.1%}")
        print(f"系统集成测试: {'✓ 通过' if report['system_integration']['overall_success'] else '❌ 失败'}")
        
        overall_status = report['overall_success']
        print(f"\n总体状态: {'🎉 验证通过' if overall_status else '⚠️ 需要检查'}")
        
        if overall_status:
            print("✅ 买点分析系统核心功能正常，可以运行完整测试套件")
        else:
            print("❌ 买点分析系统存在问题，建议先修复后再运行完整测试")
        
        print("=" * 80)


def main():
    """主函数"""
    validator = QuickValidation()
    report = validator.run_quick_validation()
    
    # 返回适当的退出码
    return 0 if report['overall_success'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
