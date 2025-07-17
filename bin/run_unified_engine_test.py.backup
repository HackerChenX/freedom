#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
统一分析引擎测试执行脚本

支持配置化测试、早停机制、分组测试等功能
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from scripts.comprehensive_unified_engine_test import Unified_engine_comprehensive_test
from scripts.indicator_logic_validator import Indicator_logic_validator
from utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedEngineTestRunner:
    """统一分析引擎测试运行器"""
    
    def __init___15(self, config_file: str = None):
        """
        初始化测试运行器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file or os.path.join(root_dir, 'config', 'unified_engine_test_config.json')
        self.config = self._load_config()
        self.test_results = {}
        
        logger.info(f"测试运行器初始化完成，配置文件: {self.config_file}")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("配置文件加载成功")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            # 返回默认配置
            return {
                "test_configuration": {
                    "test_stock_count": 100,
                    "enable_early_stop": True,
                    "max_concurrent_tests": 3,
                    "debug_mode": True
                }
            }
    
    def run_comprehensive_test(self, test_groups: List[str] = None) -> Dict[str, Any]:
        """运行全面测试"""
        logger.info("🚀 开始运行统一分析引擎全面测试")
        
        test_config = self.config.get('test_configuration', {})
        
        # 创建测试实例
        tester = Unified_engine_comprehensive_test(
            test_stock_count=test_config.get('test_stock_count', 100),
            enable_early_stop=test_config.get('enable_early_stop', True),
            max_concurrent_tests=test_config.get('max_concurrent_tests', 3)
        )
        
        try:
            # 如果指定了测试组，只测试特定组的指标
            if test_groups:
                report = self._run_grouped_test(tester, test_groups)
            else:
                # 运行全面测试
                report = tester.run_comprehensive_test()
            
            # 显示测试摘要
            tester.print_test_summary(report)
            
            # 保存测试报告
            self._save_test_report(report, 'comprehensive_test')
            
            self.test_results['comprehensive_test'] = report
            return report
            
        except Exception as e:
            logger.error(f"全面测试失败: {e}")
            raise
    
    def run_logic_validation(self, sample_stocks: int = None) -> Dict[str, Any]:
        """运行逻辑验证测试"""
        logger.info("🔍 开始运行指标逻辑验证测试")
        
        test_config = self.config.get('test_configuration', {})
        sample_stocks = sample_stocks or test_config.get('sample_stocks_per_indicator', 2)
        
        # 创建验证器
        validator = Indicator_logic_validator(
            debug_mode=test_config.get('debug_mode', True)
        )
        
        try:
            # 运行验证
            report = validator.validate_all_indicators(sample_stocks=sample_stocks)
            
            # 显示验证摘要
            validator.print_validation_summary(report)
            
            # 保存验证报告
            self._save_test_report(report, 'logic_validation')
            
            self.test_results['logic_validation'] = report
            return report
            
        except Exception as e:
            logger.error(f"逻辑验证失败: {e}")
            raise
    
    def _run_grouped_test(self, tester: Unified_engine_comprehensive_test, 
                         test_groups: List[str]) -> Dict[str, Any]:
        """运行分组测试"""
        logger.info(f"运行分组测试: {test_groups}")
        
        # 获取指标分组配置
        indicator_groups = self.config.get('indicator_groups', {})
        
        # 收集要测试的指标
        indicators_to_test = []
        for group in test_groups:
            if group in indicator_groups:
                indicators_to_test.extend(indicator_groups[group])
            else:
                logger.warning(f"未找到指标组: {group}")
        
        if not indicators_to_test:
            raise ValueError(f"没有找到要测试的指标，检查分组配置: {test_groups}")
        
        logger.info(f"将测试 {len(indicators_to_test)} 个指标")
        
        # 临时修改测试器的指标列表
        original_method = tester._generate_indicator_test_strategies
        
        def filtered_strategies():
            all_strategies = original_method()
            # 过滤出指定的指标策略
            filtered = []
            for strategy in all_strategies:
                strategy_name = strategy.get('strategy_id', '').replace('test_', '').upper()
                if strategy_name in indicators_to_test:
                    filtered.append(strategy)
            return filtered
        
        tester._generate_indicator_test_strategies = filtered_strategies
        
        try:
            return tester.run_comprehensive_test()
        finally:
            # 恢复原方法
            tester._generate_indicator_test_strategies = original_method
    
    def run_priority_test(self) -> Dict[str, Any]:
        """运行优先级测试（测试最重要的指标）"""
        logger.info("⭐ 运行优先级指标测试")
        return self.run_comprehensive_test(['priority_indicators'])
    
    def run_zxm_test(self) -> Dict[str, Any]:
        """运行ZXM指标测试"""
        logger.info("🎯 运行ZXM指标测试")
        return self.run_comprehensive_test(['zxm_indicators'])
    
    def run_enhanced_test(self) -> Dict[str, Any]:
        """运行增强指标测试"""
        logger.info("🔧 运行增强指标测试")
        return self.run_comprehensive_test(['enhanced_indicators'])
    
    def run_quick_test(self) -> Dict[str, Any]:
        """运行快速测试（少量指标和股票）"""
        logger.info("⚡ 运行快速测试")
        
        # 临时修改配置为快速测试模式
        original_config = self.config['test_configuration'].copy()
        self.config['test_configuration'].update({
            'test_stock_count': 20,
            'max_concurrent_tests': 2,
            'sample_stocks_per_indicator': 1
        })
        
        try:
            return self.run_comprehensive_test(['priority_indicators'])
        finally:
            # 恢复原配置
            self.config['test_configuration'] = original_config
    
    def _save_test_report(self, report: Dict[str, Any], test_type: str):
        """保存测试报告"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{test_type}_report_{timestamp}.json"
            filepath = os.path.join(root_dir, 'data', 'result', filename)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"📄 测试报告已保存: {filepath}")
            
            # 同时保存文本格式的摘要
            txt_filepath = filepath.replace('.json', '_summary.txt')
            self._save_text_summary(report, txt_filepath, test_type)
            
        except Exception as e:
            logger.error(f"保存测试报告失败: {e}")
    
    def _save_text_summary(self, report: Dict[str, Any], filepath: str, test_type: str):
        """保存文本格式的测试摘要"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"统一分析引擎{test_type}测试报告\n")
                f.write("=" * 80 + "\n")
                f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if test_type == 'comprehensive_test':
                    self._write_comprehensive_summary(f, report)
                elif test_type == 'logic_validation':
                    self._write_validation_summary(f, report)
                
            logger.info(f"📄 文本摘要已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存文本摘要失败: {e}")
    
    def _write_comprehensive_summary(self, f, report: Dict[str, Any]):
        """写入全面测试摘要"""
        summary = report.get('test_summary', {})
        
        f.write("测试概览:\n")
        f.write(f"  总测试数: {summary.get('total_tests', 0)}\n")
        f.write(f"  成功测试: {summary.get('successful_tests', 0)}\n")
        f.write(f"  失败测试: {summary.get('failed_tests', 0)}\n")
        f.write(f"  成功率: {summary.get('success_rate', 0)}%\n")
        f.write(f"  总耗时: {summary.get('total_duration', 0)} 秒\n")
        f.write(f"  测试股票数: {summary.get('test_stock_count', 0)}\n\n")
        
        # 分类分析
        if 'category_analysis' in report:
            f.write("分类分析:\n")
            for category, stats in report['category_analysis'].items():
                f.write(f"  {category}: {stats['successful']}/{stats['total']} "
                       f"({stats['success_rate']}%) 平均选股: {stats['avg_selected_count']}\n")
            f.write("\n")
        
        # 失败测试
        if 'failed_tests' in report and report['failed_tests']:
            f.write("失败测试:\n")
            for error in report['failed_tests'][:10]:
                f.write(f"  {error.get('indicator', 'Unknown')}: {error.get('error', 'Unknown error')}\n")
            f.write("\n")
    
    def _write_validation_summary(self, f, report: Dict[str, Any]):
        """写入验证测试摘要"""
        summary = report.get('validation_summary', {})
        
        f.write("验证概览:\n")
        f.write(f"  总指标数: {summary.get('total_indicators', 0)}\n")
        f.write(f"  验证通过: {summary.get('valid_indicators', 0)}\n")
        f.write(f"  验证失败: {summary.get('invalid_indicators', 0)}\n")
        f.write(f"  成功率: {summary.get('success_rate', 0)}%\n")
        f.write(f"  计算错误: {summary.get('calculation_errors', 0)}\n")
        f.write(f"  逻辑错误: {summary.get('logic_errors', 0)}\n")
        f.write(f"  数据错误: {summary.get('data_errors', 0)}\n\n")
        
        # 失败分析
        if 'failure_analysis' in report:
            failure = report['failure_analysis']
            f.write("失败分析:\n")
            f.write(f"  计算失败: {len(failure['failure_categories']['calculation_failures'])} 个\n")
            f.write(f"  逻辑失败: {len(failure['failure_categories']['logic_failures'])} 个\n")
            f.write(f"  条件失败: {len(failure['failure_categories']['condition_failures'])} 个\n\n")
            
            if failure['most_common_errors']:
                f.write("最常见错误:\n")
                for error, count in failure['most_common_errors'][:5]:
                    f.write(f"  {error} (出现 {count} 次)\n")
                f.write("\n")
        
        # 修复建议
        if 'fix_suggestions' in report:
            f.write("修复建议:\n")
            for suggestion in report['fix_suggestions'][:5]:
                f.write(f"  {suggestion['category']}: {suggestion['suggestion']}\n")
            f.write("\n")
    
    def generate_final_report(self) -> Dict[str, Any]:
        """生成最终测试报告"""
        logger.info("📊 生成最终测试报告")
        
        final_report = {
            'test_overview': {
                'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config_file': self.config_file,
                'tests_executed': list(self.test_results.keys())
            },
            'test_results': self.test_results,
            'overall_assessment': self._assess_overall_results()
        }
        
        # 保存最终报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(root_dir, 'data', 'result', f'final_unified_engine_report_{timestamp}.json')
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📄 最终报告已保存: {filepath}")
        
        return final_report
    
    def _assess_overall_results(self) -> Dict[str, Any]:
        """评估总体测试结果"""
        assessment = {
            'overall_status': 'UNKNOWN',
            'total_indicators_tested': 0,
            'total_successful': 0,
            'total_failed': 0,
            'overall_success_rate': 0.0,
            'recommendations': []
        }
        
        # 汇总所有测试结果
        for test_type, result in self.test_results.items():
            if test_type == 'comprehensive_test':
                summary = result.get('test_summary', {})
                assessment['total_indicators_tested'] += summary.get('total_tests', 0)
                assessment['total_successful'] += summary.get('successful_tests', 0)
                assessment['total_failed'] += summary.get('failed_tests', 0)
            
            elif test_type == 'logic_validation':
                summary = result.get('validation_summary', {})
                # 逻辑验证结果也计入总体评估
                pass
        
        # 计算总体成功率
        if assessment['total_indicators_tested'] > 0:
            assessment['overall_success_rate'] = (
                assessment['total_successful'] / assessment['total_indicators_tested'] * 100
            )
        
        # 确定总体状态
        success_rate = assessment['overall_success_rate']
        if success_rate >= 90:
            assessment['overall_status'] = 'EXCELLENT'
            assessment['recommendations'].append('系统运行状态优秀，可以投入生产使用')
        elif success_rate >= 80:
            assessment['overall_status'] = 'GOOD'
            assessment['recommendations'].append('系统运行状态良好，建议修复少量问题后投入使用')
        elif success_rate >= 60:
            assessment['overall_status'] = 'FAIR'
            assessment['recommendations'].append('系统存在一些问题，建议修复后再投入使用')
        else:
            assessment['overall_status'] = 'POOR'
            assessment['recommendations'].append('系统存在严重问题，需要全面检查和修复')
        
        return assessment


def main_29():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一分析引擎测试运行器')
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--test-type', '-t', 
                       choices=['comprehensive', 'logic', 'priority', 'zxm', 'enhanced', 'quick', 'all'],
                       default='all', help='测试类型')
    parser.add_argument('--groups', '-g', nargs='+', help='指定测试组')
    parser.add_argument('--sample-stocks', '-s', type=int, help='每个指标的样本股票数')
    
    args = parser.parse_args()
    
    print("🚀 启动统一分析引擎测试运行器")
    
    try:
        # 创建测试运行器
        runner = Unified_engine_test_runner(config_file=args.config)
        
        # 根据参数运行不同类型的测试
        if args.test_type == 'comprehensive':
            runner.run_comprehensive_test(args.groups)
        elif args.test_type == 'logic':
            runner.run_logic_validation(args.sample_stocks)
        elif args.test_type == 'priority':
            runner.run_priority_test()
        elif args.test_type == 'zxm':
            runner.run_zxm_test()
        elif args.test_type == 'enhanced':
            runner.run_enhanced_test()
        elif args.test_type == 'quick':
            runner.run_quick_test()
        elif args.test_type == 'all':
            # 运行所有测试
            print("📋 运行全套测试...")
            runner.run_quick_test()  # 先快速测试
            runner.run_priority_test()  # 然后优先级测试
            runner.run_logic_validation(2)  # 逻辑验证
            # runner.run_comprehensive_test()  # 最后全面测试（可选）
        
        # 生成最终报告
        final_report = runner.generate_final_report()
        
        # 显示总体评估
        assessment = final_report['overall_assessment']
        print(f"\n🎯 总体评估: {assessment['overall_status']}")
        print(f"📊 成功率: {assessment['overall_success_rate']:.1f}%")
        print(f"✅ 成功: {assessment['total_successful']}")
        print(f"❌ 失败: {assessment['total_failed']}")
        
        for recommendation in assessment['recommendations']:
            print(f"💡 建议: {recommendation}")
        
        # 返回状态码
        if assessment['overall_success_rate'] >= 80:
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"❌ 测试运行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    exit(main_29())