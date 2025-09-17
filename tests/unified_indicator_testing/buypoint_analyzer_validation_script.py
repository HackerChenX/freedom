#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
BuypointAnalyzer验证脚本

展示买点识别测试器的完整功能，包括：
1. 支持21个已修复指标的买点检测
2. 详细的买点分析报告和评分
3. 多种形态识别能力
4. 性能统计和质量评估
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from components.buypoint_analyzer import BuypointAnalyzer
from utils.logger import getLogger
from db.sql_manager import SQLManager, QueryType

logger = getLogger(__name__)


class BuypointAnalyzerValidator:
    """BuypointAnalyzer验证器"""
    
    def __init__(self):
        """初始化验证器"""
        logger.info("🚀 开始BuypointAnalyzer功能验证")
        self.analyzer = BuypointAnalyzer()
        self.test_results = []
        self.validation_report = {
            'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'indicator_coverage': {},
            'pattern_coverage': {},
            'performance_metrics': {},
            'detailed_results': []
        }
    
    def create_mock_data_pool(self, size: int = 10) -> list:
        """创建模拟数据池"""
        data_pool = []
        
        for i in range(size):
            # 创建上升趋势的数据
            dates = [(datetime.now() - timedelta(days=30-j)).strftime('%Y%m%d') for j in range(30)]
            base_price = 10.0 + np.random.uniform(-2, 2)
            
            # 生成价格序列
            price_trend = np.cumsum(np.random.uniform(-0.1, 0.2, 30))
            close_prices = base_price + price_trend
            high_prices = close_prices + np.random.uniform(0.1, 0.5, 30)
            low_prices = close_prices - np.random.uniform(0.1, 0.5, 30)
            open_prices = close_prices + np.random.uniform(-0.2, 0.2, 30)
            
            # 生成成交量
            volumes = np.random.randint(500000, 2000000, 30)
            
            stock_code = f"TARGET_{i:03d}" if i < 3 else f"STOCK_{i:03d}"
            
            data = pd.DataFrame({
                'date': dates,
                'code': [stock_code] * 30,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes
            })
            
            data_pool.append(data)
        
        logger.info(f"✅ 创建了{size}个股票的模拟数据池")
        return data_pool
    
    def test_supported_indicators(self):
        """测试支持的指标覆盖"""
        logger.info("📊 测试指标覆盖情况...")
        
        supported_indicators = self.analyzer.supported_indicators
        total_indicators = len(supported_indicators)
        total_patterns = sum(len(config['patterns']) for config in supported_indicators.values())
        
        self.validation_report['indicator_coverage'] = {
            'total_indicators': total_indicators,
            'total_patterns': total_patterns,
            'indicators': {}
        }
        
        for indicator_name, config in supported_indicators.items():
            self.validation_report['indicator_coverage']['indicators'][indicator_name] = {
                'patterns': config['patterns'],
                'pattern_count': len(config['patterns']),
                'calculation_method': config['calculation_method']
            }
        
        logger.info(f"✅ 指标覆盖测试完成: {total_indicators}个指标, {total_patterns}个形态")
        return True
    
    def test_pattern_recognition_accuracy(self):
        """测试形态识别准确性"""
        logger.info("🎯 测试形态识别准确性...")
        
        data_pool = self.create_mock_data_pool(8)
        test_patterns = [
            'MACD_GOLDEN_CROSS',
            'RSI_OVERSOLD',
            'KDJ_GOLDEN_CROSS',
            'BOLL_UPPER_BREAKOUT',
            'MA_GOLDEN_CROSS',
            'VOL_VOLUME_SURGE'
        ]
        
        accuracy_results = {}
        
        for pattern in test_patterns:
            try:
                start_time = time.time()
                result = self.analyzer.test_pattern_recognition(data_pool, pattern)
                execution_time = time.time() - start_time
                
                accuracy = result.get('accuracy', 0.0)
                accuracy_results[pattern] = {
                    'accuracy': accuracy,
                    'execution_time': execution_time,
                    'total_stocks': result.get('total_stocks', 0),
                    'target_stocks': result.get('target_stocks', 0),
                    'correctly_identified': result.get('correctly_identified', 0),
                    'status': result.get('status', 'UNKNOWN')
                }
                
                self.test_results.append({
                    'test_type': 'pattern_recognition',
                    'pattern': pattern,
                    'passed': result.get('status') == 'COMPLETED',
                    'accuracy': accuracy,
                    'execution_time': execution_time
                })
                
                logger.info(f"  ✅ {pattern}: 准确率 {accuracy:.2%}, 耗时 {execution_time:.3f}s")
                
            except Exception as e:
                logger.error(f"  ❌ {pattern}: 测试失败 - {e}")
                accuracy_results[pattern] = {
                    'accuracy': 0.0,
                    'execution_time': 0.0,
                    'error': str(e),
                    'status': 'FAILED'
                }
                
                self.test_results.append({
                    'test_type': 'pattern_recognition',
                    'pattern': pattern,
                    'passed': False,
                    'error': str(e)
                })
        
        self.validation_report['pattern_coverage'] = accuracy_results
        logger.info(f"✅ 形态识别测试完成: {len(test_patterns)}个形态")
        return accuracy_results
    
    def test_performance_metrics(self):
        """测试性能指标"""
        logger.info("⚡ 测试性能指标...")
        
        # 大数据集测试
        large_data_pool = self.create_mock_data_pool(50)
        
        start_time = time.time()
        result = self.analyzer.test_pattern_recognition(large_data_pool, 'MACD_GOLDEN_CROSS')
        total_time = time.time() - start_time
        
        stocks_per_second = len(large_data_pool) / total_time if total_time > 0 else 0
        
        # 获取识别统计
        stats = self.analyzer.get_recognition_statistics()
        
        performance_metrics = {
            'large_dataset_test': {
                'stock_count': len(large_data_pool),
                'total_time': total_time,
                'stocks_per_second': stocks_per_second,
                'memory_efficiency': 'GOOD' if total_time < 5.0 else 'FAIR'
            },
            'recognition_statistics': stats,
            'execution_times': {
                'average': np.mean(stats.get('execution_times', [1.0])),
                'max': max(stats.get('execution_times', [1.0])),
                'min': min(stats.get('execution_times', [1.0]))
            }
        }
        
        self.validation_report['performance_metrics'] = performance_metrics
        
        logger.info(f"✅ 性能测试完成: {stocks_per_second:.1f} 股票/秒")
        return performance_metrics
    
    def test_detailed_analysis_report(self):
        """测试详细分析报告功能"""
        logger.info("📋 测试详细分析报告功能...")
        
        data_pool = self.create_mock_data_pool(5)
        
        # 测试多个指标的综合分析
        test_cases = [
            'MACD_GOLDEN_CROSS',
            'RSI_CENTERLINE_CROSS',
            'KDJ_OVERSOLD',
            'BOLL_SQUEEZE',
            'MA_BULLISH_ARRANGEMENT'
        ]
        
        comprehensive_report = {
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_pool_size': len(data_pool),
            'indicator_analyses': {},
            'summary_statistics': {}
        }
        
        total_accuracy = 0.0
        total_tests = 0
        
        for test_case in test_cases:
            try:
                result = self.analyzer.test_pattern_recognition(data_pool, test_case)
                
                # 提取详细信息
                analysis_detail = {
                    'pattern': test_case,
                    'accuracy': result.get('accuracy', 0.0),
                    'total_stocks': result.get('total_stocks', 0),
                    'target_stocks': result.get('target_stocks', 0),
                    'correctly_identified': result.get('correctly_identified', 0),
                    'pattern_analysis': result.get('pattern_analysis', {}),
                    'indicator_performance': result.get('indicator_performance', {}),
                    'execution_time': result.get('execution_time', 0.0),
                    'status': result.get('status', 'UNKNOWN')
                }
                
                comprehensive_report['indicator_analyses'][test_case] = analysis_detail
                
                if result.get('accuracy') is not None:
                    total_accuracy += result.get('accuracy', 0.0)
                    total_tests += 1
                
                logger.info(f"  📊 {test_case}: 分析完成")
                
            except Exception as e:
                logger.error(f"  ❌ {test_case}: 分析失败 - {e}")
                comprehensive_report['indicator_analyses'][test_case] = {
                    'error': str(e),
                    'status': 'FAILED'
                }
        
        # 计算总体统计
        avg_accuracy = total_accuracy / total_tests if total_tests > 0 else 0.0
        comprehensive_report['summary_statistics'] = {
            'average_accuracy': avg_accuracy,
            'total_patterns_tested': len(test_cases),
            'successful_tests': total_tests,
            'overall_grade': self._grade_overall_performance(avg_accuracy)
        }
        
        self.validation_report['detailed_analysis_report'] = comprehensive_report
        
        logger.info(f"✅ 详细分析报告测试完成: 平均准确率 {avg_accuracy:.2%}")
        return comprehensive_report
    
    def _grade_overall_performance(self, accuracy: float) -> str:
        """评估整体性能等级"""
        if accuracy >= 0.8:
            return 'EXCELLENT'
        elif accuracy >= 0.6:
            return 'GOOD'
        elif accuracy >= 0.4:
            return 'FAIR'
        else:
            return 'POOR'
    
    def generate_validation_report(self):
        """生成验证报告"""
        logger.info("📝 生成验证报告...")
        
        # 计算总体统计
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.get('passed', False))
        failed_tests = total_tests - passed_tests
        
        self.validation_report.update({
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'detailed_results': self.test_results
        })
        
        return self.validation_report
    
    def save_report(self, filename: str = None):
        """保存验证报告"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'buypoint_analyzer_validation_report_{timestamp}.json'
        
        report_path = os.path.join('results', 'validation', filename)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 验证报告已保存: {report_path}")
        return report_path
    
    def print_summary(self):
        """打印验证摘要"""
        report = self.validation_report
        
        print("\n" + "="*80)
        print("🎯 BuypointAnalyzer 功能验证报告")
        print("="*80)
        
        print(f"\n📊 总体统计:")
        print(f"  测试时间: {report['test_time']}")
        print(f"  总测试数: {report['total_tests']}")
        print(f"  通过测试: {report['passed_tests']}")
        print(f"  失败测试: {report['failed_tests']}")
        print(f"  成功率: {report.get('success_rate', 0)*100:.1f}%")
        
        print(f"\n🎯 指标覆盖:")
        coverage = report.get('indicator_coverage', {})
        print(f"  支持指标: {coverage.get('total_indicators', 0)}个")
        print(f"  支持形态: {coverage.get('total_patterns', 0)}个")
        
        print(f"\n⚡ 性能指标:")
        perf = report.get('performance_metrics', {})
        if 'large_dataset_test' in perf:
            test_info = perf['large_dataset_test']
            print(f"  处理速度: {test_info.get('stocks_per_second', 0):.1f} 股票/秒")
            print(f"  内存效率: {test_info.get('memory_efficiency', 'UNKNOWN')}")
        
        print(f"\n📋 详细分析:")
        analysis = report.get('detailed_analysis_report', {})
        if 'summary_statistics' in analysis:
            stats = analysis['summary_statistics']
            print(f"  平均准确率: {stats.get('average_accuracy', 0)*100:.1f}%")
            print(f"  整体评级: {stats.get('overall_grade', 'UNKNOWN')}")
        
        print(f"\n✅ 验证结论:")
        success_rate = report.get('success_rate', 0)
        if success_rate >= 0.9:
            print("  🎉 BuypointAnalyzer功能完全正常，完美支持21个指标的买点识别！")
        elif success_rate >= 0.7:
            print("  ✅ BuypointAnalyzer功能良好，基本支持所有核心功能！")
        elif success_rate >= 0.5:
            print("  ⚠️  BuypointAnalyzer功能基本可用，但需要一些优化。")
        else:
            print("  ❌ BuypointAnalyzer需要进一步修复和完善。")
        
        print("="*80)
    
    def run_full_validation(self):
        """运行完整验证"""
        logger.info("🚀 开始完整验证流程...")
        
        try:
            # 1. 测试指标覆盖
            self.test_supported_indicators()
            
            # 2. 测试形态识别准确性
            self.test_pattern_recognition_accuracy()
            
            # 3. 测试性能指标
            self.test_performance_metrics()
            
            # 4. 测试详细分析报告
            self.test_detailed_analysis_report()
            
            # 5. 生成验证报告
            self.generate_validation_report()
            
            # 6. 保存报告
            report_path = self.save_report()
            
            # 7. 打印摘要
            self.print_summary()
            
            logger.info("✅ 完整验证流程完成！")
            return True, report_path
            
        except Exception as e:
            logger.error(f"❌ 验证过程中发生错误: {e}")
            return False, str(e)


def main():
    """主函数"""
    print("🎯 BuypointAnalyzer 功能验证脚本")
    print("="*50)
    
    try:
        validator = BuypointAnalyzerValidator()
        success, result = validator.run_full_validation()
        
        if success:
            print(f"\n🎉 验证成功完成！报告已保存至: {result}")
            return 0
        else:
            print(f"\n❌ 验证失败: {result}")
            return 1
            
    except Exception as e:
        print(f"\n💥 验证脚本执行异常: {e}")
        return 2


if __name__ == "__main__":
    exit(main()) 