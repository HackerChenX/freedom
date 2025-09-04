#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标测试修复通用模板

基于MACD指标测试修复经验，提供标准化的指标测试修复流程模板
可用于任何技术指标的测试、验证和修复

使用方法:
python validation/indicator_test_template.py --indicator RSI --mode full
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

class IndicatorTestTemplate:
    """技术指标测试修复通用模板"""
    
    def __init__(self, indicator_name: str):
        """
        初始化测试模板
        
        Args:
            indicator_name: 指标名称 (如 'MACD', 'RSI', 'BOLL')
        """
        self.indicator_name = indicator_name
        self.stock_data_service = get_stock_data_service()
        
        # 标准测试配置
        self.test_config = {
            'test_stocks': ['000001', '000002', '000066', '000088'],  # 标准测试股票
            'test_date': '2025-05-12',  # 标准测试日期
            'min_history_days': 250,   # 最少历史数据
            'accuracy_threshold': 0.99,  # 准确率阈值
            'max_test_stocks': 50      # 最大测试股票数
        }
        
        # 质量标准
        self.quality_standards = {
            'accuracy_rate': 0.99,      # 准确率标准
            'data_quality_rate': 0.95,  # 数据质量率标准
            'calculation_stability': 1.0  # 计算稳定性标准
        }
        
        print(f"✅ {indicator_name}指标测试模板初始化完成")
    
    def load_benchmark_data(self, benchmark_file: Optional[str] = None) -> Dict:
        """
        加载基准测试数据
        
        Args:
            benchmark_file: 基准数据文件路径
            
        Returns:
            基准测试数据字典
        """
        
        if benchmark_file and Path(benchmark_file).exists():
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 默认基准数据模板
        return {
            '000001': {
                'date': self.test_config['test_date'],
                'benchmark': {
                    'primary_value': 0.0,  # 主要指标值
                    'secondary_value': 0.0,  # 次要指标值
                    'additional_values': {}  # 其他指标值
                },
                'data_source': 'manual_input',
                'verification_status': 'unverified'
            }
        }
    
    def get_indicator_class(self):
        """
        获取指标类
        
        Returns:
            指标类实例
        """
        
        # 动态导入指标类
        try:
            if self.indicator_name.upper() == 'MACD':
                from indicators.macd import MacdMacd
                return MacdMacd()
            elif self.indicator_name.upper() == 'RSI':
                from indicators.rsi import RsiRsi
                return RsiRsi()
            elif self.indicator_name.upper() == 'BOLL':
                from indicators.boll import BollBoll
                return BollBoll()
            # 添加更多指标类...
            else:
                raise ImportError(f"未找到{self.indicator_name}指标类")
        except ImportError as e:
            print(f"❌ 指标类导入失败: {e}")
            return None
    
    def validate_data_quality(self, df: pd.DataFrame, stock_code: str) -> Dict:
        """
        验证数据质量
        
        Args:
            df: 股票数据DataFrame
            stock_code: 股票代码
            
        Returns:
            数据质量报告
        """
        
        quality_report = {
            'stock_code': stock_code,
            'total_records': len(df),
            'date_range': f"{df['date'].min()} 到 {df['date'].max()}",
            'issues': []
        }
        
        # 检查空值
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            quality_report['issues'].append(f"发现{null_count}个空值")
        
        # 检查重复日期
        duplicate_dates = df['date'].duplicated().sum()
        if duplicate_dates > 0:
            quality_report['issues'].append(f"发现{duplicate_dates}个重复日期")
        
        # 检查价格合理性
        if 'close' in df.columns:
            abnormal_prices = (df['close'] <= 0).sum()
            if abnormal_prices > 0:
                quality_report['issues'].append(f"发现{abnormal_prices}个异常价格")
        
        # 检查数据充足性
        if len(df) < self.test_config['min_history_days']:
            quality_report['issues'].append(f"数据量不足：{len(df)} < {self.test_config['min_history_days']}")
        
        quality_report['quality_score'] = 1.0 - len(quality_report['issues']) * 0.1
        quality_report['status'] = 'PASS' if quality_report['quality_score'] >= self.quality_standards['data_quality_rate'] else 'FAIL'
        
        return quality_report
    
    def calculate_with_multiple_methods(self, df: pd.DataFrame, indicator_instance) -> Dict:
        """
        使用多种方法计算指标
        
        Args:
            df: 股票数据
            indicator_instance: 指标实例
            
        Returns:
            多方法计算结果
        """
        
        methods = ['standard', 'sma_init', 'pandas']  # 可根据指标调整
        results = {}
        
        for method in methods:
            try:
                # 尝试使用不同方法计算
                if hasattr(indicator_instance, '_calculate_with_method'):
                    result = indicator_instance._calculate_with_method(df, method=method)
                else:
                    # 默认计算方法
                    result = indicator_instance.calculate(df)
                
                results[method] = {
                    'success': True,
                    'result': result,
                    'method_info': f"{method}方法计算"
                }
                
            except Exception as e:
                results[method] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def validate_against_benchmark(self, calculated_data: Dict, benchmark_data: Dict, 
                                 stock_code: str, target_date: str) -> Dict:
        """
        与基准数据验证
        
        Args:
            calculated_data: 计算结果
            benchmark_data: 基准数据
            stock_code: 股票代码
            target_date: 目标日期
            
        Returns:
            验证结果
        """
        
        validation_result = {
            'stock_code': stock_code,
            'target_date': target_date,
            'methods_tested': [],
            'best_method': None,
            'best_accuracy': 0.0,
            'validation_details': {}
        }
        
        benchmark = benchmark_data.get(stock_code, {}).get('benchmark', {})
        
        if not benchmark:
            validation_result['status'] = 'NO_BENCHMARK'
            return validation_result
        
        best_accuracy = 0.0
        best_method = None
        
        for method, method_data in calculated_data.items():
            if not method_data['success']:
                continue
            
            validation_result['methods_tested'].append(method)
            
            # 提取目标日期的计算结果
            target_data = self.extract_target_date_data(
                method_data['result'], target_date
            )
            
            if target_data is None:
                continue
            
            # 计算准确率
            accuracy = self.calculate_accuracy(target_data, benchmark)
            
            validation_result['validation_details'][method] = {
                'accuracy': accuracy,
                'target_data': target_data,
                'benchmark': benchmark
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = method
        
        validation_result['best_method'] = best_method
        validation_result['best_accuracy'] = best_accuracy
        validation_result['status'] = 'PASS' if best_accuracy >= self.quality_standards['accuracy_rate'] else 'FAIL'
        
        return validation_result
    
    def extract_target_date_data(self, result_df: pd.DataFrame, target_date: str) -> Optional[Dict]:
        """
        提取目标日期的数据
        
        Args:
            result_df: 计算结果DataFrame
            target_date: 目标日期
            
        Returns:
            目标日期的数据字典
        """
        
        try:
            target_date_obj = pd.to_datetime(target_date).date()
            
            # 查找目标日期
            if 'date' in result_df.columns:
                target_rows = result_df[result_df['date'].dt.date == target_date_obj]
            else:
                # 如果没有date列，使用索引
                target_rows = result_df.iloc[-1:] if len(result_df) > 0 else pd.DataFrame()
            
            if target_rows.empty:
                return None
            
            target_row = target_rows.iloc[0]
            
            # 根据指标类型提取相关数据
            if self.indicator_name.upper() == 'MACD':
                return {
                    'DIFF': float(target_row.get('macd_line', 0)),
                    'DEA': float(target_row.get('macd_signal', 0)),
                    'MACD': float(target_row.get('macd_histogram', 0))
                }
            elif self.indicator_name.upper() == 'RSI':
                return {
                    'RSI': float(target_row.get('rsi', 0))
                }
            # 添加更多指标的数据提取逻辑...
            else:
                # 通用提取方法
                return {col: float(target_row[col]) for col in target_row.index if pd.notna(target_row[col])}
        
        except Exception as e:
            print(f"提取目标日期数据失败: {e}")
            return None
    
    def calculate_accuracy(self, calculated: Dict, benchmark: Dict) -> float:
        """
        计算准确率
        
        Args:
            calculated: 计算结果
            benchmark: 基准数据
            
        Returns:
            准确率 (0-1)
        """
        
        if not calculated or not benchmark:
            return 0.0
        
        total_error = 0.0
        total_values = 0
        
        for key in benchmark:
            if key in calculated:
                if benchmark[key] != 0:
                    error = abs(calculated[key] - benchmark[key]) / abs(benchmark[key])
                else:
                    error = abs(calculated[key] - benchmark[key])
                
                total_error += error
                total_values += 1
        
        if total_values == 0:
            return 0.0
        
        average_error = total_error / total_values
        accuracy = max(0.0, 1.0 - average_error)
        
        return accuracy
    
    def run_comprehensive_test(self, benchmark_data: Dict) -> Dict:
        """
        运行综合测试
        
        Args:
            benchmark_data: 基准测试数据
            
        Returns:
            综合测试结果
        """
        
        print(f"\n🎯 开始{self.indicator_name}指标综合测试")
        print("=" * 80)
        
        start_time = datetime.now()
        
        # 获取指标类
        indicator_instance = self.get_indicator_class()
        if indicator_instance is None:
            return {'error': '无法获取指标类'}
        
        test_results = {
            'indicator_name': self.indicator_name,
            'test_timestamp': start_time.isoformat(),
            'test_config': self.test_config,
            'quality_standards': self.quality_standards,
            'stock_results': {},
            'summary': {}
        }
        
        stocks_tested = 0
        stocks_passed = 0
        total_accuracy = 0.0
        
        # 测试每支股票
        for stock_code in self.test_config['test_stocks']:
            print(f"\n📊 测试{stock_code}股票:")
            
            try:
                # 获取股票数据
                df = self.stock_data_service.get_stock_data(
                    stock_code, 
                    days=self.test_config['min_history_days']
                )
                
                if df is None or len(df) == 0:
                    print(f"  ❌ 无法获取{stock_code}数据")
                    continue
                
                # 数据质量检查
                quality_report = self.validate_data_quality(df, stock_code)
                print(f"  📊 数据质量: {quality_report['quality_score']:.2%}")
                
                # 多方法计算
                calculation_results = self.calculate_with_multiple_methods(df, indicator_instance)
                print(f"  🔄 计算方法: {len([m for m, r in calculation_results.items() if r['success']])}种成功")
                
                # 基准验证
                validation_result = self.validate_against_benchmark(
                    calculation_results, 
                    benchmark_data, 
                    stock_code, 
                    self.test_config['test_date']
                )
                
                print(f"  🎯 最佳方法: {validation_result['best_method']}")
                print(f"  📈 准确率: {validation_result['best_accuracy']:.2%}")
                
                # 记录结果
                test_results['stock_results'][stock_code] = {
                    'quality_report': quality_report,
                    'calculation_results': calculation_results,
                    'validation_result': validation_result
                }
                
                stocks_tested += 1
                if validation_result['status'] == 'PASS':
                    stocks_passed += 1
                
                total_accuracy += validation_result['best_accuracy']
                
            except Exception as e:
                print(f"  ❌ 测试{stock_code}异常: {e}")
                test_results['stock_results'][stock_code] = {'error': str(e)}
        
        end_time = datetime.now()
        
        # 生成汇总
        test_results['summary'] = {
            'stocks_tested': stocks_tested,
            'stocks_passed': stocks_passed,
            'pass_rate': stocks_passed / stocks_tested if stocks_tested > 0 else 0,
            'average_accuracy': total_accuracy / stocks_tested if stocks_tested > 0 else 0,
            'processing_time': (end_time - start_time).total_seconds(),
            'overall_status': 'PASS' if stocks_passed >= len(self.test_config['test_stocks']) * 0.8 else 'FAIL'
        }
        
        return test_results
    
    def generate_test_report(self, test_results: Dict, output_file: Optional[str] = None):
        """
        生成测试报告
        
        Args:
            test_results: 测试结果
            output_file: 输出文件路径
        """
        
        if output_file is None:
            output_file = f"validation/reports/{self.indicator_name}_测试报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 确保输出目录存在
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        # 显示汇总报告
        summary = test_results['summary']
        
        print(f"\n📊 {self.indicator_name}指标测试报告")
        print("=" * 80)
        print(f"🕐 测试时间: {test_results['test_timestamp']}")
        print(f"📈 测试股票: {summary['stocks_tested']}支")
        print(f"✅ 通过股票: {summary['stocks_passed']}支")
        print(f"📊 通过率: {summary['pass_rate']:.1%}")
        print(f"🎯 平均准确率: {summary['average_accuracy']:.2%}")
        print(f"⏱️ 处理时间: {summary['processing_time']:.1f}秒")
        print(f"🏆 总体状态: {summary['overall_status']}")
        
        print(f"\n📄 详细报告已保存到: {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='技术指标测试修复通用模板')
    parser.add_argument('--indicator', required=True, help='指标名称 (如 MACD, RSI, BOLL)')
    parser.add_argument('--mode', default='quick', choices=['quick', 'full'], help='测试模式')
    parser.add_argument('--benchmark', help='基准数据文件路径')
    parser.add_argument('--output', help='输出报告文件路径')
    
    args = parser.parse_args()
    
    print(f"🎯 {args.indicator}指标测试修复模板")
    print(f"模式: {args.mode}")
    
    # 创建测试实例
    tester = IndicatorTestTemplate(args.indicator)
    
    # 加载基准数据
    benchmark_data = tester.load_benchmark_data(args.benchmark)
    
    # 运行测试
    if args.mode == 'full':
        results = tester.run_comprehensive_test(benchmark_data)
    else:
        # 快速测试模式
        tester.test_config['test_stocks'] = tester.test_config['test_stocks'][:2]
        results = tester.run_comprehensive_test(benchmark_data)
    
    # 生成报告
    tester.generate_test_report(results, args.output)

if __name__ == "__main__":
    main()
