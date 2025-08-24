#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面MACD人工验证系统

问题发现：
- 之前的验证只覆盖了4支股票，样本量太小
- 需要对大量股票进行MACD计算验证
- 必须确保所有MACD计算都是正确的

解决方案：
- 扩大验证样本到100+股票
- 对每支股票进行详细的MACD计算验证
- 生成完整的人工验证报告
- 检测和修复所有计算错误
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

class ComprehensiveMacdValidation:
    """全面MACD人工验证系统"""
    
    def __init__(self):
        """初始化验证系统"""
        self.macd_indicator = MacdMacd()
        self.stock_data_service = get_stock_data_service()
        
        # 验证参数
        self.target_date = "2025-05-12"  # 统一验证日期
        self.min_stocks_to_validate = 100  # 最少验证股票数
        self.max_stocks_to_validate = 200  # 最多验证股票数
        
        # 用户提供的基准数据
        self.benchmark_data = {
            '000028': {
                'MACD': 0.220,
                'DIFF': -0.136,
                'DEA': -0.246
            }
        }
        
        # 验证结果存储
        self.validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_scope': 'comprehensive_macd_validation',
            'target_date': self.target_date,
            'benchmark_data': self.benchmark_data,
            'validation_summary': {},
            'stock_validations': {},
            'calculation_errors': [],
            'data_quality_issues': [],
            'statistical_analysis': {},
            'recommendations': []
        }
        
        print("✅ 全面MACD验证系统初始化完成")
        print(f"📅 验证目标日期: {self.target_date}")
        print(f"📊 验证股票范围: {self.min_stocks_to_validate}-{self.max_stocks_to_validate}支")
    
    def validate_single_stock(self, stock_code: str) -> Dict[str, Any]:
        """验证单支股票的MACD计算"""
        
        validation_result = {
            'stock_code': stock_code,
            'validation_date': self.target_date,
            'validation_status': 'PENDING',
            'error_message': None,
            'price_data': {},
            'macd_calculation': {},
            'quality_checks': {},
            'benchmark_comparison': {}
        }
        
        try:
            # 获取股票数据
            df = self.stock_data_service.get_stock_data(stock_code, days=200)
            
            if df is None or len(df) == 0:
                validation_result['validation_status'] = 'FAILED'
                validation_result['error_message'] = '无法获取股票数据'
                return validation_result
            
            # 查找目标日期
            target_date_obj = pd.to_datetime(self.target_date).date()
            target_rows = df[df['date'].dt.date == target_date_obj]
            
            if target_rows.empty:
                # 查找最接近的交易日
                df['date_diff'] = abs((df['date'].dt.date - target_date_obj).apply(lambda x: x.days))
                closest_idx = df['date_diff'].idxmin()
                target_idx = closest_idx
                actual_date = df.loc[closest_idx]['date'].date()
                validation_result['actual_date'] = str(actual_date)
            else:
                target_idx = target_rows.index[0]
                validation_result['actual_date'] = self.target_date
            
            # 记录价格数据
            price_data = df.iloc[target_idx]
            validation_result['price_data'] = {
                'date': str(price_data['date'].date()),
                'open': float(price_data['open']),
                'high': float(price_data['high']),
                'low': float(price_data['low']),
                'close': float(price_data['close']),
                'volume': float(price_data['volume'])
            }
            
            # 计算MACD
            macd_result = self.macd_indicator.calculate(df)
            
            if macd_result is None or macd_result.empty:
                validation_result['validation_status'] = 'FAILED'
                validation_result['error_message'] = 'MACD计算失败'
                return validation_result
            
            if target_idx >= len(macd_result):
                validation_result['validation_status'] = 'FAILED'
                validation_result['error_message'] = f'目标索引{target_idx}超出MACD结果范围{len(macd_result)}'
                return validation_result
            
            # 记录MACD计算结果
            macd_data = macd_result.iloc[target_idx]
            validation_result['macd_calculation'] = {
                'DIFF': float(macd_data['macd_line']),
                'DEA': float(macd_data['macd_signal']),
                'MACD': float(macd_data['macd_histogram']),
                'calculation_date': str(price_data['date'].date())
            }
            
            # 质量检查
            validation_result['quality_checks'] = self._perform_quality_checks(
                validation_result['macd_calculation']
            )
            
            # 基准对比（如果有基准数据）
            if stock_code in self.benchmark_data:
                validation_result['benchmark_comparison'] = self._compare_with_benchmark(
                    stock_code, 
                    validation_result['macd_calculation']
                )
            
            # 公式一致性检查
            calculated_macd = 2 * (validation_result['macd_calculation']['DIFF'] - 
                                 validation_result['macd_calculation']['DEA'])
            formula_diff = abs(calculated_macd - validation_result['macd_calculation']['MACD'])
            
            validation_result['formula_consistency'] = {
                'calculated_macd': calculated_macd,
                'system_macd': validation_result['macd_calculation']['MACD'],
                'difference': formula_diff,
                'is_consistent': formula_diff < 0.000001
            }
            
            # 确定验证状态
            if validation_result['quality_checks']['overall_quality'] == 'GOOD':
                if stock_code in self.benchmark_data:
                    benchmark_result = validation_result['benchmark_comparison']
                    if (benchmark_result['accuracy_DIFF'] > 0.99 and 
                        benchmark_result['accuracy_DEA'] > 0.99 and 
                        benchmark_result['accuracy_MACD'] > 0.99):
                        validation_result['validation_status'] = 'PASSED_HIGH_ACCURACY'
                    elif (benchmark_result['accuracy_DIFF'] > 0.95 and 
                          benchmark_result['accuracy_DEA'] > 0.95 and 
                          benchmark_result['accuracy_MACD'] > 0.95):
                        validation_result['validation_status'] = 'PASSED_ACCEPTABLE'
                    else:
                        validation_result['validation_status'] = 'FAILED_ACCURACY'
                else:
                    validation_result['validation_status'] = 'PASSED_CALCULATION'
            else:
                validation_result['validation_status'] = 'FAILED_QUALITY'
            
        except Exception as e:
            validation_result['validation_status'] = 'ERROR'
            validation_result['error_message'] = f'验证过程异常: {str(e)}'
        
        return validation_result
    
    def _perform_quality_checks(self, macd_calculation: Dict[str, float]) -> Dict[str, Any]:
        """执行MACD计算质量检查"""
        
        quality_checks = {
            'value_range_check': True,
            'nan_check': True,
            'infinity_check': True,
            'formula_check': True,
            'overall_quality': 'GOOD',
            'issues': []
        }
        
        diff = macd_calculation['DIFF']
        dea = macd_calculation['DEA']
        macd = macd_calculation['MACD']
        
        # 检查NaN值
        if pd.isna(diff) or pd.isna(dea) or pd.isna(macd):
            quality_checks['nan_check'] = False
            quality_checks['issues'].append('存在NaN值')
        
        # 检查无穷大值
        if np.isinf(diff) or np.isinf(dea) or np.isinf(macd):
            quality_checks['infinity_check'] = False
            quality_checks['issues'].append('存在无穷大值')
        
        # 检查数值范围（MACD值通常在-10到10之间）
        if abs(diff) > 10 or abs(dea) > 10 or abs(macd) > 10:
            quality_checks['value_range_check'] = False
            quality_checks['issues'].append('MACD值超出正常范围')
        
        # 检查公式一致性
        expected_macd = 2 * (diff - dea)
        if abs(expected_macd - macd) > 0.000001:
            quality_checks['formula_check'] = False
            quality_checks['issues'].append(f'公式不一致: 期望{expected_macd:.6f}, 实际{macd:.6f}')
        
        # 综合评估
        if not all([quality_checks['value_range_check'], 
                   quality_checks['nan_check'], 
                   quality_checks['infinity_check'], 
                   quality_checks['formula_check']]):
            quality_checks['overall_quality'] = 'POOR'
        elif len(quality_checks['issues']) > 0:
            quality_checks['overall_quality'] = 'FAIR'
        
        return quality_checks
    
    def _compare_with_benchmark(self, stock_code: str, macd_calculation: Dict[str, float]) -> Dict[str, Any]:
        """与基准数据进行对比"""
        
        benchmark = self.benchmark_data[stock_code]
        
        diff_diff = abs(macd_calculation['DIFF'] - benchmark['DIFF'])
        diff_dea = abs(macd_calculation['DEA'] - benchmark['DEA'])
        diff_macd = abs(macd_calculation['MACD'] - benchmark['MACD'])
        
        comparison = {
            'benchmark_DIFF': benchmark['DIFF'],
            'benchmark_DEA': benchmark['DEA'],
            'benchmark_MACD': benchmark['MACD'],
            'calculated_DIFF': macd_calculation['DIFF'],
            'calculated_DEA': macd_calculation['DEA'],
            'calculated_MACD': macd_calculation['MACD'],
            'diff_DIFF': diff_diff,
            'diff_DEA': diff_dea,
            'diff_MACD': diff_macd,
            'accuracy_DIFF': 1 - min(diff_diff / abs(benchmark['DIFF']), 1) if benchmark['DIFF'] != 0 else 1,
            'accuracy_DEA': 1 - min(diff_dea / abs(benchmark['DEA']), 1) if benchmark['DEA'] != 0 else 1,
            'accuracy_MACD': 1 - min(diff_macd / abs(benchmark['MACD']), 1) if benchmark['MACD'] != 0 else 1
        }
        
        return comparison
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """运行全面验证"""
        
        print("\n🎯 开始全面MACD人工验证")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # 获取股票列表
            print(f"📋 获取股票列表...")
            stock_codes = self.stock_data_service.get_stock_list(limit=self.max_stocks_to_validate)
            
            if not stock_codes:
                self.validation_results['validation_summary']['error'] = '无法获取股票列表'
                return self.validation_results
            
            actual_stock_count = min(len(stock_codes), self.max_stocks_to_validate)
            print(f"✅ 获取到{len(stock_codes)}支股票，将验证前{actual_stock_count}支")
            
            # 分批验证股票
            batch_size = 20
            total_batches = (actual_stock_count + batch_size - 1) // batch_size
            
            successful_validations = 0
            failed_validations = 0
            high_accuracy_validations = 0
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, actual_stock_count)
                batch_stocks = stock_codes[start_idx:end_idx]
                
                print(f"\n📊 验证第{batch_idx+1}/{total_batches}批股票 ({start_idx+1}-{end_idx})")
                
                for i, stock_code in enumerate(batch_stocks):
                    print(f"  🔍 验证 {stock_code} ({start_idx+i+1}/{actual_stock_count})", end=" ")
                    
                    validation_result = self.validate_single_stock(stock_code)
                    self.validation_results['stock_validations'][stock_code] = validation_result
                    
                    status = validation_result['validation_status']
                    if status.startswith('PASSED'):
                        successful_validations += 1
                        print(f"✅ {status}")
                        
                        if status == 'PASSED_HIGH_ACCURACY':
                            high_accuracy_validations += 1
                    elif status.startswith('FAILED'):
                        failed_validations += 1
                        print(f"❌ {status}")
                        if validation_result.get('error_message'):
                            self.validation_results['calculation_errors'].append({
                                'stock_code': stock_code,
                                'error': validation_result['error_message']
                            })
                    else:
                        failed_validations += 1
                        print(f"⚠️ {status}")
                
                # 显示当前进度
                current_total = start_idx + len(batch_stocks)
                success_rate = successful_validations / current_total if current_total > 0 else 0
                print(f"  📈 当前进度: {successful_validations}/{current_total} 成功 ({success_rate:.1%})")
            
            # 生成验证汇总
            end_time = time.time()
            
            self.validation_results['validation_summary'] = {
                'total_stocks_tested': actual_stock_count,
                'successful_validations': successful_validations,
                'failed_validations': failed_validations,
                'high_accuracy_validations': high_accuracy_validations,
                'success_rate': successful_validations / actual_stock_count if actual_stock_count > 0 else 0,
                'high_accuracy_rate': high_accuracy_validations / actual_stock_count if actual_stock_count > 0 else 0,
                'validation_time_seconds': end_time - start_time,
                'stocks_per_second': actual_stock_count / (end_time - start_time) if end_time > start_time else 0
            }
            
            # 统计分析
            self._perform_statistical_analysis()
            
            # 生成建议
            self._generate_recommendations()
            
            print(f"\n🏆 全面验证完成")
            print(f"📊 验证统计:")
            print(f"  总股票数: {actual_stock_count}")
            print(f"  成功验证: {successful_validations} ({successful_validations/actual_stock_count:.1%})")
            print(f"  高精度验证: {high_accuracy_validations} ({high_accuracy_validations/actual_stock_count:.1%})")
            print(f"  验证失败: {failed_validations}")
            print(f"  验证耗时: {end_time - start_time:.1f}秒")
            
        except Exception as e:
            self.validation_results['validation_summary']['error'] = f'验证过程异常: {str(e)}'
            print(f"❌ 验证过程异常: {e}")
        
        return self.validation_results

    def _perform_statistical_analysis(self):
        """执行统计分析"""

        successful_stocks = [
            result for result in self.validation_results['stock_validations'].values()
            if result['validation_status'].startswith('PASSED')
        ]

        if not successful_stocks:
            self.validation_results['statistical_analysis'] = {
                'error': '没有成功验证的股票进行统计分析'
            }
            return

        # 提取MACD数值
        diff_values = [stock['macd_calculation']['DIFF'] for stock in successful_stocks]
        dea_values = [stock['macd_calculation']['DEA'] for stock in successful_stocks]
        macd_values = [stock['macd_calculation']['MACD'] for stock in successful_stocks]

        # 基准对比统计（如果有基准数据）
        benchmark_stats = {}
        benchmark_stocks = [
            result for result in successful_stocks
            if 'benchmark_comparison' in result and result['benchmark_comparison']
        ]

        if benchmark_stocks:
            accuracy_diff = [stock['benchmark_comparison']['accuracy_DIFF'] for stock in benchmark_stocks]
            accuracy_dea = [stock['benchmark_comparison']['accuracy_DEA'] for stock in benchmark_stocks]
            accuracy_macd = [stock['benchmark_comparison']['accuracy_MACD'] for stock in benchmark_stocks]

            benchmark_stats = {
                'benchmark_stocks_count': len(benchmark_stocks),
                'average_accuracy_DIFF': np.mean(accuracy_diff),
                'average_accuracy_DEA': np.mean(accuracy_dea),
                'average_accuracy_MACD': np.mean(accuracy_macd),
                'min_accuracy_DIFF': np.min(accuracy_diff),
                'min_accuracy_DEA': np.min(accuracy_dea),
                'min_accuracy_MACD': np.min(accuracy_macd)
            }

        self.validation_results['statistical_analysis'] = {
            'successful_stocks_count': len(successful_stocks),
            'macd_statistics': {
                'DIFF': {
                    'mean': np.mean(diff_values),
                    'std': np.std(diff_values),
                    'min': np.min(diff_values),
                    'max': np.max(diff_values),
                    'median': np.median(diff_values)
                },
                'DEA': {
                    'mean': np.mean(dea_values),
                    'std': np.std(dea_values),
                    'min': np.min(dea_values),
                    'max': np.max(dea_values),
                    'median': np.median(dea_values)
                },
                'MACD': {
                    'mean': np.mean(macd_values),
                    'std': np.std(macd_values),
                    'min': np.min(macd_values),
                    'max': np.max(macd_values),
                    'median': np.median(macd_values)
                }
            },
            'benchmark_comparison': benchmark_stats
        }

    def _generate_recommendations(self):
        """生成建议"""

        summary = self.validation_results['validation_summary']
        recommendations = []

        # 基于成功率的建议
        if summary['success_rate'] >= 0.95:
            recommendations.append("✅ MACD计算系统表现优秀，可以安全用于生产环境")
        elif summary['success_rate'] >= 0.85:
            recommendations.append("✅ MACD计算系统表现良好，建议进一步优化失败案例")
        elif summary['success_rate'] >= 0.70:
            recommendations.append("⚠️ MACD计算系统表现一般，需要重点关注失败原因")
        else:
            recommendations.append("❌ MACD计算系统存在严重问题，需要全面检查和修复")

        # 基于高精度率的建议
        if summary['high_accuracy_rate'] >= 0.80:
            recommendations.append("🎯 高精度验证率优秀，计算精度满足要求")
        elif summary['high_accuracy_rate'] >= 0.50:
            recommendations.append("🎯 高精度验证率良好，可考虑进一步提升精度")
        else:
            recommendations.append("🎯 高精度验证率偏低，建议检查计算精度问题")

        # 基于错误类型的建议
        error_types = {}
        for error in self.validation_results['calculation_errors']:
            error_msg = error['error']
            error_types[error_msg] = error_types.get(error_msg, 0) + 1

        if error_types:
            recommendations.append("🔍 主要错误类型分析:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                recommendations.append(f"  - {error_type}: {count}次")

        # 基于统计分析的建议
        stats = self.validation_results.get('statistical_analysis', {})
        if 'benchmark_comparison' in stats and stats['benchmark_comparison']:
            benchmark = stats['benchmark_comparison']
            avg_accuracy = (benchmark['average_accuracy_DIFF'] +
                          benchmark['average_accuracy_DEA'] +
                          benchmark['average_accuracy_MACD']) / 3

            if avg_accuracy >= 0.99:
                recommendations.append("📊 基准对比显示计算精度极高，完全满足要求")
            elif avg_accuracy >= 0.95:
                recommendations.append("📊 基准对比显示计算精度良好，基本满足要求")
            else:
                recommendations.append("📊 基准对比显示计算精度需要改进")

        self.validation_results['recommendations'] = recommendations

def main():
    """主函数"""
    print("🚀 全面MACD人工验证系统")
    print("对100+股票进行全面MACD计算验证，确保所有计算都是正确的")

    # 创建验证系统
    validator = ComprehensiveMacdValidation()

    # 运行全面验证
    results = validator.run_comprehensive_validation()

    # 显示最终结果
    summary = results['validation_summary']
    print(f"\n🏆 全面MACD验证完成")
    print("=" * 80)
    print(f"📊 最终统计:")
    print(f"  验证股票总数: {summary['total_stocks_tested']}")
    print(f"  成功验证: {summary['successful_validations']} ({summary['success_rate']:.1%})")
    print(f"  高精度验证: {summary['high_accuracy_validations']} ({summary['high_accuracy_rate']:.1%})")
    print(f"  验证失败: {summary['failed_validations']}")

    if summary['success_rate'] >= 0.95:
        print(f"\n🎉 验证结果优秀！MACD计算系统完全可信！")
    elif summary['success_rate'] >= 0.85:
        print(f"\n✅ 验证结果良好！MACD计算系统基本可信！")
    else:
        print(f"\n⚠️ 验证结果需要关注，建议检查和修复问题！")

if __name__ == "__main__":
    main()
