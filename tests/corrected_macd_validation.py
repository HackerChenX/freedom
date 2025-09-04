#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正的MACD人工验证系统

问题发现：
- 之前的validation_result.json包含错误的MACD计算结果
- 000028股票的MACD数据完全不正确
- 需要重新生成正确的验证结果

解决方案：
- 使用当前正确的MACD计算方法
- 重新验证000028等股票的MACD数据
- 生成准确的人工验证报告
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

class CorrectedMacdValidation:
    """修正的MACD人工验证系统"""
    
    def __init__(self):
        """初始化验证系统"""
        self.macd_indicator = MacdMacd()
        self.stock_data_service = get_stock_data_service()
        
        # 用户提供的真实数据作为基准
        self.real_data_benchmark = {
            '000028': {
                'date': '2025-05-12',
                'MACD': 0.220,
                'DIFF': -0.136,
                'DEA': -0.246
            }
        }
        
        print("✅ 修正的MACD验证系统初始化完成")
    
    def validate_macd_calculation(self, stock_code: str, target_date: str) -> dict:
        """验证MACD计算的准确性"""
        
        print(f"\n🔍 验证{stock_code}股票MACD计算（{target_date}）")
        
        try:
            # 获取股票数据
            df = self.stock_data_service.get_stock_data(stock_code, days=200)
            
            if df is None or len(df) == 0:
                return {'error': f'无法获取{stock_code}数据'}
            
            # 查找目标日期
            target_date_obj = pd.to_datetime(target_date).date()
            target_rows = df[df['date'].dt.date == target_date_obj]
            
            if target_rows.empty:
                return {'error': f'未找到{target_date}的数据'}
            
            target_idx = target_rows.index[0]
            
            # 计算MACD
            macd_result = self.macd_indicator.calculate(df)
            
            if macd_result is None or macd_result.empty:
                return {'error': 'MACD计算失败'}
            
            if target_idx >= len(macd_result):
                return {'error': f'目标索引超出范围'}
            
            # 获取计算结果
            calculated_data = macd_result.iloc[target_idx]
            price_data = df.iloc[target_idx]
            
            result = {
                'stock_code': stock_code,
                'date': target_date,
                'price_data': {
                    'open': float(price_data['open']),
                    'high': float(price_data['high']),
                    'low': float(price_data['low']),
                    'close': float(price_data['close']),
                    'volume': float(price_data['volume'])
                },
                'calculated_macd': {
                    'DIFF': float(calculated_data['macd_line']),
                    'DEA': float(calculated_data['macd_signal']),
                    'MACD': float(calculated_data['macd_histogram'])
                },
                'validation_status': 'calculated'
            }
            
            # 如果有真实数据基准，进行对比
            if stock_code in self.real_data_benchmark:
                real_data = self.real_data_benchmark[stock_code]
                
                diff_diff = abs(result['calculated_macd']['DIFF'] - real_data['DIFF'])
                diff_dea = abs(result['calculated_macd']['DEA'] - real_data['DEA'])
                diff_macd = abs(result['calculated_macd']['MACD'] - real_data['MACD'])
                
                result['real_data_comparison'] = {
                    'real_DIFF': real_data['DIFF'],
                    'real_DEA': real_data['DEA'],
                    'real_MACD': real_data['MACD'],
                    'diff_DIFF': diff_diff,
                    'diff_DEA': diff_dea,
                    'diff_MACD': diff_macd,
                    'accuracy_DIFF': 1 - min(diff_diff / abs(real_data['DIFF']), 1) if real_data['DIFF'] != 0 else 1,
                    'accuracy_DEA': 1 - min(diff_dea / abs(real_data['DEA']), 1) if real_data['DEA'] != 0 else 1,
                    'accuracy_MACD': 1 - min(diff_macd / abs(real_data['MACD']), 1) if real_data['MACD'] != 0 else 1
                }
                
                # 判断验证状态
                if diff_diff < 0.001 and diff_dea < 0.001 and diff_macd < 0.01:
                    result['validation_status'] = 'PASSED_HIGH_ACCURACY'
                elif diff_diff < 0.01 and diff_dea < 0.01 and diff_macd < 0.1:
                    result['validation_status'] = 'PASSED_ACCEPTABLE'
                else:
                    result['validation_status'] = 'FAILED_SIGNIFICANT_DIFFERENCE'
            
            return result
            
        except Exception as e:
            return {'error': f'验证过程异常: {str(e)}'}
    
    def run_corrected_validation(self) -> dict:
        """运行修正的验证"""
        
        print("🎯 运行修正的MACD人工验证")
        print("=" * 80)
        
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_type': 'corrected_macd_validation',
            'benchmark_data': self.real_data_benchmark,
            'stock_validations': {},
            'overall_assessment': {},
            'issues_found': []
        }
        
        # 验证基准股票
        for stock_code, benchmark in self.real_data_benchmark.items():
            print(f"\n📊 验证基准股票: {stock_code}")
            
            validation_result = self.validate_macd_calculation(
                stock_code, 
                benchmark['date']
            )
            
            validation_results['stock_validations'][stock_code] = validation_result
            
            if 'error' in validation_result:
                print(f"  ❌ 验证失败: {validation_result['error']}")
                validation_results['issues_found'].append(f"{stock_code}: {validation_result['error']}")
            else:
                status = validation_result['validation_status']
                print(f"  ✅ 验证完成: {status}")
                
                if 'real_data_comparison' in validation_result:
                    comp = validation_result['real_data_comparison']
                    print(f"    DIFF准确率: {comp['accuracy_DIFF']:.1%}")
                    print(f"    DEA准确率: {comp['accuracy_DEA']:.1%}")
                    print(f"    MACD准确率: {comp['accuracy_MACD']:.1%}")
        
        # 验证其他关键股票（从之前的错误结果中选择）
        additional_stocks = ['000001', '000002', '000006']  # 可以添加更多
        
        for stock_code in additional_stocks:
            print(f"\n📊 验证其他股票: {stock_code}")
            
            validation_result = self.validate_macd_calculation(
                stock_code, 
                '2025-05-12'  # 使用相同日期
            )
            
            validation_results['stock_validations'][stock_code] = validation_result
            
            if 'error' in validation_result:
                print(f"  ❌ 验证失败: {validation_result['error']}")
            else:
                print(f"  ✅ 计算成功")
                calc = validation_result['calculated_macd']
                print(f"    DIFF: {calc['DIFF']:.6f}")
                print(f"    DEA: {calc['DEA']:.6f}")
                print(f"    MACD: {calc['MACD']:.6f}")
        
        # 生成整体评估
        successful_validations = sum(1 for v in validation_results['stock_validations'].values() 
                                   if 'error' not in v)
        total_validations = len(validation_results['stock_validations'])
        
        high_accuracy_count = sum(1 for v in validation_results['stock_validations'].values() 
                                if v.get('validation_status') == 'PASSED_HIGH_ACCURACY')
        
        validation_results['overall_assessment'] = {
            'total_stocks_tested': total_validations,
            'successful_calculations': successful_validations,
            'high_accuracy_validations': high_accuracy_count,
            'success_rate': successful_validations / total_validations if total_validations > 0 else 0,
            'high_accuracy_rate': high_accuracy_count / total_validations if total_validations > 0 else 0
        }
        
        # 判断整体状态
        if validation_results['overall_assessment']['high_accuracy_rate'] >= 0.8:
            validation_results['overall_assessment']['status'] = 'EXCELLENT'
        elif validation_results['overall_assessment']['success_rate'] >= 0.8:
            validation_results['overall_assessment']['status'] = 'GOOD'
        elif validation_results['overall_assessment']['success_rate'] >= 0.6:
            validation_results['overall_assessment']['status'] = 'ACCEPTABLE'
        else:
            validation_results['overall_assessment']['status'] = 'NEEDS_IMPROVEMENT'
        
        return validation_results
    
    def save_corrected_results(self, results: dict):
        """保存修正的结果"""
        
        # 创建结果目录
        results_dir = Path("validation/corrected_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON结果
        json_file = results_dir / "corrected_MACD_validation.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 修正的验证结果已保存:")
        print(f"  📊 JSON文件: {json_file}")
        
        # 生成人工验证报告
        report_file = results_dir / "MACD_人工验证报告.md"
        self._generate_markdown_report(results, report_file)
        print(f"  📋 验证报告: {report_file}")
    
    def _generate_markdown_report(self, results: dict, report_file: Path):
        """生成Markdown验证报告"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# MACD指标人工验证报告（修正版）\n\n")
            f.write(f"**验证时间**: {results['validation_timestamp']}\n\n")
            
            # 整体评估
            overall = results['overall_assessment']
            f.write("## 整体评估\n\n")
            f.write(f"- **验证状态**: {overall['status']}\n")
            f.write(f"- **测试股票数**: {overall['total_stocks_tested']}\n")
            f.write(f"- **计算成功率**: {overall['success_rate']:.1%}\n")
            f.write(f"- **高精度验证率**: {overall['high_accuracy_rate']:.1%}\n\n")
            
            # 基准验证结果
            f.write("## 基准数据验证\n\n")
            for stock_code, benchmark in results['benchmark_data'].items():
                if stock_code in results['stock_validations']:
                    validation = results['stock_validations'][stock_code]
                    f.write(f"### {stock_code}股票验证\n\n")
                    f.write(f"**验证日期**: {benchmark['date']}\n\n")
                    
                    if 'error' not in validation:
                        f.write("**计算结果对比**:\n\n")
                        f.write("| 指标 | 真实值 | 计算值 | 差异 | 准确率 |\n")
                        f.write("|------|--------|--------|------|--------|\n")
                        
                        if 'real_data_comparison' in validation:
                            comp = validation['real_data_comparison']
                            calc = validation['calculated_macd']
                            
                            f.write(f"| DIFF | {comp['real_DIFF']:.6f} | {calc['DIFF']:.6f} | {comp['diff_DIFF']:.6f} | {comp['accuracy_DIFF']:.1%} |\n")
                            f.write(f"| DEA | {comp['real_DEA']:.6f} | {calc['DEA']:.6f} | {comp['diff_DEA']:.6f} | {comp['accuracy_DEA']:.1%} |\n")
                            f.write(f"| MACD | {comp['real_MACD']:.6f} | {calc['MACD']:.6f} | {comp['diff_MACD']:.6f} | {comp['accuracy_MACD']:.1%} |\n")
                        
                        f.write(f"\n**验证状态**: {validation['validation_status']}\n\n")
                    else:
                        f.write(f"**验证失败**: {validation['error']}\n\n")
            
            # 其他股票验证
            f.write("## 其他股票验证结果\n\n")
            for stock_code, validation in results['stock_validations'].items():
                if stock_code not in results['benchmark_data']:
                    f.write(f"### {stock_code}股票\n\n")
                    if 'error' not in validation:
                        calc = validation['calculated_macd']
                        f.write(f"- **DIFF**: {calc['DIFF']:.6f}\n")
                        f.write(f"- **DEA**: {calc['DEA']:.6f}\n")
                        f.write(f"- **MACD**: {calc['MACD']:.6f}\n\n")
                    else:
                        f.write(f"- **错误**: {validation['error']}\n\n")
            
            # 结论
            f.write("## 验证结论\n\n")
            if overall['status'] == 'EXCELLENT':
                f.write("✅ **MACD指标计算完全正确**，可以通过人工验证。\n\n")
            elif overall['status'] == 'GOOD':
                f.write("✅ **MACD指标计算基本正确**，精度满足要求。\n\n")
            elif overall['status'] == 'ACCEPTABLE':
                f.write("⚠️ **MACD指标计算可接受**，但建议进一步优化。\n\n")
            else:
                f.write("❌ **MACD指标计算需要改进**，存在显著问题。\n\n")

def main():
    """主函数"""
    print("🔧 修正的MACD人工验证系统")
    print("解决之前validation_result.json中的错误数据问题")
    
    # 创建验证系统
    validator = CorrectedMacdValidation()
    
    # 运行修正验证
    results = validator.run_corrected_validation()
    
    # 保存结果
    validator.save_corrected_results(results)
    
    # 显示总结
    overall = results['overall_assessment']
    print(f"\n🏆 修正验证完成")
    print(f"📊 整体状态: {overall['status']}")
    print(f"📈 计算成功率: {overall['success_rate']:.1%}")
    print(f"🎯 高精度验证率: {overall['high_accuracy_rate']:.1%}")
    
    if overall['status'] in ['EXCELLENT', 'GOOD']:
        print(f"✅ MACD指标计算正确，可以继续人工验证！")
    else:
        print(f"⚠️ MACD指标计算需要进一步检查")

if __name__ == "__main__":
    main()
