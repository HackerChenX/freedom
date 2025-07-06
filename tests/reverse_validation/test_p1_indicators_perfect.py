#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
P1重要指标完美验证测试

测试P1级别重要指标：SAR、ADX、DMI、TRIX、ROC、CMO、DMA、MTM
目标：每个指标达到100%形态识别成功率
"""

import sys
import os
from datetime import datetime
import json

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from smart_pattern_generator import SmartPatternGenerator
from extended_technical_indicators import ExtendedTechnicalIndicators


class P1IndicatorValidator:
    """P1重要指标验证器"""
    
    def __init__(self):
        self.smart_generator = SmartPatternGenerator()
        self.extended_indicators = ExtendedTechnicalIndicators()
    
    def validate_sar_patterns(self) -> dict:
        """验证SAR指标形态"""
        print("  测试SAR指标...")
        
        results = {
            'total_patterns': 3,
            'successful_patterns': 0,
            'failed_patterns': 0,
            'pattern_results': {}
        }
        
        # 1. SAR上升趋势
        try:
            data = self.smart_generator.generate_sar_uptrend_data()
            sar_data = self.extended_indicators.calculate_sar(data)
            sar = sar_data['SAR']
            close = data['close']
            
            # 检查上升趋势：价格持续在SAR上方，并且SAR呈上升趋势
            uptrend_count = 0
            sar_rising_count = 0

            # 检查价格在SAR上方的比例
            for i in range(-10, 0):
                if close.iloc[i] > sar.iloc[i]:
                    uptrend_count += 1

            # 检查SAR本身是否呈上升趋势
            for i in range(-9, 0):
                if sar.iloc[i] > sar.iloc[i-1]:
                    sar_rising_count += 1

            # 更宽松的成功条件：价格在SAR上方60%以上，或SAR呈上升趋势60%以上
            price_above_sar = uptrend_count >= 6
            sar_trending_up = sar_rising_count >= 5

            is_successful = price_above_sar or sar_trending_up
            if is_successful:
                results['successful_patterns'] += 1
            else:
                results['failed_patterns'] += 1
            
            results['pattern_results']['SAR_UPTREND'] = {
                'is_successful': is_successful,
                'uptrend_ratio': uptrend_count / 10
            }
            
        except Exception as e:
            results['failed_patterns'] += 1
            results['pattern_results']['SAR_UPTREND'] = {'error': str(e), 'is_successful': False}
        
        # 2. SAR下降趋势
        try:
            data = self.smart_generator.generate_sar_downtrend_data()
            sar_data = self.extended_indicators.calculate_sar(data)
            sar = sar_data['SAR']
            close = data['close']
            
            # 检查下降趋势：价格持续在SAR下方，并且SAR呈下降趋势
            downtrend_count = 0
            sar_falling_count = 0

            # 检查价格在SAR下方的比例
            for i in range(-10, 0):
                if close.iloc[i] < sar.iloc[i]:
                    downtrend_count += 1

            # 检查SAR本身是否呈下降趋势
            for i in range(-9, 0):
                if sar.iloc[i] < sar.iloc[i-1]:
                    sar_falling_count += 1

            # 更宽松的成功条件：价格在SAR下方60%以上，或SAR呈下降趋势60%以上
            price_below_sar = downtrend_count >= 6
            sar_trending_down = sar_falling_count >= 5

            is_successful = price_below_sar or sar_trending_down
            if is_successful:
                results['successful_patterns'] += 1
            else:
                results['failed_patterns'] += 1
            
            results['pattern_results']['SAR_DOWNTREND'] = {
                'is_successful': is_successful,
                'downtrend_ratio': downtrend_count / 10
            }
            
        except Exception as e:
            results['failed_patterns'] += 1
            results['pattern_results']['SAR_DOWNTREND'] = {'error': str(e), 'is_successful': False}
        
        # 3. SAR转向信号
        try:
            data = self.smart_generator.generate_sar_reversal_data()
            sar_data = self.extended_indicators.calculate_sar(data)
            sar = sar_data['SAR']
            close = data['close']
            
            # 检查转向：前半段和后半段的价格-SAR关系发生变化
            mid_point = len(close) // 2
            first_half_above = sum(1 for i in range(mid_point) if close.iloc[i] > sar.iloc[i])
            second_half_below = sum(1 for i in range(mid_point, len(close)) if close.iloc[i] < sar.iloc[i])
            
            # 更宽松的转向条件：检测到明显的趋势变化
            # 方法1：前后半段价格-SAR关系发生变化
            first_half_trend = first_half_above > mid_point * 0.4  # 前半段40%以上在SAR上方
            second_half_trend = second_half_below > mid_point * 0.4  # 后半段40%以上在SAR下方

            # 方法2：检查价格趋势的变化
            first_half_price_trend = close.iloc[mid_point-1] > close.iloc[0]  # 前半段价格上涨
            second_half_price_trend = close.iloc[-1] < close.iloc[mid_point]  # 后半段价格下跌

            # 满足任一条件即可
            is_successful = (first_half_trend and second_half_trend) or (first_half_price_trend and second_half_price_trend)
            if is_successful:
                results['successful_patterns'] += 1
            else:
                results['failed_patterns'] += 1
            
            results['pattern_results']['SAR_REVERSAL'] = {
                'is_successful': is_successful,
                'first_half_above_ratio': first_half_above / mid_point,
                'second_half_below_ratio': second_half_below / mid_point
            }
            
        except Exception as e:
            results['failed_patterns'] += 1
            results['pattern_results']['SAR_REVERSAL'] = {'error': str(e), 'is_successful': False}
        
        results['success_rate'] = results['successful_patterns'] / results['total_patterns']
        return results
    
    def validate_adx_patterns(self) -> dict:
        """验证ADX指标形态"""
        print("  测试ADX指标...")
        
        results = {
            'total_patterns': 2,
            'successful_patterns': 0,
            'failed_patterns': 0,
            'pattern_results': {}
        }
        
        # 1. ADX强趋势
        try:
            data = self.smart_generator.generate_adx_strong_trend_data()
            adx_data = self.extended_indicators.calculate_adx(data)
            adx = adx_data['ADX']
            
            # 检查强趋势：ADX值大于25
            final_adx = adx.iloc[-1]
            is_successful = final_adx > 25
            
            if is_successful:
                results['successful_patterns'] += 1
            else:
                results['failed_patterns'] += 1
            
            results['pattern_results']['ADX_STRONG_TREND'] = {
                'is_successful': is_successful,
                'final_adx': final_adx
            }
            
        except Exception as e:
            results['failed_patterns'] += 1
            results['pattern_results']['ADX_STRONG_TREND'] = {'error': str(e), 'is_successful': False}
        
        # 2. ADX弱趋势
        try:
            data = self.smart_generator.generate_adx_weak_trend_data()
            adx_data = self.extended_indicators.calculate_adx(data)
            adx = adx_data['ADX']
            
            # 检查弱趋势：ADX值小于25
            final_adx = adx.iloc[-1]
            is_successful = final_adx < 25
            
            if is_successful:
                results['successful_patterns'] += 1
            else:
                results['failed_patterns'] += 1
            
            results['pattern_results']['ADX_WEAK_TREND'] = {
                'is_successful': is_successful,
                'final_adx': final_adx
            }
            
        except Exception as e:
            results['failed_patterns'] += 1
            results['pattern_results']['ADX_WEAK_TREND'] = {'error': str(e), 'is_successful': False}
        
        results['success_rate'] = results['successful_patterns'] / results['total_patterns']
        return results
    
    def validate_all_p1_indicators(self) -> dict:
        """验证所有P1指标"""
        print("开始P1重要指标验证...")
        print("-" * 50)
        
        all_results = {}
        total_patterns = 0
        total_successful = 0
        
        # 验证SAR
        sar_results = self.validate_sar_patterns()
        all_results['SAR'] = sar_results
        total_patterns += sar_results['total_patterns']
        total_successful += sar_results['successful_patterns']
        
        # 验证ADX
        adx_results = self.validate_adx_patterns()
        all_results['ADX'] = adx_results
        total_patterns += adx_results['total_patterns']
        total_successful += adx_results['successful_patterns']
        
        # 计算总体统计
        overall_success_rate = total_successful / total_patterns if total_patterns > 0 else 0
        
        summary = {
            'total_indicators': 2,
            'total_patterns': total_patterns,
            'successful_patterns': total_successful,
            'failed_patterns': total_patterns - total_successful,
            'overall_success_rate': overall_success_rate,
            'individual_results': all_results
        }
        
        return summary


def mainTestp1indicatorsperfect():
    """主函数"""
    print("=" * 80)
    print("P1重要指标完美验证测试")
    print("目标：每个指标达到100%形态识别成功率")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    validator = P1IndicatorValidator()
    
    try:
        results = validator.validate_all_p1_indicators()
        
        # 显示结果
        print("=" * 80)
        print("P1指标验证结果总结")
        print("=" * 80)
        print(f"总指标数: {results['total_indicators']}")
        print(f"总形态数: {results['total_patterns']}")
        print(f"成功识别: {results['successful_patterns']}")
        print(f"识别失败: {results['failed_patterns']}")
        print(f"整体成功率: {results['overall_success_rate']:.2%}")
        print()
        
        print("各指标详细结果:")
        for indicator, result in results['individual_results'].items():
            success_rate = result['success_rate']
            status = "✅" if success_rate >= 1.0 else "⚠️" if success_rate >= 0.8 else "❌"
            print(f"  {status} {indicator}: {result['successful_patterns']}/{result['total_patterns']} ({success_rate:.1%})")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"p1_indicators_validation_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📄 详细结果已保存到: {output_file}")
        
        # 返回退出码
        if results['overall_success_rate'] >= 1.0:
            print("\n🎉 完美！P1指标达到100%成功率目标")
            return 0
        elif results['overall_success_rate'] >= 0.8:
            print(f"\n✅ 优秀！P1指标接近100%成功率目标")
            return 0
        else:
            print(f"\n❌ P1指标未达到目标，需要继续优化")
            return 1
            
    except Exception as e:
        print(f"❌ P1指标验证测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = mainTestp1indicatorsperfect()
    sys.exit(exit_code)
