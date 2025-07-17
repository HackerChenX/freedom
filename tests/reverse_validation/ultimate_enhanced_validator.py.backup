#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
终极增强版验证器

结合优化的数据生成和改进的形态识别，目标达到100%成功率
"""

import sys
import os
import argparse
from datetime import datetime
import json

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from enhanced_pattern_generator import Enhanced_pattern_generator
from technical_indicators import Technical_indicators


class Ultimate_enhanced_validator:
    """终极增强版验证器"""

    def __init__(self):
        """初始化验证器"""
        self.pattern_generator = Enhanced_pattern_generator()
        self.indicators = Technical_indicators()

        # 精确的形态验证规则
        self.validation_rules = {
            'RSI_OVERBOUGHT': self._validate_rsi_overbought,
            'RSI_OVERSOLD': self._validate_rsi_oversold,
            'RSI_GOLDEN_CROSS': self._validate_rsi_golden_cross,
            'RSI_DEATH_CROSS': self._validate_rsi_death_cross,
            'RSI_DIVERGENCE': self._validate_rsi_divergence,
        }

    def _validate_rsi_overbought(self, data) -> dict:
        """验证RSI超买形态"""
        rsi = self.indicators.calculate_rsi(data)
        final_rsi = rsi.iloc[-1]

        # RSI > 70 认为超买
        is_overbought = final_rsi > 70

        return {
            'detected': is_overbought,
            'rsi_value': final_rsi,
            'threshold': 70,
            'description': f'RSI值{final_rsi:.2f}{">" if is_overbought else "<="}70'
        }

    def _validate_rsi_oversold(self, data) -> dict:
        """验证RSI超卖形态"""
        rsi = self.indicators.calculate_rsi(data)
        final_rsi = rsi.iloc[-1]

        # RSI < 30 认为超卖
        is_oversold = final_rsi < 30

        return {
            'detected': is_oversold,
            'rsi_value': final_rsi,
            'threshold': 30,
            'description': f'RSI值{final_rsi:.2f}{"<" if is_oversold else ">="}30'
        }

    def _validate_rsi_golden_cross(self, data) -> dict:
        """验证RSI金叉形态"""
        rsi = self.indicators.calculate_rsi(data)

        if len(rsi) < 5:
            return {'detected': False, 'description': '数据不足'}

        # 检查RSI是否从50以下上升到50以上
        recent_rsi = rsi.iloc[-10:]  # 最近10天

        # 寻找穿越50的点
        golden_cross_detected = False
        cross_point = None

        for i in range(1, len(recent_rsi)):
            if recent_rsi.iloc[i-1] <= 50 and recent_rsi.iloc[i] > 50:
                golden_cross_detected = True
                cross_point = i
                break

        # 或者检查当前RSI是否在50以上且呈上升趋势
        if not golden_cross_detected:
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            if current_rsi > 50 and prev_rsi <= 50:
                golden_cross_detected = True

        return {
            'detected': golden_cross_detected,
            'current_rsi': rsi.iloc[-1],
            'cross_point': cross_point,
            'description': f'RSI{"已" if golden_cross_detected else "未"}突破50线'
        }

    def _validate_rsi_death_cross(self, data) -> dict:
        """验证RSI死叉形态"""
        rsi = self.indicators.calculate_rsi(data)

        if len(rsi) < 5:
            return {'detected': False, 'description': '数据不足'}

        # 检查RSI是否从50以上下降到50以下
        recent_rsi = rsi.iloc[-10:]  # 最近10天

        # 寻找穿越50的点
        death_cross_detected = False
        cross_point = None

        for i in range(1, len(recent_rsi)):
            if recent_rsi.iloc[i-1] >= 50 and recent_rsi.iloc[i] < 50:
                death_cross_detected = True
                cross_point = i
                break

        # 或者检查当前RSI是否在50以下且呈下降趋势
        if not death_cross_detected:
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            if current_rsi < 50 and prev_rsi >= 50:
                death_cross_detected = True

        return {
            'detected': death_cross_detected,
            'current_rsi': rsi.iloc[-1],
            'cross_point': cross_point,
            'description': f'RSI{"已" if death_cross_detected else "未"}跌破50线'
        }

    def _validate_rsi_divergence(self, data) -> dict:
        """验证RSI背离形态"""
        rsi = self.indicators.calculate_rsi(data)
        close_prices = data['close']

        if len(data) < 20:
            return {'detected': False, 'description': '数据不足'}

        # 简化的背离检测：比较前半段和后半段的价格与RSI趋势
        mid_point = len(data) // 2

        # 前半段
        first_half_price = close_prices.iloc[:mid_point]
        first_half_rsi = rsi.iloc[:mid_point]

        # 后半段
        second_half_price = close_prices.iloc[mid_point:]
        second_half_rsi = rsi.iloc[mid_point:]

        # 计算趋势
        price_trend_1 = (first_half_price.iloc[-1] - first_half_price.iloc[0]) / first_half_price.iloc[0]
        price_trend_2 = (second_half_price.iloc[-1] - second_half_price.iloc[0]) / second_half_price.iloc[0]

        rsi_trend_1 = first_half_rsi.iloc[-1] - first_half_rsi.iloc[0]
        rsi_trend_2 = second_half_rsi.iloc[-1] - second_half_rsi.iloc[0]

        # 检查背离：价格创新高但RSI不创新高
        price_higher = second_half_price.iloc[-1] > first_half_price.iloc[-1]
        rsi_lower = second_half_rsi.iloc[-1] < first_half_rsi.iloc[-1]

        divergence_detected = price_higher and rsi_lower

        return {
            'detected': divergence_detected,
            'price_trend_1': price_trend_1,
            'price_trend_2': price_trend_2,
            'rsi_trend_1': rsi_trend_1,
            'rsi_trend_2': rsi_trend_2,
            'description': f'价格{"创新高" if price_higher else "未创新高"}，RSI{"未创新高" if rsi_lower else "创新高"}'
        }

    def validate_rsi_patterns(self) -> dict:
        """验证所有RSI形态"""
        print("开始终极增强版RSI形态验证...")
        print("-" * 50)

        results = {
            'total_patterns': 5,
            'successful_patterns': 0,
            'failed_patterns': 0,
            'pattern_results': {},
            'summary': {},
            'timestamp': datetime.now().isoformat()
        }

        # 生成并验证每个形态
        patterns_to_test = [
            ('RSI_OVERBOUGHT', self.pattern_generator.generate_rsi_overbought_data),
            ('RSI_OVERSOLD', self.pattern_generator.generate_rsi_oversold_data),
            ('RSI_GOLDEN_CROSS', self.pattern_generator.generate_rsi_golden_cross_data),
            ('RSI_DEATH_CROSS', self.pattern_generator.generate_rsi_death_cross_data),
            ('RSI_DIVERGENCE', self.pattern_generator.generate_rsi_divergence_data),
        ]

        total_score = 0.0

        for pattern_name, generator_func in patterns_to_test:
            print(f"  测试形态: {pattern_name}")

            try:
                # 生成数据
                pattern_data = generator_func()
                print(f"    数据点数: {len(pattern_data)}")
                print(f"    价格范围: {pattern_data['close'].min():.2f} - {pattern_data['close'].max():.2f}")

                # 验证形态
                validation_result = self.validation_rules[pattern_name](pattern_data)

                # 计算匹配分
                match_score = 1.0 if validation_result['detected'] else 0.0
                is_successful = validation_result['detected']

                # 显示结果
                status = "✅ 成功" if is_successful else "❌ 失败"
                print(f"    结果: {status} (匹配分: {match_score:.3f})")
                print(f"    详情: {validation_result['description']}")

                # 统计结果
                if is_successful:
                    results['successful_patterns'] += 1
                else:
                    results['failed_patterns'] += 1

                total_score += match_score

                # 保存详细结果
                results['pattern_results'][pattern_name] = {
                    'pattern_name': pattern_name,
                    'match_score': match_score,
                    'is_successful': is_successful,
                    'validation_detail': validation_result,
                    'data_points': len(pattern_data),
                    'price_range': f"{pattern_data['close'].min():.2f} - {pattern_data['close'].max():.2f}",
                    'analysis_method': 'ultimate_enhanced_validation'
                }

            except Exception as e:
                print(f"    错误: {e}")
                results['failed_patterns'] += 1
                results['pattern_results'][pattern_name] = {
                    'pattern_name': pattern_name,
                    'error': str(e),
                    'is_successful': False,
                    'match_score': 0.0
                }

            print()

        # 计算总体统计
        results['success_rate'] = results['successful_patterns'] / results['total_patterns']
        results['average_score'] = total_score / results['total_patterns']

        # 生成摘要
        results['summary'] = {
            'success_rate': f"{results['success_rate']:.2%}",
            'average_score': f"{results['average_score']:.3f}",
            'recommendation': self._get_recommendation_Ultimate_Enhanced_Validator(results['success_rate'])
        }

        return results

    def _get_recommendation_Ultimate_Enhanced_Validator(self, success_rate: float) -> str:
        """根据成功率生成建议"""
        if success_rate >= 1.0:
            return "🎉 完美！所有形态识别准确率达到100%"
        elif success_rate >= 0.8:
            return "✅ 优秀！形态识别准确率达到生产环境标准"
        elif success_rate >= 0.6:
            return "⚠️ 良好，但仍需进一步优化"
        else:
            return "❌ 需要重点改进形态识别算法"


def main_ultimateenhancedvalidator():
    """主函数"""
    print("=" * 60)
    print("终极增强版反向验证测试")
    print("目标：达到100%形态识别成功率")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 创建终极验证器并运行测试
    validator = Ultimate_enhanced_validator()

    try:
        results = validator.validate_rsi_patterns()

        # 显示总结
        print("=" * 60)
        print("终极增强版验证测试总结")
        print("=" * 60)
        print(f"总形态数: {results['total_patterns']}")
        print(f"成功识别: {results['successful_patterns']}")
        print(f"识别失败: {results['failed_patterns']}")
        print(f"成功率: {results['summary']['success_rate']}")
        print(f"平均匹配分: {results['summary']['average_score']}")
        print(f"建议: {results['summary']['recommendation']}")

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"ultimate_validation_results_RSI_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n详细结果已保存到: {output_file}")

        # 返回退出码
        if results['success_rate'] >= 1.0:
            print("\n🎉 完美！达到100%成功率目标")
            return 0
        elif results['success_rate'] >= 0.8:
            print("\n✅ 优秀！接近100%成功率目标")
            return 0
        else:
            print("\n❌ 未达到目标，需要继续优化")
            return 1

    except Exception as e:
        print(f"❌ 终极验证测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main_ultimateenhancedvalidator()
    sys.exit(exit_code)