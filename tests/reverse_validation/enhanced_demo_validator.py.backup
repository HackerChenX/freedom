#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
增强版单个指标反向验证演示脚本

使用真实的技术指标计算和形态识别算法，替换简化的验证逻辑
目标是将成功率从20%提升到100%
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

from pattern_data_generator import Pattern_data_generator
from technical_indicators import Pattern_recognizer


class Enhanced_indicator_validator:
    """增强版指标验证器 - 使用真实技术指标计算"""

    def __init__(self):
        """初始化验证器"""
        self.pattern_generator = Pattern_data_generator()
        self.pattern_recognizer = Pattern_recognizer()

        # 更精确的预期形态映射
        self.expected_patterns = {
            'RSI_OVERBOUGHT': ['RSI_OVERBOUGHT', 'overbought', '超买'],
            'RSI_OVERSOLD': ['RSI_OVERSOLD', 'oversold', '超卖'],
            'RSI_GOLDEN_CROSS': ['RSI_GOLDEN_CROSS', 'golden_cross', '金叉'],
            'RSI_DEATH_CROSS': ['RSI_DEATH_CROSS', 'death_cross', '死叉'],
            'RSI_DIVERGENCE': ['RSI_DIVERGENCE', 'divergence', '背离'],

            'MACD_GOLDEN_CROSS': ['MACD_GOLDEN_CROSS', 'golden_cross', '金叉'],
            'MACD_DEATH_CROSS': ['MACD_DEATH_CROSS', 'death_cross', '死叉'],
            'MACD_ABOVE_ZERO_GOLDEN': ['MACD_ABOVE_ZERO_GOLDEN', 'above_zero', '零轴上'],
            'MACD_BELOW_ZERO_DEATH': ['MACD_BELOW_ZERO_DEATH', 'below_zero', '零轴下'],
            'MACD_HISTOGRAM_DIVERGENCE': ['MACD_HISTOGRAM_DIVERGENCE', 'divergence', '背离'],

            'KDJ_GOLDEN_CROSS': ['KDJ_GOLDEN_CROSS', 'golden_cross', '金叉'],
            'KDJ_DEATH_CROSS': ['KDJ_DEATH_CROSS', 'death_cross', '死叉'],
            'KDJ_OVERBOUGHT': ['KDJ_OVERBOUGHT', 'overbought', '超买'],
            'KDJ_OVERSOLD': ['KDJ_OVERSOLD', 'oversold', '超卖'],
            'KDJ_BLUNT': ['KDJ_BLUNT', 'blunt', '钝化'],

            'BOLL_UPPER_BREAKOUT': ['BOLL_UPPER_BREAKOUT', 'upper_breakout', '上轨突破'],
            'BOLL_LOWER_BREAKOUT': ['BOLL_LOWER_BREAKOUT', 'lower_breakout', '下轨突破'],
            'BOLL_SQUEEZE': ['BOLL_SQUEEZE', 'squeeze', '收口'],
            'BOLL_EXPANSION': ['BOLL_EXPANSION', 'expansion', '开口'],
            'BOLL_MIDDLE_SUPPORT': ['BOLL_MIDDLE_SUPPORT', 'middle_support', '中轨支撑'],

            'MA_GOLDEN_CROSS': ['MA_GOLDEN_CROSS', 'golden_cross', '金叉'],
            'MA_DEATH_CROSS': ['MA_DEATH_CROSS', 'death_cross', '死叉'],
            'MA_BULLISH_ALIGNMENT': ['MA_BULLISH_ALIGNMENT', 'bullish', '多头排列'],
            'MA_BEARISH_ALIGNMENT': ['MA_BEARISH_ALIGNMENT', 'bearish', '空头排列'],
            'MA_SUPPORT': ['MA_SUPPORT', 'support', '支撑'],

            'EMA_GOLDEN_CROSS': ['EMA_GOLDEN_CROSS', 'golden_cross', '金叉'],
            'EMA_DEATH_CROSS': ['EMA_DEATH_CROSS', 'death_cross', '死叉'],
            'EMA_TREND_CONFIRMATION': ['EMA_TREND_CONFIRMATION', 'trend', '趋势确认'],
            'EMA_DIVERGENCE': ['EMA_DIVERGENCE', 'divergence', '背离'],
            'EMA_SUPPORT_RESISTANCE': ['EMA_SUPPORT_RESISTANCE', 'support', '支撑阻力']
        }

    def real_indicator_analysis(self, indicator: str, pattern_name: str, pattern_data) -> dict:
        """
        使用真实技术指标分析

        Args:
            indicator: 指标名称
            pattern_name: 形态名称
            pattern_data: 形态数据

        Returns:
            分析结果
        """
        try:
            # 根据指标类型调用相应的形态识别方法
            if indicator.upper() == 'RSI':
                detected_patterns = self.pattern_recognizer.detect_rsi_patterns(pattern_data)
            elif indicator.upper() == 'MACD':
                detected_patterns = self.pattern_recognizer.detect_macd_patterns(pattern_data)
            elif indicator.upper() == 'KDJ':
                detected_patterns = self.pattern_recognizer.detect_kdj_patterns(pattern_data)
            elif indicator.upper() == 'BOLL':
                detected_patterns = self.pattern_recognizer.detect_boll_patterns(pattern_data)
            elif indicator.upper() == 'MA':
                detected_patterns = self.pattern_recognizer.detect_ma_patterns(pattern_data)
            elif indicator.upper() == 'EMA':
                detected_patterns = self.pattern_recognizer.detect_ema_patterns(pattern_data)
            else:
                detected_patterns = {}

            # 获取预期形态
            expected_patterns = self.expected_patterns.get(pattern_name, [])

            # 提取识别出的形态名称
            identified_patterns = [name for name, detected in detected_patterns.items() if detected]

            # 计算匹配度
            match_score = self._calculate_enhanced_match_score(expected_patterns, identified_patterns, pattern_name)

            # 计算价格趋势（用于参考）
            close_prices = pattern_data['close'].values
            price_trend = (close_prices[-1] - close_prices[0]) / close_prices[0]

            return {
                'indicator': indicator,
                'pattern_name': pattern_name,
                'expected_patterns': expected_patterns,
                'identified_patterns': identified_patterns,
                'detected_patterns_detail': detected_patterns,
                'match_score': match_score,
                'is_successful': match_score > 0.7,  # 提高成功阈值
                'price_trend': f"{price_trend:.2%}",
                'data_points': len(pattern_data),
                'price_range': f"{close_prices.min():.2f} - {close_prices.max():.2f}",
                'analysis_method': 'real_technical_indicators'
            }

        except Exception as e:
            return {
                'indicator': indicator,
                'pattern_name': pattern_name,
                'error': str(e),
                'is_successful': False,
                'match_score': 0.0,
                'analysis_method': 'real_technical_indicators'
            }

    def _calculate_enhanced_match_score(self, expected_patterns: list, identified_patterns: list, pattern_name: str) -> float:
        """
        增强的匹配评分算法

        Args:
            expected_patterns: 预期形态列表
            identified_patterns: 识别出的形态列表
            pattern_name: 形态名称

        Returns:
            匹配评分（0-1）
        """
        if not expected_patterns:
            return 0.0

        if not identified_patterns:
            return 0.0

        # 精确匹配：检查是否直接识别出了目标形态
        if pattern_name in identified_patterns:
            return 1.0

        # 语义匹配：检查关键词匹配
        matches = 0
        for expected in expected_patterns:
            for identified in identified_patterns:
                # 完全匹配
                if expected.lower() == identified.lower():
                    matches += 2  # 完全匹配给更高分
                # 包含匹配
                elif expected.lower() in identified.lower() or identified.lower() in expected.lower():
                    matches += 1

        # 计算匹配率，考虑权重
        max_possible_score = len(expected_patterns) * 2  # 最高可能分数
        match_score = min(matches / max_possible_score, 1.0)

        return match_score

    def validate_single_indicator_Validator(self, indicator: str) -> dict:
        """
        验证单个指标的所有形态

        Args:
            indicator: 指标名称 (RSI, MACD, KDJ, BOLL, MA, EMA)

        Returns:
            dict: 验证结果
        """
        print(f"开始增强验证指标: {indicator}")
        print("-" * 50)

        # 生成该指标的形态数据
        if indicator.upper() == 'RSI':
            patterns = self.pattern_generator.generate_rsi_patterns()
        elif indicator.upper() == 'MACD':
            patterns = self.pattern_generator.generate_macd_patterns()
        elif indicator.upper() == 'KDJ':
            patterns = self.pattern_generator.generate_kdj_patterns()
        elif indicator.upper() == 'BOLL':
            patterns = self.pattern_generator.generate_boll_patterns()
        elif indicator.upper() == 'MA':
            patterns = self.pattern_generator.generate_ma_patterns()
        elif indicator.upper() == 'EMA':
            patterns = self.pattern_generator.generate_ema_patterns()
        else:
            raise ValueError(f"不支持的指标: {indicator}")

        results = {
            'indicator': indicator,
            'total_patterns': len(patterns),
            'successful_patterns': 0,
            'failed_patterns': 0,
            'pattern_results': {},
            'summary': {},
            'timestamp': datetime.now().isoformat()
        }

        total_score = 0.0

        for pattern_name, pattern_data in patterns.items():
            print(f"  测试形态: {pattern_name}")

            # 显示数据基本信息
            print(f"    数据点数: {len(pattern_data)}")
            print(f"    价格范围: {pattern_data['close'].min():.2f} - {pattern_data['close'].max():.2f}")

            # 运行增强验证
            result = self.real_indicator_analysis(indicator, pattern_name, pattern_data)

            # 显示结果
            status = "✅ 成功" if result['is_successful'] else "❌ 失败"
            print(f"    结果: {status} (匹配分: {result['match_score']:.3f})")
            print(f"    预期形态: {', '.join(result['expected_patterns'])}")
            print(f"    识别形态: {', '.join(result['identified_patterns'])}")

            if 'detected_patterns_detail' in result:
                detected_detail = result['detected_patterns_detail']
                detected_true = [k for k, v in detected_detail.items() if v]
                print(f"    技术分析: {', '.join(detected_true) if detected_true else '无形态检测到'}")

            print(f"    价格趋势: {result['price_trend']}")

            if 'error' in result:
                print(f"    错误: {result['error']}")

            print()

            # 统计结果
            if result['is_successful']:
                results['successful_patterns'] += 1
            else:
                results['failed_patterns'] += 1

            total_score += result['match_score']
            results['pattern_results'][pattern_name] = result

        # 计算总体统计
        results['success_rate'] = results['successful_patterns'] / results['total_patterns']
        results['average_score'] = total_score / results['total_patterns']

        # 生成摘要
        results['summary'] = {
            'success_rate': f"{results['success_rate']:.2%}",
            'average_score': f"{results['average_score']:.3f}",
            'recommendation': self._get_recommendation_Enhanced_Demo_Validator(results['success_rate'])
        }

        return results

    def _get_recommendation_Enhanced_Demo_Validator(self, success_rate: float) -> str:
        """根据成功率生成建议"""
        if success_rate >= 0.9:
            return "指标形态识别表现优秀，已达到生产环境标准"
        elif success_rate >= 0.8:
            return "指标形态识别表现良好，接近生产环境要求"
        elif success_rate >= 0.6:
            return "指标形态识别表现一般，需要进一步优化"
        elif success_rate >= 0.4:
            return "指标形态识别表现较差，需要重点改进算法"
        else:
            return "指标形态识别表现很差，需要全面重新设计"

    def save_results_Validator(self, results: dict, output_file: str = None):
        """保存结果到文件"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"enhanced_validation_results_{results['indicator']}_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        print(f"增强验证结果已保存到: {output_file}")
        return output_file


def main_enhanceddemovalidator():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版单个指标反向验证演示')
    parser.add_argument('indicator', choices=['RSI', 'MACD', 'KDJ', 'BOLL', 'MA', 'EMA'],
                       help='要测试的指标名称')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--save-results', action='store_true',
                       help='保存详细结果到JSON文件')

    args = parser.parse_args()

    print("=" * 60)
    print("增强版单个指标反向验证演示")
    print("使用真实技术指标计算和形态识别算法")
    print("=" * 60)
    print(f"测试指标: {args.indicator}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 创建增强验证器并运行测试
    validator = Enhanced_indicator_validator()

    try:
        results = validator.validate_single_indicator_Validator(args.indicator)

        # 显示总结
        print("=" * 60)
        print("增强验证测试总结")
        print("=" * 60)
        print(f"指标: {results['indicator']}")
        print(f"总形态数: {results['total_patterns']}")
        print(f"成功识别: {results['successful_patterns']}")
        print(f"识别失败: {results['failed_patterns']}")
        print(f"成功率: {results['summary']['success_rate']}")
        print(f"平均匹配分: {results['summary']['average_score']}")
        print(f"建议: {results['summary']['recommendation']}")

        # 保存结果
        if args.save_results:
            output_file = validator.save_results_Validator(results, args.output)
            print(f"\n详细结果已保存到: {output_file}")

        # 返回退出码
        if results['success_rate'] >= 0.8:
            print("\n✅ 增强验证测试结果优秀")
            return 0
        elif results['success_rate'] >= 0.6:
            print("\n⚠️ 增强验证测试结果良好")
            return 0
        else:
            print("\n❌ 增强验证测试结果需要改进")
            return 1

    except Exception as e:
        print(f"❌ 增强验证测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main_enhanceddemovalidator()
    sys.exit(exit_code)