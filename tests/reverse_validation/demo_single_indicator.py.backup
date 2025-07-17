#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
单个指标反向验证演示脚本

展示如何使用反向验证框架对单个技术指标进行形态识别验证
这是一个简化的演示版本，不依赖复杂的系统环境
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


class Simplified_indicator_validator:
    """简化的指标验证器"""

    def __init__(self):
        """初始化验证器"""
        self.pattern_generator = Pattern_data_generator()

        # 简化的预期形态映射
        self.expected_patterns = {
            'RSI_OVERBOUGHT': ['超买', '高位', 'overbought'],
            'RSI_OVERSOLD': ['超卖', '低位', 'oversold'],
            'RSI_GOLDEN_CROSS': ['金叉', '上升', '突破'],
            'RSI_DEATH_CROSS': ['死叉', '下降', '跌破'],
            'RSI_DIVERGENCE': ['背离', 'divergence'],

            'MACD_GOLDEN_CROSS': ['金叉', 'golden', '上穿'],
            'MACD_DEATH_CROSS': ['死叉', 'death', '下穿'],
            'MACD_ABOVE_ZERO_GOLDEN': ['零轴上', '强势'],
            'MACD_BELOW_ZERO_DEATH': ['零轴下', '弱势'],
            'MACD_HISTOGRAM_DIVERGENCE': ['背离', '柱状图'],

            'KDJ_GOLDEN_CROSS': ['金叉', 'K线上穿'],
            'KDJ_DEATH_CROSS': ['死叉', 'K线下穿'],
            'KDJ_OVERBOUGHT': ['超买', '高位'],
            'KDJ_OVERSOLD': ['超卖', '低位'],
            'KDJ_BLUNT': ['钝化', '高位钝化', '低位钝化']
        }

    def simulate_indicator_analysis(self, indicator: str, pattern_name: str, pattern_data) -> dict:
        """
        模拟指标分析过程

        在实际环境中，这里会调用真实的技术指标分析器
        为了演示目的，我们使用简化的模拟逻辑
        """
        # 模拟分析结果
        expected_patterns = self.expected_patterns.get(pattern_name, [])

        # 简单的模拟逻辑：基于形态名称和价格趋势判断
        close_prices = pattern_data['close'].values
        price_trend = (close_prices[-1] - close_prices[0]) / close_prices[0]

        # 模拟识别出的形态
        identified_patterns = []

        if 'OVERBOUGHT' in pattern_name:
            if price_trend > 0.15:  # 价格上涨超过15%
                identified_patterns.extend(['RSI超买', '高位'])
        elif 'OVERSOLD' in pattern_name:
            if price_trend < -0.15:  # 价格下跌超过15%
                identified_patterns.extend(['RSI超卖', '低位'])
        elif 'GOLDEN_CROSS' in pattern_name:
            if price_trend > 0.05:  # 价格上涨
                identified_patterns.extend(['金叉', '上升趋势'])
        elif 'DEATH_CROSS' in pattern_name:
            if price_trend < -0.05:  # 价格下跌
                identified_patterns.extend(['死叉', '下降趋势'])
        elif 'DIVERGENCE' in pattern_name:
            # 模拟背离检测
            identified_patterns.extend(['价格背离'])

        # 计算匹配度
        match_score = self._calculate_match_score(expected_patterns, identified_patterns)

        return {
            'indicator': indicator,
            'pattern_name': pattern_name,
            'expected_patterns': expected_patterns,
            'identified_patterns': identified_patterns,
            'match_score': match_score,
            'is_successful': match_score > 0.3,  # 降低成功阈值用于演示
            'price_trend': f"{price_trend:.2%}",
            'data_points': len(pattern_data),
            'price_range': f"{close_prices.min():.2f} - {close_prices.max():.2f}",
            'analysis_method': 'simplified_simulation'
        }

    def _calculate_match_score(self, expected_patterns, identified_patterns):
        """计算匹配评分"""
        if not expected_patterns or not identified_patterns:
            return 0.0

        matches = 0
        for expected in expected_patterns:
            for identified in identified_patterns:
                if expected.lower() in identified.lower() or identified.lower() in expected.lower():
                    matches += 1
                    break

        return min(matches / len(expected_patterns), 1.0)

    def validate_single_indicator_Indicator(self, indicator: str) -> dict:
        """
        验证单个指标的所有形态

        Args:
            indicator: 指标名称 (RSI, MACD, KDJ, BOLL, MA, EMA)

        Returns:
            dict: 验证结果
        """
        print(f"开始验证指标: {indicator}")
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

            # 运行验证
            result = self.simulate_indicator_analysis(indicator, pattern_name, pattern_data)

            # 显示结果
            status = "✅ 成功" if result['is_successful'] else "❌ 失败"
            print(f"    结果: {status} (匹配分: {result['match_score']:.3f})")
            print(f"    预期形态: {', '.join(result['expected_patterns'])}")
            print(f"    识别形态: {', '.join(result['identified_patterns'])}")
            print(f"    价格趋势: {result['price_trend']}")
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
            'recommendation': self._get_recommendation(results['success_rate'])
        }

        return results

    def _get_recommendation(self, success_rate: float) -> str:
        """根据成功率生成建议"""
        if success_rate >= 0.8:
            return "指标形态识别表现优秀，系统运行正常"
        elif success_rate >= 0.6:
            return "指标形态识别表现良好，但仍有改进空间"
        elif success_rate >= 0.4:
            return "指标形态识别表现一般，建议检查算法实现"
        else:
            return "指标形态识别表现较差，需要重点优化"

    def save_results_Indicator(self, results: dict, output_file: str = None):
        """保存结果到文件"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"demo_validation_results_{results['indicator']}_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        print(f"结果已保存到: {output_file}")
        return output_file


def main_demosingleindicator():
    """主函数"""
    parser = argparse.ArgumentParser(description='单个指标反向验证演示')
    parser.add_argument('indicator', choices=['RSI', 'MACD', 'KDJ', 'BOLL', 'MA', 'EMA'],
                       help='要测试的指标名称')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--save-results', action='store_true',
                       help='保存详细结果到JSON文件')

    args = parser.parse_args()

    print("=" * 60)
    print("单个指标反向验证演示")
    print("=" * 60)
    print(f"测试指标: {args.indicator}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 创建验证器并运行测试
    validator = Simplified_indicator_validator()

    try:
        results = validator.validate_single_indicator_Indicator(args.indicator)

        # 显示总结
        print("=" * 60)
        print("测试总结")
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
            output_file = validator.save_results_Indicator(results, args.output)
            print(f"\n详细结果已保存到: {output_file}")

        # 返回退出码
        if results['success_rate'] >= 0.6:
            print("\n✅ 测试结果良好")
            return 0
        else:
            print("\n⚠️ 测试结果需要改进")
            return 1

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main_demosingleindicator()
    sys.exit(exit_code)