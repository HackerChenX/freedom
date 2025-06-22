#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
选股系统反向验证测试框架

通过构造符合特定形态的模拟数据来验证技术指标的形态识别准确性
实现自动化的反向验证测试流程
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.reverse_validation.pattern_data_generator import PatternDataGenerator
from indicators.complete_indicator_registry import complete_registry
from models.stock_info import StockInfo
from analysis.auto_indicator_analyzer import AutoIndicatorAnalyzer


class ReverseValidationFramework:
    """反向验证测试框架"""

    def __init__(self):
        """初始化框架"""
        self.pattern_generator = PatternDataGenerator()
        self.indicator_registry = complete_registry
        self.auto_analyzer = AutoIndicatorAnalyzer()

        # 测试结果存储
        self.test_results = {}
        self.validation_report = {}

        # 核心指标列表
        self.core_indicators = ['KDJ', 'RSI', 'MACD', 'BOLL', 'MA', 'EMA']

        # 预期形态映射（定义每个模拟形态应该被识别出的技术形态）
        self.expected_patterns = {
            'RSI_OVERBOUGHT': ['RSI超买', 'RSI_OVERBOUGHT', 'RSI高位'],
            'RSI_OVERSOLD': ['RSI超卖', 'RSI_OVERSOLD', 'RSI低位'],
            'RSI_GOLDEN_CROSS': ['RSI金叉', 'RSI上升', 'RSI突破'],
            'RSI_DEATH_CROSS': ['RSI死叉', 'RSI下降', 'RSI跌破'],
            'RSI_DIVERGENCE': ['RSI背离', 'RSI顶背离', 'RSI底背离'],

            'MACD_GOLDEN_CROSS': ['MACD金叉', 'MACD_GOLDEN_CROSS', 'DIF上穿DEA'],
            'MACD_DEATH_CROSS': ['MACD死叉', 'MACD_DEATH_CROSS', 'DIF下穿DEA'],
            'MACD_ABOVE_ZERO_GOLDEN': ['MACD零轴上金叉', 'MACD强势金叉'],
            'MACD_BELOW_ZERO_DEATH': ['MACD零轴下死叉', 'MACD弱势死叉'],
            'MACD_HISTOGRAM_DIVERGENCE': ['MACD背离', 'MACD柱状图背离'],

            'KDJ_GOLDEN_CROSS': ['KDJ金叉', 'KDJ_GOLDEN_CROSS', 'K线上穿D线'],
            'KDJ_DEATH_CROSS': ['KDJ死叉', 'KDJ_DEATH_CROSS', 'K线下穿D线'],
            'KDJ_OVERBOUGHT': ['KDJ超买', 'KDJ高位', 'KDJ_OVERBOUGHT'],
            'KDJ_OVERSOLD': ['KDJ超卖', 'KDJ低位', 'KDJ_OVERSOLD'],
            'KDJ_BLUNT': ['KDJ钝化', 'KDJ高位钝化', 'KDJ低位钝化'],

            'BOLL_UPPER_BREAKOUT': ['BOLL上轨突破', 'BOLL_UPPER_BREAKOUT', '布林上轨突破'],
            'BOLL_LOWER_BREAKOUT': ['BOLL下轨突破', 'BOLL_LOWER_BREAKOUT', '布林下轨突破'],
            'BOLL_SQUEEZE': ['BOLL收口', 'BOLL_SQUEEZE', '布林收口'],
            'BOLL_EXPANSION': ['BOLL开口', 'BOLL_EXPANSION', '布林开口'],
            'BOLL_MIDDLE_SUPPORT': ['BOLL中轨支撑', 'BOLL中轨阻力'],

            'MA_GOLDEN_CROSS': ['MA金叉', 'MA_GOLDEN_CROSS', '均线金叉'],
            'MA_DEATH_CROSS': ['MA死叉', 'MA_DEATH_CROSS', '均线死叉'],
            'MA_BULLISH_ALIGNMENT': ['MA多头排列', '均线多头排列'],
            'MA_BEARISH_ALIGNMENT': ['MA空头排列', '均线空头排列'],
            'MA_SUPPORT': ['MA支撑', '均线支撑'],

            'EMA_GOLDEN_CROSS': ['EMA金叉', 'EMA_GOLDEN_CROSS'],
            'EMA_DEATH_CROSS': ['EMA死叉', 'EMA_DEATH_CROSS'],
            'EMA_TREND_CONFIRMATION': ['EMA趋势确认'],
            'EMA_DIVERGENCE': ['EMA背离'],
            'EMA_SUPPORT_RESISTANCE': ['EMA支撑', 'EMA阻力']
        }

    def run_single_pattern_validation(self, indicator: str, pattern_name: str,
                                    pattern_data: pd.DataFrame) -> Dict[str, Any]:
        """
        运行单个形态的验证测试

        Args:
            indicator: 指标名称
            pattern_name: 形态名称
            pattern_data: 形态数据

        Returns:
            Dict[str, Any]: 验证结果
        """
        try:
            # 转换为StockInfo对象
            stock_info = StockInfo(pattern_data)

            # 运行指标分析
            analysis_result = self.auto_analyzer.analyze_stock(
                stock_code=pattern_data['code'].iloc[0],
                stock_data=stock_info,
                date=pattern_data['date'].iloc[-1]  # 使用最后一个日期
            )

            # 提取相关指标的分析结果
            indicator_results = self._extract_indicator_results(analysis_result, indicator)

            # 检查是否识别出预期形态
            expected_patterns = self.expected_patterns.get(pattern_name, [])
            identified_patterns = self._extract_identified_patterns(indicator_results)

            # 计算匹配度
            match_score = self._calculate_pattern_match_score(expected_patterns, identified_patterns)

            # 构建验证结果
            validation_result = {
                'indicator': indicator,
                'pattern_name': pattern_name,
                'data_points': len(pattern_data),
                'expected_patterns': expected_patterns,
                'identified_patterns': identified_patterns,
                'match_score': match_score,
                'is_successful': match_score > 0.5,  # 50%以上匹配认为成功
                'analysis_result': indicator_results,
                'timestamp': datetime.now().isoformat()
            }

            return validation_result

        except Exception as e:
            return {
                'indicator': indicator,
                'pattern_name': pattern_name,
                'error': str(e),
                'is_successful': False,
                'match_score': 0.0,
                'timestamp': datetime.now().isoformat()
            }

    def _extract_indicator_results(self, analysis_result: Dict[str, Any],
                                 indicator: str) -> Dict[str, Any]:
        """
        从分析结果中提取特定指标的结果

        Args:
            analysis_result: 完整分析结果
            indicator: 指标名称

        Returns:
            Dict[str, Any]: 指标分析结果
        """
        # 从分析结果中提取指标相关信息
        indicator_data = {}

        if 'indicators' in analysis_result:
            for ind_name, ind_result in analysis_result['indicators'].items():
                if indicator.upper() in ind_name.upper():
                    indicator_data[ind_name] = ind_result

        if 'patterns' in analysis_result:
            for pattern_name, pattern_result in analysis_result['patterns'].items():
                if indicator.upper() in pattern_name.upper():
                    indicator_data[f'pattern_{pattern_name}'] = pattern_result

        return indicator_data

    def _extract_identified_patterns(self, indicator_results: Dict[str, Any]) -> List[str]:
        """
        从指标结果中提取识别出的形态

        Args:
            indicator_results: 指标分析结果

        Returns:
            List[str]: 识别出的形态列表
        """
        identified_patterns = []

        for key, value in indicator_results.items():
            if isinstance(value, dict):
                # 检查是否有形态信息
                if 'patterns' in value:
                    if isinstance(value['patterns'], dict):
                        for pattern_name, pattern_value in value['patterns'].items():
                            if pattern_value:  # 如果形态为True
                                identified_patterns.append(pattern_name)
                    elif isinstance(value['patterns'], list):
                        identified_patterns.extend(value['patterns'])

                # 检查是否有信号信息
                if 'signals' in value:
                    if isinstance(value['signals'], dict):
                        for signal_name, signal_value in value['signals'].items():
                            if signal_value:  # 如果信号为True
                                identified_patterns.append(signal_name)
                    elif isinstance(value['signals'], list):
                        identified_patterns.extend(value['signals'])

        return list(set(identified_patterns))  # 去重

    def _calculate_pattern_match_score(self, expected_patterns: List[str],
                                     identified_patterns: List[str]) -> float:
        """
        计算形态匹配评分

        Args:
            expected_patterns: 预期形态列表
            identified_patterns: 识别出的形态列表

        Returns:
            float: 匹配评分（0-1）
        """
        if not expected_patterns:
            return 0.0

        if not identified_patterns:
            return 0.0

        # 计算匹配数量
        matches = 0
        for expected in expected_patterns:
            for identified in identified_patterns:
                # 模糊匹配（包含关系）
                if expected.lower() in identified.lower() or identified.lower() in expected.lower():
                    matches += 1
                    break

        # 计算匹配率
        match_score = matches / len(expected_patterns)
        return min(match_score, 1.0)  # 确保不超过1.0

    def run_batch_validation(self, indicators: List[str] = None) -> Dict[str, Any]:
        """
        运行批量验证测试

        Args:
            indicators: 要测试的指标列表，None表示测试所有核心指标

        Returns:
            Dict[str, Any]: 批量验证结果
        """
        if indicators is None:
            indicators = self.core_indicators

        batch_results = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'average_match_score': 0.0,
            'indicator_results': {},
            'summary': {},
            'timestamp': datetime.now().isoformat()
        }

        total_score = 0.0

        for indicator in indicators:
            print(f"正在测试指标: {indicator}")

            # 生成该指标的所有形态数据
            if indicator == 'RSI':
                patterns = self.pattern_generator.generate_rsi_patterns()
            elif indicator == 'MACD':
                patterns = self.pattern_generator.generate_macd_patterns()
            elif indicator == 'KDJ':
                patterns = self.pattern_generator.generate_kdj_patterns()
            elif indicator == 'BOLL':
                patterns = self.pattern_generator.generate_boll_patterns()
            elif indicator == 'MA':
                patterns = self.pattern_generator.generate_ma_patterns()
            elif indicator == 'EMA':
                patterns = self.pattern_generator.generate_ema_patterns()
            else:
                print(f"跳过未支持的指标: {indicator}")
                continue

            indicator_results = {
                'total_patterns': len(patterns),
                'successful_patterns': 0,
                'failed_patterns': 0,
                'pattern_results': {},
                'average_score': 0.0
            }

            indicator_total_score = 0.0

            for pattern_name, pattern_data in patterns.items():
                print(f"  测试形态: {pattern_name}")

                # 运行单个形态验证
                result = self.run_single_pattern_validation(indicator, pattern_name, pattern_data)

                # 统计结果
                batch_results['total_tests'] += 1
                if result['is_successful']:
                    batch_results['successful_tests'] += 1
                    indicator_results['successful_patterns'] += 1
                else:
                    batch_results['failed_tests'] += 1
                    indicator_results['failed_patterns'] += 1

                # 累计评分
                score = result.get('match_score', 0.0)
                total_score += score
                indicator_total_score += score

                # 保存详细结果
                indicator_results['pattern_results'][pattern_name] = result

            # 计算指标平均分
            if indicator_results['total_patterns'] > 0:
                indicator_results['average_score'] = indicator_total_score / indicator_results['total_patterns']

            batch_results['indicator_results'][indicator] = indicator_results

        # 计算总体平均分
        if batch_results['total_tests'] > 0:
            batch_results['average_match_score'] = total_score / batch_results['total_tests']

        # 生成摘要
        batch_results['summary'] = self._generate_batch_summary(batch_results)

        return batch_results

    def _generate_batch_summary(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成批量测试摘要

        Args:
            batch_results: 批量测试结果

        Returns:
            Dict[str, Any]: 摘要信息
        """
        summary = {
            'success_rate': 0.0,
            'total_indicators': len(batch_results['indicator_results']),
            'best_indicator': None,
            'worst_indicator': None,
            'recommendations': []
        }

        # 计算成功率
        if batch_results['total_tests'] > 0:
            summary['success_rate'] = batch_results['successful_tests'] / batch_results['total_tests']

        # 找出最佳和最差指标
        best_score = 0.0
        worst_score = 1.0

        for indicator, results in batch_results['indicator_results'].items():
            score = results['average_score']

            if score > best_score:
                best_score = score
                summary['best_indicator'] = {
                    'name': indicator,
                    'score': score,
                    'success_rate': results['successful_patterns'] / results['total_patterns'] if results['total_patterns'] > 0 else 0
                }

            if score < worst_score:
                worst_score = score
                summary['worst_indicator'] = {
                    'name': indicator,
                    'score': score,
                    'success_rate': results['successful_patterns'] / results['total_patterns'] if results['total_patterns'] > 0 else 0
                }

        # 生成建议
        if summary['success_rate'] < 0.6:
            summary['recommendations'].append("整体识别准确率较低，建议检查指标实现和形态定义")

        if summary['success_rate'] > 0.8:
            summary['recommendations'].append("整体识别准确率良好，系统运行正常")

        return summary

    def generate_detailed_report(self, batch_results: Dict[str, Any],
                               output_file: str = None) -> str:
        """
        生成详细的验证报告

        Args:
            batch_results: 批量测试结果
            output_file: 输出文件路径

        Returns:
            str: 报告内容
        """
        report_lines = []

        # 报告标题
        report_lines.append("# 选股系统反向验证测试报告")
        report_lines.append("")
        report_lines.append(f"**生成时间**: {batch_results['timestamp']}")
        report_lines.append("")

        # 总体统计
        report_lines.append("## 总体统计")
        report_lines.append("")
        report_lines.append(f"- **总测试数**: {batch_results['total_tests']}")
        report_lines.append(f"- **成功测试数**: {batch_results['successful_tests']}")
        report_lines.append(f"- **失败测试数**: {batch_results['failed_tests']}")
        report_lines.append(f"- **成功率**: {batch_results['summary']['success_rate']:.2%}")
        report_lines.append(f"- **平均匹配分**: {batch_results['average_match_score']:.3f}")
        report_lines.append("")

        # 指标详细结果
        report_lines.append("## 指标详细结果")
        report_lines.append("")

        for indicator, results in batch_results['indicator_results'].items():
            report_lines.append(f"### {indicator} 指标")
            report_lines.append("")
            report_lines.append(f"- **形态总数**: {results['total_patterns']}")
            report_lines.append(f"- **成功识别**: {results['successful_patterns']}")
            report_lines.append(f"- **识别失败**: {results['failed_patterns']}")
            report_lines.append(f"- **成功率**: {results['successful_patterns']/results['total_patterns']:.2%}")
            report_lines.append(f"- **平均分**: {results['average_score']:.3f}")
            report_lines.append("")

            # 形态详细结果
            report_lines.append("#### 形态识别详情")
            report_lines.append("")
            report_lines.append("| 形态名称 | 匹配分 | 状态 | 预期形态 | 识别形态 |")
            report_lines.append("|---------|--------|------|----------|----------|")

            for pattern_name, pattern_result in results['pattern_results'].items():
                status = "✅ 成功" if pattern_result['is_successful'] else "❌ 失败"
                expected = ", ".join(pattern_result.get('expected_patterns', []))
                identified = ", ".join(pattern_result.get('identified_patterns', []))

                report_lines.append(f"| {pattern_name} | {pattern_result.get('match_score', 0):.3f} | {status} | {expected} | {identified} |")

            report_lines.append("")

        # 最佳和最差指标
        if batch_results['summary']['best_indicator']:
            best = batch_results['summary']['best_indicator']
            report_lines.append(f"## 最佳指标: {best['name']}")
            report_lines.append(f"- 平均分: {best['score']:.3f}")
            report_lines.append(f"- 成功率: {best['success_rate']:.2%}")
            report_lines.append("")

        if batch_results['summary']['worst_indicator']:
            worst = batch_results['summary']['worst_indicator']
            report_lines.append(f"## 需要改进的指标: {worst['name']}")
            report_lines.append(f"- 平均分: {worst['score']:.3f}")
            report_lines.append(f"- 成功率: {worst['success_rate']:.2%}")
            report_lines.append("")

        # 建议
        if batch_results['summary']['recommendations']:
            report_lines.append("## 建议")
            report_lines.append("")
            for i, recommendation in enumerate(batch_results['summary']['recommendations'], 1):
                report_lines.append(f"{i}. {recommendation}")
            report_lines.append("")

        # 生成报告内容
        report_content = "\n".join(report_lines)

        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"报告已保存到: {output_file}")

        return report_content

    def save_results_to_json(self, batch_results: Dict[str, Any],
                           output_file: str) -> None:
        """
        将结果保存为JSON文件

        Args:
            batch_results: 批量测试结果
            output_file: 输出文件路径
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"结果已保存到: {output_file}")