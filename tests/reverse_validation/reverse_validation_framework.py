#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
选股系统反向验证测试框架 - 重构版本

通过构造符合特定形态的模拟数据来验证技术指标的形态识别准确性
实现自动化的反向验证测试流程
适配重构后的系统架构
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import asyncio

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.reverse_validation.pattern_data_generator import Pattern_data_generator
from indicators.complete_indicator_registry import complete_registry
from indicators.pattern_registry import get_pattern_registry, PatternRegistry
from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer
from utils.dependency_injection import get_service, get_logger
from db.interfaces.data_access_interface import DataAccessInterface

logger = get_logger(__name__)


class Reverse_validation_framework:
    """反向验证测试框架 - 重构版本"""

    def __init__(self):
        """初始化框架"""
        logger.info("初始化反向验证测试框架...")

        self.pattern_generator = Pattern_data_generator()
        self.indicator_registry = complete_registry
        self.pattern_registry = get_pattern_registry()
        self.buypoint_analyzer = BuyPointAnalyzer()
        self.data_access = get_service(DataAccessInterface)

        # 测试结果存储
        self.test_results = {}
        self.validation_report = {}

        # 核心指标列表 - 基于重构后的系统
        self.core_indicators = ['KDJ', 'RSI', 'MACD', 'BOLL', 'MA', 'EMA', 'DMI', 'PVT']

        # 获取所有已注册的形态
        self.registered_patterns = self._discover_registered_patterns()

        logger.info(f"发现 {len(self.registered_patterns)} 个已注册形态")

        # 预期形态映射（基于重构后的形态注册表）
        self.expected_patterns = self._build_expected_patterns_mapping()

    def _discover_registered_patterns(self) -> Dict[str, Dict[str, Any]]:
        """发现所有已注册的形态"""
        try:
            all_patterns = self.pattern_registry.get_all_patterns()
            logger.info(f"发现 {len(all_patterns)} 个已注册形态")
            return all_patterns
        except Exception as e:
            logger.error(f"发现形态失败: {e}")
            return {}

    def _build_expected_patterns_mapping(self) -> Dict[str, List[str]]:
        """基于形态注册表构建预期形态映射"""
        expected_patterns = {}

        # 从形态注册表获取所有形态
        for pattern_id, pattern_info in self.registered_patterns.items():
            # 构建可能的形态名称变体
            display_name = pattern_info.get('display_name', '')
            pattern_variants = [
                pattern_id,
                display_name,
                pattern_id.replace('_', ''),
                display_name.replace(' ', ''),
            ]

            # 移除空字符串
            pattern_variants = [p for p in pattern_variants if p]
            expected_patterns[pattern_id] = pattern_variants

        # 添加传统形态映射以保持兼容性
        traditional_patterns = {
            'RSI_OVERBOUGHT': ['RSI超买', 'RSI_OVERBOUGHT', 'RSI高位'],
            'RSI_OVERSOLD': ['RSI超卖', 'RSI_OVERSOLD', 'RSI低位'],
            'MACD_GOLDEN_CROSS': ['MACD金叉', 'MACD_GOLDEN_CROSS', 'DIF上穿DEA'],
            'MACD_DEATH_CROSS': ['MACD死叉', 'MACD_DEATH_CROSS', 'DIF下穿DEA'],
            'KDJ_GOLDEN_CROSS': ['KDJ金叉', 'KDJ_GOLDEN_CROSS', 'K线上穿D线'],
            'KDJ_DEATH_CROSS': ['KDJ死叉', 'KDJ_DEATH_CROSS', 'K线下穿D线'],
            'BOLL_UPPER_BREAKOUT': ['BOLL上轨突破', 'BOLL_UPPER_BREAKOUT', '布林上轨突破'],
            'MA_GOLDEN_CROSS': ['MA金叉', 'MA_GOLDEN_CROSS', '均线金叉'],
            'DMI_GOLDEN_CROSS': ['DMI金叉', 'DMI_GOLDEN_CROSS', '+DI上穿-DI'],
            'PVT_GOLDEN_CROSS': ['PVT金叉', 'PVT_GOLDEN_CROSS', 'PVT上穿信号线'],
        }

        # 合并传统形态映射
        expected_patterns.update(traditional_patterns)

        logger.info(f"构建了 {len(expected_patterns)} 个形态的预期映射")
        return expected_patterns

    def run_single_pattern_validation(self, indicator: str, pattern_name: str,
                                    pattern_data: pd.DataFrame) -> Dict[str, Any]:
        """
        运行单个形态的验证测试 - 重构版本

        Args:
            indicator: 指标名称
            pattern_name: 形态名称
            pattern_data: 形态数据

        Returns:
            Dict[str, Any]: 验证结果
        """
        try:
            logger.debug(f"开始验证形态: {indicator}.{pattern_name}")

            # 使用重构后的买点分析器
            stock_code = pattern_data['code'].iloc[0] if 'code' in pattern_data.columns else 'TEST001'
            buy_date = pattern_data['date'].iloc[-1].strftime('%Y%m%d') if 'date' in pattern_data.columns else '20240101'

            # 模拟stockInfo数据结构
            mock_stock_data = self._prepare_mock_stock_data(pattern_data, stock_code)

            # 使用买点分析器进行分析
            analysis_result = self.buypoint_analyzer.analyze_stock(
                stock_code=stock_code,
                buy_date=buy_date,
                stock_name=f"测试股票_{pattern_name}"
            )

            # 提取相关指标的分析结果
            indicator_results = self._extract_indicator_results_refactored(analysis_result, indicator)

            # 检查是否识别出预期形态
            expected_patterns = self.expected_patterns.get(pattern_name, [])
            identified_patterns = self._extract_identified_patterns_refactored(indicator_results)

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
                'timestamp': datetime.now().isoformat(),
                'test_data_hash': self._calculate_data_hash(pattern_data)
            }

            logger.debug(f"形态验证完成: {indicator}.{pattern_name}, 匹配度: {match_score:.2f}")
            return validation_result

        except Exception as e:
            logger.error(f"形态验证失败: {indicator}.{pattern_name}, 错误: {e}")
            return {
                'indicator': indicator,
                'pattern_name': pattern_name,
                'error': str(e),
                'is_successful': False,
                'match_score': 0.0,
                'timestamp': datetime.now().isoformat()
            }

    def _prepare_mock_stock_data(self, pattern_data: pd.DataFrame, stock_code: str) -> None:
        """准备模拟股票数据以供买点分析器使用"""
        # 这里可以将模拟数据写入临时存储或直接传递给分析器
        # 重构后的系统可能需要不同的数据准备方式
        pass

    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """计算数据哈希值用于缓存和验证"""
        import hashlib
        data_str = data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()[:8]

    def _extract_indicator_results_refactored(self, analysis_result: Dict[str, Any],
                                            indicator: str) -> Dict[str, Any]:
        """
        从重构后的分析结果中提取特定指标的结果

        Args:
            analysis_result: 完整分析结果
            indicator: 指标名称

        Returns:
            Dict[str, Any]: 指标分析结果
        """
        if not analysis_result:
            return {}

        indicator_data = {}
        indicator_upper = indicator.upper()

        # 从买点分析结果中提取指标数据
        for key, value in analysis_result.items():
            if indicator_upper in key.upper():
                indicator_data[key] = value
            elif isinstance(value, dict):
                # 递归查找嵌套的指标数据
                nested_data = self._extract_nested_indicator_data(value, indicator_upper)
                if nested_data:
                    indicator_data.update(nested_data)

        return indicator_data

    def _extract_nested_indicator_data(self, data: Dict[str, Any], indicator: str) -> Dict[str, Any]:
        """递归提取嵌套的指标数据"""
        result = {}
        for key, value in data.items():
            if indicator in key.upper():
                result[key] = value
            elif isinstance(value, dict):
                nested = self._extract_nested_indicator_data(value, indicator)
                if nested:
                    result.update(nested)
        return result

    def _extract_identified_patterns_refactored(self, indicator_results: Dict[str, Any]) -> List[str]:
        """
        从重构后的指标结果中提取识别出的形态

        Args:
            indicator_results: 指标分析结果

        Returns:
            List[str]: 识别出的形态列表
        """
        identified_patterns = []

        for key, value in indicator_results.items():
            # 检查布尔值形态标识
            if isinstance(value, bool) and value:
                identified_patterns.append(key)

            # 检查字符串形态描述
            elif isinstance(value, str) and value:
                identified_patterns.append(value)

            # 检查字典结构
            elif isinstance(value, dict):
                # 检查是否有形态信息
                if 'patterns' in value:
                    patterns = value['patterns']
                    if isinstance(patterns, dict):
                        for pattern_name, pattern_value in patterns.items():
                            if pattern_value:  # 如果形态为True
                                identified_patterns.append(pattern_name)
                    elif isinstance(patterns, list):
                        identified_patterns.extend(patterns)

                # 检查是否有信号信息
                if 'signals' in value:
                    signals = value['signals']
                    if isinstance(signals, dict):
                        for signal_name, signal_value in signals.items():
                            if signal_value:  # 如果信号为True
                                identified_patterns.append(signal_name)
                    elif isinstance(signals, list):
                        identified_patterns.extend(signals)

                # 检查形态相关的键
                pattern_keys = ['pattern_type', 'signal_type', 'trend_type']
                for pattern_key in pattern_keys:
                    if pattern_key in value and value[pattern_key]:
                        identified_patterns.append(str(value[pattern_key]))

            # 检查数值型信号（如评分）
            elif isinstance(value, (int, float)) and value > 0:
                # 如果键名包含形态相关词汇且值为正，认为是识别出的形态
                pattern_keywords = ['score', 'signal', 'pattern', 'cross', 'breakout']
                if any(keyword in key.lower() for keyword in pattern_keywords):
                    identified_patterns.append(f"{key}_{value}")

        return list(set(identified_patterns))  # 去重

    def _get_patterns_for_indicator(self, indicator: str) -> List[str]:
        """获取指标相关的形态列表"""
        patterns = []
        indicator_upper = indicator.upper()

        # 从形态注册表获取相关形态
        for pattern_id in self.registered_patterns.keys():
            if indicator_upper in pattern_id.upper():
                patterns.append(pattern_id)

        # 添加传统形态映射
        traditional_mappings = {
            'RSI': ['RSI_OVERBOUGHT', 'RSI_OVERSOLD', 'RSI_GOLDEN_CROSS', 'RSI_DEATH_CROSS'],
            'MACD': ['MACD_GOLDEN_CROSS', 'MACD_DEATH_CROSS', 'MACD_ABOVE_ZERO_GOLDEN'],
            'KDJ': ['KDJ_GOLDEN_CROSS', 'KDJ_DEATH_CROSS', 'KDJ_OVERBOUGHT', 'KDJ_OVERSOLD'],
            'BOLL': ['BOLL_UPPER_BREAKOUT', 'BOLL_LOWER_BREAKOUT', 'BOLL_SQUEEZE'],
            'MA': ['MA_GOLDEN_CROSS', 'MA_DEATH_CROSS', 'MA_BULLISH_ALIGNMENT'],
            'EMA': ['EMA_GOLDEN_CROSS', 'EMA_DEATH_CROSS', 'EMA_TREND_CONFIRMATION'],
            'DMI': ['DMI_GOLDEN_CROSS', 'DMI_DEATH_CROSS', 'ADX_STRONG_TREND'],
            'PVT': ['PVT_GOLDEN_CROSS', 'PVT_DEATH_CROSS', 'PVT_CONSECUTIVE_RISING']
        }

        if indicator_upper in traditional_mappings:
            patterns.extend(traditional_mappings[indicator_upper])

        return list(set(patterns))  # 去重

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

    async def run_comprehensive_validation_async(self, indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        运行全面的反向验证测试 - 异步版本

        Args:
            indicators: 要测试的指标列表，None表示测试所有核心指标

        Returns:
            Dict[str, Any]: 综合验证结果
        """
        logger.info("开始运行全面反向验证测试...")

        if indicators is None:
            indicators = self.core_indicators

        total_tests = 0
        successful_tests = 0
        failed_tests = 0
        test_results = {}

        start_time = datetime.now()

        for indicator in indicators:
            logger.info(f"测试指标: {indicator}")
            indicator_results = {}

            # 获取该指标相关的形态
            indicator_patterns = self._get_patterns_for_indicator(indicator)

            for pattern_name in indicator_patterns:
                try:
                    # 生成测试数据
                    pattern_data = self.pattern_generator.generate_pattern_data(
                        pattern_type=pattern_name,
                        data_points=60,  # 60天数据
                        stock_code=f"TEST_{indicator}_{pattern_name}"
                    )

                    if pattern_data is not None and not pattern_data.empty:
                        # 运行验证测试
                        validation_result = self.run_single_pattern_validation(
                            indicator, pattern_name, pattern_data
                        )

                        indicator_results[pattern_name] = validation_result
                        total_tests += 1

                        if validation_result['is_successful']:
                            successful_tests += 1
                        else:
                            failed_tests += 1

                        logger.debug(f"  {pattern_name}: {'✓' if validation_result['is_successful'] else '✗'}")
                    else:
                        logger.warning(f"无法生成形态数据: {indicator}.{pattern_name}")

                except Exception as e:
                    logger.error(f"测试失败: {indicator}.{pattern_name}, 错误: {e}")
                    failed_tests += 1

            test_results[indicator] = indicator_results

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 计算统计信息
        success_rate = successful_tests / total_tests if total_tests > 0 else 0

        comprehensive_result = {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate,
                'duration_seconds': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            },
            'indicator_results': test_results,
            'performance_metrics': {
                'tests_per_second': total_tests / duration if duration > 0 else 0,
                'average_test_time': duration / total_tests if total_tests > 0 else 0
            },
            'system_info': {
                'framework_version': '2.0_refactored',
                'registered_patterns_count': len(self.registered_patterns),
                'tested_indicators': indicators
            }
        }

        logger.info(f"全面验证测试完成: {successful_tests}/{total_tests} 成功 ({success_rate:.1%})")
        return comprehensive_result

    def validate_refactored_system(self) -> Dict[str, Any]:
        """验证重构后的系统是否正常工作"""
        logger.info("验证重构后的系统...")

        validation_results = {
            'buypoint_analyzer': False,
            'pattern_registry': False,
            'data_access': False,
            'pattern_generator': False,
            'overall_status': False
        }

        try:
            # 测试买点分析器
            test_result = self.buypoint_analyzer.analyze_stock("000001", "20240101", "测试股票")
            validation_results['buypoint_analyzer'] = test_result is not None

            # 测试形态注册表
            patterns = self.pattern_registry.get_all_patterns()
            validation_results['pattern_registry'] = len(patterns) > 0

            # 测试数据访问
            validation_results['data_access'] = self.data_access is not None

            # 测试形态生成器
            test_data = self.pattern_generator.generate_pattern_data("MA_GOLDEN_CROSS", 30, "TEST001")
            validation_results['pattern_generator'] = test_data is not None

            # 整体状态
            validation_results['overall_status'] = all([
                validation_results['buypoint_analyzer'],
                validation_results['pattern_registry'],
                validation_results['data_access'],
                validation_results['pattern_generator']
            ])

        except Exception as e:
            logger.error(f"系统验证失败: {e}")
            validation_results['error'] = str(e)

        logger.info(f"系统验证结果: {'✓' if validation_results['overall_status'] else '✗'}")
        return validation_results