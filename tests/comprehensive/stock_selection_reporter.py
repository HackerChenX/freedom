#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股测试报告生成器

专门为选股测试系统设计的报告生成器，复用现有的报告框架
生成包含股票代码和日期的详细报告
"""

import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import pandas as pd

from utils.logger import getLogger
from .integration_test_reporter import IntegrationTestReporter
from .stock_selection_tester import (
    TestResults, IndicatorTestResult, PatternTestResult, 
    StockSelection, VerificationResult
)

logger = getLogger(__name__)


@dataclass
class StockSelectionTestReport:
    """选股测试报告"""
    report_id: str
    generation_time: datetime
    test_duration: float
    total_indicators: int
    total_patterns: int
    total_stocks_selected: int
    total_verifications: int
    overall_success_rate: float
    indicator_statistics: Dict[str, Any] = field(default_factory=dict)
    pattern_statistics: Dict[str, Any] = field(default_factory=dict)
    stock_statistics: Dict[str, Any] = field(default_factory=dict)
    verification_statistics: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class StockSelectionReporter:
    """选股测试报告生成器"""
    
    def __init__(self):
        """初始化选股测试报告生成器"""
        self.logger = getLogger(__name__)
        self.base_reporter = IntegrationTestReporter()
        
    def generate_report(self, test_results: TestResults) -> StockSelectionTestReport:
        """
        生成选股测试报告
        
        Args:
            test_results: 测试结果
            
        Returns:
            StockSelectionTestReport: 选股测试报告
        """
        self.logger.info("开始生成选股测试报告...")
        
        # 计算测试持续时间
        test_duration = (test_results.end_time - test_results.start_time).total_seconds()
        
        # 生成报告ID
        report_id = f"stock_selection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 计算指标统计信息
        indicator_statistics = self._calculate_indicator_statistics(test_results)
        
        # 计算形态统计信息
        pattern_statistics = self._calculate_pattern_statistics(test_results)
        
        # 计算股票统计信息
        stock_statistics = self._calculate_stock_statistics(test_results)
        
        # 计算验证统计信息
        verification_statistics = self._calculate_verification_statistics(test_results)
        
        # 计算性能指标
        performance_metrics = {
            'execution_time': test_duration,
            'average_time_per_indicator': test_duration / max(1, test_results.total_indicators_tested),
            'average_time_per_pattern': test_duration / max(1, test_results.total_patterns_tested),
            'stocks_per_second': test_results.total_stocks_selected / max(1, test_duration),
            'verifications_per_second': test_results.total_verifications_performed / max(1, test_duration)
        }
        
        # 生成建议
        recommendations = self._generate_recommendations(test_results)
        
        # 创建报告
        report = StockSelectionTestReport(
            report_id=report_id,
            generation_time=datetime.now(),
            test_duration=test_duration,
            total_indicators=test_results.total_indicators_tested,
            total_patterns=test_results.total_patterns_tested,
            total_stocks_selected=test_results.total_stocks_selected,
            total_verifications=test_results.total_verifications_performed,
            overall_success_rate=test_results.overall_success_rate,
            indicator_statistics=indicator_statistics,
            pattern_statistics=pattern_statistics,
            stock_statistics=stock_statistics,
            verification_statistics=verification_statistics,
            performance_metrics=performance_metrics,
            recommendations=recommendations
        )
        
        self.logger.info(f"选股测试报告生成完成: {report_id}")
        return report
    
    def _calculate_indicator_statistics(self, test_results: TestResults) -> Dict[str, Any]:
        """
        计算指标统计信息
        
        Args:
            test_results: 测试结果
            
        Returns:
            Dict[str, Any]: 指标统计信息
        """
        stats = {
            'total_indicators': test_results.total_indicators_tested,
            'indicators_with_patterns': 0,
            'indicators_with_stocks': 0,
            'indicators_with_verifications': 0,
            'top_indicators': [],
            'indicator_success_rates': {}
        }
        
        # 计算各项统计
        indicators_with_patterns = 0
        indicators_with_stocks = 0
        indicators_with_verifications = 0
        
        # 指标成功率
        indicator_success_rates = {}
        
        for indicator_name, indicator_result in test_results.indicator_results.items():
            if indicator_result.patterns_tested > 0:
                indicators_with_patterns += 1
            
            if indicator_result.total_stocks_selected > 0:
                indicators_with_stocks += 1
                
            if indicator_result.verification_success_rate > 0:
                indicators_with_verifications += 1
                
            indicator_success_rates[indicator_name] = indicator_result.verification_success_rate
        
        stats['indicators_with_patterns'] = indicators_with_patterns
        stats['indicators_with_stocks'] = indicators_with_stocks
        stats['indicators_with_verifications'] = indicators_with_verifications
        stats['indicator_success_rates'] = indicator_success_rates
        
        # 获取前5个最成功的指标
        top_indicators = sorted(
            indicator_success_rates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        stats['top_indicators'] = [
            {'name': name, 'success_rate': rate} 
            for name, rate in top_indicators
        ]
        
        return stats
    
    def _calculate_pattern_statistics(self, test_results: TestResults) -> Dict[str, Any]:
        """
        计算形态统计信息
        
        Args:
            test_results: 测试结果
            
        Returns:
            Dict[str, Any]: 形态统计信息
        """
        stats = {
            'total_patterns': test_results.total_patterns_tested,
            'patterns_with_stocks': 0,
            'patterns_with_verifications': 0,
            'top_patterns': [],
            'pattern_success_rates': {},
            'pattern_stock_counts': {}
        }
        
        # 收集所有形态结果
        all_patterns = {}
        patterns_with_stocks = 0
        patterns_with_verifications = 0
        pattern_success_rates = {}
        pattern_stock_counts = {}
        
        for indicator_result in test_results.indicator_results.values():
            for pattern_id, pattern_result in indicator_result.pattern_results.items():
                all_patterns[pattern_id] = pattern_result
                
                if pattern_result.stocks_selected > 0:
                    patterns_with_stocks += 1
                    pattern_stock_counts[pattern_id] = pattern_result.stocks_selected
                
                if pattern_result.verifications_attempted > 0:
                    patterns_with_verifications += 1
                    pattern_success_rates[pattern_id] = pattern_result.success_rate
        
        stats['patterns_with_stocks'] = patterns_with_stocks
        stats['patterns_with_verifications'] = patterns_with_verifications
        stats['pattern_success_rates'] = pattern_success_rates
        stats['pattern_stock_counts'] = pattern_stock_counts
        
        # 获取前5个最成功的形态
        top_patterns = sorted(
            pattern_success_rates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        stats['top_patterns'] = [
            {'pattern_id': pattern_id, 'success_rate': rate} 
            for pattern_id, rate in top_patterns
        ]
        
        return stats
    
    def _calculate_stock_statistics(self, test_results: TestResults) -> Dict[str, Any]:
        """
        计算股票统计信息
        
        Args:
            test_results: 测试结果
            
        Returns:
            Dict[str, Any]: 股票统计信息
        """
        stats = {
            'total_stocks_selected': test_results.total_stocks_selected,
            'unique_stocks': 0,
            'stocks_by_date': {},
            'most_selected_stocks': []
        }
        
        # 收集所有选中的股票
        all_stocks = {}
        stock_dates = {}
        stock_counts = {}
        
        for indicator_result in test_results.indicator_results.values():
            for pattern_result in indicator_result.pattern_results.values():
                for stock in pattern_result.selected_stocks:
                    stock_key = stock.stock_code
                    
                    if stock_key not in all_stocks:
                        all_stocks[stock_key] = {
                            'code': stock.stock_code,
                            'name': stock.stock_name,
                            'dates': set()
                        }
                    
                    all_stocks[stock_key]['dates'].add(stock.date)
                    
                    # 统计日期
                    if stock.date not in stock_dates:
                        stock_dates[stock.date] = 0
                    stock_dates[stock.date] += 1
                    
                    # 统计股票出现次数
                    if stock_key not in stock_counts:
                        stock_counts[stock_key] = 0
                    stock_counts[stock_key] += 1
        
        stats['unique_stocks'] = len(all_stocks)
        stats['stocks_by_date'] = stock_dates
        
        # 获取前10个最常被选中的股票
        most_selected = sorted(
            stock_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        stats['most_selected_stocks'] = [
            {
                'code': code, 
                'name': all_stocks[code]['name'], 
                'count': count,
                'dates': list(all_stocks[code]['dates'])
            } 
            for code, count in most_selected
        ]
        
        return stats
    
    def _calculate_verification_statistics(self, test_results: TestResults) -> Dict[str, Any]:
        """
        计算验证统计信息
        
        Args:
            test_results: 测试结果
            
        Returns:
            Dict[str, Any]: 验证统计信息
        """
        stats = {
            'total_verifications': test_results.total_verifications_performed,
            'successful_verifications': 0,
            'verification_success_rate': test_results.overall_success_rate,
            'verification_by_confidence': {
                'high': 0,    # 0.8-1.0
                'medium': 0,  # 0.5-0.8
                'low': 0      # 0.0-0.5
            }
        }
        
        # 收集所有验证结果
        successful_verifications = 0
        high_confidence = 0
        medium_confidence = 0
        low_confidence = 0
        
        for indicator_result in test_results.indicator_results.values():
            for pattern_result in indicator_result.pattern_results.values():
                for verification in pattern_result.verification_results:
                    if verification.pattern_match:
                        successful_verifications += 1
                    
                    # 按置信度分类
                    if verification.confidence_score >= 0.8:
                        high_confidence += 1
                    elif verification.confidence_score >= 0.5:
                        medium_confidence += 1
                    else:
                        low_confidence += 1
        
        stats['successful_verifications'] = successful_verifications
        stats['verification_by_confidence'] = {
            'high': high_confidence,
            'medium': medium_confidence,
            'low': low_confidence
        }
        
        return stats
    
    def _generate_recommendations(self, test_results: TestResults) -> List[str]:
        """
        生成建议
        
        Args:
            test_results: 测试结果
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        # 基于总体成功率的建议
        if test_results.overall_success_rate < 0.5:
            recommendations.append("总体验证成功率较低，建议检查买点分析与选股逻辑的一致性")
        elif test_results.overall_success_rate < 0.7:
            recommendations.append("总体验证成功率一般，可以进一步优化形态检测算法")
        else:
            recommendations.append("总体验证成功率良好，选股系统与买点分析系统匹配度高")
        
        # 基于指标覆盖的建议
        indicators_with_no_stocks = 0
        for indicator_result in test_results.indicator_results.values():
            if indicator_result.total_stocks_selected == 0:
                indicators_with_no_stocks += 1
        
        if indicators_with_no_stocks > 0:
            recommendations.append(f"有 {indicators_with_no_stocks} 个指标未选出任何股票，建议检查这些指标的形态定义")
        
        # 基于性能的建议
        test_duration = (test_results.end_time - test_results.start_time).total_seconds()
        if test_duration > 240:  # 超过4分钟
            recommendations.append("测试执行时间接近超时限制，建议优化数据库查询和并行处理")
        
        # 基于验证结果的建议
        low_confidence_count = 0
        for indicator_result in test_results.indicator_results.values():
            for pattern_result in indicator_result.pattern_results.values():
                for verification in pattern_result.verification_results:
                    if verification.confidence_score < 0.5:
                        low_confidence_count += 1
        
        if low_confidence_count > test_results.total_verifications_performed * 0.3:
            recommendations.append("低置信度验证结果较多，建议优化买点分析的形态匹配逻辑")
        
        return recommendations
    
    def export_json_report(self, report: StockSelectionTestReport, output_path: str) -> str:
        """
        导出JSON格式报告
        
        Args:
            report: 选股测试报告
            output_path: 输出路径
            
        Returns:
            str: 输出文件路径
        """
        # 转换为可序列化的字典
        report_dict = asdict(report)
        
        # 处理datetime对象
        report_dict['generation_time'] = report.generation_time.isoformat()
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        output_file = os.path.join(output_path, f"{report.report_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"JSON报告已保存到: {output_file}")
        return output_file
    
    def export_csv_report(self, test_results: TestResults, output_path: str) -> Dict[str, str]:
        """
        导出CSV格式报告
        
        Args:
            test_results: 测试结果
            output_path: 输出路径
            
        Returns:
            Dict[str, str]: 输出文件路径字典
        """
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        report_id = f"stock_selection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_files = {}
        
        # 1. 导出指标结果
        indicator_data = []
        for indicator_name, indicator_result in test_results.indicator_results.items():
            indicator_data.append({
                'indicator_name': indicator_name,
                'total_patterns': indicator_result.total_patterns,
                'patterns_tested': indicator_result.patterns_tested,
                'patterns_with_selections': indicator_result.patterns_with_selections,
                'total_stocks_selected': indicator_result.total_stocks_selected,
                'verification_success_rate': indicator_result.verification_success_rate
            })
        
        if indicator_data:
            indicator_file = os.path.join(output_path, f"{report_id}_indicators.csv")
            pd.DataFrame(indicator_data).to_csv(indicator_file, index=False)
            output_files['indicators'] = indicator_file
        
        # 2. 导出形态结果
        pattern_data = []
        for indicator_name, indicator_result in test_results.indicator_results.items():
            for pattern_id, pattern_result in indicator_result.pattern_results.items():
                pattern_data.append({
                    'indicator_name': indicator_name,
                    'pattern_id': pattern_id,
                    'pattern_name': pattern_result.pattern_name,
                    'stocks_selected': pattern_result.stocks_selected,
                    'verifications_attempted': pattern_result.verifications_attempted,
                    'verifications_successful': pattern_result.verifications_successful,
                    'success_rate': pattern_result.success_rate
                })
        
        if pattern_data:
            pattern_file = os.path.join(output_path, f"{report_id}_patterns.csv")
            pd.DataFrame(pattern_data).to_csv(pattern_file, index=False)
            output_files['patterns'] = pattern_file
        
        # 3. 导出选股结果
        stock_data = []
        for indicator_name, indicator_result in test_results.indicator_results.items():
            for pattern_id, pattern_result in indicator_result.pattern_results.items():
                for stock in pattern_result.selected_stocks:
                    stock_data.append({
                        'indicator_name': indicator_name,
                        'pattern_id': pattern_id,
                        'stock_code': stock.stock_code,
                        'stock_name': stock.stock_name,
                        'date': stock.date,
                        'confidence_score': stock.confidence_score
                    })
        
        if stock_data:
            stock_file = os.path.join(output_path, f"{report_id}_stocks.csv")
            pd.DataFrame(stock_data).to_csv(stock_file, index=False)
            output_files['stocks'] = stock_file
        
        # 4. 导出验证结果
        verification_data = []
        for indicator_name, indicator_result in test_results.indicator_results.items():
            for pattern_id, pattern_result in indicator_result.pattern_results.items():
                for verification in pattern_result.verification_results:
                    verification_data.append({
                        'indicator_name': indicator_name,
                        'pattern_id': pattern_id,
                        'stock_code': verification.stock_code,
                        'date': verification.date,
                        'expected_pattern': verification.expected_pattern,
                        'detected_patterns': ','.join(verification.detected_patterns),
                        'pattern_match': verification.pattern_match,
                        'confidence_score': verification.confidence_score
                    })
        
        if verification_data:
            verification_file = os.path.join(output_path, f"{report_id}_verifications.csv")
            pd.DataFrame(verification_data).to_csv(verification_file, index=False)
            output_files['verifications'] = verification_file
        
        self.logger.info(f"CSV报告已保存到: {output_path}")
        return output_files
    
    def export_html_report(self, report: StockSelectionTestReport, test_results: TestResults, output_path: str) -> str:
        """
        导出HTML格式报告
        
        Args:
            report: 选股测试报告
            test_results: 测试结果
            output_path: 输出路径
            
        Returns:
            str: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 生成HTML内容
        html_content = self._generate_html_content(report, test_results)
        
        output_file = os.path.join(output_path, f"{report.report_id}.html")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML报告已保存到: {output_file}")
        return output_file
    
    def _generate_html_content(self, report: StockSelectionTestReport, test_results: TestResults) -> str:
        """
        生成HTML内容
        
        Args:
            report: 选股测试报告
            test_results: 测试结果
            
        Returns:
            str: HTML内容
        """
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>选股测试报告</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 5px solid #007bff;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            flex: 1;
            min-width: 200px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .metric-card h3 {
            margin-top: 0;
            color: #007bff;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .success-rate {
            font-weight: bold;
        }
        .high {
            color: #28a745;
        }
        .medium {
            color: #ffc107;
        }
        .low {
            color: #dc3545;
        }
        .recommendations {
            background-color: #e9f7ef;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #28a745;
        }
        .recommendations h3 {
            color: #28a745;
            margin-top: 0;
        }
        .chart-container {
            height: 300px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>选股测试综合报告</h1>
            <p>报告ID: {report_id}</p>
            <p>生成时间: {generation_time}</p>
            <p>测试持续时间: {test_duration:.2f} 秒</p>
        </div>
        
        <div class="section">
            <h2>测试概览</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>指标统计</h3>
                    <p>总指标数: {total_indicators}</p>
                    <p>有形态指标: {indicators_with_patterns}</p>
                    <p>有选股指标: {indicators_with_stocks}</p>
                </div>
                <div class="metric-card">
                    <h3>形态统计</h3>
                    <p>总形态数: {total_patterns}</p>
                    <p>有选股形态: {patterns_with_stocks}</p>
                    <p>有验证形态: {patterns_with_verifications}</p>
                </div>
                <div class="metric-card">
                    <h3>选股统计</h3>
                    <p>总选股数: {total_stocks_selected}</p>
                    <p>唯一股票数: {unique_stocks}</p>
                </div>
                <div class="metric-card">
                    <h3>验证统计</h3>
                    <p>总验证数: {total_verifications}</p>
                    <p>成功验证: {successful_verifications}</p>
                    <p class="success-rate {success_rate_class}">成功率: {success_rate:.2%}</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>指标表现</h2>
            <table>
                <thead>
                    <tr>
                        <th>指标名称</th>
                        <th>形态数</th>
                        <th>选股数</th>
                        <th>验证成功率</th>
                    </tr>
                </thead>
                <tbody>
                    {indicator_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>最佳形态</h2>
            <table>
                <thead>
                    <tr>
                        <th>形态ID</th>
                        <th>选股数</th>
                        <th>验证成功率</th>
                    </tr>
                </thead>
                <tbody>
                    {pattern_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>最常被选中的股票</h2>
            <table>
                <thead>
                    <tr>
                        <th>股票代码</th>
                        <th>股票名称</th>
                        <th>选中次数</th>
                        <th>日期</th>
                    </tr>
                </thead>
                <tbody>
                    {stock_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>性能指标</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>执行时间</h3>
                    <p>总执行时间: {execution_time:.2f} 秒</p>
                    <p>平均每指标: {avg_time_per_indicator:.2f} 秒</p>
                    <p>平均每形态: {avg_time_per_pattern:.2f} 秒</p>
                </div>
                <div class="metric-card">
                    <h3>处理速度</h3>
                    <p>每秒选股数: {stocks_per_second:.2f}</p>
                    <p>每秒验证数: {verifications_per_second:.2f}</p>
                </div>
            </div>
        </div>
        
        <div class="section recommendations">
            <h2>改进建议</h2>
            <ul>
                {recommendations}
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        # 准备指标行
        indicator_rows = ""
        for indicator_name, indicator_result in test_results.indicator_results.items():
            success_rate = indicator_result.verification_success_rate
            rate_class = "high" if success_rate >= 0.7 else ("medium" if success_rate >= 0.5 else "low")
            
            indicator_rows += f"""
                <tr>
                    <td>{indicator_name}</td>
                    <td>{indicator_result.patterns_tested}</td>
                    <td>{indicator_result.total_stocks_selected}</td>
                    <td class="{rate_class}">{success_rate:.2%}</td>
                </tr>
            """
        
        # 准备形态行
        pattern_rows = ""
        top_patterns = sorted(
            [(pattern_id, result) for indicator in test_results.indicator_results.values() 
             for pattern_id, result in indicator.pattern_results.items()],
            key=lambda x: x[1].success_rate,
            reverse=True
        )[:10]  # 取前10个
        
        for pattern_id, pattern_result in top_patterns:
            success_rate = pattern_result.success_rate
            rate_class = "high" if success_rate >= 0.7 else ("medium" if success_rate >= 0.5 else "low")
            
            pattern_rows += f"""
                <tr>
                    <td>{pattern_id}</td>
                    <td>{pattern_result.stocks_selected}</td>
                    <td class="{rate_class}">{success_rate:.2%}</td>
                </tr>
            """
        
        # 准备股票行
        stock_rows = ""
        for stock_info in report.stock_statistics.get('most_selected_stocks', []):
            stock_rows += f"""
                <tr>
                    <td>{stock_info['code']}</td>
                    <td>{stock_info['name']}</td>
                    <td>{stock_info['count']}</td>
                    <td>{', '.join(stock_info['dates'][:5])}{'...' if len(stock_info['dates']) > 5 else ''}</td>
                </tr>
            """
        
        # 准备建议
        recommendations = "".join([f"<li>{rec}</li>" for rec in report.recommendations])
        
        # 确定成功率类别
        success_rate_class = "high" if report.overall_success_rate >= 0.7 else (
            "medium" if report.overall_success_rate >= 0.5 else "low"
        )
        
        # 填充模板
        return html_template.format(
            report_id=report.report_id,
            generation_time=report.generation_time.strftime('%Y-%m-%d %H:%M:%S'),
            test_duration=report.test_duration,
            total_indicators=report.total_indicators,
            indicators_with_patterns=report.indicator_statistics.get('indicators_with_patterns', 0),
            indicators_with_stocks=report.indicator_statistics.get('indicators_with_stocks', 0),
            total_patterns=report.total_patterns,
            patterns_with_stocks=report.pattern_statistics.get('patterns_with_stocks', 0),
            patterns_with_verifications=report.pattern_statistics.get('patterns_with_verifications', 0),
            total_stocks_selected=report.total_stocks_selected,
            unique_stocks=report.stock_statistics.get('unique_stocks', 0),
            total_verifications=report.total_verifications,
            successful_verifications=report.verification_statistics.get('successful_verifications', 0),
            success_rate=report.overall_success_rate,
            success_rate_class=success_rate_class,
            indicator_rows=indicator_rows,
            pattern_rows=pattern_rows,
            stock_rows=stock_rows,
            execution_time=report.performance_metrics.get('execution_time', 0),
            avg_time_per_indicator=report.performance_metrics.get('average_time_per_indicator', 0),
            avg_time_per_pattern=report.performance_metrics.get('average_time_per_pattern', 0),
            stocks_per_second=report.performance_metrics.get('stocks_per_second', 0),
            verifications_per_second=report.performance_metrics.get('verifications_per_second', 0),
            recommendations=recommendations
        )


def main():
    """测试报告生成器"""
    from datetime import datetime, timedelta
    
    # 创建模拟测试结果
    test_results = TestResults(
        test_id="test_123",
        start_time=datetime.now() - timedelta(minutes=5),
        end_time=datetime.now(),
        total_indicators_tested=10,
        total_patterns_tested=30,
        total_stocks_selected=150,
        total_verifications_performed=150,
        overall_success_rate=0.75
    )
    
    # 添加一些模拟的指标结果
    test_results.indicator_results = {
        "MA": IndicatorTestResult(
            indicator_name="MA",
            total_patterns=5,
            patterns_tested=5,
            patterns_with_selections=4,
            total_stocks_selected=40,
            verification_success_rate=0.8
        ),
        "MACD": IndicatorTestResult(
            indicator_name="MACD",
            total_patterns=3,
            patterns_tested=3,
            patterns_with_selections=3,
            total_stocks_selected=30,
            verification_success_rate=0.7
        ),
        "KDJ": IndicatorTestResult(
            indicator_name="KDJ",
            total_patterns=4,
            patterns_tested=4,
            patterns_with_selections=3,
            total_stocks_selected=25,
            verification_success_rate=0.6
        )
    }
    
    # 创建报告生成器
    reporter = StockSelectionReporter()
    
    # 生成报告
    report = reporter.generate_report(test_results)
    
    # 导出报告
    output_dir = "test_reports"
    json_file = reporter.export_json_report(report, output_dir)
    html_file = reporter.export_html_report(report, test_results, output_dir)
    csv_files = reporter.export_csv_report(test_results, output_dir)
    
    print(f"报告已生成:")
    print(f"- JSON: {json_file}")
    print(f"- HTML: {html_file}")
    print(f"- CSV: {', '.join(csv_files.values())}")


if __name__ == "__main__":
    main()