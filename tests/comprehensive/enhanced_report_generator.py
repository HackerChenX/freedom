#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版报告生成器

提供多种格式的测试报告生成功能，支持详细的测试结果展示
包括JSON、CSV、HTML和Markdown格式
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

from utils.logger import getLogger
from .test_result_models import (
    TestResults, IndicatorTestResult, PatternTestResult, 
    StockSelection, VerificationResult, TestResultSummary
)

logger = getLogger(__name__)


class EnhancedReportGenerator:
    """增强版报告生成器"""
    
    def __init__(self):
        """初始化报告生成器"""
        self.logger = logger
        
    def generate_reports(self, 
                        test_results: TestResults, 
                        output_dir: str = "test_reports",
                        formats: List[str] = None) -> Dict[str, str]:
        """
        生成多种格式的报告
        
        Args:
            test_results: 测试结果
            output_dir: 输出目录
            formats: 报告格式列表，默认为["json", "csv", "html", "md"]
            
        Returns:
            Dict[str, str]: 报告文件路径字典
        """
        if formats is None:
            formats = ["json", "csv", "html", "md"]
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        report_files = {}
        
        # 生成各种格式的报告
        if "json" in formats:
            json_file = self.generate_json_report(test_results, output_dir)
            report_files["json"] = json_file
        
        if "csv" in formats:
            csv_files = self.generate_csv_reports(test_results, output_dir)
            report_files.update(csv_files)
        
        if "html" in formats:
            html_file = self.generate_html_report(test_results, output_dir)
            report_files["html"] = html_file
        
        if "md" in formats:
            md_file = self.generate_markdown_report(test_results, output_dir)
            report_files["md"] = md_file
        
        self.logger.info(f"报告生成完成，保存在: {output_dir}")
        return report_files
    
    def generate_json_report(self, test_results: TestResults, output_dir: str) -> str:
        """
        生成JSON格式报告
        
        Args:
            test_results: 测试结果
            output_dir: 输出目录
            
        Returns:
            str: 报告文件路径
        """
        # 创建报告ID
        report_id = f"stock_selection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建报告文件路径
        output_file = os.path.join(output_dir, f"{report_id}.json")
        
        # 保存为JSON
        test_results.to_json(output_file)
        
        return output_file
    
    def generate_csv_reports(self, test_results: TestResults, output_dir: str) -> Dict[str, str]:
        """
        生成CSV格式报告
        
        Args:
            test_results: 测试结果
            output_dir: 输出目录
            
        Returns:
            Dict[str, str]: 报告文件路径字典
        """
        # 创建报告ID
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
                'verification_success_rate': indicator_result.verification_success_rate,
                'execution_time': indicator_result.execution_time
            })
        
        if indicator_data:
            indicator_file = os.path.join(output_dir, f"{report_id}_indicators.csv")
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
                    'success_rate': pattern_result.success_rate,
                    'execution_time': pattern_result.execution_time
                })
        
        if pattern_data:
            pattern_file = os.path.join(output_dir, f"{report_id}_patterns.csv")
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
            stock_file = os.path.join(output_dir, f"{report_id}_stocks.csv")
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
            verification_file = os.path.join(output_dir, f"{report_id}_verifications.csv")
            pd.DataFrame(verification_data).to_csv(verification_file, index=False)
            output_files['verifications'] = verification_file
        
        return output_files    

    def generate_html_report(self, test_results: TestResults, output_dir: str) -> str:
        """
        生成HTML格式报告
        
        Args:
            test_results: 测试结果
            output_dir: 输出目录
            
        Returns:
            str: 报告文件路径
        """
        # 创建报告ID
        report_id = f"stock_selection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 获取测试结果摘要
        summary = test_results.get_summary()
        
        # 获取最成功的形态
        top_patterns = test_results.get_top_patterns(10)
        
        # 获取最常被选中的股票
        top_stocks = test_results.get_top_stocks(10)
        
        # 生成HTML内容
        html_content = self._generate_html_content(test_results, summary, top_patterns, top_stocks)
        
        # 创建报告文件路径
        output_file = os.path.join(output_dir, f"{report_id}.html")
        
        # 保存HTML报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file
    
    def _generate_html_content(self, 
                             test_results: TestResults, 
                             summary: TestResultSummary,
                             top_patterns: List[Tuple[str, PatternTestResult]],
                             top_stocks: List[Tuple[str, str, int]]) -> str:
        """
        生成HTML内容
        
        Args:
            test_results: 测试结果
            summary: 测试结果摘要
            top_patterns: 最成功的形态
            top_stocks: 最常被选中的股票
            
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
            <p>报告ID: {{ test_id }}</p>
            <p>生成时间: {{ generation_time }}</p>
            <p>测试持续时间: {{ test_duration:.2f }} 秒</p>
        </div>
        
        <div class="section">
            <h2>测试概览</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>指标统计</h3>
                    <p>总指标数: {{ total_indicators }}</p>
                    <p>有形态指标: {{ indicators_with_stocks }}</p>
                </div>
                <div class="metric-card">
                    <h3>形态统计</h3>
                    <p>总形态数: {{ total_patterns }}</p>
                    <p>有选股形态: {{ patterns_with_stocks }}</p>
                </div>
                <div class="metric-card">
                    <h3>选股统计</h3>
                    <p>总选股数: {{ total_stocks_selected }}</p>
                    <p>唯一股票数: {{ unique_stocks }}</p>
                </div>
                <div class="metric-card">
                    <h3>验证统计</h3>
                    <p>总验证数: {{ total_verifications }}</p>
                    <p>成功验证: {{ successful_verifications }}</p>
                    <p class="success-rate {{ success_rate_class }}">成功率: {{ success_rate:.2%}}</p>
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
                    {{ indicator_rows }}
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
                    {{ pattern_rows }}
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
                    </tr>
                </thead>
                <tbody>
                    {{ stock_rows }}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>性能指标</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>执行时间</h3>
                    <p>总执行时间: {{ execution_time:.2f }} 秒</p>
                    <p>平均每指标: {{ avg_time_per_indicator:.2f }} 秒</p>
                    <p>平均每形态: {{ avg_time_per_pattern:.2f }} 秒</p>
                </div>
                <div class="metric-card">
                    <h3>处理速度</h3>
                    <p>每秒选股数: {{ stocks_per_second:.2f }}</p>
                    <p>每秒验证数: {{ verifications_per_second:.2f }}</p>
                </div>
            </div>
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
        for stock_code, stock_name, count in top_stocks:
            stock_rows += f"""
                <tr>
                    <td>{stock_code}</td>
                    <td>{stock_name}</td>
                    <td>{count}</td>
                </tr>
            """
        
        # 确定成功率类别
        success_rate_class = "high" if summary.overall_success_rate >= 0.7 else (
            "medium" if summary.overall_success_rate >= 0.5 else "low"
        )
        
        # 计算性能指标
        execution_time = summary.execution_time
        avg_time_per_indicator = execution_time / max(1, summary.total_indicators)
        avg_time_per_pattern = execution_time / max(1, summary.total_patterns)
        stocks_per_second = summary.total_stocks_selected / max(1, execution_time)
        verifications_per_second = summary.total_verifications / max(1, execution_time)
        
        # 填充模板
        template = Template(html_template)
        html_content = template.render(
            test_id=test_results.test_id,
            generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            test_duration=summary.execution_time,
            total_indicators=summary.total_indicators,
            indicators_with_stocks=summary.indicators_with_stocks,
            total_patterns=summary.total_patterns,
            patterns_with_stocks=summary.patterns_with_stocks,
            total_stocks_selected=summary.total_stocks_selected,
            unique_stocks=summary.unique_stocks,
            total_verifications=summary.total_verifications,
            successful_verifications=summary.successful_verifications,
            success_rate=summary.overall_success_rate,
            success_rate_class=success_rate_class,
            indicator_rows=indicator_rows,
            pattern_rows=pattern_rows,
            stock_rows=stock_rows,
            execution_time=execution_time,
            avg_time_per_indicator=avg_time_per_indicator,
            avg_time_per_pattern=avg_time_per_pattern,
            stocks_per_second=stocks_per_second,
            verifications_per_second=verifications_per_second
        )
        
        return html_content
    
    def generate_markdown_report(self, test_results: TestResults, output_dir: str) -> str:
        """
        生成Markdown格式报告
        
        Args:
            test_results: 测试结果
            output_dir: 输出目录
            
        Returns:
            str: 报告文件路径
        """
        # 创建报告ID
        report_id = f"stock_selection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 获取测试结果摘要
        summary = test_results.get_summary()
        
        # 获取最成功的形态
        top_patterns = test_results.get_top_patterns(10)
        
        # 获取最常被选中的股票
        top_stocks = test_results.get_top_stocks(10)
        
        # 生成Markdown内容
        md_content = self._generate_markdown_content(test_results, summary, top_patterns, top_stocks)
        
        # 创建报告文件路径
        output_file = os.path.join(output_dir, f"{report_id}.md")
        
        # 保存Markdown报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return output_file
    
    def _generate_markdown_content(self, 
                                 test_results: TestResults, 
                                 summary: TestResultSummary,
                                 top_patterns: List[Tuple[str, PatternTestResult]],
                                 top_stocks: List[Tuple[str, str, int]]) -> str:
        """
        生成Markdown内容
        
        Args:
            test_results: 测试结果
            summary: 测试结果摘要
            top_patterns: 最成功的形态
            top_stocks: 最常被选中的股票
            
        Returns:
            str: Markdown内容
        """
        md_content = f"""# 选股测试综合报告

## 报告信息
- **报告ID**: {test_results.test_id}
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **测试持续时间**: {summary.execution_time:.2f} 秒

## 测试概览

### 指标统计
- 总指标数: {summary.total_indicators}
- 有形态指标: {summary.indicators_with_stocks}

### 形态统计
- 总形态数: {summary.total_patterns}
- 有选股形态: {summary.patterns_with_stocks}

### 选股统计
- 总选股数: {summary.total_stocks_selected}
- 唯一股票数: {summary.unique_stocks}

### 验证统计
- 总验证数: {summary.total_verifications}
- 成功验证: {summary.successful_verifications}
- 成功率: {summary.overall_success_rate:.2%}

## 指标表现

| 指标名称 | 形态数 | 选股数 | 验证成功率 |
|---------|-------|-------|-----------|
"""
        
        # 添加指标行
        for indicator_name, indicator_result in test_results.indicator_results.items():
            md_content += f"| {indicator_name} | {indicator_result.patterns_tested} | {indicator_result.total_stocks_selected} | {indicator_result.verification_success_rate:.2%} |\n"
        
        md_content += """
## 最佳形态

| 形态ID | 选股数 | 验证成功率 |
|-------|-------|-----------|
"""
        
        # 添加形态行
        for pattern_id, pattern_result in top_patterns:
            md_content += f"| {pattern_id} | {pattern_result.stocks_selected} | {pattern_result.success_rate:.2%} |\n"
        
        md_content += """
## 最常被选中的股票

| 股票代码 | 股票名称 | 选中次数 |
|---------|---------|---------|
"""
        
        # 添加股票行
        for stock_code, stock_name, count in top_stocks:
            md_content += f"| {stock_code} | {stock_name} | {count} |\n"
        
        md_content += f"""
## 性能指标

### 执行时间
- 总执行时间: {summary.execution_time:.2f} 秒
- 平均每指标: {summary.execution_time / max(1, summary.total_indicators):.2f} 秒
- 平均每形态: {summary.execution_time / max(1, summary.total_patterns):.2f} 秒

### 处理速度
- 每秒选股数: {summary.total_stocks_selected / max(1, summary.execution_time):.2f}
- 每秒验证数: {summary.total_verifications / max(1, summary.execution_time):.2f}
"""
        
        return md_content


def main():
    """测试报告生成器"""
    from datetime import datetime, timedelta
from db.sql_manager import SQLManager, QueryType
    
    # 创建测试结果
    test_results = TestResults(
        test_id=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        start_time=datetime.now() - timedelta(minutes=5),
        end_time=datetime.now(),
        total_indicators_tested=10,
        total_patterns_tested=50,
        total_stocks_selected=200,
        total_verifications_performed=200,
        overall_success_rate=0.75
    )
    
    # 创建报告生成器
    report_generator = EnhancedReportGenerator()
    
    # 生成报告
    reports = report_generator.generate_reports(test_results, "test_reports")
    
    print(f"生成的报告: {reports}")


if __name__ == "__main__":
    main()