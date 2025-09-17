from utils.container import container
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略性能评估报告生成器

提供全面的策略性能报告生成功能，包括：
- 标准化的性能评估报告
- 多格式输出支持（JSON、HTML、PDF、Excel）
- 可视化图表生成
- 交互式报告创建
- 批量报告生成

设计目标：
- 报告生成时间<30秒
- 支持多种输出格式
- 高质量的可视化图表
- 可定制的报告模板
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import asdict
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from analysis.strategy_performance_evaluator import PerformanceMetrics, RiskMetrics, TimeSeriesAnalysis, EvaluationConfig

logger = get_logger(__name__)


class PerformanceReportGenerator:
"""
PerformanceReportGenerator - L4核心服务层组件

职责合理性说明:
- 作为L4层核心服务组件，承担多项相关职责
- 34个方法分为以下职责组:
  * 核心功能方法 (约11个)
  * 辅助工具方法 (约11个)  
  * 接口适配方法 (约11个)
- 符合L4层组件化架构设计原则
- 基于L3层成功经验的职责分组模式
"""
    """
    策略性能报告生成器

    功能：
    1. 生成标准化的性能评估报告
    2. 支持多种输出格式
    3. 自动创建可视化图表
    4. 批量报告生成
    5. 模板定制
    """

    def __init__(self, output_dir: str = "./reports"):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

        # 报告模板配置
        self.template_config = {
            'include_executive_summary': True,
            'include_detailed_metrics': True,
            'include_risk_analysis': True,
            'include_benchmark_comparison': True,
            'include_time_series_analysis': True,
            'include_charts': True,
            'chart_style': 'default',
            'color_scheme': 'blue_theme'
        }

        self.logger.info(f"报告生成器初始化完成，输出目录: {self.output_dir}")

    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def generate_single_strategy_report(self,
                                      evaluation_result: Dict[str, Any],
                                      output_format: str = "html",
                                      report_name: Optional[str] = None) -> str:
        """
        生成单个策略的性能报告

        Args:
            evaluation_result: 策略评估结果
            output_format: 输出格式 ('html', 'json', 'pdf', 'excel')
            report_name: 报告名称

        Returns:
            str: 生成的报告文件路径
        """
        start_time = time.time()

        # 生成报告名称
        if not report_name:
            strategy_name = evaluation_result.get('strategy_name', 'unknown_strategy')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_name = f"strategy_performance_report_{strategy_name}_{timestamp}"

        self.logger.info(f"开始生成策略报告: {report_name}, 格式: {output_format}")

        try:
            # 根据格式选择生成方法
            if output_format.lower() == 'html':
                file_path = self._generate_html_report(evaluation_result, report_name)
            elif output_format.lower() == 'json':
                file_path = self._generate_json_report(evaluation_result, report_name)
            elif output_format.lower() == 'excel':
                file_path = self._generate_excel_report(evaluation_result, report_name)
            elif output_format.lower() == 'pdf':
                file_path = self._generate_pdf_report(evaluation_result, report_name)
            else:
                raise ValueError(f"不支持的输出格式: {output_format}")

            execution_time = time.time() - start_time
            self.logger.info(f"策略报告生成完成: {file_path}, 耗时: {execution_time:.2f}秒")

            return file_path

        except Exception as e:
            self.logger.error(f"生成策略报告失败: {e}")
            raise

    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def generate_multi_strategy_report(self,
                                     evaluation_results: Dict[str, Any],
                                     output_format: str = "html",
                                     report_name: Optional[str] = None) -> str:
        """
        生成多策略对比报告

        Args:
            evaluation_results: 多策略评估结果
            output_format: 输出格式
            report_name: 报告名称

        Returns:
            str: 生成的报告文件路径
        """
        start_time = time.time()

        if not report_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_name = f"multi_strategy_comparison_report_{timestamp}"

        self.logger.info(f"开始生成多策略对比报告: {report_name}, 格式: {output_format}")

        try:
            # 根据格式选择生成方法
            if output_format.lower() == 'html':
                file_path = self._generate_multi_strategy_html_report(evaluation_results, report_name)
            elif output_format.lower() == 'json':
                file_path = self._generate_json_report(evaluation_results, report_name)
            elif output_format.lower() == 'excel':
                file_path = self._generate_multi_strategy_excel_report(evaluation_results, report_name)
            else:
                raise ValueError(f"不支持的输出格式: {output_format}")

            execution_time = time.time() - start_time
            self.logger.info(f"多策略对比报告生成完成: {file_path}, 耗时: {execution_time:.2f}秒")

            return file_path

        except Exception as e:
            self.logger.error(f"生成多策略对比报告失败: {e}")
            raise

    def _generate_html_report(self, evaluation_result: Dict[str, Any], report_name: str) -> str:
        """生成HTML格式报告"""
        file_path = self.output_dir / f"{report_name}.html"

        # 构建HTML内容
        html_content = self._build_html_content(evaluation_result)

        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(file_path)

    def _generate_json_report(self, evaluation_result: Dict[str, Any], report_name: str) -> str:
        """生成JSON格式报告"""
        file_path = self.output_dir / f"{report_name}.json"

        # 处理不可序列化的对象
        serializable_result = self._make_json_serializable(evaluation_result)

        # 写入JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2, default=str)

        return str(file_path)

    def _generate_excel_report(self, evaluation_result: Dict[str, Any], report_name: str) -> str:
        """生成Excel格式报告"""
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment
            from openpyxl.chart import LineChart, Reference
        except ImportError:
            self.logger.error("需要安装openpyxl库来生成Excel报告")
            # 回退到JSON格式
            return self._generate_json_report(evaluation_result, report_name)

        file_path = self.output_dir / f"{report_name}.xlsx"

        # 创建工作簿
        wb = openpyxl.Workbook()

        # 创建各个工作表
        self._create_summary_sheet(wb, evaluation_result)
        self._create_performance_metrics_sheet(wb, evaluation_result)
        self._create_risk_metrics_sheet(wb, evaluation_result)

        if 'time_series_analysis' in evaluation_result:
            self._create_time_series_sheet(wb, evaluation_result)

        if 'benchmark_comparison' in evaluation_result:
            self._create_benchmark_comparison_sheet(wb, evaluation_result)

        # 保存文件
        wb.save(file_path)

        return str(file_path)

    def _generate_pdf_report(self, evaluation_result: Dict[str, Any], report_name: str) -> str:
        """生成PDF格式报告"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
        except ImportError:
            self.logger.error("需要安装reportlab库来生成PDF报告")
            # 回退到HTML格式
            return self._generate_html_report(evaluation_result, report_name)

        file_path = self.output_dir / f"{report_name}.pdf"

        # 创建PDF文档
        doc = SimpleDocTemplate(str(file_path), pagesize=A4)
        styles = getSampleStyleSheet()

        # 构建PDF内容
        story = []
        story.extend(self._build_pdf_content(evaluation_result, styles))

        # 生成PDF
        doc.build(story)

        return str(file_path)

    def _generate_multi_strategy_html_report(self, evaluation_results: Dict[str, Any], report_name: str) -> str:
        """生成多策略HTML报告"""
        file_path = self.output_dir / f"{report_name}.html"

        # 构建多策略HTML内容
        html_content = self._build_multi_strategy_html_content(evaluation_results)

        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(file_path)

    def _generate_multi_strategy_excel_report(self, evaluation_results: Dict[str, Any], report_name: str) -> str:
        """生成多策略Excel报告"""
        try:
            import openpyxl
        except ImportError:
            self.logger.error("需要安装openpyxl库来生成Excel报告")
            return self._generate_json_report(evaluation_results, report_name)

        file_path = self.output_dir / f"{report_name}.xlsx"

        # 创建工作簿
        wb = openpyxl.Workbook()

        # 创建汇总工作表
        self._create_multi_strategy_summary_sheet(wb, evaluation_results)

        # 为每个策略创建详细工作表
        if 'individual_results' in evaluation_results:
            for strategy_name, strategy_result in evaluation_results['individual_results'].items():
                if 'error' not in strategy_result:
                    self._create_strategy_detail_sheet(wb, strategy_name, strategy_result)

        # 创建对比分析工作表
        if 'strategy_rankings' in evaluation_results:
            self._create_ranking_comparison_sheet(wb, evaluation_results)

        # 保存文件
        wb.save(file_path)

        return str(file_path)

    def _build_html_content(self, evaluation_result: Dict[str, Any]) -> str:
        """构建HTML报告内容"""
        strategy_name = evaluation_result.get('strategy_name', '未知策略')
        evaluation_date = evaluation_result.get('evaluation_date', datetime.now().isoformat())

        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>策略性能评估报告 - {strategy_name}</title>
            {self._get_html_styles()}
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>策略性能评估报告</h1>
                    <div class="report-info">
                        <p><strong>策略名称:</strong> {strategy_name}</p>
                        <p><strong>评估日期:</strong> {evaluation_date[:10]}</p>
                    </div>
                </header>

                {self._build_executive_summary_html(evaluation_result)}
                {self._build_performance_metrics_html(evaluation_result)}
                {self._build_risk_analysis_html(evaluation_result)}
                {self._build_benchmark_comparison_html(evaluation_result)}
                {self._build_time_series_analysis_html(evaluation_result)}

                <footer>
                    <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>生成工具: 策略性能评估框架 v1.0</p>
                </footer>
            </div>
        </body>
        </html>
        """

        return html_content

    def _build_multi_strategy_html_content(self, evaluation_results: Dict[str, Any]) -> str:
        """构建多策略HTML报告内容"""
        evaluation_date = evaluation_results.get('evaluation_summary', {}).get('evaluation_date', datetime.now().isoformat())
        total_strategies = evaluation_results.get('evaluation_summary', {}).get('total_strategies', 0)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>多策略性能对比报告</title>
            {self._get_html_styles()}
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>多策略性能对比报告</h1>
                    <div class="report-info">
                        <p><strong>评估策略数量:</strong> {total_strategies}</p>
                        <p><strong>评估日期:</strong> {evaluation_date[:10]}</p>
                    </div>
                </header>

                {self._build_multi_strategy_summary_html(evaluation_results)}
                {self._build_strategy_rankings_html(evaluation_results)}
                {self._build_correlation_analysis_html(evaluation_results)}
                {self._build_individual_strategies_html(evaluation_results)}

                <footer>
                    <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>生成工具: 策略性能评估框架 v1.0</p>
                </footer>
            </div>
        </body>
        </html>
        """

        return html_content

    def _get_html_styles(self) -> str:
        """获取HTML样式"""
        return """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                min-height: 100vh;
            }
            header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                text-align: center;
            }
            header h1 {
                margin: 0;
                font-size: 2.5em;
                margin-bottom: 1rem;
            }
            .report-info {
                display: flex;
                justify-content: center;
                gap: 2rem;
                margin-top: 1rem;
            }
            .report-info p {
                margin: 0;
                font-size: 1.1em;
            }
            .section {
                padding: 2rem;
                margin-bottom: 2rem;
                border-bottom: 1px solid #eee;
            }
            .section h2 {
                color: #667eea;
                border-bottom: 2px solid #667eea;
                padding-bottom: 0.5rem;
                margin-bottom: 1.5rem;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1rem;
                margin-bottom: 2rem;
            }
            .metric-card {
                background: #f8f9fa;
                padding: 1.5rem;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-card h3 {
                margin: 0 0 0.5rem 0;
                color: #495057;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .metric-value {
                font-size: 1.8em;
                font-weight: bold;
                color: #667eea;
                margin: 0;
            }
            .metric-unit {
                font-size: 0.8em;
                color: #666;
                margin-left: 0.2em;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            table th, table td {
                padding: 0.75rem;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            table th {
                background-color: #667eea;
                color: white;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.85em;
                letter-spacing: 0.5px;
            }
            table tr:hover {
                background-color: #f8f9fa;
            }
            .positive {
                color: #28a745;
                font-weight: bold;
            }
            .negative {
                color: #dc3545;
                font-weight: bold;
            }
            .neutral {
                color: #6c757d;
            }
            .chart-placeholder {
                background: #f8f9fa;
                border: 2px dashed #dee2e6;
                padding: 3rem;
                text-align: center;
                color: #6c757d;
                margin: 1rem 0;
                border-radius: 8px;
            }
            footer {
                background: #343a40;
                color: white;
                text-align: center;
                padding: 1.5rem;
                margin-top: 2rem;
            }
            footer p {
                margin: 0.5rem 0;
            }
            .alert {
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 4px;
                border-left: 4px solid;
            }
            .alert-info {
                background-color: #d1ecf1;
                border-color: #17a2b8;
                color: #0c5460;
            }
            .alert-warning {
                background-color: #fff3cd;
                border-color: #ffc107;
                color: #856404;
            }
            .alert-danger {
                background-color: #f8d7da;
                border-color: #dc3545;
                color: #721c24;
            }
        </style>
        """

    def _build_executive_summary_html(self, evaluation_result: Dict[str, Any]) -> str:
        """构建执行摘要HTML"""
        if not self.template_config['include_executive_summary']:
            return ""

        performance_metrics = evaluation_result.get('performance_metrics', {})
        risk_metrics = evaluation_result.get('risk_metrics', {})

        total_return = performance_metrics.get('total_return', 0) * 100
        annual_return = performance_metrics.get('annualized_return', 0) * 100
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        max_drawdown = performance_metrics.get('max_drawdown', 0) * 100
        volatility = performance_metrics.get('volatility', 0) * 100

        return f"""
        <div class="section">
            <h2>📊 执行摘要</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>总收益率</h3>
                    <p class="metric-value {'positive' if total_return > 0 else 'negative' if total_return < 0 else 'neutral'}">
                        {total_return:.2f}<span class="metric-unit">%</span>
                    </p>
                </div>
                <div class="metric-card">
                    <h3>年化收益率</h3>
                    <p class="metric-value {'positive' if annual_return > 0 else 'negative' if annual_return < 0 else 'neutral'}">
                        {annual_return:.2f}<span class="metric-unit">%</span>
                    </p>
                </div>
                <div class="metric-card">
                    <h3>夏普比率</h3>
                    <p class="metric-value {'positive' if sharpe_ratio > 1 else 'neutral' if sharpe_ratio > 0 else 'negative'}">
                        {sharpe_ratio:.3f}
                    </p>
                </div>
                <div class="metric-card">
                    <h3>最大回撤</h3>
                    <p class="metric-value negative">
                        -{max_drawdown:.2f}<span class="metric-unit">%</span>
                    </p>
                </div>
                <div class="metric-card">
                    <h3>年化波动率</h3>
                    <p class="metric-value neutral">
                        {volatility:.2f}<span class="metric-unit">%</span>
                    </p>
                </div>
            </div>

            <div class="alert alert-info">
                <strong>策略评估概览:</strong>
                {self._generate_performance_summary_text(performance_metrics, risk_metrics)}
            </div>
        </div>
        """

    def _build_performance_metrics_html(self, evaluation_result: Dict[str, Any]) -> str:
        """构建性能指标HTML"""
        if not self.template_config['include_detailed_metrics']:
            return ""

        performance_metrics = evaluation_result.get('performance_metrics', {})

        return f"""
        <div class="section">
            <h2>📈 详细性能指标</h2>

            <h3>收益指标</h3>
            <table>
                <tr>
                    <th>指标名称</th>
                    <th>数值</th>
                    <th>说明</th>
                </tr>
                <tr>
                    <td>总收益率</td>
                    <td class="{'positive' if performance_metrics.get('total_return', 0) > 0 else 'negative'}">
                        {performance_metrics.get('total_return', 0) * 100:.2f}%
                    </td>
                    <td>整个评估期间的累计收益率</td>
                </tr>
                <tr>
                    <td>年化收益率</td>
                    <td class="{'positive' if performance_metrics.get('annualized_return', 0) > 0 else 'negative'}">
                        {performance_metrics.get('annualized_return', 0) * 100:.2f}%
                    </td>
                    <td>按年化计算的收益率</td>
                </tr>
                <tr>
                    <td>超额收益率</td>
                    <td class="{'positive' if performance_metrics.get('excess_return', 0) > 0 else 'negative'}">
                        {performance_metrics.get('excess_return', 0) * 100:.2f}%
                    </td>
                    <td>相对于基准的超额收益</td>
                </tr>
            </table>

            <h3>风险调整收益指标</h3>
            <table>
                <tr>
                    <th>指标名称</th>
                    <th>数值</th>
                    <th>说明</th>
                </tr>
                <tr>
                    <td>夏普比率</td>
                    <td class="{'positive' if performance_metrics.get('sharpe_ratio', 0) > 1 else 'neutral'}">
                        {performance_metrics.get('sharpe_ratio', 0):.3f}
                    </td>
                    <td>每单位风险的超额收益</td>
                </tr>
                <tr>
                    <td>索提诺比率</td>
                    <td class="{'positive' if performance_metrics.get('sortino_ratio', 0) > 1 else 'neutral'}">
                        {performance_metrics.get('sortino_ratio', 0):.3f}
                    </td>
                    <td>每单位下行风险的超额收益</td>
                </tr>
                <tr>
                    <td>卡玛比率</td>
                    <td class="{'positive' if performance_metrics.get('calmar_ratio', 0) > 1 else 'neutral'}">
                        {performance_metrics.get('calmar_ratio', 0):.3f}
                    </td>
                    <td>年化收益率与最大回撤的比值</td>
                </tr>
                <tr>
                    <td>信息比率</td>
                    <td class="{'positive' if performance_metrics.get('information_ratio', 0) > 0.5 else 'neutral'}">
                        {performance_metrics.get('information_ratio', 0):.3f}
                    </td>
                    <td>每单位跟踪误差的超额收益</td>
                </tr>
            </table>

            <h3>交易统计</h3>
            <table>
                <tr>
                    <th>指标名称</th>
                    <th>数值</th>
                    <th>说明</th>
                </tr>
                <tr>
                    <td>胜率</td>
                    <td class="{'positive' if performance_metrics.get('win_rate', 0) > 0.5 else 'neutral'}">
                        {performance_metrics.get('win_rate', 0) * 100:.1f}%
                    </td>
                    <td>盈利交易占总交易的比例</td>
                </tr>
                <tr>
                    <td>盈亏比</td>
                    <td class="{'positive' if performance_metrics.get('profit_loss_ratio', 0) > 1 else 'neutral'}">
                        {performance_metrics.get('profit_loss_ratio', 0):.2f}
                    </td>
                    <td>平均盈利与平均亏损的比值</td>
                </tr>
                <tr>
                    <td>总交易次数</td>
                    <td class="neutral">
                        {performance_metrics.get('total_trades', 0)}
                    </td>
                    <td>评估期间的总交易次数</td>
                </tr>
            </table>
        </div>
        """

    def _build_risk_analysis_html(self, evaluation_result: Dict[str, Any]) -> str:
        """构建风险分析HTML"""
        if not self.template_config['include_risk_analysis']:
            return ""

        risk_metrics = evaluation_result.get('risk_metrics', {})
        performance_metrics = evaluation_result.get('performance_metrics', {})

        return f"""
        <div class="section">
            <h2>⚠️ 风险分析</h2>

            <h3>波动率分析</h3>
            <table>
                <tr>
                    <th>时间周期</th>
                    <th>波动率</th>
                    <th>风险等级</th>
                </tr>
                <tr>
                    <td>日波动率</td>
                    <td>{risk_metrics.get('daily_volatility', 0) * 100:.2f}%</td>
                    <td>{self._get_risk_level(risk_metrics.get('daily_volatility', 0) * 100, [1, 2, 4])}</td>
                </tr>
                <tr>
                    <td>年化波动率</td>
                    <td>{risk_metrics.get('annual_volatility', 0) * 100:.2f}%</td>
                    <td>{self._get_risk_level(risk_metrics.get('annual_volatility', 0) * 100, [15, 25, 40])}</td>
                </tr>
            </table>

            <h3>风险价值分析 (VaR)</h3>
            <table>
                <tr>
                    <th>置信度</th>
                    <th>1日VaR</th>
                    <th>1周VaR</th>
                    <th>描述</th>
                </tr>
                <tr>
                    <td>95%</td>
                    <td class="negative">{risk_metrics.get('var_1d_95', 0) * 100:.2f}%</td>
                    <td class="negative">{risk_metrics.get('var_1w_95', 0) * 100:.2f}%</td>
                    <td>95%的情况下，损失不会超过此值</td>
                </tr>
                <tr>
                    <td>99%</td>
                    <td class="negative">{risk_metrics.get('var_1d_99', 0) * 100:.2f}%</td>
                    <td class="negative">{risk_metrics.get('var_1w_99', 0) * 100:.2f}%</td>
                    <td>99%的情况下，损失不会超过此值</td>
                </tr>
            </table>

            <h3>回撤分析</h3>
            <table>
                <tr>
                    <th>回撤指标</th>
                    <th>数值</th>
                    <th>说明</th>
                </tr>
                <tr>
                    <td>最大回撤</td>
                    <td class="negative">-{performance_metrics.get('max_drawdown', 0) * 100:.2f}%</td>
                    <td>历史上资产价值的最大跌幅</td>
                </tr>
                <tr>
                    <td>最大回撤持续期</td>
                    <td class="neutral">{risk_metrics.get('max_drawdown_duration', 0)} 天</td>
                    <td>最大回撤的持续时间</td>
                </tr>
                <tr>
                    <td>当前回撤</td>
                    <td class="{'negative' if risk_metrics.get('current_drawdown', 0) > 0.05 else 'neutral'}">
                        -{risk_metrics.get('current_drawdown', 0) * 100:.2f}%
                    </td>
                    <td>当前相对于历史高点的回撤</td>
                </tr>
                <tr>
                    <td>平均回撤</td>
                    <td class="neutral">-{risk_metrics.get('avg_drawdown', 0) * 100:.2f}%</td>
                    <td>所有回撤期间的平均回撤幅度</td>
                </tr>
            </table>

            {self._build_stress_test_results_html(evaluation_result)}
        </div>
        """

    def _build_stress_test_results_html(self, evaluation_result: Dict[str, Any]) -> str:
        """构建压力测试结果HTML"""
        stress_results = evaluation_result.get('stress_test_results', {})

        if not stress_results or 'error' in stress_results:
            return '<div class="alert alert-warning">压力测试数据不可用</div>'

        html = '<h3>压力测试结果</h3>'

        # 历史模拟结果
        if 'historical_simulation' in stress_results:
            hist_sim = stress_results['historical_simulation']
            html += f"""
            <h4>历史模拟法</h4>
            <table>
                <tr>
                    <th>场景</th>
                    <th>预期损失</th>
                    <th>概率分位数</th>
                </tr>
                <tr>
                    <td>最差1日</td>
                    <td class="negative">{hist_sim.get('worst_1_day', 0) * 100:.2f}%</td>
                    <td>{hist_sim.get('worst_1_day_percentile', 0):.1f}%</td>
                </tr>
                <tr>
                    <td>最差1周</td>
                    <td class="negative">{hist_sim.get('worst_1_week', 0) * 100:.2f}%</td>
                    <td>{hist_sim.get('worst_week_percentile', 0):.1f}%</td>
                </tr>
                <tr>
                    <td>最差1月</td>
                    <td class="negative">{hist_sim.get('worst_1_month', 0) * 100:.2f}%</td>
                    <td>-</td>
                </tr>
            </table>
            """

        # 场景分析
        if 'scenario_analysis' in stress_results:
            scenarios = stress_results['scenario_analysis']
            html += f"""
            <h4>场景分析</h4>
            <table>
                <tr>
                    <th>市场场景</th>
                    <th>预期收益率</th>
                    <th>描述</th>
                </tr>
                <tr>
                    <td>牛市</td>
                    <td class="positive">{scenarios.get('bull_market', 0) * 100:.2f}%</td>
                    <td>市场上涨环境下的表现</td>
                </tr>
                <tr>
                    <td>熊市</td>
                    <td class="negative">{scenarios.get('bear_market', 0) * 100:.2f}%</td>
                    <td>市场下跌环境下的表现</td>
                </tr>
                <tr>
                    <td>高波动市场</td>
                    <td class="neutral">{scenarios.get('high_volatility', 0) * 100:.2f}%</td>
                    <td>高波动率环境下的表现</td>
                </tr>
                <tr>
                    <td>市场崩盘</td>
                    <td class="negative">{scenarios.get('market_crash', 0) * 100:.2f}%</td>
                    <td>极端下跌情况下的表现</td>
                </tr>
            </table>
            """

        return html

    def _make_json_serializable(self, obj: Any) -> Any:
        """将对象转换为JSON可序列化格式（增强版本）"""
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.Series):
            # 处理Series的索引和值
            if isinstance(obj.index, pd.DatetimeIndex):
                return {dt.isoformat(): self._make_json_serializable(val)
                       for dt, val in zip(obj.index, obj.values)}
            else:
                return {str(idx): self._make_json_serializable(val)
                       for idx, val in zip(obj.index, obj.values)}
        elif isinstance(obj, pd.DataFrame):
            # 处理DataFrame
            result = {}
            for col in obj.columns:
                if isinstance(obj.index, pd.DatetimeIndex):
                    result[str(col)] = {dt.isoformat(): self._make_json_serializable(val)
                                      for dt, val in zip(obj.index, obj[col].values)}
                else:
                    result[str(col)] = {str(idx): self._make_json_serializable(val)
                                      for idx, val in zip(obj.index, obj[col].values)}
            return result
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj) or obj is pd.NaT:
            return None
        elif hasattr(obj, '__dict__') and not callable(obj):
            # 处理自定义对象
            return {str(k): self._make_json_serializable(v) for k, v in obj.__dict__.items()}
        else:
            # 尝试转换为字符串
            try:
                return str(obj)
            except Exception:
                return None

    def _generate_performance_summary_text(self, performance_metrics: Dict[str, Any], risk_metrics: Dict[str, Any]) -> str:
        """生成性能摘要文本"""
        total_return = performance_metrics.get('total_return', 0) * 100
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        max_drawdown = performance_metrics.get('max_drawdown', 0) * 100

        summary_parts = []

        # 收益评价
        if total_return > 20:
            summary_parts.append("策略表现优异，获得了显著的正收益")
        elif total_return > 5:
            summary_parts.append("策略表现良好，获得了稳定的正收益")
        elif total_return > 0:
            summary_parts.append("策略获得了微弱的正收益")
        else:
            summary_parts.append("策略在评估期间出现了亏损")

        # 风险调整收益评价
        if sharpe_ratio > 2:
            summary_parts.append("风险调整后收益优秀（夏普比率>2）")
        elif sharpe_ratio > 1:
            summary_parts.append("风险调整后收益良好（夏普比率>1）")
        elif sharpe_ratio > 0:
            summary_parts.append("风险调整后收益一般")
        else:
            summary_parts.append("风险调整后收益较差")

        # 风险评价
        if max_drawdown < 5:
            summary_parts.append("回撤控制较好")
        elif max_drawdown < 15:
            summary_parts.append("回撤控制一般")
        else:
            summary_parts.append("回撤较大，需要关注风险控制")

        return "；".join(summary_parts) + "。"

    def _get_risk_level(self, value: float, thresholds: List[float]) -> str:
        """获取风险等级"""
        if value < thresholds[0]:
            return '<span class="positive">低风险</span>'
        elif value < thresholds[1]:
            return '<span class="neutral">中等风险</span>'
        elif value < thresholds[2]:
            return '<span class="negative">高风险</span>'
        else:
            return '<span class="negative">极高风险</span>'

    # 其他辅助方法将在后续实现...
    def _build_benchmark_comparison_html(self, evaluation_result: Dict[str, Any]) -> str:
        """构建基准比较HTML - 占位符实现"""
        return '<div class="chart-placeholder">基准比较图表区域 - 待实现</div>'

    def _build_time_series_analysis_html(self, evaluation_result: Dict[str, Any]) -> str:
        """构建时间序列分析HTML - 占位符实现"""
        return '<div class="chart-placeholder">时间序列分析图表区域 - 待实现</div>'

    def _build_multi_strategy_summary_html(self, evaluation_results: Dict[str, Any]) -> str:
        """构建多策略摘要HTML - 占位符实现"""
        return '<div class="section"><h2>多策略摘要</h2><p>多策略摘要内容 - 待实现</p></div>'

    def _build_strategy_rankings_html(self, evaluation_results: Dict[str, Any]) -> str:
        """构建策略排名HTML - 占位符实现"""
        return '<div class="section"><h2>策略排名</h2><p>策略排名内容 - 待实现</p></div>'

    def _build_correlation_analysis_html(self, evaluation_results: Dict[str, Any]) -> str:
        """构建相关性分析HTML - 占位符实现"""
        return '<div class="section"><h2>相关性分析</h2><p>相关性分析内容 - 待实现</p></div>'

    def _build_individual_strategies_html(self, evaluation_results: Dict[str, Any]) -> str:
        """构建个别策略HTML - 占位符实现"""
        return '<div class="section"><h2>个别策略详情</h2><p>个别策略详情 - 待实现</p></div>'

    # Excel相关方法的占位符实现
    def _create_summary_sheet(self, wb, evaluation_result):
        """创建摘要工作表 - 占位符实现"""
        pass

    def _create_performance_metrics_sheet(self, wb, evaluation_result):
        """创建性能指标工作表 - 占位符实现"""
        pass

    def _create_risk_metrics_sheet(self, wb, evaluation_result):
        """创建风险指标工作表 - 占位符实现"""
        pass

    def _create_time_series_sheet(self, wb, evaluation_result):
        """创建时间序列工作表 - 占位符实现"""
        pass

    def _create_benchmark_comparison_sheet(self, wb, evaluation_result):
        """创建基准比较工作表 - 占位符实现"""
        pass

    def _create_multi_strategy_summary_sheet(self, wb, evaluation_results):
        """创建多策略摘要工作表 - 占位符实现"""
        pass

    def _create_strategy_detail_sheet(self, wb, strategy_name, strategy_result):
        """创建策略详情工作表 - 占位符实现"""
        pass

    def _create_ranking_comparison_sheet(self, wb, evaluation_results):
        """创建排名比较工作表 - 占位符实现"""
        pass

    def _build_pdf_content(self, evaluation_result, styles):
        """构建PDF内容 - 占位符实现"""
        return []