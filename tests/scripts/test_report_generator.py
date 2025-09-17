#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试报告生成器

生成全面的指标形态策略测试报告，包括详细的测试结果、
性能分析、问题诊断和改进建议。

核心功能：
1. 生成详细的测试报告
2. 性能分析和优化建议
3. 问题诊断和修复建议
4. 多格式输出支持

Author: AI Assistant
Date: 2025-07-19
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class TestReportGenerator:
    """测试报告生成器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化报告生成器
        
        Args:
            config: 生成配置
        """
        self.config = config or self._get_default_config()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        logger.info("📋 测试报告生成器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'output': {
                'base_dir': 'data/comprehensive_test_results',
                'generate_html': True,
                'generate_pdf': False,
                'generate_charts': True,
                'generate_csv': True
            },
            'report': {
                'include_detailed_results': True,
                'include_performance_analysis': True,
                'include_issue_analysis': True,
                'include_recommendations': True,
                'max_detail_items': 100
            },
            'charts': {
                'figure_size': (12, 8),
                'dpi': 300,
                'style': 'seaborn-v0_8',
                'color_palette': 'viridis'
            }
        }
    
    def generate_comprehensive_report(self, test_data: Dict[str, Any]) -> Dict[str, str]:
        """
        生成全面测试报告
        
        Args:
            test_data: 测试数据
            
        Returns:
            Dict[str, str]: 生成的报告文件路径
        """
        try:
            logger.info("📋 开始生成全面测试报告")
            
            # 创建输出目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(self.config['output']['base_dir'], f'report_{timestamp}')
            os.makedirs(output_dir, exist_ok=True)
            
            generated_files = {}
            
            # 生成HTML报告
            if self.config['output']['generate_html']:
                html_file = self._generate_html_report(test_data, output_dir)
                generated_files['html'] = html_file
            
            # 生成CSV数据
            if self.config['output']['generate_csv']:
                csv_files = self._generate_csv_reports(test_data, output_dir)
                generated_files.update(csv_files)
            
            # 生成图表
            if self.config['output']['generate_charts']:
                chart_files = self._generate_charts(test_data, output_dir)
                generated_files.update(chart_files)
            
            # 生成汇总报告
            summary_file = self._generate_summary_report(test_data, output_dir)
            generated_files['summary'] = summary_file
            
            logger.info(f"✅ 报告生成完成，输出目录: {output_dir}")
            return generated_files
            
        except Exception as e:
            logger.error(f"❌ 生成报告失败: {e}")
            raise
    
    def _generate_html_report(self, test_data: Dict[str, Any], output_dir: str) -> str:
        """生成HTML报告"""
        try:
            html_file = os.path.join(output_dir, 'comprehensive_test_report.html')
            
            html_content = self._build_html_content(test_data)
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"📄 HTML报告已生成: {html_file}")
            return html_file
            
        except Exception as e:
            logger.error(f"❌ 生成HTML报告失败: {e}")
            return ""
    
    def _build_html_content(self, test_data: Dict[str, Any]) -> str:
        """构建HTML内容"""
        try:
            # 获取测试数据
            test_summary = test_data.get('test_summary', {})
            coverage_metrics = test_data.get('coverage_metrics', {})
            success_metrics = test_data.get('success_metrics', {})
            performance_metrics = test_data.get('performance_metrics', {})
            issue_summary = test_data.get('issue_summary', {})
            recommendations = test_data.get('recommendations', [])
            
            html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>全面指标形态策略测试报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .success-rate {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .performance-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .issue-card {{
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .table th, .table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .table th {{
            background-color: #3498db;
            color: white;
        }}
        .table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .recommendations {{
            background-color: #e8f5e8;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 20px 0;
        }}
        .recommendations li {{
            margin: 10px 0;
        }}
        .status-success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .status-warning {{
            color: #f39c12;
            font-weight: bold;
        }}
        .status-error {{
            color: #e74c3c;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 全面指标形态策略测试报告</h1>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{test_summary.get('total_execution_time', 0):.1f}s</div>
                <div class="metric-label">总执行时间</div>
            </div>
            <div class="metric-card success-rate">
                <div class="metric-value">{success_metrics.get('success_rate', 0):.1f}%</div>
                <div class="metric-label">策略成功率</div>
            </div>
            <div class="metric-card performance-card">
                <div class="metric-value">{'✅' if performance_metrics.get('performance_compliant', False) else '❌'}</div>
                <div class="metric-label">性能达标</div>
            </div>
            <div class="metric-card issue-card">
                <div class="metric-value">{issue_summary.get('total_issues', 0)}</div>
                <div class="metric-label">发现问题</div>
            </div>
        </div>
        
        <h2>📊 测试覆盖范围</h2>
        <table class="table">
            <tr>
                <th>指标</th>
                <th>数值</th>
            </tr>
            <tr>
                <td>技术指标数量</td>
                <td>{coverage_metrics.get('total_indicators', 0)}</td>
            </tr>
            <tr>
                <td>形态模式数量</td>
                <td>{coverage_metrics.get('total_patterns', 0)}</td>
            </tr>
            <tr>
                <td>生成策略数量</td>
                <td>{coverage_metrics.get('total_strategies', 0)}</td>
            </tr>
            <tr>
                <td>测试股票数量</td>
                <td>{coverage_metrics.get('total_stocks_tested', 0)}</td>
            </tr>
        </table>
        
        <h2>🎯 成功指标</h2>
        <table class="table">
            <tr>
                <th>指标</th>
                <th>数值</th>
                <th>百分比</th>
            </tr>
            <tr>
                <td>成功策略</td>
                <td>{success_metrics.get('successful_strategies', 0)}</td>
                <td class="status-success">{success_metrics.get('success_rate', 0):.1f}%</td>
            </tr>
            <tr>
                <td>有选股策略</td>
                <td>{success_metrics.get('strategies_with_selections', 0)}</td>
                <td class="status-success">{success_metrics.get('selection_rate', 0):.1f}%</td>
            </tr>
            <tr>
                <td>闭环验证成功</td>
                <td>{success_metrics.get('closed_loop_success', 0)}</td>
                <td class="status-success">{success_metrics.get('validation_rate', 0):.1f}%</td>
            </tr>
        </table>
        
        <h2>⚡ 性能分析</h2>
        <table class="table">
            <tr>
                <th>指标</th>
                <th>数值</th>
                <th>状态</th>
            </tr>
            <tr>
                <td>总执行时间</td>
                <td>{performance_metrics.get('total_execution_time', 0):.2f}秒</td>
                <td class="{'status-success' if performance_metrics.get('performance_compliant', False) else 'status-error'}">
                    {'达标' if performance_metrics.get('performance_compliant', False) else '超时'}
                </td>
            </tr>
            <tr>
                <td>策略处理速度</td>
                <td>{performance_metrics.get('strategies_per_second', 0):.2f} 策略/秒</td>
                <td class="status-success">正常</td>
            </tr>
            <tr>
                <td>股票处理速度</td>
                <td>{performance_metrics.get('stocks_per_second', 0):.2f} 股票/秒</td>
                <td class="status-success">正常</td>
            </tr>
        </table>
        
        <h2>🔧 改进建议</h2>
        <div class="recommendations">
            <ul>
                {self._format_recommendations_html(recommendations)}
            </ul>
        </div>
        
        <h2>📝 测试总结</h2>
        <p>本次全面指标形态策略测试于 <strong>{test_summary.get('test_timestamp', 'N/A')}</strong> 完成。</p>
        <p>测试覆盖了 <strong>{coverage_metrics.get('total_indicators', 0)}</strong> 个技术指标的 
           <strong>{coverage_metrics.get('total_patterns', 0)}</strong> 个形态模式，
           生成了 <strong>{coverage_metrics.get('total_strategies', 0)}</strong> 个独立选股策略。</p>
        <p>在 <strong>{coverage_metrics.get('total_stocks_tested', 0)}</strong> 只股票的测试中，
           策略成功率达到 <strong>{success_metrics.get('success_rate', 0):.1f}%</strong>，
           闭环验证成功率为 <strong>{success_metrics.get('validation_rate', 0):.1f}%</strong>。</p>
        
        <hr style="margin: 30px 0;">
        <p style="text-align: center; color: #7f8c8d;">
            报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
</body>
</html>
            """
            
            return html_template
            
        except Exception as e:
            logger.error(f"❌ 构建HTML内容失败: {e}")
            return "<html><body><h1>报告生成失败</h1></body></html>"
    
    def _format_recommendations_html(self, recommendations: List[str]) -> str:
        """格式化建议为HTML"""
        try:
            if not recommendations:
                return "<li>暂无改进建议</li>"
            
            html_items = []
            for rec in recommendations:
                html_items.append(f"<li>{rec}</li>")
            
            return "\n".join(html_items)
            
        except Exception as e:
            logger.error(f"❌ 格式化建议失败: {e}")
            return "<li>建议格式化失败</li>"
    
    def _generate_csv_reports(self, test_data: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """生成CSV报告"""
        try:
            csv_files = {}
            
            # 策略结果CSV
            strategy_results = test_data.get('detailed_results', {}).get('test_results', {})
            if strategy_results:
                csv_file = os.path.join(output_dir, 'strategy_results.csv')
                self._export_strategy_results_csv(strategy_results, csv_file)
                csv_files['strategy_results'] = csv_file
            
            # 验证结果CSV
            validation_results = test_data.get('detailed_results', {}).get('validation_results', {})
            if validation_results:
                csv_file = os.path.join(output_dir, 'validation_results.csv')
                self._export_validation_results_csv(validation_results, csv_file)
                csv_files['validation_results'] = csv_file
            
            # 问题日志CSV
            issue_log = test_data.get('detailed_results', {}).get('issue_log', [])
            if issue_log:
                csv_file = os.path.join(output_dir, 'issue_log.csv')
                self._export_issue_log_csv(issue_log, csv_file)
                csv_files['issue_log'] = csv_file
            
            return csv_files
            
        except Exception as e:
            logger.error(f"❌ 生成CSV报告失败: {e}")
            return {}
    
    def _export_strategy_results_csv(self, strategy_results: Dict[str, Any], csv_file: str):
        """导出策略结果CSV"""
        try:
            data = []
            for strategy_id, result in strategy_results.items():
                row = {
                    'strategy_id': strategy_id,
                    'success': result.get('success', False),
                    'selected_stocks': result.get('selected_stocks', 0),
                    'execution_time': result.get('execution_time', 0),
                    'error': result.get('error', '')
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"📊 策略结果CSV已生成: {csv_file}")
            
        except Exception as e:
            logger.error(f"❌ 导出策略结果CSV失败: {e}")
    
    def _export_validation_results_csv(self, validation_results: Dict[str, Any], csv_file: str):
        """导出验证结果CSV"""
        try:
            data = []

            # 处理不同的验证结果数据结构
            if isinstance(validation_results, dict):
                for strategy_id, result in validation_results.items():
                    if isinstance(result, dict):
                        row = {
                            'strategy_id': strategy_id,
                            'validation_success': result.get('validation_success', False),
                            'success_rate': result.get('success_rate', 0),
                            'sample_count': result.get('sample_count', 0),
                            'consistent': result.get('consistent', False),
                            'issues': ', '.join(result.get('issues', [])) if isinstance(result.get('issues'), list) else str(result.get('issues', ''))
                        }
                    else:
                        # 如果result不是字典，创建默认行
                        row = {
                            'strategy_id': strategy_id,
                            'validation_success': bool(result),
                            'success_rate': 0,
                            'sample_count': 0,
                            'consistent': bool(result),
                            'issues': ''
                        }
                    data.append(row)

            if not data:
                # 如果没有数据，创建一个默认行
                data.append({
                    'strategy_id': 'no_validation_data',
                    'validation_success': False,
                    'success_rate': 0,
                    'sample_count': 0,
                    'consistent': False,
                    'issues': 'No validation data available'
                })

            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"📊 验证结果CSV已生成: {csv_file}")

        except Exception as e:
            logger.error(f"❌ 导出验证结果CSV失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
    
    def _export_issue_log_csv(self, issue_log: List[Dict[str, Any]], csv_file: str):
        """导出问题日志CSV"""
        try:
            df = pd.DataFrame(issue_log)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"📊 问题日志CSV已生成: {csv_file}")
            
        except Exception as e:
            logger.error(f"❌ 导出问题日志CSV失败: {e}")
    
    def _generate_charts(self, test_data: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """生成图表"""
        try:
            chart_files = {}
            
            # 设置图表样式
            plt.style.use('default')
            
            # 成功率分析图
            success_chart = self._create_success_rate_chart(test_data, output_dir)
            if success_chart:
                chart_files['success_rate'] = success_chart
            
            # 性能分析图
            performance_chart = self._create_performance_chart(test_data, output_dir)
            if performance_chart:
                chart_files['performance'] = performance_chart
            
            return chart_files
            
        except Exception as e:
            logger.error(f"❌ 生成图表失败: {e}")
            return {}
    
    def _create_success_rate_chart(self, test_data: Dict[str, Any], output_dir: str) -> str:
        """创建成功率分析图"""
        try:
            success_metrics = test_data.get('success_metrics', {})
            
            categories = ['策略成功率', '选股成功率', '验证成功率']
            values = [
                success_metrics.get('success_rate', 0),
                success_metrics.get('selection_rate', 0),
                success_metrics.get('validation_rate', 0)
            ]
            
            plt.figure(figsize=self.config['charts']['figure_size'])
            bars = plt.bar(categories, values, color=['#3498db', '#2ecc71', '#e74c3c'])
            
            plt.title('策略测试成功率分析', fontsize=16, fontweight='bold')
            plt.ylabel('成功率 (%)', fontsize=12)
            plt.ylim(0, 100)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            chart_file = os.path.join(output_dir, 'success_rate_analysis.png')
            plt.savefig(chart_file, dpi=self.config['charts']['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"📈 成功率分析图已生成: {chart_file}")
            return chart_file
            
        except Exception as e:
            logger.error(f"❌ 创建成功率分析图失败: {e}")
            return ""
    
    def _create_performance_chart(self, test_data: Dict[str, Any], output_dir: str) -> str:
        """创建性能分析图"""
        try:
            performance_metrics = test_data.get('performance_metrics', {})
            
            # 创建性能对比图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 执行时间对比
            execution_time = performance_metrics.get('total_execution_time', 0)
            max_time = performance_metrics.get('max_allowed_time', 300)
            
            ax1.bar(['实际执行时间', '最大允许时间'], [execution_time, max_time], 
                   color=['#e74c3c' if execution_time > max_time else '#2ecc71', '#95a5a6'])
            ax1.set_title('执行时间分析', fontweight='bold')
            ax1.set_ylabel('时间 (秒)')
            
            # 处理速度分析
            strategies_per_sec = performance_metrics.get('strategies_per_second', 0)
            stocks_per_sec = performance_metrics.get('stocks_per_second', 0)
            
            ax2.bar(['策略处理速度', '股票处理速度'], [strategies_per_sec, stocks_per_sec],
                   color=['#3498db', '#9b59b6'])
            ax2.set_title('处理速度分析', fontweight='bold')
            ax2.set_ylabel('处理速度 (个/秒)')
            
            plt.tight_layout()
            
            chart_file = os.path.join(output_dir, 'performance_analysis.png')
            plt.savefig(chart_file, dpi=self.config['charts']['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"📈 性能分析图已生成: {chart_file}")
            return chart_file
            
        except Exception as e:
            logger.error(f"❌ 创建性能分析图失败: {e}")
            return ""
    
    def _generate_summary_report(self, test_data: Dict[str, Any], output_dir: str) -> str:
        """生成汇总报告"""
        try:
            summary_file = os.path.join(output_dir, 'test_summary.txt')
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("全面指标形态策略测试汇总报告\n")
                f.write("=" * 60 + "\n\n")
                
                # 基本信息
                test_summary = test_data.get('test_summary', {})
                f.write(f"测试时间: {test_summary.get('test_timestamp', 'N/A')}\n")
                f.write(f"总执行时间: {test_summary.get('total_execution_time', 0):.2f}秒\n")
                f.write(f"性能达标: {'是' if test_summary.get('performance_compliant', False) else '否'}\n\n")
                
                # 覆盖范围
                coverage = test_data.get('coverage_metrics', {})
                f.write("测试覆盖范围:\n")
                f.write("-" * 30 + "\n")
                f.write(f"技术指标: {coverage.get('total_indicators', 0)} 个\n")
                f.write(f"形态模式: {coverage.get('total_patterns', 0)} 个\n")
                f.write(f"生成策略: {coverage.get('total_strategies', 0)} 个\n")
                f.write(f"测试股票: {coverage.get('total_stocks_tested', 0)} 只\n\n")
                
                # 成功指标
                success = test_data.get('success_metrics', {})
                f.write("成功指标:\n")
                f.write("-" * 30 + "\n")
                f.write(f"策略成功率: {success.get('success_rate', 0):.1f}%\n")
                f.write(f"选股成功率: {success.get('selection_rate', 0):.1f}%\n")
                f.write(f"验证成功率: {success.get('validation_rate', 0):.1f}%\n\n")
                
                # 改进建议
                recommendations = test_data.get('recommendations', [])
                if recommendations:
                    f.write("改进建议:\n")
                    f.write("-" * 30 + "\n")
                    for i, rec in enumerate(recommendations, 1):
                        f.write(f"{i}. {rec}\n")
            
            logger.info(f"📝 汇总报告已生成: {summary_file}")
            return summary_file
            
        except Exception as e:
            logger.error(f"❌ 生成汇总报告失败: {e}")
            return ""
