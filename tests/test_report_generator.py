"""
测试报告生成器

用于生成测试覆盖率和性能测试报告
"""

import os
import sys
import json
import time
import datetime
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import coverage
from xml.etree import ElementTree as ET

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.path_utils import get_result_dir
from utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


class TestReportGenerator:
    """测试报告生成器类"""
    
    def __init__(self, report_dir=None):
        """初始化测试报告生成器
        
        Args:
            report_dir: 测试报告输出目录，默认为data/result/test_reports
        """
        self.report_dir = report_dir or os.path.join(get_result_dir(), 'test_reports')
        os.makedirs(self.report_dir, exist_ok=True)
        
        # 测试运行时间
        self.run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 覆盖率对象
        self.cov = coverage.Coverage(
            source=['strategy', 'db', 'indicators', 'utils', 'formula'],
            omit=['*/__init__.py', '*/tests/*'],
            config_file=False
        )
    
    def start_coverage(self):
        """开始收集覆盖率数据"""
        self.cov.start()
        logger.info("开始收集覆盖率数据")
    
    def stop_coverage(self):
        """停止收集覆盖率数据"""
        self.cov.stop()
        logger.info("停止收集覆盖率数据")
    
    def generate_coverage_report(self):
        """生成覆盖率报告"""
        # HTML报告
        html_dir = os.path.join(self.report_dir, f'coverage_{self.run_timestamp}')
        self.cov.html_report(directory=html_dir)
        
        # XML报告（用于CI集成）
        xml_file = os.path.join(self.report_dir, f'coverage_{self.run_timestamp}.xml')
        self.cov.xml_report(outfile=xml_file)
        
        # 终端报告
        self.cov.report()
        
        # 解析XML报告，提取关键指标
        coverage_summary = self._parse_coverage_xml(xml_file)
        
        logger.info(f"覆盖率报告已生成: {html_dir}")
        return coverage_summary
    
    def _parse_coverage_xml(self, xml_file):
        """解析覆盖率XML报告
        
        Args:
            xml_file: XML报告文件路径
            
        Returns:
            dict: 覆盖率摘要
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 获取总覆盖率
            coverage_elem = root.find('coverage')
            total_lines = int(coverage_elem.get('lines-valid', 0))
            covered_lines = int(coverage_elem.get('lines-covered', 0))
            line_rate = float(coverage_elem.get('line-rate', 0))
            
            # 获取各包覆盖率
            packages = []
            for package in root.findall('.//package'):
                pkg_name = package.get('name')
                pkg_line_rate = float(package.get('line-rate', 0))
                packages.append({
                    'name': pkg_name,
                    'line_rate': pkg_line_rate,
                    'coverage_percent': round(pkg_line_rate * 100, 2)
                })
            
            return {
                'total_lines': total_lines,
                'covered_lines': covered_lines,
                'coverage_percent': round(line_rate * 100, 2),
                'packages': packages
            }
        except Exception as e:
            logger.error(f"解析覆盖率XML报告失败: {e}")
            return {
                'total_lines': 0,
                'covered_lines': 0,
                'coverage_percent': 0,
                'packages': []
            }
    
    def generate_performance_report(self, performance_results):
        """生成性能测试报告
        
        Args:
            performance_results: 性能测试结果列表，每项包含:
                - name: 测试名称
                - execution_time: 执行时间(秒)
                - throughput: 吞吐量(每秒处理的请求/项目)
                - test_data: 测试数据大小
                - config: 测试配置
                
        Returns:
            str: 报告文件路径
        """
        # 创建DataFrame
        df = pd.DataFrame(performance_results)
        
        # 保存CSV文件
        csv_file = os.path.join(self.report_dir, f'performance_{self.run_timestamp}.csv')
        df.to_csv(csv_file, index=False)
        
        # 生成图表
        self._generate_performance_charts(df)
        
        logger.info(f"性能测试报告已生成: {csv_file}")
        return csv_file
    
    def _generate_performance_charts(self, performance_df):
        """生成性能测试图表
        
        Args:
            performance_df: 性能测试结果DataFrame
        """
        # 创建图表目录
        charts_dir = os.path.join(self.report_dir, f'performance_charts_{self.run_timestamp}')
        os.makedirs(charts_dir, exist_ok=True)
        
        # 执行时间柱状图
        plt.figure(figsize=(12, 8))
        plt.bar(performance_df['name'], performance_df['execution_time'], color='skyblue')
        plt.title('各测试执行时间')
        plt.xlabel('测试名称')
        plt.ylabel('执行时间(秒)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'execution_time.png'))
        plt.close()
        
        # 吞吐量柱状图
        if 'throughput' in performance_df.columns:
            plt.figure(figsize=(12, 8))
            plt.bar(performance_df['name'], performance_df['throughput'], color='lightgreen')
            plt.title('各测试吞吐量')
            plt.xlabel('测试名称')
            plt.ylabel('吞吐量(每秒)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'throughput.png'))
            plt.close()
        
        # 测试数据大小与执行时间关系图
        if 'test_data' in performance_df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(performance_df['test_data'], performance_df['execution_time'])
            plt.title('测试数据大小与执行时间关系')
            plt.xlabel('测试数据大小')
            plt.ylabel('执行时间(秒)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'data_size_vs_time.png'))
            plt.close()
    
    def generate_test_summary(self, test_results, coverage_summary, performance_file=None):
        """生成测试总结报告
        
        Args:
            test_results: 测试结果（来自unittest的TestResult）
            coverage_summary: 覆盖率摘要
            performance_file: 性能测试报告文件路径
            
        Returns:
            str: 总结报告文件路径
        """
        # 创建报告数据
        summary = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_results': {
                'total': test_results.testsRun,
                'failures': len(test_results.failures),
                'errors': len(test_results.errors),
                'skipped': len(test_results.skipped),
                'success_rate': round((test_results.testsRun - len(test_results.failures) - len(test_results.errors)) / test_results.testsRun * 100, 2) if test_results.testsRun > 0 else 0
            },
            'coverage': coverage_summary,
            'performance_report': performance_file
        }
        
        # 保存JSON报告
        json_file = os.path.join(self.report_dir, f'test_summary_{self.run_timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 生成HTML报告
        html_file = os.path.join(self.report_dir, f'test_summary_{self.run_timestamp}.html')
        self._generate_html_summary(summary, html_file)
        
        logger.info(f"测试总结报告已生成: {html_file}")
        return html_file
    
    def _generate_html_summary(self, summary, output_file):
        """生成HTML总结报告
        
        Args:
            summary: 总结数据
            output_file: 输出文件路径
        """
        # 创建HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>可配置选股系统测试报告</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .summary-box {{
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .test-results {{
                    display: flex;
                    justify-content: space-between;
                    flex-wrap: wrap;
                    margin-bottom: 20px;
                }}
                .result-item {{
                    background-color: #fff;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    width: 22%;
                    text-align: center;
                    margin-bottom: 10px;
                }}
                .result-item h3 {{
                    margin-top: 0;
                }}
                .success {{
                    color: #27ae60;
                }}
                .failure {{
                    color: #e74c3c;
                }}
                .coverage-table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .coverage-table th, .coverage-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .coverage-table th {{
                    background-color: #f2f2f2;
                }}
                .coverage-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .coverage-bar {{
                    background-color: #eee;
                    height: 20px;
                    border-radius: 10px;
                    overflow: hidden;
                    margin-top: 5px;
                }}
                .coverage-value {{
                    background-color: #3498db;
                    height: 100%;
                }}
                .timestamp {{
                    text-align: right;
                    color: #7f8c8d;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>可配置选股系统测试报告</h1>
                <p class="timestamp">生成时间: {summary['timestamp']}</p>
                
                <div class="summary-box">
                    <h2>测试结果总览</h2>
                    <div class="test-results">
                        <div class="result-item">
                            <h3>总用例数</h3>
                            <p>{summary['test_results']['total']}</p>
                        </div>
                        <div class="result-item">
                            <h3>通过率</h3>
                            <p class="success">{summary['test_results']['success_rate']}%</p>
                        </div>
                        <div class="result-item">
                            <h3>失败</h3>
                            <p class="failure">{summary['test_results']['failures']}</p>
                        </div>
                        <div class="result-item">
                            <h3>错误</h3>
                            <p class="failure">{summary['test_results']['errors']}</p>
                        </div>
                    </div>
                </div>
                
                <div class="summary-box">
                    <h2>代码覆盖率</h2>
                    <div class="coverage-bar">
                        <div class="coverage-value" style="width: {summary['coverage']['coverage_percent']}%"></div>
                    </div>
                    <p>总覆盖率: {summary['coverage']['coverage_percent']}% ({summary['coverage']['covered_lines']}/{summary['coverage']['total_lines']}行)</p>
                    
                    <h3>各模块覆盖率</h3>
                    <table class="coverage-table">
                        <tr>
                            <th>模块</th>
                            <th>覆盖率</th>
                            <th>可视化</th>
                        </tr>
        """
        
        # 添加各包覆盖率
        for package in summary['coverage']['packages']:
            html_content += f"""
                        <tr>
                            <td>{package['name']}</td>
                            <td>{package['coverage_percent']}%</td>
                            <td>
                                <div class="coverage-bar">
                                    <div class="coverage-value" style="width: {package['coverage_percent']}%"></div>
                                </div>
                            </td>
                        </tr>
            """
        
        # 添加报告链接
        html_content += f"""
                    </table>
                </div>
                
                <div class="summary-box">
                    <h2>详细报告链接</h2>
                    <ul>
                        <li><a href="coverage_{self.run_timestamp}/index.html" target="_blank">详细覆盖率报告</a></li>
        """
        
        if summary['performance_report']:
            html_content += f"""
                        <li><a href="performance_charts_{self.run_timestamp}/execution_time.png" target="_blank">性能测试执行时间图表</a></li>
                        <li><a href="performance_charts_{self.run_timestamp}/throughput.png" target="_blank">性能测试吞吐量图表</a></li>
                        <li><a href="{os.path.basename(summary['performance_report'])}" target="_blank">性能测试原始数据</a></li>
            """
        
        # 关闭HTML
        html_content += """
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)


def run_tests_with_coverage():
    """运行所有测试并生成覆盖率报告"""
    # 创建报告生成器
    report_generator = TestReportGenerator()
    
    # 开始收集覆盖率
    report_generator.start_coverage()
    
    # 运行测试
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # 停止覆盖率收集
    report_generator.stop_coverage()
    
    # 生成覆盖率报告
    coverage_summary = report_generator.generate_coverage_report()
    
    # 如果有性能测试结果，生成性能报告
    # 这里示例性能测试结果，实际项目中应从性能测试中收集
    performance_results = collect_performance_results()
    performance_file = report_generator.generate_performance_report(performance_results)
    
    # 生成总结报告
    summary_file = report_generator.generate_test_summary(
        result, coverage_summary, performance_file)
    
    return summary_file


def collect_performance_results():
    """收集性能测试结果
    
    在实际项目中，这应该从性能测试运行中收集数据
    这里仅作为示例
    """
    # 模拟性能测试结果
    return [
        {
            'name': '策略执行基准测试',
            'execution_time': 5.2,
            'throughput': 19.2,
            'test_data': 100,
            'config': 'baseline'
        },
        {
            'name': '策略执行缓存测试',
            'execution_time': 1.1,
            'throughput': 90.9,
            'test_data': 100,
            'config': 'with_cache'
        },
        {
            'name': '数据管理器获取K线',
            'execution_time': 0.8,
            'throughput': 125.0,
            'test_data': 100,
            'config': 'baseline'
        },
        {
            'name': '数据管理器缓存测试',
            'execution_time': 0.1,
            'throughput': 1000.0,
            'test_data': 100,
            'config': 'with_cache'
        },
        {
            'name': '单线程策略执行',
            'execution_time': 7.5,
            'throughput': 13.3,
            'test_data': 100,
            'config': 'threads=1'
        },
        {
            'name': '多线程策略执行(4线程)',
            'execution_time': 2.3,
            'throughput': 43.5,
            'test_data': 100,
            'config': 'threads=4'
        },
        {
            'name': '多线程策略执行(8线程)',
            'execution_time': 1.5,
            'throughput': 66.7,
            'test_data': 100,
            'config': 'threads=8'
        }
    ]


if __name__ == '__main__':
    # 设置日志级别
    setup_logger(level="INFO")
    
    # 运行测试并生成报告
    summary_file = run_tests_with_coverage()
    
    print(f"\n测试总结报告已生成: {summary_file}")
    print("\n测试完成！") 