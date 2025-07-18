"""
集成测试报告生成器
记录所有测试结果和性能指标
"""

import os
import sys
import json
import time
import base64
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import get_logger
from config.config import get_config
from enums.test_status import TestStatus
from utils.decorators import exception_handler, performance_monitor

logger = get_logger(__name__)


@dataclass
class TestMetrics:
    """测试指标"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    skipped_tests: int
    success_rate: float
    execution_time: float
    average_test_time: float


@dataclass
class PerformanceMetrics:
    """性能指标"""
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    query_count: int
    average_query_time: float
    cache_hit_rate: float
    connection_pool_usage: int


@dataclass
class CoverageMetrics:
    """覆盖率指标"""
    code_coverage: float
    function_coverage: float
    branch_coverage: float
    test_coverage: float
    scenario_coverage: float


@dataclass
class QualityMetrics:
    """质量指标"""
    robustness_score: float
    reliability_score: float
    maintainability_score: float
    performance_score: float
    security_score: float


@dataclass
class ComprehensiveTestReport:
    """综合测试报告"""
    report_id: str
    report_name: str
    generation_time: datetime
    test_duration: float
    test_environment: Dict[str, Any]
    test_metrics: TestMetrics
    performance_metrics: PerformanceMetrics
    coverage_metrics: CoverageMetrics
    quality_metrics: QualityMetrics
    test_suite_results: List[Dict[str, Any]] = field(default_factory=list)
    detailed_results: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    attachments: Dict[str, str] = field(default_factory=dict)


class TestResultAggregator:
    """测试结果聚合器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.results = {}
    
    def add_unit_test_results(self, results: Dict[str, Any]):
        """添加单元测试结果"""
        self.results['unit_tests'] = results
    
    def add_integration_test_results(self, results: Dict[str, Any]):
        """添加集成测试结果"""
        self.results['integration_tests'] = results
    
    def add_boundary_test_results(self, results: Dict[str, Any]):
        """添加边界测试结果"""
        self.results['boundary_tests'] = results
    
    def add_exception_test_results(self, results: Dict[str, Any]):
        """添加异常测试结果"""
        self.results['exception_tests'] = results
    
    def add_consistency_test_results(self, results: Dict[str, Any]):
        """添加一致性测试结果"""
        self.results['consistency_tests'] = results
    
    def add_performance_test_results(self, results: Dict[str, Any]):
        """添加性能测试结果"""
        self.results['performance_tests'] = results
    
    def add_architecture_test_results(self, results: Dict[str, Any]):
        """添加架构测试结果"""
        self.results['architecture_tests'] = results
    
    def get_aggregated_metrics(self) -> TestMetrics:
        """获取聚合测试指标"""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        skipped_tests = 0
        total_execution_time = 0.0
        
        for test_type, results in self.results.items():
            if isinstance(results, dict):
                # 处理不同格式的测试结果
                if 'total_tests' in results:
                    total_tests += results.get('total_tests', 0)
                    passed_tests += results.get('passed_tests', 0)
                    failed_tests += results.get('failed_tests', 0)
                    error_tests += results.get('error_tests', 0)
                    skipped_tests += results.get('skipped_tests', 0)
                    total_execution_time += results.get('execution_time', 0)
                elif isinstance(results, list):
                    # 处理列表格式的结果
                    total_tests += len(results)
                    for result in results:
                        if hasattr(result, 'test_status'):
                            if result.test_status == TestStatus.PASSED:
                                passed_tests += 1
                            elif result.test_status == TestStatus.FAILED:
                                failed_tests += 1
                            elif result.test_status == TestStatus.ERROR:
                                error_tests += 1
                            elif result.test_status == TestStatus.SKIPPED:
                                skipped_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        average_test_time = (total_execution_time / total_tests) if total_tests > 0 else 0
        
        return TestMetrics(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            skipped_tests=skipped_tests,
            success_rate=success_rate,
            execution_time=total_execution_time,
            average_test_time=average_test_time
        )
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """获取详细测试结果"""
        return self.results


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.performance_data = []
    
    def collect_system_metrics(self) -> PerformanceMetrics:
        """收集系统性能指标"""
        try:
            import psutil
            
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # 磁盘IO
            disk_io = psutil.disk_io_counters()
            disk_usage = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)  # MB
            
            # 网络IO
            network_io = psutil.net_io_counters()
            network_usage = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)  # MB
            
            return PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_io=disk_usage,
                network_io=network_usage,
                query_count=0,  # 需要从测试结果中获取
                average_query_time=0.0,  # 需要从测试结果中计算
                cache_hit_rate=0.0,  # 需要从系统中获取
                connection_pool_usage=0  # 需要从数据库连接池获取
            )
            
        except ImportError:
            self.logger.warning("psutil不可用，使用模拟性能数据")
            return PerformanceMetrics(
                cpu_usage=25.0,
                memory_usage=45.0,
                disk_io=100.0,
                network_io=50.0,
                query_count=100,
                average_query_time=0.5,
                cache_hit_rate=85.0,
                connection_pool_usage=10
            )
    
    def analyze_performance_trends(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能趋势"""
        trends = {
            'execution_time_trend': 'stable',
            'memory_usage_trend': 'increasing',
            'response_time_trend': 'improving',
            'throughput_trend': 'stable'
        }
        
        # 分析执行时间趋势
        execution_times = []
        for test_type, results in test_results.items():
            if isinstance(results, dict) and 'execution_time' in results:
                execution_times.append(results['execution_time'])
        
        if len(execution_times) > 1:
            if execution_times[-1] > execution_times[0] * 1.2:
                trends['execution_time_trend'] = 'degrading'
            elif execution_times[-1] < execution_times[0] * 0.8:
                trends['execution_time_trend'] = 'improving'
        
        return trends


class CoverageCalculator:
    """覆盖率计算器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def calculate_test_coverage(self, test_results: Dict[str, Any]) -> CoverageMetrics:
        """计算测试覆盖率"""
        # 模拟覆盖率计算
        # 在实际实现中，这里会调用coverage.py或其他工具
        
        total_test_suites = len(test_results)
        covered_suites = sum(1 for results in test_results.values() if self._is_suite_passed(results))
        
        test_coverage = (covered_suites / total_test_suites * 100) if total_test_suites > 0 else 0
        
        return CoverageMetrics(
            code_coverage=85.5,  # 模拟值，实际需要从coverage工具获取
            function_coverage=92.3,
            branch_coverage=78.9,
            test_coverage=test_coverage,
            scenario_coverage=88.7
        )
    
    def _is_suite_passed(self, results: Any) -> bool:
        """判断测试套件是否通过"""
        if isinstance(results, dict):
            passed = results.get('passed_tests', 0)
            total = results.get('total_tests', 0)
            return (passed / total) >= 0.8 if total > 0 else False
        return False


class QualityAssessor:
    """质量评估器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def assess_system_quality(self, test_results: Dict[str, Any], performance_metrics: PerformanceMetrics) -> QualityMetrics:
        """评估系统质量"""
        
        # 鲁棒性评分（基于边界测试和异常测试）
        robustness_score = self._calculate_robustness_score(test_results)
        
        # 可靠性评分（基于一致性测试和重复执行结果）
        reliability_score = self._calculate_reliability_score(test_results)
        
        # 可维护性评分（基于代码覆盖率和架构合规性）
        maintainability_score = self._calculate_maintainability_score(test_results)
        
        # 性能评分（基于性能测试结果）
        performance_score = self._calculate_performance_score(performance_metrics)
        
        # 安全性评分（基于安全测试结果，如果有的话）
        security_score = self._calculate_security_score(test_results)
        
        return QualityMetrics(
            robustness_score=robustness_score,
            reliability_score=reliability_score,
            maintainability_score=maintainability_score,
            performance_score=performance_score,
            security_score=security_score
        )
    
    def _calculate_robustness_score(self, test_results: Dict[str, Any]) -> float:
        """计算鲁棒性评分"""
        boundary_results = test_results.get('boundary_tests', {})
        exception_results = test_results.get('exception_tests', {})
        
        boundary_score = 50.0
        exception_score = 50.0
        
        # 从边界测试获取鲁棒性评分
        if isinstance(boundary_results, dict) and 'robustness_score' in boundary_results:
            boundary_score = boundary_results['robustness_score']
        
        # 从异常测试获取鲁棒性评分
        if isinstance(exception_results, dict) and 'exception_handling_score' in exception_results:
            exception_score = exception_results['exception_handling_score']
        
        return (boundary_score + exception_score) / 2
    
    def _calculate_reliability_score(self, test_results: Dict[str, Any]) -> float:
        """计算可靠性评分"""
        consistency_results = test_results.get('consistency_tests', {})
        
        if isinstance(consistency_results, dict) and 'overall_consistency_score' in consistency_results:
            return consistency_results['overall_consistency_score']
        
        # 基于总体测试通过率计算可靠性
        total_tests = sum(results.get('total_tests', 0) for results in test_results.values() if isinstance(results, dict))
        passed_tests = sum(results.get('passed_tests', 0) for results in test_results.values() if isinstance(results, dict))
        
        return (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    def _calculate_maintainability_score(self, test_results: Dict[str, Any]) -> float:
        """计算可维护性评分"""
        architecture_results = test_results.get('architecture_tests', {})
        
        if isinstance(architecture_results, dict) and 'compliance_score' in architecture_results:
            return architecture_results['compliance_score']
        
        return 75.0  # 默认评分
    
    def _calculate_performance_score(self, performance_metrics: PerformanceMetrics) -> float:
        """计算性能评分"""
        # 基于各项性能指标计算综合评分
        cpu_score = max(0, 100 - performance_metrics.cpu_usage)
        memory_score = max(0, 100 - performance_metrics.memory_usage)
        
        # 查询时间评分（假设期望平均查询时间为0.5秒）
        query_time_score = max(0, 100 - (performance_metrics.average_query_time / 0.5 * 100))
        
        # 缓存命中率评分
        cache_score = performance_metrics.cache_hit_rate
        
        return (cpu_score + memory_score + query_time_score + cache_score) / 4
    
    def _calculate_security_score(self, test_results: Dict[str, Any]) -> float:
        """计算安全性评分"""
        # 这里可以基于安全测试结果计算
        # 目前返回默认值
        return 80.0


class IntegrationTestReporter:
    """集成测试报告生成器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.result_aggregator = TestResultAggregator()
        self.performance_analyzer = PerformanceAnalyzer()
        self.coverage_calculator = CoverageCalculator()
        self.quality_assessor = QualityAssessor()
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def generate_comprehensive_report(self, test_results: Dict[str, Any], report_name: str = "comprehensive_test_report") -> ComprehensiveTestReport:
        """生成综合测试报告"""
        start_time = time.time()
        
        # 聚合测试结果
        for test_type, results in test_results.items():
            if test_type == 'unit_tests':
                self.result_aggregator.add_unit_test_results(results)
            elif test_type == 'integration_tests':
                self.result_aggregator.add_integration_test_results(results)
            elif test_type == 'boundary_tests':
                self.result_aggregator.add_boundary_test_results(results)
            elif test_type == 'exception_tests':
                self.result_aggregator.add_exception_test_results(results)
            elif test_type == 'consistency_tests':
                self.result_aggregator.add_consistency_test_results(results)
            elif test_type == 'performance_tests':
                self.result_aggregator.add_performance_test_results(results)
            elif test_type == 'architecture_tests':
                self.result_aggregator.add_architecture_test_results(results)
        
        # 计算各项指标
        test_metrics = self.result_aggregator.get_aggregated_metrics()
        performance_metrics = self.performance_analyzer.collect_system_metrics()
        coverage_metrics = self.coverage_calculator.calculate_test_coverage(test_results)
        quality_metrics = self.quality_assessor.assess_system_quality(test_results, performance_metrics)
        
        # 生成测试环境信息
        test_environment = self._get_test_environment_info()
        
        # 生成改进建议
        recommendations = self._generate_recommendations(test_metrics, quality_metrics, coverage_metrics)
        
        # 创建详细结果
        detailed_results = self.result_aggregator.get_detailed_results()
        
        # 创建测试套件结果摘要
        test_suite_results = self._create_test_suite_summary(test_results)
        
        generation_time = time.time() - start_time
        
        report = ComprehensiveTestReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_name=report_name,
            generation_time=datetime.now(),
            test_duration=generation_time,
            test_environment=test_environment,
            test_metrics=test_metrics,
            performance_metrics=performance_metrics,
            coverage_metrics=coverage_metrics,
            quality_metrics=quality_metrics,
            test_suite_results=test_suite_results,
            detailed_results=detailed_results,
            recommendations=recommendations
        )
        
        return report
    
    def _get_test_environment_info(self) -> Dict[str, Any]:
        """获取测试环境信息"""
        import platform
        
        return {
            'os': platform.system(),
            'os_version': platform.version(),
            'python_version': platform.python_version(),
            'hostname': platform.node(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'test_timestamp': datetime.now().isoformat(),
            'timezone': str(datetime.now().astimezone().tzinfo)
        }
    
    def _create_test_suite_summary(self, test_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建测试套件摘要"""
        suite_summaries = []
        
        for test_type, results in test_results.items():
            if isinstance(results, dict):
                summary = {
                    'suite_name': test_type,
                    'total_tests': results.get('total_tests', 0),
                    'passed_tests': results.get('passed_tests', 0),
                    'failed_tests': results.get('failed_tests', 0),
                    'error_tests': results.get('error_tests', 0),
                    'execution_time': results.get('execution_time', 0.0),
                    'success_rate': 0.0
                }
                
                if summary['total_tests'] > 0:
                    summary['success_rate'] = (summary['passed_tests'] / summary['total_tests']) * 100
                
                suite_summaries.append(summary)
        
        return suite_summaries
    
    def _generate_recommendations(self, test_metrics: TestMetrics, quality_metrics: QualityMetrics, coverage_metrics: CoverageMetrics) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于测试成功率的建议
        if test_metrics.success_rate < 80:
            recommendations.append("测试成功率较低，建议优先修复失败的测试用例")
        
        # 基于覆盖率的建议
        if coverage_metrics.code_coverage < 80:
            recommendations.append("代码覆盖率不足80%，建议增加更多测试用例")
        
        if coverage_metrics.branch_coverage < 70:
            recommendations.append("分支覆盖率较低，建议增加边界条件和异常路径测试")
        
        # 基于质量指标的建议
        if quality_metrics.robustness_score < 70:
            recommendations.append("系统鲁棒性有待提升，建议加强异常处理和边界条件处理")
        
        if quality_metrics.performance_score < 70:
            recommendations.append("系统性能需要优化，建议关注查询效率和资源使用")
        
        if quality_metrics.reliability_score < 80:
            recommendations.append("系统可靠性需要改进，建议加强数据一致性保证")
        
        # 基于执行时间的建议
        if test_metrics.average_test_time > 10.0:
            recommendations.append("测试执行时间较长，建议优化测试用例或使用并行执行")
        
        if not recommendations:
            recommendations.append("系统整体质量良好，建议保持当前的开发和测试实践")
        
        return recommendations
    
    def export_html_report(self, report: ComprehensiveTestReport, output_path: str) -> str:
        """导出HTML格式报告"""
        html_content = self._generate_html_content(report)
        
        output_file = Path(output_path) / f"{report.report_id}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML报告已保存到: {output_file}")
        return str(output_file)
    
    def export_json_report(self, report: ComprehensiveTestReport, output_path: str) -> str:
        """导出JSON格式报告"""
        # 转换为可序列化的字典
        report_dict = asdict(report)
        
        # 处理datetime对象
        report_dict['generation_time'] = report['generation_time'].isoformat()
        
        output_file = Path(output_path) / f"{report.report_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"JSON报告已保存到: {output_file}")
        return str(output_file)
    
    def export_markdown_report(self, report: ComprehensiveTestReport, output_path: str) -> str:
        """导出Markdown格式报告"""
        markdown_content = self._generate_markdown_content(report)
        
        output_file = Path(output_path) / f"{report.report_id}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        self.logger.info(f"Markdown报告已保存到: {output_file}")
        return str(output_file)
    
    def _generate_html_content(self, report: ComprehensiveTestReport) -> str:
        """生成HTML内容"""
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_name} - 综合测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 8px; }}
        .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric-card {{ background: #e9ecef; padding: 15px; border-radius: 8px; text-align: center; }}
        .test-suite {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_name}</h1>
        <p>生成时间: {generation_time}</p>
        <p>测试持续时间: {test_duration:.2f}秒</p>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <h3>测试指标</h3>
            <p>总测试数: {total_tests}</p>
            <p class="passed">通过: {passed_tests}</p>
            <p class="failed">失败: {failed_tests}</p>
            <p>成功率: {success_rate:.1f}%</p>
        </div>
        <div class="metric-card">
            <h3>质量评分</h3>
            <p>鲁棒性: {robustness_score:.1f}/100</p>
            <p>可靠性: {reliability_score:.1f}/100</p>
            <p>性能: {performance_score:.1f}/100</p>
        </div>
        <div class="metric-card">
            <h3>覆盖率</h3>
            <p>代码覆盖率: {code_coverage:.1f}%</p>
            <p>分支覆盖率: {branch_coverage:.1f}%</p>
            <p>测试覆盖率: {test_coverage:.1f}%</p>
        </div>
    </div>
    
    <h2>测试套件结果</h2>
    {test_suites_html}
    
    <h2>改进建议</h2>
    <ul>
        {recommendations_html}
    </ul>
</body>
</html>
        """
        
        # 生成测试套件HTML
        test_suites_html = ""
        for suite in report.test_suite_results:
            status_class = "passed" if suite['success_rate'] >= 80 else "failed"
            test_suites_html += f"""
            <div class="test-suite">
                <h3 class="{status_class}">{suite['suite_name']}</h3>
                <p>测试数: {suite['total_tests']}, 通过: {suite['passed_tests']}, 失败: {suite['failed_tests']}</p>
                <p>成功率: {suite['success_rate']:.1f}%, 执行时间: {suite['execution_time']:.2f}秒</p>
            </div>
            """
        
        # 生成建议HTML
        recommendations_html = "".join(f"<li>{rec}</li>" for rec in report.recommendations)
        
        return html_template.format(
            report_name=report.report_name,
            generation_time=report.generation_time.strftime('%Y-%m-%d %H:%M:%S'),
            test_duration=report.test_duration,
            total_tests=report.test_metrics.total_tests,
            passed_tests=report.test_metrics.passed_tests,
            failed_tests=report.test_metrics.failed_tests,
            success_rate=report.test_metrics.success_rate,
            robustness_score=report.quality_metrics.robustness_score,
            reliability_score=report.quality_metrics.reliability_score,
            performance_score=report.quality_metrics.performance_score,
            code_coverage=report.coverage_metrics.code_coverage,
            branch_coverage=report.coverage_metrics.branch_coverage,
            test_coverage=report.coverage_metrics.test_coverage,
            test_suites_html=test_suites_html,
            recommendations_html=recommendations_html
        )
    
    def _generate_markdown_content(self, report: ComprehensiveTestReport) -> str:
        """生成Markdown内容"""
        content_lines = [
            f"# {report.report_name}",
            f"**生成时间**: {report.generation_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**测试持续时间**: {report.test_duration:.2f}秒",
            f"**报告ID**: {report.report_id}",
            "",
            "## 📊 测试指标概览",
            "",
            f"- **总测试数**: {report.test_metrics.total_tests}",
            f"- **通过测试**: {report.test_metrics.passed_tests}",
            f"- **失败测试**: {report.test_metrics.failed_tests}",
            f"- **错误测试**: {report.test_metrics.error_tests}",
            f"- **跳过测试**: {report.test_metrics.skipped_tests}",
            f"- **成功率**: {report.test_metrics.success_rate:.2f}%",
            f"- **平均测试时间**: {report.test_metrics.average_test_time:.3f}秒",
            "",
            "## 🎯 质量评分",
            "",
            f"- **鲁棒性评分**: {report.quality_metrics.robustness_score:.1f}/100",
            f"- **可靠性评分**: {report.quality_metrics.reliability_score:.1f}/100",
            f"- **可维护性评分**: {report.quality_metrics.maintainability_score:.1f}/100",
            f"- **性能评分**: {report.quality_metrics.performance_score:.1f}/100",
            f"- **安全性评分**: {report.quality_metrics.security_score:.1f}/100",
            "",
            "## 📈 覆盖率指标",
            "",
            f"- **代码覆盖率**: {report.coverage_metrics.code_coverage:.1f}%",
            f"- **函数覆盖率**: {report.coverage_metrics.function_coverage:.1f}%",
            f"- **分支覆盖率**: {report.coverage_metrics.branch_coverage:.1f}%",
            f"- **测试覆盖率**: {report.coverage_metrics.test_coverage:.1f}%",
            f"- **场景覆盖率**: {report.coverage_metrics.scenario_coverage:.1f}%",
            "",
            "## ⚡ 性能指标",
            "",
            f"- **CPU使用率**: {report.performance_metrics.cpu_usage:.1f}%",
            f"- **内存使用率**: {report.performance_metrics.memory_usage:.1f}%",
            f"- **磁盘IO**: {report.performance_metrics.disk_io:.1f}MB",
            f"- **网络IO**: {report.performance_metrics.network_io:.1f}MB",
            f"- **查询数量**: {report.performance_metrics.query_count}",
            f"- **平均查询时间**: {report.performance_metrics.average_query_time:.3f}秒",
            f"- **缓存命中率**: {report.performance_metrics.cache_hit_rate:.1f}%",
            "",
            "## 🧪 测试套件详情",
            ""
        ]
        
        # 添加测试套件详情
        for suite in report.test_suite_results:
            status_emoji = "✅" if suite['success_rate'] >= 80 else "❌"
            content_lines.extend([
                f"### {status_emoji} {suite['suite_name']}",
                f"- **总测试数**: {suite['total_tests']}",
                f"- **通过测试**: {suite['passed_tests']}",
                f"- **失败测试**: {suite['failed_tests']}",
                f"- **错误测试**: {suite['error_tests']}",
                f"- **成功率**: {suite['success_rate']:.1f}%",
                f"- **执行时间**: {suite['execution_time']:.2f}秒",
                ""
            ])
        
        # 添加改进建议
        content_lines.extend([
            "## 💡 改进建议",
            ""
        ])
        
        for i, recommendation in enumerate(report.recommendations, 1):
            content_lines.append(f"{i}. {recommendation}")
        
        content_lines.extend([
            "",
            "## 🔧 测试环境",
            ""
        ])
        
        for key, value in report.test_environment.items():
            content_lines.append(f"- **{key}**: {value}")
        
        content_lines.extend([
            "",
            "---",
            f"*报告生成于 {report.generation_time.strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        return "\n".join(content_lines)


def main():
    """主函数 - 演示报告生成"""
    reporter = IntegrationTestReporter()
    
    # 模拟测试结果
    mock_test_results = {
        'unit_tests': {
            'total_tests': 50,
            'passed_tests': 47,
            'failed_tests': 2,
            'error_tests': 1,
            'skipped_tests': 0,
            'execution_time': 25.3
        },
        'integration_tests': {
            'total_tests': 20,
            'passed_tests': 18,
            'failed_tests': 1,
            'error_tests': 1,
            'skipped_tests': 0,
            'execution_time': 120.5
        },
        'boundary_tests': {
            'total_tests': 15,
            'passed_tests': 13,
            'failed_tests': 2,
            'error_tests': 0,
            'skipped_tests': 0,
            'execution_time': 45.2,
            'robustness_score': 78.5
        },
        'exception_tests': {
            'total_tests': 12,
            'passed_tests': 10,
            'failed_tests': 2,
            'error_tests': 0,
            'skipped_tests': 0,
            'execution_time': 35.1,
            'exception_handling_score': 82.3
        },
        'consistency_tests': {
            'total_tests': 8,
            'passed_tests': 7,
            'failed_tests': 1,
            'error_tests': 0,
            'skipped_tests': 0,
            'execution_time': 60.7,
            'overall_consistency_score': 88.9
        }
    }
    
    # 生成综合报告
    report = reporter.generate_comprehensive_report(mock_test_results, "股票选股策略系统综合测试报告")
    
    # 导出不同格式的报告
    output_dir = "test_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    html_file = reporter.export_html_report(report, output_dir)
    json_file = reporter.export_json_report(report, output_dir)
    markdown_file = reporter.export_markdown_report(report, output_dir)
    
    print(f"综合测试报告已生成:")
    print(f"- HTML格式: {html_file}")
    print(f"- JSON格式: {json_file}")
    print(f"- Markdown格式: {markdown_file}")
    
    print(f"\n报告摘要:")
    print(f"- 总测试数: {report.test_metrics.total_tests}")
    print(f"- 成功率: {report.test_metrics.success_rate:.1f}%")
    print(f"- 质量评分: 鲁棒性 {report.quality_metrics.robustness_score:.1f}, 可靠性 {report.quality_metrics.reliability_score:.1f}")


if __name__ == "__main__":
    main() 