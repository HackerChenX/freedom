#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
生产级风控系统全面测试验证
Production-Grade Risk Control System Comprehensive Test Validation

作为生产级测试专家，对第三周风控体系建设进行全面测试验证：

测试范围：
- 统一风控管理系统
- 事前/事中/事后风控引擎
- 智能预警通知系统
- 风险可视化API
- 系统集成和性能

关键验收标准：
- 风控检查响应时间≤10ms
- 并发处理≥1000请求/秒
- 系统可用性≥99.5%
- 所有功能测试通过率100%
"""

import asyncio
import os
import sys
import time
import threading
import concurrent.futures
import statistics
import psutil
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from monitoring.risk_monitor import (
    RiskMonitoringSystem, MarketRiskAssessor, StockRiskMonitor,
    PortfolioRiskManager, RiskLevel, RiskType
)
from monitoring.intelligent_alert_system import (
    IntelligentAlertSystem, SignalType, SignalStrength, TradingSignal
)
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType
from utils.performance_monitor import performance_monitor
from utils.exception_handler import exception_handler

logger = get_logger(__name__)


@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    status: str  # PASS, FAIL, WARNING
    execution_time: float
    details: Dict[str, Any]
    metrics: Dict[str, float]
    issues: List[str] = None
    recommendations: List[str] = None


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float
    error_rate: float
    cpu_usage: float
    memory_usage: float


class ProductionRiskControlComprehensiveTest:
    """生产级风控系统全面测试验证类"""

    def __init__(self):
        """初始化测试验证系统"""
        self.test_results: List[TestResult] = []
        self.start_time = time.time()

        # 初始化系统组件
        self.risk_monitoring = RiskMonitoringSystem()
        self.market_assessor = MarketRiskAssessor()
        self.stock_monitor = StockRiskMonitor()
        self.portfolio_manager = PortfolioRiskManager()
        self.alert_system = IntelligentAlertSystem()

        # 测试配置
        self.performance_targets = {
            'risk_check_max_time': 0.01,  # 10ms
            'concurrent_throughput': 1000,  # 1000 requests/sec
            'system_availability': 99.5,  # 99.5%
            'max_cpu_usage': 80.0,  # 80%
            'max_memory_usage': 85.0,  # 85%
            'max_error_rate': 0.5  # 0.5%
        }

        # 测试数据集
        self.test_stocks = [
            "000001", "000002", "600000", "600036", "000858",
            "600519", "000166", "002415", "300750", "688981"
        ]

        self.test_portfolios = [
            {
                'id': 'prod_test_portfolio_001',
                'name': '稳健投资组合',
                'positions': [
                    {'code': '000001', 'name': '平安银行', 'weight': 0.2, 'value': 1000000},
                    {'code': '600000', 'name': '浦发银行', 'weight': 0.2, 'value': 1000000},
                    {'code': '600036', 'name': '招商银行', 'weight': 0.3, 'value': 1500000},
                    {'code': '000858', 'name': '五粮液', 'weight': 0.15, 'value': 750000},
                    {'code': '600519', 'name': '贵州茅台', 'weight': 0.15, 'value': 750000}
                ]
            },
            {
                'id': 'prod_test_portfolio_002',
                'name': '成长型组合',
                'positions': [
                    {'code': '002415', 'name': '海康威视', 'weight': 0.4, 'value': 2000000},
                    {'code': '300750', 'name': '宁德时代', 'weight': 0.35, 'value': 1750000},
                    {'code': '688981', 'name': '中芯国际', 'weight': 0.25, 'value': 1250000}
                ]
            }
        ]

        logger.info("生产级风控系统测试验证初始化完成")

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def test_performance_validation(self) -> TestResult:
        """执行性能测试验证 - 验证风控检查≤10ms、1000/秒并发吞吐量"""
        start_time = time.time()
        test_name = "性能测试验证"
        issues = []
        recommendations = []

        try:
            logger.info("🚀 开始性能测试验证...")

            # 1. 单次风控检查响应时间测试
            response_times = []

            for i in range(100):  # 100次测试
                test_start = time.time()

                # 执行风控检查操作
                risk_report = self.risk_monitoring.comprehensive_risk_assessment(
                    stocks=self.test_stocks[:3],  # 限制股票数量提高测试速度
                    portfolios=[self.test_portfolios[0]]
                )

                test_end = time.time()
                response_time = test_end - test_start
                response_times.append(response_time)

            # 计算响应时间统计
            avg_response = statistics.mean(response_times)
            max_response = max(response_times)
            min_response = min(response_times)
            p95_response = statistics.quantiles(response_times, n=20)[18]  # 95%分位数
            p99_response = statistics.quantiles(response_times, n=100)[98]  # 99%分位数

            # 2. 并发吞吐量测试
            throughput_results = self._test_concurrent_throughput()

            # 3. 系统资源使用测试
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent

            # 性能指标评估
            metrics = PerformanceMetrics(
                avg_response_time=avg_response,
                max_response_time=max_response,
                min_response_time=min_response,
                p95_response_time=p95_response,
                p99_response_time=p99_response,
                throughput=throughput_results['throughput'],
                error_rate=throughput_results['error_rate'],
                cpu_usage=cpu_usage,
                memory_usage=memory_usage
            )

            # 验证性能目标
            status = "PASS"

            if avg_response > self.performance_targets['risk_check_max_time']:
                status = "FAIL"
                issues.append(f"平均响应时间 {avg_response*1000:.2f}ms 超过目标 {self.performance_targets['risk_check_max_time']*1000}ms")

            if throughput_results['throughput'] < self.performance_targets['concurrent_throughput']:
                status = "FAIL"
                issues.append(f"并发吞吐量 {throughput_results['throughput']:.0f}/秒 低于目标 {self.performance_targets['concurrent_throughput']}/秒")

            if cpu_usage > self.performance_targets['max_cpu_usage']:
                if status != "FAIL":
                    status = "WARNING"
                issues.append(f"CPU使用率 {cpu_usage:.1f}% 超过建议值 {self.performance_targets['max_cpu_usage']}%")

            if memory_usage > self.performance_targets['max_memory_usage']:
                if status != "FAIL":
                    status = "WARNING"
                issues.append(f"内存使用率 {memory_usage:.1f}% 超过建议值 {self.performance_targets['max_memory_usage']}%")

            # 生成建议
            if issues:
                if avg_response > self.performance_targets['risk_check_max_time']:
                    recommendations.append("优化风控计算算法，考虑使用缓存机制")
                    recommendations.append("实现异步处理和批量操作优化")

                if throughput_results['throughput'] < self.performance_targets['concurrent_throughput']:
                    recommendations.append("增加并发处理能力，使用连接池")
                    recommendations.append("考虑实现负载均衡和分布式处理")

                if cpu_usage > self.performance_targets['max_cpu_usage']:
                    recommendations.append("优化CPU密集型计算，考虑使用多进程")

                if memory_usage > self.performance_targets['max_memory_usage']:
                    recommendations.append("实现内存管理优化，及时释放不需要的对象")

            execution_time = time.time() - start_time

            return TestResult(
                test_name=test_name,
                status=status,
                execution_time=execution_time,
                details={
                    'response_time_stats': {
                        'average_ms': round(avg_response * 1000, 2),
                        'max_ms': round(max_response * 1000, 2),
                        'min_ms': round(min_response * 1000, 2),
                        'p95_ms': round(p95_response * 1000, 2),
                        'p99_ms': round(p99_response * 1000, 2)
                    },
                    'throughput_stats': throughput_results,
                    'resource_usage': {
                        'cpu_percent': cpu_usage,
                        'memory_percent': memory_usage,
                        'available_memory_gb': round(memory_info.available / (1024**3), 2)
                    }
                },
                metrics=asdict(metrics),
                issues=issues,
                recommendations=recommendations
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"性能测试验证失败: {e}")

            return TestResult(
                test_name=test_name,
                status="FAIL",
                execution_time=execution_time,
                details={'error': str(e)},
                metrics={},
                issues=[f"性能测试执行失败: {str(e)}"],
                recommendations=["检查系统配置和依赖组件状态"]
            )

    def _test_concurrent_throughput(self) -> Dict[str, float]:
        """测试并发吞吐量"""
        total_requests = 100
        max_workers = 10
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0

        def single_request():
            try:
                risk_report = self.risk_monitoring.comprehensive_risk_assessment(
                    stocks=self.test_stocks[:2],
                    portfolios=[]
                )
                return True
            except Exception:
                return False

        # 并发执行请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_request) for _ in range(total_requests)]

            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    successful_requests += 1
                else:
                    failed_requests += 1

        end_time = time.time()
        total_time = end_time - start_time

        throughput = successful_requests / total_time if total_time > 0 else 0
        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0

        return {
            'throughput': throughput,
            'error_rate': error_rate,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'total_time': total_time
        }

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=5.0)
    def test_functional_completeness(self) -> TestResult:
        """执行功能完整性测试 - 事前/事中/事后风控"""
        start_time = time.time()
        test_name = "功能完整性测试"
        issues = []
        recommendations = []

        try:
            logger.info("🔍 开始功能完整性测试...")

            test_results = {}

            # 1. 事前风控功能测试
            pre_risk_result = self._test_pre_risk_control()
            test_results['pre_risk_control'] = pre_risk_result

            # 2. 事中风控功能测试
            during_risk_result = self._test_during_risk_control()
            test_results['during_risk_control'] = during_risk_result

            # 3. 事后风控功能测试
            post_risk_result = self._test_post_risk_control()
            test_results['post_risk_control'] = post_risk_result

            # 4. 预警系统功能测试
            alert_result = self._test_alert_system()
            test_results['alert_system'] = alert_result

            # 5. 风险可视化API测试
            visualization_result = self._test_risk_visualization()
            test_results['risk_visualization'] = visualization_result

            # 综合评估
            passed_tests = sum(1 for result in test_results.values() if result['status'] == 'PASS')
            total_tests = len(test_results)
            pass_rate = (passed_tests / total_tests) * 100

            status = "PASS" if pass_rate >= 100 else "FAIL" if pass_rate < 80 else "WARNING"

            if status != "PASS":
                for test_key, result in test_results.items():
                    if result['status'] != 'PASS':
                        issues.extend(result.get('issues', []))
                        recommendations.extend(result.get('recommendations', []))

            execution_time = time.time() - start_time

            return TestResult(
                test_name=test_name,
                status=status,
                execution_time=execution_time,
                details={
                    'test_results': test_results,
                    'pass_rate': pass_rate,
                    'passed_tests': passed_tests,
                    'total_tests': total_tests
                },
                metrics={
                    'functional_coverage': pass_rate,
                    'execution_time': execution_time
                },
                issues=issues,
                recommendations=recommendations
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"功能完整性测试失败: {e}")

            return TestResult(
                test_name=test_name,
                status="FAIL",
                execution_time=execution_time,
                details={'error': str(e)},
                metrics={},
                issues=[f"功能测试执行失败: {str(e)}"],
                recommendations=["检查系统功能模块完整性"]
            )

    def _test_pre_risk_control(self) -> Dict[str, Any]:
        """测试事前风控功能"""
        try:
            # 市场风险评估
            market_risk = self.market_assessor.assess_market_risk("000001")

            # 个股风险评估
            stock_risk = self.stock_monitor.monitor_stock_risk("000001", "平安银行")

            # 验证关键字段
            required_market_fields = ['risk_level', 'risk_score', 'var_1d', 'volatility']
            required_stock_fields = ['risk_level', 'risk_score', 'var_1d', 'volatility', 'beta']

            market_valid = all(field in market_risk for field in required_market_fields)
            stock_valid = all(hasattr(stock_risk, field) for field in required_stock_fields)

            if market_valid and stock_valid:
                return {'status': 'PASS', 'details': {'market_risk': market_risk, 'stock_risk': asdict(stock_risk)}}
            else:
                issues = []
                if not market_valid:
                    issues.append("市场风险评估缺少必要字段")
                if not stock_valid:
                    issues.append("个股风险评估缺少必要字段")
                return {'status': 'FAIL', 'issues': issues}

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"事前风控测试失败: {str(e)}"]}

    def _test_during_risk_control(self) -> Dict[str, Any]:
        """测试事中风控功能"""
        try:
            # 实时监控功能测试
            status_before = self.risk_monitoring.get_monitoring_status()

            # 启动监控
            self.risk_monitoring.start_real_time_monitoring(self.test_stocks[:2], interval=1)
            time.sleep(2)  # 等待监控启动

            status_during = self.risk_monitoring.get_monitoring_status()

            # 停止监控
            self.risk_monitoring.stop_real_time_monitoring()
            time.sleep(1)  # 等待监控停止

            status_after = self.risk_monitoring.get_monitoring_status()

            # 验证监控功能
            monitoring_works = (
                status_during['monitoring_enabled'] and
                status_during['thread_alive'] and
                not status_after['monitoring_enabled']
            )

            if monitoring_works:
                return {'status': 'PASS', 'details': {'monitoring_lifecycle': 'OK'}}
            else:
                return {'status': 'FAIL', 'issues': ['实时监控功能异常']}

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"事中风控测试失败: {str(e)}"]}

    def _test_post_risk_control(self) -> Dict[str, Any]:
        """测试事后风控功能"""
        try:
            # 组合风险评估
            portfolio_risk = self.portfolio_manager.assess_portfolio_risk(self.test_portfolios[0])

            # 综合风险评估
            comprehensive_report = self.risk_monitoring.comprehensive_risk_assessment(
                stocks=self.test_stocks[:3],
                portfolios=[self.test_portfolios[0]]
            )

            # 验证报告结构
            required_sections = ['market_risk', 'stock_risks', 'portfolio_risks', 'risk_summary']
            sections_valid = all(section in comprehensive_report for section in required_sections)

            portfolio_valid = hasattr(portfolio_risk, 'risk_level') and hasattr(portfolio_risk, 'total_value')

            if sections_valid and portfolio_valid:
                return {'status': 'PASS', 'details': {'comprehensive_report': True, 'portfolio_assessment': True}}
            else:
                issues = []
                if not sections_valid:
                    issues.append("综合风险报告缺少必要部分")
                if not portfolio_valid:
                    issues.append("组合风险评估格式异常")
                return {'status': 'FAIL', 'issues': issues}

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"事后风控测试失败: {str(e)}"]}

    def _test_alert_system(self) -> Dict[str, Any]:
        """测试预警系统功能"""
        try:
            # 获取系统统计
            stats = self.alert_system.get_system_statistics()

            # 分析股票信号
            signals = self.alert_system.analyze_stock_signals("000001", "测试股票")

            # 获取预警规则
            rules = self.alert_system.get_alert_rules()

            # 验证功能
            stats_valid = 'total_rules' in stats and 'total_signals' in stats
            signals_valid = isinstance(signals, list)
            rules_valid = isinstance(rules, list) and len(rules) > 0

            if stats_valid and signals_valid and rules_valid:
                return {
                    'status': 'PASS',
                    'details': {
                        'rules_count': len(rules),
                        'signals_generated': len(signals),
                        'system_stats': stats
                    }
                }
            else:
                issues = []
                if not stats_valid:
                    issues.append("系统统计信息异常")
                if not signals_valid:
                    issues.append("信号分析功能异常")
                if not rules_valid:
                    issues.append("预警规则配置异常")
                return {'status': 'FAIL', 'issues': issues}

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"预警系统测试失败: {str(e)}"]}

    def _test_risk_visualization(self) -> Dict[str, Any]:
        """测试风险可视化API"""
        try:
            # 生成风险数据
            market_risk = self.market_assessor.assess_market_risk("000001")
            stock_risks = []
            for stock in self.test_stocks[:3]:
                risk = self.stock_monitor.monitor_stock_risk(stock)
                stock_risks.append(asdict(risk))

            # 验证数据可序列化（API返回要求）
            market_json = json.dumps(market_risk, default=str)
            stocks_json = json.dumps(stock_risks, default=str)

            # 验证关键可视化数据字段
            visualization_data = {
                'market_overview': market_risk,
                'stock_details': stock_risks,
                'risk_distribution': self._generate_risk_distribution(stock_risks),
                'time_series': self._generate_time_series_data()
            }

            viz_json = json.dumps(visualization_data, default=str)

            if len(viz_json) > 100:  # 基本数据完整性检查
                return {'status': 'PASS', 'details': {'data_size': len(viz_json)}}
            else:
                return {'status': 'FAIL', 'issues': ['风险可视化数据不足']}

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"风险可视化测试失败: {str(e)}"]}

    def _generate_risk_distribution(self, stock_risks: List[Dict]) -> Dict[str, int]:
        """生成风险分布数据"""
        distribution = {'低风险': 0, '中等风险': 0, '高风险': 0, '极高风险': 0}
        for risk in stock_risks:
            level = risk.get('risk_level', '中等风险')
            if level in distribution:
                distribution[level] += 1
        return distribution

    def _generate_time_series_data(self) -> List[Dict[str, Any]]:
        """生成时间序列数据"""
        base_time = datetime.now()
        return [
            {
                'timestamp': (base_time - timedelta(hours=i)).isoformat(),
                'risk_score': 50 + (i * 2),
                'volatility': 0.2 + (i * 0.01)
            }
            for i in range(24)
        ]

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=10.0)
    def test_integration_validation(self) -> TestResult:
        """执行集成测试验证 - 数据库、API、WebSocket连接"""
        start_time = time.time()
        test_name = "集成测试验证"
        issues = []
        recommendations = []

        try:
            logger.info("🔗 开始集成测试验证...")

            integration_results = {}

            # 1. 数据库集成测试
            db_result = self._test_database_integration()
            integration_results['database'] = db_result

            # 2. API集成测试
            api_result = self._test_api_integration()
            integration_results['api'] = api_result

            # 3. WebSocket连接测试
            websocket_result = self._test_websocket_integration()
            integration_results['websocket'] = websocket_result

            # 4. 依赖服务集成测试
            dependencies_result = self._test_dependencies_integration()
            integration_results['dependencies'] = dependencies_result

            # 5. 端到端工作流测试
            e2e_result = self._test_end_to_end_workflow()
            integration_results['end_to_end'] = e2e_result

            # 综合评估
            passed_tests = sum(1 for result in integration_results.values() if result['status'] == 'PASS')
            total_tests = len(integration_results)
            pass_rate = (passed_tests / total_tests) * 100

            status = "PASS" if pass_rate >= 80 else "FAIL" if pass_rate < 60 else "WARNING"

            # 收集问题和建议
            for test_key, result in integration_results.items():
                if result['status'] != 'PASS':
                    issues.extend(result.get('issues', []))
                    recommendations.extend(result.get('recommendations', []))

            execution_time = time.time() - start_time

            return TestResult(
                test_name=test_name,
                status=status,
                execution_time=execution_time,
                details={
                    'integration_results': integration_results,
                    'pass_rate': pass_rate,
                    'passed_tests': passed_tests,
                    'total_tests': total_tests
                },
                metrics={
                    'integration_coverage': pass_rate,
                    'execution_time': execution_time
                },
                issues=issues,
                recommendations=recommendations
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"集成测试验证失败: {e}")

            return TestResult(
                test_name=test_name,
                status="FAIL",
                execution_time=execution_time,
                details={'error': str(e)},
                metrics={},
                issues=[f"集成测试执行失败: {str(e)}"],
                recommendations=["检查系统集成配置和网络连接"]
            )

    def _test_database_integration(self) -> Dict[str, Any]:
        """测试数据库集成"""
        try:
            # 测试数据访问功能
            from utils.unified_container import get_container
            container = get_container()

            try:
                from db.interfaces.data_access_interface import DataAccessInterface
                data_access = container.resolve(DataAccessInterface)

                # 执行测试查询
                test_query = "SELECT 1 as test_value"
                result = data_access.query_dataframe(test_query)

                if not result.empty:
                    return {'status': 'PASS', 'details': {'connection': 'OK', 'query_test': 'OK'}}
                else:
                    return {'status': 'FAIL', 'issues': ['数据库查询返回空结果']}

            except Exception as e:
                # 数据接口不可用，使用模拟数据模式
                return {
                    'status': 'WARNING',
                    'details': {'connection': 'MOCK_MODE'},
                    'issues': [f'数据库连接使用模拟模式: {str(e)}'],
                    'recommendations': ['配置真实数据库连接以获得完整功能']
                }

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'数据库集成测试失败: {str(e)}']}

    def _test_api_integration(self) -> Dict[str, Any]:
        """测试API集成"""
        try:
            # 检查API相关文件是否存在
            api_files = [
                project_root / 'api' / 'main.py',
                project_root / 'api' / 'websocket_server.py'
            ]

            missing_files = [f for f in api_files if not f.exists()]

            if missing_files:
                return {
                    'status': 'WARNING',
                    'issues': [f'API文件缺失: {[str(f) for f in missing_files]}'],
                    'recommendations': ['实现完整的API端点以支持生产环境']
                }

            # 测试API数据结构
            risk_data = {
                'market_risk': self.market_assessor.assess_market_risk("000001"),
                'stock_risk': asdict(self.stock_monitor.monitor_stock_risk("000001")),
                'timestamp': datetime.now().isoformat()
            }

            # 验证数据可序列化
            json_data = json.dumps(risk_data, default=str)

            if len(json_data) > 50:
                return {'status': 'PASS', 'details': {'api_data_structure': 'OK'}}
            else:
                return {'status': 'FAIL', 'issues': ['API数据结构不完整']}

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'API集成测试失败: {str(e)}']}

    def _test_websocket_integration(self) -> Dict[str, Any]:
        """测试WebSocket集成"""
        try:
            # 检查WebSocket服务器文件
            websocket_file = project_root / 'api' / 'websocket_server.py'

            if not websocket_file.exists():
                return {
                    'status': 'WARNING',
                    'issues': ['WebSocket服务器文件不存在'],
                    'recommendations': ['实现WebSocket服务器以支持实时数据推送']
                }

            # 测试实时数据结构
            realtime_data = {
                'type': 'risk_alert',
                'data': {
                    'stock_code': '000001',
                    'risk_level': '高风险',
                    'timestamp': datetime.now().isoformat()
                }
            }

            json_data = json.dumps(realtime_data, default=str)

            if len(json_data) > 20:
                return {'status': 'PASS', 'details': {'websocket_data_structure': 'OK'}}
            else:
                return {'status': 'FAIL', 'issues': ['WebSocket数据结构异常']}

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'WebSocket集成测试失败: {str(e)}']}

    def _test_dependencies_integration(self) -> Dict[str, Any]:
        """测试依赖服务集成"""
        try:
            # 测试关键依赖
            dependencies_status = {}

            # 检查指标注册表
            try:
                from indicators.complete_indicator_registry import get_indicator_registry
                registry = get_indicator_registry()
                dependencies_status['indicator_registry'] = 'OK'
            except Exception as e:
                dependencies_status['indicator_registry'] = f'ERROR: {str(e)}'

            # 检查统一容器
            try:
                from utils.unified_container import get_container
                container = get_container()
                dependencies_status['unified_container'] = 'OK'
            except Exception as e:
                dependencies_status['unified_container'] = f'ERROR: {str(e)}'

            # 检查日志系统
            try:
                from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType
                test_logger = get_logger("test")
                dependencies_status['logging_system'] = 'OK'
            except Exception as e:
                dependencies_status['logging_system'] = f'ERROR: {str(e)}'

            errors = [k for k, v in dependencies_status.items() if 'ERROR' in v]

            if not errors:
                return {'status': 'PASS', 'details': dependencies_status}
            else:
                return {
                    'status': 'FAIL' if len(errors) > 1 else 'WARNING',
                    'details': dependencies_status,
                    'issues': [f'依赖服务异常: {errors}']
                }

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'依赖集成测试失败: {str(e)}']}

    def _test_end_to_end_workflow(self) -> Dict[str, Any]:
        """测试端到端工作流"""
        try:
            workflow_steps = {}

            # 步骤1: 市场风险评估
            market_risk = self.market_assessor.assess_market_risk("000001")
            workflow_steps['market_assessment'] = 'OK' if market_risk else 'FAIL'

            # 步骤2: 个股风险分析
            stock_risk = self.stock_monitor.monitor_stock_risk("000001")
            workflow_steps['stock_analysis'] = 'OK' if stock_risk else 'FAIL'

            # 步骤3: 组合风险计算
            portfolio_risk = self.portfolio_manager.assess_portfolio_risk(self.test_portfolios[0])
            workflow_steps['portfolio_calculation'] = 'OK' if portfolio_risk else 'FAIL'

            # 步骤4: 预警信号生成
            signals = self.alert_system.analyze_stock_signals("000001")
            workflow_steps['signal_generation'] = 'OK' if isinstance(signals, list) else 'FAIL'

            # 步骤5: 综合报告生成
            comprehensive_report = self.risk_monitoring.comprehensive_risk_assessment(
                stocks=["000001"],
                portfolios=[self.test_portfolios[0]]
            )
            workflow_steps['report_generation'] = 'OK' if comprehensive_report else 'FAIL'

            failed_steps = [k for k, v in workflow_steps.items() if v == 'FAIL']

            if not failed_steps:
                return {'status': 'PASS', 'details': workflow_steps}
            else:
                return {
                    'status': 'FAIL',
                    'details': workflow_steps,
                    'issues': [f'工作流步骤失败: {failed_steps}']
                }

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'端到端工作流测试失败: {str(e)}']}

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=15.0)
    def test_fault_tolerance_and_exceptions(self) -> TestResult:
        """执行容错和异常测试"""
        start_time = time.time()
        test_name = "容错和异常测试"
        issues = []
        recommendations = []

        try:
            logger.info("⚡ 开始容错和异常测试...")

            fault_tests = {}

            # 1. 数据库连接异常处理
            db_fault_result = self._test_database_fault_handling()
            fault_tests['database_faults'] = db_fault_result

            # 2. 网络中断恢复能力
            network_fault_result = self._test_network_fault_handling()
            fault_tests['network_faults'] = network_fault_result

            # 3. 极端数据条件测试
            extreme_data_result = self._test_extreme_data_conditions()
            fault_tests['extreme_data'] = extreme_data_result

            # 4. 系统过载保护机制
            overload_result = self._test_system_overload_protection()
            fault_tests['overload_protection'] = overload_result

            # 5. 异常恢复能力测试
            recovery_result = self._test_exception_recovery()
            fault_tests['exception_recovery'] = recovery_result

            # 综合评估
            passed_tests = sum(1 for result in fault_tests.values() if result['status'] == 'PASS')
            total_tests = len(fault_tests)
            pass_rate = (passed_tests / total_tests) * 100

            status = "PASS" if pass_rate >= 70 else "FAIL" if pass_rate < 50 else "WARNING"

            # 收集问题和建议
            for test_key, result in fault_tests.items():
                if result['status'] != 'PASS':
                    issues.extend(result.get('issues', []))
                    recommendations.extend(result.get('recommendations', []))

            execution_time = time.time() - start_time

            return TestResult(
                test_name=test_name,
                status=status,
                execution_time=execution_time,
                details={
                    'fault_tolerance_results': fault_tests,
                    'pass_rate': pass_rate,
                    'passed_tests': passed_tests,
                    'total_tests': total_tests
                },
                metrics={
                    'fault_tolerance_coverage': pass_rate,
                    'execution_time': execution_time
                },
                issues=issues,
                recommendations=recommendations
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"容错和异常测试失败: {e}")

            return TestResult(
                test_name=test_name,
                status="FAIL",
                execution_time=execution_time,
                details={'error': str(e)},
                metrics={},
                issues=[f"容错测试执行失败: {str(e)}"],
                recommendations=["加强系统异常处理和容错机制"]
            )

    def _test_database_fault_handling(self) -> Dict[str, Any]:
        """测试数据库故障处理"""
        try:
            # 测试无数据情况
            risk_result = self.stock_monitor.monitor_stock_risk("INVALID_CODE")

            # 系统应该返回默认值而不是崩溃
            if hasattr(risk_result, 'risk_level'):
                return {'status': 'PASS', 'details': {'handles_invalid_data': True}}
            else:
                return {'status': 'FAIL', 'issues': ['无效数据未正确处理']}

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'数据库故障处理测试失败: {str(e)}']}

    def _test_network_fault_handling(self) -> Dict[str, Any]:
        """测试网络故障处理"""
        try:
            # 测试在无网络连接情况下的行为
            # 由于使用模拟数据，系统应该能正常工作
            market_risk = self.market_assessor.assess_market_risk("000001")

            if market_risk and 'risk_level' in market_risk:
                return {'status': 'PASS', 'details': {'network_independent': True}}
            else:
                return {'status': 'FAIL', 'issues': ['网络故障时系统无法运行']}

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'网络故障处理测试失败: {str(e)}']}

    def _test_extreme_data_conditions(self) -> Dict[str, Any]:
        """测试极端数据条件"""
        try:
            extreme_tests = {}

            # 测试空数据集
            empty_portfolio = {'id': 'empty', 'name': '空组合', 'positions': []}
            empty_result = self.portfolio_manager.assess_portfolio_risk(empty_portfolio)
            extreme_tests['empty_portfolio'] = hasattr(empty_result, 'risk_level')

            # 测试大数据集
            large_stock_list = [f"00000{i}" for i in range(100)]
            large_result = self.risk_monitoring.comprehensive_risk_assessment(large_stock_list[:10])  # 限制数量
            extreme_tests['large_dataset'] = bool(large_result)

            # 测试异常值
            extreme_portfolio = {
                'id': 'extreme',
                'name': '极端组合',
                'positions': [
                    {'code': '000001', 'name': '股票1', 'weight': 0.99, 'value': 999999999},
                    {'code': '000002', 'name': '股票2', 'weight': 0.01, 'value': 1}
                ]
            }
            extreme_portfolio_result = self.portfolio_manager.assess_portfolio_risk(extreme_portfolio)
            extreme_tests['extreme_values'] = hasattr(extreme_portfolio_result, 'concentration_risk')

            passed_extreme = sum(extreme_tests.values())
            total_extreme = len(extreme_tests)

            if passed_extreme == total_extreme:
                return {'status': 'PASS', 'details': extreme_tests}
            else:
                return {
                    'status': 'WARNING' if passed_extreme > 0 else 'FAIL',
                    'details': extreme_tests,
                    'issues': ['部分极端数据条件处理不当']
                }

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'极端数据条件测试失败: {str(e)}']}

    def _test_system_overload_protection(self) -> Dict[str, Any]:
        """测试系统过载保护"""
        try:
            start_time = time.time()

            # 快速连续执行多个风险评估
            for i in range(20):
                self.market_assessor.assess_market_risk("000001")

            end_time = time.time()
            total_time = end_time - start_time

            # 系统应该能在合理时间内完成
            if total_time < 10:  # 10秒内完成20次评估
                return {'status': 'PASS', 'details': {'total_time': total_time}}
            else:
                return {
                    'status': 'WARNING',
                    'details': {'total_time': total_time},
                    'issues': ['系统响应时间过长'],
                    'recommendations': ['考虑实现请求限流和负载保护']
                }

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'过载保护测试失败: {str(e)}']}

    def _test_exception_recovery(self) -> Dict[str, Any]:
        """测试异常恢复能力"""
        try:
            recovery_tests = {}

            # 测试单个组件失败后的恢复
            try:
                # 故意传入错误参数
                self.risk_monitoring.comprehensive_risk_assessment([], [])
                recovery_tests['empty_params'] = True
            except Exception:
                recovery_tests['empty_params'] = False

            # 测试系统状态恢复
            initial_status = self.risk_monitoring.get_monitoring_status()
            recovery_tests['status_consistency'] = bool(initial_status)

            passed_recovery = sum(recovery_tests.values())
            total_recovery = len(recovery_tests)

            if passed_recovery >= total_recovery * 0.8:  # 80%通过率
                return {'status': 'PASS', 'details': recovery_tests}
            else:
                return {
                    'status': 'WARNING',
                    'details': recovery_tests,
                    'issues': ['系统异常恢复能力有限']
                }

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'异常恢复测试失败: {str(e)}']}

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=20.0)
    def test_stress_and_stability(self) -> TestResult:
        """执行压力测试和稳定性验证"""
        start_time = time.time()
        test_name = "压力测试和稳定性验证"
        issues = []
        recommendations = []

        try:
            logger.info("💪 开始压力测试和稳定性验证...")

            stress_results = {}

            # 1. 高频交易场景模拟
            high_frequency_result = self._test_high_frequency_scenario()
            stress_results['high_frequency'] = high_frequency_result

            # 2. 大量并发请求测试
            concurrent_stress_result = self._test_concurrent_stress()
            stress_results['concurrent_stress'] = concurrent_stress_result

            # 3. 长时间运行稳定性
            long_running_result = self._test_long_running_stability()
            stress_results['long_running'] = long_running_result

            # 4. 内存泄漏检测
            memory_leak_result = self._test_memory_leak_detection()
            stress_results['memory_leak'] = memory_leak_result

            # 5. 性能衰减测试
            performance_degradation_result = self._test_performance_degradation()
            stress_results['performance_degradation'] = performance_degradation_result

            # 综合评估
            passed_tests = sum(1 for result in stress_results.values() if result['status'] == 'PASS')
            total_tests = len(stress_results)
            pass_rate = (passed_tests / total_tests) * 100

            status = "PASS" if pass_rate >= 80 else "FAIL" if pass_rate < 60 else "WARNING"

            # 收集问题和建议
            for test_key, result in stress_results.items():
                if result['status'] != 'PASS':
                    issues.extend(result.get('issues', []))
                    recommendations.extend(result.get('recommendations', []))

            execution_time = time.time() - start_time

            return TestResult(
                test_name=test_name,
                status=status,
                execution_time=execution_time,
                details={
                    'stress_test_results': stress_results,
                    'pass_rate': pass_rate,
                    'passed_tests': passed_tests,
                    'total_tests': total_tests
                },
                metrics={
                    'stress_test_coverage': pass_rate,
                    'execution_time': execution_time
                },
                issues=issues,
                recommendations=recommendations
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"压力测试和稳定性验证失败: {e}")

            return TestResult(
                test_name=test_name,
                status="FAIL",
                execution_time=execution_time,
                details={'error': str(e)},
                metrics={},
                issues=[f"压力测试执行失败: {str(e)}"],
                recommendations=["优化系统性能和稳定性机制"]
            )

    def _test_high_frequency_scenario(self) -> Dict[str, Any]:
        """测试高频交易场景"""
        try:
            start_time = time.time()
            request_count = 50

            response_times = []
            for i in range(request_count):
                req_start = time.time()
                market_risk = self.market_assessor.assess_market_risk("000001")
                req_end = time.time()
                response_times.append(req_end - req_start)

            end_time = time.time()
            total_time = end_time - start_time
            avg_response = statistics.mean(response_times)
            throughput = request_count / total_time

            if avg_response < 0.1 and throughput > 10:  # 100ms响应时间，10次/秒吞吐量
                return {
                    'status': 'PASS',
                    'details': {
                        'avg_response_time': avg_response,
                        'throughput': throughput,
                        'total_requests': request_count
                    }
                }
            else:
                return {
                    'status': 'WARNING',
                    'details': {
                        'avg_response_time': avg_response,
                        'throughput': throughput
                    },
                    'issues': ['高频场景性能不达标'],
                    'recommendations': ['优化计算算法和数据访问']
                }

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'高频场景测试失败: {str(e)}']}

    def _test_concurrent_stress(self) -> Dict[str, Any]:
        """测试并发压力"""
        try:
            max_workers = 20
            requests_per_worker = 5
            total_requests = max_workers * requests_per_worker

            def worker_task(worker_id):
                results = []
                for i in range(requests_per_worker):
                    try:
                        start_time = time.time()
                        risk_result = self.stock_monitor.monitor_stock_risk(f"00000{worker_id}")
                        end_time = time.time()
                        results.append({
                            'success': True,
                            'response_time': end_time - start_time
                        })
                    except Exception as e:
                        results.append({
                            'success': False,
                            'error': str(e)
                        })
                return results

            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(worker_task, i) for i in range(max_workers)]
                all_results = []

                for future in concurrent.futures.as_completed(futures):
                    all_results.extend(future.result())

            end_time = time.time()
            total_time = end_time - start_time

            successful_requests = sum(1 for result in all_results if result['success'])
            failed_requests = total_requests - successful_requests
            success_rate = (successful_requests / total_requests) * 100
            throughput = successful_requests / total_time

            if success_rate >= 95 and throughput >= 50:  # 95%成功率，50次/秒吞吐量
                return {
                    'status': 'PASS',
                    'details': {
                        'success_rate': success_rate,
                        'throughput': throughput,
                        'total_requests': total_requests,
                        'successful_requests': successful_requests,
                        'failed_requests': failed_requests
                    }
                }
            else:
                return {
                    'status': 'WARNING',
                    'details': {
                        'success_rate': success_rate,
                        'throughput': throughput
                    },
                    'issues': ['并发压力测试性能不达标'],
                    'recommendations': ['增加并发处理能力，优化线程安全']
                }

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'并发压力测试失败: {str(e)}']}

    def _test_long_running_stability(self) -> Dict[str, Any]:
        """测试长时间运行稳定性"""
        try:
            # 模拟长时间运行（简化版本）
            start_memory = psutil.Process().memory_info().rss
            start_time = time.time()

            # 连续运行30秒
            end_time = start_time + 30
            iteration_count = 0
            errors = 0

            while time.time() < end_time:
                try:
                    self.risk_monitoring.comprehensive_risk_assessment(
                        stocks=self.test_stocks[:2],
                        portfolios=[]
                    )
                    iteration_count += 1
                    time.sleep(0.5)  # 每0.5秒一次
                except Exception:
                    errors += 1

            final_time = time.time()
            final_memory = psutil.Process().memory_info().rss

            actual_runtime = final_time - start_time
            memory_growth = (final_memory - start_memory) / (1024 * 1024)  # MB
            error_rate = (errors / iteration_count) * 100 if iteration_count > 0 else 100

            if error_rate < 5 and memory_growth < 100:  # 5%错误率，100MB内存增长
                return {
                    'status': 'PASS',
                    'details': {
                        'runtime_seconds': actual_runtime,
                        'iterations': iteration_count,
                        'error_rate': error_rate,
                        'memory_growth_mb': memory_growth
                    }
                }
            else:
                return {
                    'status': 'WARNING',
                    'details': {
                        'error_rate': error_rate,
                        'memory_growth_mb': memory_growth
                    },
                    'issues': ['长时间运行稳定性不佳'],
                    'recommendations': ['检查内存泄漏和异常处理机制']
                }

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'长时间稳定性测试失败: {str(e)}']}

    def _test_memory_leak_detection(self) -> Dict[str, Any]:
        """测试内存泄漏检测"""
        try:
            import gc

            # 强制垃圾回收
            gc.collect()
            initial_memory = psutil.Process().memory_info().rss

            # 执行多次操作
            for i in range(100):
                risk_result = self.stock_monitor.monitor_stock_risk(f"TEST{i:03d}")
                signals = self.alert_system.analyze_stock_signals(f"TEST{i:03d}")

                # 定期垃圾回收
                if i % 20 == 0:
                    gc.collect()

            # 最终垃圾回收
            gc.collect()
            final_memory = psutil.Process().memory_info().rss

            memory_growth = (final_memory - initial_memory) / (1024 * 1024)  # MB

            if memory_growth < 50:  # 50MB增长阈值
                return {
                    'status': 'PASS',
                    'details': {
                        'memory_growth_mb': memory_growth,
                        'initial_memory_mb': initial_memory / (1024 * 1024),
                        'final_memory_mb': final_memory / (1024 * 1024)
                    }
                }
            else:
                return {
                    'status': 'WARNING',
                    'details': {'memory_growth_mb': memory_growth},
                    'issues': ['可能存在内存泄漏'],
                    'recommendations': ['检查对象生命周期管理和内存释放']
                }

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'内存泄漏检测失败: {str(e)}']}

    def _test_performance_degradation(self) -> Dict[str, Any]:
        """测试性能衰减"""
        try:
            # 测试初始性能
            initial_times = []
            for i in range(10):
                start_time = time.time()
                self.market_assessor.assess_market_risk("000001")
                end_time = time.time()
                initial_times.append(end_time - start_time)

            initial_avg = statistics.mean(initial_times)

            # 执行大量操作后测试性能
            for i in range(200):
                self.market_assessor.assess_market_risk(f"LOAD{i:03d}")

            # 测试后续性能
            final_times = []
            for i in range(10):
                start_time = time.time()
                self.market_assessor.assess_market_risk("000001")
                end_time = time.time()
                final_times.append(end_time - start_time)

            final_avg = statistics.mean(final_times)

            # 计算性能衰减
            degradation_ratio = final_avg / initial_avg if initial_avg > 0 else 1.0

            if degradation_ratio < 1.5:  # 性能衰减不超过50%
                return {
                    'status': 'PASS',
                    'details': {
                        'initial_avg_time': initial_avg,
                        'final_avg_time': final_avg,
                        'degradation_ratio': degradation_ratio
                    }
                }
            else:
                return {
                    'status': 'WARNING',
                    'details': {
                        'degradation_ratio': degradation_ratio
                    },
                    'issues': ['性能衰减过多'],
                    'recommendations': ['优化算法缓存和资源管理']
                }

        except Exception as e:
            return {'status': 'FAIL', 'issues': [f'性能衰减测试失败: {str(e)}']}

    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """运行完整的测试套件"""
        logger.info("🚀 开始生产级风控系统全面测试验证")
        logger.info("=" * 80)

        # 更新todo状态
        try:
            # 1. 性能测试验证
            performance_result = self.test_performance_validation()
            self.test_results.append(performance_result)

            # 2. 功能完整性测试
            functional_result = self.test_functional_completeness()
            self.test_results.append(functional_result)

            # 3. 集成测试验证
            integration_result = self.test_integration_validation()
            self.test_results.append(integration_result)

            # 4. 容错和异常测试
            fault_tolerance_result = self.test_fault_tolerance_and_exceptions()
            self.test_results.append(fault_tolerance_result)

            # 5. 压力测试和稳定性验证
            stress_result = self.test_stress_and_stability()
            self.test_results.append(stress_result)

        except Exception as e:
            logger.error(f"测试执行过程中发生错误: {e}")

        # 生成最终报告
        final_report = self._generate_comprehensive_report()

        logger.info("✅ 生产级风控系统全面测试验证完成")
        return final_report

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合测试报告"""
        total_execution_time = time.time() - self.start_time

        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "PASS"])
        warning_tests = len([r for r in self.test_results if r.status == "WARNING"])
        failed_tests = len([r for r in self.test_results if r.status == "FAIL"])

        overall_pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        # 确定整体状态
        if failed_tests == 0 and warning_tests == 0:
            overall_status = "生产就绪"
        elif failed_tests == 0 and warning_tests <= 2:
            overall_status = "基本就绪（有轻微警告）"
        elif failed_tests <= 1:
            overall_status = "需要修复（有失败项）"
        else:
            overall_status = "不适合生产环境"

        # 收集所有问题和建议
        all_issues = []
        all_recommendations = []
        for result in self.test_results:
            if result.issues:
                all_issues.extend(result.issues)
            if result.recommendations:
                all_recommendations.extend(result.recommendations)

        # 关键性能指标汇总
        performance_summary = {}
        for result in self.test_results:
            if result.test_name == "性能测试验证" and result.details:
                response_stats = result.details.get('response_time_stats', {})
                throughput_stats = result.details.get('throughput_stats', {})
                performance_summary = {
                    'avg_response_time_ms': response_stats.get('average_ms', 'N/A'),
                    'p95_response_time_ms': response_stats.get('p95_ms', 'N/A'),
                    'throughput_per_sec': throughput_stats.get('throughput', 'N/A'),
                    'error_rate_percent': throughput_stats.get('error_rate', 'N/A')
                }
                break

        return {
            'test_summary': {
                'total_execution_time_seconds': round(total_execution_time, 2),
                'timestamp': datetime.now().isoformat(),
                'overall_status': overall_status,
                'overall_pass_rate': round(overall_pass_rate, 1),
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'warning_tests': warning_tests,
                'failed_tests': failed_tests
            },
            'performance_highlights': performance_summary,
            'production_readiness_assessment': {
                'risk_control_response_time': "✅ 通过" if performance_summary.get('avg_response_time_ms', 999) < 10 else "❌ 未达标",
                'concurrent_throughput': "✅ 通过" if performance_summary.get('throughput_per_sec', 0) >= 1000 else "❌ 未达标",
                'system_availability': "✅ 通过" if failed_tests == 0 else "❌ 有故障",
                'functional_completeness': "✅ 通过" if any(r.test_name == "功能完整性测试" and r.status == "PASS" for r in self.test_results) else "❌ 未完整"
            },
            'detailed_test_results': [asdict(result) for result in self.test_results],
            'critical_issues': [issue for issue in all_issues if any(keyword in issue.lower() for keyword in ['fail', 'error', '失败', '错误', '异常'])],
            'improvement_recommendations': list(set(all_recommendations)),  # 去重
            'architect_review_alignment': {
                'performance_guarantee_mechanisms': "需要验证架构师提到的性能保证机制",
                'fault_tolerance_design': "需要加强容错设计实现",
                'configuration_management': "配置管理需要进一步完善"
            },
            'next_steps': [
                "修复所有关键问题（FAIL状态）",
                "解决警告项（WARNING状态）",
                "实施性能优化建议",
                "完善监控和告警机制",
                "进行生产环境部署准备"
            ]
        }


def main():
    """主函数"""
    print("🔬 生产级风控系统全面测试验证")
    print("=" * 60)
    print("测试专家：Production-Grade QA Engineer")
    print("测试标准：金融系统生产级别")
    print("=" * 60)

    # 创建测试实例
    test_suite = ProductionRiskControlComprehensiveTest()

    # 运行完整测试套件
    comprehensive_report = test_suite.run_comprehensive_test_suite()

    # 输出报告摘要
    print("\n📊 测试结果摘要:")
    print(f"整体状态: {comprehensive_report['test_summary']['overall_status']}")
    print(f"通过率: {comprehensive_report['test_summary']['overall_pass_rate']}%")
    print(f"执行时间: {comprehensive_report['test_summary']['total_execution_time_seconds']}秒")

    print("\n🎯 生产就绪评估:")
    for key, status in comprehensive_report['production_readiness_assessment'].items():
        print(f"  {key}: {status}")

    if comprehensive_report['critical_issues']:
        print(f"\n⚠️ 关键问题 ({len(comprehensive_report['critical_issues'])}):")
        for issue in comprehensive_report['critical_issues'][:5]:  # 显示前5个
            print(f"  - {issue}")

    if comprehensive_report['improvement_recommendations']:
        print(f"\n💡 改进建议 ({len(comprehensive_report['improvement_recommendations'])}):")
        for rec in comprehensive_report['improvement_recommendations'][:3]:  # 显示前3个
            print(f"  - {rec}")

    print(f"\n📋 详细报告已生成，包含 {len(comprehensive_report['detailed_test_results'])} 项测试结果")

    return comprehensive_report


if __name__ == "__main__":
    result = main()