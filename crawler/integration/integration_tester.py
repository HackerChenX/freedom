"""
集成测试器

端到端集成测试用例，验证爬虫系统与现有系统的兼容性
"""

import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class TestCase:
    """测试用例基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time = None
        self.end_time = None
        self.status = 'pending'  # pending, running, passed, failed
        self.error_message = None
        self.test_data = {}

    def run(self) -> bool:
        """运行测试用例"""
        self.start_time = datetime.now()
        self.status = 'running'

        try:
            logger.info(f"开始测试: {self.name}")
            result = self.execute()

            if result:
                self.status = 'passed'
                logger.info(f"测试通过: {self.name}")
            else:
                self.status = 'failed'
                logger.error(f"测试失败: {self.name}")

            return result

        except Exception as e:
            self.status = 'failed'
            self.error_message = str(e)
            logger.error(f"测试异常: {self.name} - {e}")
            return False

        finally:
            self.end_time = datetime.now()

    def execute(self) -> bool:
        """执行测试逻辑，子类需要实现"""
        raise NotImplementedError

    def get_duration(self) -> float:
        """获取测试执行时间"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def get_result(self) -> Dict[str, Any]:
        """获取测试结果"""
        return {
            'name': self.name,
            'description': self.description,
            'status': self.status,
            'duration': self.get_duration(),
            'error_message': self.error_message,
            'test_data': self.test_data
        }


class CrawlerSystemTest(TestCase):
    """爬虫系统基础测试"""

    def __init__(self):
        super().__init__(
            name="crawler_system_test",
            description="测试爬虫系统基础功能"
        )

    def execute(self) -> bool:
        """执行爬虫系统测试"""
        try:
            # 测试概念股提取器
            from crawler.processors.concept_extractor import ConceptStockExtractor

            extractor = ConceptStockExtractor()
            test_text = "赛轮轮胎(601058)、森麒麟(002984)等新能源概念股值得关注"
            result = extractor.extract_stocks(test_text)

            # 验证结果
            if not result['stock_codes']:
                self.error_message = "股票代码提取失败"
                return False

            if '601058' not in result['stock_codes'] or '002984' not in result['stock_codes']:
                self.error_message = "股票代码提取不完整"
                return False

            self.test_data['extraction_result'] = result
            return True

        except Exception as e:
            self.error_message = f"爬虫系统测试异常: {e}"
            return False


class MonitoringSystemTest(TestCase):
    """监控系统集成测试"""

    def __init__(self):
        super().__init__(
            name="monitoring_system_test",
            description="测试监控系统与爬虫系统的集成"
        )

    def execute(self) -> bool:
        """执行监控系统测试"""
        try:
            from crawler.monitoring.performance_monitor import PerformanceMonitor
            from crawler.monitoring.alert_manager import AlertManager
            from crawler.monitoring.data_quality_checker import DataQualityChecker

            # 测试性能监控
            monitor = PerformanceMonitor()
            monitor.record_request(True, 2.5)
            monitor.record_request(False, 0.0, 'timeout')

            metrics = monitor.get_metrics()
            if metrics['total_requests'] != 2:
                self.error_message = "性能监控记录错误"
                return False

            # 测试告警管理
            alert_manager = AlertManager()
            test_metrics = {
                'error_rate': 20.0,  # 触发告警
                'total_requests': 100
            }
            alerts = alert_manager.check_alerts(test_metrics)

            if not alerts:
                self.error_message = "告警系统未正常触发"
                return False

            # 测试数据质量检查
            checker = DataQualityChecker()
            test_data = {
                'id': 'test_001',
                'title': '测试标题',
                'content': '测试内容',
                'source': '测试源',
                'url': 'https://example.com/test'
            }

            quality_result = checker.check_data(test_data)
            if quality_result['quality_score'] < 50:
                self.error_message = "数据质量检查异常"
                return False

            self.test_data = {
                'performance_metrics': metrics,
                'alerts_triggered': len(alerts),
                'quality_score': quality_result['quality_score']
            }

            return True

        except Exception as e:
            self.error_message = f"监控系统测试异常: {e}"
            return False


class DatabaseIntegrationTest(TestCase):
    """数据库集成测试"""

    def __init__(self):
        super().__init__(
            name="database_integration_test",
            description="测试与ClickHouse数据库的集成"
        )

    def execute(self) -> bool:
        """执行数据库集成测试"""
        try:
            from crawler.integration.system_integrator import SystemIntegrator

            # 创建系统集成器
            integrator = SystemIntegrator()

            # 初始化集成系统
            if not integrator.initialize():
                self.error_message = "系统集成初始化失败"
                return False

            # 测试文章数据同步
            test_article = {
                'id': 'test_article_001',
                'title': '测试文章标题',
                'content': '这是一篇测试文章，包含股票信息：赛轮轮胎(601058)',
                'source': '测试源',
                'url': 'https://example.com/test',
                'author': '测试作者',
                'stock_codes': ['601058'],
                'concepts': ['轮胎', '新材料']
            }

            if not integrator.sync_article_data(test_article):
                self.error_message = "文章数据同步失败"
                return False

            # 检查集成状态
            status = integrator.get_integration_status()
            if not status['clickhouse_connected']:
                self.error_message = "ClickHouse连接状态异常"
                return False

            # 健康检查
            health = integrator.health_check()
            if health['overall_status'] == 'unhealthy':
                self.error_message = f"系统健康检查失败: {health['issues']}"
                return False

            self.test_data = {
                'integration_status': status,
                'health_status': health
            }

            return True

        except Exception as e:
            self.error_message = f"数据库集成测试异常: {e}"
            return False


class IntegrationTester:
    """集成测试器主类"""

    def __init__(self):
        self.test_cases = []
        self.test_results = []

        # 添加测试用例
        self._add_test_cases()

    def _add_test_cases(self):
        """添加测试用例"""
        self.test_cases = [
            CrawlerSystemTest(),
            MonitoringSystemTest(),
            DatabaseIntegrationTest()
        ]

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试用例"""
        logger.info("开始运行集成测试...")
        start_time = datetime.now()

        self.test_results = []
        passed_count = 0
        failed_count = 0

        for test_case in self.test_cases:
            result = test_case.run()
            test_result = test_case.get_result()
            self.test_results.append(test_result)

            if result:
                passed_count += 1
            else:
                failed_count += 1

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # 计算总体结果
        total_tests = len(self.test_cases)
        success_rate = (passed_count / total_tests * 100) if total_tests > 0 else 0

        overall_result = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration': total_duration,
            'total_tests': total_tests,
            'passed_tests': passed_count,
            'failed_tests': failed_count,
            'success_rate': success_rate,
            'overall_status': 'PASSED' if failed_count == 0 else 'FAILED',
            'test_results': self.test_results
        }

        logger.info(f"集成测试完成: {passed_count}/{total_tests} 通过")
        return overall_result

    def run_specific_test(self, test_name: str) -> Optional[Dict[str, Any]]:
        """运行特定测试用例"""
        for test_case in self.test_cases:
            if test_case.name == test_name:
                result = test_case.run()
                return test_case.get_result()

        logger.warning(f"未找到测试用例: {test_name}")
        return None

    def get_test_summary(self) -> Dict[str, Any]:
        """获取测试摘要"""
        if not self.test_results:
            return {'message': '尚未运行测试'}

        summary = {
            'total_tests': len(self.test_results),
            'passed_tests': len([r for r in self.test_results if r['status'] == 'passed']),
            'failed_tests': len([r for r in self.test_results if r['status'] == 'failed']),
            'total_duration': sum(r['duration'] for r in self.test_results),
            'failed_test_names': [r['name'] for r in self.test_results if r['status'] == 'failed']
        }

        summary['success_rate'] = (summary['passed_tests'] / summary['total_tests'] * 100) if summary['total_tests'] > 0 else 0

        return summary

    def generate_test_report(self) -> str:
        """生成测试报告"""
        if not self.test_results:
            return "尚未运行测试，无法生成报告"

        summary = self.get_test_summary()

        report = f"""
=== 股市信息爬虫系统集成测试报告 ===

测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

总体结果:
- 总测试数: {summary['total_tests']}
- 通过测试: {summary['passed_tests']}
- 失败测试: {summary['failed_tests']}
- 成功率: {summary['success_rate']:.1f}%
- 总耗时: {summary['total_duration']:.2f}秒

详细结果:
"""

        for result in self.test_results:
            status_icon = "✅" if result['status'] == 'passed' else "❌"
            report += f"\n{status_icon} {result['name']}: {result['status'].upper()}"
            report += f" ({result['duration']:.2f}秒)"

            if result['error_message']:
                report += f"\n   错误: {result['error_message']}"

            if result['test_data']:
                report += f"\n   测试数据: {result['test_data']}"

        if summary['failed_tests'] > 0:
            report += f"\n\n⚠️  失败的测试: {', '.join(summary['failed_test_names'])}"
            report += "\n建议检查相关模块的配置和依赖"
        else:
            report += "\n\n🎉 所有测试通过！系统集成正常"

        return report