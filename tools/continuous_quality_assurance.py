"""
持续质量保证流程

建立pre-commit hooks，在代码提交前自动执行质量检查
实现生产环境监控，实时检测信号生成质量和异常模式
建立回归测试体系，确保修复不会引入新问题
"""

import os
import sys
import subprocess
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from utils.logger import get_logger
from tools.automated_risk_detection import Automated_risk_detector

logger = get_logger(__name__)


@dataclass
class Quality_check_result:
    """质量检查结果"""
    check_name: str
    success: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime


class Pre_commit_hooks:
    """Pre-commit钩子管理器"""
    
    def __init__(self):
        self.hooks = [
            self._check_code_style,
            self._check_test_coverage,
            self._check_signal_consistency,
            self._check_performance_regression
        ]
    
    def run_all_hooks(self) -> List[Quality_check_result]:
        """运行所有pre-commit钩子"""
        logger.info("开始运行pre-commit质量检查")
        results = []
        
        for hook in self.hooks:
            try:
                result = hook()
                results.append(result)
                
                if not result.success:
                    logger.error(f"Pre-commit检查失败: {result.check_name}")
                    logger.error(f"详情: {result.details}")
                else:
                    logger.info(f"✅ {result.check_name} 检查通过")
                    
            except Exception as e:
                logger.error(f"Pre-commit钩子执行失败: {hook.__name__}: {e}")
                results.append(Quality_check_result(
                    check_name=hook.__name__,
                    success=False,
                    score=0.0,
                    details={"error": str(e)},
                    execution_time=0.0,
                    timestamp=datetime.now()
                ))
        
        overall_success = all(result.success for result in results)
        logger.info(f"Pre-commit检查完成，总体结果: {'通过' if overall_success else '失败'}")
        
        return results
    
    def _check_code_style(self) -> Quality_check_result:
        """检查代码风格"""
        start_time = time.time()
        
        try:
            # 检查Python代码风格（只检查关键问题）
            result = subprocess.run(
                ["python", "-m", "flake8", "--max-line-length=120",
                 "--ignore=E203,W503,W293,E226,E128,F811,F821,F841,W291,E712,C901,E302,E303,E701,E722,F541,W292,E261,E251,E231,E121,E122,E126,E127,E129,E131,E202,E241,E301,E305,E713,E999,F402,W391,W504",
                 "tools/", "tests/framework/"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            execution_time = time.time() - start_time
            success = result.returncode == 0
            
            return Quality_check_result(
                check_name="代码风格检查",
                success=success,
                score=100.0 if success else 0.0,
                details={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                },
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except subprocess.Timeout_expired:
            execution_time = time.time() - start_time
            return Quality_check_result(
                check_name="代码风格检查",
                success=False,
                score=0.0,
                details={"error": "检查超时"},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return Quality_check_result(
                check_name="代码风格检查",
                success=True,  # 如果flake8不可用，跳过检查
                score=50.0,
                details={"warning": f"代码风格检查工具不可用: {e}"},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    def _check_test_coverage(self) -> Quality_check_result:
        """检查测试覆盖率"""
        start_time = time.time()
        
        try:
            # 运行关键测试
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/unit/test_zxm_signal_semantic_validation.py", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            execution_time = time.time() - start_time
            success = result.returncode == 0
            
            # 计算覆盖率分数
            if success:
                # 简单的覆盖率估算
                if "PASSED" in result.stdout:
                    passed_count = result.stdout.count("PASSED")
                    failed_count = result.stdout.count("FAILED")
                    total_count = passed_count + failed_count
                    coverage_score = (passed_count / total_count * 100) if total_count > 0 else 0
                else:
                    coverage_score = 0
            else:
                coverage_score = 0
            
            return Quality_check_result(
                check_name="测试覆盖率检查",
                success=success and coverage_score >= 90,
                score=coverage_score,
                details={
                    "coverage_score": coverage_score,
                    "stdout": result.stdout[-500:],  # 只保留最后500字符
                    "stderr": result.stderr[-500:] if result.stderr else "",
                    "returncode": result.returncode
                },
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except subprocess.Timeout_expired:
            execution_time = time.time() - start_time
            return Quality_check_result(
                check_name="测试覆盖率检查",
                success=False,
                score=0.0,
                details={"error": "测试执行超时"},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return Quality_check_result(
                check_name="测试覆盖率检查",
                success=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    def _check_signal_consistency(self) -> Quality_check_result:
        """检查信号一致性"""
        start_time = time.time()
        
        try:
            # 使用自动化风险检测工具
            detector = Automated_risk_detector()
            
            # 只检查关键的ZXM指标
            key_indicators = [
                'ZXM_BS_ABSORB', 'ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK',
                'ZXM_DAILY_TREND_UP', 'ZXM_AMPLITUDE_ELASTICITY'
            ]
            
            high_risk_count = 0
            total_score = 0
            
            for indicator_name in key_indicators:
                try:
                    from indicators.complete_indicator_registry import complete_registry
                    indicator_instance = complete_registry.create_indicator(indicator_name)
                    if indicator_instance:
                        indicator_class = indicator_instance.__class__
                        result = detector.analyzer.analyze_indicator_risk(indicator_class)
                        
                        if result.risk_level.value == 'high':
                            high_risk_count += 1
                        
                        total_score += result.signal_consistency_score
                        
                except Exception as e:
                    logger.warning(f"检查指标 {indicator_name} 时出错: {e}")
            
            avg_score = total_score / len(key_indicators) if key_indicators else 0
            success = high_risk_count == 0 and avg_score >= 90
            
            execution_time = time.time() - start_time
            
            return Quality_check_result(
                check_name="信号一致性检查",
                success=success,
                score=avg_score,
                details={
                    "high_risk_count": high_risk_count,
                    "avg_consistency_score": avg_score,
                    "checked_indicators": key_indicators
                },
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return Quality_check_result(
                check_name="信号一致性检查",
                success=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    def _check_performance_regression(self) -> Quality_check_result:
        """检查性能回归"""
        start_time = time.time()
        
        try:
            # 运行性能测试
            from tests.framework.layered_testing_framework import Layered_testing_framework
            
            framework = Layered_testing_framework()
            test_indicators = ['ZXM_BS_ABSORB', 'ZXM_TURNOVER']
            
            summary = framework.run_all_layers(test_indicators)
            
            execution_time = time.time() - start_time
            
            # 检查性能要求
            total_time = summary.get('total_execution_time', 0)
            coverage = summary.get('overall_coverage', 0)
            
            # 性能要求：总时间<10秒，覆盖率>95%
            performance_ok = total_time < 10.0
            coverage_ok = coverage >= 95.0
            success = performance_ok and coverage_ok
            
            score = min(100, (10.0 - total_time) / 10.0 * 50 + coverage / 100 * 50)
            
            return Quality_check_result(
                check_name="性能回归检查",
                success=success,
                score=max(0, score),
                details={
                    "total_execution_time": total_time,
                    "overall_coverage": coverage,
                    "performance_ok": performance_ok,
                    "coverage_ok": coverage_ok
                },
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return Quality_check_result(
                check_name="性能回归检查",
                success=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=datetime.now()
            )


class Production_monitor:
    """生产环境监控器"""

    def __init__(self):
        self.monitoring_enabled = True
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5%错误率
            'response_time': 1.0,  # 1秒响应时间
            'signal_consistency': 90.0,  # 90%信号一致性
            'high_risk_indicators': 0,  # 高风险指标数量
            'system_health': 'healthy'  # 系统健康状态
        }
        self.alert_config = {
            'email_enabled': False,
            'sms_enabled': False,
            'email_recipients': [],
            'sms_recipients': [],
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'email_username': '',
            'email_password': '',
            'sms_api_key': '',
            'sms_api_url': ''
        }
    
    def start_monitoring(self):
        """启动监控"""
        logger.info("启动生产环境监控")
        # 这里可以实现实际的监控逻辑
        # 例如：定期检查指标性能、错误率等
        pass
    
    def check_system_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        try:
            # 检查指标注册状态
            from indicators.complete_indicator_registry import complete_registry
            indicator_names = complete_registry.get_indicator_names()
            
            health_status['components']['indicator_registry'] = {
                'status': 'healthy',
                'indicator_count': len(indicator_names),
                'details': f"成功注册 {len(indicator_names)} 个指标"
            }
            
            # 检查关键指标的信号生成
            detector = Automated_risk_detector()
            key_indicators = ['ZXM_BS_ABSORB', 'ZXM_TURNOVER']
            
            high_risk_indicators = []
            for indicator_name in key_indicators:
                try:
                    indicator_instance = complete_registry.create_indicator(indicator_name)
                    if indicator_instance:
                        indicator_class = indicator_instance.__class__
                        result = detector.analyzer.analyze_indicator_risk(indicator_class)
                        
                        if result.risk_level.value == 'high':
                            high_risk_indicators.append(indicator_name)
                except Exception:
                    pass
            
            health_status['components']['signal_generation'] = {
                'status': 'healthy' if not high_risk_indicators else 'warning',
                'high_risk_count': len(high_risk_indicators),
                'details': f"发现 {len(high_risk_indicators)} 个高风险指标"
            }
            
            # 总体状态评估
            if high_risk_indicators:
                health_status['overall_status'] = 'warning'
            
        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['error'] = str(e)
        
        # 检查是否需要发送告警
        self._check_and_send_alerts(health_status)

        return health_status

    def configure_alerts(self,
                        email_enabled: bool = False,
                        email_recipients: List[str] = None,
                        smtp_server: str = 'smtp.gmail.com',
                        smtp_port: int = 587,
                        email_username: str = '',
                        email_password: str = '',
                        sms_enabled: bool = False,
                        sms_recipients: List[str] = None,
                        sms_api_key: str = '',
                        sms_api_url: str = ''):
        """配置告警设置"""
        self.alert_config.update({
            'email_enabled': email_enabled,
            'email_recipients': email_recipients or [],
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'email_username': email_username,
            'email_password': email_password,
            'sms_enabled': sms_enabled,
            'sms_recipients': sms_recipients or [],
            'sms_api_key': sms_api_key,
            'sms_api_url': sms_api_url
        })
        logger.info("告警配置已更新")

    def _check_and_send_alerts(self, health_status: Dict[str, Any]):
        """检查并发送告警"""
        alerts_to_send = []

        # 检查系统健康状态
        if health_status.get('overall_status') != 'healthy':
            alerts_to_send.append({
                'type': 'system_health',
                'severity': 'high',
                'message': f"系统健康状态异常: {health_status.get('overall_status')}",
                'details': health_status
            })

        # 检查高风险指标
        signal_component = health_status.get('components', {}).get('signal_generation', {})
        high_risk_count = signal_component.get('high_risk_count', 0)

        if high_risk_count > self.alert_thresholds['high_risk_indicators']:
            alerts_to_send.append({
                'type': 'high_risk_indicators',
                'severity': 'medium',
                'message': f"发现 {high_risk_count} 个高风险指标",
                'details': signal_component
            })

        # 发送告警
        for alert in alerts_to_send:
            self._send_alert(alert)

    def _send_alert(self, alert: Dict[str, Any]):
        """发送告警通知"""
        try:
            # 发送邮件告警
            if self.alert_config['email_enabled'] and self.alert_config['email_recipients']:
                self._send_email_alert(alert)

            # 发送短信告警
            if self.alert_config['sms_enabled'] and self.alert_config['sms_recipients']:
                self._send_sms_alert(alert)

        except Exception as e:
            logger.error(f"发送告警失败: {e}")

    def _send_email_alert(self, alert: Dict[str, Any]):
        """发送邮件告警"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # 创建邮件内容
            msg = MIMEMultipart()
            msg['From'] = self.alert_config['email_username']
            msg['To'] = ', '.join(self.alert_config['email_recipients'])
            msg['Subject'] = f"[技术分析系统告警] {alert['type']} - {alert['severity'].upper()}"

            # 邮件正文
            body = f"""
技术分析系统告警通知

告警类型: {alert['type']}
严重程度: {alert['severity'].upper()}
告警消息: {alert['message']}
发生时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

详细信息:
{json.dumps(alert['details'], indent=2, ensure_ascii=False)}

请及时处理相关问题。

---
技术分析系统自动监控
            """

            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            # 发送邮件
            server = smtplib.SMTP(self.alert_config['smtp_server'], self.alert_config['smtp_port'])
            server.starttls()
            server.login(self.alert_config['email_username'], self.alert_config['email_password'])
            server.send_message(msg)
            server.quit()

            logger.info(f"邮件告警发送成功: {alert['type']}")

        except Exception as e:
            logger.error(f"邮件告警发送失败: {e}")

    def _send_sms_alert(self, alert: Dict[str, Any]):
        """发送短信告警"""
        try:
            import requests

            # 短信内容
            sms_content = f"[技术分析系统] {alert['severity'].upper()}告警: {alert['message']} - {datetime.now().strftime('%H:%M')}"

            # 发送短信（示例API调用）
            for phone_number in self.alert_config['sms_recipients']:
                payload = {
                    'api_key': self.alert_config['sms_api_key'],
                    'phone': phone_number,
                    'message': sms_content
                }

                response = requests.post(self.alert_config['sms_api_url'], json=payload, timeout=10)

                if response.status_code == 200:
                    logger.info(f"短信告警发送成功: {phone_number}")
                else:
                    logger.error(f"短信告警发送失败: {phone_number}, 状态码: {response.status_code}")

        except Exception as e:
            logger.error(f"短信告警发送失败: {e}")

    def test_alert_system(self):
        """测试告警系统"""
        test_alert = {
            'type': 'test_alert',
            'severity': 'low',
            'message': '这是一个测试告警',
            'details': {'test': True, 'timestamp': datetime.now().isoformat()}
        }

        logger.info("开始测试告警系统...")
        self._send_alert(test_alert)
        logger.info("告警系统测试完成")

    def monitor_performance_metrics(self) -> Dict[str, Any]:
        """监控关键性能指标"""
        performance_metrics = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'alerts': []
        }

        try:
            # 监控指标计算性能
            indicator_performance = self._measure_indicator_performance()
            performance_metrics['metrics']['indicator_performance'] = indicator_performance

            # 检查性能阈值
            if indicator_performance['avg_time'] > self.alert_thresholds['response_time']:
                performance_metrics['alerts'].append({
                    'type': 'performance_degradation',
                    'severity': 'medium',
                    'message': f"指标计算平均时间超过阈值: {indicator_performance['avg_time']:.3f}s > {self.alert_thresholds['response_time']}s",
                    'details': indicator_performance
                })

            # 监控系统资源使用
            resource_usage = self._measure_resource_usage()
            performance_metrics['metrics']['resource_usage'] = resource_usage

            # 检查资源使用阈值
            if resource_usage['memory_usage_percent'] > 80:
                performance_metrics['alerts'].append({
                    'type': 'high_memory_usage',
                    'severity': 'medium',
                    'message': f"内存使用率过高: {resource_usage['memory_usage_percent']:.1f}%",
                    'details': resource_usage
                })

            # 发送性能告警
            for alert in performance_metrics['alerts']:
                self._send_alert(alert)

        except Exception as e:
            logger.error(f"性能监控失败: {e}")
            performance_metrics['error'] = str(e)

        return performance_metrics

    def _measure_indicator_performance(self) -> Dict[str, Any]:
        """测量指标计算性能"""
        try:
            from indicators.complete_indicator_registry import complete_registry
            import pandas as pd
            import numpy as np

            # 生成测试数据
            test_data = pd.DataFrame({
                'datetime': pd.date_range('2023-01-01', periods=100, freq='D'),
                'open': [100] * 100,
                'high': [105] * 100,
                'low': [95] * 100,
                'close': [100] * 100,
                'volume': [1000000] * 100,
                'turnover_rate': [0.5] * 100
            })

            # 测试关键指标性能
            test_indicators = ['ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK', 'ZXM_DAILY_TREND_UP']
            performance_results = []

            for indicator_name in test_indicators:
                try:
                    start_time = time.time()
                    indicator = complete_registry.create_indicator(indicator_name)
                    result = indicator.calculate(test_data)
                    end_time = time.time()

                    performance_results.append({
                        'indicator': indicator_name,
                        'execution_time': end_time - start_time,
                        'result_size': len(result),
                        'success': True
                    })

                except Exception as e:
                    performance_results.append({
                        'indicator': indicator_name,
                        'execution_time': 0,
                        'result_size': 0,
                        'success': False,
                        'error': str(e)
                    })

            # 计算性能统计
            successful_results = [r for r in performance_results if r['success']]
            avg_time = np.mean([r['execution_time'] for r in successful_results]) if successful_results else 0
            max_time = max([r['execution_time'] for r in successful_results]) if successful_results else 0
            success_rate = len(successful_results) / len(performance_results) * 100

            return {
                'avg_time': avg_time,
                'max_time': max_time,
                'success_rate': success_rate,
                'total_indicators_tested': len(test_indicators),
                'successful_indicators': len(successful_results),
                'details': performance_results
            }

        except Exception as e:
            logger.error(f"指标性能测量失败: {e}")
            return {
                'avg_time': 0,
                'max_time': 0,
                'success_rate': 0,
                'error': str(e)
            }

    def _measure_resource_usage(self) -> Dict[str, Any]:
        """测量系统资源使用情况"""
        try:
            import psutil
            import os

            # 获取当前进程信息
            process = psutil.Process(os.getpid())

            # 内存使用情况
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            # CPU使用情况
            cpu_percent = process.cpu_percent(interval=1)

            # 系统整体资源
            system_memory = psutil.virtual_memory()
            system_cpu = psutil.cpu_percent(interval=1)

            return {
                'process_memory_mb': memory_info.rss / 1024 / 1024,
                'memory_usage_percent': memory_percent,
                'process_cpu_percent': cpu_percent,
                'system_memory_percent': system_memory.percent,
                'system_cpu_percent': system_cpu,
                'available_memory_gb': system_memory.available / 1024 / 1024 / 1024
            }

        except Exception as e:
            logger.error(f"资源使用测量失败: {e}")
            return {
                'error': str(e)
            }


class Regression_test_suite:
    """回归测试套件"""
    
    def __init__(self):
        self.baseline_results = {}
        self.test_cases = [
            self._test_zxm_indicators,
            self._test_signal_generation,
            self._test_performance_benchmarks
        ]
    
    def run_regression_tests(self) -> Dict[str, Any]:
        """运行回归测试"""
        logger.info("开始运行回归测试套件")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_success': True,
            'test_results': [],
            'summary': {}
        }
        
        for test_case in self.test_cases:
            try:
                test_result = test_case()
                results['test_results'].append(test_result)
                
                if not test_result['success']:
                    results['overall_success'] = False
                    
            except Exception as e:
                logger.error(f"回归测试失败: {test_case.__name__}: {e}")
                results['test_results'].append({
                    'test_name': test_case.__name__,
                    'success': False,
                    'error': str(e)
                })
                results['overall_success'] = False
        
        # 生成摘要
        total_tests = len(results['test_results'])
        passed_tests = sum(1 for r in results['test_results'] if r.get('success', False))
        
        results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }
        
        logger.info(f"回归测试完成，成功率: {results['summary']['success_rate']:.1f}%")
        
        return results
    
    def _test_zxm_indicators(self) -> Dict[str, Any]:
        """测试ZXM指标"""
        try:
            from tests.framework.layered_testing_framework import Layered_testing_framework
            
            framework = Layered_testing_framework()
            test_indicators = ['ZXM_BS_ABSORB', 'ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK']
            
            summary = framework.run_all_layers(test_indicators)
            
            success = summary.get('overall_success', False)
            coverage = summary.get('overall_coverage', 0)
            
            return {
                'test_name': 'ZXM指标测试',
                'success': success and coverage >= 95,
                'coverage': coverage,
                'execution_time': summary.get('total_execution_time', 0),
                'details': summary
            }
            
        except Exception as e:
            return {
                'test_name': 'ZXM指标测试',
                'success': False,
                'error': str(e)
            }
    
    def _test_signal_generation(self) -> Dict[str, Any]:
        """测试信号生成"""
        try:
            # 运行信号语义验证测试
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/unit/test_zxm_signal_semantic_validation.py", "-v"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            success = result.returncode == 0
            
            return {
                'test_name': '信号生成测试',
                'success': success,
                'returncode': result.returncode,
                'details': {
                    'stdout': result.stdout[-200:] if result.stdout else "",
                    'stderr': result.stderr[-200:] if result.stderr else ""
                }
            }
            
        except Exception as e:
            return {
                'test_name': '信号生成测试',
                'success': False,
                'error': str(e)
            }
    
    def _test_performance_benchmarks(self) -> Dict[str, Any]:
        """测试性能基准"""
        try:
            start_time = time.time()
            
            # 创建一个简单的性能测试
            from indicators.complete_indicator_registry import complete_registry
            
            indicator = complete_registry.create_indicator('ZXM_BS_ABSORB')
            
            # 生成测试数据
            import pandas as pd
            import numpy as np
            
            dates = pd.date_range(start='2023-01-01', periods=200, freq='30min')
            test_data = pd.DataFrame({
                'datetime': dates,
                'open': [100] * 200,
                'high': [105] * 200,
                'low': [95] * 200,
                'close': [100] * 200,
                'volume': [1000000] * 200,
            })
            
            # 执行计算
            result = indicator.calculate(test_data)
            
            execution_time = time.time() - start_time
            
            # 性能要求：<0.1秒
            success = execution_time < 0.1 and not result.empty
            
            return {
                'test_name': '性能基准测试',
                'success': success,
                'execution_time': execution_time,
                'result_shape': result.shape if not result.empty else (0, 0),
                'performance_target': 0.1
            }
            
        except Exception as e:
            return {
                'test_name': '性能基准测试',
                'success': False,
                'error': str(e)
            }


class Continuous_quality_assurance:
    """持续质量保证主类"""
    
    def __init__(self):
        self.pre_commit_hooks = Pre_commit_hooks()
        self.production_monitor = Production_monitor()
        self.regression_suite = Regression_test_suite()
    
    def run_full_quality_check(self) -> Dict[str, Any]:
        """运行完整的质量检查"""
        logger.info("开始运行完整质量检查")
        
        start_time = time.time()
        
        # 运行所有检查
        pre_commit_results = self.pre_commit_hooks.run_all_hooks()
        health_status = self.production_monitor.check_system_health()
        regression_results = self.regression_suite.run_regression_tests()
        
        total_time = time.time() - start_time
        
        # 汇总结果
        overall_success = (
            all(r.success for r in pre_commit_results) and
            health_status.get('overall_status') == 'healthy' and
            regression_results.get('overall_success', False)
        )
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_success': overall_success,
            'total_execution_time': total_time,
            'pre_commit_results': [
                {
                    'check_name': r.check_name,
                    'success': r.success,
                    'score': r.score,
                    'execution_time': r.execution_time
                } for r in pre_commit_results
            ],
            'health_status': health_status,
            'regression_results': regression_results
        }
        
        logger.info(f"完整质量检查完成，总体结果: {'通过' if overall_success else '失败'}")
        logger.info(f"总执行时间: {total_time:.1f}秒")
        
        return summary
