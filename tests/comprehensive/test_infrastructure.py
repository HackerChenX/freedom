#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票选股策略系统综合测试基础设施

提供统一的测试基础设施管理，包括测试执行引擎、配置管理、监控系统等核心组件。
严格遵循六层架构原则，确保测试系统的可维护性和扩展性。

L6: 测试应用层 - 本文件提供测试执行和管理功能
L5: 测试业务层 - 各专门测试器（StockSelectionTester等）
L4: 测试服务层 - 指标测试、数据质量测试等服务
L3: 测试数据层 - 测试数据管理和模拟服务
L2: 测试基础设施层 - 本文件的基础功能
L1: 测试数据存储层 - 测试数据和结果存储
"""

import os
import sys
import time
import json
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from .config import get_test_config, TestEnvironmentConfig
from .logging_config import get_test_logger
from db.sql_manager import SQLManager, QueryType

logger = get_test_logger('test_infrastructure')


class TestStatus(Enum):
    """测试状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestPriority(Enum):
    """测试优先级枚举"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestResult:
    """单个测试结果"""
    test_name: str
    test_type: str
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    priority: TestPriority = TestPriority.MEDIUM
    
    @property
    def success(self) -> bool:
        """测试是否成功"""
        return self.status == TestStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'test_name': self.test_name,
            'test_type': self.test_type,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'metrics': self.metrics,
            'details': self.details,
            'priority': self.priority.value,
            'success': self.success
        }


@dataclass
class TestSuiteResult:
    """测试套件结果"""
    suite_name: str
    test_results: List[TestResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def total_tests(self) -> int:
        """总测试数"""
        return len(self.test_results)
    
    @property
    def passed_tests(self) -> int:
        """通过测试数"""
        return len([r for r in self.test_results if r.status == TestStatus.PASSED])
    
    @property
    def failed_tests(self) -> int:
        """失败测试数"""
        return len([r for r in self.test_results if r.status == TestStatus.FAILED])
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    @property
    def execution_time(self) -> float:
        """执行时间"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        """初始化性能监控器"""
        self.monitoring = False
        self.metrics = {}
        self.start_time = None
        self.monitor_thread = None
        
    def start_monitoring(self, test_name: str) -> None:
        """开始监控"""
        self.test_name = test_name
        self.monitoring = True
        self.start_time = time.time()
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': [],
            'peak_memory': 0.0,
            'average_cpu': 0.0
        }
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.debug(f"开始监控测试 {test_name} 的性能指标")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """停止监控并返回指标"""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # 计算汇总指标
        if self.metrics['cpu_usage']:
            self.metrics['average_cpu'] = sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage'])
        
        if self.metrics['memory_usage']:
            self.metrics['peak_memory'] = max(self.metrics['memory_usage'])
        
        execution_time = time.time() - self.start_time if self.start_time else 0.0
        self.metrics['execution_time'] = execution_time
        
        logger.debug(f"停止监控，执行时间: {execution_time:.2f}秒")
        return self.metrics.copy()
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=None)
                self.metrics['cpu_usage'].append(cpu_percent)
                
                # 内存使用情况
                memory = psutil.virtual_memory()
                memory_usage_gb = memory.used / (1024**3)
                self.metrics['memory_usage'].append(memory_usage_gb)
                
                # 磁盘I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.metrics['disk_io'].append({
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes
                    })
                
                # 网络I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    self.metrics['network_io'].append({
                        'bytes_sent': network_io.bytes_sent,
                        'bytes_recv': network_io.bytes_recv
                    })
                
                time.sleep(1.0)  # 每秒监控一次
                
            except Exception as e:
                logger.warning(f"性能监控异常: {e}")
                time.sleep(1.0)


class TestRunner:
    """测试运行器"""
    
    def __init__(self, max_workers: int = 4):
        """
        初始化测试运行器
        
        Args:
            max_workers: 最大并发工作线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.performance_monitor = PerformanceMonitor()
        self.running_tests = {}
        
    def run_test(self, test_func: Callable, test_name: str, 
                 test_type: str = "general", priority: TestPriority = TestPriority.MEDIUM,
                 **kwargs) -> TestResult:
        """
        运行单个测试
        
        Args:
            test_func: 测试函数
            test_name: 测试名称
            test_type: 测试类型
            priority: 测试优先级
            **kwargs: 传递给测试函数的参数
            
        Returns:
            TestResult: 测试结果
        """
        result = TestResult(
            test_name=test_name,
            test_type=test_type,
            status=TestStatus.RUNNING,
            start_time=datetime.now(),
            priority=priority
        )
        
        # 开始性能监控
        self.performance_monitor.start_monitoring(test_name)
        
        try:
            logger.info(f"开始执行测试: {test_name}")
            
            # 执行测试
            test_result = test_func(**kwargs)
            
            # 更新结果
            result.status = TestStatus.PASSED
            result.details = test_result if isinstance(test_result, dict) else {'result': test_result}
            
            logger.info(f"测试通过: {test_name}")
            
        except Exception as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
            logger.error(f"测试失败: {test_name}, 错误: {e}")
            
        finally:
            # 停止性能监控
            performance_metrics = self.performance_monitor.stop_monitoring()
            result.metrics = performance_metrics
            
            # 更新结束时间和执行时间
            result.end_time = datetime.now()
            result.execution_time = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    def run_test_suite(self, test_suite: Dict[str, Callable], 
                      suite_name: str, parallel: bool = True) -> TestSuiteResult:
        """
        运行测试套件
        
        Args:
            test_suite: 测试套件字典 {test_name: test_function}
            suite_name: 套件名称
            parallel: 是否并行执行
            
        Returns:
            TestSuiteResult: 测试套件结果
        """
        suite_result = TestSuiteResult(
            suite_name=suite_name,
            start_time=datetime.now()
        )
        
        logger.info(f"开始执行测试套件: {suite_name}, 包含 {len(test_suite)} 个测试")
        
        if parallel:
            # 并行执行
            futures = {}
            for test_name, test_func in test_suite.items():
                future = self.executor.submit(
                    self.run_test, test_func, test_name, suite_name
                )
                futures[future] = test_name
            
            # 收集结果
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5分钟超时
                    suite_result.test_results.append(result)
                except Exception as e:
                    # 创建失败结果
                    failed_result = TestResult(
                        test_name=futures[future],
                        test_type=suite_name,
                        status=TestStatus.ERROR,
                        start_time=datetime.now(),
                        error_message=f"测试执行异常: {e}"
                    )
                    suite_result.test_results.append(failed_result)
        else:
            # 串行执行
            for test_name, test_func in test_suite.items():
                result = self.run_test(test_func, test_name, suite_name)
                suite_result.test_results.append(result)
        
        suite_result.end_time = datetime.now()
        
        logger.info(f"测试套件完成: {suite_name}, "
                   f"成功率: {suite_result.success_rate:.1%}, "
                   f"执行时间: {suite_result.execution_time:.2f}秒")
        
        return suite_result
    
    def cleanup(self):
        """清理资源"""
        self.executor.shutdown(wait=True)


class ComprehensiveTestEngine:
    """综合测试执行引擎"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化综合测试引擎
        
        Args:
            config_file: 配置文件路径
        """
        self.config = get_test_config(config_file)
        self.test_runner = TestRunner(max_workers=self.config.test_execution.max_concurrent_tests)
        self.test_suites = {}
        self.results = {}
        
        # 创建输出目录
        self.output_dir = Path(self.config.reporting.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("综合测试引擎初始化完成")
    
    def register_test_suite(self, suite_name: str, test_suite: Dict[str, Callable]) -> None:
        """
        注册测试套件
        
        Args:
            suite_name: 套件名称
            test_suite: 测试套件
        """
        self.test_suites[suite_name] = test_suite
        logger.info(f"注册测试套件: {suite_name}, 包含 {len(test_suite)} 个测试")
    
    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def run_all_tests(self) -> Dict[str, Any]:
        """
        执行所有测试套件
        
        Returns:
            Dict[str, Any]: 综合测试结果
        """
        logger.info("开始执行综合测试")
        start_time = datetime.now()
        
        # 执行所有测试套件
        for suite_name, test_suite in self.test_suites.items():
            suite_result = self.test_runner.run_test_suite(
                test_suite, suite_name, 
                parallel=True
            )
            self.results[suite_name] = suite_result
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # 计算总体统计
        total_tests = sum(suite.total_tests for suite in self.results.values())
        total_passed = sum(suite.passed_tests for suite in self.results.values())
        total_failed = sum(suite.failed_tests for suite in self.results.values())
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        summary = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_execution_time': total_time,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'overall_success_rate': overall_success_rate,
            'suite_results': {name: suite.test_results for name, suite in self.results.items()},
            'configuration': {
                'max_concurrent_tests': self.config.test_execution.max_concurrent_tests,
                'test_timeout': self.config.test_execution.test_timeout,
                'performance_thresholds': self.config.performance_thresholds.__dict__
            }
        }
        
        # 保存结果
        self._save_results(summary)
        
        logger.info(f"综合测试完成，总体成功率: {overall_success_rate:.1%}, "
                   f"执行时间: {total_time:.2f}秒")
        
        return summary
    
    def run_specific_suite(self, suite_name: str) -> Optional[TestSuiteResult]:
        """
        执行特定测试套件
        
        Args:
            suite_name: 套件名称
            
        Returns:
            Optional[TestSuiteResult]: 测试套件结果，如果套件不存在则返回None
        """
        if suite_name not in self.test_suites:
            logger.error(f"测试套件不存在: {suite_name}")
            return None
        
        test_suite = self.test_suites[suite_name]
        result = self.test_runner.run_test_suite(test_suite, suite_name)
        self.results[suite_name] = result
        
        return result
    
    def _save_results(self, summary: Dict[str, Any]) -> None:
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON格式结果
        json_file = self.output_dir / f"comprehensive_test_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存文本格式摘要
        txt_file = self.output_dir / f"comprehensive_test_summary_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("股票选股策略系统综合测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试时间: {summary['start_time']} - {summary['end_time']}\n")
            f.write(f"总执行时间: {summary['total_execution_time']:.2f}秒\n")
            f.write(f"总测试数: {summary['total_tests']}\n")
            f.write(f"通过数: {summary['total_passed']}\n")
            f.write(f"失败数: {summary['total_failed']}\n")
            f.write(f"成功率: {summary['overall_success_rate']:.1%}\n\n")
            
            for suite_name, suite_result in self.results.items():
                f.write(f"测试套件: {suite_name}\n")
                f.write(f"  测试数: {suite_result.total_tests}\n")
                f.write(f"  通过数: {suite_result.passed_tests}\n")
                f.write(f"  失败数: {suite_result.failed_tests}\n")
                f.write(f"  成功率: {suite_result.success_rate:.1%}\n")
                f.write(f"  执行时间: {suite_result.execution_time:.2f}秒\n\n")
        
        logger.info(f"测试结果已保存: {json_file}")
    
    def cleanup(self):
        """清理资源"""
        self.test_runner.cleanup()
        logger.info("综合测试引擎资源清理完成")


def initialize_test_infrastructure() -> ComprehensiveTestEngine:
    """
    初始化测试基础设施
    
    Returns:
        ComprehensiveTestEngine: 综合测试引擎实例
    """
    logger.info("正在初始化测试基础设施...")
    
    try:
        # 初始化测试环境
        config = get_test_config()
        
        # 验证配置
        if not config.validate_config():
            raise RuntimeError("测试环境配置验证失败")
        
        # 创建测试引擎
        engine = ComprehensiveTestEngine()
        
        logger.info("测试基础设施初始化成功")
        return engine
        
    except Exception as e:
        logger.error(f"测试基础设施初始化失败: {e}")
        raise


if __name__ == "__main__":
    # 测试基础设施功能
    engine = initialize_test_infrastructure()
    
    # 示例测试套件
    def sample_test_1():
        time.sleep(1)
        return {"result": "success", "value": 42}
    
    def sample_test_2():
        time.sleep(0.5)
        return {"result": "success", "value": 24}
    
    def sample_test_3():
        raise ValueError("示例测试失败")
    
    sample_suite = {
        "test_1": sample_test_1,
        "test_2": sample_test_2,
        "test_3": sample_test_3
    }
    
    # 注册并运行测试
    engine.register_test_suite("sample_suite", sample_suite)
    results = engine.run_all_tests()
    
    print(f"测试完成，总体成功率: {results['overall_success_rate']:.1%}")
    
    # 清理资源
    engine.cleanup() 