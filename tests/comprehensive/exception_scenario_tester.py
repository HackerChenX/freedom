"""
异常场景测试模块
提供错误信息和恢复机制验证
"""

import os
import sys
import time
import traceback
import threading
import signal
import random
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Type
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

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
class ExceptionScenario:
    """异常场景定义"""
    scenario_name: str
    scenario_type: str
    description: str
    exception_type: Type[Exception]
    trigger_condition: str
    expected_recovery: str
    recovery_time_limit: float = 30.0


@dataclass
class ExceptionTestResult:
    """异常测试结果"""
    test_name: str
    scenario_name: str
    exception_triggered: bool
    exception_caught: bool
    error_message_quality: float  # 0-1评分
    recovery_attempted: bool
    recovery_successful: bool
    recovery_time: float
    test_status: TestStatus
    execution_time: float
    error_details: Optional[str] = None
    recovery_details: Optional[str] = None


@dataclass
class ExceptionTestSuite:
    """异常测试套件结果"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    total_execution_time: float
    test_results: List[ExceptionTestResult] = field(default_factory=list)
    exception_handling_score: float = 0.0
    recovery_success_rate: float = 0.0


class ExceptionSimulator:
    """异常模拟器"""
    
    def __init__(self):
        self.active_simulations = {}
        self.failure_rates = {}
    
    def simulate_database_error(self, error_type: str = "connection"):
        """模拟数据库错误"""
        if error_type == "connection":
            raise ConnectionError("数据库连接失败：连接超时")
        elif error_type == "query":
            raise Exception("SQL查询执行失败：语法错误")
        elif error_type == "timeout":
            raise TimeoutError("数据库查询超时：查询执行时间过长")
        else:
            raise Exception(f"数据库未知错误：{error_type}")
    
    def simulate_memory_error(self):
        """模拟内存错误"""
        raise MemoryError("内存不足：无法分配更多内存")
    
    def simulate_computation_error(self, error_type: str = "division"):
        """模拟计算错误"""
        if error_type == "division":
            raise ZeroDivisionError("除零错误：分母为零")
        elif error_type == "overflow":
            raise OverflowError("数值溢出：计算结果超出范围")
        elif error_type == "value":
            raise ValueError("数值错误：无效的输入参数")
        else:
            raise ArithmeticError(f"计算错误：{error_type}")
    
    def simulate_io_error(self, error_type: str = "file_not_found"):
        """模拟IO错误"""
        if error_type == "file_not_found":
            raise FileNotFoundError("文件未找到：指定的文件不存在")
        elif error_type == "permission":
            raise PermissionError("权限错误：没有访问权限")
        elif error_type == "disk_full":
            raise OSError("磁盘空间不足：无法写入文件")
        else:
            raise IOError(f"IO错误：{error_type}")
    
    def simulate_network_error(self, error_type: str = "timeout"):
        """模拟网络错误"""
        if error_type == "timeout":
            raise TimeoutError("网络超时：请求响应超时")
        elif error_type == "connection":
            raise ConnectionError("网络连接失败：服务器不可达")
        elif error_type == "dns":
            raise Exception("DNS解析失败：域名无法解析")
        else:
            raise Exception(f"网络错误：{error_type}")
    
    def simulate_random_failure(self, failure_rate: float = 0.3):
        """模拟随机失败"""
        if random.random() < failure_rate:
            error_types = [
                ("database", "connection"),
                ("computation", "division"),
                ("io", "file_not_found"),
                ("network", "timeout")
            ]
            error_category, error_type = random.choice(error_types)
            
            if error_category == "database":
                self.simulate_database_error(error_type)
            elif error_category == "computation":
                self.simulate_computation_error(error_type)
            elif error_category == "io":
                self.simulate_io_error(error_type)
            elif error_category == "network":
                self.simulate_network_error(error_type)


class ErrorRecoveryManager:
    """错误恢复管理器"""
    
    def __init__(self):
        self.recovery_strategies = {}
        self.recovery_history = []
        self.max_retry_attempts = 3
        self.retry_delay = 1.0
    
    def register_recovery_strategy(self, exception_type: Type[Exception], strategy: Callable):
        """注册恢复策略"""
        self.recovery_strategies[exception_type] = strategy
    
    def attempt_recovery(self, exception: Exception, operation: Callable, *args, **kwargs) -> tuple[bool, Any, str]:
        """尝试错误恢复"""
        recovery_start_time = time.time()
        recovery_details = []
        
        exception_type = type(exception)
        
        # 查找具体的恢复策略
        strategy = self.recovery_strategies.get(exception_type)
        if not strategy:
            # 查找父类的恢复策略
            for exc_type, strat in self.recovery_strategies.items():
                if issubclass(exception_type, exc_type):
                    strategy = strat
                    break
        
        if not strategy:
            recovery_details.append("未找到适合的恢复策略")
            return False, None, "; ".join(recovery_details)
        
        # 执行恢复策略
        for attempt in range(self.max_retry_attempts):
            try:
                recovery_details.append(f"尝试恢复 (第{attempt + 1}次)")
                
                # 执行恢复策略
                strategy(exception, attempt)
                
                # 等待后重试
                if attempt > 0:
                    time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                
                # 重新执行原操作
                result = operation(*args, **kwargs)
                
                recovery_time = time.time() - recovery_start_time
                recovery_details.append(f"恢复成功 (耗时{recovery_time:.2f}秒)")
                
                # 记录成功的恢复
                self.recovery_history.append({
                    'exception_type': exception_type.__name__,
                    'recovery_time': recovery_time,
                    'attempts': attempt + 1,
                    'success': True
                })
                
                return True, result, "; ".join(recovery_details)
                
            except Exception as retry_exception:
                recovery_details.append(f"第{attempt + 1}次恢复失败: {str(retry_exception)}")
                if attempt == self.max_retry_attempts - 1:
                    # 最后一次尝试失败
                    recovery_time = time.time() - recovery_start_time
                    self.recovery_history.append({
                        'exception_type': exception_type.__name__,
                        'recovery_time': recovery_time,
                        'attempts': attempt + 1,
                        'success': False
                    })
                    return False, None, "; ".join(recovery_details)
        
        return False, None, "; ".join(recovery_details)
    
    def database_connection_recovery(self, exception: Exception, attempt: int):
        """数据库连接恢复策略"""
        if isinstance(exception, ConnectionError):
            # 模拟重置连接池
            time.sleep(0.5)
            logger.info(f"重置数据库连接池 (尝试{attempt + 1})")
        elif isinstance(exception, TimeoutError):
            # 模拟增加超时时间
            logger.info(f"增加数据库超时时间 (尝试{attempt + 1})")
    
    def memory_error_recovery(self, exception: Exception, attempt: int):
        """内存错误恢复策略"""
        if isinstance(exception, MemoryError):
            # 模拟垃圾回收
            import gc
            gc.collect()
            logger.info(f"执行垃圾回收 (尝试{attempt + 1})")
    
    def computation_error_recovery(self, exception: Exception, attempt: int):
        """计算错误恢复策略"""
        if isinstance(exception, ZeroDivisionError):
            logger.info(f"检测到除零错误，使用默认值 (尝试{attempt + 1})")
        elif isinstance(exception, (ValueError, OverflowError)):
            logger.info(f"检测到计算错误，使用安全计算模式 (尝试{attempt + 1})")
    
    def network_error_recovery(self, exception: Exception, attempt: int):
        """网络错误恢复策略"""
        if isinstance(exception, (ConnectionError, TimeoutError)):
            # 模拟切换网络或重试
            time.sleep(1.0)
            logger.info(f"网络重试 (尝试{attempt + 1})")


class ExceptionScenarioTester:
    """异常场景测试器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.exception_simulator = ExceptionSimulator()
        self.recovery_manager = ErrorRecoveryManager()
        self.exception_scenarios = self._define_exception_scenarios()
        self._setup_recovery_strategies()
    
    def _define_exception_scenarios(self) -> Dict[str, ExceptionScenario]:
        """定义异常场景"""
        return {
            'database_connection_failure': ExceptionScenario(
                scenario_name='database_connection_failure',
                scenario_type='database',
                description='数据库连接失败场景',
                exception_type=ConnectionError,
                trigger_condition='数据库服务不可用',
                expected_recovery='重置连接池并重试'
            ),
            'database_query_timeout': ExceptionScenario(
                scenario_name='database_query_timeout',
                scenario_type='database',
                description='数据库查询超时场景',
                exception_type=TimeoutError,
                trigger_condition='查询执行时间过长',
                expected_recovery='增加超时时间并重试'
            ),
            'memory_exhaustion': ExceptionScenario(
                scenario_name='memory_exhaustion',
                scenario_type='system',
                description='内存耗尽场景',
                exception_type=MemoryError,
                trigger_condition='系统可用内存不足',
                expected_recovery='释放内存并使用分批处理'
            ),
            'division_by_zero': ExceptionScenario(
                scenario_name='division_by_zero',
                scenario_type='computation',
                description='除零错误场景',
                exception_type=ZeroDivisionError,
                trigger_condition='计算中出现除零操作',
                expected_recovery='使用默认值或跳过计算'
            ),
            'invalid_data_format': ExceptionScenario(
                scenario_name='invalid_data_format',
                scenario_type='data',
                description='无效数据格式场景',
                exception_type=ValueError,
                trigger_condition='输入数据格式不符合要求',
                expected_recovery='数据清洗和格式转换'
            ),
            'network_timeout': ExceptionScenario(
                scenario_name='network_timeout',
                scenario_type='network',
                description='网络超时场景',
                exception_type=TimeoutError,
                trigger_condition='网络请求响应超时',
                expected_recovery='重试请求或使用缓存数据'
            ),
            'file_not_found': ExceptionScenario(
                scenario_name='file_not_found',
                scenario_type='io',
                description='文件未找到场景',
                exception_type=FileNotFoundError,
                trigger_condition='访问不存在的文件',
                expected_recovery='使用默认文件或跳过操作'
            ),
            'permission_denied': ExceptionScenario(
                scenario_name='permission_denied',
                scenario_type='io',
                description='权限拒绝场景',
                exception_type=PermissionError,
                trigger_condition='没有足够的访问权限',
                expected_recovery='降级访问或使用备选方案'
            )
        }
    
    def _setup_recovery_strategies(self):
        """设置恢复策略"""
        self.recovery_manager.register_recovery_strategy(
            ConnectionError, 
            self.recovery_manager.database_connection_recovery
        )
        self.recovery_manager.register_recovery_strategy(
            TimeoutError, 
            self.recovery_manager.network_error_recovery
        )
        self.recovery_manager.register_recovery_strategy(
            MemoryError, 
            self.recovery_manager.memory_error_recovery
        )
        self.recovery_manager.register_recovery_strategy(
            (ZeroDivisionError, ValueError, OverflowError), 
            self.recovery_manager.computation_error_recovery
        )
    
    def _evaluate_error_message_quality(self, error_message: str) -> float:
        """评估错误信息质量"""
        if not error_message:
            return 0.0
        
        quality_score = 0.0
        
        # 检查错误信息是否包含有用信息
        quality_indicators = [
            ('具体错误类型', any(word in error_message.lower() for word in ['error', '错误', 'exception', '异常'])),
            ('错误原因', any(word in error_message.lower() for word in ['because', '因为', 'due to', '由于'])),
            ('错误位置', any(word in error_message.lower() for word in ['line', '行', 'function', '函数', 'method', '方法'])),
            ('建议操作', any(word in error_message.lower() for word in ['try', '尝试', 'check', '检查', 'verify', '验证'])),
            ('错误代码', any(char.isdigit() for char in error_message)),
        ]
        
        for indicator_name, has_indicator in quality_indicators:
            if has_indicator:
                quality_score += 0.2
        
        # 检查错误信息长度是否合适
        if 10 <= len(error_message) <= 200:
            quality_score += 0.1
        
        # 检查是否为中文或英文
        if any('\u4e00' <= char <= '\u9fff' for char in error_message) or error_message.isascii():
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=60.0)
    def test_database_connection_failure_scenario(self) -> ExceptionTestResult:
        """测试数据库连接失败场景"""
        scenario = self.exception_scenarios['database_connection_failure']
        test_start_time = time.time()
        
        result = ExceptionTestResult(
            test_name='test_database_connection_failure',
            scenario_name=scenario.scenario_name,
            exception_triggered=False,
            exception_caught=False,
            error_message_quality=0.0,
            recovery_attempted=False,
            recovery_successful=False,
            recovery_time=0.0,
            test_status=TestStatus.FAILED,
            execution_time=0.0
        )
        
        def mock_database_operation():
            """模拟数据库操作"""
            self.exception_simulator.simulate_database_error("connection")
            return pd.DataFrame({'result': ['success']})
        
        try:
            # 第一步：触发异常
            try:
                mock_database_operation()
            except ConnectionError as e:
                result.exception_triggered = True
                result.exception_caught = True
                result.error_details = str(e)
                result.error_message_quality = self._evaluate_error_message_quality(str(e))
                
                self.logger.info(f"成功触发和捕获数据库连接异常: {e}")
                
                # 第二步：尝试恢复
                result.recovery_attempted = True
                recovery_start_time = time.time()
                
                try:
                    # 模拟恢复操作（这里简化为不再抛出异常）
                    def recovered_operation():
                        return pd.DataFrame({'result': ['recovered']})
                    
                    success, recovered_result, recovery_details = self.recovery_manager.attempt_recovery(
                        e, recovered_operation
                    )
                    
                    result.recovery_time = time.time() - recovery_start_time
                    result.recovery_successful = success
                    result.recovery_details = recovery_details
                    
                    if success:
                        result.test_status = TestStatus.PASSED
                        self.logger.info("数据库连接恢复成功")
                    else:
                        self.logger.warning("数据库连接恢复失败")
                
                except Exception as recovery_error:
                    result.recovery_time = time.time() - recovery_start_time
                    result.recovery_details = f"恢复过程异常: {str(recovery_error)}"
                    self.logger.error(f"恢复过程异常: {recovery_error}")
            
        except Exception as e:
            result.error_details = f"测试执行异常: {str(e)}"
            result.test_status = TestStatus.ERROR
            self.logger.error(f"数据库连接失败测试异常: {e}")
        
        result.execution_time = time.time() - test_start_time
        return result
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=45.0)
    def test_computation_error_scenario(self) -> ExceptionTestResult:
        """测试计算错误场景"""
        scenario = self.exception_scenarios['division_by_zero']
        test_start_time = time.time()
        
        result = ExceptionTestResult(
            test_name='test_computation_error',
            scenario_name=scenario.scenario_name,
            exception_triggered=False,
            exception_caught=False,
            error_message_quality=0.0,
            recovery_attempted=False,
            recovery_successful=False,
            recovery_time=0.0,
            test_status=TestStatus.FAILED,
            execution_time=0.0
        )
        
        def risky_calculation(a: float, b: float) -> float:
            """有风险的计算操作"""
            if b == 0:
                self.exception_simulator.simulate_computation_error("division")
            return a / b
        
        try:
            # 触发除零异常
            try:
                risky_calculation(10.0, 0.0)
            except ZeroDivisionError as e:
                result.exception_triggered = True
                result.exception_caught = True
                result.error_details = str(e)
                result.error_message_quality = self._evaluate_error_message_quality(str(e))
                
                self.logger.info(f"成功触发除零异常: {e}")
                
                # 尝试恢复
                result.recovery_attempted = True
                recovery_start_time = time.time()
                
                def safe_calculation():
                    """安全计算操作"""
                    return 0.0  # 使用默认值
                
                success, recovered_result, recovery_details = self.recovery_manager.attempt_recovery(
                    e, safe_calculation
                )
                
                result.recovery_time = time.time() - recovery_start_time
                result.recovery_successful = success
                result.recovery_details = recovery_details
                
                if success:
                    result.test_status = TestStatus.PASSED
                    self.logger.info("计算错误恢复成功")
                else:
                    self.logger.warning("计算错误恢复失败")
        
        except Exception as e:
            result.error_details = f"测试执行异常: {str(e)}"
            result.test_status = TestStatus.ERROR
            self.logger.error(f"计算错误测试异常: {e}")
        
        result.execution_time = time.time() - test_start_time
        return result
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=60.0)
    def test_memory_exhaustion_scenario(self) -> ExceptionTestResult:
        """测试内存耗尽场景"""
        scenario = self.exception_scenarios['memory_exhaustion']
        test_start_time = time.time()
        
        result = ExceptionTestResult(
            test_name='test_memory_exhaustion',
            scenario_name=scenario.scenario_name,
            exception_triggered=False,
            exception_caught=False,
            error_message_quality=0.0,
            recovery_attempted=False,
            recovery_successful=False,
            recovery_time=0.0,
            test_status=TestStatus.FAILED,
            execution_time=0.0
        )
        
        def memory_intensive_operation():
            """内存密集型操作"""
            self.exception_simulator.simulate_memory_error()
            return "operation_completed"
        
        try:
            # 触发内存异常
            try:
                memory_intensive_operation()
            except MemoryError as e:
                result.exception_triggered = True
                result.exception_caught = True
                result.error_details = str(e)
                result.error_message_quality = self._evaluate_error_message_quality(str(e))
                
                self.logger.info(f"成功触发内存异常: {e}")
                
                # 尝试恢复
                result.recovery_attempted = True
                recovery_start_time = time.time()
                
                def memory_efficient_operation():
                    """内存高效操作"""
                    import gc
                    gc.collect()
                    return "memory_efficient_completed"
                
                success, recovered_result, recovery_details = self.recovery_manager.attempt_recovery(
                    e, memory_efficient_operation
                )
                
                result.recovery_time = time.time() - recovery_start_time
                result.recovery_successful = success
                result.recovery_details = recovery_details
                
                if success:
                    result.test_status = TestStatus.PASSED
                    self.logger.info("内存错误恢复成功")
                else:
                    self.logger.warning("内存错误恢复失败")
        
        except Exception as e:
            result.error_details = f"测试执行异常: {str(e)}"
            result.test_status = TestStatus.ERROR
            self.logger.error(f"内存耗尽测试异常: {e}")
        
        result.execution_time = time.time() - test_start_time
        return result
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=45.0)
    def test_data_format_error_scenario(self) -> ExceptionTestResult:
        """测试数据格式错误场景"""
        scenario = self.exception_scenarios['invalid_data_format']
        test_start_time = time.time()
        
        result = ExceptionTestResult(
            test_name='test_data_format_error',
            scenario_name=scenario.scenario_name,
            exception_triggered=False,
            exception_caught=False,
            error_message_quality=0.0,
            recovery_attempted=False,
            recovery_successful=False,
            recovery_time=0.0,
            test_status=TestStatus.FAILED,
            execution_time=0.0
        )
        
        def process_invalid_data(data):
            """处理无效数据"""
            if not isinstance(data, (int, float)):
                raise ValueError(f"无效的数据类型: {type(data)}, 期望数值类型")
            return data * 2
        
        try:
            # 触发数据格式异常
            try:
                process_invalid_data("invalid_string")
            except ValueError as e:
                result.exception_triggered = True
                result.exception_caught = True
                result.error_details = str(e)
                result.error_message_quality = self._evaluate_error_message_quality(str(e))
                
                self.logger.info(f"成功触发数据格式异常: {e}")
                
                # 尝试恢复
                result.recovery_attempted = True
                recovery_start_time = time.time()
                
                def process_with_conversion():
                    """带类型转换的处理"""
                    try:
                        converted_data = float("0")  # 使用默认值
                        return converted_data * 2
                    except (ValueError, TypeError):
                        return 0.0  # 最终默认值
                
                success, recovered_result, recovery_details = self.recovery_manager.attempt_recovery(
                    e, process_with_conversion
                )
                
                result.recovery_time = time.time() - recovery_start_time
                result.recovery_successful = success
                result.recovery_details = recovery_details
                
                if success:
                    result.test_status = TestStatus.PASSED
                    self.logger.info("数据格式错误恢复成功")
                else:
                    self.logger.warning("数据格式错误恢复失败")
        
        except Exception as e:
            result.error_details = f"测试执行异常: {str(e)}"
            result.test_status = TestStatus.ERROR
            self.logger.error(f"数据格式错误测试异常: {e}")
        
        result.execution_time = time.time() - test_start_time
        return result
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=300.0)
    def run_all_exception_tests(self) -> ExceptionTestSuite:
        """运行所有异常测试"""
        test_methods = [
            self.test_database_connection_failure_scenario,
            self.test_computation_error_scenario,
            self.test_memory_exhaustion_scenario,
            self.test_data_format_error_scenario
        ]
        
        suite_result = ExceptionTestSuite(
            suite_name="exception_scenario_tests",
            total_tests=len(test_methods),
            passed_tests=0,
            failed_tests=0,
            error_tests=0,
            total_execution_time=0.0
        )
        
        start_time = time.time()
        
        # 串行执行异常测试（避免相互干扰）
        for test_method in test_methods:
            try:
                result = test_method()
                suite_result.test_results.append(result)
                
                if result.test_status == TestStatus.PASSED:
                    suite_result.passed_tests += 1
                elif result.test_status == TestStatus.FAILED:
                    suite_result.failed_tests += 1
                else:
                    suite_result.error_tests += 1
                
                self.logger.info(f"异常测试 {result.test_name} 完成: {result.test_status}")
                
            except Exception as e:
                self.logger.error(f"异常测试执行异常: {e}")
                suite_result.error_tests += 1
                
                error_result = ExceptionTestResult(
                    test_name=test_method.__name__,
                    scenario_name="unknown",
                    exception_triggered=False,
                    exception_caught=False,
                    error_message_quality=0.0,
                    recovery_attempted=False,
                    recovery_successful=False,
                    recovery_time=0.0,
                    test_status=TestStatus.ERROR,
                    execution_time=0.0,
                    error_details=str(e)
                )
                suite_result.test_results.append(error_result)
        
        suite_result.total_execution_time = time.time() - start_time
        
        # 计算异常处理评分
        if suite_result.test_results:
            exception_handling_scores = []
            recovery_successes = []
            
            for result in suite_result.test_results:
                # 异常处理评分
                handling_score = 0.0
                if result.exception_triggered:
                    handling_score += 0.3
                if result.exception_caught:
                    handling_score += 0.3
                handling_score += result.error_message_quality * 0.2
                if result.recovery_attempted:
                    handling_score += 0.2
                
                exception_handling_scores.append(handling_score)
                recovery_successes.append(result.recovery_successful)
            
            suite_result.exception_handling_score = sum(exception_handling_scores) / len(exception_handling_scores) * 100
            suite_result.recovery_success_rate = sum(recovery_successes) / len(recovery_successes) * 100
        
        return suite_result
    
    def generate_exception_test_report(self, suite_result: ExceptionTestSuite) -> str:
        """生成异常测试报告"""
        report_lines = [
            "# 异常场景测试报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 测试汇总",
            f"- 总测试数: {suite_result.total_tests}",
            f"- 通过测试: {suite_result.passed_tests}",
            f"- 失败测试: {suite_result.failed_tests}",
            f"- 错误测试: {suite_result.error_tests}",
            f"- 成功率: {(suite_result.passed_tests/suite_result.total_tests*100):.2f}%",
            f"- 总执行时间: {suite_result.total_execution_time:.2f}秒",
            f"- 异常处理评分: {suite_result.exception_handling_score:.2f}/100",
            f"- 恢复成功率: {suite_result.recovery_success_rate:.2f}%",
            ""
        ]
        
        # 详细测试结果
        for result in suite_result.test_results:
            status_icon = {
                TestStatus.PASSED: "✓",
                TestStatus.FAILED: "✗",
                TestStatus.ERROR: "⚠"
            }.get(result.test_status, "?")
            
            report_lines.extend([
                f"## {status_icon} {result.test_name}",
                f"- 异常场景: {result.scenario_name}",
                f"- 测试状态: {result.test_status}",
                f"- 执行时间: {result.execution_time:.2f}秒",
                f"- 异常触发: {'✓' if result.exception_triggered else '✗'}",
                f"- 异常捕获: {'✓' if result.exception_caught else '✗'}",
                f"- 错误信息质量: {result.error_message_quality:.2f}/1.0",
                f"- 恢复尝试: {'✓' if result.recovery_attempted else '✗'}",
                f"- 恢复成功: {'✓' if result.recovery_successful else '✗'}",
                f"- 恢复时间: {result.recovery_time:.2f}秒",
                ""
            ])
            
            if result.error_details:
                report_lines.extend([
                    f"**错误详情**: {result.error_details}",
                    ""
                ])
            
            if result.recovery_details:
                report_lines.extend([
                    f"**恢复详情**: {result.recovery_details}",
                    ""
                ])
        
        # 异常处理能力分析
        report_lines.extend([
            "## 异常处理能力分析",
            "",
            f"系统的异常处理能力评分为 {suite_result.exception_handling_score:.2f}/100。",
            f"异常恢复成功率为 {suite_result.recovery_success_rate:.2f}%。",
            "",
            "### 异常处理维度评估",
        ])
        
        # 计算各维度评分
        exception_triggered_rate = sum(1 for r in suite_result.test_results if r.exception_triggered) / len(suite_result.test_results) * 100
        exception_caught_rate = sum(1 for r in suite_result.test_results if r.exception_caught) / len(suite_result.test_results) * 100
        recovery_attempted_rate = sum(1 for r in suite_result.test_results if r.recovery_attempted) / len(suite_result.test_results) * 100
        avg_error_message_quality = sum(r.error_message_quality for r in suite_result.test_results) / len(suite_result.test_results)
        
        report_lines.extend([
            f"- 异常触发能力: {exception_triggered_rate:.1f}%",
            f"- 异常捕获能力: {exception_caught_rate:.1f}%",
            f"- 错误信息质量: {avg_error_message_quality:.2f}/1.0",
            f"- 恢复尝试率: {recovery_attempted_rate:.1f}%",
            f"- 恢复成功率: {suite_result.recovery_success_rate:.1f}%",
            "",
            "### 改进建议",
        ])
        
        # 根据测试结果提供改进建议
        suggestions = []
        if exception_caught_rate < 90:
            suggestions.append("- 增强异常捕获机制，确保所有异常都能被适当处理")
        if avg_error_message_quality < 0.7:
            suggestions.append("- 改进错误信息质量，提供更详细和有用的错误描述")
        if recovery_attempted_rate < 80:
            suggestions.append("- 实现更多的自动恢复机制")
        if suite_result.recovery_success_rate < 70:
            suggestions.append("- 优化恢复策略，提高恢复成功率")
        
        if suggestions:
            report_lines.extend(suggestions)
        else:
            report_lines.append("- 系统异常处理能力良好，无需特别改进")
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    tester = ExceptionScenarioTester()
    
    # 运行所有异常测试
    suite_result = tester.run_all_exception_tests()
    
    # 生成报告
    report = tester.generate_exception_test_report(suite_result)
    
    # 保存报告
    report_file = f"exception_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"异常场景测试完成，报告已保存到: {report_file}")
    print(f"测试结果: {suite_result.passed_tests}/{suite_result.total_tests} 通过")
    print(f"异常处理评分: {suite_result.exception_handling_score:.2f}/100")
    print(f"恢复成功率: {suite_result.recovery_success_rate:.2f}%")


if __name__ == "__main__":
    main() 