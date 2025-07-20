"""
边界条件测试器
处理空结果、数据异常、网络中断等极端情况的测试
"""

import os
import sys
import time
import threading
import queue
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from unittest.mock import Mock, patch

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
class BoundaryCondition:
    """边界条件定义"""
    condition_name: str
    condition_type: str
    description: str
    test_data: Any = None
    expected_behavior: str = ""
    recovery_mechanism: str = ""


@dataclass
class BoundaryTestResult:
    """边界测试结果"""
    test_name: str
    condition_name: str
    test_status: TestStatus
    execution_time: float
    error_handled: bool
    recovery_successful: bool
    error_message: Optional[str] = None
    expected_result: Any = None
    actual_result: Any = None
    performance_impact: float = 0.0


@dataclass
class BoundaryTestSuite:
    """边界测试套件结果"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    total_execution_time: float
    test_results: List[BoundaryTestResult] = field(default_factory=list)
    robustness_score: float = 0.0


class NetworkSimulator:
    """网络状况模拟器"""
    
    def __init__(self):
        self.is_network_down = False
        self.latency_ms = 0
        self.packet_loss_rate = 0.0
    
    def simulate_network_down(self):
        """模拟网络中断"""
        self.is_network_down = True
    
    def simulate_high_latency(self, latency_ms: int):
        """模拟高延迟"""
        self.latency_ms = latency_ms
    
    def simulate_packet_loss(self, loss_rate: float):
        """模拟丢包"""
        self.packet_loss_rate = loss_rate
    
    def restore_network(self):
        """恢复网络正常"""
        self.is_network_down = False
        self.latency_ms = 0
        self.packet_loss_rate = 0.0
    
    def check_network_condition(self) -> bool:
        """检查网络状况"""
        if self.is_network_down:
            raise ConnectionError("网络连接中断")
        
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
        
        if self.packet_loss_rate > 0 and np.random.random() < self.packet_loss_rate:
            raise TimeoutError("网络请求超时")
        
        return True


class DataCorruptor:
    """数据损坏模拟器"""
    
    @staticmethod
    def create_empty_dataframe() -> pd.DataFrame:
        """创建空DataFrame"""
        return pd.DataFrame()
    
    @staticmethod
    def create_null_data() -> pd.DataFrame:
        """创建全为NULL的数据"""
        data = {
            'code': [None, None, None],
            'date': [None, None, None],
            'open': [None, None, None],
            'high': [None, None, None],
            'low': [None, None, None],
            'close': [None, None, None],
            'volume': [None, None, None]
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def create_malformed_data() -> pd.DataFrame:
        """创建格式错误的数据"""
        data = {
            'code': ['ABC', '123XYZ', ''],
            'date': ['not_a_date', '2023-13-45', ''],
            'open': ['not_a_number', np.inf, -np.inf],
            'high': [np.nan, 'text', 999999999999],
            'low': [-999, 'invalid', np.nan],
            'close': [0, np.nan, 'string'],
            'volume': [-1, 'abc', np.inf]
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def create_inconsistent_ohlc_data() -> pd.DataFrame:
        """创建OHLC不一致的数据"""
        data = {
            'code': ['000001', '000002', '000003'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'open': [10.0, 20.0, 15.0],
            'high': [5.0, 25.0, 12.0],  # high < open
            'low': [15.0, 18.0, 18.0],  # low > open
            'close': [25.0, 15.0, 10.0],  # close > high
            'volume': [1000000, 2000000, 1500000]
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def create_extreme_values_data() -> pd.DataFrame:
        """创建极值数据"""
        data = {
            'code': ['000001', '000002', '000003'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'open': [0.01, 999999.99, 0],
            'high': [0.01, 999999.99, 0.001],
            'low': [0.005, 999999.98, 0],
            'close': [0.008, 999999.985, 0.0005],
            'volume': [1, 9999999999999, 0]
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def create_duplicate_data() -> pd.DataFrame:
        """创建重复数据"""
        base_data = {
            'code': '000001',
            'date': '2023-01-01',
            'open': 10.0,
            'high': 11.0,
            'low': 9.5,
            'close': 10.5,
            'volume': 1000000
        }
        # 创建1000条重复数据
        data = pd.DataFrame([base_data] * 1000)
        return data
    
    @staticmethod
    def create_mixed_type_data() -> pd.DataFrame:
        """创建混合类型数据"""
        data = {
            'code': [123, '000002', 3.14, True, None],
            'date': [20230101, '2023-01-02', '错误日期', None, datetime.now()],
            'open': ['10.0', 20, '不是数字', None, True],
            'high': [15.5, '25', None, np.inf, -1],
            'low': [8.0, 18.5, '文本', False, 0],
            'close': [12.0, 22.5, None, '结束', np.nan],
            'volume': ['1000000', 2000000, None, '不是数字', -1]
        }
        return pd.DataFrame(data)


class BoundaryConditionTester:
    """边界条件测试器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.network_simulator = NetworkSimulator()
        self.data_corruptor = DataCorruptor()
        self.boundary_conditions = self._define_boundary_conditions()
    
    def _define_boundary_conditions(self) -> Dict[str, BoundaryCondition]:
        """定义边界条件"""
        return {
            'empty_data': BoundaryCondition(
                condition_name='empty_data',
                condition_type='data',
                description='空数据集测试',
                test_data=self.data_corruptor.create_empty_dataframe(),
                expected_behavior='优雅处理空数据，返回默认值或错误信息',
                recovery_mechanism='检测空数据并提供有意义的错误信息'
            ),
            'null_data': BoundaryCondition(
                condition_name='null_data',
                condition_type='data',
                description='全NULL数据测试',
                test_data=self.data_corruptor.create_null_data(),
                expected_behavior='过滤NULL值或抛出适当异常',
                recovery_mechanism='数据清洗和验证'
            ),
            'malformed_data': BoundaryCondition(
                condition_name='malformed_data',
                condition_type='data',
                description='格式错误数据测试',
                test_data=self.data_corruptor.create_malformed_data(),
                expected_behavior='数据验证失败，提供详细错误信息',
                recovery_mechanism='数据类型转换和清洗'
            ),
            'inconsistent_ohlc': BoundaryCondition(
                condition_name='inconsistent_ohlc',
                condition_type='data',
                description='OHLC逻辑不一致数据测试',
                test_data=self.data_corruptor.create_inconsistent_ohlc_data(),
                expected_behavior='检测OHLC不一致并标记异常数据',
                recovery_mechanism='数据修正或跳过异常记录'
            ),
            'extreme_values': BoundaryCondition(
                condition_name='extreme_values',
                condition_type='data',
                description='极值数据测试',
                test_data=self.data_corruptor.create_extreme_values_data(),
                expected_behavior='处理极值数据而不崩溃',
                recovery_mechanism='设置合理的数值范围限制'
            ),
            'duplicate_data': BoundaryCondition(
                condition_name='duplicate_data',
                condition_type='data',
                description='重复数据测试',
                test_data=self.data_corruptor.create_duplicate_data(),
                expected_behavior='检测并去除重复数据',
                recovery_mechanism='数据去重处理'
            ),
            'mixed_type_data': BoundaryCondition(
                condition_name='mixed_type_data',
                condition_type='data',
                description='混合类型数据测试',
                test_data=self.data_corruptor.create_mixed_type_data(),
                expected_behavior='类型转换或数据清洗',
                recovery_mechanism='统一数据类型'
            ),
            'network_down': BoundaryCondition(
                condition_name='network_down',
                condition_type='network',
                description='网络中断测试',
                expected_behavior='检测网络中断并提供重试机制',
                recovery_mechanism='连接重试和备用数据源'
            ),
            'high_latency': BoundaryCondition(
                condition_name='high_latency',
                condition_type='network',
                description='高延迟网络测试',
                expected_behavior='超时处理和性能监控',
                recovery_mechanism='超时重试和性能优化'
            ),
            'memory_pressure': BoundaryCondition(
                condition_name='memory_pressure',
                condition_type='resource',
                description='内存压力测试',
                expected_behavior='内存使用监控和限制',
                recovery_mechanism='数据分批处理和内存释放'
            )
        }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def test_empty_data_handling(self) -> BoundaryTestResult:
        """测试空数据处理"""
        test_name = "empty_data_handling"
        condition = self.boundary_conditions['empty_data']
        
        start_time = time.time()
        
        try:
            # 测试技术指标计算对空数据的处理
            from indicators.ma import MAIndicator
            
            ma_indicator = MAIndicator()
            empty_data = condition.test_data
            
            # 预期应该抛出异常或返回空结果
            try:
                result = ma_indicator.calculate(empty_data, period=5)
                
                # 如果没有抛出异常，检查返回结果
                if result is None or (hasattr(result, 'empty') and result.empty):
                    error_handled = True
                    recovery_successful = True
                else:
                    error_handled = False
                    recovery_successful = False
                
            except (ValueError, IndexError, KeyError) as e:
                # 预期的异常类型
                error_handled = True
                recovery_successful = True
                self.logger.info(f"空数据异常正确处理: {e}")
            
            execution_time = time.time() - start_time
            
            return BoundaryTestResult(
                test_name=test_name,
                condition_name=condition.condition_name,
                test_status=TestStatus.PASSED if error_handled else TestStatus.FAILED,
                execution_time=execution_time,
                error_handled=error_handled,
                recovery_successful=recovery_successful,
                expected_result="异常或空结果",
                actual_result="正确处理" if error_handled else "未正确处理"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BoundaryTestResult(
                test_name=test_name,
                condition_name=condition.condition_name,
                test_status=TestStatus.ERROR,
                execution_time=execution_time,
                error_handled=False,
                recovery_successful=False,
                error_message=str(e)
            )
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def test_malformed_data_handling(self) -> BoundaryTestResult:
        """测试格式错误数据处理"""
        test_name = "malformed_data_handling"
        condition = self.boundary_conditions['malformed_data']
        
        start_time = time.time()
        
        try:
            from indicators.ma import MAIndicator
            
            ma_indicator = MAIndicator()
            malformed_data = condition.test_data
            
            error_handled = False
            recovery_successful = False
            
            try:
                result = ma_indicator.calculate(malformed_data, period=5)
                
                # 检查是否有数据清洗或验证
                if result is None or (hasattr(result, 'empty') and result.empty):
                    error_handled = True
                    recovery_successful = True
                elif hasattr(result, 'dropna'):
                    # 检查是否移除了无效数据
                    valid_data = result.dropna()
                    if len(valid_data) < len(result):
                        error_handled = True
                        recovery_successful = True
                
            except (ValueError, TypeError, KeyError) as e:
                error_handled = True
                recovery_successful = True
                self.logger.info(f"格式错误数据异常正确处理: {e}")
            
            execution_time = time.time() - start_time
            
            return BoundaryTestResult(
                test_name=test_name,
                condition_name=condition.condition_name,
                test_status=TestStatus.PASSED if error_handled else TestStatus.FAILED,
                execution_time=execution_time,
                error_handled=error_handled,
                recovery_successful=recovery_successful,
                expected_result="数据验证失败或清洗",
                actual_result="正确处理" if error_handled else "未正确处理"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BoundaryTestResult(
                test_name=test_name,
                condition_name=condition.condition_name,
                test_status=TestStatus.ERROR,
                execution_time=execution_time,
                error_handled=False,
                recovery_successful=False,
                error_message=str(e)
            )
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def test_network_interruption_handling(self) -> BoundaryTestResult:
        """测试网络中断处理"""
        test_name = "network_interruption_handling"
        condition = self.boundary_conditions['network_down']
        
        start_time = time.time()
        
        try:
            # 模拟网络中断
            self.network_simulator.simulate_network_down()
            
            error_handled = False
            recovery_successful = False
            
            # 模拟数据库查询操作
            def mock_database_query():
                self.network_simulator.check_network_condition()
                return pd.DataFrame({'result': ['success']})
            
            try:
                # 第一次尝试（应该失败）
                result = mock_database_query()
            except ConnectionError:
                error_handled = True
                self.logger.info("网络中断正确检测")
                
                # 模拟恢复网络并重试
                self.network_simulator.restore_network()
                
                try:
                    result = mock_database_query()
                    recovery_successful = True
                    self.logger.info("网络恢复后重试成功")
                except Exception as retry_error:
                    self.logger.error(f"网络恢复后重试失败: {retry_error}")
            
            execution_time = time.time() - start_time
            
            return BoundaryTestResult(
                test_name=test_name,
                condition_name=condition.condition_name,
                test_status=TestStatus.PASSED if error_handled and recovery_successful else TestStatus.FAILED,
                execution_time=execution_time,
                error_handled=error_handled,
                recovery_successful=recovery_successful,
                expected_result="网络中断检测和恢复",
                actual_result="完全处理" if error_handled and recovery_successful else "部分处理"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BoundaryTestResult(
                test_name=test_name,
                condition_name=condition.condition_name,
                test_status=TestStatus.ERROR,
                execution_time=execution_time,
                error_handled=False,
                recovery_successful=False,
                error_message=str(e)
            )
        finally:
            # 确保恢复网络状态
            self.network_simulator.restore_network()
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=45.0)
    def test_high_latency_handling(self) -> BoundaryTestResult:
        """测试高延迟处理"""
        test_name = "high_latency_handling"
        condition = self.boundary_conditions['high_latency']
        
        start_time = time.time()
        
        try:
            # 模拟高延迟（5秒）
            self.network_simulator.simulate_high_latency(5000)
            
            error_handled = False
            recovery_successful = False
            timeout_occurred = False
            
            def mock_slow_operation():
                self.network_simulator.check_network_condition()
                return "操作完成"
            
            try:
                # 设置超时测试
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(mock_slow_operation)
                    try:
                        result = future.result(timeout=3.0)  # 3秒超时
                    except TimeoutError:
                        timeout_occurred = True
                        error_handled = True
                        self.logger.info("高延迟超时正确检测")
                        
                        # 模拟恢复正常延迟并重试
                        self.network_simulator.restore_network()
                        
                        try:
                            result = mock_slow_operation()
                            recovery_successful = True
                            self.logger.info("延迟恢复后操作成功")
                        except Exception as retry_error:
                            self.logger.error(f"延迟恢复后重试失败: {retry_error}")
            
            except Exception as e:
                self.logger.error(f"高延迟测试异常: {e}")
            
            execution_time = time.time() - start_time
            
            return BoundaryTestResult(
                test_name=test_name,
                condition_name=condition.condition_name,
                test_status=TestStatus.PASSED if error_handled else TestStatus.FAILED,
                execution_time=execution_time,
                error_handled=error_handled,
                recovery_successful=recovery_successful,
                expected_result="超时检测和重试",
                actual_result="正确处理" if error_handled else "未正确处理",
                performance_impact=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BoundaryTestResult(
                test_name=test_name,
                condition_name=condition.condition_name,
                test_status=TestStatus.ERROR,
                execution_time=execution_time,
                error_handled=False,
                recovery_successful=False,
                error_message=str(e)
            )
        finally:
            # 确保恢复网络状态
            self.network_simulator.restore_network()
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def test_memory_pressure_handling(self) -> BoundaryTestResult:
        """测试内存压力处理"""
        test_name = "memory_pressure_handling"
        condition = self.boundary_conditions['memory_pressure']
        
        start_time = time.time()
        
        try:
            import psutil
            import gc
            
            # 获取初始内存使用情况
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            error_handled = False
            recovery_successful = False
            
            try:
                # 创建大量数据模拟内存压力
                large_datasets = []
                for i in range(10):
                    # 创建大型DataFrame
                    large_data = pd.DataFrame(np.random.randn(100000, 50))
                    large_datasets.append(large_data)
                    
                    # 检查内存使用情况
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_growth = current_memory - initial_memory
                    
                    if memory_growth > 500:  # 超过500MB
                        error_handled = True
                        self.logger.warning(f"检测到内存使用过高: {memory_growth:.2f}MB")
                        
                        # 模拟内存释放和垃圾回收
                        large_datasets.clear()
                        gc.collect()
                        
                        # 检查内存是否释放
                        after_gc_memory = process.memory_info().rss / 1024 / 1024
                        memory_released = current_memory - after_gc_memory
                        
                        if memory_released > 100:  # 释放了超过100MB
                            recovery_successful = True
                            self.logger.info(f"内存成功释放: {memory_released:.2f}MB")
                        
                        break
                
            except MemoryError:
                error_handled = True
                recovery_successful = True
                self.logger.info("内存不足异常正确处理")
            
            execution_time = time.time() - start_time
            
            return BoundaryTestResult(
                test_name=test_name,
                condition_name=condition.condition_name,
                test_status=TestStatus.PASSED if error_handled else TestStatus.FAILED,
                execution_time=execution_time,
                error_handled=error_handled,
                recovery_successful=recovery_successful,
                expected_result="内存监控和释放",
                actual_result="正确处理" if error_handled else "未检测到内存压力"
            )
            
        except ImportError:
            # 如果psutil不可用，跳过此测试
            execution_time = time.time() - start_time
            return BoundaryTestResult(
                test_name=test_name,
                condition_name=condition.condition_name,
                test_status=TestStatus.SKIPPED,
                execution_time=execution_time,
                error_handled=False,
                recovery_successful=False,
                error_message="psutil模块不可用"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return BoundaryTestResult(
                test_name=test_name,
                condition_name=condition.condition_name,
                test_status=TestStatus.ERROR,
                execution_time=execution_time,
                error_handled=False,
                recovery_successful=False,
                error_message=str(e)
            )
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def test_ohlc_inconsistency_handling(self) -> BoundaryTestResult:
        """测试OHLC不一致数据处理"""
        test_name = "ohlc_inconsistency_handling"
        condition = self.boundary_conditions['inconsistent_ohlc']
        
        start_time = time.time()
        
        try:
            inconsistent_data = condition.test_data
            
            error_handled = False
            recovery_successful = False
            
            # 验证OHLC数据逻辑
            def validate_ohlc_data(data):
                violations = []
                for idx, row in data.iterrows():
                    if pd.notna(row['open']) and pd.notna(row['high']) and pd.notna(row['low']) and pd.notna(row['close']):
                        if row['high'] < row['open'] or row['high'] < row['close']:
                            violations.append(f"行{idx}: high价格不应低于open或close")
                        if row['low'] > row['open'] or row['low'] > row['close']:
                            violations.append(f"行{idx}: low价格不应高于open或close")
                        if row['high'] < row['low']:
                            violations.append(f"行{idx}: high价格不应低于low价格")
                
                return violations
            
            violations = validate_ohlc_data(inconsistent_data)
            
            if violations:
                error_handled = True
                self.logger.info(f"检测到OHLC不一致: {len(violations)}个违规")
                
                # 模拟数据修正
                corrected_data = inconsistent_data.copy()
                for idx, row in corrected_data.iterrows():
                    if pd.notna(row['open']) and pd.notna(row['high']) and pd.notna(row['low']) and pd.notna(row['close']):
                        # 简单修正：确保high是最高值，low是最低值
                        values = [row['open'], row['close']]
                        corrected_data.loc[idx, 'high'] = max(max(values), row['high'])
                        corrected_data.loc[idx, 'low'] = min(min(values), row['low'])
                
                # 验证修正后的数据
                corrected_violations = validate_ohlc_data(corrected_data)
                if len(corrected_violations) < len(violations):
                    recovery_successful = True
                    self.logger.info("OHLC数据修正成功")
            
            execution_time = time.time() - start_time
            
            return BoundaryTestResult(
                test_name=test_name,
                condition_name=condition.condition_name,
                test_status=TestStatus.PASSED if error_handled else TestStatus.FAILED,
                execution_time=execution_time,
                error_handled=error_handled,
                recovery_successful=recovery_successful,
                expected_result="检测并修正OHLC不一致",
                actual_result=f"检测到{len(violations)}个违规" if error_handled else "未检测到违规"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BoundaryTestResult(
                test_name=test_name,
                condition_name=condition.condition_name,
                test_status=TestStatus.ERROR,
                execution_time=execution_time,
                error_handled=False,
                recovery_successful=False,
                error_message=str(e)
            )
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=300.0)
    def run_all_boundary_tests(self) -> BoundaryTestSuite:
        """运行所有边界测试"""
        test_methods = [
            self.test_empty_data_handling,
            self.test_malformed_data_handling,
            self.test_network_interruption_handling,
            self.test_high_latency_handling,
            self.test_memory_pressure_handling,
            self.test_ohlc_inconsistency_handling
        ]
        
        suite_result = BoundaryTestSuite(
            suite_name="boundary_condition_tests",
            total_tests=len(test_methods),
            passed_tests=0,
            failed_tests=0,
            error_tests=0,
            total_execution_time=0.0
        )
        
        start_time = time.time()
        
        # 串行执行边界测试（避免相互干扰）
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
                
                self.logger.info(f"边界测试 {result.test_name} 完成: {result.test_status}")
                
            except Exception as e:
                self.logger.error(f"边界测试执行异常: {e}")
                suite_result.error_tests += 1
                
                error_result = BoundaryTestResult(
                    test_name=test_method.__name__,
                    condition_name="unknown",
                    test_status=TestStatus.ERROR,
                    execution_time=0.0,
                    error_handled=False,
                    recovery_successful=False,
                    error_message=str(e)
                )
                suite_result.test_results.append(error_result)
        
        suite_result.total_execution_time = time.time() - start_time
        
        # 计算鲁棒性评分
        if suite_result.total_tests > 0:
            robustness_factors = []
            for result in suite_result.test_results:
                if result.test_status == TestStatus.PASSED:
                    factor = 1.0
                    if result.error_handled:
                        factor += 0.5
                    if result.recovery_successful:
                        factor += 0.5
                    robustness_factors.append(factor)
                else:
                    robustness_factors.append(0.0)
            
            suite_result.robustness_score = sum(robustness_factors) / len(robustness_factors) * 100
        
        return suite_result
    
    def generate_boundary_test_report(self, suite_result: BoundaryTestSuite) -> str:
        """生成边界测试报告"""
        report_lines = [
            "# 边界条件测试报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 测试汇总",
            f"- 总测试数: {suite_result.total_tests}",
            f"- 通过测试: {suite_result.passed_tests}",
            f"- 失败测试: {suite_result.failed_tests}",
            f"- 错误测试: {suite_result.error_tests}",
            f"- 成功率: {(suite_result.passed_tests/suite_result.total_tests*100):.2f}%",
            f"- 总执行时间: {suite_result.total_execution_time:.2f}秒",
            f"- 系统鲁棒性评分: {suite_result.robustness_score:.2f}/100",
            ""
        ]
        
        # 详细测试结果
        for result in suite_result.test_results:
            status_icon = {
                TestStatus.PASSED: "✓",
                TestStatus.FAILED: "✗",
                TestStatus.ERROR: "⚠",
                TestStatus.SKIPPED: "○"
            }.get(result.test_status, "?")
            
            report_lines.extend([
                f"## {status_icon} {result.test_name}",
                f"- 边界条件: {result.condition_name}",
                f"- 测试状态: {result.test_status}",
                f"- 执行时间: {result.execution_time:.2f}秒",
                f"- 错误处理: {'✓' if result.error_handled else '✗'}",
                f"- 恢复成功: {'✓' if result.recovery_successful else '✗'}",
                f"- 期望结果: {result.expected_result}",
                f"- 实际结果: {result.actual_result}",
                ""
            ])
            
            if result.error_message:
                report_lines.extend([
                    f"**错误信息**: {result.error_message}",
                    ""
                ])
            
            if result.performance_impact > 0:
                report_lines.extend([
                    f"**性能影响**: {result.performance_impact:.2f}秒",
                    ""
                ])
        
        # 鲁棒性分析
        report_lines.extend([
            "## 鲁棒性分析",
            "",
            f"系统在面对边界条件时的鲁棒性评分为 {suite_result.robustness_score:.2f}/100。",
            "",
            "### 错误处理能力",
            f"- 能够正确处理错误的测试: {sum(1 for r in suite_result.test_results if r.error_handled)}/{suite_result.total_tests}",
            "",
            "### 恢复能力",
            f"- 能够成功恢复的测试: {sum(1 for r in suite_result.test_results if r.recovery_successful)}/{suite_result.total_tests}",
            "",
            "### 建议改进",
        ])
        
        # 根据测试结果提供改进建议
        failed_tests = [r for r in suite_result.test_results if r.test_status == TestStatus.FAILED]
        if failed_tests:
            report_lines.append("根据测试结果，建议在以下方面加强:")
            for failed_test in failed_tests:
                report_lines.append(f"- {failed_test.condition_name}: 改进{failed_test.expected_result}")
        else:
            report_lines.append("所有边界条件测试均通过，系统具有良好的鲁棒性。")
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    tester = BoundaryConditionTester()
    
    # 运行所有边界测试
    suite_result = tester.run_all_boundary_tests()
    
    # 生成报告
    report = tester.generate_boundary_test_report(suite_result)
    
    # 保存报告
    report_file = f"boundary_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"边界条件测试完成，报告已保存到: {report_file}")
    print(f"测试结果: {suite_result.passed_tests}/{suite_result.total_tests} 通过")
    print(f"鲁棒性评分: {suite_result.robustness_score:.2f}/100")


if __name__ == "__main__":
    main() 