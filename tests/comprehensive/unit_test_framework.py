"""
单元测试框架
为股票选股策略系统的核心组件提供独立测试用例
"""

import os
import sys
import time
import traceback
import unittest
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, Callable
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
class UnitTestResult:
    """单元测试结果"""
    test_name: str
    test_class: str
    test_method: str
    status: TestStatus
    execution_time: float
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    assertions_count: int = 0
    passed_assertions: int = 0
    test_data_size: int = 0
    memory_usage: float = 0.0
    setup_time: float = 0.0
    teardown_time: float = 0.0


@dataclass
class UnitTestSuiteResult:
    """单元测试套件结果"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_execution_time: float
    setup_time: float
    teardown_time: float
    test_results: List[UnitTestResult] = field(default_factory=list)
    coverage_percentage: float = 0.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class UnitTestBase(unittest.TestCase, ABC):
    """单元测试基类"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        self.test_start_time = None
        self.test_data = {}
        self.mock_objects = {}
        self.assertions_count = 0
        self.passed_assertions = 0
    
    def setUp(self):
        """测试设置"""
        self.test_start_time = time.time()
        self.setup_test_environment()
        self.create_test_data()
        self.setup_mocks()
    
    def tearDown(self):
        """测试清理"""
        self.cleanup_test_data()
        self.cleanup_mocks()
        self.cleanup_test_environment()
    
    @abstractmethod
    def setup_test_environment(self):
        """设置测试环境"""
        pass
    
    @abstractmethod
    def create_test_data(self):
        """创建测试数据"""
        pass
    
    @abstractmethod
    def cleanup_test_data(self):
        """清理测试数据"""
        pass
    
    def setup_mocks(self):
        """设置模拟对象"""
        pass
    
    def cleanup_mocks(self):
        """清理模拟对象"""
        for mock_obj in self.mock_objects.values():
            if hasattr(mock_obj, 'stop'):
                mock_obj.stop()
        self.mock_objects.clear()
    
    def cleanup_test_environment(self):
        """清理测试环境"""
        pass
    
    def assert_with_tracking(self, condition: bool, message: str = ""):
        """带跟踪的断言"""
        self.assertions_count += 1
        try:
            self.assertTrue(condition, message)
            self.passed_assertions += 1
        except AssertionError:
            self.logger.error(f"断言失败: {message}")
            raise
    
    def assert_dataframe_equal_with_tracking(self, df1: pd.DataFrame, df2: pd.DataFrame, message: str = ""):
        """带跟踪的DataFrame相等断言"""
        self.assertions_count += 1
        try:
            pd.testing.assert_frame_equal(df1, df2)
            self.passed_assertions += 1
        except AssertionError as e:
            self.logger.error(f"DataFrame断言失败: {message}, 错误: {str(e)}")
            raise
    
    def assert_approximately_equal(self, actual: float, expected: float, tolerance: float = 0.01, message: str = ""):
        """近似相等断言"""
        self.assertions_count += 1
        try:
            difference = abs(actual - expected)
            relative_error = difference / abs(expected) if expected != 0 else difference
            self.assertLessEqual(relative_error, tolerance, 
                               f"{message}: 实际值={actual}, 期望值={expected}, 相对误差={relative_error:.4f}")
            self.passed_assertions += 1
        except AssertionError:
            self.logger.error(f"近似相等断言失败: {message}")
            raise


class IndicatorUnitTests(UnitTestBase):
    """技术指标单元测试"""
    
    def setup_test_environment(self):
        """设置测试环境"""
        self.test_stock_code = "000001"
        self.test_period = 20
        
    def create_test_data(self):
        """创建测试数据"""
        # 创建标准测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)  # 确保可重复性
        
        # 生成OHLCV数据
        base_price = 10.0
        prices = []
        for i in range(len(dates)):
            if i == 0:
                open_price = base_price
            else:
                open_price = prices[-1]['close'] * (1 + np.random.normal(0, 0.02))
            
            high_price = open_price * (1 + abs(np.random.normal(0, 0.03)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.03)))
            close_price = low_price + (high_price - low_price) * np.random.random()
            volume = np.random.randint(1000000, 10000000)
            
            prices.append({
                'date': dates[i].strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        self.test_data['standard_ohlcv'] = pd.DataFrame(prices)
        
        # 创建边界条件测试数据
        self.test_data['minimal_data'] = self.test_data['standard_ohlcv'].head(5)
        self.test_data['single_row'] = self.test_data['standard_ohlcv'].head(1)
        
        # 创建异常数据
        abnormal_data = self.test_data['standard_ohlcv'].copy()
        abnormal_data.loc[10, 'close'] = np.nan
        abnormal_data.loc[20, 'volume'] = 0
        self.test_data['abnormal_data'] = abnormal_data
    
    def cleanup_test_data(self):
        """清理测试数据"""
        self.test_data.clear()
    
    def test_ma_calculation(self):
        """测试移动平均线计算"""
        from indicators.ma import MAIndicator
        
        ma_indicator = MAIndicator()
        data = self.test_data['standard_ohlcv']
        
        # 测试MA5计算
        ma5_result = ma_indicator.calculate(data, period=5)
        self.assert_with_tracking(isinstance(ma5_result, pd.Series), "MA5结果应为Series类型")
        self.assert_with_tracking(len(ma5_result) == len(data), "MA5结果长度应与输入数据相同")
        
        # 验证前4个值为NaN（因为周期为5）
        self.assert_with_tracking(ma5_result.iloc[:4].isna().all(), "MA5前4个值应为NaN")
        
        # 验证第5个值的计算正确性
        expected_ma5 = data['close'].iloc[:5].mean()
        self.assert_approximately_equal(ma5_result.iloc[4], expected_ma5, 0.001, "MA5第5个值计算错误")
        
        # 测试边界条件
        minimal_result = ma_indicator.calculate(self.test_data['minimal_data'], period=5)
        self.assert_with_tracking(minimal_result.iloc[4] == self.test_data['minimal_data']['close'].mean(), 
                                "边界条件下MA计算错误")
    
    def test_macd_calculation(self):
        """测试MACD计算"""
        from indicators.macd import MACDIndicator
        
        macd_indicator = MACDIndicator()
        data = self.test_data['standard_ohlcv']
        
        result = macd_indicator.calculate(data, fast_period=12, slow_period=26, signal_period=9)
        
        # 验证返回结果包含必要字段
        expected_columns = ['macd', 'signal', 'histogram']
        for col in expected_columns:
            self.assert_with_tracking(col in result.columns, f"MACD结果应包含{col}列")
        
        # 验证数据类型
        self.assert_with_tracking(all(result.dtypes == 'float64'), "MACD结果应为float64类型")
        
        # 验证计算逻辑
        self.assert_with_tracking((result['histogram'] == result['macd'] - result['signal']).all(), 
                                "MACD histogram计算错误")
    
    def test_rsi_calculation(self):
        """测试RSI计算"""
        from indicators.rsi import RSIIndicator
        
        rsi_indicator = RSIIndicator()
        data = self.test_data['standard_ohlcv']
        
        rsi_result = rsi_indicator.calculate(data, period=14)
        
        # 验证RSI值范围
        valid_rsi = rsi_result.dropna()
        self.assert_with_tracking((valid_rsi >= 0).all() and (valid_rsi <= 100).all(), 
                                "RSI值应在0-100范围内")
        
        # 验证数据类型和长度
        self.assert_with_tracking(isinstance(rsi_result, pd.Series), "RSI结果应为Series类型")
        self.assert_with_tracking(len(rsi_result) == len(data), "RSI结果长度应与输入数据相同")
    
    def test_indicator_with_insufficient_data(self):
        """测试数据不足情况"""
        from indicators.ma import MAIndicator
        
        ma_indicator = MAIndicator()
        
        # 测试空数据
        empty_data = pd.DataFrame(columns=['close'])
        with self.assertRaises(ValueError):
            ma_indicator.calculate(empty_data, period=5)
        
        # 测试数据不足
        insufficient_data = self.test_data['single_row']
        result = ma_indicator.calculate(insufficient_data, period=5)
        self.assert_with_tracking(result.isna().all(), "数据不足时应返回NaN")
    
    def test_indicator_with_abnormal_data(self):
        """测试异常数据处理"""
        from indicators.ma import MAIndicator
        
        ma_indicator = MAIndicator()
        abnormal_data = self.test_data['abnormal_data']
        
        # 测试包含NaN的数据
        result = ma_indicator.calculate(abnormal_data, period=5)
        
        # 验证在NaN值附近的计算
        self.assert_with_tracking(isinstance(result, pd.Series), "异常数据处理后结果应为Series类型")


class StrategyUnitTests(UnitTestBase):
    """选股策略单元测试"""
    
    def setup_test_environment(self):
        """设置测试环境"""
        self.test_stock_codes = ["000001", "000002", "600000", "600036"]
    
    def create_test_data(self):
        """创建测试数据"""
        # 为每个股票创建测试数据
        for code in self.test_stock_codes:
            dates = pd.date_range('2023-01-01', periods=50, freq='D')
            np.random.seed(hash(code) % 1000)  # 基于股票代码生成不同的随机种子
            
            base_price = 10.0 + hash(code) % 20
            data = []
            
            for i, date in enumerate(dates):
                if i == 0:
                    close = base_price
                else:
                    # 模拟不同的趋势
                    if code == "000001":  # 上涨趋势
                        trend = 0.001
                    elif code == "000002":  # 下跌趋势
                        trend = -0.001
                    else:  # 震荡趋势
                        trend = 0
                    
                    close = data[-1]['close'] * (1 + trend + np.random.normal(0, 0.02))
                
                volume = np.random.randint(1000000, 10000000)
                
                data.append({
                    'code': code,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': close * (1 + np.random.normal(0, 0.01)),
                    'high': close * (1 + abs(np.random.normal(0, 0.02))),
                    'low': close * (1 - abs(np.random.normal(0, 0.02))),
                    'close': close,
                    'volume': volume
                })
            
            self.test_data[f'stock_{code}'] = pd.DataFrame(data)
        
        # 创建组合数据
        all_data = []
        for code in self.test_stock_codes:
            all_data.append(self.test_data[f'stock_{code}'])
        self.test_data['combined_data'] = pd.concat(all_data, ignore_index=True)
    
    def cleanup_test_data(self):
        """清理测试数据"""
        self.test_data.clear()
    
    def setup_mocks(self):
        """设置模拟对象"""
        # 模拟数据库查询
        self.mock_objects['query_executor'] = Mock()
        self.mock_objects['query_executor'].execute_query.return_value = self.test_data['combined_data']
    
    def test_dual_ma_strategy(self):
        """测试双均线策略"""
        from strategy.dual_ma.dual_ma_strategy import DualMAStrategy
        
        strategy = DualMAStrategy()
        
        # 测试单股票
        stock_data = self.test_data['stock_000001']
        result = strategy.analyze_single_stock(stock_data)
        
        # 验证结果结构
        self.assert_with_tracking('signal' in result, "双均线策略结果应包含信号")
        self.assert_with_tracking('score' in result, "双均线策略结果应包含评分")
        self.assert_with_tracking('reason' in result, "双均线策略结果应包含原因")
        
        # 验证信号类型
        self.assert_with_tracking(result['signal'] in ['BUY', 'SELL', 'HOLD'], 
                                "双均线策略信号应为BUY/SELL/HOLD")
        
        # 验证评分范围
        self.assert_with_tracking(0 <= result['score'] <= 100, "双均线策略评分应在0-100范围内")
    
    def test_strategy_with_insufficient_data(self):
        """测试数据不足的策略处理"""
        from strategy.dual_ma.dual_ma_strategy import DualMAStrategy
        
        strategy = DualMAStrategy()
        
        # 测试空数据
        empty_data = pd.DataFrame(columns=['code', 'date', 'open', 'high', 'low', 'close', 'volume'])
        with self.assertRaises(ValueError):
            strategy.analyze_single_stock(empty_data)
        
        # 测试数据不足
        minimal_data = self.test_data['stock_000001'].head(5)
        result = strategy.analyze_single_stock(minimal_data)
        self.assert_with_tracking(result['signal'] == 'HOLD', "数据不足时应返回HOLD信号")
    
    def test_strategy_consistency(self):
        """测试策略一致性"""
        from strategy.dual_ma.dual_ma_strategy import DualMAStrategy
        
        strategy = DualMAStrategy()
        stock_data = self.test_data['stock_000001']
        
        # 多次执行相同策略
        results = []
        for _ in range(3):
            result = strategy.analyze_single_stock(stock_data.copy())
            results.append(result)
        
        # 验证结果一致性
        first_result = results[0]
        for result in results[1:]:
            self.assert_with_tracking(result['signal'] == first_result['signal'], 
                                    "相同数据多次执行策略应返回一致结果")
            self.assert_approximately_equal(result['score'], first_result['score'], 0.001, 
                                          "相同数据多次执行策略评分应一致")


class DataManagerUnitTests(UnitTestBase):
    """数据管理器单元测试"""
    
    def setup_test_environment(self):
        """设置测试环境"""
        self.test_codes = ["000001", "000002"]
        self.test_date_range = ("2023-01-01", "2023-12-31")
    
    def create_test_data(self):
        """创建测试数据"""
        # 创建模拟数据库数据
        data = []
        for code in self.test_codes:
            dates = pd.date_range(self.test_date_range[0], self.test_date_range[1], freq='D')
            for date in dates[:100]:  # 限制数据量
                data.append({
                    'code': code,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': 10.0 + np.random.random(),
                    'high': 11.0 + np.random.random(),
                    'low': 9.0 + np.random.random(),
                    'close': 10.0 + np.random.random(),
                    'volume': np.random.randint(1000000, 10000000),
                    'turnover_rate': np.random.random() * 10
                })
        
        self.test_data['database_data'] = pd.DataFrame(data)
    
    def cleanup_test_data(self):
        """清理测试数据"""
        self.test_data.clear()
    
    def setup_mocks(self):
        """设置模拟对象"""
        # 模拟数据库连接
        self.mock_objects['db_connection'] = Mock()
        
        # 模拟查询执行器
        self.mock_objects['query_executor'] = Mock()
        self.mock_objects['query_executor'].execute_query.return_value = self.test_data['database_data']
    
    def test_data_retrieval(self):
        """测试数据检索"""
        from db.managers.unified_query_executor import UnifiedQueryExecutor
        
        # 使用模拟对象
        with patch('db.managers.unified_query_executor.get_clickhouse_db') as mock_db:
            mock_db.return_value = self.mock_objects['db_connection']
            
            executor = UnifiedQueryExecutor()
            
            # 测试数据检索
            query = "SELECT * FROM stock_info WHERE code IN ('000001', '000002')"
            result = executor.execute_query(query)
            
            # 验证结果
            self.assert_with_tracking(isinstance(result, pd.DataFrame), "查询结果应为DataFrame类型")
            self.assert_with_tracking(len(result) > 0, "查询结果不应为空")
            
            # 验证数据完整性
            expected_columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume']
            for col in expected_columns:
                self.assert_with_tracking(col in result.columns, f"查询结果应包含{col}列")
    
    def test_data_validation(self):
        """测试数据验证"""
        data = self.test_data['database_data']
        
        # 验证数据类型
        self.assert_with_tracking(data['code'].dtype == 'object', "股票代码应为字符串类型")
        self.assert_with_tracking(data['date'].dtype == 'object', "日期应为字符串类型")
        
        # 验证数值列
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            self.assert_with_tracking(pd.api.types.is_numeric_dtype(data[col]), f"{col}应为数值类型")
        
        # 验证OHLC逻辑
        valid_ohlc = (data['low'] <= data['open']) & (data['open'] <= data['high']) & \
                     (data['low'] <= data['close']) & (data['close'] <= data['high'])
        self.assert_with_tracking(valid_ohlc.all(), "OHLC数据应满足逻辑关系")
    
    def test_error_handling(self):
        """测试错误处理"""
        from db.managers.unified_query_executor import UnifiedQueryExecutor
        
        # 模拟数据库连接错误
        with patch('db.managers.unified_query_executor.get_clickhouse_db') as mock_db:
            mock_db.side_effect = Exception("数据库连接失败")
            
            executor = UnifiedQueryExecutor()
            
            # 验证异常处理
            with self.assertRaises(Exception):
                executor.execute_query("SELECT * FROM stock_info")


class UnitTestFramework:
    """单元测试框架主类"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.test_suites = {}
        self.register_test_suites()
    
    def register_test_suites(self):
        """注册测试套件"""
        self.test_suites = {
            'indicator_tests': IndicatorUnitTests,
            'strategy_tests': StrategyUnitTests,
            'data_manager_tests': DataManagerUnitTests
        }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=300.0)
    def run_unit_test_suite(self, suite_name: str, test_methods: Optional[List[str]] = None) -> UnitTestSuiteResult:
        """运行单元测试套件"""
        if suite_name not in self.test_suites:
            raise ValueError(f"未找到测试套件: {suite_name}")
        
        suite_start_time = time.time()
        test_class = self.test_suites[suite_name]
        
        # 创建测试套件
        loader = unittest.TestLoader()
        if test_methods:
            # 运行指定测试方法
            suite = unittest.TestSuite()
            for method in test_methods:
                suite.addTest(test_class(method))
        else:
            # 运行所有测试方法
            suite = loader.loadTestsFromTestCase(test_class)
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        test_result = runner.run(suite)
        
        # 收集测试结果
        test_results = []
        total_tests = test_result.testsRun
        passed_tests = total_tests - len(test_result.failures) - len(test_result.errors)
        failed_tests = len(test_result.failures)
        error_tests = len(test_result.errors)
        
        # 处理失败的测试
        for test, error in test_result.failures + test_result.errors:
            test_name = test._testMethodName
            test_results.append(UnitTestResult(
                test_name=test_name,
                test_class=test_class.__name__,
                test_method=test_name,
                status=TestStatus.FAILED,
                execution_time=0.0,
                error_message=str(error),
                stack_trace=traceback.format_exc()
            ))
        
        # 处理成功的测试
        for i in range(passed_tests):
            test_results.append(UnitTestResult(
                test_name=f"test_{i+1}",
                test_class=test_class.__name__,
                test_method=f"test_{i+1}",
                status=TestStatus.PASSED,
                execution_time=0.0
            ))
        
        total_execution_time = time.time() - suite_start_time
        
        return UnitTestSuiteResult(
            suite_name=suite_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=0,
            error_tests=error_tests,
            total_execution_time=total_execution_time,
            setup_time=0.0,
            teardown_time=0.0,
            test_results=test_results
        )
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=600.0)
    def run_all_unit_tests(self) -> Dict[str, UnitTestSuiteResult]:
        """运行所有单元测试"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_suite = {
                executor.submit(self.run_unit_test_suite, suite_name): suite_name
                for suite_name in self.test_suites.keys()
            }
            
            for future in as_completed(future_to_suite):
                suite_name = future_to_suite[future]
                try:
                    result = future.result()
                    results[suite_name] = result
                    self.logger.info(f"单元测试套件 {suite_name} 完成: {result.passed_tests}/{result.total_tests} 通过")
                except Exception as e:
                    self.logger.error(f"单元测试套件 {suite_name} 执行失败: {e}")
                    results[suite_name] = UnitTestSuiteResult(
                        suite_name=suite_name,
                        total_tests=0,
                        passed_tests=0,
                        failed_tests=0,
                        skipped_tests=0,
                        error_tests=1,
                        total_execution_time=0.0,
                        setup_time=0.0,
                        teardown_time=0.0,
                        test_results=[UnitTestResult(
                            test_name="framework_error",
                            test_class="UnitTestFramework",
                            test_method="run_unit_test_suite",
                            status=TestStatus.ERROR,
                            execution_time=0.0,
                            error_message=str(e)
                        )]
                    )
        
        return results
    
    def generate_unit_test_report(self, results: Dict[str, UnitTestSuiteResult]) -> str:
        """生成单元测试报告"""
        report_lines = [
            "# 单元测试执行报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 测试套件汇总",
            ""
        ]
        
        total_tests = sum(result.total_tests for result in results.values())
        total_passed = sum(result.passed_tests for result in results.values())
        total_failed = sum(result.failed_tests for result in results.values())
        total_errors = sum(result.error_tests for result in results.values())
        
        report_lines.extend([
            f"- 总测试数: {total_tests}",
            f"- 通过测试: {total_passed}",
            f"- 失败测试: {total_failed}",
            f"- 错误测试: {total_errors}",
            f"- 通过率: {(total_passed/total_tests*100):.2f}%" if total_tests > 0 else "- 通过率: 0%",
            ""
        ])
        
        # 详细结果
        for suite_name, result in results.items():
            report_lines.extend([
                f"## {suite_name}",
                f"- 测试数: {result.total_tests}",
                f"- 通过: {result.passed_tests}",
                f"- 失败: {result.failed_tests}",
                f"- 错误: {result.error_tests}",
                f"- 执行时间: {result.total_execution_time:.2f}秒",
                ""
            ])
            
            # 失败的测试详情
            failed_tests = [t for t in result.test_results if t.status in [TestStatus.FAILED, TestStatus.ERROR]]
            if failed_tests:
                report_lines.append("### 失败测试详情")
                for test in failed_tests:
                    report_lines.extend([
                        f"- **{test.test_name}**: {test.error_message}",
                        ""
                    ])
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    framework = UnitTestFramework()
    
    # 运行所有单元测试
    results = framework.run_all_unit_tests()
    
    # 生成报告
    report = framework.generate_unit_test_report(results)
    
    # 保存报告
    report_file = f"unit_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"单元测试完成，报告已保存到: {report_file}")


if __name__ == "__main__":
    main() 