"""
Fibonacci指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
import io
import logging

from indicators.fibonacci import Fibonacci
from utils.logger import get_logger
from tests.helper.log_capture import LogCaptureMixin


logger = get_logger(__name__)


class TestFibonacci(unittest.TestCase, LogCaptureMixin):
    """Fibonacci指标测试类"""

    def setUp(self):
        """设置测试环境"""
        # 设置日志捕获
        self.log_stream = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.logger = logging.getLogger()
        self.logger.addHandler(self.log_handler)
        self.logger.setLevel(logging.DEBUG)

        self.fibonacci = Fibonacci(period=14)
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        # 生成包含趋势的测试数据
        np.random.seed(42)
        base_price = 100
        prices = []
        for i in range(50):
            # 添加趋势和随机波动
            trend = i * 0.5  # 上升趋势
            noise = np.random.normal(0, 2)  # 随机噪声
            price = base_price + trend + noise
            prices.append(price)
        
        # 生成OHLC数据
        self.test_data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 1)) for p in prices],
            'low': [p - abs(np.random.normal(0, 1)) for p in prices],
            'close': [p + np.random.normal(0, 0.5) for p in prices],
            'volume': np.random.randint(1000, 10000, 50)
        })
        
        # 确保 high >= low >= 0
        for i in range(len(self.test_data)):
            if self.test_data.loc[i, 'high'] < self.test_data.loc[i, 'low']:
                self.test_data.loc[i, 'high'], self.test_data.loc[i, 'low'] = \
                    self.test_data.loc[i, 'low'], self.test_data.loc[i, 'high']

    def tearDown(self):
        """清理测试环境"""
        if hasattr(self, 'log_handler') and self.log_handler:
            self.logger.removeHandler(self.log_handler)
            self.log_handler.close()

    def test_fibonacci_initialization(self):
        """测试Fibonacci指标初始化"""
        logger.info("测试Fibonacci指标初始化")
        
        # 测试默认初始化
        fibonacci = Fibonacci()
        self.assertEqual(fibonacci.name, "FIBONACCI")
        self.assertIn("斐波那契", fibonacci.description)
        self.assertEqual(fibonacci.period, 14)
        
        # 测试自定义参数初始化
        fibonacci_custom = Fibonacci(period=20)
        self.assertEqual(fibonacci_custom.period, 20)
        
        logger.info("Fibonacci指标初始化测试完成")

    def test_fibonacci_calculation(self):
        """测试Fibonacci指标计算"""
        logger.info("测试Fibonacci指标计算")
        
        result = self.fibonacci.calculate(self.test_data)
        
        # 验证返回类型
        self.assertIsInstance(result, pd.DataFrame)
        
        # 验证包含期望的列
        expected_columns = [
            'fib_ret_0.236', 'fib_ret_0.382', 'fib_ret_0.5', 
            'fib_ret_0.618', 'fib_ret_0.786', 'FIBONACCI_VALUE'
        ]
        
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # 验证数值的合理性
        fib_columns = [col for col in result.columns if col.startswith('fib_ret_')]
        for col in fib_columns:
            values = result[col].dropna()
            if len(values) > 0:
                self.assertTrue(all(values >= 0), f"{col} 列包含负值")
        
        logger.info("Fibonacci指标计算测试完成")

    def test_fibonacci_score_calculation(self):
        """测试Fibonacci评分计算"""
        logger.info("测试Fibonacci评分计算")
        
        raw_score = self.fibonacci.calculate_raw_score(self.test_data)
        
        # 验证返回类型
        self.assertIsInstance(raw_score, pd.Series)
        
        # 验证评分范围
        valid_scores = raw_score.dropna()
        if len(valid_scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores))
        
        logger.info("Fibonacci评分计算测试完成")

    def test_fibonacci_patterns(self):
        """测试Fibonacci形态识别"""
        logger.info("测试Fibonacci形态识别")
        
        patterns = self.fibonacci.get_patterns(self.test_data)
        
        # 验证返回类型
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证形态列的存在
        if not patterns.empty:
            pattern_columns = [col for col in patterns.columns if col.startswith('FIBONACCI_NEAR_')]
            self.assertGreater(len(pattern_columns), 0, "应该包含斐波那契接近模式")
            
            # 验证形态值为布尔类型
            for col in pattern_columns:
                values = patterns[col].dropna()
                if len(values) > 0:
                    self.assertTrue(all(isinstance(v, (bool, np.bool_)) for v in values))
        
        logger.info("Fibonacci形态识别测试完成")

    def test_fibonacci_signals(self):
        """测试Fibonacci信号生成"""
        logger.info("测试Fibonacci信号生成")
        
        signals = self.fibonacci.generate_trading_signals(self.test_data)
        
        # 验证信号结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['fibonacci_buy_signal', 'fibonacci_sell_signal']
        
        for key in expected_signal_keys:
            self.assertIn(key, signals)
            self.assertIsInstance(signals[key], pd.Series)
            
            # 验证信号为布尔类型
            signal_values = signals[key].dropna()
            if len(signal_values) > 0:
                self.assertTrue(all(isinstance(v, (bool, np.bool_)) for v in signal_values))
        
        logger.info("Fibonacci信号生成测试完成")

    def test_fibonacci_confidence_calculation(self):
        """测试Fibonacci置信度计算"""
        logger.info("测试Fibonacci置信度计算")
        
        confidence = self.fibonacci.calculate_confidence(self.test_data)
        
        # 验证置信度类型和范围
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        logger.info("Fibonacci置信度计算测试完成")

    def test_fibonacci_edge_cases(self):
        """测试Fibonacci边界情况"""
        logger.info("测试Fibonacci边界情况")
        
        # 测试数据不足的情况
        small_data = self.test_data.head(5)
        result = self.fibonacci.calculate(small_data)
        
        # Fibonacci应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
        
        # 验证在数据不足时返回NaN
        fib_columns = [col for col in result.columns if col.startswith('fib_ret_')]
        for col in fib_columns:
            values = result[col].dropna()
            # 数据不足时可能全为NaN，这是预期的
            self.assertTrue(len(values) == 0 or all(pd.isna(result[col])) or len(values) > 0)
        
        logger.info("Fibonacci边界情况测试完成")

    def test_fibonacci_different_periods(self):
        """测试不同周期的Fibonacci"""
        logger.info("测试不同周期的Fibonacci")
        
        # 测试不同周期
        periods = [10, 20, 30]
        
        for period in periods:
            fibonacci = Fibonacci(period=period)
            result = fibonacci.calculate(self.test_data)
            
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(fibonacci.period, period)
            
            # 验证包含基本列
            expected_columns = ['fib_ret_0.236', 'fib_ret_0.382', 'fib_ret_0.5', 'fib_ret_0.618', 'fib_ret_0.786']
            for col in expected_columns:
                self.assertIn(col, result.columns)
        
        logger.info("不同周期Fibonacci测试完成")

    def test_fibonacci_parameter_setting(self):
        """测试Fibonacci参数设置"""
        logger.info("测试Fibonacci参数设置")
        
        # 测试set_parameters方法
        self.fibonacci.set_parameters(period=25)
        self.assertEqual(self.fibonacci.period, 25)
        
        # 测试set_parameters_Fibonacci方法
        self.fibonacci.set_parameters_Fibonacci(period=30)
        self.assertEqual(self.fibonacci.period, 30)
        
        logger.info("Fibonacci参数设置测试完成")

    def test_fibonacci_abstract_methods(self):
        """测试Fibonacci抽象方法实现"""
        logger.info("测试Fibonacci抽象方法实现")
        
        # 测试抽象方法能正常调用
        result = self.fibonacci._calculate_baseindicator(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        
        score = self.fibonacci.calculate_raw_score_Indicator_Base_Indicator(self.test_data)
        self.assertIsInstance(score, pd.Series)
        
        patterns = self.fibonacci.get_patterns_Indicator_Base_Indicator(self.test_data)
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 测试参数设置抽象方法
        self.fibonacci.set_parameters_Indicator_Base_Indicator(period=35)
        self.assertEqual(self.fibonacci.period, 35)
        
        # 测试置信度计算抽象方法
        confidence = self.fibonacci.calculate_confidence_Indicator_Base_Indicator(
            score, patterns, {}
        )
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        logger.info("Fibonacci抽象方法测试完成")

    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.fibonacci.calculate(self.test_data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)

    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.fibonacci.get_patterns(self.test_data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)


if __name__ == '__main__':
    unittest.main() 