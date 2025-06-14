"""
FibonacciTools指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.fibonacci_tools import FibonacciTools, FibonacciType
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestFibonacciTools(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """FibonacciTools指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = FibonacciTools()
        self.expected_columns = [
            'FIB_GOLDEN_RATIO_SUPPORT', 'FIB_GOLDEN_RATIO_RESISTANCE',
            'FIB_50_PERCENT_RETRACEMENT', 'FIB_382_RETRACEMENT', 'FIB_618_RETRACEMENT',
            'FIB_EXTENSION_TARGET', 'FIB_GOLDEN_EXTENSION', 'FIB_100_EXTENSION',
            'FIB_CLUSTER_SUPPORT', 'FIB_CLUSTER_RESISTANCE',
            'FIB_BREAKOUT_UP', 'FIB_BREAKOUT_DOWN',
            'FIB_SUPPORT_BOUNCE', 'FIB_RESISTANCE_PULLBACK',
            'FIB_TIME_CYCLE', 'FIB_VOLUME_CONFIRMATION',
            'FIB_TREND_ALIGNMENT', 'FIB_REVERSAL_SIGNAL'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_fibonacci_tools_initialization(self):
        """测试FibonacciTools初始化"""
        # 测试默认初始化
        default_indicator = FibonacciTools()
        self.assertEqual(default_indicator.name, "FibonacciTools")
        self.assertIn("斐波那契工具指标", default_indicator.description)
    
    def test_fibonacci_tools_calculation_accuracy(self):
        """测试FibonacciTools计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证FibonacciTools列存在
        self.assertIsInstance(result, pd.DataFrame)
        
        # 验证包含斐波那契水平线
        fib_columns = [col for col in result.columns if 'fib_' in col]
        self.assertGreater(len(fib_columns), 0, "应该包含斐波那契水平线")
    
    def test_fibonacci_tools_score_range(self):
        """测试FibonacciTools评分范围"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        if 'score' in raw_score_df.columns:
            valid_scores = raw_score_df['score'].dropna()
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_fibonacci_tools_confidence_calculation(self):
        """测试FibonacciTools置信度计算"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        raw_score = raw_score_df['score'] if 'score' in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_fibonacci_tools_required_columns(self):
        """测试FibonacciTools必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_fibonacci_tools_patterns(self):
        """测试FibonacciTools形态识别"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 然后获取形态
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证基本的形态列存在
        if not patterns.empty and len(patterns.columns) > 0:
            for pattern in self.expected_columns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_fibonacci_tools_signals(self):
        """测试FibonacciTools信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_fibonacci_tools_retracement_calculation(self):
        """测试FibonacciTools回调线计算"""
        # 使用足够的数据进行回调线计算
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        # 手动指定摆动点
        swing_high_idx = 80
        swing_low_idx = 20
        
        result = self.indicator.calculate_retracement(long_data, swing_high_idx, swing_low_idx)
        
        # 验证回调线列存在
        retracement_columns = [col for col in result.columns if 'fib_' in col and 'ext_' not in col]
        self.assertGreater(len(retracement_columns), 0, "应该包含回调线")
        
        # 验证关键回调位
        key_levels = ['fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786']
        for level in key_levels:
            self.assertIn(level, result.columns, f"缺少回调位: {level}")
    
    def test_fibonacci_tools_extension_calculation(self):
        """测试FibonacciTools扩展线计算"""
        # 使用足够的数据进行扩展线计算
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        # 手动指定摆动点
        swing_high_idx = 80
        swing_low_idx = 20
        
        result = self.indicator.calculate_extension(long_data, swing_high_idx, swing_low_idx)
        
        # 验证扩展线列存在
        extension_columns = [col for col in result.columns if 'fib_ext_' in col]
        self.assertGreater(len(extension_columns), 0, "应该包含扩展线")
    
    def test_fibonacci_tools_time_series_calculation(self):
        """测试FibonacciTools时间序列计算"""
        # 使用足够的数据进行时间序列计算
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 150}
        ])
        
        start_idx = 10
        result = self.indicator.calculate_time_series(long_data, start_idx)
        
        # 验证时间序列列存在
        time_columns = [col for col in result.columns if 'fib_time_' in col]
        self.assertGreater(len(time_columns), 0, "应该包含时间序列")
    
    def test_fibonacci_tools_swing_point_detection(self):
        """测试FibonacciTools摆动点检测"""
        # 使用足够的数据进行摆动点检测
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        swing_high_idx, swing_low_idx = self.indicator._detect_swing_points(long_data)
        
        # 验证摆动点索引有效
        self.assertIsInstance(swing_high_idx, int)
        self.assertIsInstance(swing_low_idx, int)
        self.assertGreaterEqual(swing_high_idx, 0)
        self.assertGreaterEqual(swing_low_idx, 0)
        self.assertLess(swing_high_idx, len(long_data))
        self.assertLess(swing_low_idx, len(long_data))
    
    def test_fibonacci_tools_pattern_identification(self):
        """测试FibonacciTools形态识别"""
        # 使用足够的数据进行形态识别
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        patterns = self.indicator.identify_patterns(long_data)
        
        # 验证返回形态列表
        self.assertIsInstance(patterns, list)
    
    def test_fibonacci_tools_score_calculation(self):
        """测试FibonacciTools评分计算"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(raw_score_df, pd.DataFrame)
        
        # 验证包含score列
        self.assertIn('score', raw_score_df.columns)
        
        # 验证评分值的合理性
        scores = raw_score_df['score'].dropna()
        if len(scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in scores), "评分应在0-100范围内")
    
    def test_fibonacci_tools_fibonacci_types(self):
        """测试FibonacciTools斐波那契类型"""
        # 测试回调线类型
        result_retracement = self.indicator.calculate(self.data, fib_type=FibonacciType.RETRACEMENT)
        self.assertIsInstance(result_retracement, pd.DataFrame)
        
        # 测试扩展线类型
        result_extension = self.indicator.calculate(self.data, fib_type=FibonacciType.EXTENSION)
        self.assertIsInstance(result_extension, pd.DataFrame)
        
        # 测试时间序列类型
        result_time = self.indicator.calculate(self.data, fib_type=FibonacciType.TIME_SERIES)
        self.assertIsInstance(result_time, pd.DataFrame)
    
    def test_fibonacci_tools_volume_confirmation(self):
        """测试FibonacciTools成交量确认"""
        # 生成带成交量的数据
        data_with_volume = self.data.copy()
        data_with_volume['volume'] = np.random.randint(1000, 10000, len(data_with_volume))
        
        raw_score_df = self.indicator.calculate_raw_score(data_with_volume)
        
        # 验证成交量确认在评分中的影响
        self.assertIsInstance(raw_score_df, pd.DataFrame)
        self.assertIn('score', raw_score_df.columns)
    
    def test_fibonacci_tools_edge_cases(self):
        """测试FibonacciTools边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(5)
        result = self.indicator.calculate(small_data)
        
        # FibonacciTools应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_fibonacci_tools_validation(self):
        """测试FibonacciTools数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['high'], axis=1)

        # BaseIndicator会处理缺失列并返回空DataFrame
        result = self.indicator.calculate(invalid_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_fibonacci_tools_indicator_type(self):
        """测试FibonacciTools指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "FIBONACCITOOLS", "指标类型应该是FIBONACCITOOLS")
    
    def test_fibonacci_tools_register_patterns(self):
        """测试FibonacciTools形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
