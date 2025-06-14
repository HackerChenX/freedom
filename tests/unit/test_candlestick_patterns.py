"""
CandlestickPatterns指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.pattern.candlestick_patterns import CandlestickPatterns, PatternType
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestCandlestickPatterns(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """CandlestickPatterns指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = CandlestickPatterns()
        self.expected_columns = [
            'doji', 'hammer', 'hanging_man', 'long_legged_doji', 'gravestone_doji', 'shooting_star',
            'engulfing_bullish', 'engulfing_bearish', 'dark_cloud_cover', 'piercing_line',
            'morning_star', 'evening_star', 'harami_bullish', 'single_needle_bottom',
            'head_shoulders_top', 'head_shoulders_bottom', 'double_top', 'double_bottom',
            'island_reversal', 'v_reversal'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_candlestick_patterns_initialization(self):
        """测试CandlestickPatterns初始化"""
        # 测试默认初始化
        default_indicator = CandlestickPatterns()
        self.assertEqual(default_indicator.name, "CandlestickPatterns")
        self.assertIn("K线形态识别指标", default_indicator.description)
    
    def test_candlestick_patterns_calculation_accuracy(self):
        """测试CandlestickPatterns计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证CandlestickPatterns列存在
        for col in self.expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证形态值的合理性
        for col in self.expected_columns:
            if col in result.columns:
                pattern_values = result[col].dropna()
                if len(pattern_values) > 0:
                    # 形态值应该是布尔值
                    unique_values = pattern_values.unique()
                    for val in unique_values:
                        self.assertIsInstance(val, (bool, np.bool_), f"{col}应该是布尔值")
    
    def test_candlestick_patterns_score_range(self):
        """测试CandlestickPatterns评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        if 'score' in raw_score.columns:
            valid_scores = raw_score['score'].dropna()
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_candlestick_patterns_confidence_calculation(self):
        """测试CandlestickPatterns置信度计算"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        raw_score = raw_score_df['score'] if 'score' in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_candlestick_patterns_parameter_update(self):
        """测试CandlestickPatterns参数更新"""
        # CandlestickPatterns通常没有可变参数，但应该有set_parameters方法
        self.indicator.set_parameters()
        # 验证方法存在且不抛出异常
        self.assertTrue(True, "set_parameters方法应该存在")
    
    def test_candlestick_patterns_required_columns(self):
        """测试CandlestickPatterns必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_candlestick_patterns_patterns(self):
        """测试CandlestickPatterns形态识别"""
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
    
    def test_candlestick_patterns_signals(self):
        """测试CandlestickPatterns信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_candlestick_patterns_single_patterns(self):
        """测试CandlestickPatterns单日形态"""
        result = self.indicator.calculate(self.data)
        
        # 验证单日形态列存在
        single_patterns = ['doji', 'hammer', 'hanging_man', 'long_legged_doji', 
                          'gravestone_doji', 'shooting_star']
        
        for pattern in single_patterns:
            self.assertIn(pattern, result.columns, f"缺少单日形态: {pattern}")
            
            # 验证形态值的合理性
            pattern_values = result[pattern].dropna()
            if len(pattern_values) > 0:
                unique_values = pattern_values.unique()
                for val in unique_values:
                    self.assertIsInstance(val, (bool, np.bool_), f"{pattern}应该是布尔值")
    
    def test_candlestick_patterns_combined_patterns(self):
        """测试CandlestickPatterns组合形态"""
        result = self.indicator.calculate(self.data)
        
        # 验证组合形态列存在
        combined_patterns = ['engulfing_bullish', 'engulfing_bearish', 'dark_cloud_cover',
                           'piercing_line', 'morning_star', 'evening_star', 
                           'harami_bullish', 'single_needle_bottom']
        
        for pattern in combined_patterns:
            self.assertIn(pattern, result.columns, f"缺少组合形态: {pattern}")
            
            # 验证形态值的合理性
            pattern_values = result[pattern].dropna()
            if len(pattern_values) > 0:
                unique_values = pattern_values.unique()
                for val in unique_values:
                    self.assertIsInstance(val, (bool, np.bool_), f"{pattern}应该是布尔值")
    
    def test_candlestick_patterns_complex_patterns(self):
        """测试CandlestickPatterns复合形态"""
        # 需要足够的数据进行复合形态识别
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        result = self.indicator.calculate(long_data)
        
        # 验证复合形态列存在
        complex_patterns = ['head_shoulders_top', 'head_shoulders_bottom', 'double_top',
                          'double_bottom', 'island_reversal', 'v_reversal']
        
        for pattern in complex_patterns:
            self.assertIn(pattern, result.columns, f"缺少复合形态: {pattern}")
            
            # 验证形态值的合理性
            pattern_values = result[pattern].dropna()
            if len(pattern_values) > 0:
                unique_values = pattern_values.unique()
                for val in unique_values:
                    self.assertIsInstance(val, (bool, np.bool_), f"{pattern}应该是布尔值")
    
    def test_candlestick_patterns_latest_patterns(self):
        """测试CandlestickPatterns最新形态获取"""
        latest_patterns = self.indicator.get_latest_patterns(self.data, lookback=5)
        
        # 验证返回字典
        self.assertIsInstance(latest_patterns, dict)
        
        # 验证字典值为布尔值
        for pattern_name, pattern_found in latest_patterns.items():
            self.assertIsInstance(pattern_found, bool, f"{pattern_name}应该是布尔值")
    
    def test_candlestick_patterns_identify_patterns(self):
        """测试CandlestickPatterns形态识别"""
        identified_patterns = self.indicator.identify_patterns(self.data)
        
        # 验证返回列表
        self.assertIsInstance(identified_patterns, list)
        
        # 验证列表元素为字符串
        for pattern in identified_patterns:
            self.assertIsInstance(pattern, str, "识别的形态应该是字符串")
    
    def test_candlestick_patterns_generate_signals_method(self):
        """测试CandlestickPatterns信号生成方法"""
        signals = self.indicator.generate_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, pd.DataFrame)
        
        # 验证基本信号列存在
        expected_signal_columns = ['buy_signal', 'sell_signal', 'neutral_signal', 
                                 'trend', 'score', 'signal_type', 'confidence']
        
        for col in expected_signal_columns:
            self.assertIn(col, signals.columns, f"缺少信号列: {col}")
    
    def test_candlestick_patterns_pattern_types(self):
        """测试CandlestickPatterns形态类型枚举"""
        # 验证PatternType枚举存在
        self.assertTrue(hasattr(PatternType, 'DOJI'))
        self.assertTrue(hasattr(PatternType, 'HAMMER'))
        self.assertTrue(hasattr(PatternType, 'ENGULFING_BULLISH'))
        self.assertTrue(hasattr(PatternType, 'HEAD_SHOULDERS_TOP'))
        
        # 验证枚举值
        self.assertEqual(PatternType.DOJI.value, "十字星")
        self.assertEqual(PatternType.HAMMER.value, "锤头线")
        self.assertEqual(PatternType.ENGULFING_BULLISH.value, "阳包阴")
    
    def test_candlestick_patterns_score_calculation(self):
        """测试CandlestickPatterns评分计算"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(raw_score_df, pd.DataFrame)
        
        # 验证包含score列
        self.assertIn('score', raw_score_df.columns)
        
        # 验证评分值的合理性
        scores = raw_score_df['score'].dropna()
        if len(scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in scores), "评分应在0-100范围内")
    
    def test_candlestick_patterns_volume_confirmation(self):
        """测试CandlestickPatterns成交量确认"""
        # 生成带成交量的数据
        data_with_volume = self.data.copy()
        data_with_volume['volume'] = np.random.randint(1000, 10000, len(data_with_volume))
        
        signals = self.indicator.generate_signals(data_with_volume)
        
        # 验证成交量确认列存在
        self.assertIn('volume_confirmation', signals.columns)
        
        # 验证成交量确认值为布尔值
        vol_confirm_values = signals['volume_confirmation'].dropna()
        if len(vol_confirm_values) > 0:
            unique_values = vol_confirm_values.unique()
            for val in unique_values:
                self.assertIsInstance(val, (bool, np.bool_), "成交量确认应该是布尔值")
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        for col in self.expected_columns:
            self.assertIn(col, result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_candlestick_patterns_register_patterns(self):
        """测试CandlestickPatterns形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_candlestick_patterns_edge_cases(self):
        """测试CandlestickPatterns边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(10)
        result = self.indicator.calculate(small_data)
        
        # CandlestickPatterns应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
        for col in self.expected_columns:
            self.assertIn(col, result.columns)
    
    def test_candlestick_patterns_validation(self):
        """测试CandlestickPatterns数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['high'], axis=1)

        # BaseIndicator会处理缺失列并返回空DataFrame
        result = self.indicator.calculate(invalid_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_candlestick_patterns_indicator_type(self):
        """测试CandlestickPatterns指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "CANDLESTICKPATTERNS", "指标类型应该是CANDLESTICKPATTERNS")


if __name__ == '__main__':
    unittest.main()
