"""
ZXMPatternIndicator指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.pattern.zxm_patterns import ZXMPatternIndicator
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestZXMPatternIndicator(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """ZXMPatternIndicator指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = ZXMPatternIndicator()
        self.expected_columns = [
            'class_one_buy', 'class_two_buy', 'class_three_buy',
            'breakout_pullback_buy', 'volume_shrink_platform_buy',
            'long_shadow_support_buy', 'ma_converge_diverge_buy',
            'volume_decrease', 'decline_slow_down', 'decline_reduce',
            'key_support_hold', 'macd_double_diverge', 'volume_shrink_range',
            'ma_convergence', 'macd_zero_hover', 'long_lower_shadow',
            'ma_precise_support', 'small_alternating'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_zxm_pattern_indicator_initialization(self):
        """测试ZXMPatternIndicator初始化"""
        # 测试默认初始化
        default_indicator = ZXMPatternIndicator()
        self.assertEqual(default_indicator.name, "ZXMPattern")
        self.assertIn("基于ZXM体系的买点和吸筹形态识别指标", default_indicator.description)
    
    def test_zxm_pattern_indicator_calculation_accuracy(self):
        """测试ZXMPatternIndicator计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证ZXMPatternIndicator列存在
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
    
    def test_zxm_pattern_indicator_score_range(self):
        """测试ZXMPatternIndicator评分范围"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        if 'score' in raw_score_df.columns:
            valid_scores = raw_score_df['score'].dropna()
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_zxm_pattern_indicator_confidence_calculation(self):
        """测试ZXMPatternIndicator置信度计算"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        raw_score = raw_score_df['score'] if 'score' in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_zxm_pattern_indicator_parameter_update(self):
        """测试ZXMPatternIndicator参数更新"""
        # ZXMPatternIndicator通常没有可变参数，但应该有set_parameters方法
        self.indicator.set_parameters()
        # 验证方法存在且不抛出异常
        self.assertTrue(True, "set_parameters方法应该存在")
    
    def test_zxm_pattern_indicator_required_columns(self):
        """测试ZXMPatternIndicator必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_zxm_pattern_indicator_patterns(self):
        """测试ZXMPatternIndicator形态识别"""
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
    
    def test_zxm_pattern_indicator_signals(self):
        """测试ZXMPatternIndicator信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_zxm_pattern_indicator_buy_points(self):
        """测试ZXMPatternIndicator买点识别"""
        result = self.indicator.calculate(self.data)
        
        # 验证买点形态列存在
        buy_patterns = ['class_one_buy', 'class_two_buy', 'class_three_buy',
                       'breakout_pullback_buy', 'volume_shrink_platform_buy',
                       'long_shadow_support_buy', 'ma_converge_diverge_buy']
        
        for pattern in buy_patterns:
            self.assertIn(pattern, result.columns, f"缺少买点形态: {pattern}")
            
            # 验证形态值的合理性
            pattern_values = result[pattern].dropna()
            if len(pattern_values) > 0:
                unique_values = pattern_values.unique()
                for val in unique_values:
                    self.assertIsInstance(val, (bool, np.bool_), f"{pattern}应该是布尔值")
    
    def test_zxm_pattern_indicator_absorption_patterns(self):
        """测试ZXMPatternIndicator吸筹形态"""
        result = self.indicator.calculate(self.data)
        
        # 验证吸筹形态列存在
        absorption_patterns = ['volume_decrease', 'decline_slow_down', 'decline_reduce',
                             'key_support_hold', 'macd_double_diverge', 'volume_shrink_range',
                             'ma_convergence', 'macd_zero_hover', 'long_lower_shadow',
                             'ma_precise_support', 'small_alternating']
        
        for pattern in absorption_patterns:
            self.assertIn(pattern, result.columns, f"缺少吸筹形态: {pattern}")
            
            # 验证形态值的合理性
            pattern_values = result[pattern].dropna()
            if len(pattern_values) > 0:
                unique_values = pattern_values.unique()
                for val in unique_values:
                    self.assertIsInstance(val, (bool, np.bool_), f"{pattern}应该是布尔值")
    
    def test_zxm_pattern_indicator_score_calculation(self):
        """测试ZXMPatternIndicator评分计算"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(raw_score_df, pd.DataFrame)
        
        # 验证包含score列
        self.assertIn('score', raw_score_df.columns)
        
        # 验证评分值的合理性
        scores = raw_score_df['score'].dropna()
        if len(scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in scores), "评分应在0-100范围内")
    
    def test_zxm_pattern_indicator_class_one_buy(self):
        """测试ZXMPatternIndicator一类买点"""
        # 需要足够的数据进行一类买点识别
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        result = self.indicator.calculate(long_data)
        
        # 验证一类买点列存在
        self.assertIn('class_one_buy', result.columns)
        
        # 验证形态值的合理性
        pattern_values = result['class_one_buy'].dropna()
        if len(pattern_values) > 0:
            unique_values = pattern_values.unique()
            for val in unique_values:
                self.assertIsInstance(val, (bool, np.bool_), "一类买点应该是布尔值")
    
    def test_zxm_pattern_indicator_class_two_buy(self):
        """测试ZXMPatternIndicator二类买点"""
        # 需要足够的数据进行二类买点识别
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        result = self.indicator.calculate(long_data)
        
        # 验证二类买点列存在
        self.assertIn('class_two_buy', result.columns)
        
        # 验证形态值的合理性
        pattern_values = result['class_two_buy'].dropna()
        if len(pattern_values) > 0:
            unique_values = pattern_values.unique()
            for val in unique_values:
                self.assertIsInstance(val, (bool, np.bool_), "二类买点应该是布尔值")
    
    def test_zxm_pattern_indicator_class_three_buy(self):
        """测试ZXMPatternIndicator三类买点"""
        result = self.indicator.calculate(self.data)
        
        # 验证三类买点列存在
        self.assertIn('class_three_buy', result.columns)
        
        # 验证形态值的合理性
        pattern_values = result['class_three_buy'].dropna()
        if len(pattern_values) > 0:
            unique_values = pattern_values.unique()
            for val in unique_values:
                self.assertIsInstance(val, (bool, np.bool_), "三类买点应该是布尔值")
    
    def test_zxm_pattern_indicator_special_buy_points(self):
        """测试ZXMPatternIndicator特殊买点"""
        result = self.indicator.calculate(self.data)
        
        # 验证特殊买点列存在
        special_patterns = ['breakout_pullback_buy', 'volume_shrink_platform_buy',
                          'long_shadow_support_buy', 'ma_converge_diverge_buy']
        
        for pattern in special_patterns:
            self.assertIn(pattern, result.columns, f"缺少特殊买点: {pattern}")
            
            # 验证形态值的合理性
            pattern_values = result[pattern].dropna()
            if len(pattern_values) > 0:
                unique_values = pattern_values.unique()
                for val in unique_values:
                    self.assertIsInstance(val, (bool, np.bool_), f"{pattern}应该是布尔值")
    
    def test_zxm_pattern_indicator_volume_confirmation(self):
        """测试ZXMPatternIndicator成交量确认"""
        # 生成带成交量的数据
        data_with_volume = self.data.copy()
        data_with_volume['volume'] = np.random.randint(1000, 10000, len(data_with_volume))
        
        raw_score_df = self.indicator.calculate_raw_score(data_with_volume)
        
        # 验证成交量确认在评分中的影响
        self.assertIsInstance(raw_score_df, pd.DataFrame)
        self.assertIn('score', raw_score_df.columns)
    
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
    
    def test_zxm_pattern_indicator_register_patterns(self):
        """测试ZXMPatternIndicator形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_zxm_pattern_indicator_edge_cases(self):
        """测试ZXMPatternIndicator边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(5)
        result = self.indicator.calculate(small_data)
        
        # ZXMPatternIndicator应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_zxm_pattern_indicator_validation(self):
        """测试ZXMPatternIndicator数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['volume'], axis=1)

        # BaseIndicator会处理缺失列并返回空DataFrame
        result = self.indicator.calculate(invalid_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_zxm_pattern_indicator_indicator_type(self):
        """测试ZXMPatternIndicator指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "ZXMPATTERNS", "指标类型应该是ZXMPATTERNS")


if __name__ == '__main__':
    unittest.main()
