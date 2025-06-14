"""
EnhancedMACD指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.trend.enhanced_macd import EnhancedMACD
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestEnhancedMACD(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """EnhancedMACD指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = EnhancedMACD(fast_period=12, slow_period=26, signal_period=9)
        self.expected_columns = [
            'macd', 'macd_signal', 'macd_hist', 'fast_ema', 'slow_ema',
            'hist_change_rate', 'trend_strength', 'zero_cross_angle',
            'signal_cross_angle', 'macd_deviation'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_enhanced_macd_calculation_accuracy(self):
        """测试EnhancedMACD计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证EnhancedMACD列存在
        for col in self.expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证MACD值的合理性
        macd_values = result['macd'].dropna()
        signal_values = result['macd_signal'].dropna()
        hist_values = result['macd_hist'].dropna()
        
        if len(macd_values) > 0:
            # MACD值应该是有限的数值
            self.assertTrue(all(np.isfinite(v) for v in macd_values), "MACD值应该是有限数值")
            self.assertTrue(all(np.isfinite(v) for v in signal_values), "信号线值应该是有限数值")
            self.assertTrue(all(np.isfinite(v) for v in hist_values), "柱状体值应该是有限数值")
    
    def test_enhanced_macd_manual_calculation(self):
        """测试EnhancedMACD手动计算验证"""
        # 创建简单的测试数据
        simple_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104] * 10,
            'high': [101, 102, 103, 104, 105] * 10,
            'low': [99, 100, 101, 102, 103] * 10,
            'close': [100.5, 101.5, 102.5, 103.5, 104.5] * 10,
            'volume': [1000, 1200, 800, 1500, 900] * 10
        })
        
        # 使用较小的周期便于验证
        test_indicator = EnhancedMACD(fast_period=5, slow_period=10, signal_period=3)
        result = test_indicator.calculate(simple_data)
        
        # 验证MACD计算逻辑
        if len(result) >= 10:
            macd_value = result['macd'].iloc[9]
            signal_value = result['macd_signal'].iloc[9]
            hist_value = result['macd_hist'].iloc[9]
            
            if not pd.isna(macd_value) and not pd.isna(signal_value) and not pd.isna(hist_value):
                # 验证柱状体 = MACD - 信号线
                self.assertAlmostEqual(hist_value, macd_value - signal_value, places=6,
                                     msg="柱状体应该等于MACD减去信号线")
    
    def test_enhanced_macd_score_range(self):
        """测试EnhancedMACD评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_macd_confidence_calculation(self):
        """测试EnhancedMACD置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_enhanced_macd_parameter_update(self):
        """测试EnhancedMACD参数更新"""
        new_fast = 8
        new_slow = 21
        new_signal = 6
        self.indicator.set_parameters(fast_period=new_fast, slow_period=new_slow, signal_period=new_signal)
        
        # 验证参数更新
        self.assertEqual(self.indicator._fast_period, new_fast)
        self.assertEqual(self.indicator._slow_period, new_slow)
        self.assertEqual(self.indicator._signal_period, new_signal)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('macd', result.columns)
        self.assertIn('macd_signal', result.columns)
        self.assertIn('macd_hist', result.columns)
    
    def test_enhanced_macd_required_columns(self):
        """测试EnhancedMACD必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_enhanced_macd_comprehensive_score(self):
        """测试EnhancedMACD综合评分"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 然后计算评分
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        self.assertIsInstance(raw_score, pd.Series)
        
        # 验证评分范围
        valid_scores = raw_score.dropna()
        if len(valid_scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_macd_patterns(self):
        """测试EnhancedMACD形态识别"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 然后获取形态
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证基本的形态列存在
        if not patterns.empty and len(patterns.columns) > 0:
            expected_patterns = [
                'MACD_GOLDEN_CROSS', 'MACD_DEATH_CROSS',
                'MACD_ZERO_CROSS_UP', 'MACD_ZERO_CROSS_DOWN',
                'MACD_HIST_POSITIVE', 'MACD_HIST_NEGATIVE'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_enhanced_macd_multi_period_analysis(self):
        """测试EnhancedMACD多周期分析"""
        # 使用多周期参数
        multi_indicator = EnhancedMACD(
            fast_period=12, slow_period=26, signal_period=9,
            multi_periods=[(8, 17, 9), (12, 26, 9), (24, 52, 18)]
        )
        result = multi_indicator.calculate(self.data)
        
        # 验证多周期MACD列存在
        expected_multi_columns = [
            'macd_8_17_9', 'macd_signal_8_17_9', 'macd_hist_8_17_9',
            'macd_24_52_18', 'macd_signal_24_52_18', 'macd_hist_24_52_18'
        ]
        
        for col in expected_multi_columns:
            self.assertIn(col, result.columns, f"缺少多周期列: {col}")
    
    def test_enhanced_macd_volume_weighted(self):
        """测试EnhancedMACD成交量加权"""
        # 测试成交量加权模式
        volume_indicator = EnhancedMACD(
            fast_period=12, slow_period=26, signal_period=9,
            volume_weighted=True
        )
        result = volume_indicator.calculate(self.data)
        
        # 验证成交量加权MACD计算
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('macd', result.columns)
        self.assertIn('macd_signal', result.columns)
        self.assertIn('macd_hist', result.columns)
    
    def test_enhanced_macd_volatility_adaptation(self):
        """测试EnhancedMACD波动率自适应"""
        # 测试自适应模式
        adaptive_indicator = EnhancedMACD(
            fast_period=12, slow_period=26, signal_period=9,
            adapt_to_volatility=True
        )
        result = adaptive_indicator.calculate(self.data)
        
        # 验证自适应功能
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('macd', result.columns)
        
        # 测试非自适应模式
        non_adaptive_indicator = EnhancedMACD(
            fast_period=12, slow_period=26, signal_period=9,
            adapt_to_volatility=False
        )
        result2 = non_adaptive_indicator.calculate(self.data)
        
        self.assertIsInstance(result2, pd.DataFrame)
        self.assertIn('macd', result2.columns)
    
    def test_enhanced_macd_sensitivity_adjustment(self):
        """测试EnhancedMACD灵敏度调整"""
        # 测试不同灵敏度设置
        sensitivities = [0.5, 1.0, 1.5, 2.0]
        
        for sensitivity in sensitivities:
            test_indicator = EnhancedMACD(
                fast_period=12, slow_period=26, signal_period=9,
                sensitivity=sensitivity
            )
            result = test_indicator.calculate(self.data)
            
            # 验证灵敏度调整功能
            self.assertIsInstance(result, pd.DataFrame)
            self.assertIn('macd', result.columns)
            
            # 验证MACD值的合理性
            macd_values = result['macd'].dropna()
            if len(macd_values) > 0:
                self.assertTrue(all(np.isfinite(v) for v in macd_values), 
                               f"灵敏度{sensitivity}时MACD值应该是有限数值")
    
    def test_enhanced_macd_signals(self):
        """测试EnhancedMACD信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_enhanced_macd_trend_strength(self):
        """测试EnhancedMACD趋势强度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证趋势强度列存在
        self.assertIn('trend_strength', result.columns)
        
        # 验证趋势强度值的合理性
        trend_strength = result['trend_strength'].dropna()
        if len(trend_strength) > 0:
            # 趋势强度应该在-1到1范围内
            self.assertTrue(all(-1 <= v <= 1 for v in trend_strength), 
                           "趋势强度应该在-1到1范围内")
    
    def test_enhanced_macd_histogram_change_rate(self):
        """测试EnhancedMACD柱状体变化率"""
        result = self.indicator.calculate(self.data)
        
        # 验证柱状体变化率列存在
        self.assertIn('hist_change_rate', result.columns)
        
        # 验证变化率值的合理性
        change_rate = result['hist_change_rate'].dropna()
        if len(change_rate) > 0:
            # 变化率应该是有限数值
            self.assertTrue(all(np.isfinite(v) for v in change_rate), 
                           "柱状体变化率应该是有限数值")
    
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
    
    def test_enhanced_macd_register_patterns(self):
        """测试EnhancedMACD形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_enhanced_macd_edge_cases(self):
        """测试EnhancedMACD边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(10)
        result = self.indicator.calculate(small_data)
        
        # EnhancedMACD应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('macd', result.columns)
    
    def test_enhanced_macd_validation(self):
        """测试EnhancedMACD数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['close'], axis=1)
        
        with self.assertRaises((ValueError, KeyError)):
            self.indicator.calculate(invalid_data)
    
    def test_enhanced_macd_indicator_type(self):
        """测试EnhancedMACD指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "trend", "指标类型应该是trend")


if __name__ == '__main__':
    unittest.main()
