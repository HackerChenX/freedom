"""
EnhancedTRIX指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.trend.enhanced_trix import EnhancedTRIX
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestEnhancedTRIX(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """EnhancedTRIX指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = EnhancedTRIX(n=12, m=9, secondary_n=24)
        self.expected_columns = [
            'TRIX', 'MATRIX', 'trix_secondary', 'matrix_secondary',
            'trix_momentum', 'trix_slope', 'trix_accel', 'trix_volatility'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_enhanced_trix_calculation_accuracy(self):
        """测试EnhancedTRIX计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证EnhancedTRIX列存在
        for col in self.expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证TRIX值的合理性
        trix_values = result['TRIX'].dropna()
        matrix_values = result['MATRIX'].dropna()
        
        if len(trix_values) > 0:
            # TRIX值应该是有限的数值
            self.assertTrue(all(np.isfinite(v) for v in trix_values), "TRIX值应该是有限数值")
            self.assertTrue(all(np.isfinite(v) for v in matrix_values), "MATRIX值应该是有限数值")
    
    def test_enhanced_trix_manual_calculation(self):
        """测试EnhancedTRIX手动计算验证"""
        # 创建简单的测试数据
        simple_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104] * 10,
            'high': [101, 102, 103, 104, 105] * 10,
            'low': [99, 100, 101, 102, 103] * 10,
            'close': [100.5, 101.5, 102.5, 103.5, 104.5] * 10,
            'volume': [1000, 1200, 800, 1500, 900] * 10
        })
        
        # 使用较小的周期便于验证
        test_indicator = EnhancedTRIX(n=6, m=3, secondary_n=12)
        result = test_indicator.calculate(simple_data)
        
        # 验证TRIX计算逻辑
        if len(result) >= 10:
            trix_value = result['TRIX'].iloc[9]
            matrix_value = result['MATRIX'].iloc[9]
            
            if not pd.isna(trix_value) and not pd.isna(matrix_value):
                # TRIX值应该是合理的数值
                self.assertTrue(-10 <= trix_value <= 10, "TRIX值应该在合理范围内")
                self.assertTrue(-10 <= matrix_value <= 10, "MATRIX值应该在合理范围内")
    
    def test_enhanced_trix_score_range(self):
        """测试EnhancedTRIX评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_trix_confidence_calculation(self):
        """测试EnhancedTRIX置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_enhanced_trix_parameter_update(self):
        """测试EnhancedTRIX参数更新"""
        new_n = 8
        new_m = 6
        new_secondary_n = 16
        self.indicator.set_parameters(n=new_n, m=new_m, secondary_n=new_secondary_n)
        
        # 验证参数更新
        self.assertEqual(self.indicator.n, new_n)
        self.assertEqual(self.indicator.m, new_m)
        self.assertEqual(self.indicator.secondary_n, new_secondary_n)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('TRIX', result.columns)
        self.assertIn('MATRIX', result.columns)
    
    def test_enhanced_trix_required_columns(self):
        """测试EnhancedTRIX必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_enhanced_trix_comprehensive_score(self):
        """测试EnhancedTRIX综合评分"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 然后计算评分
        score = self.indicator.calculate_score()
        
        self.assertIsInstance(score, pd.Series)
        
        # 验证评分范围
        valid_scores = score.dropna()
        if len(valid_scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_trix_patterns(self):
        """测试EnhancedTRIX形态识别"""
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
                'above_zero', 'below_zero', 'rising', 'falling',
                'golden_cross', 'death_cross', 'cross_up_zero', 'cross_down_zero'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_enhanced_trix_divergence_detection(self):
        """测试EnhancedTRIX背离检测"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 检测背离
        divergence = self.indicator.detect_divergence()
        
        # 验证背离检测结果
        self.assertIsInstance(divergence, pd.DataFrame)
        
        if not divergence.empty:
            expected_divergence_columns = [
                'bullish_divergence', 'bearish_divergence',
                'hidden_bullish_divergence', 'hidden_bearish_divergence',
                'divergence_strength'
            ]
            
            for col in expected_divergence_columns:
                self.assertIn(col, divergence.columns, f"缺少背离分析列: {col}")
    
    def test_enhanced_trix_multi_period_synergy(self):
        """测试EnhancedTRIX多周期协同分析"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 分析多周期协同
        synergy = self.indicator.analyze_multi_period_synergy()
        
        # 验证协同分析结果
        self.assertIsInstance(synergy, pd.DataFrame)
        
        if not synergy.empty:
            expected_synergy_columns = [
                'primary_above_zero', 'primary_below_zero',
                'secondary_above_zero', 'secondary_below_zero',
                'bullish_ratio', 'bearish_ratio', 'consensus_score'
            ]
            
            for col in expected_synergy_columns:
                self.assertIn(col, synergy.columns, f"缺少协同分析列: {col}")
    
    def test_enhanced_trix_zero_cross_quality(self):
        """测试EnhancedTRIX零轴交叉质量评估"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 评估零轴交叉质量
        quality = self.indicator.evaluate_zero_cross_quality()
        
        # 验证质量评估结果
        self.assertIsInstance(quality, pd.DataFrame)
        
        if not quality.empty:
            expected_quality_columns = [
                'cross_up_zero', 'cross_down_zero', 'cross_angle',
                'post_cross_acceleration', 'cross_persistence', 'cross_quality_score'
            ]
            
            for col in expected_quality_columns:
                self.assertIn(col, quality.columns, f"缺少质量评估列: {col}")
    
    def test_enhanced_trix_signals(self):
        """测试EnhancedTRIX信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_enhanced_trix_market_environment(self):
        """测试EnhancedTRIX市场环境设置"""
        # 测试设置不同的市场环境
        environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        
        for env in environments:
            self.indicator.set_market_environment(env)
            self.assertEqual(self.indicator.market_environment, env)
        
        # 测试无效环境
        with self.assertRaises(ValueError):
            self.indicator.set_market_environment('invalid_environment')
    
    def test_enhanced_trix_adaptive_period(self):
        """测试EnhancedTRIX自适应周期"""
        # 测试自适应模式
        adaptive_indicator = EnhancedTRIX(n=12, m=9, adaptive_period=True)
        result = adaptive_indicator.calculate(self.data)
        
        # 验证自适应周期功能
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('TRIX', result.columns)
        
        # 测试非自适应模式
        non_adaptive_indicator = EnhancedTRIX(n=12, m=9, adaptive_period=False)
        result2 = non_adaptive_indicator.calculate(self.data)
        
        self.assertIsInstance(result2, pd.DataFrame)
        self.assertIn('TRIX', result2.columns)
    
    def test_enhanced_trix_multi_periods(self):
        """测试EnhancedTRIX多周期计算"""
        # 使用多周期参数
        multi_indicator = EnhancedTRIX(
            n=12, m=9, secondary_n=24,
            multi_periods=[6, 12, 24, 48]
        )
        result = multi_indicator.calculate(self.data)
        
        # 验证多周期TRIX列存在
        expected_multi_columns = [
            'trix_6', 'matrix_6', 'trix_48', 'matrix_48'
        ]
        
        for col in expected_multi_columns:
            self.assertIn(col, result.columns, f"缺少多周期列: {col}")
    
    def test_enhanced_trix_slope_calculation(self):
        """测试EnhancedTRIX斜率计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证斜率列存在
        self.assertIn('trix_slope', result.columns)
        
        # 验证斜率值的合理性
        slope_values = result['trix_slope'].dropna()
        if len(slope_values) > 0:
            # 斜率应该是有限数值
            self.assertTrue(all(np.isfinite(v) for v in slope_values), 
                           "TRIX斜率应该是有限数值")
    
    def test_enhanced_trix_momentum_calculation(self):
        """测试EnhancedTRIX动量计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证动量列存在
        self.assertIn('trix_momentum', result.columns)
        
        # 验证动量值的合理性
        momentum_values = result['trix_momentum'].dropna()
        if len(momentum_values) > 0:
            # 动量应该是有限数值
            self.assertTrue(all(np.isfinite(v) for v in momentum_values), 
                           "TRIX动量应该是有限数值")
    
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
    
    def test_enhanced_trix_register_patterns(self):
        """测试EnhancedTRIX形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_enhanced_trix_edge_cases(self):
        """测试EnhancedTRIX边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(10)
        result = self.indicator.calculate(small_data)
        
        # EnhancedTRIX应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('TRIX', result.columns)
    
    def test_enhanced_trix_validation(self):
        """测试EnhancedTRIX数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['close'], axis=1)
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(invalid_data)
    
    def test_enhanced_trix_indicator_type(self):
        """测试EnhancedTRIX指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "ENHANCEDTRIX", "指标类型应该是ENHANCEDTRIX")


if __name__ == '__main__':
    unittest.main()
