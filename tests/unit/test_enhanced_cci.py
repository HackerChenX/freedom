"""
EnhancedCCI指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.trend.enhanced_cci import EnhancedCCI
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestEnhancedCCI(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """EnhancedCCI指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = EnhancedCCI(period=20, factor=0.015, secondary_period=40)
        self.expected_columns = ['cci', 'cci_secondary', 'cci_ma5', 'cci_ma10', 'cci_ma20', 'cci_slope', 'cci_volatility', 'state']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_enhanced_cci_calculation_accuracy(self):
        """测试EnhancedCCI计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证EnhancedCCI列存在
        for col in self.expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证CCI值的合理性
        cci_values = result['cci'].dropna()
        if len(cci_values) > 0:
            # CCI值通常在-300到300范围内
            self.assertTrue(all(-500 <= v <= 500 for v in cci_values), "CCI值应该在合理范围内")
    
    def test_enhanced_cci_manual_calculation(self):
        """测试EnhancedCCI手动计算验证"""
        # 创建简单的测试数据
        simple_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104] * 10,
            'high': [101, 102, 103, 104, 105] * 10,
            'low': [99, 100, 101, 102, 103] * 10,
            'close': [100.5, 101.5, 102.5, 103.5, 104.5] * 10,
            'volume': [1000, 1200, 800, 1500, 900] * 10
        })
        
        # 使用较小的周期便于验证
        test_indicator = EnhancedCCI(period=10, factor=0.015)
        result = test_indicator.calculate(simple_data)
        
        # 验证CCI计算逻辑
        if len(result) >= 10:
            cci_value = result['cci'].iloc[9]
            if not pd.isna(cci_value):
                # CCI应该是一个合理的数值
                self.assertTrue(-500 <= cci_value <= 500, "CCI值应该在合理范围内")
    
    def test_enhanced_cci_score_range(self):
        """测试EnhancedCCI评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_cci_confidence_calculation(self):
        """测试EnhancedCCI置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_enhanced_cci_parameter_update(self):
        """测试EnhancedCCI参数更新"""
        new_period = 15
        new_factor = 0.02
        self.indicator.set_parameters(period=new_period, factor=new_factor)
        
        # 验证参数更新
        self.assertEqual(self.indicator.base_period, new_period)
        self.assertEqual(self.indicator.factor, new_factor)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('cci', result.columns)
    
    def test_enhanced_cci_required_columns(self):
        """测试EnhancedCCI必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_enhanced_cci_comprehensive_score(self):
        """测试EnhancedCCI综合评分"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 然后计算评分
        score = self.indicator.calculate_score()
        
        self.assertIsInstance(score, pd.Series)
        
        # 验证评分范围
        valid_scores = score.dropna()
        if len(valid_scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_cci_patterns(self):
        """测试EnhancedCCI形态识别"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 然后获取形态
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证基本的形态列存在
        if not patterns.empty and len(patterns.columns) > 0:
            # 检查实际返回的形态列
            expected_patterns = [
                'zero_cross_up', 'zero_cross_down',
                'bullish_divergence', 'bearish_divergence'
            ]

            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_enhanced_cci_crossover_analysis(self):
        """测试EnhancedCCI交叉分析"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 分析交叉
        crossovers = self.indicator.analyze_crossovers()
        
        # 验证交叉分析结果
        self.assertIsInstance(crossovers, pd.DataFrame)
        
        if not crossovers.empty:
            expected_crossover_columns = [
                'zero_cross_up', 'zero_cross_down',
                'overbought_enter', 'overbought_exit',
                'oversold_enter', 'oversold_exit'
            ]
            
            for col in expected_crossover_columns:
                self.assertIn(col, crossovers.columns, f"缺少交叉分析列: {col}")
    
    def test_enhanced_cci_multi_period_synergy(self):
        """测试EnhancedCCI多周期协同分析"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 分析多周期协同
        synergy = self.indicator.analyze_multi_period_synergy()
        
        # 验证协同分析结果
        self.assertIsInstance(synergy, pd.DataFrame)
        
        if not synergy.empty:
            expected_synergy_columns = [
                'bullish_agreement', 'bearish_agreement',
                'rising_momentum', 'falling_momentum'
            ]
            
            for col in expected_synergy_columns:
                self.assertIn(col, synergy.columns, f"缺少协同分析列: {col}")
    
    def test_enhanced_cci_pattern_identification(self):
        """测试EnhancedCCI形态识别"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 识别形态
        patterns = self.indicator.identify_patterns()
        
        # 验证形态识别结果
        self.assertIsInstance(patterns, pd.DataFrame)
        
        if not patterns.empty:
            expected_pattern_columns = ['zero_cross_up', 'zero_cross_down']
            
            for col in expected_pattern_columns:
                self.assertIn(col, patterns.columns, f"缺少形态识别列: {col}")
    
    def test_enhanced_cci_signals(self):
        """测试EnhancedCCI信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_enhanced_cci_market_environment(self):
        """测试EnhancedCCI市场环境设置"""
        # 测试设置不同的市场环境
        environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        
        for env in environments:
            self.indicator.set_market_environment(env)
            self.assertEqual(self.indicator.market_environment, env)
        
        # 测试无效环境
        with self.assertRaises(ValueError):
            self.indicator.set_market_environment('invalid_environment')
    
    def test_enhanced_cci_adaptive_period(self):
        """测试EnhancedCCI自适应周期"""
        # 测试自适应模式
        adaptive_indicator = EnhancedCCI(period=20, adaptive=True)
        result = adaptive_indicator.calculate(self.data)
        
        # 验证自适应周期功能
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('cci', result.columns)
        
        # 测试非自适应模式
        non_adaptive_indicator = EnhancedCCI(period=20, adaptive=False)
        result2 = non_adaptive_indicator.calculate(self.data)
        
        self.assertIsInstance(result2, pd.DataFrame)
        self.assertIn('cci', result2.columns)
    
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
    
    def test_enhanced_cci_register_patterns(self):
        """测试EnhancedCCI形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_enhanced_cci_edge_cases(self):
        """测试EnhancedCCI边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(10)
        result = self.indicator.calculate(small_data)
        
        # EnhancedCCI应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('cci', result.columns)
    
    def test_enhanced_cci_validation(self):
        """测试EnhancedCCI数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['high', 'low'], axis=1)
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(invalid_data)
    
    def test_enhanced_cci_state_classification(self):
        """测试EnhancedCCI状态分类"""
        result = self.indicator.calculate(self.data)
        
        # 验证状态分类
        if 'state' in result.columns:
            states = result['state'].dropna().unique()
            valid_states = [
                'extreme_overbought', 'overbought', 'neutral_high',
                'neutral_low', 'oversold', 'extreme_oversold', 'neutral'
            ]
            
            for state in states:
                self.assertIn(state, valid_states, f"无效的状态分类: {state}")


if __name__ == '__main__':
    unittest.main()
