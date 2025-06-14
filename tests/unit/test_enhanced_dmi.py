"""
EnhancedDMI指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.trend.enhanced_dmi import EnhancedDMI
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestEnhancedDMI(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """EnhancedDMI指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = EnhancedDMI(period=14, adx_period=14, adaptive=True)
        self.expected_columns = ['plus_di', 'minus_di', 'adx', 'adxr', 'dx', 'tr']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_enhanced_dmi_calculation_accuracy(self):
        """测试EnhancedDMI计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证EnhancedDMI列存在
        for col in self.expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证DMI值的合理性
        plus_di_values = result['plus_di'].dropna()
        minus_di_values = result['minus_di'].dropna()
        adx_values = result['adx'].dropna()
        
        if len(plus_di_values) > 0:
            # DI值应该在0-100范围内
            self.assertTrue(all(0 <= v <= 100 for v in plus_di_values), "+DI值应该在0-100范围内")
            self.assertTrue(all(0 <= v <= 100 for v in minus_di_values), "-DI值应该在0-100范围内")
            self.assertTrue(all(0 <= v <= 100 for v in adx_values), "ADX值应该在0-100范围内")
    
    def test_enhanced_dmi_manual_calculation(self):
        """测试EnhancedDMI手动计算验证"""
        # 创建简单的测试数据
        simple_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104] * 10,
            'high': [101, 102, 103, 104, 105] * 10,
            'low': [99, 100, 101, 102, 103] * 10,
            'close': [100.5, 101.5, 102.5, 103.5, 104.5] * 10,
            'volume': [1000, 1200, 800, 1500, 900] * 10
        })
        
        # 使用较小的周期便于验证
        test_indicator = EnhancedDMI(period=10, adx_period=10, adaptive=False)
        result = test_indicator.calculate(simple_data)
        
        # 验证DMI计算逻辑
        if len(result) >= 10:
            plus_di_value = result['plus_di'].iloc[9]
            minus_di_value = result['minus_di'].iloc[9]
            adx_value = result['adx'].iloc[9]
            
            if not pd.isna(plus_di_value) and not pd.isna(minus_di_value) and not pd.isna(adx_value):
                # DMI值应该是合理的数值
                self.assertTrue(0 <= plus_di_value <= 100, "+DI值应该在0-100范围内")
                self.assertTrue(0 <= minus_di_value <= 100, "-DI值应该在0-100范围内")
                self.assertTrue(0 <= adx_value <= 100, "ADX值应该在0-100范围内")
    
    def test_enhanced_dmi_score_range(self):
        """测试EnhancedDMI评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_dmi_confidence_calculation(self):
        """测试EnhancedDMI置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_enhanced_dmi_parameter_update(self):
        """测试EnhancedDMI参数更新"""
        new_period = 10
        new_adx_period = 10
        self.indicator.set_parameters(period=new_period, adx_period=new_adx_period)
        
        # 验证参数更新
        self.assertEqual(self.indicator.base_period, new_period)
        self.assertEqual(self.indicator.adx_period, new_adx_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('plus_di', result.columns)
        self.assertIn('minus_di', result.columns)
        self.assertIn('adx', result.columns)
    
    def test_enhanced_dmi_required_columns(self):
        """测试EnhancedDMI必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_enhanced_dmi_comprehensive_score(self):
        """测试EnhancedDMI综合评分"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 然后计算评分
        score = self.indicator.calculate_score()
        
        self.assertIsInstance(score, pd.Series)
        
        # 验证评分范围
        valid_scores = score.dropna()
        if len(valid_scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_dmi_patterns(self):
        """测试EnhancedDMI形态识别"""
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
                'trend_start', 'trend_acceleration', 'trend_exhaustion',
                'trend_reversal_warning', 'no_trend_zone', 'strong_trend'
            ]

            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_enhanced_dmi_crossover_quality(self):
        """测试EnhancedDMI交叉质量评估"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 评估交叉质量
        crossover_quality = self.indicator.evaluate_di_crossover_quality()
        
        # 验证交叉质量评估结果
        self.assertIsInstance(crossover_quality, pd.Series)
        
        if not crossover_quality.empty:
            # 交叉质量分数应该在合理范围内
            quality_values = crossover_quality.dropna()
            if len(quality_values) > 0:
                self.assertTrue(all(-100 <= v <= 100 for v in quality_values), 
                               "交叉质量分数应该在合理范围内")
    
    def test_enhanced_dmi_three_line_synergy(self):
        """测试EnhancedDMI三线协同分析"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 分析三线协同
        synergy = self.indicator.analyze_three_line_synergy()
        
        # 验证协同分析结果
        self.assertIsInstance(synergy, pd.DataFrame)
        
        if not synergy.empty:
            expected_synergy_columns = [
                'strong_uptrend', 'strong_downtrend',
                'weakening_trend', 'potential_reversal',
                'no_trend', 'emerging_trend'
            ]
            
            for col in expected_synergy_columns:
                self.assertIn(col, synergy.columns, f"缺少协同分析列: {col}")
    
    def test_enhanced_dmi_pattern_identification(self):
        """测试EnhancedDMI形态识别"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 识别形态
        patterns = self.indicator.identify_patterns()
        
        # 验证形态识别结果
        self.assertIsInstance(patterns, pd.DataFrame)
        
        if not patterns.empty:
            expected_pattern_columns = [
                'trend_start', 'trend_acceleration', 'trend_exhaustion',
                'trend_reversal_warning', 'no_trend_zone', 'strong_trend'
            ]
            
            for col in expected_pattern_columns:
                self.assertIn(col, patterns.columns, f"缺少形态识别列: {col}")
    
    def test_enhanced_dmi_signals(self):
        """测试EnhancedDMI信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_enhanced_dmi_market_environment(self):
        """测试EnhancedDMI市场环境设置"""
        # 测试设置不同的市场环境
        environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        
        for env in environments:
            self.indicator.set_market_environment(env)
            self.assertEqual(self.indicator.market_environment, env)
        
        # 测试无效环境
        with self.assertRaises(ValueError):
            self.indicator.set_market_environment('invalid_environment')
    
    def test_enhanced_dmi_adaptive_period(self):
        """测试EnhancedDMI自适应周期"""
        # 测试自适应模式
        adaptive_indicator = EnhancedDMI(period=14, adaptive=True)
        result = adaptive_indicator.calculate(self.data)
        
        # 验证自适应周期功能
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('plus_di', result.columns)
        
        # 测试非自适应模式
        non_adaptive_indicator = EnhancedDMI(period=14, adaptive=False)
        result2 = non_adaptive_indicator.calculate(self.data)
        
        self.assertIsInstance(result2, pd.DataFrame)
        self.assertIn('plus_di', result2.columns)
    
    def test_enhanced_dmi_adx_strength_classification(self):
        """测试EnhancedDMI ADX强度分类"""
        # 测试不同ADX值的强度分类
        test_values = [10, 25, 35, 45, 55]
        expected_classifications = ["无趋势", "弱趋势", "中等趋势", "强趋势", "极强趋势"]
        
        for value, expected in zip(test_values, expected_classifications):
            classification = self.indicator.classify_adx_strength(value)
            self.assertEqual(classification, expected, f"ADX值{value}的分类不正确")
    
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
    
    def test_enhanced_dmi_register_patterns(self):
        """测试EnhancedDMI形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_enhanced_dmi_edge_cases(self):
        """测试EnhancedDMI边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(10)
        result = self.indicator.calculate(small_data)
        
        # EnhancedDMI应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_enhanced_dmi_validation(self):
        """测试EnhancedDMI数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['high', 'low'], axis=1)
        
        # 由于EnhancedDMI内部没有显式验证，这里只验证计算不会崩溃
        try:
            result = self.indicator.calculate(invalid_data)
            # 如果没有抛出异常，验证结果是否为空或合理
            self.assertIsInstance(result, pd.DataFrame)
        except (KeyError, ValueError):
            # 如果抛出异常，这是预期的行为
            pass
    
    def test_enhanced_dmi_smoothed_values(self):
        """测试EnhancedDMI平滑值计算"""
        # 创建简单的测试序列
        test_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # 测试平滑值计算
        smoothed = self.indicator._calculate_smoothed_values(test_series, 5)
        
        # 验证平滑值计算结果
        self.assertIsInstance(smoothed, pd.Series)
        self.assertEqual(len(smoothed), len(test_series))
        
        # 验证第一个平滑值
        first_smoothed = smoothed.iloc[4]  # 第5个值（索引4）
        if not pd.isna(first_smoothed):
            expected_first = test_series.iloc[:5].mean()
            self.assertAlmostEqual(first_smoothed, expected_first, places=6)


if __name__ == '__main__':
    unittest.main()
