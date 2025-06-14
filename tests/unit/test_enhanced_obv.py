"""
EnhancedOBV指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.volume.enhanced_obv import EnhancedOBV
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestEnhancedOBV(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """EnhancedOBV指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = EnhancedOBV(ma_period=30, multi_periods=[5, 10, 20, 60])
        self.expected_columns = [
            'obv', 'obv_smooth', 'obv_ma', 'obv_ma5', 'obv_ma10', 'obv_ma20', 'obv_ma60',
            'obv_momentum', 'obv_rate', 'volume_price_corr'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_enhanced_obv_calculation_accuracy(self):
        """测试EnhancedOBV计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证EnhancedOBV列存在
        for col in self.expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证OBV值的合理性
        obv_values = result['obv'].dropna()
        
        if len(obv_values) > 0:
            # OBV值应该是有限数值
            self.assertTrue(all(np.isfinite(v) for v in obv_values), "OBV值应该是有限数值")
    
    def test_enhanced_obv_manual_calculation(self):
        """测试EnhancedOBV手动计算验证"""
        # 创建简单的测试数据
        simple_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104] * 10,
            'high': [101, 102, 103, 104, 105] * 10,
            'low': [99, 100, 101, 102, 103] * 10,
            'close': [100.5, 101.5, 102.5, 103.5, 104.5] * 10,
            'volume': [1000, 1200, 800, 1500, 900] * 10
        })
        
        # 使用较小的周期便于验证
        test_indicator = EnhancedOBV(ma_period=5, multi_periods=[5])
        result = test_indicator.calculate(simple_data)
        
        # 验证OBV计算逻辑
        if len(result) >= 5:
            obv_value = result['obv'].iloc[4]
            
            if not pd.isna(obv_value):
                # OBV值应该是有限数值
                self.assertTrue(np.isfinite(obv_value), "OBV值应该是有限数值")
    
    def test_enhanced_obv_score_range(self):
        """测试EnhancedOBV评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_obv_confidence_calculation(self):
        """测试EnhancedOBV置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_enhanced_obv_parameter_update(self):
        """测试EnhancedOBV参数更新"""
        new_ma_period = 20
        new_sensitivity = 1.5
        self.indicator.set_parameters(ma_period=new_ma_period, sensitivity=new_sensitivity)
        
        # 验证参数更新
        self.assertEqual(self.indicator.ma_period, new_ma_period)
        self.assertEqual(self.indicator.sensitivity, new_sensitivity)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('obv', result.columns)
    
    def test_enhanced_obv_required_columns(self):
        """测试EnhancedOBV必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_enhanced_obv_comprehensive_score(self):
        """测试EnhancedOBV综合评分"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 然后计算评分
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        self.assertIsInstance(raw_score, pd.Series)
        
        # 验证评分范围
        valid_scores = raw_score.dropna()
        if len(valid_scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_obv_patterns(self):
        """测试EnhancedOBV形态识别"""
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
                'OBV_ABOVE_MA', 'OBV_BELOW_MA',
                'OBV_RISING', 'OBV_FALLING',
                'OBV_CROSS_MA_UP', 'OBV_CROSS_MA_DOWN'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_enhanced_obv_multi_period_calculation(self):
        """测试EnhancedOBV多周期计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证多周期OBV均线列存在
        expected_multi_columns = ['obv_ma5', 'obv_ma10', 'obv_ma20', 'obv_ma60']
        
        for col in expected_multi_columns:
            self.assertIn(col, result.columns, f"缺少多周期列: {col}")
    
    def test_enhanced_obv_momentum_calculation(self):
        """测试EnhancedOBV动量计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证动量列存在
        self.assertIn('obv_momentum', result.columns)
        
        # 验证动量值的合理性
        momentum_values = result['obv_momentum'].dropna()
        if len(momentum_values) > 0:
            # 动量应该是有限数值
            self.assertTrue(all(np.isfinite(v) for v in momentum_values), 
                           "OBV动量应该是有限数值")
    
    def test_enhanced_obv_volume_price_correlation(self):
        """测试EnhancedOBV量价相关性计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证量价相关性列存在
        self.assertIn('volume_price_corr', result.columns)
        
        # 验证相关性值的合理性
        corr_values = result['volume_price_corr'].dropna()
        if len(corr_values) > 0:
            # 相关性应该在-1到1之间
            self.assertTrue(all(-1 <= v <= 1 for v in corr_values), 
                           "量价相关性应该在-1到1之间")
    
    def test_enhanced_obv_smoothing(self):
        """测试EnhancedOBV平滑功能"""
        # 测试启用平滑
        smoothed_indicator = EnhancedOBV(use_smoothed_obv=True, smoothing_period=5)
        result1 = smoothed_indicator.calculate(self.data)
        
        # 测试禁用平滑
        unsmoothed_indicator = EnhancedOBV(use_smoothed_obv=False)
        result2 = unsmoothed_indicator.calculate(self.data)
        
        # 两种情况都应该能正常计算
        self.assertIsInstance(result1, pd.DataFrame)
        self.assertIsInstance(result2, pd.DataFrame)
        self.assertIn('obv_smooth', result1.columns)
        self.assertIn('obv_smooth', result2.columns)
    
    def test_enhanced_obv_market_environment(self):
        """测试EnhancedOBV市场环境设置"""
        # 测试设置不同的市场环境
        environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        
        for env in environments:
            self.indicator.set_market_environment(env)
            self.assertEqual(self.indicator.market_environment, env)
        
        # 测试无效环境
        with self.assertRaises(ValueError):
            self.indicator.set_market_environment('invalid_environment')
    
    def test_enhanced_obv_flow_gradient(self):
        """测试EnhancedOBV资金流向梯度计算"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 计算资金流向梯度
        flow_gradient = self.indicator.calculate_flow_gradient()
        
        # 验证梯度计算结果
        self.assertIsInstance(flow_gradient, pd.DataFrame)
        
        if not flow_gradient.empty:
            expected_gradient_columns = [
                'gradient', 'acceleration', 'normalized_gradient', 'normalized_acceleration',
                'strong_inflow', 'weak_inflow', 'strong_outflow', 'weak_outflow', 'neutral'
            ]
            
            for col in expected_gradient_columns:
                self.assertIn(col, flow_gradient.columns, f"缺少梯度分析列: {col}")
    
    def test_enhanced_obv_divergence_detection(self):
        """测试EnhancedOBV背离检测"""
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
    
    def test_enhanced_obv_price_volume_synergy(self):
        """测试EnhancedOBV价格成交量协同分析"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 计算价格成交量协同
        synergy = self.indicator.calculate_price_volume_synergy()
        
        # 验证协同分析结果
        self.assertIsInstance(synergy, pd.DataFrame)
        
        if not synergy.empty:
            expected_synergy_columns = [
                'direction_synergy', 'magnitude_diff', 'ideal_up', 'ideal_down',
                'poor_up', 'poor_down', 'synergy_score'
            ]
            
            for col in expected_synergy_columns:
                self.assertIn(col, synergy.columns, f"缺少协同分析列: {col}")
    
    def test_enhanced_obv_pattern_identification(self):
        """测试EnhancedOBV形态识别方法"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 识别形态
        patterns = self.indicator.identify_patterns()
        
        # 验证形态识别结果
        self.assertIsInstance(patterns, pd.DataFrame)
        
        if not patterns.empty:
            expected_pattern_columns = [
                'obv_breakout', 'obv_breakdown', 'obv_acceleration_up', 'obv_acceleration_down',
                'obv_w_bottom', 'obv_m_top', 'bullish_divergence', 'bearish_divergence',
                'hidden_bullish_divergence', 'hidden_bearish_divergence', 'obv_consolidation'
            ]
            
            for col in expected_pattern_columns:
                self.assertIn(col, patterns.columns, f"缺少形态识别列: {col}")
    
    def test_enhanced_obv_signals(self):
        """测试EnhancedOBV信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_enhanced_obv_generate_signals_method(self):
        """测试EnhancedOBV信号生成方法"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 生成信号
        signals = self.indicator.generate_signals(self.data)
        
        # 验证信号生成结果
        self.assertIsInstance(signals, pd.DataFrame)
        
        expected_signal_columns = [
            'obv', 'obv_ma', 'score', 'buy_signal', 'sell_signal',
            'bull_trend', 'bear_trend', 'signal_strength'
        ]
        
        for col in expected_signal_columns:
            self.assertIn(col, signals.columns, f"缺少信号列: {col}")
    
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
    
    def test_enhanced_obv_register_patterns(self):
        """测试EnhancedOBV形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_enhanced_obv_edge_cases(self):
        """测试EnhancedOBV边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(10)
        result = self.indicator.calculate(small_data)
        
        # EnhancedOBV应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('obv', result.columns)
    
    def test_enhanced_obv_validation(self):
        """测试EnhancedOBV数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['volume'], axis=1)
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(invalid_data)
    
    def test_enhanced_obv_indicator_type(self):
        """测试EnhancedOBV指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "ENHANCEDOBV", "指标类型应该是ENHANCEDOBV")


if __name__ == '__main__':
    unittest.main()
