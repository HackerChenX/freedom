"""
EnhancedKDJ指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.oscillator.enhanced_kdj import EnhancedKDJ
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestEnhancedKDJ(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """EnhancedKDJ指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = EnhancedKDJ(n=9, m1=3, m2=3, multi_periods=[5, 9, 14])
        self.expected_columns = [
            'K', 'D', 'J', 'K_5', 'D_5', 'J_5', 'rsv_5',
            'K_14', 'D_14', 'J_14', 'rsv_14', 'j_acceleration',
            'kd_cross_angle', 'kd_distance', 'j_normalized'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_enhanced_kdj_calculation_accuracy(self):
        """测试EnhancedKDJ计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证EnhancedKDJ列存在
        for col in self.expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证KDJ值的合理性
        k_values = result['K'].dropna()
        d_values = result['D'].dropna()
        j_values = result['J'].dropna()
        
        if len(k_values) > 0:
            # KDJ值应该在合理范围内
            self.assertTrue(all(0 <= v <= 100 for v in k_values), "K值应该在0-100范围内")
            self.assertTrue(all(0 <= v <= 100 for v in d_values), "D值应该在0-100范围内")
            # J值可能超出0-100范围
            self.assertTrue(all(np.isfinite(v) for v in j_values), "J值应该是有限数值")
    
    def test_enhanced_kdj_manual_calculation(self):
        """测试EnhancedKDJ手动计算验证"""
        # 创建简单的测试数据
        simple_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104] * 10,
            'high': [101, 102, 103, 104, 105] * 10,
            'low': [99, 100, 101, 102, 103] * 10,
            'close': [100.5, 101.5, 102.5, 103.5, 104.5] * 10,
            'volume': [1000, 1200, 800, 1500, 900] * 10
        })
        
        # 使用较小的周期便于验证
        test_indicator = EnhancedKDJ(n=5, m1=3, m2=3, multi_periods=[5])
        result = test_indicator.calculate(simple_data)
        
        # 验证KDJ计算逻辑
        if len(result) >= 5:
            k_value = result['K'].iloc[4]
            d_value = result['D'].iloc[4]
            j_value = result['J'].iloc[4]
            
            if not pd.isna(k_value) and not pd.isna(d_value) and not pd.isna(j_value):
                # 验证J = 3K - 2D
                expected_j = 3 * k_value - 2 * d_value
                self.assertAlmostEqual(j_value, expected_j, places=6,
                                     msg="J值应该等于3K-2D")
    
    def test_enhanced_kdj_score_range(self):
        """测试EnhancedKDJ评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_kdj_confidence_calculation(self):
        """测试EnhancedKDJ置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_enhanced_kdj_parameter_update(self):
        """测试EnhancedKDJ参数更新"""
        new_n = 6
        new_m1 = 2
        new_m2 = 2
        self.indicator.set_parameters(n=new_n, m1=new_m1, m2=new_m2)
        
        # 验证参数更新
        self.assertEqual(self.indicator.n, new_n)
        self.assertEqual(self.indicator.m1, new_m1)
        self.assertEqual(self.indicator.m2, new_m2)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('K', result.columns)
        self.assertIn('D', result.columns)
        self.assertIn('J', result.columns)
    
    def test_enhanced_kdj_required_columns(self):
        """测试EnhancedKDJ必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['high', 'low', 'close']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_enhanced_kdj_comprehensive_score(self):
        """测试EnhancedKDJ综合评分"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 然后计算评分
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        self.assertIsInstance(raw_score, pd.Series)
        
        # 验证评分范围
        valid_scores = raw_score.dropna()
        if len(valid_scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_kdj_patterns(self):
        """测试EnhancedKDJ形态识别"""
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
                'KDJ_GOLDEN_CROSS', 'KDJ_DEATH_CROSS',
                'KDJ_OVERSOLD', 'KDJ_OVERBOUGHT',
                'KDJ_J_OVERSOLD', 'KDJ_J_OVERBOUGHT'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_enhanced_kdj_multi_period_calculation(self):
        """测试EnhancedKDJ多周期计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证多周期KDJ列存在
        expected_multi_columns = ['K_5', 'D_5', 'J_5', 'K_14', 'D_14', 'J_14']
        
        for col in expected_multi_columns:
            self.assertIn(col, result.columns, f"缺少多周期列: {col}")
    
    def test_enhanced_kdj_j_acceleration(self):
        """测试EnhancedKDJ J线加速度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证J线加速度列存在
        self.assertIn('j_acceleration', result.columns)
        
        # 验证加速度值的合理性
        j_accel = result['j_acceleration'].dropna()
        if len(j_accel) > 0:
            # 加速度应该是有限数值
            self.assertTrue(all(np.isfinite(v) for v in j_accel), 
                           "J线加速度应该是有限数值")
    
    def test_enhanced_kdj_kd_cross_angle(self):
        """测试EnhancedKDJ KD交叉角度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证KD交叉角度列存在
        self.assertIn('kd_cross_angle', result.columns)
        
        # 验证角度值的合理性
        kd_angle = result['kd_cross_angle'].dropna()
        if len(kd_angle) > 0:
            # 角度应该是有限数值
            self.assertTrue(all(np.isfinite(v) for v in kd_angle), 
                           "KD交叉角度应该是有限数值")
    
    def test_enhanced_kdj_j_normalized(self):
        """测试EnhancedKDJ J线归一化"""
        result = self.indicator.calculate(self.data)
        
        # 验证J线归一化列存在
        self.assertIn('j_normalized', result.columns)
        
        # 验证归一化值的合理性
        j_norm = result['j_normalized'].dropna()
        if len(j_norm) > 0:
            # 归一化值应该在0-100范围内
            self.assertTrue(all(0 <= v <= 100 for v in j_norm), 
                           "J线归一化值应该在0-100范围内")
    
    def test_enhanced_kdj_sensitivity_adjustment(self):
        """测试EnhancedKDJ灵敏度调整"""
        # 测试不同灵敏度设置
        sensitivities = [0.5, 1.0, 1.5, 2.0]
        
        for sensitivity in sensitivities:
            test_indicator = EnhancedKDJ(n=9, m1=3, m2=3, sensitivity=sensitivity)
            result = test_indicator.calculate(self.data)
            
            # 验证灵敏度调整功能
            self.assertIsInstance(result, pd.DataFrame)
            self.assertIn('K', result.columns)
            self.assertIn('D', result.columns)
            self.assertIn('J', result.columns)
    
    def test_enhanced_kdj_signals(self):
        """测试EnhancedKDJ信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_enhanced_kdj_pattern_identification(self):
        """测试EnhancedKDJ形态识别方法"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 识别形态
        patterns = self.indicator.identify_patterns(self.data)
        
        # 验证形态识别结果
        self.assertIsInstance(patterns, list)
        
        # 验证形态类型
        valid_patterns = [
            "KD超卖区域", "KD超买区域", "J线超卖区域", "J线超买区域",
            "KD金叉", "KD死叉", "KD高质量金叉", "KD高质量死叉",
            "KDJ三重底", "KDJ三重顶", "KDJ正背离", "KDJ负背离"
        ]
        
        for pattern in patterns:
            self.assertIn(pattern, valid_patterns, f"无效的形态类型: {pattern}")
    
    def test_enhanced_kdj_multi_period_consistency(self):
        """测试EnhancedKDJ多周期一致性"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 计算多周期一致性
        consistency = self.indicator._calculate_multi_period_consistency()
        
        # 验证一致性计算结果
        self.assertIsInstance(consistency, pd.Series)
        
        # 验证一致性值的合理性
        consistency_values = consistency.dropna()
        if len(consistency_values) > 0:
            # 一致性分数应该在合理范围内
            self.assertTrue(all(-20 <= v <= 20 for v in consistency_values), 
                           "多周期一致性分数应该在-20到20范围内")
    
    def test_enhanced_kdj_generate_signals_method(self):
        """测试EnhancedKDJ信号生成方法"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 生成信号
        signals = self.indicator.generate_signals(self.data)
        
        # 验证信号生成结果
        self.assertIsInstance(signals, pd.DataFrame)
        
        expected_signal_columns = [
            'K', 'D', 'J', 'score', 'buy_signal', 'sell_signal',
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
    
    def test_enhanced_kdj_register_patterns(self):
        """测试EnhancedKDJ形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_enhanced_kdj_edge_cases(self):
        """测试EnhancedKDJ边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(5)
        result = self.indicator.calculate(small_data)
        
        # EnhancedKDJ应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('K', result.columns)
        self.assertIn('D', result.columns)
        self.assertIn('J', result.columns)
    
    def test_enhanced_kdj_validation(self):
        """测试EnhancedKDJ数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['high', 'low'], axis=1)
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(invalid_data)
    
    def test_enhanced_kdj_indicator_type(self):
        """测试EnhancedKDJ指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "ENHANCEDKDJ", "指标类型应该是ENHANCEDKDJ")


if __name__ == '__main__':
    unittest.main()
