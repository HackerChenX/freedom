"""
CMO指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.cmo import CMO
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestCMO(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """CMO指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = CMO(period=14, oversold=-40, overbought=40)
        self.expected_columns = ['cmo']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_cmo_calculation_accuracy(self):
        """测试CMO计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 手动计算CMO验证
        price_change = self.data['close'].diff()
        up = price_change.apply(lambda x: x if x > 0 else 0)
        down = price_change.apply(lambda x: abs(x) if x < 0 else 0)
        
        up_sum = up.rolling(window=14).sum()
        down_sum = down.rolling(window=14).sum()
        
        up_down_sum = up_sum + down_sum
        expected_cmo = np.where(
            up_down_sum > 0,
            100 * ((up_sum - down_sum) / up_down_sum),
            0
        )
        
        # 比较计算结果（允许小的数值误差）
        calculated_cmo = result['cmo']
        diff = abs(calculated_cmo - expected_cmo).dropna()
        self.assertTrue(all(d < 0.001 for d in diff), "CMO计算结果不正确")
    
    def test_cmo_score_range(self):
        """测试CMO评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_cmo_confidence_calculation(self):
        """测试CMO置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_cmo_parameter_update(self):
        """测试CMO参数更新"""
        new_period = 20
        new_oversold = -50
        new_overbought = 50
        self.indicator.set_parameters(period=new_period, oversold=new_oversold, overbought=new_overbought)
        
        # 验证参数更新
        self.assertEqual(self.indicator.period, new_period)
        self.assertEqual(self.indicator.oversold, new_oversold)
        self.assertEqual(self.indicator.overbought, new_overbought)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('cmo', result.columns)
    
    def test_cmo_required_columns(self):
        """测试CMO必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        self.assertIn('close', self.indicator.REQUIRED_COLUMNS)
    
    def test_cmo_comprehensive_score(self):
        """测试CMO综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_cmo_patterns(self):
        """测试CMO形态识别"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证预期的形态列存在
        expected_patterns = [
            'CMO_OVERSOLD', 'CMO_OVERBOUGHT',
            'CMO_CROSS_UP_ZERO', 'CMO_CROSS_DOWN_ZERO',
            'CMO_ABOVE_ZERO', 'CMO_BELOW_ZERO'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_cmo_value_range(self):
        """测试CMO数值范围"""
        result = self.indicator.calculate(self.data)
        
        # CMO值应该在-100到+100范围内
        cmo_values = result['cmo'].dropna()
        self.assertTrue(all(-100 <= v <= 100 for v in cmo_values), "CMO值应在-100到+100范围内")
    
    def test_cmo_signals(self):
        """测试CMO信号生成"""
        # 生成信号
        signals = self.indicator.generate_signals(self.data)
        
        # 验证信号
        self.assertIsInstance(signals, list)
        
        if len(signals) > 0:
            signal = signals[0]
            self.assertIsInstance(signal, dict)
            
            # 验证必需的信号字段
            required_fields = ['indicator', 'buy_signal', 'sell_signal', 'score', 'confidence']
            for field in required_fields:
                self.assertIn(field, signal, f"缺少信号字段: {field}")
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('cmo', result.columns)
    
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
