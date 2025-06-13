"""
KC指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.kc import KC
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestKC(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """KC指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = KC(period=20, atr_period=10, multiplier=2.0)
        self.expected_columns = ['kc_middle', 'kc_upper', 'kc_lower', 'kc_position', 'kc_width', 'kc_width_chg']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 60}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_kc_calculation_accuracy(self):
        """测试KC计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证KC数值合理性
        middle_values = result['kc_middle'].dropna()
        upper_values = result['kc_upper'].dropna()
        lower_values = result['kc_lower'].dropna()
        
        self.assertTrue(len(middle_values) > 0, "KC中轨值全为NaN")
        self.assertTrue(len(upper_values) > 0, "KC上轨值全为NaN")
        self.assertTrue(len(lower_values) > 0, "KC下轨值全为NaN")
        
        # 验证上轨 > 中轨 > 下轨
        for i in range(len(result)):
            if not pd.isna(result['kc_upper'].iloc[i]):
                self.assertGreater(result['kc_upper'].iloc[i], result['kc_middle'].iloc[i])
                self.assertGreater(result['kc_middle'].iloc[i], result['kc_lower'].iloc[i])
    
    def test_kc_score_range(self):
        """测试KC评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_kc_confidence_calculation(self):
        """测试KC置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_kc_parameter_update(self):
        """测试KC参数更新"""
        new_period = 14
        new_atr_period = 14
        new_multiplier = 1.5
        
        self.indicator.set_parameters(
            period=new_period,
            atr_period=new_atr_period,
            multiplier=new_multiplier
        )
        
        # 验证参数更新
        self.assertEqual(self.indicator.period, new_period)
        self.assertEqual(self.indicator.atr_period, new_atr_period)
        self.assertEqual(self.indicator.multiplier, new_multiplier)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('kc_middle', result.columns)
        self.assertIn('kc_upper', result.columns)
        self.assertIn('kc_lower', result.columns)
    
    def test_kc_required_columns(self):
        """测试KC必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_cols = ['high', 'low', 'close']
        for col in expected_cols:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_kc_comprehensive_score(self):
        """测试KC综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_kc_patterns(self):
        """测试KC形态识别"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证预期的形态列存在
        expected_patterns = [
            'KC_ABOVE_UPPER', 'KC_BELOW_LOWER',
            'KC_ABOVE_MIDDLE', 'KC_BELOW_MIDDLE',
            'KC_BREAK_UPPER', 'KC_BREAK_LOWER'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_kc_channel_width(self):
        """测试KC通道宽度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证通道宽度计算
        width_values = result['kc_width'].dropna()
        self.assertTrue(len(width_values) > 0, "KC通道宽度值全为NaN")
        
        # 通道宽度应该都是正数
        self.assertTrue(all(v > 0 for v in width_values), "KC通道宽度应该都是正数")
    
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


if __name__ == '__main__':
    unittest.main()
