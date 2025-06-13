"""
DMI指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.dmi import DMI
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestDMI(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """DMI指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = DMI(period=14, adx_period=14)
        self.expected_columns = ['PDI', 'MDI', 'ADX', 'ADXR']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 60}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_dmi_calculation_accuracy(self):
        """测试DMI计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证DMI数值合理性
        pdi_values = result['PDI'].dropna()
        mdi_values = result['MDI'].dropna()
        adx_values = result['ADX'].dropna()
        
        self.assertTrue(len(pdi_values) > 0, "PDI值全为NaN")
        self.assertTrue(len(mdi_values) > 0, "MDI值全为NaN")
        self.assertTrue(len(adx_values) > 0, "ADX值全为NaN")
        
        # DMI值应该在0-100范围内
        self.assertTrue(all(0 <= v <= 100 for v in pdi_values), "PDI值应在0-100范围内")
        self.assertTrue(all(0 <= v <= 100 for v in mdi_values), "MDI值应在0-100范围内")
        self.assertTrue(all(0 <= v <= 100 for v in adx_values), "ADX值应在0-100范围内")
        
        # PDI和MDI应该都是非负数
        self.assertTrue(all(v >= 0 for v in pdi_values), "PDI应该都是非负数")
        self.assertTrue(all(v >= 0 for v in mdi_values), "MDI应该都是非负数")
    
    def test_dmi_score_range(self):
        """测试DMI评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_dmi_confidence_calculation(self):
        """测试DMI置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_dmi_parameter_update(self):
        """测试DMI参数更新"""
        new_period = 20
        new_adx_period = 20
        self.indicator.set_parameters(period=new_period, adx_period=new_adx_period)
        
        # 验证参数更新
        self.assertEqual(self.indicator.period, new_period)
        self.assertEqual(self.indicator.adx_period, new_adx_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('PDI', result.columns)
        self.assertIn('MDI', result.columns)
        self.assertIn('ADX', result.columns)
    
    def test_dmi_required_columns(self):
        """测试DMI必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_cols = ['high', 'low', 'close']
        for col in expected_cols:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_dmi_comprehensive_score(self):
        """测试DMI综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_dmi_patterns(self):
        """测试DMI形态识别"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证预期的形态列存在
        expected_patterns = [
            'DMI_GOLDEN_CROSS', 'DMI_DEATH_CROSS',
            'ADX_STRONG_TREND', 'ADX_WEAK_TREND',
            'ADX_RISING', 'ADX_FALLING'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_dmi_adx_strength_classification(self):
        """测试ADX强度分类"""
        # 测试ADX强度分类功能
        test_values = [10, 18, 22, 28, 35, 45, 55]
        
        for value in test_values:
            classification = self.indicator._classify_adx_strength(value)
            self.assertIsInstance(classification, str)
            self.assertIn("趋势", classification)
    
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
