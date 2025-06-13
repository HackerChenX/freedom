"""
Ichimoku指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.ichimoku import Ichimoku
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestIchimoku(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """Ichimoku指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = Ichimoku(tenkan_period=9, kijun_period=26, senkou_b_period=52, chikou_period=26)
        self.expected_columns = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 
                                'chikou_span', 'kumo_top', 'kumo_bottom', 'kumo_thickness']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 120}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_ichimoku_calculation_accuracy(self):
        """测试Ichimoku计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证Ichimoku数值合理性
        tenkan_values = result['tenkan_sen'].dropna()
        kijun_values = result['kijun_sen'].dropna()
        
        self.assertTrue(len(tenkan_values) > 0, "转换线值全为NaN")
        self.assertTrue(len(kijun_values) > 0, "基准线值全为NaN")
        
        # 验证数值都是有限数
        self.assertTrue(all(np.isfinite(v) for v in tenkan_values), "转换线值应该都是有限数")
        self.assertTrue(all(np.isfinite(v) for v in kijun_values), "基准线值应该都是有限数")
        
        # 手动计算转换线验证
        tenkan_high = self.data['high'].rolling(window=9).max()
        tenkan_low = self.data['low'].rolling(window=9).min()
        expected_tenkan = (tenkan_high + tenkan_low) / 2
        
        # 比较转换线计算结果（允许小的数值误差）
        calculated_tenkan = result['tenkan_sen']
        diff = abs(calculated_tenkan - expected_tenkan).dropna()
        self.assertTrue(all(d < 0.001 for d in diff), "转换线计算结果不正确")
    
    def test_ichimoku_score_range(self):
        """测试Ichimoku评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_ichimoku_confidence_calculation(self):
        """测试Ichimoku置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_ichimoku_parameter_update(self):
        """测试Ichimoku参数更新"""
        new_tenkan = 7
        new_kijun = 22
        new_senkou_b = 44
        new_chikou = 22
        
        self.indicator.set_parameters(
            tenkan_period=new_tenkan,
            kijun_period=new_kijun,
            senkou_b_period=new_senkou_b,
            chikou_period=new_chikou
        )
        
        # 验证参数更新
        self.assertEqual(self.indicator.tenkan_period, new_tenkan)
        self.assertEqual(self.indicator.kijun_period, new_kijun)
        self.assertEqual(self.indicator.senkou_b_period, new_senkou_b)
        self.assertEqual(self.indicator.chikou_period, new_chikou)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('tenkan_sen', result.columns)
        self.assertIn('kijun_sen', result.columns)
    
    def test_ichimoku_required_columns(self):
        """测试Ichimoku必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_cols = ['high', 'low', 'close']
        for col in expected_cols:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_ichimoku_comprehensive_score(self):
        """测试Ichimoku综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_ichimoku_patterns(self):
        """测试Ichimoku形态识别"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证预期的形态列存在
        expected_patterns = [
            'ICHIMOKU_TK_GOLDEN_CROSS', 'ICHIMOKU_TK_DEATH_CROSS',
            'ICHIMOKU_PRICE_ABOVE_KUMO', 'ICHIMOKU_PRICE_BELOW_KUMO',
            'ICHIMOKU_KUMO_BULLISH', 'ICHIMOKU_KUMO_BEARISH'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_ichimoku_kumo_thickness(self):
        """测试Ichimoku云图厚度计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证云图厚度计算
        kumo_thickness = result['kumo_thickness'].dropna()
        self.assertTrue(len(kumo_thickness) > 0, "云图厚度值全为NaN")
        
        # 云图厚度应该都是非负数
        self.assertTrue(all(v >= 0 for v in kumo_thickness), "云图厚度应该都是非负数")
    
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
