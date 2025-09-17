"""
DMI指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.complete_indicator_registry import complete_registry
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin
from db.sql_manager import SQLManager, QueryType


class Testdmi_dmi(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """DMI指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 手动初始化日志处理器
        import io
        import logging
        self.log_stream = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.log_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        self.log_handler.setFormatter(formatter)

        # 添加到根日志记录器
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.DEBUG)

        # 显式调用LogCaptureMixin的setUp（如果存在）
        try:
            super().setUp()
        except AttributeError:
            pass

        self.indicator = complete_registry.create_indicator('DMI', period=14, adx_period=14)
        self.expected_columns = ['PDI', 'MDI', 'ADX', 'ADXR']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 60}
        ])

    def clear_logs(self):
        """清除捕获的日志"""
        if hasattr(self, 'log_stream') and hasattr(self, 'log_handler'):
            import io
            self.log_stream = io.StringIO()
            self.log_handler.setStream(self.log_stream)

    def assert_no_logs(self, level=None):
        """断言没有日志输出"""
        if hasattr(self, 'log_stream'):
            log_content = self.log_stream.getvalue().strip()
            if level:
                # 检查特定级别的日志
                lines = log_content.split('\n')
                error_lines = [line for line in lines if level in line]
                self.assertEqual(len(error_lines), 0, f"发现{level}级别日志: {error_lines}")
            else:
                # 检查是否有任何日志
                self.assertEqual(log_content, '', f"发现意外日志: {log_content}")

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
    
    def test_dmi_patterns_Dmi(self):
        """测试DMI形态识别"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证预期的形态列存在
        expected_patterns = [
            'DMI_STRONG_TREND', 'DMI_WEAK_TREND', 'DMI_UPTREND', 'DMI_DOWNTREND',
            'DMI_PDI_CROSS_UP', 'DMI_MDI_CROSS_UP', 'DMI_ADX_RISING', 'DMI_ADX_FALLING'
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
            # 验证分类结果是有效的字符串
            valid_classifications = ["no_trend", "weak", "medium", "strong", "very_strong"]
            self.assertIn(classification, valid_classifications)
    
    def test_no_errors_during_calculation_Dmi_Test_Dmi(self):
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
    
    def test_no_errors_during_pattern_detection_Dmi_Test_Dmi(self):
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
