"""
BIAS指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.complete_indicator_registry import complete_registry
from tests.unit.indicator_test_mixin import Indicator_test_mixin
from tests.helper.data_generator import Test_data_generator
from tests.helper.log_capture import Log_capture_mixin


class Test_bIAS(unittest.TestCase, Indicator_test_mixin, Log_capture_mixin):
    """BIAS指标测试类"""
    
    def set_up_Bias(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        Log_capture_mixin.set_up_Bias(self)

        self.indicator = complete_registry.create_indicator('BIAS', periods=[6, 12, 24])
        self.expected_columns = ['BIAS6', 'BIAS12', 'BIAS24', 'BIAS', 'BIAS_MA']
        self.data = Test_data_generator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])

    def tear_down_Bias(self):
        """清理日志捕获器"""
        Log_capture_mixin.tear_down_Bias(self)
    
    def test_bias_calculation_accuracy(self):
        """测试BIAS计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证BIAS计算公式：(close - MA) / MA * 100
        for period in [6, 12, 24]:
            col_name = f'BIAS{period}'
            if col_name in result.columns:
                # 手动计算验证
                ma = self.data['close'].rolling(window=period, min_periods=1).mean()
                expected_bias = (self.data['close'] - ma) / ma * 100
                
                # 比较计算结果（允许小的数值误差）
                calculated_bias = result[col_name]
                diff = abs(calculated_bias - expected_bias).dropna()
                self.assertTrue(all(d < 0.001 for d in diff), f"{col_name}计算结果不正确")
    
    def test_bias_score_range(self):
        """测试BIAS评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_bias_confidence_calculation(self):
        """测试BIAS置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assert_is_instance(confidence, float)
        self.assert_greater_equal(confidence, 0.0)
        self.assert_less_equal(confidence, 1.0)
    
    def test_bias_parameter_update(self):
        """测试BIAS参数更新"""
        new_periods = [5, 10, 20]
        self.indicator.set_parameters(periods=new_periods)
        
        # 验证参数更新
        self.assert_equal(self.indicator.periods, new_periods)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('BIAS5', result.columns)
        self.assertIn('BIAS10', result.columns)
        self.assertIn('BIAS20', result.columns)
    
    def test_bias_required_columns(self):
        """测试BIAS必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        self.assertIn('close', self.indicator.REQUIRED_COLUMNS)
    
    def test_bias_comprehensive_score(self):
        """测试BIAS综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assert_is_instance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)


if __name__ == '__main__':
    unittest.main()
