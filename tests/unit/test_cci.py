"""
CCI指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.cci import CCI
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestCCI(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """CCI指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = CCI(period=14)
        self.expected_columns = ['CCI']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_cci_calculation_accuracy(self):
        """测试CCI计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证CCI计算公式：(TP - MA) / (0.015 * MD)
        # 其中TP = (high + low + close) / 3
        tp = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        ma = tp.rolling(window=14, min_periods=1).mean()
        md = tp.rolling(window=14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        expected_cci = (tp - ma) / (0.015 * md)
        
        # 比较计算结果（允许小的数值误差）
        calculated_cci = result['CCI']
        diff = abs(calculated_cci - expected_cci).dropna()
        self.assertTrue(all(d < 0.001 for d in diff), "CCI计算结果不正确")
    
    def test_cci_score_range(self):
        """测试CCI评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_cci_confidence_calculation(self):
        """测试CCI置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_cci_parameter_update(self):
        """测试CCI参数更新"""
        new_period = 20
        self.indicator.set_parameters(period=new_period)
        
        # 验证参数更新
        self.assertEqual(self.indicator.period, new_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('CCI', result.columns)
    
    def test_cci_required_columns(self):
        """测试CCI必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_cols = ['high', 'low', 'close']
        for col in expected_cols:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_cci_comprehensive_score(self):
        """测试CCI综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_cci_patterns(self):
        """测试CCI形态识别"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证预期的形态列存在
        expected_patterns = [
            'CCI_OVERSOLD', 'CCI_EXTREME_OVERSOLD',
            'CCI_OVERBOUGHT', 'CCI_EXTREME_OVERBOUGHT',
            'CCI_CROSS_UP_OVERSOLD', 'CCI_CROSS_DOWN_OVERBOUGHT'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_cci_signals(self):
        """测试CCI信号生成"""
        # 先计算CCI
        result = self.indicator.calculate(self.data)
        
        # 生成信号
        signals = self.indicator.get_signals(result)
        
        # 验证信号
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('cci_signal', signals.columns)
        
        # 验证信号值
        signal_values = signals['cci_signal'].dropna().unique()
        valid_signals = {-1, 0, 1}
        self.assertTrue(all(s in valid_signals for s in signal_values), "信号值应为-1, 0, 1")
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('CCI', result.columns)
    
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
