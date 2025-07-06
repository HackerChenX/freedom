"""
Chaikin指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.complete_indicator_registry import complete_registry
from tests.unit.indicator_test_mixin import Indicator_test_mixin
from tests.helper.data_generator import Test_data_generator
from tests.helper.log_capture import Log_capture_mixin


class Testchaikin_chaikin(unittest.Test_case, Indicator_test_mixin, Log_capture_mixin):
    """Chaikin指标测试类"""
    
    def set_up_Chaikin(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        Log_capture_mixin.set_up_Chaikin(self)
        
        self.indicator = Chaikin(fast_period=3, slow_period=10)
        self.expected_columns = ['ad_line', 'chaikin_oscillator', 'chaikin_signal']
        self.data = Test_data_generator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tear_down_Chaikin(self):
        """清理日志捕获器"""
        Log_capture_mixin.tear_down_Chaikin(self)
    
    def test_chaikin_calculation_accuracy(self):
        """测试Chaikin计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证A/D线计算公式
        high_low_diff = self.data['high'] - self.data['low']
        high_low_diff = high_low_diff.replace(0, 0.000001)
        
        clv = ((self.data['close'] - self.data['low']) - (self.data['high'] - self.data['close'])) / high_low_diff
        expected_ad_line = (clv * self.data['volume']).cumsum()
        
        # 比较A/D线计算结果
        calculated_ad_line = result['ad_line']
        diff = abs(calculated_ad_line - expected_ad_line).dropna()
        self.assertTrue(all(d < 0.001 for d in diff), "A/D线计算结果不正确")
        
        # 验证Chaikin震荡器计算
        ad_ema_fast = expected_ad_line.ewm(span=3).mean()
        ad_ema_slow = expected_ad_line.ewm(span=10).mean()
        expected_chaikin = ad_ema_fast - ad_ema_slow
        
        calculated_chaikin = result['chaikin_oscillator']
        chaikin_diff = abs(calculated_chaikin - expected_chaikin).dropna()
        self.assertTrue(all(d < 0.001 for d in chaikin_diff), "Chaikin震荡器计算结果不正确")
    
    def test_chaikin_score_range(self):
        """测试Chaikin评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_chaikin_confidence_calculation(self):
        """测试Chaikin置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assert_is_instance(confidence, float)
        self.assert_greater_equal(confidence, 0.0)
        self.assert_less_equal(confidence, 1.0)
    
    def test_chaikin_parameter_update(self):
        """测试Chaikin参数更新"""
        new_fast_period = 5
        new_slow_period = 15
        self.indicator.set_parameters(fast_period=new_fast_period, slow_period=new_slow_period)
        
        # 验证参数更新
        self.assert_equal(self.indicator.fast_period, new_fast_period)
        self.assert_equal(self.indicator.slow_period, new_slow_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('chaikin_oscillator', result.columns)
    
    def test_chaikin_required_columns(self):
        """测试Chaikin必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_cols:
            self.assert_in(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_chaikin_comprehensive_score(self):
        """测试Chaikin综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assert_is_instance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_chaikin_patterns(self):
        """测试Chaikin形态识别"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assert_is_instance(patterns, pd.DataFrame)
        
        # 验证预期的形态列存在
        expected_patterns = [
            'CHAIKIN_CROSS_UP_ZERO', 'CHAIKIN_CROSS_DOWN_ZERO',
            'CHAIKIN_ABOVE_ZERO', 'CHAIKIN_BELOW_ZERO',
            'CHAIKIN_RISING', 'CHAIKIN_FALLING'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_chaikin_signals(self):
        """测试Chaikin信号生成"""
        # 先计算Chaikin
        result = self.indicator.calculate(self.data)
        
        # 生成信号
        signals = self.indicator.get_signals(result)
        
        # 验证信号
        self.assert_is_instance(signals, pd.DataFrame)
        self.assertIn('chaikin_buy_signal', signals.columns)
        self.assertIn('chaikin_sell_signal', signals.columns)
        
        # 验证信号值
        buy_signals = signals['chaikin_buy_signal'].dropna().unique()
        sell_signals = signals['chaikin_sell_signal'].dropna().unique()
        valid_signals = {0, 1}
        self.assertTrue(all(s in valid_signals for s in buy_signals), "买入信号值应为0, 1")
        self.assertTrue(all(s in valid_signals for s in sell_signals), "卖出信号值应为0, 1")
    
    def test_no_errors_during_calculation_Chaikin(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assert_is_instance(result, pd.DataFrame)
        for col in self.expected_columns:
            self.assert_in(col, result.columns)
    
    def test_no_errors_during_pattern_detection_Chaikin(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assert_is_instance(patterns, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
