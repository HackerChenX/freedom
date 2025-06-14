"""
MTM指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.mtm import MTM
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestMTM(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """MTM指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = MTM(period=10, ma_period=6)
        self.expected_columns = ['mtm', 'mtmma']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_mtm_calculation_accuracy(self):
        """测试MTM计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 手动计算MTM验证
        expected_mtm = self.data['close'] - self.data['close'].shift(10)
        expected_mtmma = expected_mtm.rolling(window=6).mean()
        
        # 比较计算结果（允许小的数值误差）
        calculated_mtm = result['mtm']
        calculated_mtmma = result['mtmma']
        
        # 验证MTM计算
        mtm_diff = abs(calculated_mtm - expected_mtm).dropna()
        self.assertTrue(all(d < 0.001 for d in mtm_diff), "MTM计算结果不正确")
        
        # 验证MTMMA计算
        mtmma_diff = abs(calculated_mtmma - expected_mtmma).dropna()
        self.assertTrue(all(d < 0.001 for d in mtmma_diff), "MTMMA计算结果不正确")
    
    def test_mtm_score_range(self):
        """测试MTM评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_mtm_confidence_calculation(self):
        """测试MTM置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_mtm_parameter_update(self):
        """测试MTM参数更新"""
        new_period = 12
        new_ma_period = 8
        self.indicator.set_parameters(period=new_period, ma_period=new_ma_period)
        
        # 验证参数更新
        self.assertEqual(self.indicator.period, new_period)
        self.assertEqual(self.indicator.ma_period, new_ma_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('mtm', result.columns)
        self.assertIn('mtmma', result.columns)
    
    def test_mtm_required_columns(self):
        """测试MTM必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_mtm_comprehensive_score(self):
        """测试MTM综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_mtm_patterns(self):
        """测试MTM形态识别"""
        patterns = self.indicator.get_patterns(self.data)

        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)

        # 验证预期的形态列存在
        expected_patterns = [
            'MTM_OVERBOUGHT', 'MTM_OVERSOLD',
            'MTM_GOLDEN_CROSS', 'MTM_DEATH_CROSS',
            'MTM_CROSS_UP_ZERO', 'MTM_CROSS_DOWN_ZERO',
            'MTM_ABOVE_ZERO', 'MTM_BELOW_ZERO'
        ]

        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_mtm_auto_threshold(self):
        """测试MTM自动阈值计算"""
        # 创建一个使用自动阈值的指标
        auto_indicator = MTM(period=10, ma_period=6, overbought=0, oversold=0)
        result = auto_indicator.calculate(self.data)
        
        # 验证阈值已被自动设置
        self.assertNotEqual(auto_indicator.overbought, 0)
        self.assertNotEqual(auto_indicator.oversold, 0)
        self.assertGreater(auto_indicator.overbought, 0)
        self.assertLess(auto_indicator.oversold, 0)
    
    def test_mtm_trading_signals(self):
        """测试MTM交易信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号字典结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('mtm', result.columns)
        self.assertIn('mtmma', result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_mtm_register_patterns(self):
        """测试MTM形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")


if __name__ == '__main__':
    unittest.main()
