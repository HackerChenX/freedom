"""
PSY指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.psy import PSY
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestPSY(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """PSY指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = PSY(period=12)
        self.expected_columns = ['psy', 'psyma']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_psy_calculation_accuracy(self):
        """测试PSY计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 手动计算PSY验证
        price_change = self.data['close'].diff()
        up_days = (price_change > 0).astype(int)
        expected_psy = up_days.rolling(window=12).sum() / 12 * 100
        expected_psyma = expected_psy.rolling(window=6).mean()
        
        # 比较计算结果（允许小的数值误差）
        calculated_psy = result['psy']
        calculated_psyma = result['psyma']
        
        # 验证PSY计算
        psy_diff = abs(calculated_psy - expected_psy).dropna()
        self.assertTrue(all(d < 0.001 for d in psy_diff), "PSY计算结果不正确")
        
        # 验证PSYMA计算
        psyma_diff = abs(calculated_psyma - expected_psyma).dropna()
        self.assertTrue(all(d < 0.001 for d in psyma_diff), "PSYMA计算结果不正确")
    
    def test_psy_score_range(self):
        """测试PSY评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_psy_confidence_calculation(self):
        """测试PSY置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_psy_parameter_update(self):
        """测试PSY参数更新"""
        new_period = 20
        new_secondary_period = 40
        self.indicator.set_parameters(period=new_period, secondary_period=new_secondary_period)
        
        # 验证参数更新
        self.assertEqual(self.indicator.period, new_period)
        self.assertEqual(self.indicator.secondary_period, new_secondary_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('psy', result.columns)
        self.assertIn('psyma', result.columns)
    
    def test_psy_required_columns(self):
        """测试PSY必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_psy_comprehensive_score(self):
        """测试PSY综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_psy_patterns(self):
        """测试PSY形态识别"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证预期的形态列存在
        expected_patterns = [
            'PSY_OVERBOUGHT', 'PSY_OVERSOLD',
            'PSY_EXTREME_OVERBOUGHT', 'PSY_EXTREME_OVERSOLD',
            'PSY_GOLDEN_CROSS', 'PSY_DEATH_CROSS',
            'PSY_CROSS_UP_50', 'PSY_CROSS_DOWN_50'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_psy_value_range(self):
        """测试PSY数值范围"""
        result = self.indicator.calculate(self.data)
        
        # PSY值应该在0到100范围内
        psy_values = result['psy'].dropna()
        self.assertTrue(all(0 <= v <= 100 for v in psy_values), "PSY值应在0到100范围内")
    
    def test_psy_enhanced_mode(self):
        """测试PSY增强模式"""
        enhanced_indicator = PSY(period=12, enhanced=True)
        result = enhanced_indicator.calculate(self.data)
        
        # 验证增强模式下的额外列
        enhanced_columns = ['psy_secondary', 'psyma_secondary', 'psy_momentum', 'market_sentiment']
        for col in enhanced_columns:
            self.assertIn(col, result.columns, f"增强模式缺少列: {col}")
    
    def test_psy_signals(self):
        """测试PSY信号生成"""
        signals = self.indicator.generate_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, pd.DataFrame)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'neutral_signal', 'trend', 'score']
        for key in expected_signal_keys:
            self.assertIn(key, signals.columns, f"缺少信号列: {key}")
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('psy', result.columns)
        self.assertIn('psyma', result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_psy_register_patterns(self):
        """测试PSY形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_psy_market_environment(self):
        """测试PSY市场环境设置"""
        # 测试设置市场环境
        self.indicator.set_market_environment('bull_market')
        self.assertEqual(self.indicator.market_environment, 'bull_market')
        
        # 测试无效市场环境
        with self.assertRaises(ValueError):
            self.indicator.set_market_environment('invalid_market')


if __name__ == '__main__':
    unittest.main()
