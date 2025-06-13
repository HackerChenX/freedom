"""
EMV指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.emv import EMV
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestEMV(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """EMV指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = EMV(volume_divisor=1000000, period=14)
        self.expected_columns = ['EMV', 'EMV_MA']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 60}
        ])
        
        # 添加成交量数据
        np.random.seed(42)
        self.data['volume'] = np.random.randint(1000000, 10000000, len(self.data))
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_emv_calculation_accuracy(self):
        """测试EMV计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证EMV数值合理性
        emv_values = result['EMV'].dropna()
        emv_ma_values = result['EMV_MA'].dropna()
        
        self.assertTrue(len(emv_values) > 0, "EMV值全为NaN")
        self.assertTrue(len(emv_ma_values) > 0, "EMV_MA值全为NaN")
        
        # 验证EMV值都是有限数
        self.assertTrue(all(np.isfinite(v) for v in emv_values), "EMV值应该都是有限数")
        self.assertTrue(all(np.isfinite(v) for v in emv_ma_values), "EMV_MA值应该都是有限数")
    
    def test_emv_score_range(self):
        """测试EMV评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_emv_confidence_calculation(self):
        """测试EMV置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_emv_parameter_update(self):
        """测试EMV参数更新"""
        new_volume_divisor = 500000
        new_period = 20
        self.indicator.set_parameters(volume_divisor=new_volume_divisor, period=new_period)
        
        # 验证参数更新
        self.assertEqual(self.indicator.volume_divisor, new_volume_divisor)
        self.assertEqual(self.indicator.period, new_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('EMV', result.columns)
        self.assertIn('EMV_MA', result.columns)
    
    def test_emv_required_columns(self):
        """测试EMV必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_cols = ['high', 'low', 'volume']
        for col in expected_cols:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_emv_comprehensive_score(self):
        """测试EMV综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_emv_patterns(self):
        """测试EMV形态识别"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证预期的形态列存在
        expected_patterns = [
            'EMV_CROSS_UP_ZERO', 'EMV_CROSS_DOWN_ZERO',
            'EMV_ABOVE_ZERO', 'EMV_BELOW_ZERO',
            'EMV_ABOVE_MA', 'EMV_BELOW_MA'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_emv_zero_volume_handling(self):
        """测试EMV零成交量处理"""
        # 创建包含零成交量的测试数据
        test_data = self.data.copy()
        test_data.loc[test_data.index[10:15], 'volume'] = 0
        
        # 计算EMV
        result = self.indicator.calculate(test_data)
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('EMV', result.columns)
        
        # 验证EMV值都是有限数
        emv_values = result['EMV'].dropna()
        self.assertTrue(all(np.isfinite(v) for v in emv_values), "EMV值应该都是有限数")
    
    def test_emv_signals(self):
        """测试EMV信号生成"""
        # 生成信号
        signals = self.indicator.generate_signals(self.data)
        
        # 验证信号
        self.assertIsInstance(signals, pd.DataFrame)
        
        # 验证信号列存在
        expected_signal_cols = ['buy_signal', 'sell_signal', 'neutral_signal', 'score']
        for col in expected_signal_cols:
            self.assertIn(col, signals.columns, f"缺少信号列: {col}")
    
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
