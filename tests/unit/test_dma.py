"""
DMA指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.dma import DMA
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestDMA(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """DMA指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = DMA(fast_period=10, slow_period=50, ama_period=10)
        self.expected_columns = ['DMA', 'AMA', 'DMA_PCT', 'FAST_MA_CHG']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 60}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_dma_calculation_accuracy(self):
        """测试DMA计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 手动计算DMA验证
        fast_ma = self.data['close'].rolling(window=10).mean()
        slow_ma = self.data['close'].rolling(window=50).mean()
        expected_dma = fast_ma - slow_ma
        expected_ama = expected_dma.rolling(window=10).mean()
        
        # 比较DMA计算结果（允许小的数值误差）
        calculated_dma = result['DMA']
        diff = abs(calculated_dma - expected_dma).dropna()
        self.assertTrue(all(d < 0.001 for d in diff), "DMA计算结果不正确")
        
        # 比较AMA计算结果
        calculated_ama = result['AMA']
        ama_diff = abs(calculated_ama - expected_ama).dropna()
        self.assertTrue(all(d < 0.001 for d in ama_diff), "AMA计算结果不正确")
    
    def test_dma_score_range(self):
        """测试DMA评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_dma_confidence_calculation(self):
        """测试DMA置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_dma_parameter_update(self):
        """测试DMA参数更新"""
        new_fast_period = 5
        new_slow_period = 30
        new_ama_period = 15
        self.indicator.set_parameters(
            fast_period=new_fast_period, 
            slow_period=new_slow_period, 
            ama_period=new_ama_period
        )
        
        # 验证参数更新
        self.assertEqual(self.indicator.fast_period, new_fast_period)
        self.assertEqual(self.indicator.slow_period, new_slow_period)
        self.assertEqual(self.indicator.ama_period, new_ama_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('DMA', result.columns)
        self.assertIn('AMA', result.columns)
    
    def test_dma_required_columns(self):
        """测试DMA必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        self.assertIn('close', self.indicator.REQUIRED_COLUMNS)
    
    def test_dma_comprehensive_score(self):
        """测试DMA综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_dma_patterns(self):
        """测试DMA形态识别"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证预期的形态列存在
        expected_patterns = [
            'DMA_UPTREND', 'DMA_DOWNTREND',
            'DMA_WEAK_UPTREND', 'DMA_WEAK_DOWNTREND',
            'DMA_GOLDEN_CROSS', 'DMA_DEATH_CROSS'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_dma_signals(self):
        """测试DMA信号生成"""
        # 生成信号
        signals = self.indicator.generate_signals(self.data)
        
        # 验证信号
        self.assertIsInstance(signals, list)
        
        if len(signals) > 0:
            signal = signals[0]
            self.assertIsInstance(signal, dict)
            
            # 验证必需的信号字段
            required_fields = ['indicator', 'buy_signal', 'sell_signal', 'score', 'confidence']
            for field in required_fields:
                self.assertIn(field, signal, f"缺少信号字段: {field}")
    
    def test_dma_percentage_calculation(self):
        """测试DMA百分比计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证DMA_PCT计算
        dma_pct = result['DMA_PCT'].dropna()
        self.assertTrue(len(dma_pct) > 0, "DMA_PCT值全为NaN")
        
        # DMA_PCT应该在合理范围内
        self.assertTrue(all(-100 <= v <= 100 for v in dma_pct), "DMA_PCT值超出合理范围")
    
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
