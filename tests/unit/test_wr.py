"""
WR指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.wr import WR
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestWR(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """WR指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = WR(period=14)
        self.expected_columns = ['wr']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_wr_calculation_accuracy(self):
        """测试WR计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证WR列存在
        self.assertIn('wr', result.columns)
        
        # 验证WR值的合理性
        wr_values = result['wr'].dropna()
        if len(wr_values) > 0:
            # WR值应该在-100到0之间
            self.assertTrue(all(-100 <= v <= 0 for v in wr_values), "WR值应该在-100到0之间")
    
    def test_wr_manual_calculation(self):
        """测试WR手动计算验证"""
        # 创建简单的测试数据
        simple_data = pd.DataFrame({
            'open': [100, 101, 99, 102, 98],
            'high': [101, 102, 100, 103, 99],
            'low': [99, 100, 98, 101, 97],
            'close': [100.5, 101.5, 99.5, 102.5, 98.5],
            'volume': [1000, 1200, 800, 1500, 900]
        })
        
        # 使用较小的周期便于验证
        test_indicator = WR(period=3)
        result = test_indicator.calculate(simple_data)
        
        # 手动验证WR计算
        if len(result) >= 3:
            # 计算第3个点的WR值
            highest_high = max(simple_data['high'].iloc[0:3])
            lowest_low = min(simple_data['low'].iloc[0:3])
            current_close = simple_data['close'].iloc[2]
            expected_wr = -100 * (highest_high - current_close) / (highest_high - lowest_low)
            calculated_wr = result['wr'].iloc[2]
            
            if not pd.isna(calculated_wr):
                self.assertAlmostEqual(calculated_wr, expected_wr, places=2, 
                                     msg="WR计算不正确")
    
    def test_wr_score_range(self):
        """测试WR评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_wr_confidence_calculation(self):
        """测试WR置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_wr_parameter_update(self):
        """测试WR参数更新"""
        new_period = 10
        self.indicator.set_parameters(period=new_period)
        
        # 验证参数更新
        self.assertEqual(self.indicator.period, new_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('wr', result.columns)
    
    def test_wr_required_columns(self):
        """测试WR必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_wr_comprehensive_score(self):
        """测试WR综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_wr_patterns(self):
        """测试WR形态识别"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 然后获取形态
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证基本的形态列存在
        if not patterns.empty and len(patterns.columns) > 0:
            expected_patterns = [
                'WR_EXTREME_OVERSOLD', 'WR_OVERSOLD', 'WR_NORMAL',
                'WR_OVERBOUGHT', 'WR_EXTREME_OVERBOUGHT',
                'WR_CROSS_ABOVE_OVERSOLD', 'WR_CROSS_BELOW_OVERBOUGHT'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_wr_overbought_oversold_detection(self):
        """测试WR超买超卖检测"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证超买超卖形态存在
        if not patterns.empty:
            overbought_patterns = ['WR_OVERBOUGHT', 'WR_EXTREME_OVERBOUGHT']
            oversold_patterns = ['WR_OVERSOLD', 'WR_EXTREME_OVERSOLD']
            
            for pattern in overbought_patterns + oversold_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态: {pattern}")
    
    def test_wr_cross_detection(self):
        """测试WR穿越检测"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证穿越形态存在
        if not patterns.empty:
            cross_patterns = [
                'WR_CROSS_ABOVE_OVERSOLD', 'WR_CROSS_BELOW_OVERBOUGHT',
                'WR_CROSS_ABOVE_MID', 'WR_CROSS_BELOW_MID'
            ]
            
            for pattern in cross_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少穿越形态: {pattern}")
    
    def test_wr_trend_detection(self):
        """测试WR趋势检测"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证趋势形态存在
        if not patterns.empty:
            trend_patterns = ['WR_RISING', 'WR_FALLING', 'WR_UPTREND', 'WR_DOWNTREND']
            
            for pattern in trend_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少趋势形态: {pattern}")
    
    def test_wr_reversal_detection(self):
        """测试WR反转检测"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证反转形态存在
        if not patterns.empty:
            reversal_patterns = ['WR_BULLISH_REVERSAL', 'WR_BEARISH_REVERSAL']
            
            for pattern in reversal_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少反转形态: {pattern}")
    
    def test_wr_stagnation_detection(self):
        """测试WR钝化检测"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证钝化形态存在
        if not patterns.empty:
            stagnation_patterns = ['WR_LOW_STAGNATION', 'WR_HIGH_STAGNATION']
            
            for pattern in stagnation_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少钝化形态: {pattern}")
    
    def test_wr_signals(self):
        """测试WR信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
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
        self.assertIn('wr', result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_wr_register_patterns(self):
        """测试WR形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_wr_edge_cases(self):
        """测试WR边界情况"""
        # 测试价格无变化的情况
        flat_data = self.data.copy()
        flat_data['high'] = flat_data['close']
        flat_data['low'] = flat_data['close']
        
        result = self.indicator.calculate(flat_data)
        
        # WR应该能处理无变化的价格（可能为NaN或特定值）
        wr_values = result['wr'].dropna()
        # 如果有值，应该在合理范围内
        if len(wr_values) > 0:
            self.assertTrue(all(-100 <= v <= 0 for v in wr_values), "WR值应该在合理范围内")
    
    def test_wr_compute_method(self):
        """测试WR的compute方法"""
        result = self.indicator.compute(self.data)
        
        # 验证compute方法返回正确的结果
        self.assertIn('wr', result.columns)
        
        # 验证WR值的合理性
        wr_values = result['wr'].dropna()
        if len(wr_values) > 0:
            self.assertTrue(all(-100 <= v <= 0 for v in wr_values), "WR值应该在-100到0之间")
    
    def test_wr_validation(self):
        """测试WR数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['high', 'low'], axis=1)
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(invalid_data)
    
    def test_wr_extreme_values(self):
        """测试WR极值情况"""
        # 创建包含极值的数据
        extreme_data = pd.DataFrame({
            'open': [100, 110, 90, 120, 80],
            'high': [105, 115, 95, 125, 85],
            'low': [95, 105, 85, 115, 75],
            'close': [104, 106, 94, 116, 84],  # 价格在高低点之间变化
            'volume': [1000, 1200, 800, 1500, 900]
        })
        
        result = self.indicator.calculate(extreme_data)
        wr_values = result['wr'].dropna()
        
        if len(wr_values) > 0:
            # 验证WR值在合理范围内
            self.assertTrue(all(-100 <= v <= 0 for v in wr_values), "WR值应该在-100到0之间")
    
    def test_wr_formula_verification(self):
        """测试WR公式验证"""
        # 创建简单数据验证公式
        simple_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],  # 最高价递增
            'low': [98, 99, 100],     # 最低价递增
            'close': [101, 102, 103], # 收盘价递增
            'volume': [1000, 1000, 1000]
        })
        
        test_indicator = WR(period=2)
        result = test_indicator.calculate(simple_data)
        
        # 验证第2个点的WR计算
        if len(result) >= 2:
            wr_value = result['wr'].iloc[1]
            if not pd.isna(wr_value):
                # 手动计算验证
                highest_high = max(simple_data['high'].iloc[0:2])  # 103
                lowest_low = min(simple_data['low'].iloc[0:2])     # 98
                current_close = simple_data['close'].iloc[1]       # 102
                expected_wr = -100 * (highest_high - current_close) / (highest_high - lowest_low)
                
                self.assertAlmostEqual(wr_value, expected_wr, places=6, 
                                     msg="WR公式计算验证失败")


if __name__ == '__main__':
    unittest.main()
