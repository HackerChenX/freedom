"""
VOSC指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.vosc import VOSC
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestVOSC(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """VOSC指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = VOSC(short_period=12, long_period=26)
        self.expected_columns = ['vosc', 'vosc_signal']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_vosc_calculation_accuracy(self):
        """测试VOSC计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证VOSC列存在
        self.assertIn('vosc', result.columns)
        self.assertIn('vosc_signal', result.columns)
        
        # 验证VOSC值的合理性
        vosc_values = result['vosc'].dropna()
        if len(vosc_values) > 0:
            # VOSC值应该在合理范围内（通常-100到100之间）
            self.assertTrue(all(-200 <= v <= 200 for v in vosc_values), 
                           "VOSC值应该在合理范围内")
    
    def test_vosc_manual_calculation(self):
        """测试VOSC手动计算验证"""
        # 创建简单的测试数据
        simple_data = pd.DataFrame({
            'open': [100] * 30,
            'high': [101] * 30,
            'low': [99] * 30,
            'close': [100] * 30,
            'volume': list(range(1000, 1030))  # 递增的成交量
        })
        
        # 使用较小的周期便于验证
        test_indicator = VOSC(short_period=5, long_period=10)
        result = test_indicator.calculate(simple_data)
        
        # 手动验证VOSC计算
        if len(result) >= 10:
            # 计算第10个点的VOSC值
            short_ma = np.mean(simple_data['volume'].iloc[5:10])  # 5日均线
            long_ma = np.mean(simple_data['volume'].iloc[0:10])   # 10日均线
            expected_vosc = (short_ma - long_ma) / long_ma * 100
            calculated_vosc = result['vosc'].iloc[9]
            
            if not pd.isna(calculated_vosc):
                self.assertAlmostEqual(calculated_vosc, expected_vosc, places=2, 
                                     msg="VOSC计算不正确")
    
    def test_vosc_score_range(self):
        """测试VOSC评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_vosc_confidence_calculation(self):
        """测试VOSC置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_vosc_parameter_update(self):
        """测试VOSC参数更新"""
        new_short_period = 8
        new_long_period = 20
        self.indicator.set_parameters(short_period=new_short_period, long_period=new_long_period)
        
        # 验证参数更新
        self.assertEqual(self.indicator.short_period, new_short_period)
        self.assertEqual(self.indicator.long_period, new_long_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('vosc', result.columns)
        self.assertIn('vosc_signal', result.columns)
    
    def test_vosc_required_columns(self):
        """测试VOSC必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_vosc_comprehensive_score(self):
        """测试VOSC综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_vosc_patterns(self):
        """测试VOSC形态识别"""
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
                'VOSC_ABOVE_ZERO', 'VOSC_BELOW_ZERO',
                'VOSC_GOLDEN_CROSS', 'VOSC_DEATH_CROSS',
                'VOSC_RISING', 'VOSC_FALLING'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_vosc_zero_cross_detection(self):
        """测试VOSC零轴穿越检测"""
        # 创建包含零轴穿越的数据
        cross_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 30}
        ])
        
        patterns = self.indicator.get_patterns(cross_data)
        
        # 验证零轴穿越形态存在
        if not patterns.empty:
            self.assertIn('VOSC_CROSS_ABOVE_ZERO', patterns.columns)
            self.assertIn('VOSC_CROSS_BELOW_ZERO', patterns.columns)
    
    def test_vosc_signal_cross_detection(self):
        """测试VOSC信号线交叉检测"""
        # 创建包含交叉的数据
        cross_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 30}
        ])
        
        patterns = self.indicator.get_patterns(cross_data)
        
        # 验证信号线交叉形态存在
        if not patterns.empty:
            self.assertIn('VOSC_GOLDEN_CROSS', patterns.columns)
            self.assertIn('VOSC_DEATH_CROSS', patterns.columns)
    
    def test_vosc_trend_detection(self):
        """测试VOSC趋势检测"""
        # 创建包含趋势的数据
        trend_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30}
        ])
        
        patterns = self.indicator.get_patterns(trend_data)
        
        # 验证趋势形态存在
        if not patterns.empty:
            self.assertIn('VOSC_UPTREND', patterns.columns)
            self.assertIn('VOSC_DOWNTREND', patterns.columns)
    
    def test_vosc_extreme_detection(self):
        """测试VOSC极值检测"""
        # 创建包含极端成交量变化的数据
        extreme_data = self.data.copy()
        # 创造极端成交量变化
        extreme_data.loc[extreme_data.index[25:30], 'volume'] *= 5
        
        patterns = self.indicator.get_patterns(extreme_data)
        
        # 验证极值形态存在
        if not patterns.empty:
            self.assertIn('VOSC_EXTREME_HIGH', patterns.columns)
            self.assertIn('VOSC_EXTREME_LOW', patterns.columns)
    
    def test_vosc_price_relation_detection(self):
        """测试VOSC价格关系检测"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证价格关系形态存在
        if not patterns.empty:
            self.assertIn('VOSC_PRICE_CONFIRMATION', patterns.columns)
            self.assertIn('VOSC_PRICE_DIVERGENCE', patterns.columns)
    
    def test_vosc_signals(self):
        """测试VOSC信号生成"""
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
        self.assertIn('vosc', result.columns)
        self.assertIn('vosc_signal', result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_vosc_register_patterns(self):
        """测试VOSC形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_vosc_edge_cases(self):
        """测试VOSC边界情况"""
        # 测试成交量为0的情况
        zero_vol_data = self.data.copy()
        zero_vol_data['volume'] = 0.0
        
        result = self.indicator.calculate(zero_vol_data)
        
        # VOSC应该为NaN或0（因为除以0）
        vosc_values = result['vosc'].dropna()
        # 如果有值，应该是0或NaN
        if len(vosc_values) > 0:
            self.assertTrue(all(abs(v) < 1e-10 or pd.isna(v) for v in vosc_values), 
                           "成交量为0时VOSC应该为0或NaN")
    
    def test_vosc_compute_method(self):
        """测试VOSC的compute方法"""
        result = self.indicator.compute(self.data)
        
        # 验证compute方法返回正确的结果
        self.assertIn('vosc', result.columns)
        self.assertIn('vosc_signal', result.columns)
        
        # 验证VOSC值的合理性
        vosc_values = result['vosc'].dropna()
        if len(vosc_values) > 0:
            self.assertTrue(all(-200 <= v <= 200 for v in vosc_values), 
                           "VOSC值应该在合理范围内")
    
    def test_vosc_validation(self):
        """测试VOSC数据验证"""
        # 测试缺少volume列的情况
        invalid_data = self.data.drop('volume', axis=1)
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(invalid_data)
    
    def test_vosc_signal_line_relationship(self):
        """测试VOSC与信号线的关系"""
        result = self.indicator.calculate(self.data)
        
        vosc_values = result['vosc'].dropna()
        signal_values = result['vosc_signal'].dropna()
        
        if len(vosc_values) > 10 and len(signal_values) > 10:
            # 信号线应该比VOSC更平滑（波动性更小）
            vosc_volatility = vosc_values.std()
            signal_volatility = signal_values.std()
            
            self.assertLessEqual(signal_volatility, vosc_volatility * 1.2, 
                               "信号线应该比VOSC更平滑")


if __name__ == '__main__':
    unittest.main()
