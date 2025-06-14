"""
VIX指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.vix import VIX
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestVIX(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """VIX指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = VIX(period=10, smooth_period=5)
        self.expected_columns = ['vix', 'vix_smooth']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_vix_calculation_accuracy(self):
        """测试VIX计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证VIX列存在
        self.assertIn('vix', result.columns)
        self.assertIn('vix_smooth', result.columns)
        self.assertIn('daily_range', result.columns)
        
        # 验证VIX值的合理性
        vix_values = result['vix'].dropna()
        if len(vix_values) > 0:
            # VIX值应该为正数
            self.assertTrue(all(v >= 0 for v in vix_values), "VIX值应该为正数")
            # VIX值通常在0-100范围内
            self.assertTrue(all(v <= 200 for v in vix_values), "VIX值应该在合理范围内")
    
    def test_vix_daily_range_calculation(self):
        """测试VIX日内波动率计算"""
        result = self.indicator.calculate(self.data)
        
        # 手动计算日内波动率验证
        expected_daily_range = (self.data['high'] - self.data['low']) / self.data['close'] * 100
        calculated_daily_range = result['daily_range']
        
        # 比较计算结果（允许小的数值误差）
        diff = abs(calculated_daily_range - expected_daily_range).dropna()
        self.assertTrue(all(d < 0.001 for d in diff), "日内波动率计算不正确")
    
    def test_vix_score_range(self):
        """测试VIX评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_vix_confidence_calculation(self):
        """测试VIX置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_vix_parameter_update(self):
        """测试VIX参数更新"""
        new_period = 20
        new_smooth_period = 10
        self.indicator.set_parameters(period=new_period, smooth_period=new_smooth_period)
        
        # 验证参数更新
        self.assertEqual(self.indicator.period, new_period)
        self.assertEqual(self.indicator.smooth_period, new_smooth_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('vix', result.columns)
        self.assertIn('vix_smooth', result.columns)
    
    def test_vix_required_columns(self):
        """测试VIX必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_vix_comprehensive_score(self):
        """测试VIX综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_vix_patterns(self):
        """测试VIX形态识别"""
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
                'VIX_EXTREME_PANIC', 'VIX_HIGH_PANIC',
                'VIX_EXTREME_OPTIMISM', 'VIX_LOW_FEAR',
                'VIX_RISING', 'VIX_FALLING'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_vix_extreme_values(self):
        """测试VIX极端值检测"""
        # 创建包含极端波动的数据
        extreme_data = self.data.copy()
        # 创造一个极端波动的日子
        extreme_data.loc[extreme_data.index[25], 'high'] = extreme_data.loc[extreme_data.index[25], 'close'] * 1.2
        extreme_data.loc[extreme_data.index[25], 'low'] = extreme_data.loc[extreme_data.index[25], 'close'] * 0.8
        
        patterns = self.indicator.get_patterns(extreme_data)
        
        # 验证极端形态存在
        if not patterns.empty:
            # 应该检测到某种VIX形态
            pattern_sum = patterns.sum().sum()
            self.assertGreater(pattern_sum, 0, "应该检测到VIX形态")
    
    def test_vix_smooth_relationship(self):
        """测试VIX与平滑线的关系"""
        result = self.indicator.calculate(self.data)
        
        vix_values = result['vix'].dropna()
        vix_smooth_values = result['vix_smooth'].dropna()
        
        if len(vix_values) > 10 and len(vix_smooth_values) > 10:
            # 平滑线应该比原始VIX更平滑（波动性更小）
            vix_volatility = vix_values.std()
            smooth_volatility = vix_smooth_values.std()
            
            self.assertLessEqual(smooth_volatility, vix_volatility * 1.2, 
                               "平滑线应该比原始VIX更平滑")
    
    def test_vix_signals(self):
        """测试VIX信号生成"""
        signals = self.indicator.generate_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, pd.DataFrame)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'vix_buy_signal', 'vix_sell_signal']
        for key in expected_signal_keys:
            self.assertIn(key, signals.columns, f"缺少信号列: {key}")
    
    def test_vix_reversal_detection(self):
        """测试VIX反转检测"""
        # 创建包含反转的数据
        reversal_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 80, 'periods': 15},  # 下跌增加波动
            {'type': 'trend', 'start_price': 80, 'end_price': 120, 'periods': 15}   # 上涨减少波动
        ])
        
        patterns = self.indicator.get_patterns(reversal_data)
        
        # 验证反转形态存在
        if not patterns.empty:
            self.assertIn('VIX_TOP_REVERSAL', patterns.columns)
            self.assertIn('VIX_BOTTOM_REVERSAL', patterns.columns)
    
    def test_vix_historical_position(self):
        """测试VIX历史位置检测"""
        # 创建足够长的数据以计算历史位置
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 70}
        ])
        
        patterns = self.indicator.get_patterns(long_data)
        
        # 验证历史位置形态存在
        if not patterns.empty:
            historical_patterns = [
                'VIX_HISTORICAL_HIGH', 'VIX_RELATIVE_HIGH',
                'VIX_HISTORICAL_LOW', 'VIX_RELATIVE_LOW'
            ]
            
            for pattern in historical_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少历史位置形态: {pattern}")
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('vix', result.columns)
        self.assertIn('vix_smooth', result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_vix_register_patterns(self):
        """测试VIX形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_vix_edge_cases(self):
        """测试VIX边界情况"""
        # 测试价格不变的情况
        flat_data = self.data.copy()
        flat_data['high'] = flat_data['close']
        flat_data['low'] = flat_data['close']
        
        result = self.indicator.calculate(flat_data)
        
        # VIX应该为0（因为没有波动）
        vix_values = result['vix'].dropna()
        if len(vix_values) > 0:
            self.assertTrue(all(abs(v) < 1e-10 for v in vix_values), 
                           "价格不变时VIX应该为0")
    
    def test_vix_compute_method(self):
        """测试VIX的compute方法"""
        result = self.indicator.compute(self.data)

        # 验证compute方法返回正确的结果
        self.assertIn('vix', result.columns)
        self.assertIn('vix_smooth', result.columns)

        # 验证VIX值的合理性
        vix_values = result['vix'].dropna()
        if len(vix_values) > 0:
            self.assertTrue(all(v >= 0 for v in vix_values), "VIX值应该为正数")

        # 验证平滑线存在
        vix_smooth_values = result['vix_smooth'].dropna()
        if len(vix_smooth_values) > 0:
            self.assertTrue(all(v >= 0 for v in vix_smooth_values), "VIX平滑线值应该为正数")


if __name__ == '__main__':
    unittest.main()
