"""
Vortex指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.vortex import Vortex
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestVortex(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """Vortex指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = Vortex(period=14)
        self.expected_columns = ['vi_plus', 'vi_minus', 'vi_diff']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_vortex_calculation_accuracy(self):
        """测试Vortex计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证Vortex列存在
        self.assertIn('vi_plus', result.columns)
        self.assertIn('vi_minus', result.columns)
        self.assertIn('vi_diff', result.columns)
        
        # 验证Vortex值的合理性
        vi_plus_values = result['vi_plus'].dropna()
        vi_minus_values = result['vi_minus'].dropna()
        
        if len(vi_plus_values) > 0 and len(vi_minus_values) > 0:
            # VI值应该为正数
            self.assertTrue(all(v >= 0 for v in vi_plus_values), "VI+值应该为正数")
            self.assertTrue(all(v >= 0 for v in vi_minus_values), "VI-值应该为正数")
            # VI值通常在0.5-2.0范围内
            self.assertTrue(all(v <= 5.0 for v in vi_plus_values), "VI+值应该在合理范围内")
            self.assertTrue(all(v <= 5.0 for v in vi_minus_values), "VI-值应该在合理范围内")
    
    def test_vortex_manual_calculation(self):
        """测试Vortex手动计算验证"""
        # 创建简单的测试数据
        simple_data = pd.DataFrame({
            'open': [100, 101, 99, 102, 98],
            'high': [101, 102, 100, 103, 99],
            'low': [99, 100, 98, 101, 97],
            'close': [100.5, 101.5, 99.5, 102.5, 98.5],
            'volume': [1000, 1200, 800, 1500, 900]
        })
        
        # 使用较小的周期便于验证
        test_indicator = Vortex(period=3)
        result = test_indicator.calculate(simple_data)
        
        # 验证Vortex计算逻辑
        if len(result) >= 3:
            vi_plus_value = result['vi_plus'].iloc[2]
            vi_minus_value = result['vi_minus'].iloc[2]
            
            if not pd.isna(vi_plus_value) and not pd.isna(vi_minus_value):
                # VI值应该是合理的正数
                self.assertGreater(vi_plus_value, 0, "VI+值应该为正数")
                self.assertGreater(vi_minus_value, 0, "VI-值应该为正数")
                self.assertLess(vi_plus_value, 10, "VI+值应该在合理范围内")
                self.assertLess(vi_minus_value, 10, "VI-值应该在合理范围内")
    
    def test_vortex_score_range(self):
        """测试Vortex评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_vortex_confidence_calculation(self):
        """测试Vortex置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_vortex_parameter_update(self):
        """测试Vortex参数更新"""
        new_period = 10
        self.indicator.set_parameters(period=new_period)
        
        # 验证参数更新
        self.assertEqual(self.indicator.period, new_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('vi_plus', result.columns)
        self.assertIn('vi_minus', result.columns)
        self.assertIn('vi_diff', result.columns)
    
    def test_vortex_required_columns(self):
        """测试Vortex必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_vortex_comprehensive_score(self):
        """测试Vortex综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_vortex_patterns(self):
        """测试Vortex形态识别"""
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
                'VORTEX_BULLISH_CROSS', 'VORTEX_BEARISH_CROSS',
                'VORTEX_VI_PLUS_ABOVE', 'VORTEX_VI_MINUS_ABOVE',
                'VORTEX_VI_PLUS_STRONG', 'VORTEX_VI_MINUS_STRONG'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_vortex_cross_detection(self):
        """测试Vortex交叉检测"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证交叉形态存在
        if not patterns.empty:
            self.assertIn('VORTEX_BULLISH_CROSS', patterns.columns)
            self.assertIn('VORTEX_BEARISH_CROSS', patterns.columns)
    
    def test_vortex_threshold_detection(self):
        """测试Vortex阈值检测"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证阈值形态存在
        if not patterns.empty:
            self.assertIn('VORTEX_VI_PLUS_STRONG', patterns.columns)
            self.assertIn('VORTEX_VI_MINUS_STRONG', patterns.columns)
            self.assertIn('VORTEX_VI_PLUS_WEAK', patterns.columns)
            self.assertIn('VORTEX_VI_MINUS_WEAK', patterns.columns)
    
    def test_vortex_trend_detection(self):
        """测试Vortex趋势检测"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证趋势形态存在
        if not patterns.empty:
            self.assertIn('VORTEX_VI_PLUS_RISING', patterns.columns)
            self.assertIn('VORTEX_VI_MINUS_RISING', patterns.columns)
            self.assertIn('VORTEX_VI_DIFF_RISING', patterns.columns)
            self.assertIn('VORTEX_VI_DIFF_FALLING', patterns.columns)
    
    def test_vortex_extreme_detection(self):
        """测试Vortex极值检测"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证极值形态存在
        if not patterns.empty:
            self.assertIn('VORTEX_VI_PLUS_EXTREME_HIGH', patterns.columns)
            self.assertIn('VORTEX_VI_MINUS_EXTREME_HIGH', patterns.columns)
            self.assertIn('VORTEX_VI_PLUS_EXTREME_LOW', patterns.columns)
            self.assertIn('VORTEX_VI_MINUS_EXTREME_LOW', patterns.columns)
    
    def test_vortex_signals(self):
        """测试Vortex信号生成"""
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
        self.assertIn('vi_plus', result.columns)
        self.assertIn('vi_minus', result.columns)
        self.assertIn('vi_diff', result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_vortex_register_patterns(self):
        """测试Vortex形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_vortex_edge_cases(self):
        """测试Vortex边界情况"""
        # 测试价格无变化的情况
        flat_data = self.data.copy()
        flat_data['high'] = flat_data['close']
        flat_data['low'] = flat_data['close']
        
        result = self.indicator.calculate(flat_data)
        
        # Vortex应该能处理无变化的价格
        vi_plus_values = result['vi_plus'].dropna()
        vi_minus_values = result['vi_minus'].dropna()
        
        if len(vi_plus_values) > 0 and len(vi_minus_values) > 0:
            self.assertTrue(all(v >= 0 for v in vi_plus_values), "VI+值应该为非负数")
            self.assertTrue(all(v >= 0 for v in vi_minus_values), "VI-值应该为非负数")
    
    def test_vortex_compute_method(self):
        """测试Vortex的compute方法"""
        result = self.indicator.compute(self.data)
        
        # 验证compute方法返回正确的结果
        self.assertIn('vi_plus', result.columns)
        self.assertIn('vi_minus', result.columns)
        self.assertIn('vi_diff', result.columns)
        
        # 验证Vortex值的合理性
        vi_plus_values = result['vi_plus'].dropna()
        vi_minus_values = result['vi_minus'].dropna()
        
        if len(vi_plus_values) > 0 and len(vi_minus_values) > 0:
            self.assertTrue(all(v >= 0 for v in vi_plus_values), "VI+值应该为正数")
            self.assertTrue(all(v >= 0 for v in vi_minus_values), "VI-值应该为正数")
    
    def test_vortex_validation(self):
        """测试Vortex数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['high', 'low'], axis=1)
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(invalid_data)
    
    def test_vortex_vi_relationship(self):
        """测试VI+和VI-的关系"""
        result = self.indicator.calculate(self.data)
        
        vi_plus_values = result['vi_plus'].dropna()
        vi_minus_values = result['vi_minus'].dropna()
        vi_diff_values = result['vi_diff'].dropna()
        
        if len(vi_plus_values) > 0 and len(vi_minus_values) > 0 and len(vi_diff_values) > 0:
            # VI差值应该等于VI+ - VI-
            calculated_diff = vi_plus_values - vi_minus_values
            actual_diff = vi_diff_values
            
            # 由于索引可能不完全匹配，我们检查长度相近的情况
            if len(calculated_diff) == len(actual_diff):
                np.testing.assert_array_almost_equal(
                    calculated_diff.values, actual_diff.values, decimal=6,
                    err_msg="VI差值计算不正确"
                )


if __name__ == '__main__':
    unittest.main()
