"""
TRIX指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.trix import TRIX
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestTRIX(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """TRIX指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = TRIX(n=12, m=9)
        self.expected_columns = ['TR', 'TRIX', 'MATRIX']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_trix_calculation_accuracy(self):
        """测试TRIX计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证TRIX列存在
        self.assertIn('TRIX', result.columns)
        self.assertIn('MATRIX', result.columns)
        self.assertIn('TR', result.columns)
        
        # 验证TRIX值的合理性
        trix_values = result['TRIX'].dropna()
        if len(trix_values) > 0:
            # TRIX值应该是百分比形式，通常在-10到10之间
            self.assertTrue(all(-50 <= v <= 50 for v in trix_values), 
                           "TRIX值应该在合理范围内")
    
    def test_trix_score_range(self):
        """测试TRIX评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_trix_confidence_calculation(self):
        """测试TRIX置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_trix_parameter_update(self):
        """测试TRIX参数更新"""
        new_n = 20
        new_m = 15
        self.indicator.set_parameters(n=new_n, m=new_m)
        
        # 验证参数更新
        self.assertEqual(self.indicator.n, new_n)
        self.assertEqual(self.indicator.m, new_m)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('TRIX', result.columns)
        self.assertIn('MATRIX', result.columns)
    
    def test_trix_required_columns(self):
        """测试TRIX必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_trix_comprehensive_score(self):
        """测试TRIX综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_trix_patterns(self):
        """测试TRIX形态识别"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)

        # 然后获取形态
        patterns = self.indicator.get_patterns(self.data)

        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)

        # 验证返回的是DataFrame类型
        self.assertIsInstance(patterns, pd.DataFrame)

        # 验证至少有基本的形态列
        if not patterns.empty and len(patterns.columns) > 0:
            # 至少应该有金叉死叉形态
            self.assertIn('TRIX_GOLDEN_CROSS', patterns.columns)
            self.assertIn('TRIX_DEATH_CROSS', patterns.columns)
    
    def test_trix_crossover_detection(self):
        """测试TRIX交叉检测"""
        # 创建包含交叉的数据
        crossover_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 25},
            {'type': 'trend', 'start_price': 90, 'end_price': 110, 'periods': 25}
        ])
        
        # 先计算指标
        result = self.indicator.calculate(crossover_data)
        self.assertIsInstance(result, pd.DataFrame)
        
        patterns = self.indicator.get_patterns(crossover_data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 如果形态DataFrame不为空，验证金叉死叉形态存在
        if not patterns.empty:
            self.assertIn('TRIX_GOLDEN_CROSS', patterns.columns)
            self.assertIn('TRIX_DEATH_CROSS', patterns.columns)
    
    def test_trix_zero_crossing(self):
        """测试TRIX零轴穿越"""
        # 创建包含零轴穿越的数据
        zero_cross_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 80, 'periods': 30},
            {'type': 'trend', 'start_price': 80, 'end_price': 120, 'periods': 30}
        ])
        
        patterns = self.indicator.get_patterns(zero_cross_data)
        
        # 验证零轴穿越形态存在
        if not patterns.empty:
            self.assertIn('TRIX_CROSS_UP_ZERO', patterns.columns)
            self.assertIn('TRIX_CROSS_DOWN_ZERO', patterns.columns)
    
    def test_trix_ema_calculation(self):
        """测试TRIX的EMA计算"""
        # 测试内部EMA计算方法
        test_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ema_result = self.indicator._ema(test_series, 3)
        
        # 验证EMA结果
        self.assertEqual(len(ema_result), len(test_series))
        self.assertEqual(ema_result[0], test_series[0])  # 第一个值应该相等
        
        # 验证EMA是递增的（对于递增序列）
        self.assertTrue(all(ema_result[i] <= ema_result[i+1] for i in range(len(ema_result)-1)))
    
    def test_trix_sma_calculation(self):
        """测试TRIX的SMA计算"""
        # 测试内部SMA计算方法
        test_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        sma_result = self.indicator.sma(test_series, 3)
        
        # 验证SMA结果
        self.assertEqual(len(sma_result), len(test_series))
        
        # 验证SMA计算正确性（最后几个值）
        expected_last = np.mean(test_series[-3:])  # 最后3个值的平均
        self.assertAlmostEqual(sma_result[-1], expected_last, places=5)
    
    def test_trix_trend_detection(self):
        """测试TRIX趋势检测"""
        # 创建明显的上升趋势数据
        uptrend_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 50}
        ])
        
        patterns = self.indicator.get_patterns(uptrend_data)
        
        # 验证趋势形态
        if not patterns.empty:
            self.assertIn('TRIX_RISING', patterns.columns)
            self.assertIn('TRIX_FALLING', patterns.columns)
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('TRIX', result.columns)
        self.assertIn('MATRIX', result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_trix_register_patterns(self):
        """测试TRIX形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_trix_edge_cases(self):
        """测试TRIX边界情况"""
        # 测试价格不变的情况
        flat_data = self.data.copy()
        flat_data['close'] = 100.0  # 所有价格相同
        
        result = self.indicator.calculate(flat_data)
        
        # TRIX应该接近0（价格不变时）
        trix_values = result['TRIX'].dropna()
        if len(trix_values) > 0:
            # 价格不变时TRIX应该接近0
            self.assertTrue(all(abs(v) < 1.0 for v in trix_values), 
                           "价格不变时TRIX应该接近0")
    
    def test_trix_compute_method(self):
        """测试TRIX的compute方法"""
        result = self.indicator.compute(self.data)
        
        # 验证compute方法返回正确的列名
        self.assertIn('trix', result.columns)
        self.assertIn('signal', result.columns)
        
        # 验证这些列与原始列的对应关系
        self.assertTrue((result['trix'] == result['TRIX']).all())
        self.assertTrue((result['signal'] == result['MATRIX']).all())


if __name__ == '__main__':
    unittest.main()
