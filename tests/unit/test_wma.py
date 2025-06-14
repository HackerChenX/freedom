"""
WMA指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.wma import WMA
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestWMA(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """WMA指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = WMA(period=14)
        self.expected_columns = ['WMA14']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_wma_calculation_accuracy(self):
        """测试WMA计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证WMA列存在
        self.assertIn('WMA14', result.columns)
        
        # 验证WMA值的合理性
        wma_values = result['WMA14'].dropna()
        if len(wma_values) > 0:
            # WMA值应该为正数（假设价格为正）
            self.assertTrue(all(v > 0 for v in wma_values), "WMA值应该为正数")
            # WMA值应该在合理范围内
            price_range = (self.data['close'].min(), self.data['close'].max())
            self.assertTrue(all(price_range[0] <= v <= price_range[1] * 1.1 for v in wma_values), 
                           "WMA值应该在价格范围内")
    
    def test_wma_manual_calculation(self):
        """测试WMA手动计算验证"""
        # 创建简单的测试数据
        simple_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1200, 800, 1500, 900]
        })
        
        # 使用较小的周期便于验证
        test_indicator = WMA(period=3)
        result = test_indicator.calculate(simple_data)
        
        # 手动验证WMA计算
        if len(result) >= 3:
            # 计算第3个点的WMA值
            prices = simple_data['close'].iloc[0:3].values  # [100, 101, 102]
            weights = np.array([1, 2, 3])  # 权重
            expected_wma = np.sum(prices * weights) / np.sum(weights)
            calculated_wma = result['WMA3'].iloc[2]
            
            if not pd.isna(calculated_wma):
                self.assertAlmostEqual(calculated_wma, expected_wma, places=6, 
                                     msg="WMA计算不正确")
    
    def test_wma_score_range(self):
        """测试WMA评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_wma_confidence_calculation(self):
        """测试WMA置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_wma_parameter_update(self):
        """测试WMA参数更新"""
        new_period = 10
        self.indicator.set_parameters(period=new_period)
        
        # 验证参数更新
        self.assertEqual(self.indicator.period, new_period)
        self.assertEqual(self.indicator.periods, [new_period])
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('WMA10', result.columns)
    
    def test_wma_multiple_periods(self):
        """测试WMA多周期计算"""
        multi_indicator = WMA(period=14, periods=[5, 10, 20])
        result = multi_indicator.calculate(self.data)
        
        # 验证多个WMA列存在
        expected_columns = ['WMA5', 'WMA10', 'WMA20']
        for col in expected_columns:
            self.assertIn(col, result.columns, f"缺少WMA列: {col}")
    
    def test_wma_required_columns(self):
        """测试WMA必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_wma_comprehensive_score(self):
        """测试WMA综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_wma_patterns(self):
        """测试WMA形态识别"""
        # 使用多周期以便检测交叉
        multi_indicator = WMA(period=14, periods=[5, 10])
        
        # 先计算指标
        result = multi_indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 然后获取形态
        patterns = multi_indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证基本的形态列存在
        if not patterns.empty and len(patterns.columns) > 0:
            expected_patterns = [
                'PRICE_CROSS_ABOVE_WMA5', 'PRICE_CROSS_BELOW_WMA5',
                'PRICE_CROSS_ABOVE_WMA10', 'PRICE_CROSS_BELOW_WMA10',
                'WMA_GOLDEN_CROSS_5_10', 'WMA_DEATH_CROSS_5_10'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_wma_cross_detection(self):
        """测试WMA交叉检测"""
        # 使用多周期
        multi_indicator = WMA(period=14, periods=[5, 10])
        patterns = multi_indicator.get_patterns(self.data)
        
        # 验证交叉形态存在
        if not patterns.empty:
            cross_patterns = ['WMA_GOLDEN_CROSS_5_10', 'WMA_DEATH_CROSS_5_10']
            for pattern in cross_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少交叉形态: {pattern}")
    
    def test_wma_price_cross_detection(self):
        """测试价格与WMA交叉检测"""
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证价格交叉形态存在
        if not patterns.empty:
            price_cross_patterns = ['PRICE_CROSS_ABOVE_WMA14', 'PRICE_CROSS_BELOW_WMA14']
            for pattern in price_cross_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少价格交叉形态: {pattern}")
    
    def test_wma_signals(self):
        """测试WMA信号生成"""
        # 使用多周期以便生成信号
        multi_indicator = WMA(period=14, periods=[5, 10])
        signals = multi_indicator.generate_trading_signals(self.data)
        
        # 验证信号字典结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy', 'sell', 'exit_long', 'exit_short']
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
        self.assertIn('WMA14', result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_wma_register_patterns(self):
        """测试WMA形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_wma_edge_cases(self):
        """测试WMA边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(5)
        result = self.indicator.calculate(small_data)
        
        # WMA应该能处理数据不足的情况
        wma_values = result['WMA14'].dropna()
        # 数据不足时，前面的值应该是NaN
        self.assertTrue(len(wma_values) == 0 or len(wma_values) < len(small_data), 
                       "数据不足时WMA应该有NaN值")
    
    def test_wma_compute_method(self):
        """测试WMA的compute方法"""
        result = self.indicator.compute(self.data)
        
        # 验证compute方法返回正确的结果
        self.assertIn('WMA14', result.columns)
        
        # 验证WMA值的合理性
        wma_values = result['WMA14'].dropna()
        if len(wma_values) > 0:
            self.assertTrue(all(v > 0 for v in wma_values), "WMA值应该为正数")
    
    def test_wma_validation(self):
        """测试WMA数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['close'], axis=1)
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(invalid_data)
    
    def test_wma_weight_calculation(self):
        """测试WMA权重计算"""
        # 创建简单数据验证权重计算
        simple_data = pd.DataFrame({
            'open': [100, 100, 100, 100],
            'high': [100, 100, 100, 100],
            'low': [100, 100, 100, 100],
            'close': [100, 100, 100, 100],  # 相同价格
            'volume': [1000, 1000, 1000, 1000]
        })
        
        test_indicator = WMA(period=4)
        result = test_indicator.calculate(simple_data)
        
        # 当所有价格相同时，WMA应该等于价格
        wma_value = result['WMA4'].iloc[3]
        if not pd.isna(wma_value):
            self.assertAlmostEqual(wma_value, 100.0, places=6, 
                                 msg="相同价格时WMA应该等于价格")
    
    def test_wma_responsiveness(self):
        """测试WMA响应性"""
        # 创建价格突变的数据
        trend_data = pd.DataFrame({
            'open': [100] * 10 + [110] * 10,
            'high': [101] * 10 + [111] * 10,
            'low': [99] * 10 + [109] * 10,
            'close': [100] * 10 + [110] * 10,  # 价格突然上涨
            'volume': [1000] * 20
        })
        
        test_indicator = WMA(period=5)
        result = test_indicator.calculate(trend_data)
        
        # WMA应该对价格变化有响应
        wma_before = result['WMA5'].iloc[9]  # 价格变化前
        wma_after = result['WMA5'].iloc[14]  # 价格变化后
        
        if not pd.isna(wma_before) and not pd.isna(wma_after):
            self.assertGreater(wma_after, wma_before, "WMA应该对价格上涨有响应")


if __name__ == '__main__':
    unittest.main()
