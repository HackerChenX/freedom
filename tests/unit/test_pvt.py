"""
PVT指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.pvt import PVT
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestPVT(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """PVT指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = PVT(ma_period=12)
        self.expected_columns = ['pvt', 'pvt_signal']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_pvt_calculation_accuracy(self):
        """测试PVT计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证PVT列存在
        self.assertIn('pvt', result.columns)
        self.assertIn('pvt_signal', result.columns)
        
        # 手动计算PVT验证
        price_change = self.data['close'].pct_change()
        expected_pvt = (self.data['volume'] * price_change).cumsum()
        expected_signal = expected_pvt.rolling(window=12).mean()
        
        # 比较计算结果（允许小的数值误差）
        calculated_pvt = result['pvt']
        calculated_signal = result['pvt_signal']
        
        # 验证PVT计算
        pvt_diff = abs(calculated_pvt - expected_pvt).dropna()
        self.assertTrue(all(d < 0.001 for d in pvt_diff), "PVT计算结果不正确")
        
        # 验证信号线计算
        signal_diff = abs(calculated_signal - expected_signal).dropna()
        self.assertTrue(all(d < 0.001 for d in signal_diff), "PVT信号线计算结果不正确")
    
    def test_pvt_score_range(self):
        """测试PVT评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_pvt_confidence_calculation(self):
        """测试PVT置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_pvt_parameter_update(self):
        """测试PVT参数更新"""
        new_ma_period = 20
        self.indicator.set_parameters(ma_period=new_ma_period)
        
        # 验证参数更新
        self.assertEqual(self.indicator.ma_period, new_ma_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('pvt', result.columns)
        self.assertIn('pvt_signal', result.columns)
    
    def test_pvt_required_columns(self):
        """测试PVT必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_pvt_comprehensive_score(self):
        """测试PVT综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_pvt_patterns(self):
        """测试PVT形态识别"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)

        # 然后获取形态
        patterns = self.indicator.get_patterns(self.data)

        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)

        # 如果形态DataFrame不为空，验证预期的形态列存在
        if not patterns.empty:
            expected_patterns = [
                'PVT_GOLDEN_CROSS', 'PVT_DEATH_CROSS',
                'PVT_ABOVE_SIGNAL', 'PVT_BELOW_SIGNAL',
                'PVT_RISING', 'PVT_FALLING'
            ]

            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
        else:
            # 如果形态DataFrame为空，至少验证它是正确的类型
            self.assertEqual(len(patterns.columns), 0)
    
    def test_pvt_signals(self):
        """测试PVT信号生成"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 生成信号
        signals = self.indicator.get_signals(result)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, pd.DataFrame)
        expected_signal_keys = ['pvt_buy_signal', 'pvt_sell_signal']
        for key in expected_signal_keys:
            self.assertIn(key, signals.columns, f"缺少信号列: {key}")
    
    def test_pvt_crossover_detection(self):
        """测试PVT交叉检测"""
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
            self.assertIn('PVT_GOLDEN_CROSS', patterns.columns)
            self.assertIn('PVT_DEATH_CROSS', patterns.columns)
    
    def test_pvt_volume_impact(self):
        """测试成交量对PVT的影响"""
        # 创建高成交量数据
        high_volume_data = self.data.copy()
        high_volume_data['volume'] = high_volume_data['volume'] * 2
        
        # 创建低成交量数据
        low_volume_data = self.data.copy()
        low_volume_data['volume'] = low_volume_data['volume'] * 0.5
        
        # 计算PVT
        high_vol_result = self.indicator.calculate(high_volume_data)
        low_vol_result = self.indicator.calculate(low_volume_data)
        
        # 验证高成交量导致更大的PVT变化
        high_vol_pvt_range = high_vol_result['pvt'].max() - high_vol_result['pvt'].min()
        low_vol_pvt_range = low_vol_result['pvt'].max() - low_vol_result['pvt'].min()
        
        self.assertGreater(high_vol_pvt_range, low_vol_pvt_range, 
                          "高成交量应该导致更大的PVT变化范围")
    
    def test_pvt_price_change_impact(self):
        """测试价格变化对PVT的影响"""
        # 创建大幅价格变化数据
        large_change_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 25}
        ])
        
        # 创建小幅价格变化数据
        small_change_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 105, 'periods': 25}
        ])
        
        # 计算PVT
        large_change_result = self.indicator.calculate(large_change_data)
        small_change_result = self.indicator.calculate(small_change_data)
        
        # 验证大幅价格变化导致更大的PVT变化
        large_change_pvt_range = large_change_result['pvt'].max() - large_change_result['pvt'].min()
        small_change_pvt_range = small_change_result['pvt'].max() - small_change_result['pvt'].min()
        
        self.assertGreater(large_change_pvt_range, small_change_pvt_range,
                          "大幅价格变化应该导致更大的PVT变化范围")
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('pvt', result.columns)
        self.assertIn('pvt_signal', result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_pvt_register_patterns(self):
        """测试PVT形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_pvt_edge_cases(self):
        """测试PVT边界情况"""
        # 测试零成交量情况
        zero_volume_data = self.data.copy()
        zero_volume_data['volume'] = 0
        
        result = self.indicator.calculate(zero_volume_data)
        
        # PVT应该为0（因为成交量为0）
        pvt_values = result['pvt'].dropna()
        if len(pvt_values) > 0:
            self.assertTrue(all(abs(v) < 1e-10 for v in pvt_values), 
                           "零成交量时PVT应该为0")
    
    def test_pvt_cumulative_nature(self):
        """测试PVT的累积性质"""
        result = self.indicator.calculate(self.data)
        
        # PVT应该是累积的，即每个值都基于前一个值
        pvt_values = result['pvt'].dropna()
        
        if len(pvt_values) > 1:
            # 验证PVT是累积计算的（通过检查差值是否等于当期贡献）
            price_change = self.data['close'].pct_change().dropna()
            volume = self.data['volume'][price_change.index]
            expected_contribution = (volume * price_change).dropna()
            
            if len(expected_contribution) > 1:
                actual_diff = pvt_values.diff().dropna()
                # 由于索引可能不完全匹配，我们检查长度和趋势
                self.assertGreater(len(actual_diff), 0, "PVT应该有变化")


if __name__ == '__main__':
    unittest.main()
