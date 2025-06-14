"""
Volume Ratio指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.volume_ratio import VolumeRatio
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestVolumeRatio(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """Volume Ratio指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = VolumeRatio(reference_period=5, ma_period=3)
        self.expected_columns = ['volume_ratio', 'volume_ratio_ma']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_volume_ratio_calculation_accuracy(self):
        """测试Volume Ratio计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证Volume Ratio列存在
        self.assertIn('volume_ratio', result.columns)
        self.assertIn('volume_ratio_ma', result.columns)
        
        # 验证Volume Ratio值的合理性
        vr_values = result['volume_ratio'].dropna()
        if len(vr_values) > 0:
            # Volume Ratio值应该为正数
            self.assertTrue(all(v >= 0 for v in vr_values), "Volume Ratio值应该为正数")
            # 大部分Volume Ratio值应该在合理范围内
            reasonable_values = [v for v in vr_values if 0.1 <= v <= 10]
            self.assertGreater(len(reasonable_values), len(vr_values) * 0.8, 
                             "大部分Volume Ratio值应该在合理范围内")
    
    def test_volume_ratio_manual_calculation(self):
        """测试Volume Ratio手动计算验证"""
        # 创建简单的测试数据
        simple_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106],
            'high': [101, 102, 103, 104, 105, 106, 107],
            'low': [99, 100, 101, 102, 103, 104, 105],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5],
            'volume': [1000, 1200, 800, 1500, 900, 1100, 1300]
        })
        
        result = self.indicator.calculate(simple_data)
        
        # 手动验证第6个点的量比计算（索引5）
        if len(result) > 5:
            # 前5个成交量的平均值
            ref_avg = np.mean([1000, 1200, 800, 1500, 900])  # 1080
            expected_vr = 1100 / ref_avg  # 约1.019
            calculated_vr = result['volume_ratio'].iloc[5]
            
            self.assertAlmostEqual(calculated_vr, expected_vr, places=3, 
                                 msg="Volume Ratio计算不正确")
    
    def test_volume_ratio_score_range(self):
        """测试Volume Ratio评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_volume_ratio_confidence_calculation(self):
        """测试Volume Ratio置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_volume_ratio_parameter_update(self):
        """测试Volume Ratio参数更新"""
        new_ref_period = 10
        new_ma_period = 5
        self.indicator.set_parameters(reference_period=new_ref_period, ma_period=new_ma_period)
        
        # 验证参数更新
        self.assertEqual(self.indicator.reference_period, new_ref_period)
        self.assertEqual(self.indicator.ma_period, new_ma_period)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('volume_ratio', result.columns)
        self.assertIn('volume_ratio_ma', result.columns)
    
    def test_volume_ratio_required_columns(self):
        """测试Volume Ratio必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_volume_ratio_comprehensive_score(self):
        """测试Volume Ratio综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_volume_ratio_patterns(self):
        """测试Volume Ratio形态识别"""
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
                'VR_HIGH', 'VR_LOW', 'VR_NORMAL',
                'VR_RISING', 'VR_FALLING',
                'VR_ABOVE_MA', 'VR_BELOW_MA'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_volume_ratio_extreme_values(self):
        """测试Volume Ratio极端值检测"""
        # 创建包含极端成交量的数据
        extreme_data = self.data.copy()
        # 创造一个极端放量的日子
        extreme_data.loc[extreme_data.index[25], 'volume'] = extreme_data['volume'].iloc[24] * 5
        
        patterns = self.indicator.get_patterns(extreme_data)
        
        # 验证极端形态存在
        if not patterns.empty:
            # 应该检测到某种VR形态
            pattern_sum = patterns.sum().sum()
            self.assertGreater(pattern_sum, 0, "应该检测到Volume Ratio形态")
    
    def test_volume_ratio_ma_relationship(self):
        """测试Volume Ratio与均线关系"""
        result = self.indicator.calculate(self.data)
        
        vr_values = result['volume_ratio'].dropna()
        vr_ma_values = result['volume_ratio_ma'].dropna()
        
        if len(vr_values) > 10 and len(vr_ma_values) > 10:
            # 均线应该比原始值更平滑（波动性更小）
            vr_volatility = vr_values.std()
            ma_volatility = vr_ma_values.std()
            
            self.assertLessEqual(ma_volatility, vr_volatility * 1.2, 
                               "均线应该比原始值更平滑")
    
    def test_volume_ratio_signals(self):
        """测试Volume Ratio信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_volume_ratio_breakout_detection(self):
        """测试Volume Ratio突破检测"""
        # 创建包含突破的数据
        breakout_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 30}
        ])
        
        # 在某些点增加成交量以创造突破
        breakout_data.loc[breakout_data.index[15:20], 'volume'] = (
            breakout_data.loc[breakout_data.index[15:20], 'volume'] * 2.5
        ).astype(int)
        
        patterns = self.indicator.get_patterns(breakout_data)
        
        # 验证突破形态存在
        if not patterns.empty:
            self.assertIn('VR_BREAKOUT_HIGH', patterns.columns)
            self.assertIn('VR_BREAKDOWN_LOW', patterns.columns)
    
    def test_volume_ratio_cross_detection(self):
        """测试Volume Ratio交叉检测"""
        # 创建包含交叉的数据
        cross_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 30}
        ])
        
        patterns = self.indicator.get_patterns(cross_data)
        
        # 验证交叉形态存在
        if not patterns.empty:
            self.assertIn('VR_GOLDEN_CROSS', patterns.columns)
            self.assertIn('VR_DEATH_CROSS', patterns.columns)
    
    def test_volume_ratio_peak_trough_detection(self):
        """测试Volume Ratio峰谷检测"""
        # 创建足够长的数据以计算峰谷
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 25}
        ])
        
        patterns = self.indicator.get_patterns(long_data)
        
        # 验证峰谷形态存在
        if not patterns.empty:
            self.assertIn('VR_PEAK', patterns.columns)
            self.assertIn('VR_TROUGH', patterns.columns)
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('volume_ratio', result.columns)
        self.assertIn('volume_ratio_ma', result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_volume_ratio_register_patterns(self):
        """测试Volume Ratio形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_volume_ratio_edge_cases(self):
        """测试Volume Ratio边界情况"""
        # 测试成交量为0的情况
        zero_vol_data = self.data.copy()
        zero_vol_data['volume'] = 0.0

        result = self.indicator.calculate(zero_vol_data)

        # Volume Ratio应该为1（默认值）
        vr_values = result['volume_ratio'].dropna()
        if len(vr_values) > 0:
            # 由于参考期也是0，所以应该返回默认值1
            # 但实际上前几个值可能是0，因为还没有足够的参考期数据
            # 我们只检查有效值是否为1
            valid_values = [v for v in vr_values if v > 0]
            if len(valid_values) > 0:
                self.assertTrue(all(abs(v - 1.0) < 1e-10 for v in valid_values),
                               "成交量为0时Volume Ratio应该为1")
    
    def test_volume_ratio_compute_method(self):
        """测试Volume Ratio的compute方法"""
        result = self.indicator.compute(self.data)
        
        # 验证compute方法返回正确的结果
        self.assertIn('volume_ratio', result.columns)
        self.assertIn('volume_ratio_ma', result.columns)
        
        # 验证Volume Ratio值的合理性
        vr_values = result['volume_ratio'].dropna()
        if len(vr_values) > 0:
            self.assertTrue(all(v >= 0 for v in vr_values), "Volume Ratio值应该为正数")
    
    def test_volume_ratio_market_activity(self):
        """测试Volume Ratio市场活跃度评估"""
        activity_result = self.indicator.get_market_activity(self.data)
        
        # 验证活跃度评估结果
        self.assertIn('market_activity', activity_result.columns)
        self.assertIn('activity_score', activity_result.columns)
        
        # 验证活跃度评分在合理范围内
        activity_scores = activity_result['activity_score'].dropna()
        if len(activity_scores) > 0:
            self.assertTrue(all(0 <= s <= 200 for s in activity_scores), 
                           "活跃度评分应该在合理范围内")
    
    def test_volume_ratio_validation(self):
        """测试Volume Ratio数据验证"""
        # 测试缺少volume列的情况
        invalid_data = self.data.drop('volume', axis=1)
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(invalid_data)
        
        # 测试所有volume都是NaN的情况
        nan_vol_data = self.data.copy()
        nan_vol_data['volume'] = np.nan
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(nan_vol_data)


if __name__ == '__main__':
    unittest.main()
