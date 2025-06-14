"""
VOL指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.vol import VOL
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestVOL(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """VOL指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = VOL(period=14, enable_cycles_analysis=True, enable_standardization=True)
        self.expected_columns = ['vol', 'vol_ma5', 'vol_ma10', 'vol_ma20']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_vol_calculation_accuracy(self):
        """测试VOL计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证VOL列存在
        self.assertIn('vol', result.columns)
        self.assertIn('vol_ma5', result.columns)
        self.assertIn('vol_ma10', result.columns)
        self.assertIn('vol_ma20', result.columns)
        self.assertIn('vol_ratio', result.columns)
        
        # 验证VOL值的合理性
        vol_values = result['vol'].dropna()
        if len(vol_values) > 0:
            # VOL值应该为正数
            self.assertTrue(all(v >= 0 for v in vol_values), "VOL值应该为正数")
            # VOL值应该等于原始volume
            self.assertTrue((result['vol'] == result['volume']).all(), "VOL应该等于原始volume")
    
    def test_vol_ratio_calculation(self):
        """测试VOL比率计算"""
        result = self.indicator.calculate(self.data)
        
        # 手动计算vol_ratio验证
        expected_vol_ratio = result['volume'] / result['vol_ma5']
        calculated_vol_ratio = result['vol_ratio']
        
        # 比较计算结果（允许小的数值误差）
        diff = abs(calculated_vol_ratio - expected_vol_ratio).dropna()
        self.assertTrue(all(d < 0.001 for d in diff), "VOL比率计算不正确")
    
    def test_vol_score_range(self):
        """测试VOL评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_vol_confidence_calculation(self):
        """测试VOL置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_vol_parameter_update(self):
        """测试VOL参数更新"""
        new_period = 20
        self.indicator.set_parameters(period=new_period, enable_cycles_analysis=False)
        
        # 验证参数更新
        self.assertEqual(self.indicator.period, new_period)
        self.assertEqual(self.indicator.enable_cycles_analysis, False)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('vol', result.columns)
        self.assertIn('vol_ma5', result.columns)
    
    def test_vol_required_columns(self):
        """测试VOL必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_vol_comprehensive_score(self):
        """测试VOL综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_vol_patterns(self):
        """测试VOL形态识别"""
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
                'VOL_HIGH', 'VOL_LOW',
                'VOL_RISING', 'VOL_FALLING',
                'VOL_BREAKOUT_UP', 'VOL_BREAKOUT_DOWN'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_vol_extreme_values(self):
        """测试VOL极端值检测"""
        # 创建包含极端成交量的数据
        extreme_data = self.data.copy()
        # 创造一个极端放量的日子
        extreme_data.loc[extreme_data.index[25], 'volume'] = extreme_data['volume'].iloc[24] * 5
        
        patterns = self.indicator.get_patterns(extreme_data)
        
        # 验证极端形态存在
        if not patterns.empty:
            # 应该检测到某种VOL形态
            pattern_sum = patterns.sum().sum()
            self.assertGreater(pattern_sum, 0, "应该检测到VOL形态")
    
    def test_vol_ma_relationship(self):
        """测试VOL均线关系"""
        result = self.indicator.calculate(self.data)
        
        vol_ma5 = result['vol_ma5'].dropna()
        vol_ma10 = result['vol_ma10'].dropna()
        vol_ma20 = result['vol_ma20'].dropna()
        
        if len(vol_ma5) > 10 and len(vol_ma10) > 10 and len(vol_ma20) > 10:
            # 短期均线应该比长期均线更敏感（波动性更大）
            ma5_volatility = vol_ma5.std()
            ma20_volatility = vol_ma20.std()
            
            self.assertGreaterEqual(ma5_volatility, ma20_volatility * 0.5, 
                                  "短期均线应该比长期均线更敏感")
    
    def test_vol_signals(self):
        """测试VOL信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_vol_breakout_detection(self):
        """测试VOL突破检测"""
        # 创建包含突破的数据
        breakout_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 15},  # 下跌
            {'type': 'trend', 'start_price': 90, 'end_price': 120, 'periods': 15}   # 突破上涨
        ])
        
        # 在突破点增加成交量
        breakout_data.loc[breakout_data.index[15:20], 'volume'] *= 3
        
        patterns = self.indicator.get_patterns(breakout_data)
        
        # 验证突破形态存在
        if not patterns.empty:
            self.assertIn('VOL_BREAKOUT_UP', patterns.columns)
            self.assertIn('VOL_BREAKOUT_DOWN', patterns.columns)
    
    def test_vol_ma_cross_detection(self):
        """测试VOL均线交叉检测"""
        # 创建包含交叉的数据
        cross_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 30}
        ])
        
        patterns = self.indicator.get_patterns(cross_data)
        
        # 验证交叉形态存在
        if not patterns.empty:
            self.assertIn('VOL_GOLDEN_CROSS', patterns.columns)
            self.assertIn('VOL_DEATH_CROSS', patterns.columns)
    
    def test_vol_peak_trough_detection(self):
        """测试VOL峰谷检测"""
        # 创建足够长的数据以计算峰谷
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 130}
        ])
        
        patterns = self.indicator.get_patterns(long_data)
        
        # 验证峰谷形态存在
        if not patterns.empty:
            self.assertIn('VOL_PEAK', patterns.columns)
            self.assertIn('VOL_TROUGH', patterns.columns)
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('vol', result.columns)
        self.assertIn('vol_ma5', result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_vol_register_patterns(self):
        """测试VOL形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_vol_edge_cases(self):
        """测试VOL边界情况"""
        # 测试成交量为0的情况
        zero_vol_data = self.data.copy()
        zero_vol_data['volume'] = 0.0
        
        result = self.indicator.calculate(zero_vol_data)
        
        # VOL应该为0
        vol_values = result['vol'].dropna()
        if len(vol_values) > 0:
            self.assertTrue(all(v == 0 for v in vol_values), 
                           "成交量为0时VOL应该为0")
    
    def test_vol_compute_method(self):
        """测试VOL的compute方法"""
        result = self.indicator.compute(self.data)
        
        # 验证compute方法返回正确的结果
        self.assertIn('vol', result.columns)
        self.assertIn('vol_ma5', result.columns)
        
        # 验证VOL值的合理性
        vol_values = result['vol'].dropna()
        if len(vol_values) > 0:
            self.assertTrue(all(v >= 0 for v in vol_values), "VOL值应该为正数")
    
    def test_vol_standardization(self):
        """测试VOL标准化功能"""
        # 测试启用标准化
        indicator_with_std = VOL(enable_standardization=True)
        result_with_std = indicator_with_std.calculate(self.data)
        
        # 测试禁用标准化
        indicator_without_std = VOL(enable_standardization=False)
        result_without_std = indicator_without_std.calculate(self.data)
        
        # 验证两种模式都能正常计算
        self.assertIn('vol', result_with_std.columns)
        self.assertIn('vol', result_without_std.columns)
    
    def test_vol_cycles_analysis(self):
        """测试VOL周期分析功能"""
        # 创建足够长的数据以进行周期分析
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 70}
        ])
        
        # 测试启用周期分析
        indicator_with_cycles = VOL(enable_cycles_analysis=True)
        result_with_cycles = indicator_with_cycles.calculate(long_data)
        
        # 验证周期分析结果
        self.assertIn('vol', result_with_cycles.columns)
        
        # 如果数据足够长，应该有周期分析结果
        if len(long_data) >= 60:
            # 周期分析可能会添加额外的列
            self.assertTrue(len(result_with_cycles.columns) >= len(self.expected_columns))


if __name__ == '__main__':
    unittest.main()
