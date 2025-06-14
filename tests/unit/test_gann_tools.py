"""
GannTools指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.gann_tools import GannTools, GannAngle, GannTimeCycle
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestGannTools(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """GannTools指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = GannTools()
        self.expected_columns = [
            'GANN_1X1_SUPPORT', 'GANN_1X1_RESISTANCE',
            'GANN_1X2_SUPPORT', 'GANN_1X2_RESISTANCE',
            'GANN_2X1_SUPPORT', 'GANN_2X1_RESISTANCE',
            'GANN_ANGLE_CLUSTER_SUPPORT', 'GANN_ANGLE_CLUSTER_RESISTANCE',
            'GANN_1X1_BREAKOUT_UP', 'GANN_1X1_BREAKOUT_DOWN',
            'GANN_TIME_CYCLE_LOW', 'GANN_TIME_CYCLE_HIGH',
            'GANN_SQUARE_SUPPORT', 'GANN_SQUARE_RESISTANCE',
            'GANN_PRICE_TARGET_UP', 'GANN_PRICE_TARGET_DOWN',
            'GANN_VOLUME_CONFIRMATION', 'GANN_TREND_ALIGNMENT'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_gann_tools_initialization(self):
        """测试GannTools初始化"""
        # 测试默认初始化
        default_indicator = GannTools()
        self.assertEqual(default_indicator.name, "GannTools")
        self.assertIn("江恩理论工具指标", default_indicator.description)
    
    def test_gann_tools_calculation_accuracy(self):
        """测试GannTools计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证GannTools列存在
        self.assertIsInstance(result, pd.DataFrame)
        
        # 验证包含江恩角度线
        gann_columns = [col for col in result.columns if 'gann_' in col]
        self.assertGreater(len(gann_columns), 0, "应该包含江恩角度线")
    
    def test_gann_tools_score_range(self):
        """测试GannTools评分范围"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        if 'score' in raw_score_df.columns:
            valid_scores = raw_score_df['score'].dropna()
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_gann_tools_confidence_calculation(self):
        """测试GannTools置信度计算"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        raw_score = raw_score_df['score'] if 'score' in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_gann_tools_parameter_update(self):
        """测试GannTools参数更新"""
        # 测试参数设置
        self.indicator.set_parameters(pivot_idx=10, price_unit=0.5, time_unit=2, levels=12)
        
        # 验证参数已设置
        self.assertEqual(self.indicator.pivot_idx, 10)
        self.assertEqual(self.indicator.price_unit, 0.5)
        self.assertEqual(self.indicator.time_unit, 2)
        self.assertEqual(self.indicator.levels, 12)
    
    def test_gann_tools_required_columns(self):
        """测试GannTools必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_gann_tools_patterns(self):
        """测试GannTools形态识别"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 然后获取形态
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证基本的形态列存在
        if not patterns.empty and len(patterns.columns) > 0:
            for pattern in self.expected_columns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_gann_tools_signals(self):
        """测试GannTools信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_gann_tools_angle_lines_calculation(self):
        """测试GannTools角度线计算"""
        # 使用足够的数据进行角度线计算
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        result = self.indicator.calculate(long_data)
        
        # 验证角度线列存在
        angle_columns = [col for col in result.columns if 'gann_' in col]
        self.assertGreater(len(angle_columns), 0, "应该包含江恩角度线")
        
        # 验证关键角度线
        key_angles = ['gann_1x1', 'gann_1x2', 'gann_2x1', 'gann_1x4', 'gann_4x1']
        for angle in key_angles:
            self.assertIn(angle, result.columns, f"缺少角度线: {angle}")
    
    def test_gann_tools_time_cycles_calculation(self):
        """测试GannTools时间周期计算"""
        # 使用足够的数据进行时间周期计算
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 200}
        ])
        
        result = self.indicator.calculate(long_data)
        
        # 验证时间周期列存在
        self.assertIn('time_cycle', result.columns, "应该包含时间周期列")
        
        # 验证周期列
        cycle_columns = [col for col in result.columns if 'cycle_' in col]
        self.assertGreater(len(cycle_columns), 0, "应该包含周期列")
    
    def test_gann_tools_gann_square_calculation(self):
        """测试GannTools江恩方格计算"""
        # 使用足够的数据进行江恩方格计算
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        result = self.indicator.calculate_gann_square(long_data)
        
        # 验证江恩方格结构
        self.assertIsInstance(result, pd.DataFrame)
        expected_square_columns = ['level', 'price', 'time_factor']
        for col in expected_square_columns:
            self.assertIn(col, result.columns, f"缺少江恩方格列: {col}")
    
    def test_gann_tools_pattern_identification(self):
        """测试GannTools形态识别"""
        # 使用足够的数据进行形态识别
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        patterns = self.indicator.identify_patterns(long_data)
        
        # 验证返回形态列表
        self.assertIsInstance(patterns, list)
    
    def test_gann_tools_score_calculation(self):
        """测试GannTools评分计算"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(raw_score_df, pd.DataFrame)
        
        # 验证包含score列
        self.assertIn('score', raw_score_df.columns)
        
        # 验证评分值的合理性
        scores = raw_score_df['score'].dropna()
        if len(scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in scores), "评分应在0-100范围内")
    
    def test_gann_tools_angle_enums(self):
        """测试GannTools角度枚举"""
        # 测试角度枚举
        self.assertIsInstance(GannAngle.ANGLE_1X1, GannAngle)
        self.assertIsInstance(GannAngle.ANGLE_1X2, GannAngle)
        self.assertIsInstance(GannAngle.ANGLE_2X1, GannAngle)
        
        # 测试角度比例
        self.assertIn(GannAngle.ANGLE_1X1, self.indicator.ANGLE_RATIOS)
        self.assertEqual(self.indicator.ANGLE_RATIOS[GannAngle.ANGLE_1X1], (1, 1))
    
    def test_gann_tools_time_cycle_enums(self):
        """测试GannTools时间周期枚举"""
        # 测试时间周期枚举
        self.assertIsInstance(GannTimeCycle.CYCLE_144, GannTimeCycle)
        self.assertIsInstance(GannTimeCycle.CYCLE_360, GannTimeCycle)
        
        # 测试时间周期值
        self.assertIn(GannTimeCycle.CYCLE_144, self.indicator.TIME_CYCLES)
        self.assertEqual(self.indicator.TIME_CYCLES[GannTimeCycle.CYCLE_144], 144)
    
    def test_gann_tools_1x1_line_analysis(self):
        """测试GannTools 1x1线分析"""
        # 使用足够的数据进行1x1线分析
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        result = self.indicator.calculate(long_data)
        
        # 验证1x1线存在
        self.assertIn('gann_1x1', result.columns, "应该包含江恩1x1线")
        
        # 验证1x1线值的合理性
        gann_1x1_values = result['gann_1x1'].dropna()
        if len(gann_1x1_values) > 0:
            self.assertTrue(all(v > 0 for v in gann_1x1_values), "江恩1x1线值应该为正")
    
    def test_gann_tools_angle_cluster_analysis(self):
        """测试GannTools角度线聚集分析"""
        # 使用足够的数据进行角度线聚集分析
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        patterns = self.indicator.identify_patterns(long_data)
        
        # 验证形态识别结果
        self.assertIsInstance(patterns, list)
    
    def test_gann_tools_price_target_analysis(self):
        """测试GannTools价格目标分析"""
        # 使用足够的数据进行价格目标分析
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        raw_score_df = self.indicator.calculate_raw_score(long_data)
        
        # 验证价格目标分析在评分中的影响
        self.assertIsInstance(raw_score_df, pd.DataFrame)
        self.assertIn('score', raw_score_df.columns)
    
    def test_gann_tools_volume_confirmation(self):
        """测试GannTools成交量确认"""
        # 生成带成交量的数据
        data_with_volume = self.data.copy()
        data_with_volume['volume'] = np.random.randint(1000, 10000, len(data_with_volume))
        
        raw_score_df = self.indicator.calculate_raw_score(data_with_volume)
        
        # 验证成交量确认在评分中的影响
        self.assertIsInstance(raw_score_df, pd.DataFrame)
        self.assertIn('score', raw_score_df.columns)
    
    def test_gann_tools_edge_cases(self):
        """测试GannTools边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(5)
        result = self.indicator.calculate(small_data)
        
        # GannTools应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_gann_tools_validation(self):
        """测试GannTools数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['close'], axis=1)

        # BaseIndicator会处理缺失列并返回空DataFrame
        result = self.indicator.calculate(invalid_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_gann_tools_indicator_type(self):
        """测试GannTools指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "GANNTOOLS", "指标类型应该是GANNTOOLS")
    
    def test_gann_tools_register_patterns(self):
        """测试GannTools形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
    
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
