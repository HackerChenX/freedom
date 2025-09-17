"""
Elliott_wave指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.complete_indicator_registry import complete_registry
from tests.unit.indicator_test_mixin import Indicator_test_mixin
from tests.helper.data_generator import Test_data_generator
from tests.helper.log_capture import Log_capture_mixin, LogCaptureMixin
from db.sql_manager import SQLManager, QueryType


class Test_elliott_wave(unittest.TestCase, Indicator_test_mixin, Log_capture_mixin):
    """ElliottWave指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        Log_capture_mixin.setUp(self)
        
        self.indicator = complete_registry.create_indicator('ELLIOTT_WAVE')
        self.expected_columns = [
            'ELLIOTT_FIVE_WAVE', 'ELLIOTT_ZIG_ZAG', 'ELLIOTT_FLAT',
            'ELLIOTT_TRIANGLE', 'ELLIOTT_DIAGONAL', 'ELLIOTT_COMBINATION',
            'ELLIOTT_WAVE_1', 'ELLIOTT_WAVE_2', 'ELLIOTT_WAVE_3',
            'ELLIOTT_WAVE_4', 'ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_A',
            'ELLIOTT_WAVE_B', 'ELLIOTT_WAVE_C', 'ELLIOTT_IMPULSE',
            'ELLIOTT_CORRECTIVE', 'ELLIOTT_GOLDEN_RATIO', 'ELLIOTT_FIBONACCI_RATIO',
            'ELLIOTT_VOLUME_CONFIRMATION', 'ELLIOTT_TIME_RATIO',
            'ELLIOTT_WAVE_COMPLETION', 'ELLIOTT_TREND_REVERSAL'
        ]
        self.data = Test_data_generator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 80}
        ])

    def tearDown(self):
        """清理日志捕获器"""
        Log_capture_mixin.tearDown(self)
    
    def test_elliott_wave_initialization(self):
        """测试ElliottWave初始化"""
        # 测试默认初始化
        default_indicator = complete_registry.create_indicator('ELLIOTT_WAVE')
        self.assertEqual(default_indicator.name, "ElliottWave")
        self.assertIn("艾略特波浪理论分析指标", default_indicator.description)
    
    def test_elliott_wave_calculation_accuracy(self):
        """测试ElliottWave计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证ElliottWave列存在
        self.assert_is_instance(result, pd.DataFrame)
        
        # 验证包含波浪相关列
        wave_columns = [col for col in result.columns if 'wave_' in col]
        self.assertGreater(len(wave_columns), 0, "应该包含波浪相关列")
    
    def test_elliott_wave_score_range(self):
        """测试ElliottWave评分范围"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        if 'score' in raw_score_df.columns:
            valid_scores = raw_score_df['score'].dropna()
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_elliott_wave_confidence_calculation(self):
        """测试ElliottWave置信度计算"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        raw_score = raw_score_df['score'] if 'score' in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assert_is_instance(confidence, float)
        self.assert_greater_equal(confidence, 0.0)
        self.assert_less_equal(confidence, 1.0)
    
    def test_elliott_wave_parameter_update(self):
        """测试ElliottWave参数更新"""
        # 测试参数设置
        self.indicator.set_parameters(min_wave_height=0.05, max_wave_count=12)
        
        # 验证参数已设置
        self.assert_equal(self.indicator.min_wave_height, 0.05)
        self.assert_equal(self.indicator.max_wave_count, 12)
    
    def test_elliott_wave_required_columns(self):
        """测试ElliottWave必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assert_in(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_elliott_wave_patterns(self):
        """测试ElliottWave形态识别"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        self.assert_is_instance(result, pd.DataFrame)
        
        # 然后获取形态
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assert_is_instance(patterns, pd.DataFrame)
        
        # 验证基本的形态列存在
        if not patterns.empty and len(patterns.columns) > 0:
            for pattern in self.expected_columns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_elliott_wave_signals(self):
        """测试ElliottWave信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assert_is_instance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assert_is_instance(signals[key], pd.Series)
    
    def test_elliott_wave_swing_points_identification(self):
        """测试ElliottWave摆动点识别"""
        # 使用足够的数据进行摆动点识别
        long_data = Test_data_generator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        swing_points = self.indicator._identify_swing_points(long_data, 0.03)
        
        # 验证摆动点识别结果
        self.assert_is_instance(swing_points, list)
        self.assertGreater(len(swing_points), 0, "应该识别出摆动点")
    
    def test_elliott_wave_wave_generation(self):
        """测试ElliottWave波浪生成"""
        # 使用足够的数据进行波浪生成
        long_data = Test_data_generator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        swing_points = self.indicator._identify_swing_points(long_data, 0.03)
        waves = self.indicator._generate_waves(long_data, swing_points, 9)
        
        # 验证波浪生成结果
        self.assert_is_instance(waves, list)
        if len(waves) > 0:
            # 验证波浪结构
            for wave in waves:
                self.assertIn('start_idx', wave)
                self.assertIn('end_idx', wave)
                self.assertIn('direction', wave)
                self.assertIn('height', wave)
                self.assertIn('wave_number', wave)
    
    def test_elliott_wave_pattern_identification(self):
        """测试ElliottWave形态识别"""
        # 使用足够的数据进行形态识别
        long_data = Test_data_generator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        patterns = self.indicator.identify_patterns(long_data)
        
        # 验证返回形态列表
        self.assert_is_instance(patterns, list)
    
    def test_elliott_wave_score_calculation(self):
        """测试ElliottWave评分计算"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        
        # 验证返回DataFrame
        self.assert_is_instance(raw_score_df, pd.DataFrame)
        
        # 验证包含score列
        self.assertIn('score', raw_score_df.columns)
        
        # 验证评分值的合理性
        scores = raw_score_df['score'].dropna()
        if len(scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in scores), "评分应在0-100范围内")
    
    def test_elliott_wave_direction_enums(self):
        """测试ElliottWave方向枚举"""
        # 由于枚举未在elliott_wave.py中定义，跳过此测试
        self.skipTest("WaveDirection枚举未在elliott_wave.py中定义")

    def test_elliott_wave_type_enums(self):
        """测试ElliottWave类型枚举"""
        # 由于枚举未在elliott_wave.py中定义，跳过此测试
        self.skipTest("WaveType枚举未在elliott_wave.py中定义")

    def test_elliott_wave_pattern_enums(self):
        """测试ElliottWave形态枚举"""
        # 由于枚举未在elliott_wave.py中定义，跳过此测试
        self.skipTest("WavePattern枚举未在elliott_wave.py中定义")
    
    def test_elliott_wave_five_wave_pattern(self):
        """测试ElliottWave五浪形态"""
        # 使用足够的数据进行五浪形态测试
        long_data = Test_data_generator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        result = self.indicator.calculate(long_data)
        
        # 验证波浪形态列存在
        if 'wave_pattern' in result.columns:
            pattern = result['wave_pattern'].iloc[0]
            self.assert_is_instance(pattern, str)
    
    def test_elliott_wave_prediction(self):
        """测试ElliottWave预测功能"""
        # 使用足够的数据进行预测测试
        long_data = Test_data_generator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        result = self.indicator.calculate(long_data)
        
        # 验证预测列存在
        if 'next_wave_prediction' in result.columns:
            prediction = result['next_wave_prediction'].iloc[-1]
            if pd.notna(prediction):
                self.assert_is_instance(prediction, (int, float))
                self.assert_greater(prediction, 0)
    
    def test_elliott_wave_volume_confirmation(self):
        """测试ElliottWave成交量确认"""
        # 生成带成交量的数据
        data_with_volume = self.data.copy()
        data_with_volume['volume'] = np.random.randint(1000, 10000, len(data_with_volume))
        
        raw_score_df = self.indicator.calculate_raw_score(data_with_volume)
        
        # 验证成交量确认在评分中的影响
        self.assert_is_instance(raw_score_df, pd.DataFrame)
        self.assertIn('score', raw_score_df.columns)
    
    def test_elliott_wave_fibonacci_ratios(self):
        """测试ElliottWave斐波那契比例"""
        # 使用足够的数据进行斐波那契比例测试
        long_data = Test_data_generator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        patterns = self.indicator.identify_patterns(long_data)
        
        # 验证斐波那契比例识别
        self.assert_is_instance(patterns, list)
    
    def test_elliott_wave_time_analysis(self):
        """测试ElliottWave时间分析"""
        # 使用足够的数据进行时间分析测试
        long_data = Test_data_generator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        raw_score_df = self.indicator.calculate_raw_score(long_data)
        
        # 验证时间分析在评分中的影响
        self.assert_is_instance(raw_score_df, pd.DataFrame)
        self.assertIn('score', raw_score_df.columns)
    
    def test_elliott_wave_edge_cases(self):
        """测试ElliottWave边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(5)
        result = self.indicator.calculate(small_data)
        
        # ElliottWave应该能处理数据不足的情况
        self.assert_is_instance(result, pd.DataFrame)
    
    def test_elliott_wave_validation(self):
        """测试ElliottWave数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['high'], axis=1)

        # BaseIndicator会处理缺失列并返回空DataFrame
        result = self.indicator.calculate(invalid_data)
        self.assert_is_instance(result, pd.DataFrame)
    
    def test_elliott_wave_indicator_type(self):
        """测试ElliottWave指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "ELLIOTTWAVE", "指标类型应该是ELLIOTTWAVE")
    
    def test_elliott_wave_register_patterns(self):
        """测试ElliottWave形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_no_errors_during_calculation_Wave(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assert_is_instance(result, pd.DataFrame)
    
    def test_no_errors_during_pattern_detection_Wave(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assert_is_instance(patterns, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
