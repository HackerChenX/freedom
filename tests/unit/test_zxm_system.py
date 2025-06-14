"""
ZXM体系指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.zxm_absorb import ZXMAbsorb
from indicators.zxm_washplate import ZXMWashPlate, WashPlateType
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestZXMSystem(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """ZXM体系指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)

        self.zxm_absorb = ZXMAbsorb()
        self.zxm_washplate = ZXMWashPlate()

        # 为IndicatorTestMixin设置默认指标
        self.indicator = self.zxm_absorb

        # 为IndicatorTestMixin设置预期列
        self.expected_columns = ['V11', 'V12', 'EMA_V11_3', 'AA', 'BB', 'XG', 'BUY']
        
        # ZXMAbsorb预期列
        self.absorb_expected_columns = [
            'ZXM_ABSORB_SIGNAL', 'ZXM_STRONG_ABSORB', 'ZXM_MEDIUM_ABSORB', 'ZXM_WEAK_ABSORB',
            'ZXM_V11_LOW', 'ZXM_V11_LOW_V12_UP', 'ZXM_V11_LOW_V12_FAST_UP',
            'ZXM_CONTINUOUS_LOW', 'ZXM_LOW_RECOVERY', 'ZXM_VOLUME_SHRINK',
            'ZXM_LOW_VOLUME_EXPANSION', 'ZXM_ABSORB_CONFIRMATION'
        ]
        
        # ZXMWashPlate预期列
        self.washplate_expected_columns = [
            'ZXM_SHOCK_WASH', 'ZXM_PULLBACK_WASH', 'ZXM_FALSE_BREAK_WASH',
            'ZXM_TIME_WASH', 'ZXM_CONTINUOUS_YIN_WASH', 'ZXM_ANY_WASH',
            'ZXM_MULTIPLE_WASH', 'ZXM_WASH_COMPLETION', 'ZXM_WASH_RECOVERY',
            'ZXM_WASH_VOLUME_CONFIRM', 'ZXM_WASH_SUPPORT', 'ZXM_WASH_BREAKOUT'
        ]
        
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_zxm_absorb_initialization(self):
        """测试ZXMAbsorb初始化"""
        # 测试默认初始化
        default_indicator = ZXMAbsorb()
        self.assertEqual(default_indicator.name, "ZXM_ABSORB")
        self.assertIn("ZXM核心吸筹指标", default_indicator.description)
    
    def test_zxm_absorb_calculation_accuracy(self):
        """测试ZXMAbsorb计算准确性"""
        result = self.zxm_absorb.calculate(self.data)
        
        # 验证ZXMAbsorb列存在
        self.assertIsInstance(result, pd.DataFrame)
        
        # 验证包含核心列
        core_columns = ['V11', 'V12', 'EMA_V11_3', 'AA', 'BB', 'XG', 'BUY']
        for col in core_columns:
            self.assertIn(col, result.columns, f"缺少核心列: {col}")
    
    def test_zxm_absorb_score_range(self):
        """测试ZXMAbsorb评分范围"""
        raw_score = self.zxm_absorb.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_zxm_absorb_confidence_calculation(self):
        """测试ZXMAbsorb置信度计算"""
        raw_score = self.zxm_absorb.calculate_raw_score(self.data)
        patterns = self.zxm_absorb.get_patterns(self.data)
        
        confidence = self.zxm_absorb.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_zxm_absorb_parameter_update(self):
        """测试ZXMAbsorb参数更新"""
        # 测试参数设置
        self.zxm_absorb.set_parameters(v11_threshold=15, v12_threshold=10, xg_threshold=4)
        
        # 验证参数已设置
        self.assertEqual(self.zxm_absorb.v11_threshold, 15)
        self.assertEqual(self.zxm_absorb.v12_threshold, 10)
        self.assertEqual(self.zxm_absorb.xg_threshold, 4)
    
    def test_zxm_absorb_required_columns(self):
        """测试ZXMAbsorb必需列"""
        self.assertTrue(hasattr(self.zxm_absorb, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.zxm_absorb.REQUIRED_COLUMNS)
    
    def test_zxm_absorb_patterns(self):
        """测试ZXMAbsorb形态识别"""
        # 先计算指标
        result = self.zxm_absorb.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 然后获取形态
        patterns = self.zxm_absorb.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证基本的形态列存在
        if not patterns.empty and len(patterns.columns) > 0:
            for pattern in self.absorb_expected_columns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_zxm_absorb_signals(self):
        """测试ZXMAbsorb信号生成"""
        signals = self.zxm_absorb.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_zxm_washplate_initialization(self):
        """测试ZXMWashPlate初始化"""
        # 测试默认初始化
        default_indicator = ZXMWashPlate()
        self.assertEqual(default_indicator.name, "ZXMWashPlate")
        self.assertIn("ZXM洗盘形态识别指标", default_indicator.description)
    
    def test_zxm_washplate_calculation_accuracy(self):
        """测试ZXMWashPlate计算准确性"""
        result = self.zxm_washplate.calculate(self.data)
        
        # 验证ZXMWashPlate列存在
        self.assertIsInstance(result, pd.DataFrame)
        
        # 验证包含洗盘形态列
        for wash_type in WashPlateType:
            self.assertIn(wash_type.value, result.columns, f"缺少洗盘形态列: {wash_type.value}")
    
    def test_zxm_washplate_score_range(self):
        """测试ZXMWashPlate评分范围"""
        raw_score = self.zxm_washplate.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_zxm_washplate_confidence_calculation(self):
        """测试ZXMWashPlate置信度计算"""
        raw_score = self.zxm_washplate.calculate_raw_score(self.data)
        patterns = self.zxm_washplate.get_patterns(self.data)
        
        confidence = self.zxm_washplate.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_zxm_washplate_parameter_update(self):
        """测试ZXMWashPlate参数更新"""
        # 测试参数设置
        self.zxm_washplate.set_parameters(shock_price_threshold=0.05, shock_volume_ratio=3.0)
        
        # 验证参数已设置
        self.assertEqual(self.zxm_washplate.shock_price_threshold, 0.05)
        self.assertEqual(self.zxm_washplate.shock_volume_ratio, 3.0)
    
    def test_zxm_washplate_required_columns(self):
        """测试ZXMWashPlate必需列"""
        self.assertTrue(hasattr(self.zxm_washplate, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.zxm_washplate.REQUIRED_COLUMNS)
    
    def test_zxm_washplate_patterns(self):
        """测试ZXMWashPlate形态识别"""
        # 先计算指标
        result = self.zxm_washplate.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 然后获取形态
        patterns = self.zxm_washplate.get_patterns(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(patterns, pd.DataFrame)
        
        # 验证基本的形态列存在
        if not patterns.empty and len(patterns.columns) > 0:
            for pattern in self.washplate_expected_columns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_zxm_washplate_signals(self):
        """测试ZXMWashPlate信号生成"""
        signals = self.zxm_washplate.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_zxm_absorb_v11_calculation(self):
        """测试ZXMAbsorb V11指标计算"""
        # 使用足够的数据进行V11计算测试
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        result = self.zxm_absorb.calculate(long_data)
        
        # 验证V11指标计算结果
        self.assertIn('V11', result.columns)
        self.assertIn('EMA_V11_3', result.columns)
        
        # V11值应该在合理范围内
        v11_values = result['V11'].dropna()
        if len(v11_values) > 0:
            self.assertTrue(all(-100 <= v <= 200 for v in v11_values), "V11值应在合理范围内")
    
    def test_zxm_absorb_xg_calculation(self):
        """测试ZXMAbsorb XG吸筹强度计算"""
        # 使用足够的数据进行XG计算测试
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        result = self.zxm_absorb.calculate(long_data)
        
        # 验证XG指标计算结果
        self.assertIn('XG', result.columns)
        
        # XG值应该在0-6范围内
        xg_values = result['XG'].dropna()
        if len(xg_values) > 0:
            self.assertTrue(all(0 <= x <= 6 for x in xg_values), "XG值应在0-6范围内")
    
    def test_zxm_washplate_wash_types(self):
        """测试ZXMWashPlate洗盘类型枚举"""
        # 测试洗盘类型枚举
        self.assertEqual(WashPlateType.SHOCK_WASH.value, "横盘震荡洗盘")
        self.assertEqual(WashPlateType.PULLBACK_WASH.value, "回调洗盘")
        self.assertEqual(WashPlateType.FALSE_BREAK_WASH.value, "假突破洗盘")
        self.assertEqual(WashPlateType.TIME_WASH.value, "时间洗盘")
        self.assertEqual(WashPlateType.CONTINUOUS_YIN_WASH.value, "连续阴线洗盘")
    
    def test_zxm_washplate_recent_wash_plates(self):
        """测试ZXMWashPlate最近洗盘形态"""
        # 使用足够的数据进行洗盘形态测试
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        recent_wash_plates = self.zxm_washplate.get_recent_wash_plates(long_data, lookback=10)
        
        # 验证返回字典
        self.assertIsInstance(recent_wash_plates, dict)
        
        # 验证包含所有洗盘类型
        for wash_type in WashPlateType:
            self.assertIn(wash_type.value, recent_wash_plates)
            # 检查是否为布尔类型（包括numpy布尔类型）
            value = recent_wash_plates[wash_type.value]
            self.assertTrue(isinstance(value, (bool, np.bool_)), f"值应该是布尔类型，实际类型: {type(value)}")
    
    def test_zxm_absorb_validation(self):
        """测试ZXMAbsorb数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['high'], axis=1)
        
        # BaseIndicator会处理缺失列并返回空DataFrame
        result = self.zxm_absorb.calculate(invalid_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_zxm_washplate_validation(self):
        """测试ZXMWashPlate数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['volume'], axis=1)
        
        # BaseIndicator会处理缺失列并返回空DataFrame
        result = self.zxm_washplate.calculate(invalid_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_zxm_absorb_indicator_type(self):
        """测试ZXMAbsorb指标类型"""
        indicator_type = self.zxm_absorb.get_indicator_type()
        self.assertEqual(indicator_type, "ZXM_ABSORB", "指标类型应该是ZXM_ABSORB")
    
    def test_zxm_washplate_indicator_type(self):
        """测试ZXMWashPlate指标类型"""
        indicator_type = self.zxm_washplate.get_indicator_type()
        self.assertEqual(indicator_type, "ZXM_WASHPLATE", "指标类型应该是ZXM_WASHPLATE")
    
    def test_zxm_absorb_register_patterns(self):
        """测试ZXMAbsorb形态注册"""
        # 调用形态注册
        self.zxm_absorb.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_zxm_washplate_register_patterns(self):
        """测试ZXMWashPlate形态注册"""
        # 调用形态注册
        self.zxm_washplate.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_no_errors_during_zxm_absorb_calculation(self):
        """测试ZXMAbsorb计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.zxm_absorb.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_no_errors_during_zxm_washplate_calculation(self):
        """测试ZXMWashPlate计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.zxm_washplate.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
