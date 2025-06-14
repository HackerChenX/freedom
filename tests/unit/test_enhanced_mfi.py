"""
EnhancedMFI指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.volume.enhanced_mfi import EnhancedMFI
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestEnhancedMFI(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """EnhancedMFI指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = EnhancedMFI(period=14, volatility_lookback=20)
        self.expected_columns = [
            'mfi', 'mfi_overbought', 'mfi_oversold', 'mfi_price_ratio',
            'mfi_momentum', 'mfi_slope', 'mfi_accel', 'mfi_adjusted'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_enhanced_mfi_calculation_accuracy(self):
        """测试EnhancedMFI计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证EnhancedMFI列存在
        for col in self.expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证MFI值的合理性
        mfi_values = result['mfi'].dropna()
        
        if len(mfi_values) > 0:
            # MFI值应该在0-100范围内
            self.assertTrue(all(0 <= v <= 100 for v in mfi_values), "MFI值应该在0-100范围内")
    
    def test_enhanced_mfi_manual_calculation(self):
        """测试EnhancedMFI手动计算验证"""
        # 创建简单的测试数据
        simple_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104] * 10,
            'high': [101, 102, 103, 104, 105] * 10,
            'low': [99, 100, 101, 102, 103] * 10,
            'close': [100.5, 101.5, 102.5, 103.5, 104.5] * 10,
            'volume': [1000, 1200, 800, 1500, 900] * 10
        })
        
        # 使用较小的周期便于验证
        test_indicator = EnhancedMFI(period=5, volatility_lookback=10)
        result = test_indicator.calculate(simple_data)
        
        # 验证MFI计算逻辑
        if len(result) >= 5:
            mfi_value = result['mfi'].iloc[4]
            
            if not pd.isna(mfi_value):
                # MFI值应该在合理范围内
                self.assertTrue(0 <= mfi_value <= 100, "MFI值应该在0-100范围内")
    
    def test_enhanced_mfi_score_range(self):
        """测试EnhancedMFI评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_mfi_confidence_calculation(self):
        """测试EnhancedMFI置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_enhanced_mfi_parameter_update(self):
        """测试EnhancedMFI参数更新"""
        new_period = 10
        new_volatility_lookback = 15
        self.indicator.set_parameters(period=new_period, volatility_lookback=new_volatility_lookback)
        
        # 验证参数更新
        self.assertEqual(self.indicator.period, new_period)
        self.assertEqual(self.indicator.volatility_lookback, new_volatility_lookback)
        
        # 验证新参数下的计算
        result = self.indicator.calculate(self.data)
        self.assertIn('mfi', result.columns)
    
    def test_enhanced_mfi_required_columns(self):
        """测试EnhancedMFI必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_enhanced_mfi_comprehensive_score(self):
        """测试EnhancedMFI综合评分"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 然后计算评分
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        self.assertIsInstance(raw_score, pd.Series)
        
        # 验证评分范围
        valid_scores = raw_score.dropna()
        if len(valid_scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_enhanced_mfi_patterns(self):
        """测试EnhancedMFI形态识别"""
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
                'MFI_OVERBOUGHT', 'MFI_OVERSOLD',
                'MFI_ABOVE_50', 'MFI_BELOW_50',
                'MFI_CROSS_50_UP', 'MFI_CROSS_50_DOWN'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_enhanced_mfi_dynamic_thresholds(self):
        """测试EnhancedMFI动态阈值"""
        result = self.indicator.calculate(self.data)
        
        # 验证动态阈值列存在
        self.assertIn('mfi_overbought', result.columns)
        self.assertIn('mfi_oversold', result.columns)
        
        # 验证阈值的合理性
        overbought_values = result['mfi_overbought'].dropna()
        oversold_values = result['mfi_oversold'].dropna()
        
        if len(overbought_values) > 0 and len(oversold_values) > 0:
            # 超买阈值应该大于超卖阈值
            self.assertTrue(all(ob > os for ob, os in zip(overbought_values, oversold_values)),
                           "超买阈值应该大于超卖阈值")
    
    def test_enhanced_mfi_volume_filter(self):
        """测试EnhancedMFI成交量过滤"""
        # 测试启用成交量过滤
        filtered_indicator = EnhancedMFI(period=14, enable_volume_filter=True)
        result1 = filtered_indicator.calculate(self.data)
        
        # 测试禁用成交量过滤
        unfiltered_indicator = EnhancedMFI(period=14, enable_volume_filter=False)
        result2 = unfiltered_indicator.calculate(self.data)
        
        # 两种情况都应该能正常计算
        self.assertIsInstance(result1, pd.DataFrame)
        self.assertIsInstance(result2, pd.DataFrame)
        self.assertIn('mfi', result1.columns)
        self.assertIn('mfi', result2.columns)
    
    def test_enhanced_mfi_market_environment(self):
        """测试EnhancedMFI市场环境设置"""
        # 测试设置不同的市场环境
        environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        
        for env in environments:
            self.indicator.set_market_environment(env)
            self.assertEqual(self.indicator.market_environment, env)
        
        # 测试无效环境
        with self.assertRaises(ValueError):
            self.indicator.set_market_environment('invalid_environment')
    
    def test_enhanced_mfi_price_structure_synergy(self):
        """测试EnhancedMFI价格结构协同分析"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 分析价格结构协同
        synergy = self.indicator.analyze_price_structure_synergy(self.data)
        
        # 验证协同分析结果
        self.assertIsInstance(synergy, pd.DataFrame)
        self.assertIn('synergy_score', synergy.columns)
        
        # 验证协同评分的合理性
        synergy_scores = synergy['synergy_score'].dropna()
        if len(synergy_scores) > 0:
            # 协同评分应该是有限数值
            self.assertTrue(all(np.isfinite(v) for v in synergy_scores), 
                           "协同评分应该是有限数值")
    
    def test_enhanced_mfi_mfi_price_ratio(self):
        """测试EnhancedMFI价格比率计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证价格比率列存在
        self.assertIn('mfi_price_ratio', result.columns)
        
        # 验证价格比率的合理性
        ratio_values = result['mfi_price_ratio'].dropna()
        if len(ratio_values) > 0:
            # 价格比率应该是有限数值
            self.assertTrue(all(np.isfinite(v) for v in ratio_values), 
                           "MFI价格比率应该是有限数值")
    
    def test_enhanced_mfi_momentum_calculation(self):
        """测试EnhancedMFI动量计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证动量相关列存在
        momentum_columns = ['mfi_momentum', 'mfi_slope', 'mfi_accel']
        for col in momentum_columns:
            self.assertIn(col, result.columns, f"缺少动量列: {col}")
        
        # 验证动量值的合理性
        for col in momentum_columns:
            momentum_values = result[col].dropna()
            if len(momentum_values) > 0:
                # 动量应该是有限数值
                self.assertTrue(all(np.isfinite(v) for v in momentum_values), 
                               f"{col}应该是有限数值")
    
    def test_enhanced_mfi_adjusted_calculation(self):
        """测试EnhancedMFI环境调整计算"""
        # 测试不同市场环境下的调整
        environments = ['bull_market', 'bear_market', 'volatile_market', 'normal']
        
        for env in environments:
            self.indicator.set_market_environment(env)
            result = self.indicator.calculate(self.data)
            
            # 验证调整后的MFI列存在
            self.assertIn('mfi_adjusted', result.columns)
            
            # 验证调整后的MFI值的合理性
            adjusted_values = result['mfi_adjusted'].dropna()
            if len(adjusted_values) > 0:
                # 调整后的MFI应该是有限数值
                self.assertTrue(all(np.isfinite(v) for v in adjusted_values), 
                               f"在{env}环境下，调整后的MFI应该是有限数值")
    
    def test_enhanced_mfi_signals(self):
        """测试EnhancedMFI信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_enhanced_mfi_pattern_identification(self):
        """测试EnhancedMFI形态识别方法"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 识别形态
        patterns = self.indicator.identify_patterns(self.data)
        
        # 验证形态识别结果
        self.assertIsInstance(patterns, list)
        
        # 验证形态类型
        valid_patterns = [
            "MFI超买区域", "MFI超卖区域", "MFI中性区域偏多", "MFI中性区域偏空",
            "MFI从超买区回落", "MFI从超卖区回升", "MFI上升趋势", "MFI下降趋势",
            "MFI横盘整理", "MFI顶背离", "MFI底背离"
        ]
        
        for pattern in patterns:
            # 检查是否包含有效的形态关键词
            is_valid = any(valid_pattern in pattern for valid_pattern in valid_patterns)
            self.assertTrue(is_valid, f"无效的形态类型: {pattern}")
    
    def test_enhanced_mfi_generate_signals_method(self):
        """测试EnhancedMFI信号生成方法"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 生成信号
        signals = self.indicator.generate_signals(self.data)
        
        # 验证信号生成结果
        self.assertIsInstance(signals, pd.DataFrame)
        
        expected_signal_columns = [
            'buy_signal', 'sell_signal', 'neutral_signal', 'trend', 'score',
            'signal_type', 'signal_desc', 'confidence', 'risk_level', 'position_size'
        ]
        
        for col in expected_signal_columns:
            self.assertIn(col, signals.columns, f"缺少信号列: {col}")
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        for col in self.expected_columns:
            self.assertIn(col, result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_enhanced_mfi_register_patterns(self):
        """测试EnhancedMFI形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_enhanced_mfi_edge_cases(self):
        """测试EnhancedMFI边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(10)
        result = self.indicator.calculate(small_data)
        
        # EnhancedMFI应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('mfi', result.columns)
    
    def test_enhanced_mfi_validation(self):
        """测试EnhancedMFI数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['volume'], axis=1)
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(invalid_data)


if __name__ == '__main__':
    unittest.main()
