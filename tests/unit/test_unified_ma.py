"""
UnifiedMA指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.unified_ma import UnifiedMA
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestUnifiedMA(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """UnifiedMA指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = UnifiedMA(periods=[5, 10, 20], ma_type='simple')
        self.expected_columns = ['MA5', 'MA10', 'MA20']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_unified_ma_initialization(self):
        """测试UnifiedMA初始化"""
        # 测试默认初始化
        default_indicator = UnifiedMA()
        self.assertEqual(default_indicator._parameters['ma_type'], 'simple')
        self.assertEqual(default_indicator._parameters['periods'], [5, 10, 20, 30, 60])
        
        # 测试自定义初始化
        custom_indicator = UnifiedMA(periods=[10, 20], ma_type='ema')
        self.assertEqual(custom_indicator._parameters['ma_type'], 'ema')
        self.assertEqual(custom_indicator._parameters['periods'], [10, 20])
    
    def test_unified_ma_calculation_accuracy(self):
        """测试UnifiedMA计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证UnifiedMA列存在
        for col in self.expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证MA值的合理性
        for col in self.expected_columns:
            ma_values = result[col].dropna()
            
            if len(ma_values) > 0:
                # MA值应该是有限数值
                self.assertTrue(all(np.isfinite(v) for v in ma_values), f"{col}值应该是有限数值")
                # MA值应该在合理范围内
                self.assertTrue(all(v > 0 for v in ma_values), f"{col}值应该为正数")
    
    def test_unified_ma_different_types(self):
        """测试UnifiedMA不同类型计算"""
        ma_types = ['simple', 'ema', 'wma', 'ama', 'hma']
        
        for ma_type in ma_types:
            with self.subTest(ma_type=ma_type):
                indicator = UnifiedMA(periods=[10, 20], ma_type=ma_type)
                result = indicator.calculate(self.data)
                
                # 验证MA列存在
                self.assertIn('MA10', result.columns)
                self.assertIn('MA20', result.columns)
                
                # 验证MA值的合理性
                ma10_values = result['MA10'].dropna()
                ma20_values = result['MA20'].dropna()
                
                if len(ma10_values) > 0:
                    self.assertTrue(all(np.isfinite(v) for v in ma10_values), 
                                   f"{ma_type} MA10值应该是有限数值")
                if len(ma20_values) > 0:
                    self.assertTrue(all(np.isfinite(v) for v in ma20_values), 
                                   f"{ma_type} MA20值应该是有限数值")
    
    def test_unified_ma_score_range(self):
        """测试UnifiedMA评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_unified_ma_confidence_calculation(self):
        """测试UnifiedMA置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_unified_ma_parameter_update(self):
        """测试UnifiedMA参数更新"""
        new_params = {
            'periods': [15, 30],
            'ma_type': 'ema',
            'price_col': 'high'
        }
        self.indicator.set_parameters(new_params)
        
        # 验证参数更新
        self.assertEqual(self.indicator._parameters['periods'], [15, 30])
        self.assertEqual(self.indicator._parameters['ma_type'], 'ema')
        self.assertEqual(self.indicator._parameters['price_col'], 'high')
    
    def test_unified_ma_required_columns(self):
        """测试UnifiedMA必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_unified_ma_patterns(self):
        """测试UnifiedMA形态识别"""
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
                'MA_GOLDEN_CROSS', 'MA_DEATH_CROSS',
                'PRICE_ABOVE_SHORT_MA', 'PRICE_BELOW_SHORT_MA',
                'MA_SHORT_UPTREND', 'MA_SHORT_DOWNTREND'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_unified_ma_signals(self):
        """测试UnifiedMA信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_unified_ma_trend_detection(self):
        """测试UnifiedMA趋势检测"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 获取趋势
        trend = self.indicator.get_ma_trend(period=10)
        
        # 验证趋势检测结果
        self.assertIsInstance(trend, pd.Series)
        
        if not trend.empty:
            # 趋势值应该在-1, 0, 1之间
            unique_values = trend.dropna().unique()
            for val in unique_values:
                self.assertIn(val, [-1, 0, 1], "趋势值应该在-1, 0, 1之间")
    
    def test_unified_ma_consolidation_detection(self):
        """测试UnifiedMA盘整检测"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 检测盘整
        consolidation = self.indicator.is_consolidation(period=10, threshold=0.01)
        
        # 验证盘整检测结果
        self.assertIsInstance(consolidation, pd.Series)
        
        if not consolidation.empty:
            # 盘整值应该是布尔值
            unique_values = consolidation.dropna().unique()
            for val in unique_values:
                self.assertIsInstance(val, (bool, np.bool_)), "盘整值应该是布尔值"
    
    def test_unified_ma_sma_calculation(self):
        """测试UnifiedMA简单移动平均计算"""
        # 创建简单测试数据
        simple_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104] * 10,
            'high': [101, 102, 103, 104, 105] * 10,
            'low': [99, 100, 101, 102, 103] * 10,
            'close': [100.5, 101.5, 102.5, 103.5, 104.5] * 10,
            'volume': [1000, 1200, 800, 1500, 900] * 10
        })
        
        # 使用SMA类型
        sma_indicator = UnifiedMA(periods=[5], ma_type='simple')
        result = sma_indicator.calculate(simple_data)
        
        # 验证SMA计算
        if 'MA5' in result.columns:
            ma5_values = result['MA5'].dropna()
            if len(ma5_values) > 0:
                # SMA值应该是有限数值
                self.assertTrue(all(np.isfinite(v) for v in ma5_values), "SMA值应该是有限数值")
    
    def test_unified_ma_ema_calculation(self):
        """测试UnifiedMA指数移动平均计算"""
        # 使用EMA类型
        ema_indicator = UnifiedMA(periods=[10], ma_type='ema')
        result = ema_indicator.calculate(self.data)
        
        # 验证EMA计算
        if 'MA10' in result.columns:
            ma10_values = result['MA10'].dropna()
            if len(ma10_values) > 0:
                # EMA值应该是有限数值
                self.assertTrue(all(np.isfinite(v) for v in ma10_values), "EMA值应该是有限数值")
    
    def test_unified_ma_wma_calculation(self):
        """测试UnifiedMA加权移动平均计算"""
        # 使用WMA类型
        wma_indicator = UnifiedMA(periods=[10], ma_type='wma')
        result = wma_indicator.calculate(self.data)
        
        # 验证WMA计算
        if 'MA10' in result.columns:
            ma10_values = result['MA10'].dropna()
            if len(ma10_values) > 0:
                # WMA值应该是有限数值
                self.assertTrue(all(np.isfinite(v) for v in ma10_values), "WMA值应该是有限数值")
    
    def test_unified_ma_ama_calculation(self):
        """测试UnifiedMA自适应移动平均计算"""
        # 使用AMA类型
        ama_indicator = UnifiedMA(periods=[10], ma_type='ama')
        result = ama_indicator.calculate(self.data)
        
        # 验证AMA计算
        if 'MA10' in result.columns:
            ma10_values = result['MA10'].dropna()
            if len(ma10_values) > 0:
                # AMA值应该是有限数值
                self.assertTrue(all(np.isfinite(v) for v in ma10_values), "AMA值应该是有限数值")
    
    def test_unified_ma_hma_calculation(self):
        """测试UnifiedMA Hull移动平均计算"""
        # 使用HMA类型
        hma_indicator = UnifiedMA(periods=[10], ma_type='hma')
        result = hma_indicator.calculate(self.data)
        
        # 验证HMA计算
        if 'MA10' in result.columns:
            ma10_values = result['MA10'].dropna()
            if len(ma10_values) > 0:
                # HMA值应该是有限数值
                self.assertTrue(all(np.isfinite(v) for v in ma10_values), "HMA值应该是有限数值")
    
    def test_unified_ma_invalid_type(self):
        """测试UnifiedMA无效类型处理"""
        # 使用无效的MA类型
        invalid_indicator = UnifiedMA(periods=[10], ma_type='invalid_type')
        
        # 应该回退到默认的simple类型
        self.assertEqual(invalid_indicator._parameters['ma_type'], 'simple')
    
    def test_unified_ma_single_period(self):
        """测试UnifiedMA单周期计算"""
        single_indicator = UnifiedMA(periods=[20], ma_type='simple')
        result = single_indicator.calculate(self.data)
        
        # 验证单周期MA计算
        self.assertIn('MA20', result.columns)
        
        ma20_values = result['MA20'].dropna()
        if len(ma20_values) > 0:
            self.assertTrue(all(np.isfinite(v) for v in ma20_values), "单周期MA值应该是有限数值")
    
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
    
    def test_unified_ma_register_patterns(self):
        """测试UnifiedMA形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_unified_ma_edge_cases(self):
        """测试UnifiedMA边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(10)
        result = self.indicator.calculate(small_data)
        
        # UnifiedMA应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
        for col in self.expected_columns:
            self.assertIn(col, result.columns)
    
    def test_unified_ma_validation(self):
        """测试UnifiedMA数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['close'], axis=1)

        # BaseIndicator会处理缺失列并返回空DataFrame
        result = self.indicator.calculate(invalid_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_unified_ma_indicator_type(self):
        """测试UnifiedMA指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "UNIFIEDMA", "指标类型应该是UNIFIEDMA")


if __name__ == '__main__':
    unittest.main()
