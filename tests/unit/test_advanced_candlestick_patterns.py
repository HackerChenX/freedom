"""
AdvancedCandlestickPatterns指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.pattern.advanced_candlestick_patterns import AdvancedCandlestickPatterns, AdvancedPatternType
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestAdvancedCandlestickPatterns(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """AdvancedCandlestickPatterns指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = AdvancedCandlestickPatterns()
        self.expected_columns = [
            '三白兵', '三黑鸦', '三内涨', '三内跌', '三外涨', '三外跌',
            '上升三法', '下降三法', '铺垫形态', '棍心三明治',
            '梯底形态', '塔顶形态', '脱离形态', '反冲形态', '奇特三河',
            '头肩顶', '头肩底', '双顶', '双底', '三重顶', '三重底',
            '上升三角形', '下降三角形', '对称三角形', '矩形整理',
            '钻石顶', '钻石底', '杯柄形态'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_advanced_candlestick_patterns_initialization(self):
        """测试AdvancedCandlestickPatterns初始化"""
        # 测试默认初始化
        default_indicator = AdvancedCandlestickPatterns()
        self.assertEqual(default_indicator.name, "AdvancedCandlestickPatterns")
        self.assertIn("高级K线形态识别指标", default_indicator.description)
    
    def test_advanced_candlestick_patterns_calculation_accuracy(self):
        """测试AdvancedCandlestickPatterns计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证AdvancedCandlestickPatterns列存在
        for col in self.expected_columns:
            self.assertIn(col, result.columns, f"缺少列: {col}")
        
        # 验证形态值的合理性
        for col in self.expected_columns:
            if col in result.columns:
                pattern_values = result[col].dropna()
                if len(pattern_values) > 0:
                    # 形态值应该是布尔值
                    unique_values = pattern_values.unique()
                    for val in unique_values:
                        self.assertIsInstance(val, (bool, np.bool_), f"{col}应该是布尔值")
    
    def test_advanced_candlestick_patterns_score_range(self):
        """测试AdvancedCandlestickPatterns评分范围"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        if 'score' in raw_score_df.columns:
            valid_scores = raw_score_df['score'].dropna()
            self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_advanced_candlestick_patterns_confidence_calculation(self):
        """测试AdvancedCandlestickPatterns置信度计算"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        raw_score = raw_score_df['score'] if 'score' in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_advanced_candlestick_patterns_parameter_update(self):
        """测试AdvancedCandlestickPatterns参数更新"""
        # AdvancedCandlestickPatterns通常没有可变参数，但应该有set_parameters方法
        self.indicator.set_parameters()
        # 验证方法存在且不抛出异常
        self.assertTrue(True, "set_parameters方法应该存在")
    
    def test_advanced_candlestick_patterns_required_columns(self):
        """测试AdvancedCandlestickPatterns必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_advanced_candlestick_patterns_patterns(self):
        """测试AdvancedCandlestickPatterns形态识别"""
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
    
    def test_advanced_candlestick_patterns_signals(self):
        """测试AdvancedCandlestickPatterns信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_advanced_candlestick_patterns_three_star_patterns(self):
        """测试AdvancedCandlestickPatterns三星形态"""
        result = self.indicator.calculate(self.data)
        
        # 验证三星形态列存在
        three_star_patterns = ['三白兵', '三黑鸦', '三内涨', '三内跌', '三外涨', '三外跌']
        
        for pattern in three_star_patterns:
            self.assertIn(pattern, result.columns, f"缺少三星形态: {pattern}")
            
            # 验证形态值的合理性
            pattern_values = result[pattern].dropna()
            if len(pattern_values) > 0:
                unique_values = pattern_values.unique()
                for val in unique_values:
                    self.assertIsInstance(val, (bool, np.bool_), f"{pattern}应该是布尔值")
    
    def test_advanced_candlestick_patterns_compound_patterns(self):
        """测试AdvancedCandlestickPatterns复合形态"""
        result = self.indicator.calculate(self.data)
        
        # 验证复合形态列存在
        compound_patterns = ['上升三法', '下降三法', '铺垫形态', '棍心三明治',
                           '梯底形态', '塔顶形态', '脱离形态', '反冲形态', '奇特三河']
        
        for pattern in compound_patterns:
            self.assertIn(pattern, result.columns, f"缺少复合形态: {pattern}")
            
            # 验证形态值的合理性
            pattern_values = result[pattern].dropna()
            if len(pattern_values) > 0:
                unique_values = pattern_values.unique()
                for val in unique_values:
                    self.assertIsInstance(val, (bool, np.bool_), f"{pattern}应该是布尔值")
    
    def test_advanced_candlestick_patterns_complex_patterns(self):
        """测试AdvancedCandlestickPatterns复杂形态"""
        # 需要足够的数据进行复杂形态识别
        long_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100}
        ])
        
        result = self.indicator.calculate(long_data)
        
        # 验证复杂形态列存在
        complex_patterns = ['头肩顶', '头肩底', '双顶', '双底', '三重顶', '三重底',
                          '上升三角形', '下降三角形', '对称三角形', '矩形整理',
                          '钻石顶', '钻石底', '杯柄形态']
        
        for pattern in complex_patterns:
            self.assertIn(pattern, result.columns, f"缺少复杂形态: {pattern}")
            
            # 验证形态值的合理性
            pattern_values = result[pattern].dropna()
            if len(pattern_values) > 0:
                unique_values = pattern_values.unique()
                for val in unique_values:
                    self.assertIsInstance(val, (bool, np.bool_), f"{pattern}应该是布尔值")
    
    def test_advanced_candlestick_patterns_identify_patterns(self):
        """测试AdvancedCandlestickPatterns形态识别"""
        identified_patterns = self.indicator.identify_patterns(self.data)
        
        # 验证返回列表
        self.assertIsInstance(identified_patterns, list)
        
        # 验证列表元素为字符串
        for pattern in identified_patterns:
            self.assertIsInstance(pattern, str, "识别的形态应该是字符串")
    
    def test_advanced_candlestick_patterns_generate_signals_method(self):
        """测试AdvancedCandlestickPatterns信号生成方法"""
        signals = self.indicator.generate_signals(self.indicator.calculate(self.data))
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, pd.DataFrame)
        
        # 验证基本信号列存在
        expected_signal_columns = ['buy_signal', 'sell_signal', 'watch_signal', 
                                 'signal_strength', 'trend_confirmed', 'compound_signal']
        
        for col in expected_signal_columns:
            self.assertIn(col, signals.columns, f"缺少信号列: {col}")
    
    def test_advanced_candlestick_patterns_pattern_types(self):
        """测试AdvancedCandlestickPatterns形态类型枚举"""
        # 验证AdvancedPatternType枚举存在
        self.assertTrue(hasattr(AdvancedPatternType, 'THREE_WHITE_SOLDIERS'))
        self.assertTrue(hasattr(AdvancedPatternType, 'THREE_BLACK_CROWS'))
        self.assertTrue(hasattr(AdvancedPatternType, 'HEAD_SHOULDERS_TOP'))
        self.assertTrue(hasattr(AdvancedPatternType, 'DOUBLE_BOTTOM'))
        
        # 验证枚举值
        self.assertEqual(AdvancedPatternType.THREE_WHITE_SOLDIERS.value, "三白兵")
        self.assertEqual(AdvancedPatternType.THREE_BLACK_CROWS.value, "三黑鸦")
        self.assertEqual(AdvancedPatternType.HEAD_SHOULDERS_TOP.value, "头肩顶")
    
    def test_advanced_candlestick_patterns_score_calculation(self):
        """测试AdvancedCandlestickPatterns评分计算"""
        raw_score_df = self.indicator.calculate_raw_score(self.data)
        
        # 验证返回DataFrame
        self.assertIsInstance(raw_score_df, pd.DataFrame)
        
        # 验证包含score列
        self.assertIn('score', raw_score_df.columns)
        
        # 验证评分值的合理性
        scores = raw_score_df['score'].dropna()
        if len(scores) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in scores), "评分应在0-100范围内")
    
    def test_advanced_candlestick_patterns_signal_strength(self):
        """测试AdvancedCandlestickPatterns信号强度计算"""
        result = self.indicator.calculate(self.data)
        signals = self.indicator.generate_signals(result)
        
        # 验证信号强度列存在
        self.assertIn('signal_strength', signals.columns)
        
        # 验证信号强度值的合理性
        strength_values = signals['signal_strength'].dropna()
        if len(strength_values) > 0:
            self.assertTrue(all(0 <= s <= 100 for s in strength_values), 
                           "信号强度应在0-100范围内")
    
    def test_advanced_candlestick_patterns_trend_confirmation(self):
        """测试AdvancedCandlestickPatterns趋势确认"""
        result = self.indicator.calculate(self.data)
        signals = self.indicator.generate_signals(result)
        
        # 验证趋势确认列存在
        self.assertIn('trend_confirmed', signals.columns)
        
        # 验证趋势确认值为布尔值
        trend_confirm_values = signals['trend_confirmed'].dropna()
        if len(trend_confirm_values) > 0:
            unique_values = trend_confirm_values.unique()
            for val in unique_values:
                self.assertIsInstance(val, (bool, np.bool_), "趋势确认应该是布尔值")
    
    def test_advanced_candlestick_patterns_compound_signal(self):
        """测试AdvancedCandlestickPatterns复合信号"""
        result = self.indicator.calculate(self.data)
        signals = self.indicator.generate_signals(result)
        
        # 验证复合信号列存在
        self.assertIn('compound_signal', signals.columns)
        
        # 验证复合信号值的合理性
        compound_values = signals['compound_signal'].dropna()
        if len(compound_values) > 0:
            self.assertTrue(all(0 <= v <= 10 for v in compound_values), 
                           "复合信号应在0-10范围内")
    
    def test_advanced_candlestick_patterns_volume_confirmation(self):
        """测试AdvancedCandlestickPatterns成交量确认"""
        # 生成带成交量的数据
        data_with_volume = self.data.copy()
        data_with_volume['volume'] = np.random.randint(1000, 10000, len(data_with_volume))
        
        raw_score_df = self.indicator.calculate_raw_score(data_with_volume)
        
        # 验证成交量确认在评分中的影响
        self.assertIsInstance(raw_score_df, pd.DataFrame)
        self.assertIn('score', raw_score_df.columns)
    
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
    
    def test_advanced_candlestick_patterns_register_patterns(self):
        """测试AdvancedCandlestickPatterns形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_advanced_candlestick_patterns_edge_cases(self):
        """测试AdvancedCandlestickPatterns边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(10)
        result = self.indicator.calculate(small_data)
        
        # AdvancedCandlestickPatterns应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
        for col in self.expected_columns:
            self.assertIn(col, result.columns)
    
    def test_advanced_candlestick_patterns_validation(self):
        """测试AdvancedCandlestickPatterns数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['high'], axis=1)
        
        # BaseIndicator会处理缺失列并返回空DataFrame
        result = self.indicator.calculate(invalid_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_advanced_candlestick_patterns_indicator_type(self):
        """测试AdvancedCandlestickPatterns指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "ADVANCEDCANDLESTICKPATTERNS", "指标类型应该是ADVANCEDCANDLESTICKPATTERNS")


if __name__ == '__main__':
    unittest.main()
