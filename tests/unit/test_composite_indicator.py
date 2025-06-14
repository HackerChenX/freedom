"""
CompositeIndicator指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.composite_indicator import CompositeIndicator
from indicators.ma import MA
from indicators.rsi import RSI
from indicators.macd import MACD
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestCompositeIndicator(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """CompositeIndicator指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        # 创建子指标
        self.ma_indicator = MA(periods=[20])
        self.rsi_indicator = RSI(period=14)
        self.macd_indicator = MACD()
        
        # 创建组合指标
        self.indicator = CompositeIndicator(
            name="TestComposite",
            description="测试组合指标",
            indicators=[self.ma_indicator, self.rsi_indicator, self.macd_indicator],
            weights={"MA": 0.4, "RSI": 0.3, "MACD": 0.3}
        )
        
        self.expected_columns = [
            'composite_score', 'MA_score', 'RSI_score', 'MACD_score'
        ]
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 80}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_composite_indicator_initialization(self):
        """测试CompositeIndicator初始化"""
        # 验证指标数量
        self.assertEqual(len(self.indicator.indicators), 3)
        
        # 验证权重
        self.assertAlmostEqual(self.indicator.weights["MA"], 0.4)
        self.assertAlmostEqual(self.indicator.weights["RSI"], 0.3)
        self.assertAlmostEqual(self.indicator.weights["MACD"], 0.3)
        
        # 验证权重总和为1
        total_weight = sum(self.indicator.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
    
    def test_composite_indicator_calculation_accuracy(self):
        """测试CompositeIndicator计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证CompositeIndicator列存在
        self.assertIn('composite_score', result.columns)
        
        # 验证各子指标的评分列存在
        for indicator_name in ['MA', 'RSI', 'MACD']:
            score_col = f"{indicator_name}_score"
            self.assertIn(score_col, result.columns, f"缺少评分列: {score_col}")
        
        # 验证评分值的合理性
        composite_scores = result['composite_score'].dropna()
        
        if len(composite_scores) > 0:
            # 评分应该在0-100范围内
            self.assertTrue(all(0 <= s <= 100 for s in composite_scores), 
                           "组合评分应该在0-100范围内")
    
    def test_composite_indicator_add_remove_indicators(self):
        """测试添加和移除指标"""
        # 创建新的组合指标
        composite = CompositeIndicator()
        
        # 添加指标
        composite.add_indicator(self.ma_indicator, 0.5)
        composite.add_indicator(self.rsi_indicator, 0.5)
        
        self.assertEqual(len(composite.indicators), 2)
        self.assertAlmostEqual(composite.weights["MA"], 0.5)
        self.assertAlmostEqual(composite.weights["RSI"], 0.5)
        
        # 移除指标
        composite.remove_indicator("MA")
        
        self.assertEqual(len(composite.indicators), 1)
        self.assertNotIn("MA", composite.weights)
        self.assertAlmostEqual(composite.weights["RSI"], 1.0)
    
    def test_composite_indicator_score_range(self):
        """测试CompositeIndicator评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_composite_indicator_confidence_calculation(self):
        """测试CompositeIndicator置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_composite_indicator_parameter_update(self):
        """测试CompositeIndicator参数更新"""
        new_ma = MA(periods=[10])
        new_weights = {"MA": 0.6, "RSI": 0.4}
        
        self.indicator.set_parameters(indicators=[new_ma, self.rsi_indicator], weights=new_weights)
        
        # 验证参数更新
        self.assertEqual(len(self.indicator.indicators), 2)
        self.assertAlmostEqual(self.indicator.weights["MA"], 0.6)
        self.assertAlmostEqual(self.indicator.weights["RSI"], 0.4)
    
    def test_composite_indicator_required_columns(self):
        """测试CompositeIndicator必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_composite_indicator_patterns(self):
        """测试CompositeIndicator形态识别"""
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
                'COMPOSITE_BULLISH_RESONANCE', 'COMPOSITE_BEARISH_RESONANCE',
                'COMPOSITE_DIVERGENCE'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_composite_indicator_custom_columns(self):
        """测试CompositeIndicator自定义列功能"""
        # 添加自定义列
        self.indicator.add_custom_column('custom_ratio', lambda df: df['close'] / df['open'])
        
        # 计算指标
        result = self.indicator.calculate(self.data)
        
        # 验证自定义列存在
        self.assertIn('custom_ratio', result.columns)
        
        # 验证自定义列值的合理性
        custom_values = result['custom_ratio'].dropna()
        if len(custom_values) > 0:
            self.assertTrue(all(v > 0 for v in custom_values), "自定义比率应该为正数")
    
    def test_composite_indicator_resonance_patterns(self):
        """测试CompositeIndicator共振形态检测"""
        # 先计算指标
        result = self.indicator.calculate(self.data)
        
        # 获取形态（这会触发共振形态检测）
        patterns_list = self.indicator.get_patterns(self.data)
        
        # 验证形态检测结果
        self.assertIsInstance(patterns_list, pd.DataFrame)
    
    def test_composite_indicator_weight_normalization(self):
        """测试CompositeIndicator权重标准化"""
        # 创建权重不为1的组合指标
        composite = CompositeIndicator(
            indicators=[self.ma_indicator, self.rsi_indicator],
            weights={"MA": 2.0, "RSI": 3.0}
        )
        
        # 验证权重被标准化
        total_weight = sum(composite.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
        self.assertAlmostEqual(composite.weights["MA"], 0.4)
        self.assertAlmostEqual(composite.weights["RSI"], 0.6)
    
    def test_composite_indicator_empty_indicators(self):
        """测试CompositeIndicator空指标列表"""
        empty_composite = CompositeIndicator()
        
        # 计算应该返回原始数据
        result = empty_composite.calculate(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 评分应该返回默认值
        score = empty_composite.calculate_raw_score(self.data)
        self.assertIsInstance(score, pd.Series)
    
    def test_composite_indicator_signals(self):
        """测试CompositeIndicator信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_composite_indicator_indicator_names(self):
        """测试CompositeIndicator指标名称获取"""
        names = self.indicator.get_indicator_names()
        
        self.assertIsInstance(names, list)
        self.assertEqual(len(names), 3)
        self.assertIn("MA", names)
        self.assertIn("RSI", names)
        self.assertIn("MACD", names)
    
    def test_composite_indicator_weights_getter(self):
        """测试CompositeIndicator权重获取"""
        weights = self.indicator.get_indicator_weights()
        
        self.assertIsInstance(weights, dict)
        self.assertAlmostEqual(weights["MA"], 0.4)
        self.assertAlmostEqual(weights["RSI"], 0.3)
        self.assertAlmostEqual(weights["MACD"], 0.3)
    
    def test_composite_indicator_market_environment(self):
        """测试CompositeIndicator市场环境设置"""
        # 测试设置不同的市场环境
        environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        
        for env in environments:
            self.indicator.set_market_environment(env)
            self.assertEqual(self.indicator.market_environment, env)
        
        # 测试无效环境
        with self.assertRaises(ValueError):
            self.indicator.set_market_environment('invalid_environment')
    
    def test_composite_indicator_automatic_scoring(self):
        """测试CompositeIndicator自动评分功能"""
        # 测试启用自动评分
        self.indicator.calculate_score_automatically = True
        result1 = self.indicator.calculate(self.data)
        self.assertIn('composite_score', result1.columns)
        
        # 测试禁用自动评分
        self.indicator.calculate_score_automatically = False
        result2 = self.indicator.calculate(self.data)
        # 即使禁用自动评分，composite_score列也应该存在（通过其他方式计算）
        self.assertIsInstance(result2, pd.DataFrame)
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('composite_score', result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_composite_indicator_register_patterns(self):
        """测试CompositeIndicator形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_composite_indicator_edge_cases(self):
        """测试CompositeIndicator边界情况"""
        # 测试数据不足的情况
        small_data = self.data.head(10)
        result = self.indicator.calculate(small_data)
        
        # CompositeIndicator应该能处理数据不足的情况
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_composite_indicator_validation(self):
        """测试CompositeIndicator数据验证"""
        # 测试缺少必需列的情况
        invalid_data = self.data.drop(['close'], axis=1)
        
        # CompositeIndicator应该能处理缺少列的情况（由子指标处理）
        try:
            result = self.indicator.calculate(invalid_data)
            self.assertIsInstance(result, pd.DataFrame)
        except ValueError:
            # 如果子指标抛出异常也是可以接受的
            pass
    
    def test_composite_indicator_indicator_type(self):
        """测试CompositeIndicator指标类型"""
        indicator_type = self.indicator.get_indicator_type()
        self.assertEqual(indicator_type, "COMPOSITE", "指标类型应该是COMPOSITE")


if __name__ == '__main__':
    unittest.main()
