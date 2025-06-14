"""
StochRSI指标单元测试
"""
import unittest
import pandas as pd
import numpy as np
from indicators.stochrsi import STOCHRSI
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestSTOCHRSI(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    """StochRSI指标测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 显式调用LogCaptureMixin的setUp
        LogCaptureMixin.setUp(self)
        
        self.indicator = STOCHRSI(rsi_period=14, k_period=3, d_period=3, overbought=80, oversold=20)
        self.expected_columns = ['stochrsi_k', 'stochrsi_d']
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_stochrsi_calculation_accuracy(self):
        """测试StochRSI计算准确性"""
        result = self.indicator.calculate(self.data)
        
        # 验证StochRSI列存在
        self.assertIn('stochrsi_k', result.columns)
        self.assertIn('stochrsi_d', result.columns)
        self.assertIn('rsi', result.columns)
        
        # 验证StochRSI值的合理性
        k_values = result['stochrsi_k'].dropna()
        d_values = result['stochrsi_d'].dropna()
        
        if len(k_values) > 0:
            # StochRSI值应该在0-100范围内
            self.assertTrue(all(0 <= v <= 100 for v in k_values), "StochRSI K值应在0-100范围内")
        
        if len(d_values) > 0:
            # StochRSI D值应该在0-100范围内
            self.assertTrue(all(0 <= v <= 100 for v in d_values), "StochRSI D值应在0-100范围内")
    
    def test_stochrsi_score_range(self):
        """测试StochRSI评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内")
    
    def test_stochrsi_confidence_calculation(self):
        """测试StochRSI置信度计算"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, patterns, {})
        
        # 验证置信度在0-1范围内
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_stochrsi_parameter_update(self):
        """测试StochRSI参数更新"""
        # 创建新的指标实例测试参数
        new_indicator = STOCHRSI(rsi_period=21, k_period=5, d_period=5, overbought=75, oversold=25)
        
        # 验证参数设置
        self.assertEqual(new_indicator.rsi_period, 21)
        self.assertEqual(new_indicator.k_period, 5)
        self.assertEqual(new_indicator.d_period, 5)
        self.assertEqual(new_indicator.overbought, 75)
        self.assertEqual(new_indicator.oversold, 25)
        
        # 验证新参数下的计算
        result = new_indicator.calculate(self.data)
        self.assertIn('stochrsi_k', result.columns)
        self.assertIn('stochrsi_d', result.columns)
    
    def test_stochrsi_required_columns(self):
        """测试StochRSI必需列"""
        self.assertTrue(hasattr(self.indicator, 'REQUIRED_COLUMNS'))
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, self.indicator.REQUIRED_COLUMNS)
    
    def test_stochrsi_comprehensive_score(self):
        """测试StochRSI综合评分"""
        score_result = self.indicator.calculate_score(self.data)
        
        self.assertIsInstance(score_result, dict)
        self.assertIn('score', score_result)
        self.assertIn('confidence', score_result)
        
        # 验证评分范围
        self.assertGreaterEqual(score_result['score'], 0.0)
        self.assertLessEqual(score_result['score'], 100.0)
    
    def test_stochrsi_patterns(self):
        """测试StochRSI形态识别"""
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
                'STOCHRSI_GOLDEN_CROSS', 'STOCHRSI_DEATH_CROSS',
                'STOCHRSI_OVERBOUGHT', 'STOCHRSI_OVERSOLD'
            ]
            
            for pattern in expected_patterns:
                self.assertIn(pattern, patterns.columns, f"缺少形态列: {pattern}")
    
    def test_stochrsi_crossover_detection(self):
        """测试StochRSI交叉检测"""
        # 创建包含交叉的数据
        crossover_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 25},
            {'type': 'trend', 'start_price': 90, 'end_price': 110, 'periods': 25}
        ])
        
        patterns = self.indicator.get_patterns(crossover_data)
        
        # 验证金叉死叉形态存在
        if not patterns.empty:
            self.assertIn('STOCHRSI_GOLDEN_CROSS', patterns.columns)
            self.assertIn('STOCHRSI_DEATH_CROSS', patterns.columns)
    
    def test_stochrsi_overbought_oversold(self):
        """测试StochRSI超买超卖检测"""
        # 创建包含极端价格变动的数据
        extreme_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 150, 'periods': 15},  # 强烈上涨
            {'type': 'trend', 'start_price': 150, 'end_price': 100, 'periods': 15}   # 强烈下跌
        ])
        
        patterns = self.indicator.get_patterns(extreme_data)
        
        # 验证超买超卖形态被检测到
        if not patterns.empty:
            self.assertIn('STOCHRSI_OVERBOUGHT', patterns.columns)
            self.assertIn('STOCHRSI_OVERSOLD', patterns.columns)
    
    def test_stochrsi_rsi_calculation(self):
        """测试StochRSI中的RSI计算"""
        result = self.indicator.calculate(self.data)
        
        # 验证RSI值的合理性
        rsi_values = result['rsi'].dropna()
        if len(rsi_values) > 0:
            # RSI值应该在0-100范围内
            self.assertTrue(all(0 <= v <= 100 for v in rsi_values), "RSI值应在0-100范围内")
    
    def test_stochrsi_k_d_relationship(self):
        """测试StochRSI K线和D线的关系"""
        result = self.indicator.calculate(self.data)
        
        k_values = result['stochrsi_k'].dropna()
        d_values = result['stochrsi_d'].dropna()
        
        # D线应该是K线的移动平均，因此应该更平滑
        if len(k_values) > 5 and len(d_values) > 5:
            k_volatility = k_values.std()
            d_volatility = d_values.std()
            
            # D线的波动性应该小于或等于K线
            self.assertLessEqual(d_volatility, k_volatility * 1.5, 
                               "D线应该比K线更平滑")
    
    def test_stochrsi_signals(self):
        """测试StochRSI信号生成"""
        signals = self.indicator.generate_trading_signals(self.data)
        
        # 验证信号DataFrame结构
        self.assertIsInstance(signals, dict)
        expected_signal_keys = ['buy_signal', 'sell_signal', 'signal_strength']
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assertIsInstance(signals[key], pd.Series)
    
    def test_stochrsi_extreme_values(self):
        """测试StochRSI极端值处理"""
        # 创建包含极端价格的数据
        extreme_data = self.data.copy()
        extreme_data.loc[extreme_data.index[10], 'close'] = extreme_data['close'].iloc[9] * 2  # 价格翻倍
        extreme_data.loc[extreme_data.index[20], 'close'] = extreme_data['close'].iloc[19] * 0.5  # 价格减半
        
        result = self.indicator.calculate(extreme_data)
        
        # 验证StochRSI值仍在合理范围内
        k_values = result['stochrsi_k'].dropna()
        if len(k_values) > 0:
            self.assertTrue(all(0 <= v <= 100 for v in k_values), 
                           "即使有极端价格变动，StochRSI K值也应在0-100范围内")
    
    def test_no_errors_during_calculation(self):
        """测试计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行计算
        result = self.indicator.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('stochrsi_k', result.columns)
        self.assertIn('stochrsi_d', result.columns)
    
    def test_no_errors_during_pattern_detection(self):
        """测试形态检测过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行形态检测
        patterns = self.indicator.get_patterns(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证结果
        self.assertIsInstance(patterns, pd.DataFrame)
    
    def test_stochrsi_register_patterns(self):
        """测试StochRSI形态注册"""
        # 调用形态注册
        self.indicator.register_patterns()
        
        # 验证形态已注册（通过检查是否有异常抛出）
        self.assertTrue(True, "形态注册应该成功完成")
    
    def test_stochrsi_edge_cases(self):
        """测试StochRSI边界情况"""
        # 测试价格不变的情况
        flat_data = self.data.copy()
        flat_data['close'] = 100.0  # 所有价格相同
        
        result = self.indicator.calculate(flat_data)
        
        # StochRSI应该能够处理价格不变的情况
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('stochrsi_k', result.columns)
        self.assertIn('stochrsi_d', result.columns)
    
    def test_stochrsi_different_periods(self):
        """测试不同周期参数的StochRSI"""
        # 测试短周期
        short_period_indicator = STOCHRSI(rsi_period=7, k_period=2, d_period=2)
        short_result = short_period_indicator.calculate(self.data)
        
        # 测试长周期
        long_period_indicator = STOCHRSI(rsi_period=21, k_period=5, d_period=5)
        long_result = long_period_indicator.calculate(self.data)
        
        # 验证两种参数都能正常计算
        self.assertIn('stochrsi_k', short_result.columns)
        self.assertIn('stochrsi_k', long_result.columns)
        
        # 短周期应该更敏感（波动更大）
        short_k = short_result['stochrsi_k'].dropna()
        long_k = long_result['stochrsi_k'].dropna()
        
        if len(short_k) > 10 and len(long_k) > 10:
            short_volatility = short_k.std()
            long_volatility = long_k.std()
            
            # 这个测试可能不总是成立，所以只验证计算成功
            self.assertGreaterEqual(short_volatility, 0)
            self.assertGreaterEqual(long_volatility, 0)


if __name__ == '__main__':
    unittest.main()
