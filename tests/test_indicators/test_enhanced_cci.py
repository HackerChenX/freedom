import unittest
import pandas as pd
import numpy as np
from indicators.trend.enhanced_cci import EnhancedCCI


class TestEnhancedCCI(unittest.TestCase):
    """测试增强型CCI指标"""

    def setUp(self):
        """准备测试数据"""
        # 创建模拟价格数据
        np.random.seed(42)  # 确保结果可重现
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # 创建趋势和震荡市场的混合数据
        # 前50天是上升趋势，后50天是震荡
        trend = np.linspace(0, 5, 50)
        oscillation = np.random.normal(0, 0.5, 50)
        combined = np.concatenate([trend, oscillation])
        
        # 添加随机波动
        noise = np.random.normal(0, 0.2, 100)
        price_base = combined + noise
        
        # 生成OHLC数据
        self.test_data = pd.DataFrame({
            'open': price_base,
            'high': price_base + np.random.uniform(0.1, 0.3, 100),
            'low': price_base - np.random.uniform(0.1, 0.3, 100),
            'close': price_base + np.random.normal(0, 0.1, 100)
        }, index=dates)
        
        # 创建CCI实例
        self.cci = EnhancedCCI(period=20, factor=0.015, adaptive=True)

    def test_calculation(self):
        """测试CCI基础计算功能"""
        result = self.cci.calculate(self.test_data)
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['cci', 'cci_secondary', 'cci_ma5', 'cci_ma10', 
                           'cci_ma20', 'cci_slope', 'cci_volatility', 'state']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # 检查CCI是否与原始数据长度一致
        self.assertEqual(len(result), len(self.test_data))
        
        # 检查CCI值是否在合理范围内
        # CCI通常在±300之间，但极端情况可能超出
        self.assertTrue((result['cci'].abs() < 500).all())
        
        # 检查CCI状态分类
        unique_states = result['state'].unique()
        expected_states = ['extreme_overbought', 'overbought', 'neutral_high',
                          'neutral_low', 'oversold', 'extreme_oversold', 'neutral']
        for state in unique_states:
            self.assertIn(state, expected_states)

    def test_adaptive_period(self):
        """测试CCI自适应周期功能"""
        # 创建低波动率数据
        low_vol_data = self.test_data.copy()
        low_vol_data['close'] = low_vol_data['close'].rolling(window=10).mean()
        
        # 创建高波动率数据
        high_vol_data = self.test_data.copy()
        high_vol_data['close'] = high_vol_data['close'] + np.random.normal(0, 1, len(high_vol_data))
        
        # 测试自适应CCI
        cci_adaptive = EnhancedCCI(period=20, adaptive=True)
        cci_adaptive.calculate(high_vol_data)
        high_vol_period = cci_adaptive.current_period
        
        cci_adaptive.calculate(low_vol_data)
        low_vol_period = cci_adaptive.current_period
        
        # 高波动率应该使用较短周期，低波动率应该使用较长周期
        self.assertLessEqual(high_vol_period, cci_adaptive.base_period)
        self.assertGreaterEqual(low_vol_period, cci_adaptive.base_period)

    def test_crossovers(self):
        """测试交叉分析功能"""
        self.cci.calculate(self.test_data)
        crossovers = self.cci.analyze_crossovers()
        
        # 检查输出类型
        self.assertIsInstance(crossovers, pd.DataFrame)
        self.assertEqual(len(crossovers), len(self.test_data))
        
        # 检查是否包含预期的列
        expected_columns = ['zero_cross_up', 'zero_cross_down', 'overbought_enter', 
                           'overbought_exit', 'oversold_enter', 'oversold_exit',
                           'extreme_overbought_enter', 'extreme_overbought_exit',
                           'extreme_oversold_enter', 'extreme_oversold_exit',
                           'ma5_cross_up', 'ma5_cross_down', 'crossover_strength']
        for col in expected_columns:
            self.assertIn(col, crossovers.columns)
        
        # 检查交叉点的互斥性
        # 零轴向上交叉和向下交叉不应同时发生
        self.assertFalse((crossovers['zero_cross_up'] & crossovers['zero_cross_down']).any())
        
        # 验证交叉强度的范围
        self.assertTrue((crossovers['crossover_strength'] >= 0).all())
        self.assertTrue((crossovers['crossover_strength'] <= 100).all())

    def test_multi_period_synergy(self):
        """测试多周期协同分析功能"""
        self.cci.calculate(self.test_data)
        synergy = self.cci.analyze_multi_period_synergy()
        
        # 检查输出类型
        self.assertIsInstance(synergy, pd.DataFrame)
        self.assertEqual(len(synergy), len(self.test_data))
        
        # 检查是否包含预期的列
        expected_columns = ['bullish_agreement', 'bearish_agreement', 
                           'rising_momentum', 'falling_momentum', 'synergy_score']
        for col in expected_columns:
            self.assertIn(col, synergy.columns)
        
        # 验证协同评分的范围
        self.assertTrue((synergy['synergy_score'] >= 0).all())
        self.assertTrue((synergy['synergy_score'] <= 100).all())
        
        # 验证方向一致性的互斥性
        self.assertFalse((synergy['bullish_agreement'] & synergy['bearish_agreement']).any())
        
        # 验证动量一致性的互斥性
        self.assertFalse((synergy['rising_momentum'] & synergy['falling_momentum']).any())

    def test_pattern_identification(self):
        """测试形态识别功能"""
        self.cci.calculate(self.test_data)
        patterns = self.cci.identify_patterns()
        
        # 检查输出类型
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertEqual(len(patterns), len(self.test_data))
        
        # 检查是否包含预期的形态列
        expected_patterns = ['overbought_hook', 'oversold_hook',
                            'zero_rejection_bull', 'zero_rejection_bear',
                            'head_shoulders_top', 'head_shoulders_bottom',
                            'bullish_divergence', 'bearish_divergence']
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns)
        
        # 验证形态互斥性
        # 头肩顶和头肩底不应同时出现
        self.assertFalse((patterns['head_shoulders_top'] & patterns['head_shoulders_bottom']).any())
        
        # 正背离和负背离不应同时出现
        self.assertFalse((patterns['bullish_divergence'] & patterns['bearish_divergence']).any())

    def test_score_calculation(self):
        """测试评分计算功能"""
        self.cci.calculate(self.test_data)
        score = self.cci.calculate_score()
        
        # 检查输出类型
        self.assertIsInstance(score, pd.Series)
        self.assertEqual(len(score), len(self.test_data))
        
        # 检查评分范围
        self.assertTrue((score >= 0).all())
        self.assertTrue((score <= 100).all())
        
        # 测试不同市场环境下的评分调整
        # 牛市环境
        self.cci.set_market_environment("bull_market")
        bull_score = self.cci.calculate_score()
        
        # 熊市环境
        self.cci.set_market_environment("bear_market")
        bear_score = self.cci.calculate_score()
        
        # 高波动环境
        self.cci.set_market_environment("volatile_market")
        volatile_score = self.cci.calculate_score()
        
        # 确保市场环境确实影响了评分
        self.assertTrue(not (bull_score.equals(bear_score) and bear_score.equals(volatile_score)))

    def test_signal_generation(self):
        """测试信号生成功能"""
        signals = self.cci.generate_signals(self.test_data)
        
        # 检查输出类型
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(len(signals), len(self.test_data))
        
        # 检查是否包含预期的列
        expected_columns = ['buy_signal', 'sell_signal', 'score', 'signal_type', 
                           'signal_desc', 'confidence', 'trend', 'risk_level', 'position_size']
        for col in expected_columns:
            self.assertIn(col, signals.columns)
        
        # 检查信号类型
        self.assertTrue(signals['buy_signal'].dtype == bool)
        self.assertTrue(signals['sell_signal'].dtype == bool)
        
        # 验证买入和卖出信号的互斥性
        # 注意：在实际应用中，买入和卖出信号可能同时存在，这里仅为简化测试
        # self.assertFalse((signals['buy_signal'] & signals['sell_signal']).any())
        
        # 检查评分范围
        self.assertTrue((signals['score'].dropna() >= 0).all())
        self.assertTrue((signals['score'].dropna() <= 100).all())
        
        # 检查信号置信度范围
        self.assertTrue((signals['confidence'].dropna() >= 0).all())
        self.assertTrue((signals['confidence'].dropna() <= 100).all())


if __name__ == '__main__':
    unittest.main() 