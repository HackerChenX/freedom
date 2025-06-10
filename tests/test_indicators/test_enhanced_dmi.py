import unittest
import pandas as pd
import numpy as np
from indicators.trend.enhanced_dmi import EnhancedDMI


class TestEnhancedDMI(unittest.TestCase):
    """测试增强型DMI指标"""

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
            'close': price_base + np.random.normal(0, 0.1, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # 创建DMI实例
        self.dmi = EnhancedDMI(period=14, adx_period=14, adaptive=True)

    def test_calculation(self):
        """测试DMI基础计算功能"""
        result = self.dmi.calculate(self.test_data)
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['plus_di', 'minus_di', 'adx', 'adxr', 'dx', 'tr']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # 检查NaN值是否在预期范围内（前14个周期应该是NaN）
        self.assertTrue(result['plus_di'].iloc[13:].notna().all())
        self.assertTrue(result['minus_di'].iloc[13:].notna().all())
        self.assertTrue(result['adx'].iloc[27:].notna().all())  # ADX需要更多数据
        
        # 检查数值范围
        self.assertTrue((result['plus_di'].dropna() >= 0).all())
        self.assertTrue((result['minus_di'].dropna() >= 0).all())
        self.assertTrue((result['adx'].dropna() >= 0).all())
        self.assertTrue((result['adx'].dropna() <= 100).all())

    def test_adaptive_period(self):
        """测试自适应周期调整功能"""
        # 创建高波动数据
        high_vol_data = self.test_data.copy()
        high_vol_data['close'] = high_vol_data['close'] * (1 + np.random.normal(0, 0.02, 100))
        
        # 测试自适应模式
        dmi_adaptive = EnhancedDMI(period=14, adaptive=True)
        period_adaptive = dmi_adaptive.adjust_period_by_volatility(high_vol_data)
        
        # 测试非自适应模式
        dmi_fixed = EnhancedDMI(period=14, adaptive=False)
        period_fixed = dmi_fixed.adjust_period_by_volatility(high_vol_data)
        
        # 非自适应模式应该返回原始周期
        self.assertEqual(period_fixed, 14)
        
        # 自适应模式可能会改变周期（取决于数据波动性）
        # 在这里我们只检查它返回的是一个合理的整数值
        self.assertIsInstance(period_adaptive, int)
        self.assertTrue(6 <= period_adaptive <= 26)  # 在代码中定义的合理范围内

    def test_di_crossover_quality(self):
        """测试DI交叉质量评估"""
        self.dmi.calculate(self.test_data)
        quality = self.dmi.evaluate_di_crossover_quality()
        
        # 检查输出类型
        self.assertIsInstance(quality, pd.Series)
        self.assertEqual(len(quality), len(self.test_data))
        
        # 检查非交叉点的得分应该为0
        self.assertTrue((quality == 0).any())
        
        # 检查交叉点的得分应该非0（如果有交叉的话）
        plus_di = self.dmi._result['plus_di']
        minus_di = self.dmi._result['minus_di']
        
        golden_cross = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
        death_cross = (plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1))
        
        if golden_cross.any():
            self.assertTrue((quality[golden_cross] > 0).any())
        if death_cross.any():
            self.assertTrue((quality[death_cross] < 0).any())

    def test_three_line_synergy(self):
        """测试三线协同分析"""
        self.dmi.calculate(self.test_data)
        synergy = self.dmi.analyze_three_line_synergy()
        
        # 检查输出类型
        self.assertIsInstance(synergy, pd.DataFrame)
        self.assertEqual(len(synergy), len(self.test_data))
        
        # 检查是否包含预期的列
        expected_columns = ['strong_uptrend', 'strong_downtrend', 'weakening_trend', 
                           'potential_reversal', 'no_trend', 'emerging_trend']
        for col in expected_columns:
            self.assertIn(col, synergy.columns)
        
        # 检查值是否为布尔型
        for col in synergy.columns:
            self.assertTrue(synergy[col].dtype == bool)

    def test_score_calculation(self):
        """测试评分计算功能"""
        self.dmi.calculate(self.test_data)
        score = self.dmi.calculate_score()
        
        # 检查输出类型
        self.assertIsInstance(score, pd.Series)
        self.assertEqual(len(score), len(self.test_data))
        
        # 检查得分范围是否在0-100之间
        self.assertTrue((score.dropna() >= 0).all())
        self.assertTrue((score.dropna() <= 100).all())
        
        # 测试不同市场环境下的评分调整
        # 牛市环境
        self.dmi.set_market_environment("bull_market")
        bull_score = self.dmi.calculate_score()
        
        # 熊市环境
        self.dmi.set_market_environment("bear_market")
        bear_score = self.dmi.calculate_score()
        
        # 高波动环境
        self.dmi.set_market_environment("volatile_market")
        volatile_score = self.dmi.calculate_score()
        
        # 确保市场环境确实影响了评分
        # 不能直接比较得分值，因为影响是动态的，取决于原始得分
        # 但可以验证三种评分不全都相同
        self.assertTrue(not (bull_score.equals(bear_score) and bear_score.equals(volatile_score)))

    def test_pattern_identification(self):
        """测试形态识别功能"""
        self.dmi.calculate(self.test_data)
        patterns = self.dmi.identify_patterns()
        
        # 检查输出类型
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertEqual(len(patterns), len(self.test_data))
        
        # 检查是否包含预期的形态列
        expected_patterns = ['trend_start', 'trend_acceleration', 'trend_exhaustion',
                            'trend_reversal_warning', 'no_trend_zone', 'strong_trend',
                            'false_cross', 'trend_continuation']
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns)
        
        # 检查值是否为布尔型
        for col in patterns.columns:
            self.assertTrue(patterns[col].dtype == bool)

    def test_signal_generation(self):
        """测试信号生成功能"""
        signals = self.dmi.generate_signals(self.test_data)
        
        # 检查输出类型
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(len(signals), len(self.test_data))
        
        # 检查是否包含预期的列
        expected_columns = ['buy_signal', 'sell_signal', 'score', 'signal_type', 
                           'signal_desc', 'confidence']
        for col in expected_columns:
            self.assertIn(col, signals.columns)
        
        # 检查信号类型
        self.assertTrue(signals['buy_signal'].dtype == bool)
        self.assertTrue(signals['sell_signal'].dtype == bool)
        self.assertTrue((signals['score'].dropna() >= 0).all())
        self.assertTrue((signals['score'].dropna() <= 100).all())


if __name__ == '__main__':
    unittest.main() 