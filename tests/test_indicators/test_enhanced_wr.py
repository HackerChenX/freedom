import unittest
import pandas as pd
import numpy as np
from indicators.enhanced_wr import EnhancedWR


class TestEnhancedWR(unittest.TestCase):
    """测试增强型威廉指标(Williams %R)"""

    def setUp(self):
        """准备测试数据"""
        # 创建模拟价格数据
        np.random.seed(42)  # 确保结果可重现
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # 创建趋势和震荡市场的混合数据
        # 前30天是上升趋势，中间40天是震荡，后30天是下降趋势
        trend1 = np.linspace(0, 3, 30)
        oscillation = np.random.normal(0, 0.5, 40)
        trend2 = np.linspace(3, 0, 30)
        
        combined = np.concatenate([trend1, oscillation, trend2])
        
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
        
        # 创建增强型WR实例
        self.wr = EnhancedWR(period=14, secondary_period=28, adaptive_threshold=True)

    def test_calculation(self):
        """测试WR基础计算功能"""
        result = self.wr.calculate(self.test_data)
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['wr', 'wr_secondary', 'wr_volatility', 'wr_rate_of_change', 'wr_mean', 'wr_momentum']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # 检查多周期WR列是否存在
        for period in self.wr.multi_periods:
            if period != self.wr.period and period != self.wr.secondary_period:
                self.assertIn(f'wr_{period}', result.columns)
        
        # 检查WR值的范围
        self.assertTrue((result['wr'].dropna() >= -100).all())
        self.assertTrue((result['wr'].dropna() <= 0).all())
        
        # 检查次要周期WR
        self.assertTrue((result['wr_secondary'].dropna() >= -100).all())
        self.assertTrue((result['wr_secondary'].dropna() <= 0).all())

    def test_adaptive_threshold(self):
        """测试自适应阈值调整功能"""
        # 计算WR
        self.wr.calculate(self.test_data)
        
        # 检查自适应阈值是否合理
        self.assertTrue(-25 <= self.wr.overbought_threshold <= -15)
        self.assertTrue(-85 <= self.wr.oversold_threshold <= -75)
        
        # 测试不同市场环境下的阈值调整
        # 牛市环境
        self.wr.set_market_environment('bull_market')
        self.wr._adjust_thresholds_by_market(self.wr._result)
        bull_overbought = self.wr.overbought_threshold
        bull_oversold = self.wr.oversold_threshold
        
        # 熊市环境
        self.wr.set_market_environment('bear_market')
        self.wr._adjust_thresholds_by_market(self.wr._result)
        bear_overbought = self.wr.overbought_threshold
        bear_oversold = self.wr.oversold_threshold
        
        # 检查牛市环境下超买阈值是否低于熊市环境
        self.assertTrue(bull_overbought <= bear_overbought)
        
        # 检查熊市环境下超卖阈值是否低于牛市环境
        self.assertTrue(bear_oversold <= bull_oversold)

    def test_enhanced_divergence(self):
        """测试增强型背离识别功能"""
        # 计算WR
        self.wr.calculate(self.test_data)
        
        # 获取背离结果
        divergence = self.wr.detect_enhanced_divergence()
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['bullish_divergence', 'bearish_divergence', 
                           'hidden_bullish_divergence', 'hidden_bearish_divergence',
                           'divergence_strength']
        for col in expected_columns:
            self.assertIn(col, divergence.columns)
        
        # 检查布尔型列
        self.assertTrue(divergence['bullish_divergence'].dtype == bool)
        self.assertTrue(divergence['bearish_divergence'].dtype == bool)
        self.assertTrue(divergence['hidden_bullish_divergence'].dtype == bool)
        self.assertTrue(divergence['hidden_bearish_divergence'].dtype == bool)
        
        # 检查背离强度范围
        self.assertTrue((divergence['divergence_strength'] >= 0).all())

    def test_multi_period_synergy(self):
        """测试多周期协同分析功能"""
        # 计算WR
        self.wr.calculate(self.test_data)
        
        # 获取多周期协同分析结果
        synergy = self.wr.analyze_multi_period_synergy()
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['all_overbought', 'all_oversold', 'primary_overbought', 
                           'primary_oversold', 'secondary_overbought', 'secondary_oversold',
                           'overbought_ratio', 'oversold_ratio', 'consensus_score']
        for col in expected_columns:
            self.assertIn(col, synergy.columns)
        
        # 检查布尔型列
        self.assertTrue(synergy['all_overbought'].dtype == bool)
        self.assertTrue(synergy['all_oversold'].dtype == bool)
        
        # 检查比率范围
        self.assertTrue((synergy['overbought_ratio'] >= 0).all())
        self.assertTrue((synergy['overbought_ratio'] <= 1).all())
        self.assertTrue((synergy['oversold_ratio'] >= 0).all())
        self.assertTrue((synergy['oversold_ratio'] <= 1).all())
        
        # 检查共识得分范围
        self.assertTrue((synergy['consensus_score'].dropna() >= 0).all())
        self.assertTrue((synergy['consensus_score'].dropna() <= 100).all())

    def test_oscillation_band(self):
        """测试震荡带分析功能"""
        # 计算WR
        self.wr.calculate(self.test_data)
        
        # 获取震荡带分析结果
        oscillation = self.wr.identify_oscillation_band()
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['upper_band', 'lower_band', 'middle_band', 
                           'above_upper', 'below_lower', 'in_band',
                           'breakout_up', 'breakout_down', 'band_width',
                           'band_width_expanding', 'band_width_contracting',
                           'extremely_narrow_band']
        for col in expected_columns:
            self.assertIn(col, oscillation.columns)
        
        # 检查震荡带值的关系
        self.assertTrue((oscillation['upper_band'].dropna() >= oscillation['middle_band'].dropna()).all())
        self.assertTrue((oscillation['middle_band'].dropna() >= oscillation['lower_band'].dropna()).all())
        
        # 检查布尔型列
        self.assertTrue(oscillation['above_upper'].dtype == bool)
        self.assertTrue(oscillation['below_lower'].dtype == bool)
        self.assertTrue(oscillation['in_band'].dtype == bool)
        
        # 检查带宽计算
        self.assertTrue((oscillation['band_width'].dropna() >= 0).all())
        self.assertTrue((oscillation['band_width'] == (oscillation['upper_band'] - oscillation['lower_band'])).all())

    def test_pattern_identification(self):
        """测试形态识别功能"""
        # 计算WR
        self.wr.calculate(self.test_data)
        
        # 获取形态识别结果
        patterns = self.wr.identify_patterns()
        
        # 检查基础形态列是否存在
        basic_patterns = ['overbought', 'oversold', 'extreme_overbought', 'extreme_oversold',
                         'cross_above_oversold', 'cross_below_overbought',
                         'cross_above_midline', 'cross_below_midline',
                         'w_bottom', 'm_top']
        for pattern in basic_patterns:
            self.assertIn(pattern, patterns.columns)
        
        # 检查布尔型列
        for col in patterns.columns:
            self.assertTrue(patterns[col].dtype == bool)
        
        # 检查W底和M顶形态识别
        self.assertIsNotNone(self.wr._detect_w_bottom)
        self.assertIsNotNone(self.wr._detect_m_top)
        
        # 检查钝化形态识别
        self.assertIn('overbought_stagnation', patterns.columns)
        self.assertIn('oversold_stagnation', patterns.columns)

    def test_score_calculation(self):
        """测试评分计算功能"""
        # 计算WR
        self.wr.calculate(self.test_data)
        
        # 计算评分
        score = self.wr.calculate_score()
        
        # 检查评分范围
        self.assertTrue((score.dropna() >= 0).all())
        self.assertTrue((score.dropna() <= 100).all())
        
        # 测试不同市场环境下的评分调整
        # 牛市环境
        self.wr.set_market_environment('bull_market')
        bull_score = self.wr.calculate_score()
        
        # 熊市环境
        self.wr.set_market_environment('bear_market')
        bear_score = self.wr.calculate_score()
        
        # 检查市场环境是否影响评分
        self.assertFalse(bull_score.equals(bear_score))

    def test_signal_generation(self):
        """测试信号生成功能"""
        # 计算WR并生成信号
        signals = self.wr.generate_signals(self.test_data)
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['wr', 'score', 'buy_signal', 'sell_signal', 
                           'signal_type', 'signal_desc', 'confidence', 'stop_loss']
        for col in expected_columns:
            self.assertIn(col, signals.columns)
        
        # 检查信号类型
        self.assertTrue(signals['buy_signal'].dtype == bool)
        self.assertTrue(signals['sell_signal'].dtype == bool)
        
        # 检查置信度范围
        self.assertTrue((signals['confidence'].dropna() >= 0).all())
        self.assertTrue((signals['confidence'].dropna() <= 100).all())
        
        # 检查止损价计算
        has_buy_signals = signals['buy_signal'].any()
        has_sell_signals = signals['sell_signal'].any()
        
        if has_buy_signals:
            buy_signals = signals[signals['buy_signal']]
            # 买入信号的止损价应该低于当前价格
            buy_stops = buy_signals['stop_loss'].dropna()
            if not buy_stops.empty:
                buy_prices = self.test_data.loc[buy_stops.index, 'close']
                self.assertTrue((buy_stops < buy_prices).all())
        
        if has_sell_signals:
            sell_signals = signals[signals['sell_signal']]
            # 卖出信号的止损价应该高于当前价格
            sell_stops = sell_signals['stop_loss'].dropna()
            if not sell_stops.empty:
                sell_prices = self.test_data.loc[sell_stops.index, 'close']
                self.assertTrue((sell_stops > sell_prices).all())


if __name__ == '__main__':
    unittest.main() 