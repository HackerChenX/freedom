import unittest
import pandas as pd
import numpy as np
from indicators.oscillator.enhanced_stochrsi import EnhancedSTOCHRSI


class TestEnhancedSTOCHRSI(unittest.TestCase):
    """测试增强型随机相对强弱指标(STOCHRSI)"""

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
        
        # 创建增强型STOCHRSI实例
        self.stochrsi = EnhancedSTOCHRSI(n=14, m=3, p=3, secondary_n=28, adaptive_threshold=True)

    def test_calculation(self):
        """测试STOCHRSI基础计算功能"""
        result = self.stochrsi.calculate(self.test_data)
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['RSI', 'STOCHRSI_K', 'STOCHRSI_D', 'STOCHRSI_K_SECONDARY', 'STOCHRSI_D_SECONDARY',
                           'K_SLOPE', 'D_SLOPE', 'K_ACCEL', 'D_ACCEL', 'STOCHRSI_VOLATILITY']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # 检查多周期STOCHRSI列是否存在
        for period in self.stochrsi.multi_periods:
            if period != self.stochrsi.n and period != self.stochrsi.secondary_n:
                self.assertIn(f'STOCHRSI_K_{period}', result.columns)
                self.assertIn(f'STOCHRSI_D_{period}', result.columns)
        
        # 检查STOCHRSI值不是全NaN
        self.assertFalse(result['STOCHRSI_K'].isna().all())
        self.assertFalse(result['STOCHRSI_D'].isna().all())
        
        # 检查RSI和STOCHRSI的范围
        self.assertTrue((result['RSI'].dropna() >= 0).all())
        self.assertTrue((result['RSI'].dropna() <= 100).all())
        self.assertTrue((result['STOCHRSI_K'].dropna() >= 0).all())
        self.assertTrue((result['STOCHRSI_K'].dropna() <= 100).all())

    def test_adaptive_threshold(self):
        """测试自适应阈值调整功能"""
        # 计算STOCHRSI
        self.stochrsi.calculate(self.test_data)
        
        # 记录原始阈值
        original_overbought = self.stochrsi.overbought_threshold
        original_oversold = self.stochrsi.oversold_threshold
        
        # 测试不同市场环境下的阈值调整
        # 牛市环境
        self.stochrsi.set_market_environment('bull_market')
        self.stochrsi._adjust_thresholds_by_market(self.test_data)
        bull_overbought = self.stochrsi.overbought_threshold
        bull_oversold = self.stochrsi.oversold_threshold
        
        # 熊市环境
        self.stochrsi.set_market_environment('bear_market')
        self.stochrsi._adjust_thresholds_by_market(self.test_data)
        bear_overbought = self.stochrsi.overbought_threshold
        bear_oversold = self.stochrsi.oversold_threshold
        
        # 检查牛市环境下超买阈值是否高于原始阈值
        self.assertTrue(bull_overbought >= original_overbought)
        
        # 检查熊市环境下超卖阈值是否低于原始阈值
        self.assertTrue(bear_oversold <= original_oversold)

    def test_divergence_detection(self):
        """测试背离检测功能"""
        # 计算STOCHRSI
        self.stochrsi.calculate(self.test_data)
        
        # 获取背离结果
        divergence = self.stochrsi.detect_divergence()
        
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
        # 计算STOCHRSI
        self.stochrsi.calculate(self.test_data)
        
        # 获取多周期协同分析结果
        synergy = self.stochrsi.analyze_multi_period_synergy()
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['primary_overbought', 'primary_oversold', 
                           'primary_golden_cross', 'primary_death_cross',
                           'bullish_ratio', 'bearish_ratio', 'consensus_score']
        for col in expected_columns:
            self.assertIn(col, synergy.columns)
        
        # 检查布尔型列
        self.assertTrue(synergy['primary_overbought'].dtype == bool)
        self.assertTrue(synergy['primary_oversold'].dtype == bool)
        
        # 检查比率范围
        self.assertTrue((synergy['bullish_ratio'] >= 0).all())
        self.assertTrue((synergy['bullish_ratio'] <= 1).all())
        self.assertTrue((synergy['bearish_ratio'] >= 0).all())
        self.assertTrue((synergy['bearish_ratio'] <= 1).all())
        
        # 检查共识得分范围
        self.assertTrue((synergy['consensus_score'].dropna() >= 0).all())
        self.assertTrue((synergy['consensus_score'].dropna() <= 100).all())

    def test_cross_quality(self):
        """测试交叉质量评估功能"""
        # 计算STOCHRSI
        self.stochrsi.calculate(self.test_data)
        
        # 获取交叉质量评估结果
        quality = self.stochrsi.evaluate_cross_quality()
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['golden_cross', 'death_cross', 'cross_angle', 
                           'cross_position_score', 'separation_speed', 'cross_quality_score']
        for col in expected_columns:
            self.assertIn(col, quality.columns)
        
        # 检查布尔型列
        self.assertTrue(quality['golden_cross'].dtype == bool)
        self.assertTrue(quality['death_cross'].dtype == bool)
        
        # 检查评分范围
        max_score = quality['cross_quality_score'].max()
        if not pd.isna(max_score):
            self.assertTrue(max_score <= 100)

    def test_pattern_identification(self):
        """测试形态识别功能"""
        # 计算STOCHRSI
        self.stochrsi.calculate(self.test_data)
        
        # 获取形态识别结果
        patterns = self.stochrsi.identify_patterns()
        
        # 检查基础形态列是否存在
        basic_patterns = ['overbought', 'oversold', 'extreme_overbought', 'extreme_oversold',
                         'golden_cross', 'death_cross', 'oversold_golden_cross', 'overbought_death_cross']
        for pattern in basic_patterns:
            self.assertIn(pattern, patterns.columns)
        
        # 检查特殊形态列是否存在
        special_patterns = ['w_bottom', 'm_top', 'overbought_stagnation', 'oversold_stagnation']
        for pattern in special_patterns:
            self.assertIn(pattern, patterns.columns)
        
        # 检查布尔型列
        for col in patterns.columns:
            self.assertTrue(patterns[col].dtype == bool)

    def test_score_calculation(self):
        """测试评分计算功能"""
        # 计算STOCHRSI
        self.stochrsi.calculate(self.test_data)
        
        # 计算评分
        score = self.stochrsi.calculate_score()
        
        # 检查评分范围
        self.assertTrue((score.dropna() >= 0).all())
        self.assertTrue((score.dropna() <= 100).all())
        
        # 测试不同市场环境下的评分调整
        # 牛市环境
        self.stochrsi.set_market_environment('bull_market')
        bull_score = self.stochrsi.calculate_score()
        
        # 熊市环境
        self.stochrsi.set_market_environment('bear_market')
        bear_score = self.stochrsi.calculate_score()
        
        # 检查市场环境是否影响评分
        self.assertFalse(bull_score.equals(bear_score))

    def test_signal_generation(self):
        """测试信号生成功能"""
        # 计算STOCHRSI并生成信号
        signals = self.stochrsi.generate_signals(self.test_data)
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['stochrsi_k', 'stochrsi_d', 'score', 'buy_signal', 'sell_signal', 'neutral_signal',
                           'signal_type', 'signal_desc', 'trend', 'confidence', 'stop_loss',
                           'market_env']
        for col in expected_columns:
            self.assertIn(col, signals.columns)
        
        # 检查信号类型
        self.assertTrue(signals['buy_signal'].dtype == bool)
        self.assertTrue(signals['sell_signal'].dtype == bool)
        self.assertTrue(signals['neutral_signal'].dtype == bool)
        
        # 检查趋势值
        self.assertTrue(set(signals['trend'].unique()).issubset({-1, 0, 1}))
        
        # 检查置信度范围
        self.assertTrue((signals['confidence'].dropna() >= 0).all())
        self.assertTrue((signals['confidence'].dropna() <= 100).all())
        
        # 检查买卖信号互斥
        self.assertFalse((signals['buy_signal'] & signals['sell_signal']).any())
        
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