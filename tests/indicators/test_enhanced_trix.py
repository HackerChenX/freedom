import unittest
import pandas as pd
import numpy as np
from indicators.trend.enhanced_trix import EnhancedTRIX


class TestEnhancedTRIX(unittest.TestCase):
    """测试增强型TRIX三重指数平滑移动平均线指标"""

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
        
        # 创建增强型TRIX实例
        self.trix = EnhancedTRIX(n=12, m=9, secondary_n=24, adaptive_period=True)

    def test_calculation(self):
        """测试TRIX基础计算功能"""
        result = self.trix.calculate(self.test_data)
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['TR', 'TRIX', 'MATRIX', 'trix_secondary', 'matrix_secondary', 
                           'trix_momentum', 'trix_slope', 'trix_accel', 'trix_volatility']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # 检查多周期TRIX列是否存在
        for period in self.trix.multi_periods:
            if period != self.trix._adaptive_n and period != self.trix.secondary_n:
                self.assertIn(f'trix_{period}', result.columns)
                self.assertIn(f'matrix_{period}', result.columns)
        
        # 检查TRIX值不是全NaN
        self.assertFalse(result['TRIX'].isna().all())
        self.assertFalse(result['MATRIX'].isna().all())

    def test_adaptive_period(self):
        """测试自适应周期调整功能"""
        # 计算TRIX
        self.trix.calculate(self.test_data)
        
        # 检查自适应周期是否被调整
        self.assertIsNotNone(self.trix._adaptive_n)
        
        # 测试不同市场环境下的周期调整
        # 牛市环境
        self.trix.set_market_environment('bull_market')
        original_n = self.trix._adaptive_n
        self.trix._adjust_period_by_volatility(self.test_data)
        bull_n = self.trix._adaptive_n
        
        # 熊市环境
        self.trix.set_market_environment('bear_market')
        self.trix._adjust_period_by_volatility(self.test_data)
        bear_n = self.trix._adaptive_n
        
        # 检查牛市环境下周期是否小于等于熊市环境
        self.assertTrue(bull_n <= bear_n)

    def test_divergence_detection(self):
        """测试背离检测功能"""
        # 计算TRIX
        self.trix.calculate(self.test_data)
        
        # 获取背离结果
        divergence = self.trix.detect_divergence()
        
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
        # 计算TRIX
        self.trix.calculate(self.test_data)
        
        # 获取多周期协同分析结果
        synergy = self.trix.analyze_multi_period_synergy()
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['primary_above_zero', 'primary_below_zero', 
                           'primary_rising', 'primary_falling',
                           'bullish_ratio', 'bearish_ratio', 'consensus_score']
        for col in expected_columns:
            self.assertIn(col, synergy.columns)
        
        # 检查布尔型列
        self.assertTrue(synergy['primary_above_zero'].dtype == bool)
        self.assertTrue(synergy['primary_below_zero'].dtype == bool)
        
        # 检查比率范围
        self.assertTrue((synergy['bullish_ratio'] >= 0).all())
        self.assertTrue((synergy['bullish_ratio'] <= 1).all())
        self.assertTrue((synergy['bearish_ratio'] >= 0).all())
        self.assertTrue((synergy['bearish_ratio'] <= 1).all())
        
        # 检查共识得分范围
        self.assertTrue((synergy['consensus_score'].dropna() >= 0).all())
        self.assertTrue((synergy['consensus_score'].dropna() <= 100).all())

    def test_zero_cross_quality(self):
        """测试零轴交叉质量评估功能"""
        # 计算TRIX
        self.trix.calculate(self.test_data)
        
        # 获取零轴交叉质量评估结果
        quality = self.trix.evaluate_zero_cross_quality()
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['cross_up_zero', 'cross_down_zero', 'cross_angle', 
                           'post_cross_acceleration', 'cross_persistence', 'cross_quality_score']
        for col in expected_columns:
            self.assertIn(col, quality.columns)
        
        # 检查布尔型列
        self.assertTrue(quality['cross_up_zero'].dtype == bool)
        self.assertTrue(quality['cross_down_zero'].dtype == bool)
        
        # 检查评分范围
        max_score = quality['cross_quality_score'].max()
        if not pd.isna(max_score):
            self.assertTrue(max_score <= 100)

    def test_pattern_identification(self):
        """测试形态识别功能"""
        # 计算TRIX
        self.trix.calculate(self.test_data)
        
        # 获取形态识别结果
        patterns = self.trix.identify_patterns()
        
        # 检查基础形态列是否存在
        basic_patterns = ['above_zero', 'below_zero', 'rising', 'falling',
                         'golden_cross', 'death_cross', 'cross_up_zero', 'cross_down_zero']
        for pattern in basic_patterns:
            self.assertIn(pattern, patterns.columns)
        
        # 检查布尔型列
        for col in patterns.columns:
            self.assertTrue(patterns[col].dtype == bool)
        
        # 检查钝化形态识别
        self.assertIn('stagnation_near_zero', patterns.columns)

    def test_score_calculation(self):
        """测试评分计算功能"""
        # 计算TRIX
        self.trix.calculate(self.test_data)
        
        # 计算评分
        score = self.trix.calculate_score()
        
        # 检查评分范围
        self.assertTrue((score.dropna() >= 0).all())
        self.assertTrue((score.dropna() <= 100).all())
        
        # 测试不同市场环境下的评分调整
        # 牛市环境
        self.trix.set_market_environment('bull_market')
        bull_score = self.trix.calculate_score()
        
        # 熊市环境
        self.trix.set_market_environment('bear_market')
        bear_score = self.trix.calculate_score()
        
        # 检查市场环境是否影响评分
        self.assertFalse(bull_score.equals(bear_score))

    def test_signal_generation(self):
        """测试信号生成功能"""
        # 计算TRIX并生成信号
        signals = self.trix.generate_signals(self.test_data)
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['trix', 'score', 'buy_signal', 'sell_signal', 'neutral_signal',
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