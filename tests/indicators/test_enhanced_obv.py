import unittest
import pandas as pd
import numpy as np
from indicators.volume.enhanced_obv import EnhancedOBV


class TestEnhancedOBV(unittest.TestCase):
    """测试增强型OBV指标"""

    def setUp(self):
        """准备测试数据"""
        # 创建模拟价格数据
        np.random.seed(42)  # 确保结果可重现
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # 创建价格数据：混合趋势和震荡
        trend = np.linspace(0, 5, 50)
        oscillation = np.random.normal(0, 0.5, 50)
        combined = np.concatenate([trend, oscillation])
        
        # 添加随机波动
        noise = np.random.normal(0, 0.2, 100)
        price_base = combined + noise
        
        # 生成成交量数据，与价格大致相关
        volume_base = price_base * 1000 + np.random.normal(0, 500, 100)
        volume_base = np.abs(volume_base)  # 确保成交量为正
        
        # 生成OHLCV数据
        self.test_data = pd.DataFrame({
            'open': price_base,
            'high': price_base + np.random.uniform(0.1, 0.3, 100),
            'low': price_base - np.random.uniform(0.1, 0.3, 100),
            'close': price_base + np.random.normal(0, 0.1, 100),
            'volume': volume_base.astype(int)
        }, index=dates)
        
        # 创建OBV实例
        self.obv = EnhancedOBV(smooth_period=5, adaptive=True)

    def test_calculation(self):
        """测试OBV基础计算功能"""
        result = self.obv.calculate(self.test_data)
        
        # 检查结果DataFrame是否包含预期的列
        expected_columns = ['obv', 'obv_smooth', 'obv_roc', 'obv_slope', 'obv_ma10', 'obv_ma20', 'obv_ma60']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # 检查OBV是否与原始数据长度一致
        self.assertEqual(len(result), len(self.test_data))
        
        # 检查第一个OBV值是否为0
        self.assertEqual(result['obv'].iloc[0], 0)
        
        # 检查OBV计算规则是否正确
        # 价格上升日，OBV应该增加成交量；价格下降日，OBV应该减少成交量
        price_change = self.test_data['close'].diff()
        
        # 随机抽取几个点检查
        for i in range(10, 15):
            if price_change.iloc[i] > 0:
                self.assertGreaterEqual(result['obv'].iloc[i], result['obv'].iloc[i-1])
            elif price_change.iloc[i] < 0:
                self.assertLessEqual(result['obv'].iloc[i], result['obv'].iloc[i-1])
            else:
                self.assertEqual(result['obv'].iloc[i], result['obv'].iloc[i-1])

    def test_smoothing(self):
        """测试平滑处理功能"""
        self.obv.calculate(self.test_data)
        
        # 获取原始OBV和平滑后的OBV
        obv = self.obv._result['obv']
        obv_smooth = self.obv._result['obv_smooth']
        
        # 平滑后的OBV波动应该小于原始OBV
        obv_std = obv.diff().std()
        smooth_std = obv_smooth.diff().std()
        
        # 由于使用了加权移动平均，平滑后的标准差应该小于原始标准差
        self.assertLess(smooth_std, obv_std)
        
        # 测试自适应周期调整
        high_vol_data = self.test_data.copy()
        high_vol_data['volume'] = high_vol_data['volume'] * (1 + np.random.normal(0, 0.5, len(high_vol_data)))
        
        # 重新计算
        obv_adaptive = EnhancedOBV(smooth_period=5, adaptive=True)
        obv_adaptive.calculate(high_vol_data)
        
        obv_fixed = EnhancedOBV(smooth_period=5, adaptive=False)
        obv_fixed.calculate(high_vol_data)
        
        # 验证结果存在差异
        self.assertFalse(obv_adaptive._result['obv_smooth'].equals(obv_fixed._result['obv_smooth']))

    def test_flow_gradient(self):
        """测试资金流向梯度分析"""
        self.obv.calculate(self.test_data)
        flow = self.obv.calculate_flow_gradient()
        
        # 检查输出类型
        self.assertIsInstance(flow, pd.DataFrame)
        self.assertEqual(len(flow), len(self.test_data))
        
        # 检查是否包含预期的列
        expected_columns = ['gradient', 'acceleration', 'normalized_gradient', 
                           'normalized_acceleration', 'strong_inflow', 'weak_inflow',
                           'strong_outflow', 'weak_outflow', 'neutral']
        for col in expected_columns:
            self.assertIn(col, flow.columns)
        
        # 验证资金流向状态的互斥性
        flow_states = flow[['strong_inflow', 'weak_inflow', 'strong_outflow', 'weak_outflow', 'neutral']]
        
        # 每个时间点只能有一种状态为True
        self.assertTrue(((flow_states.sum(axis=1) <= 1) | (flow_states.sum(axis=1) == 0)).all())

    def test_divergence_detection(self):
        """测试背离检测功能"""
        self.obv.calculate(self.test_data)
        divergence = self.obv.detect_divergence()
        
        # 检查输出类型
        self.assertIsInstance(divergence, pd.DataFrame)
        self.assertEqual(len(divergence), len(self.test_data))
        
        # 检查是否包含预期的列
        expected_columns = ['bullish_divergence', 'bearish_divergence', 
                           'hidden_bullish_divergence', 'hidden_bearish_divergence',
                           'divergence_strength']
        for col in expected_columns:
            self.assertIn(col, divergence.columns)
        
        # 验证背离强度值的范围
        self.assertTrue((divergence['divergence_strength'] >= 0).all())
        self.assertTrue((divergence['divergence_strength'] <= 1).all())
        
        # 验证背离标记的互斥性 (正背离和负背离不应同时发生)
        self.assertFalse(((divergence['bullish_divergence'] == 1) & 
                          (divergence['bearish_divergence'] == 1)).any())

    def test_price_volume_synergy(self):
        """测试量价协同分析"""
        self.obv.calculate(self.test_data)
        synergy = self.obv.calculate_price_volume_synergy()
        
        # 检查输出类型
        self.assertIsInstance(synergy, pd.DataFrame)
        self.assertEqual(len(synergy), len(self.test_data))
        
        # 检查是否包含预期的列
        expected_columns = ['direction_synergy', 'magnitude_diff', 'ideal_up', 
                           'ideal_down', 'poor_up', 'poor_down', 'synergy_score']
        for col in expected_columns:
            self.assertIn(col, synergy.columns)
        
        # 验证协同评分的范围
        self.assertTrue((synergy['synergy_score'] >= 0).all())
        self.assertTrue((synergy['synergy_score'] <= 100).all())
        
        # 验证理想配合和不良配合的互斥性
        self.assertFalse(((synergy['ideal_up'] & synergy['poor_up']).any() or 
                          (synergy['ideal_down'] & synergy['poor_down']).any()))

    def test_score_calculation(self):
        """测试评分计算功能"""
        self.obv.calculate(self.test_data)
        score = self.obv.calculate_score()
        
        # 检查输出类型
        self.assertIsInstance(score, pd.Series)
        self.assertEqual(len(score), len(self.test_data))
        
        # 检查评分范围
        self.assertTrue((score >= 0).all())
        self.assertTrue((score <= 100).all())
        
        # 测试不同市场环境下的评分调整
        # 牛市环境
        self.obv.set_market_environment("bull_market")
        bull_score = self.obv.calculate_score()
        
        # 熊市环境
        self.obv.set_market_environment("bear_market")
        bear_score = self.obv.calculate_score()
        
        # 高波动环境
        self.obv.set_market_environment("volatile_market")
        volatile_score = self.obv.calculate_score()
        
        # 确保市场环境确实影响了评分
        self.assertTrue(not (bull_score.equals(bear_score) and bear_score.equals(volatile_score)))

    def test_pattern_identification(self):
        """测试形态识别功能"""
        self.obv.calculate(self.test_data)
        patterns = self.obv.identify_patterns()
        
        # 检查输出类型
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertEqual(len(patterns), len(self.test_data))
        
        # 检查是否包含预期的形态列
        expected_patterns = ['obv_breakout', 'obv_breakdown', 'obv_acceleration_up',
                            'obv_acceleration_down', 'obv_w_bottom', 'obv_m_top',
                            'bullish_divergence', 'bearish_divergence',
                            'hidden_bullish_divergence', 'hidden_bearish_divergence',
                            'obv_consolidation']
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns.columns)
        
        # 检查值是否为布尔型
        for col in patterns.columns:
            self.assertTrue(patterns[col].dtype == bool)

    def test_signal_generation(self):
        """测试信号生成功能"""
        signals = self.obv.generate_signals(self.test_data)
        
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
        
        # 检查评分范围
        self.assertTrue((signals['score'].dropna() >= 0).all())
        self.assertTrue((signals['score'].dropna() <= 100).all())
        
        # 检查信号置信度范围
        self.assertTrue((signals['confidence'].dropna() >= 0).all())
        self.assertTrue((signals['confidence'].dropna() <= 100).all())


if __name__ == '__main__':
    unittest.main() 