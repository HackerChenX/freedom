"""
ZXM弹性指标测试模块
测试ZXM体系中的弹性相关指标
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from indicators.zxm.elasticity_indicators import (
    AmplitudeElasticity, ZXMRiseElasticity, Elasticity
)


class TestZXMElasticityIndicators(unittest.TestCase):
    """ZXM弹性指标测试类"""
    
    def setUp(self):
        """设置测试数据"""
        # 生成测试数据
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        # 生成价格数据
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 200)
        prices = [base_price]
        
        for i in range(1, 200):
            price = prices[-1] * (1 + returns[i])
            prices.append(max(price, base_price * 0.5))
        
        # 生成OHLCV数据
        self.test_data = pd.DataFrame({
            'datetime': dates,
            'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 + np.random.uniform(-0.02, 0)) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 200),
            'code': ['TEST001'] * 200,
            'name': ['测试股票'] * 200,
            'level': [1] * 200,
            'industry': ['科技'] * 200,
            'seq': range(200),
            'turnover': [p * v for p, v in zip(prices, np.random.randint(1000000, 5000000, 200))],
            'turnover_rate': np.random.uniform(0.1, 3.0, 200),
            'price_change': np.random.uniform(-5, 5, 200),
            'price_range': np.random.uniform(1, 10, 200)
        })
        
        # 确保OHLC逻辑正确
        for i in range(len(self.test_data)):
            high = max(self.test_data.loc[i, 'open'], self.test_data.loc[i, 'high'], self.test_data.loc[i, 'close'])
            low = min(self.test_data.loc[i, 'open'], self.test_data.loc[i, 'low'], self.test_data.loc[i, 'close'])
            self.test_data.loc[i, 'high'] = high
            self.test_data.loc[i, 'low'] = low
    
    def test_amplitude_elasticity(self):
        """测试ZXM振幅弹性指标"""
        indicator = AmplitudeElasticity()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('XG', result.columns)
        self.assertIn('Amplitude', result.columns)
        self.assertIn('A1', result.columns)
        
        # 测试评分功能
        score = indicator.calculate_raw_score(self.test_data)
        self.assertIsInstance(score, pd.Series)
        self.assertTrue(all(0 <= s <= 100 for s in score))
        
        # 测试形态识别
        patterns = indicator.identify_patterns(self.test_data)
        self.assertIsInstance(patterns, list)
        
        # 测试置信度计算
        confidence = indicator.calculate_confidence(score, patterns, {})
        self.assertIsInstance(confidence, float)
        self.assertTrue(0 <= confidence <= 1)
        
        # 测试形态获取
        patterns_df = indicator.get_patterns(self.test_data)
        self.assertIsInstance(patterns_df, pd.DataFrame)
        
        print(f"✅ ZXM振幅弹性指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_zxm_rise_elasticity(self):
        """测试ZXM涨幅弹性指标"""
        indicator = ZXMRiseElasticity()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('XG', result.columns)
        self.assertIn('RiseRatio', result.columns)
        self.assertIn('A1', result.columns)
        
        # 测试评分功能
        score = indicator.calculate_raw_score(self.test_data)
        self.assertIsInstance(score, pd.Series)
        self.assertTrue(all(0 <= s <= 100 for s in score))
        
        # 测试形态识别
        patterns = indicator.identify_patterns(self.test_data)
        self.assertIsInstance(patterns, list)
        
        # 测试置信度计算
        confidence = indicator.calculate_confidence(score, patterns, {})
        self.assertIsInstance(confidence, float)
        self.assertTrue(0 <= confidence <= 1)
        
        print(f"✅ ZXM涨幅弹性指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_elasticity(self):
        """测试ZXM弹性指标"""
        indicator = Elasticity()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('BuySignal', result.columns)
        self.assertIn('ElasticityRatio', result.columns)
        self.assertIn('BounceStrength', result.columns)
        self.assertIn('VolumeRatio', result.columns)
        
        # 测试评分功能
        score = indicator.calculate_raw_score(self.test_data)
        self.assertIsInstance(score, pd.Series)
        self.assertTrue(all(0 <= s <= 100 for s in score))
        
        # 测试形态识别
        patterns = indicator.identify_patterns(self.test_data)
        self.assertIsInstance(patterns, list)
        
        # 测试置信度计算
        confidence = indicator.calculate_confidence(score, patterns, {})
        self.assertIsInstance(confidence, float)
        self.assertTrue(0 <= confidence <= 1)
        
        # 测试信号生成
        signals = indicator.generate_signals(self.test_data)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('score', signals.columns)
        
        print(f"✅ ZXM弹性指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_all_indicators_integration(self):
        """测试所有弹性指标的集成"""
        indicators = [
            AmplitudeElasticity(),
            ZXMRiseElasticity(),
            Elasticity()
        ]
        
        results = {}
        scores = {}
        
        for indicator in indicators:
            try:
                result = indicator.calculate(self.test_data)
                score = indicator.calculate_raw_score(self.test_data)
                patterns = indicator.identify_patterns(self.test_data)
                confidence = indicator.calculate_confidence(score, patterns, {})
                
                results[indicator.name] = result
                scores[indicator.name] = score
                
                # 验证基本要求
                self.assertIsInstance(result, pd.DataFrame)
                self.assertIsInstance(score, pd.Series)
                self.assertIsInstance(patterns, list)
                self.assertIsInstance(confidence, float)
                self.assertTrue(0 <= confidence <= 1)
                self.assertTrue(all(0 <= s <= 100 for s in score))
                
            except Exception as e:
                self.fail(f"指标 {indicator.name} 测试失败: {e}")
        
        print(f"✅ 所有{len(indicators)}个ZXM弹性指标集成测试通过")
    
    def test_amplitude_calculation_accuracy(self):
        """测试振幅计算准确性"""
        # 创建简单测试数据
        test_data = pd.DataFrame({
            'high': [110, 115, 120],
            'low': [100, 105, 110],
            'close': [105, 110, 115],
            'open': [102, 107, 112],
            'volume': [1000000, 1100000, 1200000]
        })
        
        indicator = AmplitudeElasticity()
        result = indicator.calculate(test_data)
        
        # 验证振幅计算
        expected_amplitude_0 = 100 * (110 - 100) / 100  # 10%
        expected_amplitude_1 = 100 * (115 - 105) / 105  # 9.52%
        expected_amplitude_2 = 100 * (120 - 110) / 110  # 9.09%
        
        self.assertAlmostEqual(result['Amplitude'].iloc[0], expected_amplitude_0, places=2)
        self.assertAlmostEqual(result['Amplitude'].iloc[1], expected_amplitude_1, places=2)
        self.assertAlmostEqual(result['Amplitude'].iloc[2], expected_amplitude_2, places=2)
        
        print("✅ 振幅计算准确性验证通过")
    
    def test_rise_ratio_calculation_accuracy(self):
        """测试涨幅比率计算准确性"""
        # 创建简单测试数据
        test_data = pd.DataFrame({
            'close': [100, 107, 115],  # 7%和7.48%的涨幅
            'high': [102, 109, 117],
            'low': [98, 105, 113],
            'open': [99, 106, 114],
            'volume': [1000000, 1100000, 1200000]
        })
        
        indicator = ZXMRiseElasticity()
        result = indicator.calculate(test_data)
        
        # 验证涨幅比率计算
        expected_ratio_1 = 107 / 100  # 1.07
        expected_ratio_2 = 115 / 107  # 1.0748
        
        self.assertAlmostEqual(result['RiseRatio'].iloc[1], expected_ratio_1, places=3)
        self.assertAlmostEqual(result['RiseRatio'].iloc[2], expected_ratio_2, places=3)
        
        # 验证A1条件（涨幅>7%）
        self.assertFalse(result['A1'].iloc[1])  # 7%涨幅，等于1.07，不满足>1.07
        self.assertTrue(result['A1'].iloc[2])  # 7.48%涨幅，大于1.07，应该满足
        
        print("✅ 涨幅比率计算准确性验证通过")


if __name__ == '__main__':
    unittest.main()
