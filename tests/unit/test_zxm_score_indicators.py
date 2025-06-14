"""
ZXM评分指标测试模块
测试ZXM体系中的评分相关指标
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from indicators.zxm.score_indicators import (
    ZXMElasticityScore, ZXMBuyPointScore, StockScoreCalculator
)


class TestZXMScoreIndicators(unittest.TestCase):
    """ZXM评分指标测试类"""
    
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
    
    def test_zxm_elasticity_score(self):
        """测试ZXM弹性评分指标"""
        indicator = ZXMElasticityScore()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('Signal', result.columns)
        self.assertIn('ElasticityScore', result.columns)
        self.assertIn('ElasticityCount', result.columns)
        self.assertIn('AmplitudeElasticity', result.columns)
        self.assertIn('RiseElasticity', result.columns)
        
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
        
        print(f"✅ ZXM弹性评分指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_zxm_buy_point_score(self):
        """测试ZXM买点评分指标"""
        indicator = ZXMBuyPointScore()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('Signal', result.columns)
        self.assertIn('BuyPointScore', result.columns)
        self.assertIn('BuyPointCount', result.columns)
        self.assertIn('MACDBuyPoint', result.columns)
        self.assertIn('TurnoverBuyPoint', result.columns)
        self.assertIn('MACallbackBuyPoint', result.columns)
        
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
        
        print(f"✅ ZXM买点评分指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_stock_score_calculator(self):
        """测试ZXM股票综合评分指标"""
        indicator = StockScoreCalculator()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('TotalScore', result.columns)
        self.assertIn('TrendScore', result.columns)
        self.assertIn('MomentumScore', result.columns)
        self.assertIn('VolatilityScore', result.columns)
        self.assertIn('VolumeScore', result.columns)
        self.assertIn('BuySignal', result.columns)
        self.assertIn('SellSignal', result.columns)
        
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
        self.assertIn('sell_signal', signals.columns)
        self.assertIn('score', signals.columns)
        
        print(f"✅ ZXM股票综合评分指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_all_indicators_integration(self):
        """测试所有评分指标的集成"""
        indicators = [
            ZXMElasticityScore(),
            ZXMBuyPointScore(),
            StockScoreCalculator()
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
        
        print(f"✅ 所有{len(indicators)}个ZXM评分指标集成测试通过")
    
    def test_score_calculation_logic(self):
        """测试评分计算逻辑"""
        # 测试弹性评分逻辑
        elasticity_indicator = ZXMElasticityScore()
        elasticity_result = elasticity_indicator.calculate(self.test_data)
        
        # 验证弹性评分计算逻辑
        for i in range(len(elasticity_result)):
            count = elasticity_result['ElasticityCount'].iloc[i]
            expected_score = (count / 2) * 100  # 2个弹性指标
            actual_score = elasticity_result['ElasticityScore'].iloc[i]
            self.assertAlmostEqual(actual_score, expected_score, places=1)
        
        # 测试买点评分逻辑
        buy_point_indicator = ZXMBuyPointScore()
        buy_point_result = buy_point_indicator.calculate(self.test_data)
        
        # 验证买点评分计算逻辑
        for i in range(len(buy_point_result)):
            count = buy_point_result['BuyPointCount'].iloc[i]
            expected_score = (count / 3) * 100  # 3个买点指标
            actual_score = buy_point_result['BuyPointScore'].iloc[i]
            self.assertAlmostEqual(actual_score, expected_score, places=1)
        
        print("✅ 评分计算逻辑验证通过")
    
    def test_signal_threshold_logic(self):
        """测试信号阈值逻辑"""
        # 测试弹性评分信号阈值
        elasticity_indicator = ZXMElasticityScore(threshold=75)
        elasticity_result = elasticity_indicator.calculate(self.test_data)
        
        # 验证信号生成逻辑
        for i in range(len(elasticity_result)):
            score = elasticity_result['ElasticityScore'].iloc[i]
            signal = elasticity_result['Signal'].iloc[i]
            expected_signal = score >= 75
            self.assertEqual(signal, expected_signal)
        
        # 测试买点评分信号阈值
        buy_point_indicator = ZXMBuyPointScore(threshold=75)
        buy_point_result = buy_point_indicator.calculate(self.test_data)
        
        # 验证信号生成逻辑
        for i in range(len(buy_point_result)):
            score = buy_point_result['BuyPointScore'].iloc[i]
            signal = buy_point_result['Signal'].iloc[i]
            expected_signal = score >= 75
            self.assertEqual(signal, expected_signal)
        
        print("✅ 信号阈值逻辑验证通过")


if __name__ == '__main__':
    unittest.main()
