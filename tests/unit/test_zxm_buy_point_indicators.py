"""
ZXM买点指标测试模块
测试ZXM体系中的买点相关指标
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from indicators.zxm.buy_point_indicators import (
    ZXMDailyMACD, ZXMTurnover, ZXMVolumeShrink,
    ZXMMACallback, ZXMBSAbsorb
)


class TestZXMBuyPointIndicators(unittest.TestCase):
    """ZXM买点指标测试类"""
    
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
    
    def test_zxm_daily_macd(self):
        """测试ZXM日线MACD指标"""
        indicator = ZXMDailyMACD()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('XG', result.columns)
        self.assertIn('MACD', result.columns)
        self.assertIn('DIFF', result.columns)
        self.assertIn('DEA', result.columns)
        
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
        
        print(f"✅ ZXM日线MACD指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_zxm_turnover(self):
        """测试ZXM换手率指标"""
        indicator = ZXMTurnover()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('XG', result.columns)
        self.assertIn('Turnover', result.columns)
        
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
        
        print(f"✅ ZXM换手率指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_zxm_volume_shrink(self):
        """测试ZXM缩量指标"""
        indicator = ZXMVolumeShrink()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('XG', result.columns)
        self.assertIn('VOL_RATIO', result.columns)
        self.assertIn('MA_VOL_2', result.columns)
        
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
        
        print(f"✅ ZXM缩量指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_zxm_ma_callback(self):
        """测试ZXM均线回调指标"""
        indicator = ZXMMACallback()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('XG', result.columns)
        self.assertIn('A20', result.columns)
        self.assertIn('A30', result.columns)
        self.assertIn('A60', result.columns)
        self.assertIn('A120', result.columns)
        
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
        
        print(f"✅ ZXM均线回调指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_zxm_bs_absorb(self):
        """测试ZXM BS吸筹指标"""
        indicator = ZXMBSAbsorb()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('XG', result.columns)
        self.assertIn('V11', result.columns)
        self.assertIn('V12', result.columns)
        self.assertIn('AA', result.columns)
        self.assertIn('BB', result.columns)
        
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
        
        print(f"✅ ZXM BS吸筹指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_all_indicators_integration(self):
        """测试所有买点指标的集成"""
        indicators = [
            ZXMDailyMACD(),
            ZXMTurnover(),
            ZXMVolumeShrink(),
            ZXMMACallback(),
            ZXMBSAbsorb()
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
        
        print(f"✅ 所有{len(indicators)}个ZXM买点指标集成测试通过")


if __name__ == '__main__':
    unittest.main()
