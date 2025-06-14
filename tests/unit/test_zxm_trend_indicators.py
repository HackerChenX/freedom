"""
ZXM趋势指标测试模块
测试ZXM体系中的趋势相关指标
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from indicators.zxm.trend_indicators import (
    ZXMDailyTrendUp, ZXMWeeklyTrendUp, ZXMMonthlyKDJTrendUp,
    ZXMWeeklyKDJDOrDEATrendUp, ZXMWeeklyKDJDTrendUp, ZXMMonthlyMACD,
    TrendDetector, TrendDuration, ZXMWeeklyMACD
)


class TestZXMTrendIndicators(unittest.TestCase):
    """ZXM趋势指标测试类"""
    
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
            'turnover_rate': np.random.uniform(0.01, 0.1, 200),
            'price_change': np.random.uniform(-5, 5, 200),
            'price_range': np.random.uniform(1, 10, 200)
        })
        
        # 确保OHLC逻辑正确
        for i in range(len(self.test_data)):
            high = max(self.test_data.loc[i, 'open'], self.test_data.loc[i, 'high'], self.test_data.loc[i, 'close'])
            low = min(self.test_data.loc[i, 'open'], self.test_data.loc[i, 'low'], self.test_data.loc[i, 'close'])
            self.test_data.loc[i, 'high'] = high
            self.test_data.loc[i, 'low'] = low
    
    def test_zxm_daily_trend_up(self):
        """测试ZXM日线上移指标"""
        indicator = ZXMDailyTrendUp()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('XG', result.columns)
        self.assertIn('MA60', result.columns)
        self.assertIn('MA120', result.columns)
        
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
        
        print(f"✅ ZXM日线上移指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_zxm_weekly_trend_up(self):
        """测试ZXM周线上移指标"""
        indicator = ZXMWeeklyTrendUp()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('XG', result.columns)
        self.assertIn('MA10', result.columns)
        self.assertIn('MA20', result.columns)
        self.assertIn('MA30', result.columns)
        
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
        
        print(f"✅ ZXM周线上移指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_zxm_monthly_kdj_trend_up(self):
        """测试ZXM月KDJ上移指标"""
        indicator = ZXMMonthlyKDJTrendUp()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('XG', result.columns)
        self.assertIn('K', result.columns)
        self.assertIn('D', result.columns)
        self.assertIn('J', result.columns)
        
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
        
        print(f"✅ ZXM月KDJ上移指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_zxm_weekly_kdj_d_or_dea_trend_up(self):
        """测试ZXM周KDJ·D/DEA上移指标"""
        indicator = ZXMWeeklyKDJDOrDEATrendUp()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('XG', result.columns)
        
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
        
        print(f"✅ ZXM周KDJ·D/DEA上移指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_zxm_weekly_kdj_d_trend_up(self):
        """测试ZXM周KDJ·D上移指标"""
        indicator = ZXMWeeklyKDJDTrendUp()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('XG', result.columns)
        self.assertIn('K', result.columns)
        self.assertIn('D', result.columns)
        self.assertIn('J', result.columns)
        
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
        
        print(f"✅ ZXM周KDJ·D上移指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_zxm_monthly_macd(self):
        """测试ZXM月MACD指标"""
        indicator = ZXMMonthlyMACD()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('XG', result.columns)
        self.assertIn('DIF', result.columns)
        self.assertIn('DEA', result.columns)
        self.assertIn('MACD', result.columns)
        
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
        
        print(f"✅ ZXM月MACD指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_trend_detector(self):
        """测试趋势检测器"""
        indicator = TrendDetector()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('TrendState', result.columns)
        self.assertIn('TrendStrength', result.columns)
        self.assertIn('TrendHealth', result.columns)
        
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
        
        print(f"✅ 趋势检测器测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_trend_duration(self):
        """测试趋势持续性指标"""
        indicator = TrendDuration()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('TrendState', result.columns)
        self.assertIn('TrendDuration', result.columns)
        self.assertIn('TrendLifecyclePhase', result.columns)
        
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
        
        print(f"✅ 趋势持续性指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")
    
    def test_zxm_weekly_macd(self):
        """测试ZXM周线MACD指标"""
        indicator = ZXMWeeklyMACD()
        
        # 测试计算功能
        result = indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('DIF', result.columns)
        self.assertIn('DEA', result.columns)
        self.assertIn('MACD', result.columns)
        
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
        
        print(f"✅ ZXM周线MACD指标测试通过 - 评分范围: {score.min():.1f}-{score.max():.1f}, 置信度: {confidence:.3f}")


if __name__ == '__main__':
    unittest.main()
