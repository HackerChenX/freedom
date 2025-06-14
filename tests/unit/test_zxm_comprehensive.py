"""
ZXM指标综合测试模块
测试所有修复的ZXM指标的基本功能
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入所有ZXM指标
from indicators.zxm.trend_indicators import (
    ZXMDailyTrendUp, ZXMWeeklyTrendUp, ZXMMonthlyKDJTrendUp,
    ZXMWeeklyKDJDOrDEATrendUp, ZXMWeeklyKDJDTrendUp, ZXMMonthlyMACD,
    TrendDetector, TrendDuration, ZXMWeeklyMACD
)
from indicators.zxm.buy_point_indicators import (
    ZXMDailyMACD, ZXMTurnover, ZXMVolumeShrink,
    ZXMMACallback, ZXMBSAbsorb
)
from indicators.zxm.elasticity_indicators import (
    AmplitudeElasticity, ZXMRiseElasticity, Elasticity
)
from indicators.zxm.score_indicators import (
    ZXMElasticityScore, ZXMBuyPointScore, StockScoreCalculator
)
from indicators.zxm.selection_model import SelectionModel
from indicators.zxm.diagnostics import ZXMDiagnostics


class TestZXMComprehensive(unittest.TestCase):
    """ZXM指标综合测试类"""
    
    def setUp(self):
        """设置测试数据"""
        # 生成测试数据
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        np.random.seed(42)
        
        # 生成价格数据
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 150)
        prices = [base_price]
        
        for i in range(1, 150):
            price = prices[-1] * (1 + returns[i])
            prices.append(max(price, base_price * 0.5))
        
        # 生成OHLCV数据
        self.test_data = pd.DataFrame({
            'datetime': dates,
            'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 + np.random.uniform(-0.02, 0)) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 150),
            'code': ['TEST001'] * 150,
            'name': ['测试股票'] * 150,
            'level': [1] * 150,
            'industry': ['科技'] * 150,
            'seq': range(150),
            'turnover': [p * v for p, v in zip(prices, np.random.randint(1000000, 5000000, 150))],
            'turnover_rate': np.random.uniform(0.1, 3.0, 150),
            'price_change': np.random.uniform(-5, 5, 150),
            'price_range': np.random.uniform(1, 10, 150)
        })
        
        # 确保OHLC逻辑正确
        for i in range(len(self.test_data)):
            high = max(self.test_data.loc[i, 'open'], self.test_data.loc[i, 'high'], self.test_data.loc[i, 'close'])
            low = min(self.test_data.loc[i, 'open'], self.test_data.loc[i, 'low'], self.test_data.loc[i, 'close'])
            self.test_data.loc[i, 'high'] = high
            self.test_data.loc[i, 'low'] = low
    
    def test_all_zxm_trend_indicators(self):
        """测试所有ZXM趋势指标"""
        trend_indicators = [
            ZXMDailyTrendUp(),
            ZXMWeeklyTrendUp(),
            ZXMMonthlyKDJTrendUp(),
            ZXMWeeklyKDJDOrDEATrendUp(),
            ZXMWeeklyKDJDTrendUp(),
            ZXMMonthlyMACD(),
            TrendDetector(),
            TrendDuration(),
            ZXMWeeklyMACD()
        ]
        
        for indicator in trend_indicators:
            with self.subTest(indicator=indicator.name):
                try:
                    # 测试基本计算
                    result = indicator.calculate(self.test_data)
                    self.assertIsInstance(result, pd.DataFrame)
                    
                    # 测试评分
                    score = indicator.calculate_raw_score(self.test_data)
                    self.assertIsInstance(score, pd.Series)
                    self.assertTrue(all(0 <= s <= 100 for s in score))
                    
                    # 测试抽象方法
                    patterns = indicator.identify_patterns(self.test_data)
                    self.assertIsInstance(patterns, list)
                    
                    confidence = indicator.calculate_confidence(score, patterns, {})
                    self.assertIsInstance(confidence, float)
                    self.assertTrue(0 <= confidence <= 1)
                    
                    print(f"✅ {indicator.name} 测试通过")
                    
                except Exception as e:
                    self.fail(f"{indicator.name} 测试失败: {e}")
    
    def test_all_zxm_buy_point_indicators(self):
        """测试所有ZXM买点指标"""
        buy_point_indicators = [
            ZXMDailyMACD(),
            ZXMTurnover(),
            ZXMVolumeShrink(),
            ZXMMACallback(),
            ZXMBSAbsorb()
        ]
        
        for indicator in buy_point_indicators:
            with self.subTest(indicator=indicator.name):
                try:
                    # 测试基本计算
                    result = indicator.calculate(self.test_data)
                    self.assertIsInstance(result, pd.DataFrame)
                    
                    # 测试评分
                    score = indicator.calculate_raw_score(self.test_data)
                    self.assertIsInstance(score, pd.Series)
                    self.assertTrue(all(0 <= s <= 100 for s in score))
                    
                    # 测试抽象方法
                    patterns = indicator.identify_patterns(self.test_data)
                    self.assertIsInstance(patterns, list)
                    
                    confidence = indicator.calculate_confidence(score, patterns, {})
                    self.assertIsInstance(confidence, float)
                    self.assertTrue(0 <= confidence <= 1)
                    
                    print(f"✅ {indicator.name} 测试通过")
                    
                except Exception as e:
                    self.fail(f"{indicator.name} 测试失败: {e}")
    
    def test_all_zxm_elasticity_indicators(self):
        """测试所有ZXM弹性指标"""
        elasticity_indicators = [
            AmplitudeElasticity(),
            ZXMRiseElasticity(),
            Elasticity()
        ]
        
        for indicator in elasticity_indicators:
            with self.subTest(indicator=indicator.name):
                try:
                    # 测试基本计算
                    result = indicator.calculate(self.test_data)
                    self.assertIsInstance(result, pd.DataFrame)
                    
                    # 测试评分
                    score = indicator.calculate_raw_score(self.test_data)
                    self.assertIsInstance(score, pd.Series)
                    self.assertTrue(all(0 <= s <= 100 for s in score))
                    
                    # 测试抽象方法
                    patterns = indicator.identify_patterns(self.test_data)
                    self.assertIsInstance(patterns, list)
                    
                    confidence = indicator.calculate_confidence(score, patterns, {})
                    self.assertIsInstance(confidence, float)
                    self.assertTrue(0 <= confidence <= 1)
                    
                    print(f"✅ {indicator.name} 测试通过")
                    
                except Exception as e:
                    self.fail(f"{indicator.name} 测试失败: {e}")
    
    def test_all_zxm_score_indicators(self):
        """测试所有ZXM评分指标"""
        score_indicators = [
            ZXMElasticityScore(),
            ZXMBuyPointScore(),
            StockScoreCalculator()
        ]
        
        for indicator in score_indicators:
            with self.subTest(indicator=indicator.name):
                try:
                    # 测试基本计算
                    result = indicator.calculate(self.test_data)
                    self.assertIsInstance(result, pd.DataFrame)
                    
                    # 测试评分
                    score = indicator.calculate_raw_score(self.test_data)
                    self.assertIsInstance(score, pd.Series)
                    self.assertTrue(all(0 <= s <= 100 for s in score))
                    
                    # 测试抽象方法
                    patterns = indicator.identify_patterns(self.test_data)
                    self.assertIsInstance(patterns, list)
                    
                    confidence = indicator.calculate_confidence(score, patterns, {})
                    self.assertIsInstance(confidence, float)
                    self.assertTrue(0 <= confidence <= 1)
                    
                    print(f"✅ {indicator.name} 测试通过")
                    
                except Exception as e:
                    self.fail(f"{indicator.name} 测试失败: {e}")
    
    def test_zxm_selection_model(self):
        """测试ZXM选股模型"""
        indicator = SelectionModel()

        try:
            # 只测试抽象方法的存在性，不测试复杂的计算逻辑
            # 测试set_parameters方法
            indicator.set_parameters(selection_threshold=80)
            self.assertEqual(indicator.selection_threshold, 80)

            # 测试calculate_confidence方法
            score_series = pd.Series([60, 70, 80])
            patterns_list = ["选股系统买入信号"]
            confidence = indicator.calculate_confidence(score_series, patterns_list, {})
            self.assertIsInstance(confidence, float)
            self.assertTrue(0 <= confidence <= 1)

            print(f"✅ {indicator.name} 抽象方法测试通过")

        except Exception as e:
            self.fail(f"{indicator.name} 测试失败: {e}")
    
    def test_zxm_diagnostics(self):
        """测试ZXM诊断指标"""
        indicator = ZXMDiagnostics()
        
        try:
            # 测试基本计算
            result = indicator.calculate(self.test_data)
            self.assertIsInstance(result, pd.DataFrame)
            
            # 测试评分
            score_result = indicator.calculate_raw_score(self.test_data)
            self.assertIsInstance(score_result, pd.DataFrame)
            self.assertIn('raw_score', score_result.columns)
            self.assertTrue(all(0 <= s <= 100 for s in score_result['raw_score']))
            
            # 测试抽象方法
            score_series = score_result['raw_score']
            confidence = indicator.calculate_confidence(score_series, [], {})
            self.assertIsInstance(confidence, float)
            self.assertTrue(0 <= confidence <= 1)
            
            # 测试信号生成
            signals = indicator.generate_signals(self.test_data)
            self.assertIsInstance(signals, pd.DataFrame)
            
            print(f"✅ {indicator.name} 测试通过")
            
        except Exception as e:
            self.fail(f"{indicator.name} 测试失败: {e}")


if __name__ == '__main__':
    unittest.main()
