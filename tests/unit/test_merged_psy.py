import unittest
import pandas as pd
import numpy as np
from indicators.psy import PSY, EnhancedPSY
from indicators.enhanced_factory import EnhancedIndicatorFactory


class TestMergedPSY(unittest.TestCase):
    """测试合并后的PSY指标功能"""
    
    def setUp(self):
        """准备测试数据"""
        # 创建样本数据
        np.random.seed(42)
        n = 100
        self.data = pd.DataFrame({
            'close': np.cumsum(np.random.normal(0, 1, n)) + 100,
            'high': np.cumsum(np.random.normal(0, 1, n)) + 102,
            'low': np.cumsum(np.random.normal(0, 1, n)) + 98,
            'volume': np.random.randint(1000, 10000, n)
        })
    
    def test_basic_psy(self):
        """测试基础PSY功能"""
        # 创建普通PSY指标实例
        psy = PSY(period=12)
        
        # 计算PSY
        result = psy.calculate(self.data)
        
        # 检查结果
        self.assertIn('psy', result.columns)
        self.assertIn('psyma', result.columns)
        
        # 验证PSY值范围在0-100之间
        self.assertTrue((result['psy'].dropna() >= 0).all())
        self.assertTrue((result['psy'].dropna() <= 100).all())
    
    def test_enhanced_psy(self):
        """测试增强版PSY功能"""
        # 创建增强版PSY指标实例
        psy = PSY(period=12, enhanced=True)
        
        # 计算PSY
        result = psy.calculate(self.data)
        
        # 检查基础列
        self.assertIn('psy', result.columns)
        self.assertIn('psyma', result.columns)
        
        # 检查增强版特有列
        self.assertIn('psy_momentum', result.columns)
        self.assertIn('psy_slope', result.columns)
        self.assertIn('market_sentiment', result.columns)
        
        # 测试多周期协同分析
        synergy = psy.analyze_multi_period_synergy()
        self.assertIn('synergy_score', synergy.columns)
        
        # 验证模式识别
        patterns = psy.identify_patterns(self.data)
        self.assertTrue(isinstance(patterns, list))
        
        # 验证评分系统
        score = psy.calculate_raw_score(self.data)
        self.assertTrue((score >= 0).all() and (score <= 100).all())
    
    def test_deprecated_enhanced_psy_class(self):
        """测试弃用的EnhancedPSY类仍能正常工作"""
        # 使用旧的EnhancedPSY类
        enhanced_psy = EnhancedPSY(period=12)
        
        # 计算PSY
        result = enhanced_psy.calculate(self.data)
        
        # 检查增强版特有列
        self.assertIn('psy_momentum', result.columns)
        self.assertIn('market_sentiment', result.columns)
    
    def test_enhanced_factory(self):
        """测试通过EnhancedIndicatorFactory创建PSY"""
        # 使用工厂创建PSY
        psy = EnhancedIndicatorFactory.create("PSY", period=12)
        
        # 检查是否成功创建
        self.assertIsNotNone(psy)
        self.assertTrue(isinstance(psy, PSY))
        
        # 计算PSY
        result = psy.calculate(self.data)
        
        # 检查是否启用了增强功能
        self.assertIn('psy_momentum', result.columns)
        
        # 检查设置参数是否生效
        self.assertEqual(psy.period, 12)


if __name__ == '__main__':
    unittest.main() 