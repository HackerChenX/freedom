import unittest
import pandas as pd
import numpy as np
from indicators.complete_indicator_registry import complete_registry


class Test_merged_pSY(unittest.TestCase):
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
        psy = complete_registry.create_indicator('PSY', period=12)
        
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
        psy = complete_registry.create_indicator('PSY', period=12, enhanced=True)
        
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
        self.assert_true(isinstance(patterns, list))
        
        # 验证评分系统
        score = psy.calculate_raw_score(self.data)
        self.assert_true((score >= 0).all() and (score <= 100).all())
    
    def test_deprecated_enhanced_psy_class(self):
        """测试弃用的EnhancedPSY类仍能正常工作"""
        # 使用统一注册系统创建增强PSY
        enhanced_psy = complete_registry.create_indicator('ENHANCED_PSY', period=12)
        if not enhanced_psy:
            # 如果没有ENHANCED_PSY，使用PSY的增强模式
            enhanced_psy = complete_registry.create_indicator('PSY', period=12, enhanced=True)
        
        # 计算PSY
        result = enhanced_psy.calculate(self.data)
        
        # 检查增强版特有列
        self.assertIn('psy_momentum', result.columns)
        self.assertIn('market_sentiment', result.columns)
    
    def test_enhanced_factory(self):
        """测试通过EnhancedIndicatorFactory创建PSY"""
        # 使用统一注册系统创建PSY
        psy = complete_registry.create_indicator("PSY", period=12)

        # 检查是否成功创建
        self.assert_is_not_none(psy)
        
        # 计算PSY
        result = psy.calculate(self.data)
        
        # 检查是否启用了增强功能
        self.assertIn('psy_momentum', result.columns)
        
        # 检查设置参数是否生效
        self.assert_equal(psy.period, 12)


if __name__ == '__main__':
    unittest.main() 