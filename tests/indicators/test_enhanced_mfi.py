#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强型资金流向指标(EnhancedMFI)测试模块
"""

import unittest
import pandas as pd
import numpy as np
from indicators.volume.enhanced_mfi import EnhancedMFI


class TestEnhancedMFI(unittest.TestCase):
    """
    测试增强型资金流向指标(EnhancedMFI)
    """
    
    def setUp(self):
        """
        准备测试数据
        """
        # 创建测试数据
        date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # 模拟价格数据
        close = np.sin(np.linspace(0, 4*np.pi, 100)) * 10 + 50  # 正弦波
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2
        
        # 模拟成交量数据（与价格变化正相关）
        volume = np.abs(np.diff(np.append(0, close))) * 10000 + 5000
        # 添加一些异常成交量
        volume[20] = volume[20] * 5  # 异常高成交量
        volume[40] = volume[40] * 4
        volume[60] = volume[60] * 6
        volume[80] = volume[80] * 3
        
        # 创建DataFrame
        self.data = pd.DataFrame({
            'open': close - np.random.rand(100) * 0.5,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=date_range)
        
        # 初始化指标
        self.mfi = EnhancedMFI(period=14)
    
    def test_calculate(self):
        """
        测试计算方法
        """
        # 计算指标
        result = self.mfi.calculate(self.data)
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('mfi', result.columns)
        self.assertIn('mfi_overbought', result.columns)
        self.assertIn('mfi_oversold', result.columns)
        self.assertIn('mfi_price_ratio', result.columns)
        self.assertIn('mfi_adjusted', result.columns)
        
        # 验证MFI值在0-100范围内
        self.assertTrue((result['mfi'] >= 0).all() and (result['mfi'] <= 100).all())
    
    def test_smooth_abnormal_volume(self):
        """
        测试异常成交量平滑功能
        """
        # 计算原始MFI
        self.mfi.enable_volume_filter = False
        original_result = self.mfi.calculate(self.data)
        
        # 计算平滑后的MFI
        self.mfi.enable_volume_filter = True
        smoothed_result = self.mfi.calculate(self.data)
        
        # 验证异常成交量的MFI值不同
        self.assertNotEqual(original_result['mfi'].iloc[21], smoothed_result['mfi'].iloc[21])
        self.assertNotEqual(original_result['mfi'].iloc[41], smoothed_result['mfi'].iloc[41])
        self.assertNotEqual(original_result['mfi'].iloc[61], smoothed_result['mfi'].iloc[61])
        self.assertNotEqual(original_result['mfi'].iloc[81], smoothed_result['mfi'].iloc[81])
    
    def test_dynamic_thresholds(self):
        """
        测试动态阈值调整功能
        """
        # 计算指标
        result = self.mfi.calculate(self.data)
        
        # 验证默认阈值
        self.assertEqual(self.mfi._dynamic_overbought, 80)
        self.assertEqual(self.mfi._dynamic_oversold, 20)
        
        # 设置高波动市场环境
        self.mfi.set_market_environment('volatile_market')
        result = self.mfi.calculate(self.data)
        
        # 验证高波动市场阈值变化
        self.assertGreater(self.mfi._dynamic_overbought, 80)
        self.assertLess(self.mfi._dynamic_oversold, 20)
        
        # 设置牛市环境
        self.mfi.set_market_environment('bull_market')
        result = self.mfi.calculate(self.data)
        
        # 验证牛市阈值变化
        self.assertNotEqual(self.mfi._dynamic_overbought, 80)
        self.assertNotEqual(self.mfi._dynamic_oversold, 20)
    
    def test_price_structure_synergy(self):
        """
        测试价格结构协同分析功能
        """
        # 计算指标
        result = self.mfi.calculate(self.data)
        
        # 执行价格结构协同分析
        synergy_result = self.mfi.analyze_price_structure_synergy(self.data)
        
        # 验证结果
        self.assertIsInstance(synergy_result, pd.DataFrame)
        self.assertIn('synergy_score', synergy_result.columns)
    
    def test_calculate_raw_score(self):
        """
        测试原始评分计算功能
        """
        # 计算评分
        score = self.mfi.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        self.assertTrue((score >= 0).all() and (score <= 100).all())
        
        # 设置不同市场环境测试评分调整
        self.mfi.set_market_environment('bull_market')
        bull_score = self.mfi.calculate_raw_score(self.data)
        
        self.mfi.set_market_environment('bear_market')
        bear_score = self.mfi.calculate_raw_score(self.data)
        
        # 验证不同市场环境的评分不同
        self.assertFalse((bull_score == bear_score).all())
    
    def test_identify_patterns(self):
        """
        测试形态识别功能
        """
        # 计算指标
        result = self.mfi.calculate(self.data)
        
        # 识别形态
        patterns = self.mfi.identify_patterns(self.data)
        
        # 验证形态列表
        self.assertIsInstance(patterns, list)
        self.assertTrue(len(patterns) > 0)
    
    def test_generate_signals(self):
        """
        测试信号生成功能
        """
        # 计算指标
        result = self.mfi.calculate(self.data)
        
        # 生成信号
        signals = self.mfi.generate_signals(self.data)
        
        # 验证信号数据框
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)
        self.assertIn('neutral_signal', signals.columns)
        self.assertIn('score', signals.columns)
        self.assertIn('confidence', signals.columns)
    
    def test_market_environment(self):
        """
        测试市场环境设置功能
        """
        # 设置有效市场环境
        self.mfi.set_market_environment('bull_market')
        self.assertEqual(self.mfi.market_environment, 'bull_market')
        
        self.mfi.set_market_environment('bear_market')
        self.assertEqual(self.mfi.market_environment, 'bear_market')
        
        self.mfi.set_market_environment('sideways_market')
        self.assertEqual(self.mfi.market_environment, 'sideways_market')
        
        self.mfi.set_market_environment('volatile_market')
        self.assertEqual(self.mfi.market_environment, 'volatile_market')
        
        self.mfi.set_market_environment('normal')
        self.assertEqual(self.mfi.market_environment, 'normal')
        
        # 测试无效市场环境
        with self.assertRaises(ValueError):
            self.mfi.set_market_environment('invalid_environment')


if __name__ == '__main__':
    unittest.main() 