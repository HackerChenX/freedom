"""
技术指标单元测试模块

测试指标计算的正确性和稳定性
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from indicators import (
    BaseIndicator, MACD, RSI, KDJ, IndicatorFactory,
    ma, ema, macd, rsi, kdj
)


class TestIndicators(unittest.TestCase):
    """技术指标测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试数据
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
        
        # 创建模拟价格数据
        close = np.sin(np.linspace(0, 4 * np.pi, 100)) * 50 + 100
        high = close + np.random.rand(100) * 10
        low = close - np.random.rand(100) * 10
        volume = np.random.rand(100) * 1000 + 1000
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'open': close - np.random.rand(100) * 5,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        self.test_data.set_index('date', inplace=True)
    
    def test_ma_calculation(self):
        """测试MA计算"""
        # 使用通用函数计算
        ma_5 = ma(self.test_data['close'].values, 5)
        ma_10 = ma(self.test_data['close'].values, 10)
        
        # 检查结果长度
        self.assertEqual(len(ma_5), len(self.test_data))
        
        # 检查前N-1个值为NaN
        self.assertTrue(np.isnan(ma_5[0]))
        self.assertTrue(np.isnan(ma_5[3]))
        self.assertFalse(np.isnan(ma_5[4]))
        
        # 检查计算结果
        expected_ma5 = self.test_data['close'].rolling(5).mean().values
        np.testing.assert_allclose(ma_5, expected_ma5)
    
    def test_ema_calculation(self):
        """测试EMA计算"""
        # 使用通用函数计算
        ema_12 = ema(self.test_data['close'].values, 12)
        
        # 检查结果长度
        self.assertEqual(len(ema_12), len(self.test_data))
        
        # 检查计算结果
        expected_ema12 = self.test_data['close'].ewm(span=12, adjust=False).mean().values
        np.testing.assert_allclose(ema_12, expected_ema12)
    
    def test_macd_indicator(self):
        """测试MACD指标"""
        # 创建MACD指标
        macd_indicator = MACD(fast_period=12, slow_period=26, signal_period=9)
        
        # 计算MACD
        result = macd_indicator.calculate(self.test_data)
        
        # 检查结果包含所需的列
        self.assertIn('macd_line', result.columns)
        self.assertIn('macd_signal', result.columns)
        self.assertIn('macd_histogram', result.columns)
        
        # 检查MACD计算结果
        dif, dea, hist = macd(self.test_data['close'].values)
        np.testing.assert_allclose(result['macd_line'].values, dif)
        np.testing.assert_allclose(result['macd_signal'].values, dea)
        np.testing.assert_allclose(result['macd_histogram'].values, hist)
        
        # 检查信号列
        self.assertIn('macd_buy_signal', result.columns)
        self.assertIn('macd_sell_signal', result.columns)
    
    def test_rsi_indicator(self):
        """测试RSI指标"""
        # 创建RSI指标
        rsi_indicator = RSI(period=14)
        
        # 计算RSI
        result = rsi_indicator.calculate(self.test_data)
        
        # 检查结果包含RSI列
        self.assertIn('rsi', result.columns)
        
        # 检查RSI计算结果
        rsi_values = rsi(self.test_data['close'].values, 14)
        
        # 由于计算方法可能略有不同，允许一定的误差
        # 这里跳过前30个值的比较，因为初始计算可能不稳定
        np.testing.assert_allclose(
            result['rsi'].values[30:], 
            rsi_values[30:], 
            rtol=1e-2, atol=1e-2
        )
        
        # 检查RSI范围
        self.assertTrue(all((result['rsi'] >= 0) & (result['rsi'] <= 100)))
    
    def test_kdj_indicator(self):
        """测试KDJ指标"""
        # 创建KDJ指标
        kdj_indicator = KDJ(n=9, m1=3, m2=3)
        
        # 计算KDJ
        result = kdj_indicator.calculate(self.test_data)
        
        # 检查结果包含所需的列
        self.assertIn('k', result.columns)
        self.assertIn('d', result.columns)
        self.assertIn('j', result.columns)
        
        # 检查KDJ计算结果
        k, d, j = kdj(
            self.test_data['close'].values,
            self.test_data['high'].values,
            self.test_data['low'].values
        )
        
        # 由于计算方法可能略有不同，允许一定的误差
        np.testing.assert_allclose(result['k'].values, k, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(result['d'].values, d, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(result['j'].values, j, rtol=1e-2, atol=1e-2)
    
    def test_indicator_factory(self):
        """测试指标工厂"""
        # 测试创建MACD指标
        macd = IndicatorFactory.create_indicator('MACD')
        self.assertIsInstance(macd, MACD)
        
        # 测试创建RSI指标
        rsi = IndicatorFactory.create_indicator('RSI', period=10)
        self.assertIsInstance(rsi, RSI)
        self.assertEqual(rsi.period, 10)
        
        # 测试创建KDJ指标
        kdj = IndicatorFactory.create_indicator('KDJ', n=14, m1=5, m2=5)
        self.assertIsInstance(kdj, KDJ)
        self.assertEqual(kdj.n, 14)
        self.assertEqual(kdj.m1, 5)
        self.assertEqual(kdj.m2, 5)
        
        # 测试从配置创建指标
        config = {'name': 'MACD', 'fast_period': 10, 'slow_period': 30}
        macd_from_config = IndicatorFactory.create_indicator_from_config(config)
        self.assertIsInstance(macd_from_config, MACD)
        self.assertEqual(macd_from_config.fast_period, 10)
        self.assertEqual(macd_from_config.slow_period, 30)
        
        # 测试获取已注册的指标
        indicators = IndicatorFactory.get_registered_indicators()
        self.assertIn('MACD', indicators)
        self.assertIn('RSI', indicators)
        self.assertIn('KDJ', indicators)
        
        # 测试检查指标是否已注册
        self.assertTrue(IndicatorFactory.is_registered('MACD'))
        self.assertFalse(IndicatorFactory.is_registered('UNKNOWN'))


if __name__ == '__main__':
    unittest.main() 