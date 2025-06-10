#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
布林带带宽变化率测试

测试布林带带宽变化率功能是否正常
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from indicators.boll import BOLL


class TestBOLLBandwidth(unittest.TestCase):
    """测试布林带带宽变化率功能"""

    def setUp(self):
        """
        创建测试数据
        """
        # 创建100天的样本数据
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        dates.reverse()  # 按时间正序排列
        
        # 创建示例价格数据
        np.random.seed(42)  # 固定随机种子，确保测试可重复
        
        # 生成价格数据 - 前半部分相对平稳，后半部分有明显趋势和波动
        close_prices = np.zeros(100)
        
        # 前50天相对平稳的价格
        close_prices[:50] = np.cumsum(np.random.normal(0, 0.3, 50)) + 100
        
        # 后50天明显的上升趋势和增大的波动
        trend = np.linspace(0, 15, 50)
        volatility = np.linspace(0.3, 1.5, 50)
        noise = np.array([np.random.normal(0, vol) for vol in volatility])
        close_prices[50:] = close_prices[49] + trend + np.cumsum(noise)
        
        # 创建DataFrame
        self.data = pd.DataFrame({
            'close': close_prices
        }, index=dates)
        
        # 确保价格始终为正值
        self.data[self.data <= 0] = 0.01
        
        # 创建BOLL指标实例
        self.boll = BOLL()
        self.boll.set_parameters({
            'periods': 20,
            'std_dev': 2.0,
            'bw_periods': 10
        })
        
        # 计算指标
        self.result = self.boll.compute(self.data)
        
    def test_bandwidth_calculation(self):
        """测试带宽计算"""
        # 获取带宽
        bandwidth = self.boll.get_bandwidth()
        
        # 验证带宽长度与数据长度一致
        self.assertEqual(len(bandwidth), len(self.data))
        
        # 验证带宽值是否合理
        self.assertTrue(all(bandwidth > 0))  # 带宽应该是正数
        
        # 手动计算部分数据点的带宽，验证结果
        for i in range(30, 40):
            upper = self.result['upper'].iloc[i]
            lower = self.result['lower'].iloc[i]
            middle = self.result['middle'].iloc[i]
            
            expected_bandwidth = (upper - lower) / middle
            calculated_bandwidth = bandwidth.iloc[i]
            
            # 验证计算值和期望值是否接近
            self.assertAlmostEqual(calculated_bandwidth, expected_bandwidth, places=10)
    
    def test_bandwidth_rate_calculation(self):
        """测试带宽变化率计算"""
        # 获取带宽变化率
        bw_rate = self.boll.get_bandwidth_rate()
        
        # 验证带宽变化率长度与数据长度一致
        self.assertEqual(len(bw_rate), len(self.data))
        
        # 验证前N个数据点的带宽变化率应为NaN（N=bw_periods）
        self.assertTrue(all(pd.isna(bw_rate.iloc[:10])))
        
        # 手动计算部分数据点的带宽变化率，验证结果
        bandwidth = self.boll.get_bandwidth()
        for i in range(30, 40):
            current_bw = bandwidth.iloc[i]
            prev_bw = bandwidth.iloc[i-10]
            
            expected_bw_rate = (current_bw - prev_bw) / prev_bw * 100
            calculated_bw_rate = bw_rate.iloc[i]
            
            # 验证计算值和期望值是否接近
            self.assertAlmostEqual(calculated_bw_rate, expected_bw_rate, places=10)
    
    def test_bandwidth_squeeze(self):
        """测试带宽收缩判断"""
        # 设置测试数据
        # 带宽减少测试数据
        squeeze_data = pd.DataFrame({
            'close': np.linspace(100, 110, 100)  # 线性上升价格，带宽应该收缩
        })
        
        # 计算带宽指标
        self.boll.compute(squeeze_data)
        
        # 判断带宽收缩
        squeeze_signal = self.boll.is_bandwidth_squeeze(threshold=5.0)
        
        # 验证信号长度与数据长度一致
        self.assertEqual(len(squeeze_signal), len(squeeze_data))
        
        # 验证是否存在收缩信号
        # 注意：线性价格数据应该产生带宽收缩
        self.assertTrue(any(squeeze_signal[30:]))
    
    def test_bandwidth_expansion(self):
        """测试带宽扩张判断"""
        # 设置测试数据
        # 带宽扩张测试数据 - 使用指数增长的波动性
        t = np.linspace(0, 4, 100)
        volatility = np.exp(t) * 0.1
        noise = np.array([np.random.normal(0, vol) for vol in volatility])
        prices = 100 + np.cumsum(noise)
        
        expansion_data = pd.DataFrame({
            'close': prices
        })
        
        # 计算带宽指标
        self.boll.compute(expansion_data)
        
        # 判断带宽扩张
        expansion_signal = self.boll.is_bandwidth_expansion(threshold=10.0)
        
        # 验证信号长度与数据长度一致
        self.assertEqual(len(expansion_signal), len(expansion_data))
        
        # 验证是否存在扩张信号
        # 注意：波动性增加的数据应该在后期产生带宽扩张
        self.assertTrue(any(expansion_signal[50:]))
    
    def test_boll_parameters(self):
        """测试参数设置"""
        # 创建一个新的BOLL实例
        boll = BOLL()
        
        # 检查默认参数
        params = boll.parameters
        self.assertEqual(params['periods'], 20)
        self.assertEqual(params['std_dev'], 2.0)
        self.assertEqual(params['bw_periods'], 20)
        
        # 设置新参数
        new_params = {
            'periods': 10,
            'std_dev': 3.0,
            'bw_periods': 5
        }
        boll.set_parameters(new_params)
        
        # 验证参数是否正确设置
        updated_params = boll.parameters
        self.assertEqual(updated_params['periods'], 10)
        self.assertEqual(updated_params['std_dev'], 3.0)
        self.assertEqual(updated_params['bw_periods'], 5)
        
        # 使用新参数计算指标
        result = boll.compute(self.data)
        
        # 获取带宽变化率
        bw_rate = boll.get_bandwidth_rate()
        
        # 验证前5个数据点的带宽变化率应为NaN（新的bw_periods=5）
        self.assertTrue(all(pd.isna(bw_rate.iloc[:5])))
        self.assertFalse(pd.isna(bw_rate.iloc[5]))


if __name__ == "__main__":
    unittest.main() 