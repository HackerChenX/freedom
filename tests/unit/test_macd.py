"""
MACD指标单元测试
"""

import unittest
import pandas as pd
import numpy as np

from indicators import MACD


class TestMACD(unittest.TestCase):
    """MACD指标单元测试类"""
    
    def setUp(self):
        """准备测试数据"""
        # 创建一个简单的测试数据集
        data = {
            'date': pd.date_range(start='2023-01-01', periods=100),
            'close': np.random.normal(100, 10, 100)  # 生成随机价格数据
        }
        self.df = pd.DataFrame(data)
        self.df.set_index('date', inplace=True)
        
        # 创建MACD实例
        self.macd = MACD()
    
    def test_calculate(self):
        """测试计算方法"""
        # 调用计算方法
        result = self.macd.calculate(self.df)
        
        # 验证结果DataFrame包含所需的列
        self.assertIn('macd_line', result.columns)
        self.assertIn('macd_signal', result.columns)
        self.assertIn('macd_histogram', result.columns)
        
        # 验证结果长度正确
        self.assertEqual(len(result), len(self.df))
        
        # 验证前几个值为NaN（因为需要足够的数据来计算移动平均）
        self.assertTrue(pd.isna(result['macd_line'].iloc[0]))
        
        # 验证最后一个值不为NaN
        self.assertFalse(pd.isna(result['macd_line'].iloc[-1]))
    
    def test_with_custom_params(self):
        """测试自定义参数"""
        # 创建自定义参数的MACD实例
        custom_macd = MACD(fast_period=5, slow_period=10, signal_period=3)
        
        # 调用计算方法
        result = custom_macd.calculate(self.df)
        
        # 验证结果DataFrame包含所需的列
        self.assertIn('macd_line', result.columns)
        self.assertIn('macd_signal', result.columns)
        self.assertIn('macd_histogram', result.columns)
        
        # 验证自定义参数的影响（例如更快的参数应该减少NaN值）
        standard_result = self.macd.calculate(self.df)
        
        # 计算非NaN值的数量，自定义参数应该有更多非NaN值
        custom_valid_count = result['macd_signal'].notna().sum()
        standard_valid_count = standard_result['macd_signal'].notna().sum()
        
        self.assertGreaterEqual(custom_valid_count, standard_valid_count)
    
    def test_get_buy_signal(self):
        """测试买入信号"""
        # 首先计算MACD
        result = self.macd.calculate(self.df)
        
        # 获取买入信号
        buy_signal = self.macd.get_buy_signal(result)
        
        # 验证买入信号是布尔类型的Series
        self.assertIsInstance(buy_signal, pd.Series)
        self.assertEqual(buy_signal.dtype, bool)
        
        # 验证买入信号的长度正确
        self.assertEqual(len(buy_signal), len(self.df))
    
    def test_get_sell_signal(self):
        """测试卖出信号"""
        # 首先计算MACD
        result = self.macd.calculate(self.df)
        
        # 获取卖出信号
        sell_signal = self.macd.get_sell_signal(result)
        
        # 验证卖出信号是布尔类型的Series
        self.assertIsInstance(sell_signal, pd.Series)
        self.assertEqual(sell_signal.dtype, bool)
        
        # 验证卖出信号的长度正确
        self.assertEqual(len(sell_signal), len(self.df))
    
    def test_add_signals(self):
        """测试添加信号"""
        # 首先计算MACD
        result = self.macd.calculate(self.df)
        
        # 添加信号
        result_with_signals = self.macd.add_signals(result)
        
        # 验证结果DataFrame包含信号列
        self.assertIn('macd_buy_signal', result_with_signals.columns)
        self.assertIn('macd_sell_signal', result_with_signals.columns)
        
        # 验证信号列是布尔类型
        self.assertEqual(result_with_signals['macd_buy_signal'].dtype, bool)
        self.assertEqual(result_with_signals['macd_sell_signal'].dtype, bool)


if __name__ == '__main__':
    unittest.main() 