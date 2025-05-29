#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双周期RSI测试

测试RSI指标的双周期策略功能
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from indicators.rsi import RSI


class TestDualPeriodRSI(unittest.TestCase):
    """测试双周期RSI功能"""

    def setUp(self):
        """
        创建测试数据
        """
        # 创建100天的样本数据
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        dates.reverse()  # 按时间正序排列
        
        # 创建示例价格数据
        np.random.seed(42)  # 固定随机种子，确保测试可重复
        
        # 生成价格数据 - 不同的模式以验证各种信号
        close_prices = np.zeros(100)
        
        # 前30天下降趋势
        close_prices[:30] = np.linspace(100, 80, 30) + np.random.normal(0, 1, 30)
        
        # 中间40天震荡趋势
        close_prices[30:70] = np.sin(np.linspace(0, 4*np.pi, 40)) * 5 + 85 + np.random.normal(0, 1, 40)
        
        # 后30天上升趋势
        close_prices[70:] = np.linspace(85, 110, 30) + np.random.normal(0, 1.5, 30)
        
        # 创建DataFrame
        self.data = pd.DataFrame({
            'close': close_prices
        }, index=dates)
        
        # 确保价格始终为正值
        self.data[self.data <= 0] = 0.01
        
        # 创建RSI指标实例，使用双周期6和14
        self.rsi = RSI(periods=[6, 14])
        
        # 计算指标
        self.result = self.rsi.compute(self.data)
        
    def test_dual_period_signals_existence(self):
        """测试是否生成双周期RSI信号"""
        # 验证是否包含必要的列
        required_columns = [
            'RSI6', 'RSI14', 
            'dual_rsi_buy_signal', 'dual_rsi_sell_signal',
            'dual_rsi_bullish', 'dual_rsi_bearish',
            'dual_rsi_top_divergence', 'dual_rsi_bottom_divergence'
        ]
        
        for col in required_columns:
            self.assertIn(col, self.result.columns, f"结果中缺少列: {col}")
    
    def test_dual_period_buy_signals(self):
        """测试双周期RSI买入信号"""
        # 检查是否有买入信号
        buy_signals = self.result['dual_rsi_buy_signal'].sum()
        self.assertGreater(buy_signals, 0, "没有检测到买入信号")
        
        print(f"检测到 {buy_signals} 个买入信号")
        
        # 检查买入信号出现的位置
        buy_signal_indices = self.result[self.result['dual_rsi_buy_signal']].index
        for idx in buy_signal_indices:
            # 获取信号位置的数据
            i = self.result.index.get_loc(idx)
            
            # 验证信号逻辑
            # 两个RSI都处于上升状态
            self.assertTrue(
                self.result['rsi_rising_6'].iloc[i] and self.result['rsi_rising_14'].iloc[i],
                f"位置 {i} 的买入信号不满足两个RSI上升状态"
            )
            
            # 两个RSI都处于或曾处于超卖区，或者RSI6上穿RSI14
            condition1 = (
                (self.result['RSI6'].iloc[i] <= 35 or self.result['RSI6'].iloc[i-1] <= 30) and
                (self.result['RSI14'].iloc[i] <= 40 or self.result['RSI14'].iloc[i-1] <= 35)
            )
            
            condition2 = (
                self.result['RSI6'].iloc[i-1] < self.result['RSI14'].iloc[i-1] and
                self.result['RSI6'].iloc[i] > self.result['RSI14'].iloc[i]
            )
            
            self.assertTrue(
                condition1 or condition2,
                f"位置 {i} 的买入信号不满足超卖区域条件或上穿条件"
            )
    
    def test_dual_period_sell_signals(self):
        """测试双周期RSI卖出信号"""
        # 检查是否有卖出信号
        sell_signals = self.result['dual_rsi_sell_signal'].sum()
        self.assertGreater(sell_signals, 0, "没有检测到卖出信号")
        
        print(f"检测到 {sell_signals} 个卖出信号")
        
        # 检查卖出信号出现的位置
        sell_signal_indices = self.result[self.result['dual_rsi_sell_signal']].index
        for idx in sell_signal_indices:
            # 获取信号位置的数据
            i = self.result.index.get_loc(idx)
            
            # 验证信号逻辑
            # 两个RSI都处于下降状态
            self.assertTrue(
                self.result['rsi_falling_6'].iloc[i] and self.result['rsi_falling_14'].iloc[i],
                f"位置 {i} 的卖出信号不满足两个RSI下降状态"
            )
            
            # 两个RSI都处于或曾处于超买区，或者RSI6下穿RSI14
            condition1 = (
                (self.result['RSI6'].iloc[i] >= 65 or self.result['RSI6'].iloc[i-1] >= 70) and
                (self.result['RSI14'].iloc[i] >= 60 or self.result['RSI14'].iloc[i-1] >= 65)
            )
            
            condition2 = (
                self.result['RSI6'].iloc[i-1] > self.result['RSI14'].iloc[i-1] and
                self.result['RSI6'].iloc[i] < self.result['RSI14'].iloc[i]
            )
            
            self.assertTrue(
                condition1 or condition2,
                f"位置 {i} 的卖出信号不满足超买区域条件或下穿条件"
            )
    
    def test_dual_period_trend_signals(self):
        """测试双周期RSI趋势信号"""
        # 检查多头趋势信号
        bullish_signals = self.result['dual_rsi_bullish'].sum()
        self.assertGreater(bullish_signals, 0, "没有检测到多头趋势信号")
        
        # 检查空头趋势信号
        bearish_signals = self.result['dual_rsi_bearish'].sum()
        self.assertGreater(bearish_signals, 0, "没有检测到空头趋势信号")
        
        print(f"检测到 {bullish_signals} 个多头趋势信号和 {bearish_signals} 个空头趋势信号")
        
        # 验证多头趋势信号逻辑
        bullish_indices = self.result[self.result['dual_rsi_bullish']].index
        for idx in bullish_indices:
            i = self.result.index.get_loc(idx)
            self.assertTrue(
                self.result['RSI6'].iloc[i] > self.result['RSI14'].iloc[i] and
                self.result['RSI6'].iloc[i] > 50 and self.result['RSI14'].iloc[i] > 50,
                f"位置 {i} 的多头趋势信号不满足条件"
            )
        
        # 验证空头趋势信号逻辑
        bearish_indices = self.result[self.result['dual_rsi_bearish']].index
        for idx in bearish_indices:
            i = self.result.index.get_loc(idx)
            self.assertTrue(
                self.result['RSI6'].iloc[i] < self.result['RSI14'].iloc[i] and
                self.result['RSI6'].iloc[i] < 50 and self.result['RSI14'].iloc[i] < 50,
                f"位置 {i} 的空头趋势信号不满足条件"
            )
    
    def test_divergence_signals(self):
        """测试RSI背离信号"""
        # 检查顶背离信号
        top_divergence = self.result['dual_rsi_top_divergence'].sum()
        print(f"检测到 {top_divergence} 个顶背离信号")
        
        # 检查底背离信号
        bottom_divergence = self.result['dual_rsi_bottom_divergence'].sum()
        print(f"检测到 {bottom_divergence} 个底背离信号")
        
        # 注意：由于我们的测试数据可能不足以产生背离，所以这里不强制要求必须有背离信号
        # 但是我们可以验证背离信号的逻辑是否正确
        if top_divergence > 0:
            top_divergence_indices = self.result[self.result['dual_rsi_top_divergence']].index
            for idx in top_divergence_indices:
                i = self.result.index.get_loc(idx)
                # 验证价格创新高但RSI未创新高
                rolling_max_close = self.result['close'].rolling(window=20).max().iloc[i]
                rolling_max_rsi6 = self.result['RSI6'].rolling(window=20).max().iloc[i]
                rolling_max_rsi14 = self.result['RSI14'].rolling(window=20).max().iloc[i]
                
                self.assertTrue(
                    self.result['close'].iloc[i] >= rolling_max_close and
                    self.result['RSI6'].iloc[i] < rolling_max_rsi6 and
                    self.result['RSI14'].iloc[i] < rolling_max_rsi14,
                    f"位置 {i} 的顶背离信号不满足条件"
                )
        
        if bottom_divergence > 0:
            bottom_divergence_indices = self.result[self.result['dual_rsi_bottom_divergence']].index
            for idx in bottom_divergence_indices:
                i = self.result.index.get_loc(idx)
                # 验证价格创新低但RSI未创新低
                rolling_min_close = self.result['close'].rolling(window=20).min().iloc[i]
                rolling_min_rsi6 = self.result['RSI6'].rolling(window=20).min().iloc[i]
                rolling_min_rsi14 = self.result['RSI14'].rolling(window=20).min().iloc[i]
                
                self.assertTrue(
                    self.result['close'].iloc[i] <= rolling_min_close and
                    self.result['RSI6'].iloc[i] > rolling_min_rsi6 and
                    self.result['RSI14'].iloc[i] > rolling_min_rsi14,
                    f"位置 {i} 的底背离信号不满足条件"
                )
    
    def test_signal_methods(self):
        """测试信号判断方法"""
        # 找到一个买入信号位置
        buy_index = self.result['dual_rsi_buy_signal'].idxmax()
        if buy_index is not pd.NaT:
            i = self.result.index.get_loc(buy_index)
            self.assertTrue(
                self.rsi.is_dual_period_buy_signal(self.result, i),
                f"is_dual_period_buy_signal方法在位置 {i} 未返回预期结果"
            )
        
        # 找到一个卖出信号位置
        sell_index = self.result['dual_rsi_sell_signal'].idxmax()
        if sell_index is not pd.NaT:
            i = self.result.index.get_loc(sell_index)
            self.assertTrue(
                self.rsi.is_dual_period_sell_signal(self.result, i),
                f"is_dual_period_sell_signal方法在位置 {i} 未返回预期结果"
            )


if __name__ == "__main__":
    unittest.main() 