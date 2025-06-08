#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
新指标功能测试脚本
"""

import unittest
import pandas as pd
import numpy as np

from indicators.base_indicator import BaseIndicator
from indicators.ma import MA
from indicators.rsi import RSI
from indicators.boll import BOLL
from indicators.macd import MACD
from indicators.kdj import KDJ
from indicators.cci import CCI
from indicators.dmi import DMI
from indicators.bias import BIAS
from indicators.trix import TRIX
from indicators.stock_vix import StockVIX
from indicators.emv import EMV
from indicators.intraday_volatility import IntradayVolatility
from indicators.fibonacci import Fibonacci
from indicators.sentiment_analysis import SentimentAnalysis
from strategy.base_strategy import BaseStrategy
from db.data_manager import DataManager
from utils.date_utils import get_current_date
from utils.stock_utils import get_stock_list
from utils.logger import get_logger

logger = get_logger(__name__)


class TestNewIndicators(unittest.TestCase):
    """测试新指标和增强功能"""
    
    def setUp(self):
        """初始化测试环境"""
        # 创建模拟数据
        self.create_mock_data()
        
        # 初始化指标
        self.stock_vix = StockVIX()
        self.fibonacci = Fibonacci()
        self.sentiment = SentimentAnalysis()
        self.bias = BIAS()
    
    def create_mock_data(self):
        """创建模拟数据"""
        # 设置随机种子以保证可重复性
        np.random.seed(42)
        
        # 创建日期索引
        date_range = pd.date_range(start='2020-01-01', periods=300, freq='D')
        
        # 创建初始价格
        initial_price = 100.0
        
        # 生成随机价格走势
        # 生成随机涨跌幅
        returns = np.random.normal(0.0005, 0.018, len(date_range))
        
        # 累计计算价格
        price = initial_price * (1 + returns).cumprod()
        
        # 模拟高低价格
        high = price * (1 + np.abs(np.random.normal(0, 0.01, len(date_range))))
        low = price * (1 - np.abs(np.random.normal(0, 0.01, len(date_range))))
        
        # 确保 low <= open/close <= high
        open_price = low + (high - low) * np.random.random(len(date_range))
        
        # 生成成交量
        volume = np.random.randint(1000, 10000, len(date_range))
        
        # 创建DataFrame
        self.data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        }, index=date_range)
    
    def test_stock_vix(self):
        """测试个股VIX指标"""
        # 计算VIX指标
        vix_result = self.stock_vix.calculate(self.data)
        
        # 验证计算结果
        self.assertIn('stock_vix', vix_result.columns)
        self.assertIn('volatility_zone', vix_result.columns)
        self.assertIn('volatility_trend', vix_result.columns)
        
        # 测试信号生成
        signals = self.stock_vix.generate_signals(self.data)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)
        self.assertIn('signal_type', signals.columns)
    
    def test_fibonacci(self):
        """测试斐波那契工具"""
        # 计算斐波那契水平
        fib_result = self.fibonacci.calculate(self.data)
        
        # 验证结果中包含趋势方向
        self.assertIn('trend_direction', fib_result.columns)
        
        # 验证结果中包含斐波那契回调水平
        retrace_cols = [col for col in fib_result.columns if col.startswith('fib_retrace_')]
        self.assertTrue(len(retrace_cols) > 0)
        
        # 测试信号生成
        signals = self.fibonacci.generate_signals(self.data)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)
        self.assertIn('signal_type', signals.columns)
    
    def test_sentiment_analysis(self):
        """测试情绪分析"""
        # 计算情绪指标
        sentiment_result = self.sentiment.calculate(self.data)
        
        # 验证结果中包含情绪指数
        self.assertIn('sentiment_index', sentiment_result.columns)
        self.assertIn('sentiment_category', sentiment_result.columns)
        
        # 测试信号生成
        signals = self.sentiment.generate_signals(self.data)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)
        self.assertIn('signal_type', signals.columns)
    
    def test_bias_enhancements(self):
        """测试BIAS增强功能"""
        # 计算BIAS指标
        bias_result = self.bias.calculate(self.data)
        
        # 验证BIAS计算结果
        self.assertIn('BIAS6', bias_result.columns)
        self.assertIn('BIAS12', bias_result.columns)
        self.assertIn('BIAS24', bias_result.columns)
        
        # 测试多周期协同评估
        multi_period_result = self.bias.evaluate_multi_period_bias(self.data)
        
        # 验证多周期协同评估结果
        if len(multi_period_result.columns) > 0:
            self.assertIn('correlation_6_12', multi_period_result.columns)
            self.assertIn('divergence_score', multi_period_result.columns)

    def test_emv(self):
        """测试EMV指标"""
        # 使用正确的 'period' 参数初始化EMV
        emv = EMV(period=14)
        emv_result = emv.calculate(self.data)
        self.assertIn('EMV', emv_result.columns)
        self.assertIn('EMV_MA', emv_result.columns)
        
        signals = emv.generate_signals(self.data)
        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)

    def test_boll_bandwidth(self):
        """测试BOLL带宽分析"""
        # 使用正确的 'period' 参数初始化BOLL
        boll = BOLL(period=20)
        bandwidth_dynamics = boll.analyze_bandwidth_dynamics(self.data)
        self.assertIn('bandwidth', bandwidth_dynamics)
        self.assertIn('bandwidth_expanding', bandwidth_dynamics)
        self.assertIn('bandwidth_contracting', bandwidth_dynamics)


if __name__ == '__main__':
    unittest.main() 