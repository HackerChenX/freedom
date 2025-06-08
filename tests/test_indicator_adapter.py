#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
指标适配器与复合指标测试
"""

import unittest
import pandas as pd
import numpy as np
import os

from indicators.adapter import (IndicatorAdapter, register_indicator,
                            get_indicator, calculate_indicator, list_all_indicators)
from indicators.composite import TechnicalComposite, technical_composite
from indicators.macd import MACD
from indicators.rsi import RSI
from indicators.ma import MA
from utils.logger import get_logger

logger = get_logger(__name__)


class TestIndicatorAdapter(unittest.TestCase):
    """测试指标适配器功能"""
    
    def setUp(self):
        """初始化测试环境"""
        # 创建模拟数据
        self.create_mock_data()
        
        # 初始化指标
        self.macd = MACD()
        self.rsi = RSI()
    
    def create_mock_data(self):
        """创建模拟数据"""
        # 创建基础数据
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100)
        
        # 模拟价格数据
        close_prices = np.random.normal(loc=100, scale=10, size=100)
        close_prices = close_prices.cumsum()  # 累积求和创造趋势
        
        # 确保价格为正
        close_prices = np.abs(close_prices)
        
        # 创建OHLCV数据
        self.data = pd.DataFrame({
            'date': dates,
            'open': close_prices * np.random.uniform(0.98, 1.0, 100),
            'high': close_prices * np.random.uniform(1.0, 1.05, 100),
            'low': close_prices * np.random.uniform(0.95, 1.0, 100),
            'close': close_prices,
            'volume': np.random.uniform(1000000, 5000000, 100)
        })
        
        # 设置索引
        self.data.set_index('date', inplace=True)
    
    def test_adapter_registration(self):
        """测试指标适配器注册功能"""
        # 注册MACD指标
        register_indicator(self.macd)
        
        # 获取已注册的指标
        adapter = get_indicator("MACD")
        
        # 断言
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.name, "MACD")
    
    def test_adapter_calculation(self):
        """测试指标适配器计算功能"""
        # 注册MACD指标
        register_indicator(self.macd)
        
        # 使用适配器计算MACD
        result = calculate_indicator("MACD", self.data)
        
        # 断言
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(len(result) > 0)
        
        # 检查输出列 - 修改为实际列名
        macd_columns = ['DIF', 'DEA', 'MACD']  # 实际输出的MACD列名
        for col in macd_columns:
            self.assertIn(col, result.columns)
    
    def test_adapter_column_mapping(self):
        """测试指标适配器列名映射功能"""
        # 注册RSI指标
        register_indicator(self.rsi)
        
        # 创建数据副本并修改列名
        renamed_data = self.data.copy()
        renamed_data.columns = [col.upper() for col in renamed_data.columns]
        
        # 使用适配器计算RSI
        result = calculate_indicator("RSI", renamed_data)
        
        # 断言
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(len(result) > 0)
        
        # 修改为实际输出的RSI列名
        rsi_column = 'RSI14'  # 实际输出的RSI列名
        self.assertIn(rsi_column, result.columns)


class TestCompositeIndicator(unittest.TestCase):
    """测试复合指标功能"""
    
    def setUp(self):
        """初始化测试环境"""
        # 创建模拟数据
        self.create_mock_data()
        
        # 初始化技术指标组合器
        self.composite = TechnicalComposite()
        
        # 预先注册指标
        self.register_indicators()
    
    def create_mock_data(self):
        """创建模拟数据"""
        # 创建基础数据
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100)
        
        # 模拟价格数据
        close_prices = np.random.normal(loc=100, scale=10, size=100)
        close_prices = close_prices.cumsum()  # 累积求和创造趋势
        
        # 确保价格为正
        close_prices = np.abs(close_prices)
        
        # 创建OHLCV数据
        self.data = pd.DataFrame({
            'date': dates,
            'open': close_prices * np.random.uniform(0.98, 1.0, 100),
            'high': close_prices * np.random.uniform(1.0, 1.05, 100),
            'low': close_prices * np.random.uniform(0.95, 1.0, 100),
            'close': close_prices,
            'volume': np.random.uniform(1000000, 5000000, 100)
        })
        
        # 设置索引
        self.data.set_index('date', inplace=True)
    
    def register_indicators(self):
        """注册测试所需的指标"""
        from indicators.macd import MACD
        from indicators.rsi import RSI
        from indicators.atr import ATR
        from indicators.boll import BOLL
        from indicators.cci import CCI
        from indicators.vix import VIX
        from indicators.momentum import Momentum
        from indicators.roc import ROC
        from indicators.vr import VR
        from indicators.adx import ADX
        from indicators.intraday_volatility import IntradayVolatility
        from indicators.trend.trend_strength import TrendStrength
        
        # 注册所有测试需要的指标
        register_indicator(MACD())
        register_indicator(RSI())
        register_indicator(ATR())
        register_indicator(BOLL())
        register_indicator(CCI())
        register_indicator(VIX())
        register_indicator(Momentum())
        register_indicator(ROC())
        register_indicator(VR())
        register_indicator(ADX())
        register_indicator(IntradayVolatility())
        register_indicator(TrendStrength())
    
    def test_trend_strength_composite(self):
        """测试趋势强度复合指标"""
        try:
            # 创建趋势强度复合指标
            trend_indicator = self.composite.create_trend_strength_composite()
            
            # 计算指标
            result = trend_indicator.calculate(self.data)
            
            # 断言
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(len(result) > 0)
            
            # 检查输出列
            expected_columns = ['trend_strength_score', 'trend_category']
            for col in expected_columns:
                self.assertIn(col, result.columns)
        except Exception as e:
            logger.error(f"测试趋势强度复合指标失败: {str(e)}")
            raise
    
    def test_volatility_composite(self):
        """测试波动性复合指标"""
        try:
            # 创建波动性复合指标
            volatility_indicator = self.composite.create_volatility_composite()
            
            # 计算指标
            result = volatility_indicator.calculate(self.data)
            
            # 断言
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(len(result) > 0)
            
            # 检查输出列
            expected_columns = ['volatility_score', 'volatility_category']
            for col in expected_columns:
                self.assertIn(col, result.columns)
        except Exception as e:
            logger.error(f"测试波动性复合指标失败: {str(e)}")
            raise
    
    def test_momentum_composite(self):
        """测试动量复合指标"""
        try:
            # 创建动量复合指标
            momentum_indicator = self.composite.create_momentum_composite()
            
            # 计算指标
            result = momentum_indicator.calculate(self.data)
            
            # 断言
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(len(result) > 0)
            
            # 检查输出列
            expected_columns = ['momentum_score', 'momentum_category']
            for col in expected_columns:
                self.assertIn(col, result.columns)
        except Exception as e:
            logger.error(f"测试动量复合指标失败: {str(e)}")
            raise
    
    def test_market_health_composite(self):
        """测试市场健康度复合指标"""
        try:
            # 创建市场健康度复合指标
            health_indicator = self.composite.create_market_health_composite()
            
            # 计算指标
            result = health_indicator.calculate(self.data)
            
            # 断言
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(len(result) > 0)
            
            # 检查输出列
            expected_columns = ['market_health_score', 'market_health_category', 'market_state']
            for col in expected_columns:
                self.assertIn(col, result.columns)
        except Exception as e:
            logger.error(f"测试市场健康度复合指标失败: {str(e)}")
            raise


if __name__ == '__main__':
    unittest.main() 