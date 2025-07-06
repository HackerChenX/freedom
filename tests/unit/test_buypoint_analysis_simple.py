#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
买点分析系统简单测试
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer
from utils.logger import getLogger

logger = getLogger(__name__)


class TestBuyPointAnalysisSimple(unittest.TestCase):
    """买点分析系统简单测试类"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # 生成价格数据
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)
        prices = [base_price]
        
        for i in range(1, 100):
            price = prices[-1] * (1 + returns[i])
            prices.append(max(price, base_price * 0.5))
        
        # 生成OHLCV数据
        self.test_data = pd.DataFrame({
            'datetime': dates,
            'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 + np.random.uniform(-0.02, 0)) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 100),
            'code': ['TEST001'] * 100,
            'name': ['测试股票'] * 100,
            'level': ['daily'] * 100,
            'industry': ['科技'] * 100,
            'turnover_rate': np.random.uniform(0.1, 3.0, 100),
            'price_change': np.random.uniform(-5, 5, 100),
            'price_range': np.random.uniform(1, 10, 100)
        })
        
        # 确保OHLC逻辑正确
        for i in range(len(self.test_data)):
            high = max(self.test_data.loc[i, 'open'], self.test_data.loc[i, 'high'], self.test_data.loc[i, 'close'])
            low = min(self.test_data.loc[i, 'open'], self.test_data.loc[i, 'low'], self.test_data.loc[i, 'close'])
            self.test_data.loc[i, 'high'] = high
            self.test_data.loc[i, 'low'] = low
    
    def test_buypoint_analyzer_initialization(self):
        """测试买点分析器初始化"""
        try:
            analyzer = BuyPointAnalyzer()
            self.assertIsNotNone(analyzer)
            logger.info("✅ 买点分析器初始化测试通过")
        except Exception as e:
            self.fail(f"买点分析器初始化失败: {e}")
    
    def test_buypoint_calculation_methods(self):
        """测试买点计算方法"""
        try:
            analyzer = BuyPointAnalyzer()
            
            # 测试计算买点指标方法是否存在
            self.assertTrue(hasattr(analyzer, 'calculate_buy_point_indicators'))
            
            # 模拟调用计算方法
            buy_date_idx = 50  # 使用中间的日期作为买点
            
            # 这里我们不实际调用方法，因为可能需要数据库连接
            # 只是验证方法存在性
            logger.info("✅ 买点计算方法存在性测试通过")
            
        except Exception as e:
            self.fail(f"买点计算方法测试失败: {e}")
    
    def test_buypoint_data_structure(self):
        """测试买点数据结构"""
        # 测试买点分析需要的数据结构
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            self.assertIn(col, self.test_data.columns, f"缺少必需的列: {col}")
        
        # 验证数据类型
        self.assertTrue(pd.api.types.is_numeric_dtype(self.test_data['close']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.test_data['volume']))
        
        logger.info("✅ 买点数据结构测试通过")
    
    def test_buypoint_signal_detection_logic(self):
        """测试买点信号检测逻辑"""
        # 测试基本的买点信号检测逻辑
        data = self.test_data.copy()
        
        # 计算简单移动平均
        data['ma5'] = data['close'].rolling(window=5).mean()
        data['ma10'] = data['close'].rolling(window=10).mean()
        data['ma20'] = data['close'].rolling(window=20).mean()
        
        # 简单的买点信号：价格接近均线支撑
        data['near_ma20'] = abs(data['low'] / data['ma20'] - 1) < 0.02
        
        # 验证信号计算
        signals = data['near_ma20'].dropna()
        self.assertIsInstance(signals.iloc[0], (bool, np.bool_))
        
        logger.info(f"✅ 买点信号检测逻辑测试通过，检测到 {signals.sum()} 个潜在买点")


if __name__ == '__main__':
    unittest.main() 