#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主力行为模式分析测试
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.institutional_behavior import InstitutionalBehavior
from strategy.institutional_strategy import InstitutionalStrategy


class TestInstitutionalBehavior(unittest.TestCase):
    """测试主力行为模式分析功能"""
    
    def setUp(self):
        """初始化测试环境"""
        # 创建模拟数据
        self.create_mock_data()
        
        # 初始化指标和策略
        self.behavior = InstitutionalBehavior()
        self.strategy = InstitutionalStrategy()
    
    def create_mock_data(self):
        """创建模拟数据"""
        # 创建基础数据
        np.random.seed(42)  # 确保结果可复现
        n_days = 200
        
        # 基础价格序列（模拟一个完整的主力运作周期）
        base_price = 10.0
        price_trend = np.zeros(n_days)
        
        # 0-50天：吸筹期（价格窄幅波动）
        price_trend[:50] = base_price + np.random.normal(0, 0.2, 50) + np.linspace(0, 0.5, 50)
        
        # 50-100天：控盘期（价格小幅上涨）
        price_trend[50:100] = price_trend[49] + np.random.normal(0, 0.3, 50) + np.linspace(0, 1.5, 50)
        
        # 100-150天：拉升期（价格快速上涨）
        price_trend[100:150] = price_trend[99] + np.random.normal(0, 0.5, 50) + np.linspace(0, 5.0, 50)
        
        # 150-200天：出货期（价格波动加大，逐渐回落）
        price_trend[150:] = price_trend[149] + np.random.normal(0, 0.8, 50) - np.linspace(0, 3.0, 50)
        
        # 确保价格为正
        price_trend = np.maximum(price_trend, base_price * 0.5)
        
        # 生成OHLC数据
        dates = pd.date_range(start='2020-01-01', periods=n_days)
        close = price_trend
        
        # 生成高低开数据
        daily_volatility = np.random.uniform(0.02, 0.06, n_days)
        high = close * (1 + daily_volatility)
        low = close * (1 - daily_volatility)
        open_price = low + (high - low) * np.random.uniform(0, 1, n_days)
        
        # 生成成交量数据
        base_volume = 1000000
        volume = np.zeros(n_days)
        
        # 吸筹期：成交量温和放大
        volume[:50] = base_volume * (1 + np.random.uniform(0, 0.5, 50) + np.linspace(0, 1.0, 50))
        
        # 控盘期：成交量稳定
        volume[50:100] = base_volume * (1 + np.random.uniform(0, 0.3, 50))
        
        # 拉升期：成交量明显放大
        volume[100:150] = base_volume * (2 + np.random.uniform(0, 1.0, 50) + np.linspace(0, 3.0, 50))
        
        # 出货期：成交量进一步放大后逐渐萎缩
        volume[150:180] = base_volume * (3 + np.random.uniform(0, 1.5, 30))
        volume[180:] = base_volume * (2 + np.random.uniform(0, 1.0, 20) - np.linspace(0, 1.5, 20))
        
        # 特定时点添加特征性变化
        
        # 洗盘特征：价格短期急跌后快速收复
        for i in [40, 80, 120]:
            high[i] = high[i] * 1.02
            close[i] = low[i] * 1.05
            volume[i] = volume[i] * 2.0
        
        # 放量突破特征
        for i in [60, 110]:
            high[i] = high[i] * 1.05
            close[i] = high[i] * 0.98
            volume[i] = volume[i] * 2.5
        
        # 创建DataFrame
        self.data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
    
    def test_basic_calculation(self):
        """测试基本计算功能"""
        # 计算结果
        result = self.behavior.calculate(self.data)
        
        # 检查结果是否包含必要的列
        self.assertIn("behavior_pattern", result.columns)
        self.assertIn("behavior_intensity", result.columns)
        self.assertIn("behavior_description", result.columns)
        
        # 检查行为模式识别结果
        patterns = result["behavior_pattern"].unique()
        self.assertTrue(any(pattern != "未知" for pattern in patterns))
    
    def test_behavior_classification(self):
        """测试行为模式分类功能"""
        # 计算分类结果
        classifications = self.behavior.classify_institutional_behavior(self.data)
        
        # 检查分类结果是否为列表且非空
        self.assertIsInstance(classifications, list)
        self.assertTrue(len(classifications) > 0)
        
        # 检查是否包含主要分类类型
        classification_types = [item["type"] for item in classifications]
        expected_types = ["dominant_phase", "intensity_trend", "volume_price_relation"]
        for expected_type in expected_types:
            self.assertIn(expected_type, classification_types)
    
    def test_absorption_prediction(self):
        """测试吸筹完成时间预测功能"""
        # 使用吸筹期的数据
        absorption_data = self.data.iloc[:60]
        
        # 计算预测结果
        prediction = self.behavior.predict_absorption_completion(absorption_data)
        
        # 检查预测结果是否包含必要的键
        expected_keys = ["is_in_absorption", "completion_days_min", "completion_days_max", 
                         "confidence", "description"]
        for key in expected_keys:
            self.assertIn(key, prediction)
    
    def test_strategy(self):
        """测试主力行为选股策略"""
        # 创建模拟股票数据字典
        data_dict = {
            "000001": self.data,
            "000002": self.data.iloc[50:],  # 控盘期开始
            "000003": self.data.iloc[100:],  # 拉升期开始
            "000004": self.data.iloc[150:]   # 出货期开始
        }
        
        # 执行选股
        selected = self.strategy.select(data_dict)
        
        # 检查选股结果
        self.assertIsInstance(selected, list)
        
        # 详细分析单只股票
        analysis = self.strategy.analyze_stock(self.data)
        
        # 检查分析结果是否包含必要的键
        expected_keys = ["score", "investment_suggestion", "risk_level", "behavior_classifications"]
        for key in expected_keys:
            self.assertIn(key, analysis)


if __name__ == "__main__":
    unittest.main() 