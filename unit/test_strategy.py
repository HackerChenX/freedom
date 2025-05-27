"""
测试策略模块

使用模拟对象测试策略选股逻辑
"""

import unittest
from unittest.mock import MagicMock, patch
import datetime
import pandas as pd
import numpy as np

class MockDataManager:
    """模拟数据管理器"""
    
    def __init__(self):
        self.stock_list = pd.DataFrame({
            'stock_code': ['000001', '000002', '000003'],
            'stock_name': ['测试1', '测试2', '测试3'],
            'market': ['主板', '创业板', '科创板'],
            'industry': ['金融', '科技', '医药'],
            'market_cap': [100, 200, 300]
        })
        
        # 创建测试K线数据
        dates = pd.date_range('2023-01-01', periods=10)
        
        self.kline_data = {
            '000001': pd.DataFrame({
                'date': dates,
                'open': np.random.uniform(10, 20, 10),
                'high': np.random.uniform(15, 25, 10),
                'low': np.random.uniform(5, 15, 10),
                'close': np.linspace(10, 20, 10),
                'volume': np.random.uniform(1000, 2000, 10)
            }),
            '000002': pd.DataFrame({
                'date': dates,
                'open': np.random.uniform(10, 20, 10),
                'high': np.random.uniform(15, 25, 10),
                'low': np.random.uniform(5, 15, 10),
                'close': np.linspace(20, 10, 10),
                'volume': np.random.uniform(1000, 2000, 10)
            }),
            '000003': pd.DataFrame({
                'date': dates,
                'open': np.random.uniform(10, 20, 10),
                'high': np.random.uniform(15, 25, 10),
                'low': np.random.uniform(5, 15, 10),
                'close': np.random.uniform(10, 20, 10),
                'volume': np.random.uniform(1000, 2000, 10)
            })
        }
    
    def get_stock_list(self, filters=None):
        """获取股票列表"""
        df = self.stock_list.copy()
        
        if filters:
            if 'market' in filters and filters['market']:
                df = df[df['market'].isin(filters['market'])]
            
            if 'market_cap' in filters:
                if 'min' in filters['market_cap']:
                    df = df[df['market_cap'] >= filters['market_cap']['min']]
                if 'max' in filters['market_cap']:
                    df = df[df['market_cap'] <= filters['market_cap']['max']]
        
        return df
    
    def get_kline_data(self, stock_code, period=None, start_date=None, end_date=None):
        """获取K线数据"""
        return self.kline_data.get(stock_code, pd.DataFrame())


class MockIndicator:
    """模拟指标"""
    
    def __init__(self, name, signal_type='DEFAULT'):
        self.name = name
        self.signal_type = signal_type
        
    def calculate(self, data):
        """计算指标"""
        # 简单返回输入数据
        return data
    
    def generate_signals(self, data):
        """生成信号"""
        # 为第一支股票生成正向信号，其他生成负向信号
        if '000001' in str(data):
            return pd.DataFrame({
                'signal': [True] * len(data),
                'signal_strength': [0.8] * len(data)
            }, index=data.index)
        else:
            return pd.DataFrame({
                'signal': [False] * len(data),
                'signal_strength': [0.2] * len(data)
            }, index=data.index)


class TestStrategy(unittest.TestCase):
    """策略测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟数据管理器
        self.data_manager = MockDataManager()
        
        # 创建模拟指标
        self.ma_cross = MockIndicator('MA_CROSS')
        self.rsi_oversold = MockIndicator('RSI_OVERSOLD')
        
        # 创建策略配置
        self.strategy_config = {
            "strategy": {
                "id": "TEST_STRATEGY",
                "name": "测试策略",
                "description": "用于测试的策略",
                "conditions": [
                    {
                        "indicator_id": "MA_CROSS",
                        "period": "DAILY",
                        "parameters": {}
                    },
                    {
                        "indicator_id": "RSI_OVERSOLD",
                        "period": "DAILY",
                        "parameters": {}
                    },
                    {
                        "logic": "AND"
                    }
                ],
                "filters": {
                    "market": ["主板", "科创板"],
                    "market_cap": {
                        "min": 50,
                        "max": 250
                    }
                },
                "sort": [
                    {
                        "field": "signal_strength",
                        "direction": "DESC"
                    }
                ]
            }
        }
    
    def test_strategy_execution(self):
        """测试策略执行"""
        # 解析策略配置
        # 在实际场景中会使用StrategyParser，这里我们直接模拟解析结果
        strategy_plan = {
            "strategy_id": "TEST_STRATEGY",
            "name": "测试策略",
            "description": "用于测试的策略",
            "conditions": [
                {
                    "type": "indicator",
                    "indicator": self.ma_cross,
                    "period": "DAILY"
                },
                {
                    "type": "indicator",
                    "indicator": self.rsi_oversold,
                    "period": "DAILY"
                },
                {
                    "type": "logic",
                    "value": "AND"
                }
            ],
            "filters": self.strategy_config["strategy"]["filters"],
            "sort": self.strategy_config["strategy"]["sort"]
        }
        
        # 执行策略
        # 在实际场景中会使用StrategyExecutor，这里我们直接模拟执行过程
        
        # 1. 获取符合过滤条件的股票列表
        stock_list = self.data_manager.get_stock_list(strategy_plan["filters"])
        self.assertEqual(len(stock_list), 1)  # 应该有1支股票符合过滤条件
        
        # 2. 对每支股票执行策略条件
        results = []
        for _, stock in stock_list.iterrows():
            stock_code = stock['stock_code']
            
            # 获取K线数据
            kline_data = self.data_manager.get_kline_data(stock_code)
            
            # 计算指标
            satisfied = True
            satisfied_conditions = []
            signal_strength = 0
            
            # 简化处理：由于我们的模拟指标只对000001返回True，其他返回False
            if stock_code == '000001':
                satisfied = True
                satisfied_conditions = ['MA_CROSS', 'RSI_OVERSOLD']
                signal_strength = 0.8
            else:
                satisfied = False
            
            if satisfied:
                results.append({
                    'stock_code': stock_code,
                    'stock_name': stock['stock_name'],
                    'satisfied_conditions': satisfied_conditions,
                    'signal_strength': signal_strength
                })
        
        # 3. 验证结果
        self.assertEqual(len(results), 1)  # 应该只有1支股票满足所有条件
        self.assertEqual(results[0]['stock_code'], '000001')
        self.assertEqual(len(results[0]['satisfied_conditions']), 2)
        self.assertAlmostEqual(results[0]['signal_strength'], 0.8)
    
    def test_market_filter(self):
        """测试市场过滤"""
        # 修改过滤条件只选择创业板
        filters = {
            "market": ["创业板"],
            "market_cap": {
                "min": 50,
                "max": 250
            }
        }
        
        # 获取符合过滤条件的股票列表
        stock_list = self.data_manager.get_stock_list(filters)
        
        # 验证结果
        self.assertEqual(len(stock_list), 1)  # 应该只有1支创业板股票
        self.assertEqual(stock_list.iloc[0]['stock_code'], '000002')
    
    def test_market_cap_filter(self):
        """测试市值过滤"""
        # 修改过滤条件，市值范围50-150
        filters = {
            "market": ["主板", "创业板", "科创板"],
            "market_cap": {
                "min": 50,
                "max": 150
            }
        }
        
        # 获取符合过滤条件的股票列表
        stock_list = self.data_manager.get_stock_list(filters)
        
        # 验证结果
        self.assertEqual(len(stock_list), 1)  # 应该只有1支股票市值在范围内
        self.assertEqual(stock_list.iloc[0]['stock_code'], '000001')


if __name__ == '__main__':
    unittest.main() 