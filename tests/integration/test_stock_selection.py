"""
选股流程集成测试模块

测试从策略配置到选股执行的完整流程
"""

import unittest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from strategy.strategy_parser import StrategyParser
from strategy.strategy_executor import StrategyExecutor
from strategy.strategy_manager import StrategyManager
from db.data_manager import DataManager
from indicators.factory import IndicatorFactory
from strategy.selector import StockSelector


class TestStockSelectionWorkflow(unittest.TestCase):
    """选股流程集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试用临时目录
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # 创建测试用的策略配置
        self.strategy_config = {
            "strategy": {
                "id": "TEST_SELECTION_STRATEGY",
                "name": "测试选股策略",
                "description": "用于测试选股流程的策略",
                "conditions": [
                    {
                        "indicator_id": "MA_CROSS",
                        "period": "DAILY",
                        "parameters": {
                            "fast_period": 5,
                            "slow_period": 20
                        }
                    },
                    {
                        "indicator_id": "RSI_OVERSOLD",
                        "period": "DAILY",
                        "parameters": {
                            "period": 14,
                            "threshold": 30
                        }
                    },
                    {
                        "logic": "AND"
                    }
                ],
                "filters": {
                    "market": ["主板", "科创板"],
                    "market_cap": {
                        "min": 50,
                        "max": 2000
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
        
        # 创建测试用策略文件
        self.strategy_file = os.path.join(self.temp_dir.name, "test_strategy.json")
        with open(self.strategy_file, 'w') as f:
            json.dump(self.strategy_config, f)
        
        # 创建测试用股票列表
        self.stock_list = pd.DataFrame({
            'stock_code': ['000001', '000002', '000003', '000004', '000005'],
            'stock_name': ['测试1', '测试2', '测试3', '测试4', '测试5'],
            'market': ['主板', '主板', '创业板', '科创板', '主板'],
            'industry': ['金融', '科技', '医药', '能源', '消费'],
            'market_cap': [100, 200, 300, 400, 3000]
        })
        
        # 创建测试用K线数据
        dates = pd.date_range('2023-01-01', periods=30)
        
        # 股票1: 满足MA_CROSS和RSI_OVERSOLD
        stock1_data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(10, 20, 30),
            'high': np.random.uniform(15, 25, 30),
            'low': np.random.uniform(5, 15, 30),
            'close': np.linspace(10, 20, 30),  # 上升趋势
            'volume': np.random.uniform(1000, 2000, 30)
        })
        # 确保MA_CROSS条件满足 (5日均线上穿20日均线)
        stock1_data['ma5'] = stock1_data['close'].rolling(5).mean()
        stock1_data['ma20'] = stock1_data['close'].rolling(20).mean()
        stock1_data.loc[29, 'ma5'] = 15  # 确保最后一天5日均线大于20日均线
        stock1_data.loc[29, 'ma20'] = 14
        stock1_data.loc[28, 'ma5'] = 13  # 确保上一天5日均线小于等于20日均线
        stock1_data.loc[28, 'ma20'] = 14
        
        # 股票2: 只满足MA_CROSS
        stock2_data = stock1_data.copy()
        
        # 股票3: 只满足RSI_OVERSOLD (市场不符合)
        stock3_data = stock1_data.copy()
        
        # 股票4: 满足所有条件但市值超过阈值
        stock4_data = stock1_data.copy()
        
        # 股票5: 不满足任何条件
        stock5_data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(10, 20, 30),
            'high': np.random.uniform(15, 25, 30),
            'low': np.random.uniform(5, 15, 30),
            'close': np.linspace(20, 10, 30),  # 下降趋势
            'volume': np.random.uniform(1000, 2000, 30)
        })
        
        self.kline_data = {
            '000001': stock1_data,
            '000002': stock2_data,
            '000003': stock3_data,
            '000004': stock4_data,
            '000005': stock5_data
        }
        
        # 创建测试用指标
        self.mock_ma_cross = MagicMock()
        self.mock_ma_cross.generate_signals.return_value = pd.DataFrame({
            'golden_cross': [False] * 29 + [True],  # 最后一天金叉
            'signal_strength': [0] * 29 + [0.8]     # 最后一天信号强度0.8
        }, index=dates)
        
        self.mock_rsi_oversold = MagicMock()
        self.mock_rsi_oversold.generate_signals.return_value = pd.DataFrame({
            'oversold': [False] * 29 + [True],      # 最后一天超卖
            'signal_strength': [0] * 29 + [0.6]     # 最后一天信号强度0.6
        }, index=dates)
        
        # 测试用指标字典
        self.indicator_dict = {
            'MA_CROSS': self.mock_ma_cross,
            'RSI_OVERSOLD': self.mock_rsi_oversold
        }
        
        self.mock_db_conn = MagicMock()
        self.data_manager = DataManager()
        self.data_manager.db_conn = self.mock_db_conn
        self.strategy_manager = MagicMock()
        self.stock_selector = StockSelector(self.data_manager, self.strategy_manager)
        
    def tearDown(self):
        """测试后清理"""
        self.temp_dir.cleanup()
        
    @patch.object(IndicatorFactory, 'create')
    @patch.object(DataManager, 'get_stock_list')
    @patch.object(DataManager, 'get_kline_data')
    def test_end_to_end_selection(self, mock_get_kline, mock_get_stocks, mock_create_indicator):
        """测试端到端的选股流程"""
        # 配置模拟对象行为
        mock_get_stocks.return_value = self.stock_list
        mock_get_kline.side_effect = lambda stock_code, *args, **kwargs: self.kline_data[stock_code]
        mock_create_indicator.side_effect = lambda indicator_id, **kwargs: self.indicator_dict.get(indicator_id)
        
        # 创建解析器和执行器
        parser = StrategyParser()
        executor = StrategyExecutor()
        
        # 解析策略
        strategy_plan = parser.parse_from_file(self.strategy_file)
        
        # 执行选股
        result = executor.execute_strategy(
            strategy_plan=strategy_plan,
            end_date='2023-01-30'
        )
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 验证选出的股票
        # 只有000001和000004应该被选中，但000004因为市值过滤被排除
        expected_stocks = ['000001']
        self.assertEqual(len(result), len(expected_stocks))
        for stock in expected_stocks:
            self.assertIn(stock, result['stock_code'].values)
        
        # 验证信号强度计算
        # MA_CROSS:0.8, RSI_OVERSOLD:0.6, 平均:0.7
        self.assertAlmostEqual(result.iloc[0]['signal_strength'], 0.7, delta=0.01)
        
        # 验证条件满足情况
        self.assertTrue('MA_CROSS' in result.iloc[0]['satisfied_conditions'])
        self.assertTrue('RSI_OVERSOLD' in result.iloc[0]['satisfied_conditions'])
    
    @patch.object(StrategyManager, 'get_strategy')
    @patch.object(IndicatorFactory, 'create')
    @patch.object(DataManager, 'get_stock_list')
    @patch.object(DataManager, 'get_kline_data')
    def test_strategy_manager_integration(self, mock_get_kline, mock_get_stocks, mock_create_indicator, mock_get_strategy):
        """测试策略管理器与选股流程的集成"""
        # 配置模拟对象行为
        mock_get_stocks.return_value = self.stock_list
        mock_get_kline.side_effect = lambda stock_code, *args, **kwargs: self.kline_data[stock_code]
        mock_create_indicator.side_effect = lambda indicator_id, **kwargs: self.indicator_dict.get(indicator_id)
        mock_get_strategy.return_value = self.strategy_config
        
        # 创建策略管理器和执行器
        manager = StrategyManager()
        executor = StrategyExecutor()
        
        # 执行选股（通过策略ID）
        with patch.object(StrategyParser, 'parse_strategy', return_value=StrategyParser().parse_strategy(self.strategy_config)):
            result = executor.execute_strategy_by_id(
                strategy_id="TEST_SELECTION_STRATEGY",
                strategy_manager=manager,
                end_date='2023-01-30'
            )
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        
        # 验证选出的股票
        expected_stocks = ['000001']
        self.assertEqual(len(result), len(expected_stocks))
        for stock in expected_stocks:
            self.assertIn(stock, result['stock_code'].values)
    
    @patch.object(IndicatorFactory, 'create')
    @patch.object(DataManager, 'get_stock_list')
    @patch.object(DataManager, 'get_kline_data')
    def test_performance_and_caching(self, mock_get_kline, mock_get_stocks, mock_create_indicator):
        """测试性能和缓存机制"""
        # 配置模拟对象行为
        mock_get_stocks.return_value = self.stock_list
        mock_get_kline.side_effect = lambda stock_code, *args, **kwargs: self.kline_data[stock_code]
        mock_create_indicator.side_effect = lambda indicator_id, **kwargs: self.indicator_dict.get(indicator_id)
        
        # 创建解析器和执行器
        parser = StrategyParser()
        executor = StrategyExecutor()
        
        # 解析策略
        strategy_plan = parser.parse_from_file(self.strategy_file)
        
        # 首次执行选股
        executor.execute_strategy(
            strategy_plan=strategy_plan,
            end_date='2023-01-30'
        )
        
        # 重置mock计数
        mock_get_kline.reset_mock()
        mock_create_indicator.reset_mock()
        
        # 再次执行选股（应该使用缓存）
        executor.execute_strategy(
            strategy_plan=strategy_plan,
            end_date='2023-01-30'
        )
        
        # 验证缓存是否生效
        # 如果缓存正常工作，get_kline_data应该不会被调用或调用次数减少
        self.assertLess(mock_get_kline.call_count, 5)  # 少于股票数量的调用
        self.assertEqual(mock_create_indicator.call_count, 0)  # 不应再创建指标


if __name__ == '__main__':
    unittest.main() 