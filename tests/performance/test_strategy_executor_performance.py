"""
策略执行器性能测试模块

用于测试策略执行引擎在不同负载下的性能表现
"""

import unittest
import os
import sys
import time
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from strategy.strategy_parser import StrategyParser
from strategy.strategy_executor import StrategyExecutor
from db.data_manager import DataManager
from indicators.factory import IndicatorFactory
from utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


class TestStrategyExecutorPerformance(unittest.TestCase):
    """策略执行器性能测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化（仅运行一次）"""
        # 设置日志级别
        setup_logger(level="INFO")
        
        # 创建测试用的策略配置
        cls.strategy_config = {
            "strategy": {
                "id": "PERF_TEST_STRATEGY",
                "name": "性能测试策略",
                "description": "用于性能测试的策略",
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
                    "market": ["主板", "科创板", "创业板"],
                    "market_cap": {
                        "min": 0,
                        "max": 10000
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
        
        # 创建测试用股票列表
        stock_codes = [f'00000{i}' for i in range(1, 101)]  # 100支股票
        stock_names = [f'测试股票{i}' for i in range(1, 101)]
        markets = ['主板', '科创板', '创业板'] * 34
        industries = ['金融', '科技', '医药', '能源', '消费'] * 20
        market_caps = np.random.uniform(50, 2000, 100)
        
        cls.stock_list = pd.DataFrame({
            'stock_code': stock_codes,
            'stock_name': stock_names,
            'market': markets[:100],
            'industry': industries[:100],
            'market_cap': market_caps
        })
        
        # 创建测试用K线数据
        cls.kline_data = {}
        for stock_code in stock_codes:
            # 创建一年的K线数据
            dates = pd.date_range('2023-01-01', periods=252)
            
            # 随机生成价格数据
            closes = np.cumsum(np.random.normal(0, 1, 252)) + 20  # 随机游走
            opens = closes + np.random.normal(0, 0.5, 252)
            highs = np.maximum(opens, closes) + np.random.uniform(0, 1, 252)
            lows = np.minimum(opens, closes) - np.random.uniform(0, 1, 252)
            volumes = np.random.uniform(1000, 5000, 252)
            
            # 确保价格为正数
            opens = np.maximum(0.1, opens)
            highs = np.maximum(opens, highs)
            lows = np.maximum(0.1, lows)
            closes = np.maximum(lows, closes)
            
            cls.kline_data[stock_code] = pd.DataFrame({
                'date': dates,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
        
        # 创建测试用指标
        cls.mock_ma_cross = MagicMock()
        cls.mock_ma_cross.generate_signals.return_value = pd.DataFrame({
            'golden_cross': [False] * 251 + [True],  # 最后一天金叉
            'signal_strength': [0] * 251 + [0.8]     # 最后一天信号强度0.8
        }, index=dates)
        
        cls.mock_rsi_oversold = MagicMock()
        cls.mock_rsi_oversold.generate_signals.return_value = pd.DataFrame({
            'oversold': [False] * 251 + [True],      # 最后一天超卖
            'signal_strength': [0] * 251 + [0.6]     # 最后一天信号强度0.6
        }, index=dates)
        
        # 测试用指标字典
        cls.indicator_dict = {
            'MA_CROSS': cls.mock_ma_cross,
            'RSI_OVERSOLD': cls.mock_rsi_oversold
        }
    
    def setUp(self):
        """测试前准备"""
        # 清除缓存
        data_manager = DataManager()
        data_manager.clear_cache()
        
        # 创建测试执行器
        self.strategy_executor = StrategyExecutor()
        self.strategy_executor.clear_cache()
    
    @patch.object(IndicatorFactory, 'create')
    @patch.object(DataManager, 'get_stock_list')
    @patch.object(DataManager, 'get_kline_data')
    def test_executor_performance_baseline(self, mock_get_kline, mock_get_stocks, mock_create_indicator):
        """测试执行器基准性能"""
        # 配置模拟对象行为
        mock_get_stocks.return_value = self.stock_list
        mock_get_kline.side_effect = lambda stock_code, *args, **kwargs: self.kline_data[stock_code]
        mock_create_indicator.side_effect = lambda indicator_id, **kwargs: self.indicator_dict.get(indicator_id)
        
        # 解析策略
        parser = StrategyParser()
        strategy_plan = parser.parse_strategy(self.strategy_config)
        
        # 测量执行时间
        start_time = time.time()
        
        # 执行策略
        result = self.strategy_executor.execute_strategy(
            strategy_plan=strategy_plan,
            end_date='2023-12-31'
        )
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 记录性能数据
        logger.info(f"基准性能测试结果: {len(result)} 支股票, 执行时间: {execution_time:.2f} 秒")
        
        # 检查执行时间
        self.assertLess(execution_time, 30, "执行时间不应超过30秒")
    
    @patch.object(IndicatorFactory, 'create')
    @patch.object(DataManager, 'get_stock_list')
    @patch.object(DataManager, 'get_kline_data')
    def test_executor_performance_with_cache(self, mock_get_kline, mock_get_stocks, mock_create_indicator):
        """测试执行器使用缓存时的性能"""
        # 配置模拟对象行为
        mock_get_stocks.return_value = self.stock_list
        mock_get_kline.side_effect = lambda stock_code, *args, **kwargs: self.kline_data[stock_code]
        mock_create_indicator.side_effect = lambda indicator_id, **kwargs: self.indicator_dict.get(indicator_id)
        
        # 解析策略
        parser = StrategyParser()
        strategy_plan = parser.parse_strategy(self.strategy_config)
        
        # 第一次执行（无缓存）
        self.strategy_executor.execute_strategy(
            strategy_plan=strategy_plan,
            end_date='2023-12-31'
        )
        
        # 重置mock调用计数
        mock_get_kline.reset_mock()
        mock_create_indicator.reset_mock()
        
        # 测量第二次执行时间（使用缓存）
        start_time = time.time()
        
        # 第二次执行（有缓存）
        result = self.strategy_executor.execute_strategy(
            strategy_plan=strategy_plan,
            end_date='2023-12-31'
        )
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 记录性能数据
        logger.info(f"缓存性能测试结果: {len(result)} 支股票, 执行时间: {execution_time:.2f} 秒")
        
        # 检查执行时间应该显著减少
        self.assertLess(execution_time, 5, "使用缓存后执行时间应显著减少")
        
        # 验证缓存是否生效
        self.assertLess(mock_get_kline.call_count, 10, "使用缓存后应减少数据库调用")
    
    @patch.object(IndicatorFactory, 'create')
    @patch.object(DataManager, 'get_stock_list')
    @patch.object(DataManager, 'get_kline_data')
    def test_executor_performance_scaling(self, mock_get_kline, mock_get_stocks, mock_create_indicator):
        """测试执行器在不同工作线程数下的性能表现"""
        # 配置模拟对象行为
        mock_get_stocks.return_value = self.stock_list
        mock_get_kline.side_effect = lambda stock_code, *args, **kwargs: self.kline_data[stock_code]
        mock_create_indicator.side_effect = lambda indicator_id, **kwargs: self.indicator_dict.get(indicator_id)
        
        # 解析策略
        parser = StrategyParser()
        strategy_plan = parser.parse_strategy(self.strategy_config)
        
        # 测试不同线程数
        thread_counts = [1, 2, 4, 8, 16]
        execution_times = []
        
        for thread_count in thread_counts:
            # 创建执行器并设置线程数
            executor = StrategyExecutor(max_workers=thread_count)
            
            # 测量执行时间
            start_time = time.time()
            
            # 执行策略
            result = executor.execute_strategy(
                strategy_plan=strategy_plan,
                end_date='2023-12-31'
            )
            
            # 计算执行时间
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            # 记录性能数据
            logger.info(f"线程数 {thread_count} 性能测试结果: {len(result)} 支股票, 执行时间: {execution_time:.2f} 秒")
            
            # 清除缓存，确保公平比较
            executor.clear_cache()
            data_manager = DataManager()
            data_manager.clear_cache()
        
        # 验证线程数增加时性能提升
        # 理论上，线程数增加应该提高性能，但可能存在极限
        self.assertLess(execution_times[1], execution_times[0], "增加线程数应提高性能")
        
        # 记录扩展性数据
        scaling_efficiency = [execution_times[0] / t for t in execution_times]
        logger.info(f"线程数扩展效率: {list(zip(thread_counts, scaling_efficiency))}")


if __name__ == '__main__':
    unittest.main() 