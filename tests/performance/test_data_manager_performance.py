"""
数据管理器性能测试模块

用于测试数据管理器在不同负载下的性能表现
"""

import unittest
import time
import cProfile
import pstats
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from data.data_manager import DataManager
from enums.period import Period
from utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


class TestDataManagerPerformance(unittest.TestCase):
    """测试数据管理器的性能"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化（仅运行一次）"""
        # 设置日志级别
        setup_logger(level="INFO")
        
        # 生成模拟K线数据
        cls.create_mock_kline_data()
        
        # 生成模拟股票列表
        cls.create_mock_stock_list()
        
    @classmethod
    def create_mock_kline_data(cls):
        """创建模拟K线数据"""
        # 创建一年的日期数据
        dates = pd.date_range('2023-01-01', periods=252)
        
        # 为多个股票创建数据
        stock_codes = [f'00000{i}' for i in range(1, 101)]
        
        # 存储K线数据
        cls.kline_data = {}
        
        for stock_code in stock_codes:
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
                'volume': volumes,
                'amount': volumes * closes
            })
    
    @classmethod
    def create_mock_stock_list(cls):
        """创建模拟股票列表"""
        # 生成100支股票
        stock_codes = [f'00000{i}' for i in range(1, 101)]
        stock_names = [f'测试股票{i}' for i in range(1, 101)]
        markets = ['主板', '科创板', '创业板'] * 34
        industries = ['金融', '科技', '医药', '能源', '消费'] * 20
        market_caps = np.random.uniform(50, 2000, 100)
        
        cls.stock_list = pd.DataFrame({
            'stock_code': stock_codes,
            'stock_name': stock_names,
            'market': markets[:100],
            'industry': industries[:100],
            'market_cap': market_caps,
            'last_price': np.random.uniform(5, 100, 100),
            'pe_ratio': np.random.uniform(10, 50, 100),
            'pb_ratio': np.random.uniform(1, 10, 100),
            'turnover_rate': np.random.uniform(1, 10, 100)
        })
    
    def setUp(self):
        """测试前准备"""
        # 创建数据管理器
        self.data_manager = DataManager()
        # 清除缓存
        self.data_manager.clear_cache()
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    def test_get_kline_data_performance(self, mock_query):
        """测试获取K线数据的性能"""
        # 配置模拟对象
        mock_query.side_effect = lambda sql: self.kline_data['000001'] if '000001' in sql else pd.DataFrame()
        
        # 测量首次获取时间（无缓存）
        start_time = time.time()
        self.data_manager.get_kline_data(
            stock_code='000001',
            period=Period.DAILY,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        first_execution_time = time.time() - start_time
        
        # 测量再次获取时间（有缓存）
        start_time = time.time()
        self.data_manager.get_kline_data(
            stock_code='000001',
            period=Period.DAILY,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        second_execution_time = time.time() - start_time
        
        # 记录性能数据
        logger.info(f"K线数据获取性能测试结果:")
        logger.info(f"  首次获取时间: {first_execution_time:.6f} 秒")
        logger.info(f"  再次获取时间: {second_execution_time:.6f} 秒")
        logger.info(f"  缓存加速比例: {first_execution_time/second_execution_time:.2f}x")
        
        # 验证缓存加速效果
        self.assertLess(second_execution_time, first_execution_time, "缓存应该显著提高性能")
        self.assertGreater(first_execution_time/second_execution_time, 5, "缓存应该至少提高5倍性能")
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    def test_get_stock_list_performance(self, mock_query):
        """测试获取股票列表的性能"""
        # 配置模拟对象
        mock_query.return_value = self.stock_list
        
        # 测量首次获取时间（无缓存）
        start_time = time.time()
        self.data_manager.get_stock_list()
        first_execution_time = time.time() - start_time
        
        # 测量再次获取时间（有缓存）
        start_time = time.time()
        self.data_manager.get_stock_list()
        second_execution_time = time.time() - start_time
        
        # 记录性能数据
        logger.info(f"股票列表获取性能测试结果:")
        logger.info(f"  首次获取时间: {first_execution_time:.6f} 秒")
        logger.info(f"  再次获取时间: {second_execution_time:.6f} 秒")
        logger.info(f"  缓存加速比例: {first_execution_time/second_execution_time:.2f}x")
        
        # 验证缓存加速效果
        self.assertLess(second_execution_time, first_execution_time, "缓存应该显著提高性能")
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    def test_cache_eviction_performance(self, mock_query):
        """测试缓存淘汰策略的性能"""
        # 配置模拟对象
        mock_query.side_effect = lambda sql: self.kline_data[sql[-7:]] if sql[-7:] in self.kline_data else pd.DataFrame()
        
        # 测量填满缓存的时间
        start_time = time.time()
        
        # 获取100支股票的数据，填满缓存
        stock_codes = [f'00000{i}' for i in range(1, 101)]
        for stock_code in stock_codes:
            self.data_manager.get_kline_data(
                stock_code=stock_code,
                period=Period.DAILY,
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
        
        cache_fill_time = time.time() - start_time
        
        # 获取缓存统计
        cache_stats = self.data_manager.get_cache_stats()
        
        # 记录性能数据
        logger.info(f"缓存填充性能测试结果:")
        logger.info(f"  缓存填充时间: {cache_fill_time:.6f} 秒")
        logger.info(f"  缓存大小: {cache_stats['size']}")
        logger.info(f"  缓存命中率: {cache_stats['hit_rate']:.2f}")
        logger.info(f"  缓存驱逐次数: {cache_stats['evictions']}")
        
        # 验证缓存效果
        self.assertGreater(cache_stats['hit_rate'], 0, "缓存命中率应大于0")
        self.assertLessEqual(cache_stats['size'], self.data_manager.max_cache_size, "缓存大小不应超过最大限制")
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    def test_multithreaded_cache_performance(self, mock_query):
        """测试多线程环境下缓存的性能"""
        # 配置模拟对象
        mock_query.side_effect = lambda sql: self.kline_data[sql[-7:]] if sql[-7:] in self.kline_data else pd.DataFrame()
        
        # 创建线程安全测试
        import threading
        
        # 定义线程工作函数
        def worker(thread_id):
            # 每个线程获取10支不同的股票数据
            for i in range(10):
                stock_code = f'00000{thread_id*10 + i + 1}'
                self.data_manager.get_kline_data(
                    stock_code=stock_code,
                    period=Period.DAILY,
                    start_date='2023-01-01',
                    end_date='2023-12-31'
                )
        
        # 测量多线程执行时间
        start_time = time.time()
        
        # 创建并启动10个线程
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        execution_time = time.time() - start_time
        
        # 获取缓存统计
        cache_stats = self.data_manager.get_cache_stats()
        
        # 记录性能数据
        logger.info(f"多线程缓存性能测试结果:")
        logger.info(f"  执行时间: {execution_time:.6f} 秒")
        logger.info(f"  缓存大小: {cache_stats['size']}")
        logger.info(f"  缓存命中率: {cache_stats['hit_rate']:.2f}")
        logger.info(f"  缓存驱逐次数: {cache_stats['evictions']}")
        
        # 验证多线程环境下缓存的正确性
        self.assertLessEqual(cache_stats['size'], self.data_manager.max_cache_size, "缓存大小不应超过最大限制")


if __name__ == '__main__':
    unittest.main() 