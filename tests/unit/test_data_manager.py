"""
数据管理器单元测试模块

测试数据管理器的基本功能和异常处理
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from db.data_manager import DataManager
from enums.period import Period
from utils.exceptions import DataAccessError, DataNotFoundError, DataValidationError


class TestDataManager(unittest.TestCase):
    """数据管理器单元测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建数据管理器实例
        self.data_manager = DataManager()
        
        # 清除缓存
        self.data_manager.clear_cache()
        
        # 创建测试数据
        self.test_kline_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'open': np.random.rand(10) * 100,
            'high': np.random.rand(10) * 100,
            'low': np.random.rand(10) * 100,
            'close': np.random.rand(10) * 100,
            'volume': np.random.rand(10) * 10000,
            'amount': np.random.rand(10) * 1000000
        })
        
        self.test_stock_list = pd.DataFrame({
            'stock_code': ['000001', '000002', '000003'],
            'stock_name': ['测试1', '测试2', '测试3'],
            'market': ['主板', '创业板', '科创板'],
            'industry': ['金融', '科技', '医药'],
            'market_cap': [100, 200, 300],
            'last_price': [10, 20, 30],
            'pe_ratio': [15, 20, 25],
            'pb_ratio': [1.5, 2.0, 2.5],
            'turnover_rate': [2, 3, 4]
        })
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    def test_get_kline_data_success(self, mock_query):
        """测试成功获取K线数据"""
        # 配置模拟对象
        mock_query.return_value = self.test_kline_data
        
        # 调用被测方法
        result = self.data_manager.get_kline_data(
            stock_code='000001',
            period=Period.DAILY,
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 10)
        self.assertEqual(result.iloc[0]['date'].strftime('%Y-%m-%d'), '2023-01-01')
        
        # 验证模拟对象被调用
        mock_query.assert_called_once()
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    def test_get_kline_data_empty(self, mock_query):
        """测试获取空K线数据"""
        # 配置模拟对象返回空DataFrame
        mock_query.return_value = pd.DataFrame()
        
        # 调用被测方法
        result = self.data_manager.get_kline_data(
            stock_code='000001',
            period=Period.DAILY,
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        # 验证结果
        self.assertTrue(result.empty)
        
        # 验证模拟对象被调用
        mock_query.assert_called_once()
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    def test_get_kline_data_invalid_period(self, mock_query):
        """测试使用无效周期获取K线数据"""
        # 测试使用无效的周期字符串
        with self.assertRaises(DataValidationError):
            self.data_manager.get_kline_data(
                stock_code='000001',
                period='INVALID_PERIOD',
                start_date='2023-01-01',
                end_date='2023-01-10'
            )
        
        # 验证模拟对象未被调用
        mock_query.assert_not_called()
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    def test_get_kline_data_empty_stock_code(self, mock_query):
        """测试使用空股票代码获取K线数据"""
        # 测试使用空股票代码
        with self.assertRaises(DataValidationError):
            self.data_manager.get_kline_data(
                stock_code='',
                period=Period.DAILY,
                start_date='2023-01-01',
                end_date='2023-01-10'
            )
        
        # 验证模拟对象未被调用
        mock_query.assert_not_called()
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    def test_get_kline_data_db_error(self, mock_query):
        """测试数据库错误时获取K线数据"""
        # 配置模拟对象抛出异常
        mock_query.side_effect = Exception("数据库连接错误")
        
        # 测试数据库错误
        with self.assertRaises(DataAccessError):
            self.data_manager.get_kline_data(
                stock_code='000001',
                period=Period.DAILY,
                start_date='2023-01-01',
                end_date='2023-01-10'
            )
        
        # 验证模拟对象被调用
        mock_query.assert_called_once()
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    def test_get_kline_data_cache(self, mock_query):
        """测试K线数据缓存功能"""
        # 配置模拟对象
        mock_query.return_value = self.test_kline_data
        
        # 首次调用
        self.data_manager.get_kline_data(
            stock_code='000001',
            period=Period.DAILY,
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        # 验证模拟对象被调用一次
        mock_query.assert_called_once()
        mock_query.reset_mock()
        
        # 再次调用，应该使用缓存
        self.data_manager.get_kline_data(
            stock_code='000001',
            period=Period.DAILY,
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        # 验证模拟对象未被再次调用
        mock_query.assert_not_called()
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    def test_get_stock_list_success(self, mock_query):
        """测试成功获取股票列表"""
        # 配置模拟对象
        mock_query.return_value = self.test_stock_list
        
        # 调用被测方法
        result = self.data_manager.get_stock_list()
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0]['stock_code'], '000001')
        
        # 验证模拟对象被调用
        mock_query.assert_called_once()
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    def test_get_stock_list_with_filters(self, mock_query):
        """测试使用过滤器获取股票列表"""
        # 配置模拟对象
        mock_query.return_value = self.test_stock_list[self.test_stock_list['market'] == '主板']
        
        # 调用被测方法
        result = self.data_manager.get_stock_list(
            filters={'market': ['主板']}
        )
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['stock_code'], '000001')
        
        # 验证模拟对象被调用
        mock_query.assert_called_once()
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    def test_get_stock_list_db_error(self, mock_query):
        """测试数据库错误时获取股票列表"""
        # 配置模拟对象抛出异常
        mock_query.side_effect = Exception("数据库连接错误")
        
        # 测试数据库错误
        with self.assertRaises(DataAccessError):
            self.data_manager.get_stock_list()
        
        # 验证模拟对象被调用
        mock_query.assert_called_once()
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    @patch('db.clickhouse_db.ClickHouseDB.execute')
    def test_save_selection_result_success(self, mock_execute, mock_query):
        """测试成功保存选股结果"""
        # 创建测试选股结果
        selection_result = pd.DataFrame({
            'stock_code': ['000001', '000002'],
            'stock_name': ['测试1', '测试2'],
            'signal_strength': [0.8, 0.7],
            'satisfied_conditions': [['MA_CROSS', 'RSI_OVERSOLD'], ['MA_CROSS']]
        })
        
        # 配置模拟对象
        mock_execute.return_value = 2  # 影响的行数
        
        # 调用被测方法
        result = self.data_manager.save_selection_result(
            result=selection_result,
            strategy_id='TEST_STRATEGY',
            selection_date='2023-01-10'
        )
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证模拟对象被调用
        mock_execute.assert_called_once()
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    @patch('db.clickhouse_db.ClickHouseDB.execute')
    def test_save_selection_result_empty(self, mock_execute, mock_query):
        """测试保存空选股结果"""
        # 创建空选股结果
        selection_result = pd.DataFrame()
        
        # 调用被测方法
        result = self.data_manager.save_selection_result(
            result=selection_result,
            strategy_id='TEST_STRATEGY',
            selection_date='2023-01-10'
        )
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证模拟对象未被调用
        mock_execute.assert_not_called()
    
    @patch('db.clickhouse_db.ClickHouseDB.query')
    @patch('db.clickhouse_db.ClickHouseDB.execute')
    def test_save_selection_result_db_error(self, mock_execute, mock_query):
        """测试数据库错误时保存选股结果"""
        # 创建测试选股结果
        selection_result = pd.DataFrame({
            'stock_code': ['000001', '000002'],
            'stock_name': ['测试1', '测试2'],
            'signal_strength': [0.8, 0.7],
            'satisfied_conditions': [['MA_CROSS', 'RSI_OVERSOLD'], ['MA_CROSS']]
        })
        
        # 配置模拟对象抛出异常
        mock_execute.side_effect = Exception("数据库连接错误")
        
        # 测试数据库错误
        with self.assertRaises(DataAccessError):
            self.data_manager.save_selection_result(
                result=selection_result,
                strategy_id='TEST_STRATEGY',
                selection_date='2023-01-10'
            )
        
        # 验证模拟对象被调用
        mock_execute.assert_called_once()
    
    def test_cache_operations(self):
        """测试缓存操作功能"""
        # 测试清除缓存
        self.data_manager.clear_cache()
        stats = self.data_manager.get_cache_stats()
        self.assertEqual(stats['size'], 0)
        
        # 测试添加缓存（通过私有方法）
        self.data_manager._add_to_cache('test_key', 'test_value')
        stats = self.data_manager.get_cache_stats()
        self.assertEqual(stats['size'], 1)
        
        # 测试获取缓存（通过私有方法）
        value = self.data_manager._get_from_cache('test_key', 3600)
        self.assertEqual(value, 'test_value')
        
        # 测试缓存命中率
        stats = self.data_manager.get_cache_stats()
        self.assertEqual(stats['hits'], 1)
        
        # 测试清除特定模式的缓存
        self.data_manager._add_to_cache('other_key', 'other_value')
        self.data_manager.clear_cache('test')
        stats = self.data_manager.get_cache_stats()
        self.assertEqual(stats['size'], 1)  # 应该只剩下'other_key'
        
        # 测试完全清除缓存
        self.data_manager.clear_cache()
        stats = self.data_manager.get_cache_stats()
        self.assertEqual(stats['size'], 0)


if __name__ == '__main__':
    unittest.main() 