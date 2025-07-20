#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能日期管理器集成测试

测试与其他组件的集成功能
"""

import unittest
import datetime
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from analysis.engines.date_manager import Date_manager, Date_format, Date_range, Week_day
from utils.cache import LRUCache


class Test_date_manager_integration(unittest.TestCase):
    """日期管理器集成测试"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟数据库
        self.mock_db = Mock()
        
        # 创建日期管理器实例
        self.date_manager = Date_manager()
        self.date_manager.db = self.mock_db
    
    def test_database_integration(self):
        """测试与数据库的集成"""
        # 模拟数据库返回数据
        mock_df = pd.DataFrame({
            'latest_date': ['2024-01-15']
        })
        self.mock_db.query_df.return_value = mock_df
        
        # 测试获取最新交易日期
        latest_date = self.date_manager.get_latest_trading_date()
        
        # 验证结果
        self.assertEqual(latest_date, "2024-01-15")
        
        # 验证数据库查询被调用
        self.mock_db.query_df.assert_called()
    
    def test_pandas_integration(self):
        """测试与pandas的集成"""
        # 创建pandas时间序列
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        
        # 测试pandas Timestamp解析
        for date in dates:
            parsed_date = self.date_manager.parse_date(date)
            self.assert_is_instance(parsed_date, datetime.datetime)
            self.assert_equal(parsed_date.date(), date.date())
        
        # 测试时间序列生成
        time_series = self.date_manager.get_time_series_dates(
            '2024-01-01', '2024-01-10', trading_days_only=False
        )
        
        self.assert_is_instance(time_series, pd.Datetime_index)
        self.assert_equal(len(time_series), 10)
    
    def test_cache_integration(self):
        """测试与缓存系统的集成"""
        # 测试缓存功能
        test_date = "2024-01-15"
        
        # 第一次解析，应该缓存
        result1 = self.date_manager.parse_date(test_date)
        
        # 第二次解析，应该从缓存获取
        result2 = self.date_manager.parse_date(test_date)
        
        # 结果应该相同
        self.assert_equal(result1, result2)
        
        # 验证缓存命中
        stats = self.date_manager.get_performance_stats()
        self.assertGreater(stats['cache_hits'], 0)
    
    def test_complex_date_operations(self):
        """测试复杂日期操作的集成"""
        base_date = "2024-01-15"  # 周一
        
        # 测试获取前后交易日
        prev_dates = self.date_manager.get_previous_trading_dates(base_date, 5)
        next_dates = self.date_manager.get_next_trading_dates(base_date, 5)
        
        # 验证结果
        self.assert_equal(len(prev_dates), 5)
        self.assert_equal(len(next_dates), 5)
        
        # 测试日期范围生成
        date_range = self.date_manager.get_date_range(
            prev_dates[-1], next_dates[-1], trading_days_only=True
        )
        
        # 验证范围包含基准日期
        self.assert_in(base_date, date_range)
        
        # 测试交易日计算
        trading_days = self.date_manager.get_trading_days_between(
            prev_dates[-1], next_dates[-1]
        )
        
        self.assert_greater(trading_days, 0)
    
    def test_multi_format_conversion(self):
        """测试多种格式转换的集成"""
        # 测试不同输入格式
        test_cases = [
            ("2024-01-15", DateFormat.YYYY_MM_DD),
            ("20240115", DateFormat.YYYYMMDD),
            ("2024-01-15 12:00:00", DateFormat.YYYY_MM_DD_HH_MM_SS),
            ("20240115_120000", DateFormat.YYYYMMDD_HHMMSS),
        ]
        
        for input_str, input_format in test_cases:
            # 解析日期
            parsed_date = self.date_manager.parse_date(input_str, input_format)
            
            # 转换为不同格式
            for output_format in Date_format:
                if output_format != Date_format.TIMESTAMP:
                    formatted = self.date_manager.format_date(
                        parsed_date, output_format
                    )
                    self.assert_is_instance(formatted, str)
    
    def test_error_handling_integration(self):
        """测试错误处理的集成"""
        # 测试数据库错误处理
        self.mock_db.query_df.side_effect = Exception("Database connection failed")
        
        # 应该回退到智能估算
        with patch('analysis.engines.date_manager.datetime') as mock_datetime:
            mock_now = datetime.datetime(2024, 1, 15, 10, 0, 0)
            mock_datetime.datetime.now.return_value = mock_now
            mock_datetime.datetime.side_effect = lambda *args, **kw: datetime.datetime(*args, **kw)
            mock_datetime.timedelta = datetime.timedelta
            
            latest_date = self.date_manager.get_latest_trading_date()
            
            # 应该返回估算的日期
            self.assert_is_instance(latest_date, str)
            self.assertRegex(latest_date, r'\d{4}-\d{2}-\d{2}')
    
    def test_performance_integration(self):
        """测试性能集成"""
        # 大量日期操作
        test_dates = [f"2024-01-{i:02d}" for i in range(1, 32)]
        
        # 批量解析日期
        start_time = datetime.datetime.now()
        
        parsed_dates = []
        for date_str in test_dates:
            try:
                parsed_date = self.date_manager.parse_date(date_str)
                parsed_dates.append(parsed_date)
            except ValueError:
                # 跳过无效日期（如2月30日）
                pass
        
        # 重复解析相同日期以测试缓存效果
        for date_str in test_dates[:10]:  # 重复解析前10个日期
            try:
                self.date_manager.parse_date(date_str)
            except ValueError:
                pass
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 验证性能
        self.assert_greater(len(parsed_dates), 25)  # 大部分日期应该解析成功
        self.assert_less(duration, 1.0)  # 应该在1秒内完成
        
        # 验证缓存效果
        stats = self.date_manager.get_performance_stats()
        self.assertGreater(stats['cache_hits'], 0)
    
    def test_real_world_scenario(self):
        """测试真实世界场景"""
        # 模拟股票分析场景
        
        # 1. 获取最新交易日期
        mock_df = pd.DataFrame({
            'latest_date': ['2024-01-15']
        })
        self.mock_db.query_df.return_value = mock_df
        
        latest_date = self.date_manager.get_latest_trading_date()
        
        # 2. 获取过去30个交易日
        past_30_days = self.date_manager.get_previous_trading_dates(latest_date, 30)
        
        # 3. 生成时间序列索引
        time_series = self.date_manager.get_time_series_dates(
            past_30_days[-1], latest_date, trading_days_only=True
        )
        
        # 4. 验证结果
        self.assert_equal(len(past_30_days), 30)
        self.assert_is_instance(time_series, pd.Datetime_index)
        self.assert_greater_equal(len(time_series), 30)
        
        # 5. 验证日期有效性
        for date_str in past_30_days:
            self.assert_true(self.date_manager.is_trading_day(date_str))
        
        # 6. 测试日期范围验证
        self.assert_true(self.date_manager.validate_date_range(
            past_30_days[-1], latest_date, max_days=50
        ))


if __name__ == '__main__':
    unittest.main() 