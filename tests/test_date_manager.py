#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from config import get_config
"""
智能日期管理器单元测试

测试 Date_manager 类的所有核心功能：
- 日期解析和格式转换
- 交易日计算
- 日期范围生成
- 缓存机制
- 性能统计
"""

import unittest
import datetime
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, Magic_mock
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from analysis.engines.date_manager import Date_manager, Date_format, Date_range, Week_day


class Test_date_manager(unittest.Test_case):
    """智能日期管理器测试类"""
    
    def set_up_Manager(self):
        """测试前置设置"""
        # 模拟ClickHouse数据库连接
        with patch('analysis.engines.date_manager.get_clickhouse_db') as mock_db:
            mock_db.return_value = Mock()
            self.date_manager = Date_manager(get_config('cache.size'), cache_ttl=get_config('cache.ttl', 3600))
    
    def test_current_datetime_and_date(self):
        """测试获取当前日期时间"""
        # 测试获取当前datetime
        current_dt = self.date_manager.get_current_datetime()
        self.assert_is_instance(current_dt, datetime.datetime)
        
        # 测试获取当前日期字符串 - 默认格式
        current_date = self.date_manager.get_current_date()
        self.assert_is_instance(current_date, str)
        self.assertRegex(current_date, r'\d{4}-\d{2}-\d{2}')
        
        # 测试获取当前日期字符串 - YYYYMMDD格式
        current_date_yyyymmdd = self.date_manager.get_current_date(Date_format.YYYYMMDD)
        self.assertRegex(current_date_yyyymmdd, r'\d{8}')
        
        # 测试获取当前日期字符串 - 时间戳格式
        current_timestamp = self.date_manager.get_current_date(Date_format.TIMESTAMP)
        self.assert_is_instance(int(current_timestamp), int)
    
    def test_parse_date_datetime_input(self):
        """测试解析datetime输入"""
        test_dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
        parsed_dt = self.date_manager.parse_date(test_dt)
        self.assert_equal(parsed_dt, test_dt)
    
    def test_parse_date_pandas_timestamp(self):
        """测试解析pandas Timestamp输入"""
        test_ts = pd.Timestamp('2024-01-15 10:30:00')
        parsed_dt = self.date_manager.parse_date(test_ts)
        self.assert_equal(parsed_dt, test_ts.to_pydatetime())
    
    def test_parse_date_timestamp_input(self):
        """测试解析时间戳输入"""
        # 秒时间戳
        timestamp_seconds = 1705315800  # 2024-01-15 10:30:00
        parsed_dt = self.date_manager.parse_date(timestamp_seconds)
        expected_dt = datetime.datetime.fromtimestamp(timestamp_seconds)
        self.assert_equal(parsed_dt, expected_dt)
        
        # 毫秒时间戳
        timestamp_ms = 1705315800000
        parsed_dt = self.date_manager.parse_date(timestamp_ms)
        expected_dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000)
        self.assert_equal(parsed_dt, expected_dt)
    
    def test_parse_date_string_with_format(self):
        """测试使用指定格式解析日期字符串"""
        # YYYYMMDD格式
        date_str = "20240115"
        parsed_dt = self.date_manager.parse_date(date_str, Date_format.YYYYMMDD)
        expected_dt = datetime.datetime(2024, 1, 15)
        self.assert_equal(parsed_dt, expected_dt)
        
        # YYYY-MM-DD格式
        date_str = "2024-01-15"
        parsed_dt = self.date_manager.parse_date(date_str, Date_format.YYYY_MM_DD)
        expected_dt = datetime.datetime(2024, 1, 15)
        self.assert_equal(parsed_dt, expected_dt)
        
        # 时间戳字符串格式
        timestamp_str = "1705315800"
        parsed_dt = self.date_manager.parse_date(timestamp_str, Date_format.TIMESTAMP)
        expected_dt = datetime.datetime.fromtimestamp(float(timestamp_str))
        self.assert_equal(parsed_dt, expected_dt)
    
    def test_auto_parse_date_string(self):
        """测试自动解析日期字符串"""
        # YYYYMMDD格式
        parsed_dt = self.date_manager.parse_date("20240115")
        expected_dt = datetime.datetime(2024, 1, 15)
        self.assert_equal(parsed_dt, expected_dt)
        
        # YYYY-MM-DD格式
        parsed_dt = self.date_manager.parse_date("2024-01-15")
        expected_dt = datetime.datetime(2024, 1, 15)
        self.assert_equal(parsed_dt, expected_dt)
        
        # YYYY-MM-DD HH:MM:SS格式
        parsed_dt = self.date_manager.parse_date("2024-01-15 10:30:00")
        expected_dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
        self.assert_equal(parsed_dt, expected_dt)
        
        # ISO格式
        parsed_dt = self.date_manager.parse_date("2024-01-15T10:30:00")
        expected_dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
        self.assert_equal(parsed_dt, expected_dt)
    
    def test_parse_date_invalid_input(self):
        """测试解析无效日期输入"""
        with self.assert_raises(ValueError):
            self.date_manager.parse_date("invalid_date")
        
        with self.assert_raises(ValueError):
            self.date_manager.parse_date(None)
        
        with self.assert_raises(ValueError):
            self.date_manager.parse_date([])
    
    def test_format_date(self):
        """测试日期格式化"""
        test_dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
        
        # 默认格式
        formatted = self.date_manager.format_date(test_dt)
        self.assertEqual(formatted, "2024-01-15")
        
        # YYYYMMDD格式
        formatted = self.date_manager.format_date(test_dt, Date_format.YYYYMMDD)
        self.assertEqual(formatted, "20240115")
        
        # 时间戳格式
        formatted = self.date_manager.format_date(test_dt, Date_format.TIMESTAMP)
        expected_timestamp = str(int(test_dt.timestamp()))
        self.assert_equal(formatted, expected_timestamp)
        
        # 从字符串输入格式化
        formatted = self.date_manager.format_date("2024-01-15", DateFormat.YYYYMMDD)
        self.assertEqual(formatted, "20240115")
    
    def test_get_latest_trading_date_from_cache(self):
        """测试从缓存获取最新交易日期"""
        # 设置缓存
        cache_key = f"latest_trading_date_{DateFormat.YYYY_MM_DD.name}"
        cached_date = "2024-01-15"
        self.date_manager.cache.set(cache_key, cached_date)
        self.date_manager._latest_date_cache = datetime.datetime(2024, 1, 15)
        self.date_manager._latest_date_cache_time = datetime.datetime.now()
        
        # 测试从缓存获取
        result = self.date_manager.get_latest_trading_date()
        self.assert_equal(result, cached_date)
    
    def test_get_latest_trading_date_from_db(self):
        """测试从数据库获取最新交易日期"""
        # 模拟数据库查询结果
        mock_result = pd.DataFrame({'latest_date': ['2024-01-15']})
        self.date_manager.db.query_df.return_value = mock_result
        
        # 清除缓存
        self.date_manager.clear_cache()
        
        # 测试从数据库获取
        result = self.date_manager.get_latest_trading_date()
        self.assertEqual(result, "2024-01-15")
    
    def test_get_latest_trading_date_estimation(self):
        """测试最新交易日期智能估算"""
        # 模拟数据库查询失败
        self.date_manager.db.query_df.side_effect = Exception("Database error")
        
        # 清除缓存
        self.date_manager.clear_cache()
        
        # 测试智能估算
        with patch('analysis.engines.date_manager.datetime') as mock_datetime:
            # 模拟周五15点后
            mock_now = datetime.datetime(2024, 1, 12, 16, 0, 0)  # 2024年1月12日是周五
            mock_datetime.datetime.now.return_value = mock_now
            mock_datetime.datetime.side_effect = lambda *args, **kw: datetime.datetime(*args, **kw)
            mock_datetime.timedelta = datetime.timedelta
            
            result = self.date_manager.get_latest_trading_date()
            # 周五15点后应该返回当天
            self.assertEqual(result, "2024-01-12")

    def test_estimate_latest_trading_date_weekend(self):
        """测试周末的最新交易日期估算"""
        with patch('analysis.engines.date_manager.datetime') as mock_datetime:
            # 模拟周六
            mock_now = datetime.datetime(2024, 1, 13, 10, 0, 0)  # 2024年1月13日是周六
            mock_datetime.datetime.now.return_value = mock_now
            mock_datetime.datetime.side_effect = lambda *args, **kw: datetime.datetime(*args, **kw)
            mock_datetime.timedelta = datetime.timedelta

            estimated = self.date_manager._estimate_latest_trading_date()
            # 周六应该回退到周五
            expected = mock_now - datetime.timedelta(days=1)
            self.assert_equal(estimated, expected)
    
    def test_get_previous_trading_dates(self):
        """测试获取前N个交易日"""
        base_date = "2024-01-15"  # 周一
        
        # 获取前3个交易日
        previous_dates = self.date_manager.get_previous_trading_dates(base_date, 3)
        
        self.assert_equal(len(previous_dates), 3)
        # 应该是周五、周四、周三
        expected_dates = ["2024-01-12", "2024-01-11", "2024-01-10"]
        self.assert_equal(previous_dates, expected_dates)
    
    def test_get_next_trading_dates(self):
        """测试获取后N个交易日"""
        base_date = "2024-01-10"  # 周三
        
        # 获取后3个交易日
        next_dates = self.date_manager.get_next_trading_dates(base_date, 3)
        
        self.assert_equal(len(next_dates), 3)
        # 应该是周四、周五、下周一
        expected_dates = ["2024-01-11", "2024-01-12", "2024-01-15"]
        self.assert_equal(next_dates, expected_dates)
    
    def test_get_date_range_trading_days_only(self):
        """测试生成交易日范围"""
        start_date = "2024-01-10"  # 周三
        end_date = "2024-01-16"    # 周二
        
        # 只包含交易日
        date_range = self.date_manager.get_date_range(start_date, end_date, trading_days_only=True)
        
        # 应该包含周三到周五，然后周一周二，共5天
        expected_dates = ["2024-01-10", "2024-01-11", "2024-01-12", "2024-01-15", "2024-01-16"]
        self.assert_equal(date_range, expected_dates)
    
    def test_get_date_range_all_days(self):
        """测试生成所有日期范围"""
        start_date = "2024-01-10"  # 周三
        end_date = "2024-01-14"    # 周日
        
        # 包含所有日期
        date_range = self.date_manager.get_date_range(start_date, end_date, trading_days_only=False)
        
        # 应该包含周三到周日，共5天
        expected_dates = ["2024-01-10", "2024-01-11", "2024-01-12", "2024-01-13", "2024-01-14"]
        self.assert_equal(date_range, expected_dates)
    
    def test_get_date_range_invalid_order(self):
        """测试无效日期顺序"""
        start_date = "2024-01-15"
        end_date = "2024-01-10"
        
        with self.assert_raises(ValueError):
            self.date_manager.get_date_range(start_date, end_date)
    
    def test_get_predefined_date_range(self):
        """测试获取预定义日期范围"""
        base_date = "2024-01-15"
        
        # 获取最近7天
        end_date, date_list = self.date_manager.get_predefined_date_range(
            Date_range.LAST_7_DAYS, base_date, trading_days_only=True
        )
        
        self.assertEqual(end_date, "2024-01-15")
        self.assert_is_instance(date_list, list)
        self.assert_greater(len(date_list), 0)
    
    def test_is_trading_day(self):
        """测试判断是否为交易日"""
        # 周一到周五应该是交易日
        self.assertTrue(self.date_manager.is_trading_day("2024-01-15"))  # 周一
        self.assertTrue(self.date_manager.is_trading_day("2024-01-16"))  # 周二
        self.assertTrue(self.date_manager.is_trading_day("2024-01-17"))  # 周三
        self.assertTrue(self.date_manager.is_trading_day("2024-01-18"))  # 周四
        self.assertTrue(self.date_manager.is_trading_day("2024-01-19"))  # 周五
        
        # 周六周日不应该是交易日
        self.assertFalse(self.date_manager.is_trading_day("2024-01-13"))  # 周六
        self.assertFalse(self.date_manager.is_trading_day("2024-01-14"))  # 周日
    
    def test_get_trading_days_between(self):
        """测试计算两个日期之间的交易日数量"""
        start_date = "2024-01-10"  # 周三
        end_date = "2024-01-16"    # 周二
        
        trading_days = self.date_manager.get_trading_days_between(start_date, end_date)
        
        # 周三到周二，包含周三到周五，然后周一周二，共5天
        self.assert_equal(trading_days, 5)
    
    def test_add_trading_days_positive(self):
        """测试增加交易日数"""
        base_date = "2024-01-10"  # 周三
        
        # 增加3个交易日
        result_date = self.date_manager.add_trading_days(base_date, 3)
        
        # 应该是下周一
        self.assertEqual(result_date, "2024-01-15")
    
    def test_add_trading_days_negative(self):
        """测试减少交易日数"""
        base_date = "2024-01-15"  # 周一
        
        # 减少3个交易日
        result_date = self.date_manager.add_trading_days(base_date, -3)
        
        # 应该是上周三
        self.assertEqual(result_date, "2024-01-10")
    
    def test_add_trading_days_zero(self):
        """测试增加0个交易日"""
        base_date = "2024-01-15"
        
        result_date = self.date_manager.add_trading_days(base_date, 0)
        
        self.assert_equal(result_date, base_date)
    
    def test_validate_date_range_valid(self):
        """测试有效日期范围验证"""
        start_date = "2024-01-10"
        end_date = "2024-01-15"
        
        # 有效范围
        self.assert_true(self.date_manager.validate_date_range(start_date, end_date))
        
        # 有效范围且在最大天数限制内
        self.assert_true(self.date_manager.validate_date_range(start_date, end_date, max_days=10))
    
    def test_validate_date_range_invalid(self):
        """测试无效日期范围验证"""
        start_date = "2024-01-15"
        end_date = "2024-01-10"
        
        # 无效范围（开始日期晚于结束日期）
        self.assert_false(self.date_manager.validate_date_range(start_date, end_date))
        
        # 超出最大天数限制
        start_date = "2024-01-01"
        end_date = "2024-01-31"
        self.assert_false(self.date_manager.validate_date_range(start_date, end_date, max_days=10))
    
    def test_validate_date_range_parse_error(self):
        """测试日期解析错误的验证"""
        # 无效日期格式
        self.assertFalse(self.date_manager.validate_date_range("invalid_date", "2024-01-15"))
    
    def test_get_time_series_dates_trading_days(self):
        """测试生成交易日时间序列索引"""
        start_date = "2024-01-10"  # 周三
        end_date = "2024-01-16"    # 周二
        
        date_index = self.date_manager.get_time_series_dates(
            start_date, end_date, frequency='D', trading_days_only=True
        )
        
        self.assert_is_instance(date_index, pd.Datetime_index)
        # 应该包含5个交易日
        self.assert_equal(len(date_index), 5)
    
    def test_get_time_series_dates_all_days(self):
        """测试生成所有日期时间序列索引"""
        start_date = "2024-01-10"  # 周三
        end_date = "2024-01-14"    # 周日
        
        date_index = self.date_manager.get_time_series_dates(
            start_date, end_date, frequency='D', trading_days_only=False
        )
        
        self.assert_is_instance(date_index, pd.Datetime_index)
        # 应该包含5天
        self.assert_equal(len(date_index), 5)
    
    def test_cache_functionality_Manager(self):
        """测试缓存功能"""
        # 清除缓存
        self.date_manager.clear_cache()
        
        # 设置缓存
        self.date_manager.cache.set("test_key", "test_value")
        
        # 获取缓存
        cached_value = self.date_manager.cache.get("test_key")
        self.assertEqual(cached_value, "test_value")
        
        # 清除缓存
        self.date_manager.clear_cache()
        cached_value = self.date_manager.cache.get("test_key")
        self.assert_is_none(cached_value)
    
    def test_performance_stats(self):
        """测试性能统计"""
        # 设置一些缓存数据
        self.date_manager.cache.set("key1", "value1")
        self.date_manager.cache.set("key2", "value2")
        
        stats = self.date_manager.get_performance_stats()
        
        self.assert_is_instance(stats, dict)
        self.assertIn('cache_size', stats)
        self.assertIn('cache_max_size', stats)
        self.assertIn('cache_hit_rate', stats)
        self.assertIn('latest_date_cached', stats)
        self.assertIn('cache_expire_seconds', stats)
        
        # 验证统计数据
        self.assertGreaterEqual(stats['cache_size'], 2)
        self.assertEqual(stats['cache_max_size'], 100)
        self.assertEqual(stats['cache_expire_seconds'], 300)
    
    def test_string_representations(self):
        """测试字符串表示方法"""
        # 设置一些测试数据
        self.date_manager._latest_date_cache = datetime.datetime(2024, 1, 15)
        
        # 测试__str__
        str_repr = str(self.date_manager)
        self.assertIn("DateManager", str_repr)
        self.assertIn("cache_size", str_repr)
        self.assertIn("latest_date", str_repr)
        
        # 测试__repr__
        repr_str = repr(self.date_manager)
        self.assertIn("DateManager", repr_str)
        self.assertIn("cache_size", repr_str)
        self.assertIn("cache_hit_rate", repr_str)
        self.assertIn("latest_date", repr_str)


if __name__ == '__main__':
    unittest.main(verbosity=2) 