#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
数据库重构测试脚本

用于验证数据库查询重构后的功能是否正常
"""

import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd

from db.clickhouse_db import get_clickhouse_db
from db.data_manager import DataManager
from analysis.buypoints.period_data_processor import PeriodDataProcessor
from models.stock_info import StockInfo
from enums.period import Period

def test_clickhouse_db_get_stock_info():
    """测试 get_stock_info 方法"""
    db = get_clickhouse_db()
    
    # 测试单个股票查询
    stock_info = db.get_stock_info(
        stock_code="000001",
        level="日线",
        start_date="2023-01-01",
        end_date="2023-01-10"
    )
    
    assert isinstance(stock_info, StockInfo)
    df = stock_info.to_dataframe()
    assert not df.empty
    assert "000001" in df["code"].values
    # 验证返回了所有字段
    expected_fields = ["code", "name", "date", "level", "open", "high", "low", "close", 
                      "volume", "turnover_rate", "price_change", "price_range", "industry",
                      "market", "area", "list_date", "is_valid", "market_cap", "pe_ratio", "pb_ratio"]
    assert all(field in df.columns for field in expected_fields)
    
    # 测试多股票查询
    stock_info = db.get_stock_info(
        stock_code=["000001", "600000"],
        level="日线",
        start_date="2023-01-01",
        end_date="2023-01-10"
    )
    
    assert isinstance(stock_info, StockInfo)
    df = stock_info.to_dataframe()
    assert not df.empty
    assert set(["000001", "600000"]).issubset(set(df["code"].values))
    # 验证返回了所有字段
    assert all(field in df.columns for field in expected_fields)
    
    # 测试带过滤条件的查询
    stock_info = db.get_stock_info(
        level="日线",
        filters={"industry": "银行"}
    )
    
    assert isinstance(stock_info, StockInfo)
    df = stock_info.to_dataframe()
    assert not df.empty
    assert all(df["industry"] == "银行")
    # 验证返回了所有字段
    assert all(field in df.columns for field in expected_fields)
    
    # 测试空结果
    stock_info = db.get_stock_info(
        stock_code="999999",  # 不存在的股票代码
        level="日线"
    )
    
    assert isinstance(stock_info, StockInfo)
    df = stock_info.to_dataframe()
    assert df.empty
    assert stock_info.code == "999999"

def test_data_manager():
    """测试 DataManager 类"""
    data_manager = DataManager()
    
    # 测试 get_stock_info 方法
    stock_info = data_manager.get_stock_info(
        stock_code="000001",
        level="日线",
        start_date="2023-01-01",
        end_date="2023-01-10"
    )
    
    assert isinstance(stock_info, StockInfo)
    assert not stock_info.to_dataframe().empty
    
    # 测试缓存功能
    # 第二次调用应该从缓存获取
    cached_stock_info = data_manager.get_stock_info(
        stock_code="000001",
        level="日线",
        start_date="2023-01-01",
        end_date="2023-01-10"
    )
    
    assert isinstance(cached_stock_info, StockInfo)
    assert cached_stock_info.to_dataframe().equals(stock_info.to_dataframe())
    
    # 测试缓存统计
    stats = data_manager.get_cache_stats()
    assert stats["hits"] > 0
    
    # 测试清除缓存
    data_manager.clear_cache()
    stats_after_clear = data_manager.get_cache_stats()
    assert stats_after_clear["size"] == 0

def test_period_data_processor():
    """测试 PeriodDataProcessor 类"""
    processor = PeriodDataProcessor()
    
    # 测试获取多周期数据
    multi_period_data = processor.get_multi_period_data(
        stock_code="000001",
        end_date="2023-01-10",
        periods=["日线", "周线", "月线"]
    )
    
    assert isinstance(multi_period_data, dict)
    assert all(isinstance(df, pd.DataFrame) for df in multi_period_data.values())
    assert all(not df.empty for df in multi_period_data.values())
    
    # 测试获取单个周期数据
    kline_data = processor._get_kline_data(
        stock_code="000001",
        end_date="2023-01-10",
        period=KlinePeriod.DAILY
    )
    
    assert isinstance(kline_data, pd.DataFrame)
    assert not kline_data.empty

def test_limit_by_functionality():
    """测试 LIMIT BY 功能"""
    db = get_clickhouse_db()
    
    # 测试获取每个股票的最新记录
    stock_info = db.get_stock_info(
        stock_code=["000001", "600000"],
        level="日线",
        order_by="code, date DESC"
    )
    
    assert isinstance(stock_info, StockInfo)
    df = stock_info.to_dataframe()
    
    # 检查是否每个股票只有一条记录
    assert len(df["code"].unique()) == 2
    assert all(df.groupby("code").size() == 1)

def main():
    """运行所有测试"""
    test_clickhouse_db_get_stock_info()
    test_data_manager()
    test_period_data_processor()
    test_limit_by_functionality()
    print("所有测试通过！")

if __name__ == "__main__":
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, root_dir)
    
    # 运行测试
    main()
    sys.exit(0) 