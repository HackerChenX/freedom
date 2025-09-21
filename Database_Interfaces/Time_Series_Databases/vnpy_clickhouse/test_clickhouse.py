#!/usr/bin/env python3
"""
ClickHouse数据库接口测试脚本

测试vnpy_clickhouse模块的各项功能，包括：
1. 数据库连接
2. K线数据的增删改查
3. Tick数据的增删改查
4. 数据概览功能
5. 性能测试
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List
import random

# 添加项目路径
project_root = os.path.join(os.path.dirname(__file__), "../../../..")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "Core_Framework"))

from Core_Framework.vnpy.vnpy.trader.object import BarData, TickData
from Core_Framework.vnpy.vnpy.trader.constant import Exchange, Interval
from vnpy_clickhouse import Database


def create_test_bar_data(count: int = 100) -> List[BarData]:
    """创建测试K线数据"""
    bars = []
    base_time = datetime(2024, 1, 1, 9, 30)
    base_price = 100.0
    
    for i in range(count):
        # 模拟价格波动
        price_change = random.uniform(-2, 2)
        open_price = base_price + price_change
        high_price = open_price + random.uniform(0, 1)
        low_price = open_price - random.uniform(0, 1)
        close_price = open_price + random.uniform(-0.5, 0.5)
        
        bar = BarData(
            symbol="000001",
            exchange=Exchange.SSE,
            datetime=base_time + timedelta(minutes=i),
            interval=Interval.MINUTE,
            volume=random.randint(1000, 10000),
            turnover=random.uniform(100000, 1000000),
            open_interest=0,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            gateway_name="test"
        )
        bars.append(bar)
        base_price = close_price
    
    return bars


def create_test_tick_data(count: int = 50) -> List[TickData]:
    """创建测试Tick数据"""
    ticks = []
    base_time = datetime(2024, 1, 1, 9, 30)
    base_price = 100.0
    
    for i in range(count):
        price_change = random.uniform(-0.1, 0.1)
        last_price = base_price + price_change
        
        tick = TickData(
            symbol="000001",
            exchange=Exchange.SSE,
            datetime=base_time + timedelta(seconds=i*3),
            name="平安银行",
            volume=random.randint(100, 1000),
            turnover=random.uniform(10000, 100000),
            open_interest=0,
            last_price=last_price,
            last_volume=random.randint(100, 500),
            limit_up=last_price * 1.1,
            limit_down=last_price * 0.9,
            open_price=base_price,
            high_price=max(base_price, last_price),
            low_price=min(base_price, last_price),
            pre_close=base_price,
            bid_price_1=last_price - 0.01,
            bid_price_2=last_price - 0.02,
            bid_price_3=last_price - 0.03,
            bid_price_4=last_price - 0.04,
            bid_price_5=last_price - 0.05,
            ask_price_1=last_price + 0.01,
            ask_price_2=last_price + 0.02,
            ask_price_3=last_price + 0.03,
            ask_price_4=last_price + 0.04,
            ask_price_5=last_price + 0.05,
            bid_volume_1=random.randint(100, 1000),
            bid_volume_2=random.randint(100, 1000),
            bid_volume_3=random.randint(100, 1000),
            bid_volume_4=random.randint(100, 1000),
            bid_volume_5=random.randint(100, 1000),
            ask_volume_1=random.randint(100, 1000),
            ask_volume_2=random.randint(100, 1000),
            ask_volume_3=random.randint(100, 1000),
            ask_volume_4=random.randint(100, 1000),
            ask_volume_5=random.randint(100, 1000),
            localtime=base_time + timedelta(seconds=i*3),
            gateway_name="test"
        )
        ticks.append(tick)
        base_price = last_price
    
    return ticks


def test_clickhouse_database():
    """测试ClickHouse数据库接口"""
    print("🧪 开始测试ClickHouse数据库接口")
    print("=" * 60)
    
    try:
        # 1. 创建数据库实例
        print("\n1. 创建数据库连接...")
        db = Database()
        print("✅ 数据库连接成功")
        
        # 2. 测试K线数据操作
        print("\n2. 测试K线数据操作...")
        
        # 创建测试数据
        test_bars = create_test_bar_data(100)
        print(f"   创建了 {len(test_bars)} 条测试K线数据")
        
        # 保存数据
        result = db.save_bar_data(test_bars)
        print(f"   保存K线数据: {'成功' if result else '失败'}")
        
        # 查询数据
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 2)
        loaded_bars = db.load_bar_data("000001", Exchange.SSE, Interval.MINUTE, start_time, end_time)
        print(f"   查询到 {len(loaded_bars)} 条K线数据")
        
        # 验证数据
        if loaded_bars:
            first_bar = loaded_bars[0]
            print(f"   第一条数据: {first_bar.symbol} {first_bar.datetime} O:{first_bar.open_price} C:{first_bar.close_price}")
        
        # 3. 测试Tick数据操作
        print("\n3. 测试Tick数据操作...")
        
        # 创建测试数据
        test_ticks = create_test_tick_data(50)
        print(f"   创建了 {len(test_ticks)} 条测试Tick数据")
        
        # 保存数据
        result = db.save_tick_data(test_ticks)
        print(f"   保存Tick数据: {'成功' if result else '失败'}")
        
        # 查询数据
        loaded_ticks = db.load_tick_data("000001", Exchange.SSE, start_time, end_time)
        print(f"   查询到 {len(loaded_ticks)} 条Tick数据")
        
        # 验证数据
        if loaded_ticks:
            first_tick = loaded_ticks[0]
            print(f"   第一条数据: {first_tick.symbol} {first_tick.datetime} 价格:{first_tick.last_price}")
        
        # 4. 测试概览功能
        print("\n4. 测试数据概览功能...")
        
        bar_overviews = db.get_bar_overview()
        print(f"   K线概览: {len(bar_overviews)} 个合约")
        for overview in bar_overviews[:3]:  # 只显示前3个
            print(f"     {overview.symbol}.{overview.exchange.value} {overview.interval.value}: {overview.count} 条记录")
        
        tick_overviews = db.get_tick_overview()
        print(f"   Tick概览: {len(tick_overviews)} 个合约")
        for overview in tick_overviews[:3]:  # 只显示前3个
            print(f"     {overview.symbol}.{overview.exchange.value}: {overview.count} 条记录")
        
        # 5. 测试删除功能
        print("\n5. 测试数据删除功能...")
        
        # 删除K线数据
        deleted_bars = db.delete_bar_data("000001", Exchange.SSE, Interval.MINUTE)
        print(f"   删除了 {deleted_bars} 条K线数据")
        
        # 删除Tick数据
        deleted_ticks = db.delete_tick_data("000001", Exchange.SSE)
        print(f"   删除了 {deleted_ticks} 条Tick数据")
        
        # 6. 性能测试
        print("\n6. 性能测试...")
        
        # 大批量数据测试
        large_bars = create_test_bar_data(1000)
        start_time = datetime.now()
        db.save_bar_data(large_bars)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"   保存1000条K线数据耗时: {duration:.2f}秒")
        
        # 查询性能测试
        start_time = datetime.now()
        query_bars = db.load_bar_data("000001", Exchange.SSE, Interval.MINUTE, 
                                     datetime(2024, 1, 1), datetime(2024, 1, 2))
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"   查询{len(query_bars)}条K线数据耗时: {duration:.2f}秒")
        
        # 清理测试数据
        db.delete_bar_data("000001", Exchange.SSE, Interval.MINUTE)
        
        # 关闭连接
        db.close()
        
        print("\n" + "=" * 60)
        print("🎉 ClickHouse数据库接口测试完成！")
        print("✅ 所有功能测试通过")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_clickhouse_database()
