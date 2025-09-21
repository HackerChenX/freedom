#!/usr/bin/env python3
"""
测试ClickHouse模块与真实stock_info数据的集成
"""

import sys
import os
from datetime import datetime, timedelta

# 添加项目路径
project_root = os.path.join(os.path.dirname(__file__), "../../../..")
core_framework_path = os.path.join(project_root, "Core_Framework", "vnpy")
sys.path.insert(0, project_root)
sys.path.insert(0, core_framework_path)

try:
    from vnpy.trader.object import BarData, TickData
    from vnpy.trader.constant import Exchange, Interval
    from vnpy_clickhouse import Database
except ImportError as e:
    # 尝试直接导入
    try:
        import clickhouse_connect
        from datetime import datetime, timedelta
        print("✅ 基础依赖导入成功，将进行简化测试")
        SIMPLE_TEST = True
    except ImportError:
        print(f"❌ 导入失败: {e}")
        print("请确保VnPy核心框架路径正确")
        print(f"项目根目录: {project_root}")
        print(f"核心框架路径: {core_framework_path}")
        sys.exit(1)
else:
    SIMPLE_TEST = False

def test_real_data_integration():
    """测试与真实stock_info数据的集成"""
    print("🧪 测试ClickHouse模块与真实数据集成...")
    
    try:
        # 创建数据库实例
        database = Database()
        print("✅ 数据库连接成功")
        
        # 1. 测试数据概览
        print("\n📊 测试数据概览...")
        overviews = database.get_bar_overview()
        print(f"✅ 获取到 {len(overviews)} 个数据概览")
        
        # 显示前10个概览
        print("前10个数据概览:")
        for i, overview in enumerate(overviews[:10]):
            print(f"  {i+1}. {overview.symbol} {overview.exchange.value} {overview.interval.value} "
                  f"数量:{overview.count} 时间:{overview.start.date()}~{overview.end.date()}")
        
        # 2. 测试数据查询
        print("\n📈 测试数据查询...")
        
        # 选择一个有数据的股票进行测试
        if overviews:
            test_overview = overviews[0]
            symbol = test_overview.symbol
            exchange = test_overview.exchange
            interval = test_overview.interval
            
            # 查询最近30天的数据
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
            
            print(f"查询 {symbol} {exchange.value} {interval.value} 数据...")
            bars = database.load_bar_data(symbol, exchange, interval, start_time, end_time)
            
            print(f"✅ 查询到 {len(bars)} 条K线数据")
            
            # 显示前5条数据
            if bars:
                print("前5条K线数据:")
                for i, bar in enumerate(bars[:5]):
                    print(f"  {i+1}. {bar.datetime} O:{bar.open_price:.2f} H:{bar.high_price:.2f} "
                          f"L:{bar.low_price:.2f} C:{bar.close_price:.2f} V:{bar.volume:.0f}")
            
            # 3. 测试不同时间周期
            print(f"\n⏰ 测试不同时间周期数据...")
            intervals_to_test = [Interval.DAILY, Interval.WEEKLY, Interval.MINUTE_15]
            
            for test_interval in intervals_to_test:
                try:
                    bars = database.load_bar_data(symbol, exchange, test_interval, start_time, end_time)
                    print(f"  {test_interval.value}: {len(bars)} 条数据")
                except Exception as e:
                    print(f"  {test_interval.value}: 查询失败 - {e}")
        
        # 4. 测试数据统计
        print(f"\n📈 数据统计分析...")
        
        # 按交易所统计
        sse_count = sum(1 for o in overviews if o.exchange == Exchange.SSE)
        szse_count = sum(1 for o in overviews if o.exchange == Exchange.SZSE)
        print(f"上交所股票: {sse_count} 个")
        print(f"深交所股票: {szse_count} 个")
        
        # 按时间周期统计
        interval_stats = {}
        for overview in overviews:
            interval_name = overview.interval.value
            if interval_name not in interval_stats:
                interval_stats[interval_name] = 0
            interval_stats[interval_name] += 1
        
        print("时间周期分布:")
        for interval_name, count in sorted(interval_stats.items()):
            print(f"  {interval_name}: {count} 个股票")
        
        # 5. 测试数据完整性
        print(f"\n🔍 数据完整性检查...")
        
        # 检查是否有最新数据
        latest_overviews = sorted(overviews, key=lambda x: x.end, reverse=True)[:5]
        print("最新数据的股票:")
        for overview in latest_overviews:
            print(f"  {overview.symbol}: 最新数据时间 {overview.end}")
        
        print("\n🎉 所有测试通过！ClickHouse模块与真实数据集成成功！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            database.close()
            print("✅ 数据库连接已关闭")
        except:
            pass

def test_data_mapping():
    """测试数据字段映射的正确性"""
    print("\n🔄 测试数据字段映射...")
    
    try:
        database = Database()
        
        # 获取一个样本数据
        overviews = database.get_bar_overview()
        if not overviews:
            print("❌ 没有找到数据概览")
            return False
        
        # 选择日线数据进行测试
        daily_overview = None
        for overview in overviews:
            if overview.interval == Interval.DAILY:
                daily_overview = overview
                break
        
        if not daily_overview:
            print("❌ 没有找到日线数据")
            return False
        
        symbol = daily_overview.symbol
        exchange = daily_overview.exchange
        
        # 查询最近5天数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=5)
        
        bars = database.load_bar_data(symbol, exchange, Interval.DAILY, start_time, end_time)
        
        if bars:
            bar = bars[0]
            print(f"✅ 数据映射测试成功")
            print(f"  股票代码: {bar.symbol}")
            print(f"  交易所: {bar.exchange.value}")
            print(f"  时间: {bar.datetime}")
            print(f"  开盘价: {bar.open_price}")
            print(f"  最高价: {bar.high_price}")
            print(f"  最低价: {bar.low_price}")
            print(f"  收盘价: {bar.close_price}")
            print(f"  成交量: {bar.volume}")
            print(f"  成交额: {bar.turnover:.2f}")
            print(f"  持仓量: {bar.open_interest}")
            
            # 验证数据合理性
            if (bar.high_price >= bar.open_price and 
                bar.high_price >= bar.close_price and
                bar.low_price <= bar.open_price and 
                bar.low_price <= bar.close_price and
                bar.volume >= 0):
                print("✅ 数据合理性检查通过")
                return True
            else:
                print("❌ 数据合理性检查失败")
                return False
        else:
            print("❌ 没有查询到数据")
            return False
            
    except Exception as e:
        print(f"❌ 数据映射测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ClickHouse模块真实数据集成测试")
    print("=" * 60)
    
    success1 = test_real_data_integration()
    success2 = test_data_mapping()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！模块可以正常使用！")
    else:
        print("\n💥 部分测试失败，请检查配置和实现！")
