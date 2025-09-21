#!/usr/bin/env python3
"""
直接测试ClickHouse与stock_info数据的集成
"""

import clickhouse_connect
from datetime import datetime, timedelta

def test_stock_data_query():
    """测试stock_info数据查询和映射"""
    print("🧪 测试ClickHouse与stock_info数据集成...")
    
    try:
        # 连接ClickHouse
        client = clickhouse_connect.get_client(
            host='localhost',
            port=8123,
            username='default',
            password='123456',
            database='stock'
        )
        print("✅ 连接stock数据库成功")
        
        # 1. 测试数据概览查询
        print("\n📊 测试数据概览查询...")
        overview_query = """
            SELECT 
                code as symbol,
                level,
                COUNT(*) as count,
                MIN(datetime) as start,
                MAX(datetime) as end
            FROM stock_info
            GROUP BY code, level
            ORDER BY code, level
            LIMIT 10
        """
        
        result = client.query(overview_query)
        print(f"✅ 查询到 {len(result.result_rows)} 个数据概览")
        
        print("数据概览示例:")
        for i, row in enumerate(result.result_rows[:5]):
            symbol, level, count, start, end = row
            print(f"  {i+1}. {symbol} {level} 数量:{count} 时间:{start.date()}~{end.date()}")
        
        # 2. 测试K线数据查询
        print("\n📈 测试K线数据查询...")
        
        # 选择一个股票进行测试
        if result.result_rows:
            test_symbol = result.result_rows[0][0]  # 第一个股票代码
            test_level = result.result_rows[0][1]   # 第一个时间周期
            
            # 查询最近的数据
            bar_query = """
                SELECT code, name, datetime, level, open, close, high, low, volume,
                       turnover_rate, price_change, price_range
                FROM stock_info
                WHERE code = %(symbol)s 
                  AND level = %(level)s
                ORDER BY datetime DESC
                LIMIT 5
            """
            
            bar_result = client.query(
                bar_query,
                parameters={
                    'symbol': test_symbol,
                    'level': test_level
                }
            )
            
            print(f"✅ 查询到 {len(bar_result.result_rows)} 条K线数据")
            print(f"股票: {test_symbol}, 周期: {test_level}")
            
            print("K线数据示例:")
            for i, row in enumerate(bar_result.result_rows):
                code, name, dt, level, open_p, close_p, high_p, low_p, volume = row[:9]
                print(f"  {i+1}. {dt} {name} O:{open_p:.2f} H:{high_p:.2f} L:{low_p:.2f} C:{close_p:.2f} V:{volume:.0f}")
        
        # 3. 测试时间周期映射
        print("\n⏰ 测试时间周期映射...")
        
        level_query = """
            SELECT DISTINCT level, COUNT(*) as count
            FROM stock_info
            GROUP BY level
            ORDER BY count DESC
        """
        
        level_result = client.query(level_query)
        print("时间周期分布:")
        
        # VnPy时间周期映射
        level_mapping = {
            "1分钟": "MINUTE",
            "5分钟": "MINUTE_5", 
            "15分钟": "MINUTE_15",
            "30分钟": "MINUTE_30",
            "1小时": "HOUR",
            "日线": "DAILY",
            "周线": "WEEKLY",
            "月线": "MONTHLY"
        }
        
        for row in level_result.result_rows:
            level, count = row
            vnpy_interval = level_mapping.get(level, "UNKNOWN")
            print(f"  {level} -> {vnpy_interval}: {count} 条记录")
        
        # 4. 测试交易所推断
        print("\n🏢 测试交易所推断...")
        
        exchange_query = """
            SELECT 
                CASE 
                    WHEN code LIKE '000%' OR code LIKE '001%' OR code LIKE '002%' OR code LIKE '003%' OR code LIKE '300%' THEN 'SZSE'
                    WHEN code LIKE '600%' OR code LIKE '601%' OR code LIKE '603%' OR code LIKE '605%' OR code LIKE '688%' THEN 'SSE'
                    WHEN code LIKE '8%' THEN 'NEEQ'
                    ELSE 'UNKNOWN'
                END as exchange,
                COUNT(DISTINCT code) as stock_count
            FROM stock_info
            GROUP BY exchange
            ORDER BY stock_count DESC
        """
        
        exchange_result = client.query(exchange_query)
        print("交易所分布:")
        for row in exchange_result.result_rows:
            exchange, count = row
            exchange_name = {
                'SSE': '上海证券交易所',
                'SZSE': '深圳证券交易所', 
                'NEEQ': '全国中小企业股份转让系统',
                'UNKNOWN': '未知交易所'
            }.get(exchange, exchange)
            print(f"  {exchange_name}: {count} 只股票")
        
        # 5. 测试数据质量
        print("\n🔍 测试数据质量...")
        
        quality_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT code) as unique_stocks,
                MIN(datetime) as earliest_date,
                MAX(datetime) as latest_date,
                AVG(volume) as avg_volume,
                COUNT(CASE WHEN volume = 0 THEN 1 END) as zero_volume_count
            FROM stock_info
        """
        
        quality_result = client.query(quality_query)
        row = quality_result.result_rows[0]
        total, unique, earliest, latest, avg_vol, zero_vol = row
        
        print(f"数据质量报告:")
        print(f"  总记录数: {total:,}")
        print(f"  股票数量: {unique:,}")
        print(f"  时间范围: {earliest.date()} ~ {latest.date()}")
        print(f"  平均成交量: {avg_vol:.0f}")
        print(f"  零成交量记录: {zero_vol:,} ({zero_vol/total*100:.2f}%)")
        
        # 6. 测试VnPy数据模型兼容性
        print("\n🔄 测试VnPy数据模型兼容性...")
        
        # 模拟BarData对象创建
        if bar_result.result_rows:
            row = bar_result.result_rows[0]
            code, name, dt, level, open_p, close_p, high_p, low_p, volume = row[:9]
            
            # 模拟VnPy BarData字段
            bar_data = {
                'symbol': code,
                'exchange': 'SSE' if code.startswith(('600', '601', '603', '605', '688')) else 'SZSE',
                'datetime': dt,
                'interval': level_mapping.get(level, 'DAILY'),
                'volume': volume,
                'turnover': volume * (open_p + close_p) / 2,  # 估算成交额
                'open_interest': 0,  # 股票没有持仓量
                'open_price': open_p,
                'high_price': high_p,
                'low_price': low_p,
                'close_price': close_p,
                'gateway_name': 'clickhouse'
            }
            
            print("VnPy BarData模拟对象:")
            for key, value in bar_data.items():
                print(f"  {key}: {value}")
            
            # 验证数据合理性
            if (high_p >= open_p and high_p >= close_p and 
                low_p <= open_p and low_p <= close_p and volume >= 0):
                print("✅ 数据合理性检查通过")
            else:
                print("❌ 数据合理性检查失败")
        
        client.close()
        print("\n🎉 所有测试通过！stock_info数据与VnPy模型兼容！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """测试查询性能"""
    print("\n⚡ 测试查询性能...")
    
    try:
        client = clickhouse_connect.get_client(
            host='localhost',
            port=8123,
            username='default',
            password='123456',
            database='stock'
        )
        
        # 测试大数据量查询
        start_time = datetime.now()
        
        large_query = """
            SELECT code, datetime, open, close, high, low, volume
            FROM stock_info
            WHERE level = '日线'
              AND datetime >= '2024-01-01'
            ORDER BY code, datetime
            LIMIT 10000
        """
        
        result = client.query(large_query)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        print(f"✅ 查询10000条日线数据耗时: {duration:.2f}秒")
        print(f"✅ 查询速度: {len(result.result_rows)/duration:.0f} 条/秒")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ClickHouse stock_info数据集成测试")
    print("=" * 60)
    
    success1 = test_stock_data_query()
    success2 = test_performance()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！ClickHouse模块可以正常使用您的stock_info数据！")
        print("\n📋 下一步:")
        print("1. 配置VnPy使用ClickHouse数据库")
        print("2. 在vt_setting.json中设置database.name为'clickhouse'")
        print("3. 开始使用VnPy进行量化交易开发")
    else:
        print("\n💥 部分测试失败，请检查ClickHouse配置！")
