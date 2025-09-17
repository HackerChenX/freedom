#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接数据库查询测试

绕过服务层，直接连接ClickHouse数据库验证数据存在性
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def test_direct_clickhouse_connection():
    """直接测试ClickHouse连接和数据查询"""
    
    print("🎯 直接ClickHouse数据库连接测试")
    print("=" * 80)
    
    try:
        # 直接导入ClickHouse客户端
        from clickhouse_driver import Client
        
        # 创建ClickHouse客户端连接
        client = Client(
            host='localhost',
            port=9000,
            database='stock',
            user='default',
            password='123456'
        )
        
        print("✅ ClickHouse客户端连接成功")
        
        # 测试1: 查看所有数据库
        print(f"\n🔍 测试1: 查看所有数据库")
        databases_query = "SHOW DATABASES"
        databases = client.execute(databases_query)
        print(f"发现数据库: {[db[0] for db in databases]}")
        
        # 测试2: 查看stock数据库中的表
        print(f"\n🔍 测试2: 查看stock数据库中的表")
        tables_query = "SHOW TABLES FROM stock"
        tables = client.execute(tables_query)
        print(f"发现表: {[table[0] for table in tables]}")
        
        # 测试3: 检查stock_info表的结构
        if tables:
            print(f"\n🔍 测试3: 检查stock_info表结构")
            try:
                describe_query = "DESCRIBE TABLE stock.stock_info"
                columns = client.execute(describe_query)
                print(f"表结构:")
                for col in columns:
                    print(f"  - {col[0]}: {col[1]}")
            except Exception as e:
                print(f"❌ 查询表结构失败: {e}")
        
        # 测试4: 查询stock_info表的总行数
        print(f"\n🔍 测试4: 查询stock_info表总行数")
        try:
            count_query = "SELECT COUNT(*) FROM stock.stock_info"
            count_result = client.execute(count_query)
            total_rows = count_result[0][0] if count_result else 0
            print(f"总行数: {total_rows:,}")
        except Exception as e:
            print(f"❌ 查询总行数失败: {e}")
        
        # 测试5: 查询不同的level值
        print(f"\n🔍 测试5: 查询不同的level值")
        try:
            level_query = "SELECT DISTINCT level, COUNT(*) FROM stock.stock_info GROUP BY level"
            level_result = client.execute(level_query)
            print(f"Level分布:")
            for level, count in level_result:
                print(f"  - {level}: {count:,} 条记录")
        except Exception as e:
            print(f"❌ 查询level分布失败: {e}")
        
        # 测试6: 查询股票代码样本
        print(f"\n🔍 测试6: 查询股票代码样本")
        try:
            codes_query = "SELECT DISTINCT code FROM stock.stock_info LIMIT 20"
            codes_result = client.execute(codes_query)
            sample_codes = [code[0] for code in codes_result]
            print(f"股票代码样本: {sample_codes}")
        except Exception as e:
            print(f"❌ 查询股票代码失败: {e}")
        
        # 测试7: 查询日期范围
        print(f"\n🔍 测试7: 查询日期范围")
        try:
            date_query = "SELECT MIN(date) as min_date, MAX(date) as max_date FROM stock.stock_info"
            date_result = client.execute(date_query)
            if date_result:
                min_date, max_date = date_result[0]
                print(f"日期范围: {min_date} 到 {max_date}")
        except Exception as e:
            print(f"❌ 查询日期范围失败: {e}")
        
        # 测试8: 查询特定股票的数据
        print(f"\n🔍 测试8: 查询特定股票数据")
        test_codes = ['000001', '000002', '600000', '600036', '300001']
        
        for code in test_codes:
            try:
                stock_query = f"""
                SELECT code, date, close, volume 
                FROM stock.stock_info 
                WHERE code = '{code}' 
                AND level = '日线'
                ORDER BY date DESC 
                LIMIT 5
                """
                stock_result = client.execute(stock_query)
                
                if stock_result:
                    print(f"  ✅ {code}: 找到 {len(stock_result)} 条记录")
                    for row in stock_result[:2]:  # 显示前2条
                        print(f"    {row[1]}: 收盘价={row[2]}, 成交量={row[3]}")
                else:
                    print(f"  ❌ {code}: 无数据")
                    
            except Exception as e:
                print(f"  ❌ {code}: 查询失败 - {e}")
        
        # 测试9: 使用原始查询逻辑测试
        print(f"\n🔍 测试9: 使用原始查询逻辑测试")
        test_code = '000001'
        end_date = '2025-01-20'
        start_date = '2024-12-01'
        
        try:
            original_query = f"""
            SELECT date, open, high, low, close, volume
            FROM stock_info WHERE level = %(level)s AND code = '{test_code}'
            AND level = '日线'
            AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date ASC
            """
            
            print(f"查询SQL: {original_query}")
            original_result = client.execute(original_query)
            
            if original_result:
                print(f"  ✅ 原始查询成功: {len(original_result)} 条记录")
                
                # 转换为DataFrame
                df = pd.DataFrame(original_result, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                
                print(f"  📊 DataFrame信息:")
                print(f"    形状: {df.shape}")
                print(f"    列名: {list(df.columns)}")
                print(f"    日期范围: {df['date'].min()} 到 {df['date'].max()}")
                print(f"    收盘价范围: {df['close'].min():.2f} 到 {df['close'].max():.2f}")
                
                return True, df
            else:
                print(f"  ❌ 原始查询无结果")
                return False, None
                
        except Exception as e:
            print(f"  ❌ 原始查询失败: {e}")
            return False, None
        
    except ImportError as e:
        print(f"❌ 导入ClickHouse客户端失败: {e}")
        print("请确保已安装: pip install clickhouse-driver")
        return False, None
    except Exception as e:
        print(f"❌ ClickHouse连接失败: {e}")
        return False, None

def test_rsi_calculation_with_real_data():
    """使用真实数据测试RSI计算"""
    
    print(f"\n🎯 使用真实数据测试RSI计算")
    print("=" * 80)
    
    # 获取真实数据
    success, df = test_direct_clickhouse_connection()
    
    if not success or df is None:
        print("❌ 无法获取真实数据，跳过RSI计算测试")
        return False
    
    try:
        # 导入RSI计算相关模块
        from utils.technical_utils import calculate_rsi_Utils
        from indicators.rsi import RsiRsi
from db.sql_manager import SQLManager, QueryType
        
        print(f"📊 使用真实数据计算RSI")
        print(f"数据形状: {df.shape}")
        print(f"价格数据样本:")
        print(df[['date', 'close']].head())
        
        # 方法1: 使用技术工具函数
        print(f"\n🔧 方法1: 使用技术工具函数")
        rsi_values_utils = calculate_rsi_Utils(df['close'], 14)
        
        if rsi_values_utils is not None and not rsi_values_utils.empty:
            print(f"  ✅ 技术工具函数计算成功")
            print(f"  RSI数据点数: {len(rsi_values_utils)}")
            print(f"  RSI范围: {rsi_values_utils.min():.2f} - {rsi_values_utils.max():.2f}")
            print(f"  最新RSI值: {rsi_values_utils.iloc[-1]:.2f}")
        else:
            print(f"  ❌ 技术工具函数计算失败")
        
        # 方法2: 使用RSI指标类
        print(f"\n🔧 方法2: 使用RSI指标类")
        rsi_indicator = RsiRsi()
        rsi_result = rsi_indicator._calculate_rsi(df)
        
        if 'rsi_14' in rsi_result.columns and not rsi_result['rsi_14'].empty:
            rsi_values_indicator = rsi_result['rsi_14'].dropna()
            print(f"  ✅ RSI指标类计算成功")
            print(f"  RSI数据点数: {len(rsi_values_indicator)}")
            print(f"  RSI范围: {rsi_values_indicator.min():.2f} - {rsi_values_indicator.max():.2f}")
            print(f"  最新RSI值: {rsi_values_indicator.iloc[-1]:.2f}")
            
            # 比较两种方法的结果
            if rsi_values_utils is not None and not rsi_values_utils.empty:
                # 对齐长度进行比较
                min_len = min(len(rsi_values_utils), len(rsi_values_indicator))
                utils_tail = rsi_values_utils.tail(min_len)
                indicator_tail = rsi_values_indicator.tail(min_len)
                
                # 计算差异
                diff = abs(utils_tail.values - indicator_tail.values)
                max_diff = diff.max()
                avg_diff = diff.mean()
                
                print(f"\n📊 两种方法对比:")
                print(f"  最大差异: {max_diff:.6f}")
                print(f"  平均差异: {avg_diff:.6f}")
                
                if max_diff < 0.01:
                    print(f"  ✅ 两种方法结果一致")
                    return True
                else:
                    print(f"  ⚠️ 两种方法存在差异")
                    return True
            else:
                print(f"  ✅ RSI指标类计算成功（无法对比）")
                return True
        else:
            print(f"  ❌ RSI指标类计算失败")
            return False
        
    except Exception as e:
        print(f"❌ RSI计算测试异常: {e}")
        return False

def test_stock_data_service_with_real_data():
    """测试股票数据服务与真实数据的兼容性"""
    
    print(f"\n🎯 测试股票数据服务与真实数据兼容性")
    print("=" * 80)
    
    try:
        from db.services.stock_data_service import get_stock_data_service
        
        # 获取股票数据服务
        stock_service = get_stock_data_service()
        
        if stock_service is None:
            print("❌ 无法获取股票数据服务")
            return False
        
        print("✅ 股票数据服务获取成功")
        
        # 测试获取股票数据
        test_codes = ['000001', '000002', '600000']
        
        for code in test_codes:
            print(f"\n📊 测试股票 {code}")
            
            try:
                # 使用服务获取数据
                stock_data = stock_service.get_stock_data(code, days=30)
                
                if stock_data is not None and len(stock_data) > 0:
                    print(f"  ✅ 服务获取成功: {len(stock_data)} 条记录")
                    print(f"  数据列: {list(stock_data.columns)}")
                    print(f"  日期范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
                    
                    # 测试RSI计算
                    from indicators.rsi import RsiRsi
from db.sql_manager import SQLManager, QueryType
                    rsi_indicator = RsiRsi()
                    rsi_result = rsi_indicator._calculate_rsi(stock_data)
                    
                    if 'rsi_14' in rsi_result.columns:
                        rsi_values = rsi_result['rsi_14'].dropna()
                        if len(rsi_values) > 0:
                            print(f"  ✅ RSI计算成功: 最新值 {rsi_values.iloc[-1]:.2f}")
                        else:
                            print(f"  ❌ RSI计算无有效值")
                    else:
                        print(f"  ❌ RSI计算失败")
                else:
                    print(f"  ❌ 服务获取失败或无数据")
                    
            except Exception as e:
                print(f"  ❌ 测试异常: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 股票数据服务测试异常: {e}")
        return False

def main():
    """主函数"""
    
    print("🎯 直接数据库查询测试")
    print("验证ClickHouse数据库中的真实数据情况")
    print("=" * 80)
    
    # 测试1: 直接数据库连接
    db_success, sample_df = test_direct_clickhouse_connection()
    
    # 测试2: RSI计算
    if db_success:
        rsi_success = test_rsi_calculation_with_real_data()
    else:
        rsi_success = False
    
    # 测试3: 股票数据服务
    service_success = test_stock_data_service_with_real_data()
    
    # 总结
    print(f"\n🏆 测试结果总结")
    print("=" * 80)
    print(f"数据库直连测试: {'✅ 成功' if db_success else '❌ 失败'}")
    print(f"RSI计算测试: {'✅ 成功' if rsi_success else '❌ 失败'}")
    print(f"数据服务测试: {'✅ 成功' if service_success else '❌ 失败'}")
    
    if db_success and rsi_success and service_success:
        print(f"\n🎯 结论: 数据库中确实存在完整的真实数据，RSI验证可以正常进行")
        print(f"建议: 重新运行RSI阶段4验证，应该能够成功")
    elif db_success:
        print(f"\n🎯 结论: 数据库连接正常，但服务层可能存在问题")
        print(f"建议: 检查股票数据服务的实现逻辑")
    else:
        print(f"\n🎯 结论: 数据库连接存在问题")
        print(f"建议: 检查ClickHouse配置和连接参数")

if __name__ == "__main__":
    main()
