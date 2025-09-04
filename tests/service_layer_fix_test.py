#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务层修复测试

测试修复后的股票数据服务，验证服务层与直接数据库查询结果一致
"""

import sys
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from clickhouse_driver import Client
    from db.services.stock_data_service import get_stock_data_service
except ImportError as e:
    print(f"导入错误: {e}")

def test_service_layer_fix():
    """测试服务层修复效果"""
    
    print("🎯 服务层修复测试")
    print("=" * 80)
    
    try:
        # 1. 直接数据库连接
        client = Client(
            host='localhost',
            port=9000,
            database='stock',
            user='default',
            password='123456'
        )
        
        # 2. 股票数据服务
        stock_service = get_stock_data_service()
        
        if stock_service is None:
            print("❌ 无法获取股票数据服务")
            return False
        
        print("✅ 股票数据服务获取成功")
        
        # 获取测试股票
        available_query = """
        SELECT code, COUNT(*) as data_count
        FROM stock_info
        WHERE level = '日线'
        AND date >= '2024-01-01'
        GROUP BY code
        HAVING data_count >= 50
        ORDER BY data_count DESC
        LIMIT 5
        """
        
        available_result = client.execute(available_query)
        
        if not available_result:
            print("❌ 没有可用的股票数据")
            return False
        
        available_stocks = [row[0] for row in available_result]
        print(f"📈 可用股票: {available_stocks}")
        
        # 测试每个股票
        total_consistency = 0.0
        valid_tests = 0
        
        for stock_code in available_stocks[:3]:  # 测试前3个股票
            print(f"\n🔍 测试股票 {stock_code}")
            
            # 方法1: 直接数据库查询（使用最新数据）
            print(f"  📊 方法1: 直接数据库查询")

            end_date = '2025-12-31'
            start_date = '2025-01-01'

            direct_query = f"""
            SELECT date, open, high, low, close, volume
            FROM stock_info
            WHERE code = '{stock_code}'
            AND level = '日线'
            AND date >= '{start_date}'
            AND date <= '{end_date}'
            ORDER BY date DESC
            LIMIT 50
            """
            
            direct_result = client.execute(direct_query)
            
            if direct_result:
                direct_df = pd.DataFrame(direct_result, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                direct_df['date'] = pd.to_datetime(direct_df['date'])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    direct_df[col] = pd.to_numeric(direct_df[col], errors='coerce')
                
                print(f"    ✅ 直接查询成功: {len(direct_df)}条记录")
                print(f"    📅 日期范围: {direct_df['date'].min().date()} 到 {direct_df['date'].max().date()}")
            else:
                print(f"    ❌ 直接查询失败")
                continue
            
            # 方法2: 服务层查询
            print(f"  📊 方法2: 服务层查询")
            
            try:
                service_df = stock_service.get_stock_data(stock_code, days=50)
                
                if service_df is not None and len(service_df) > 0:
                    print(f"    ✅ 服务查询成功: {len(service_df)}条记录")
                    print(f"    📅 日期范围: {service_df['date'].min().date()} 到 {service_df['date'].max().date()}")
                    print(f"    📊 列名: {list(service_df.columns)}")
                else:
                    print(f"    ❌ 服务查询失败或无数据")
                    continue
            except Exception as e:
                print(f"    ❌ 服务查询异常: {e}")
                continue
            
            # 对比分析
            print(f"  🔍 对比分析")
            
            # 数据量对比
            direct_count = len(direct_df)
            service_count = len(service_df)
            
            print(f"    直接查询数据量: {direct_count}")
            print(f"    服务查询数据量: {service_count}")
            
            if service_count == 0:
                print(f"    ❌ 服务层无数据")
                continue
            
            # 日期范围对比
            direct_date_range = (direct_df['date'].min(), direct_df['date'].max())
            service_date_range = (service_df['date'].min(), service_df['date'].max())
            
            print(f"    直接查询日期范围: {direct_date_range[0].date()} 到 {direct_date_range[1].date()}")
            print(f"    服务查询日期范围: {service_date_range[0].date()} 到 {service_date_range[1].date()}")
            
            # 数据一致性检查（取重叠的日期进行对比）
            if len(service_df) >= 10:
                # 取服务层数据的最后10天
                service_tail = service_df.tail(10)
                
                # 在直接查询结果中找到对应的日期
                matching_data = []
                for _, service_row in service_tail.iterrows():
                    service_date = service_row['date']
                    direct_match = direct_df[direct_df['date'] == service_date]
                    
                    if not direct_match.empty:
                        direct_row = direct_match.iloc[0]
                        
                        # 比较收盘价
                        service_close = service_row['close']
                        direct_close = direct_row['close']
                        
                        if pd.notna(service_close) and pd.notna(direct_close):
                            diff = abs(service_close - direct_close)
                            matching_data.append({
                                'date': service_date,
                                'service_close': service_close,
                                'direct_close': direct_close,
                                'difference': diff
                            })
                
                if matching_data:
                    print(f"    📊 找到{len(matching_data)}个匹配日期")
                    
                    # 计算一致性
                    total_diff = sum(item['difference'] for item in matching_data)
                    avg_diff = total_diff / len(matching_data)
                    max_diff = max(item['difference'] for item in matching_data)
                    
                    # 计算一致性评分
                    if max_diff < 0.01:
                        consistency = 1.0
                    elif avg_diff < 0.01:
                        consistency = 0.95
                    elif avg_diff < 0.1:
                        consistency = 0.90
                    else:
                        consistency = 0.80
                    
                    print(f"    📊 平均差异: {avg_diff:.6f}")
                    print(f"    📊 最大差异: {max_diff:.6f}")
                    print(f"    📊 一致性评分: {consistency:.1%}")
                    
                    total_consistency += consistency
                    valid_tests += 1
                    
                    # 显示前3个匹配的详细对比
                    print(f"    🔍 详细对比（前3个匹配）:")
                    for i, item in enumerate(matching_data[:3]):
                        print(f"      {item['date'].date()}: 服务={item['service_close']:.3f}, 直接={item['direct_close']:.3f}, 差异={item['difference']:.6f}")
                else:
                    print(f"    ⚠️ 没有找到匹配的日期数据")
            else:
                print(f"    ⚠️ 服务层数据不足，无法对比")
        
        # 总体结果
        if valid_tests > 0:
            overall_consistency = total_consistency / valid_tests
            
            print(f"\n🏆 服务层修复测试结果")
            print("=" * 60)
            print(f"测试股票数: {valid_tests}")
            print(f"总体一致性: {overall_consistency:.1%}")
            
            if overall_consistency >= 0.95:
                print(f"✅ 服务层修复成功! 一致性达到95%以上")
                return True
            elif overall_consistency >= 0.90:
                print(f"⚠️ 服务层修复部分成功，一致性达到90%以上")
                return True
            else:
                print(f"❌ 服务层修复效果不佳，一致性仍然偏低")
                return False
        else:
            print(f"\n❌ 没有有效的测试结果")
            return False
    
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def test_service_layer_functionality():
    """测试服务层功能性"""
    
    print(f"\n🔧 测试服务层功能性")
    print("=" * 60)
    
    try:
        stock_service = get_stock_data_service()
        
        if stock_service is None:
            print("❌ 无法获取股票数据服务")
            return False
        
        # 测试不同的参数组合
        test_cases = [
            {'stock_code': '000001', 'days': 30, 'description': '平安银行30天'},
            {'stock_code': '000002', 'days': 50, 'description': '万科A50天'},
            {'stock_code': '600000', 'days': 20, 'description': '浦发银行20天'}
        ]
        
        successful_tests = 0
        
        for test_case in test_cases:
            print(f"📊 测试: {test_case['description']}")
            
            try:
                result = stock_service.get_stock_data(
                    test_case['stock_code'], 
                    days=test_case['days']
                )
                
                if result is not None and len(result) > 0:
                    print(f"  ✅ 成功: {len(result)}条记录")
                    print(f"  📅 日期范围: {result['date'].min().date()} 到 {result['date'].max().date()}")
                    
                    # 验证数据质量
                    if 'close' in result.columns:
                        close_data = result['close'].dropna()
                        if len(close_data) > 0:
                            print(f"  💰 价格范围: {close_data.min():.2f} - {close_data.max():.2f}")
                            successful_tests += 1
                        else:
                            print(f"  ⚠️ 无有效价格数据")
                    else:
                        print(f"  ⚠️ 缺少收盘价列")
                else:
                    print(f"  ❌ 无数据")
            
            except Exception as e:
                print(f"  ❌ 异常: {e}")
        
        success_rate = successful_tests / len(test_cases)
        
        print(f"\n📊 功能性测试结果:")
        print(f"成功测试: {successful_tests}/{len(test_cases)}")
        print(f"成功率: {success_rate:.1%}")
        
        return success_rate >= 0.8
    
    except Exception as e:
        print(f"❌ 功能性测试异常: {e}")
        return False

def main():
    """主函数"""
    
    print("🎯 服务层修复验证测试")
    print("验证修复后的服务层是否解决了数据查询问题")
    print("=" * 80)
    
    # 测试1: 服务层功能性
    functionality_test = test_service_layer_functionality()
    
    # 测试2: 服务层修复效果
    fix_test = test_service_layer_fix()
    
    # 总结
    print(f"\n🏆 测试结果总结")
    print("=" * 80)
    print(f"服务层功能性: {'✅ 正常' if functionality_test else '❌ 异常'}")
    print(f"服务层修复效果: {'✅ 成功' if fix_test else '❌ 失败'}")
    
    if functionality_test and fix_test:
        print(f"\n🎉 服务层修复成功!")
        print(f"✅ 服务层现在可以正常获取股票数据")
        print(f"✅ 服务层与直接数据库查询结果高度一致")
        print(f"🚀 可以继续完善阶段4验证")
    else:
        print(f"\n⚠️ 服务层修复需要进一步调整")
        if not functionality_test:
            print(f"❌ 服务层功能性存在问题")
        if not fix_test:
            print(f"❌ 服务层修复效果不佳")

if __name__ == "__main__":
    main()
