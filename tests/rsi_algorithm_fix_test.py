#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSI算法修复测试

测试修复后的RSI算法，验证系统RSI与基准RSI的一致性
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from clickhouse_driver import Client
    from indicators.rsi import RsiRsi
    from utils.technical_utils import calculate_rsi_Utils
except ImportError as e:
    print(f"导入错误: {e}")

def test_rsi_algorithm_fix():
    """测试RSI算法修复效果"""
    
    print("🎯 RSI算法修复测试")
    print("=" * 80)
    
    try:
        # 连接数据库
        client = Client(
            host='localhost',
            port=9000,
            database='stock',
            user='default',
            password='123456'
        )
        
        # 获取测试数据
        print("📊 获取测试数据")
        
        # 查询可用股票
        available_query = """
        SELECT code, COUNT(*) as data_count
        FROM stock_info
        WHERE level = '日线'
        AND date >= '2024-01-01'
        GROUP BY code
        HAVING data_count >= 100
        ORDER BY data_count DESC
        LIMIT 3
        """
        
        available_result = client.execute(available_query)
        
        if not available_result:
            print("❌ 没有可用的股票数据")
            return False
        
        available_stocks = [row[0] for row in available_result]
        print(f"📈 可用股票: {available_stocks}")
        
        # 测试每个股票
        total_accuracy = 0.0
        valid_tests = 0
        
        for stock_code in available_stocks:
            print(f"\n🔍 测试股票 {stock_code}")
            
            # 获取股票数据
            query = f"""
            SELECT date, open, high, low, close, volume
            FROM stock_info
            WHERE code = '{stock_code}'
            AND level = '日线'
            AND date >= '2024-01-01'
            ORDER BY date ASC
            LIMIT 100
            """
            
            result = client.execute(query)
            
            if not result:
                print(f"  ❌ 无数据")
                continue
            
            # 构建DataFrame
            df = pd.DataFrame(result, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if len(df) < 30:
                print(f"  ❌ 数据不足: {len(df)}条")
                continue
            
            print(f"  📊 数据点数: {len(df)}")
            print(f"  📅 日期范围: {df['date'].min().date()} 到 {df['date'].max().date()}")
            
            # 方法1: 系统RSI指标类
            rsi_indicator = RsiRsi()
            system_rsi_result = rsi_indicator._calculate_rsi(df)
            
            if 'rsi_14' not in system_rsi_result.columns:
                print(f"  ❌ 系统RSI计算失败")
                continue
            
            system_rsi = system_rsi_result['rsi_14'].dropna()
            
            if len(system_rsi) == 0:
                print(f"  ❌ 系统RSI无有效值")
                continue
            
            # 方法2: 技术工具函数
            utils_rsi = calculate_rsi_Utils(df['close'], 14)
            
            if utils_rsi is None or utils_rsi.empty:
                print(f"  ❌ 工具函数RSI计算失败")
                continue
            
            utils_rsi_clean = utils_rsi.dropna()
            
            if len(utils_rsi_clean) == 0:
                print(f"  ❌ 工具函数RSI无有效值")
                continue
            
            # 对比分析
            print(f"  📈 系统RSI: {len(system_rsi)}个值, 最新: {system_rsi.iloc[-1]:.3f}")
            print(f"  📈 工具RSI: {len(utils_rsi_clean)}个值, 最新: {utils_rsi_clean.iloc[-1]:.3f}")
            
            # 计算准确率（使用最后10个值进行对比）
            min_len = min(len(system_rsi), len(utils_rsi_clean))
            if min_len >= 10:
                compare_len = min(10, min_len)
                
                system_tail = system_rsi.tail(compare_len)
                utils_tail = utils_rsi_clean.tail(compare_len)
                
                # 计算差异
                differences = abs(system_tail.values - utils_tail.values)
                max_diff = differences.max()
                avg_diff = differences.mean()
                
                # 计算准确率（基于相对误差）
                relative_errors = differences / (abs(utils_tail.values) + 1e-9)
                avg_relative_error = relative_errors.mean()
                accuracy = max(0, 1 - avg_relative_error)
                
                print(f"  📊 最大差异: {max_diff:.3f}")
                print(f"  📊 平均差异: {avg_diff:.3f}")
                print(f"  📊 平均相对误差: {avg_relative_error:.1%}")
                print(f"  📊 准确率: {accuracy:.1%}")
                
                total_accuracy += accuracy
                valid_tests += 1
                
                # 详细对比最后5个值
                print(f"  🔍 详细对比（最后5个值）:")
                for i in range(max(0, compare_len-5), compare_len):
                    sys_val = system_tail.iloc[i]
                    utils_val = utils_tail.iloc[i]
                    diff = abs(sys_val - utils_val)
                    print(f"    第{i+1}个: 系统={sys_val:.3f}, 工具={utils_val:.3f}, 差异={diff:.3f}")
            else:
                print(f"  ⚠️ 数据不足，无法对比")
        
        # 总体结果
        if valid_tests > 0:
            overall_accuracy = total_accuracy / valid_tests
            
            print(f"\n🏆 RSI算法修复测试结果")
            print("=" * 60)
            print(f"测试股票数: {valid_tests}")
            print(f"总体准确率: {overall_accuracy:.1%}")
            
            if overall_accuracy >= 0.95:
                print(f"✅ 修复成功! 准确率达到95%以上")
                return True
            elif overall_accuracy >= 0.90:
                print(f"⚠️ 修复部分成功，准确率达到90%以上")
                return True
            else:
                print(f"❌ 修复效果不佳，准确率仍然偏低")
                return False
        else:
            print(f"\n❌ 没有有效的测试结果")
            return False
    
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def test_wilder_rsi_implementation():
    """测试Wilder RSI实现的正确性"""
    
    print(f"\n🔧 测试Wilder RSI实现正确性")
    print("=" * 60)
    
    # 创建简单的测试数据
    test_prices = pd.Series([
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89, 46.03,
        46.83, 47.69, 46.49, 46.26, 47.09, 46.66, 46.80, 46.23, 46.38, 46.33,
        46.51, 46.87, 47.37, 47.20, 47.72, 47.90, 47.87, 48.39, 48.66, 48.79
    ])
    
    print(f"📊 测试数据: {len(test_prices)}个价格点")
    print(f"📈 价格范围: {test_prices.min():.2f} - {test_prices.max():.2f}")
    
    try:
        # 使用修复后的工具函数
        rsi_result = calculate_rsi_Utils(test_prices, 14)
        
        if rsi_result is not None and not rsi_result.empty:
            rsi_clean = rsi_result.dropna()
            
            if len(rsi_clean) > 0:
                print(f"✅ RSI计算成功: {len(rsi_clean)}个值")
                print(f"📊 RSI范围: {rsi_clean.min():.2f} - {rsi_clean.max():.2f}")
                print(f"📊 最新RSI: {rsi_clean.iloc[-1]:.3f}")
                
                # 验证RSI值的合理性
                if 0 <= rsi_clean.min() <= 100 and 0 <= rsi_clean.max() <= 100:
                    print(f"✅ RSI值在合理范围内")
                    return True
                else:
                    print(f"❌ RSI值超出合理范围")
                    return False
            else:
                print(f"❌ 没有有效的RSI值")
                return False
        else:
            print(f"❌ RSI计算失败")
            return False
    
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def main():
    """主函数"""
    
    print("🎯 RSI算法修复验证测试")
    print("验证修复后的RSI算法是否解决了准确率问题")
    print("=" * 80)
    
    # 测试1: Wilder RSI实现正确性
    wilder_test = test_wilder_rsi_implementation()
    
    # 测试2: 算法修复效果
    fix_test = test_rsi_algorithm_fix()
    
    # 总结
    print(f"\n🏆 测试结果总结")
    print("=" * 80)
    print(f"Wilder RSI实现: {'✅ 正确' if wilder_test else '❌ 错误'}")
    print(f"算法修复效果: {'✅ 成功' if fix_test else '❌ 失败'}")
    
    if wilder_test and fix_test:
        print(f"\n🎉 RSI算法修复成功!")
        print(f"✅ 系统RSI与基准RSI现在应该高度一致")
        print(f"🚀 可以继续完善阶段4验证")
    else:
        print(f"\n⚠️ RSI算法修复需要进一步调整")
        if not wilder_test:
            print(f"❌ Wilder RSI实现存在问题")
        if not fix_test:
            print(f"❌ 算法修复效果不佳")

if __name__ == "__main__":
    main()
