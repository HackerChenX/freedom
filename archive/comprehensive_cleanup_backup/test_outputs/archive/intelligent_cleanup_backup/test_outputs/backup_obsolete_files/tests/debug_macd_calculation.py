#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试MACD计算结果

检查MACD指标计算的输出结构，找出形态检测失败的原因
"""

import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

def debug_macd_calculation():
    """调试MACD计算"""
    
    print("🔍 调试MACD计算结果")
    print("=" * 60)
    
    try:
        # 初始化
        macd_indicator = MacdMacd()
        stock_data_service = get_stock_data_service()
        
        # 获取一支股票的数据
        stock_codes = stock_data_service.get_stock_list(limit=5)
        
        if not stock_codes:
            print("❌ 无法获取股票列表")
            return
        
        stock_code = stock_codes[0]
        print(f"📊 测试股票: {stock_code}")
        
        # 获取股票数据
        df = stock_data_service.get_stock_data(stock_code, days=120)
        
        if df is None or len(df) < 60:
            print("❌ 股票数据不足")
            return
        
        print(f"✅ 获取到{len(df)}天的股票数据")
        print(f"📋 股票数据列: {list(df.columns)}")
        print(f"📅 数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
        
        # 显示最近几天的价格数据
        print(f"\n📈 最近5天的价格数据:")
        recent_price = df.tail(5)[['date', 'open', 'high', 'low', 'close', 'volume']]
        print(recent_price.to_string(index=False))
        
        # 计算MACD
        print(f"\n🔄 计算MACD指标...")
        macd_result = macd_indicator.calculate(df)
        
        if macd_result is None:
            print("❌ MACD计算返回None")
            return
        
        if macd_result.empty:
            print("❌ MACD计算返回空DataFrame")
            return
        
        print(f"✅ MACD计算成功")
        print(f"📋 MACD结果列: {list(macd_result.columns)}")
        print(f"📊 MACD结果行数: {len(macd_result)}")
        
        # 显示最近几天的MACD数据
        print(f"\n📊 最近10天的MACD数据:")
        if len(macd_result) >= 10:
            recent_macd = macd_result.tail(10)
            
            # 检查关键列是否存在
            key_columns = ['macd_line', 'signal_line', 'histogram']
            available_columns = [col for col in key_columns if col in macd_result.columns]
            
            if available_columns:
                display_columns = available_columns
                if 'date' in macd_result.columns:
                    display_columns = ['date'] + display_columns
                
                print(recent_macd[display_columns].to_string(index=False))
                
                # 分析MACD数据特征
                print(f"\n📈 MACD数据分析:")
                for col in available_columns:
                    values = recent_macd[col].values
                    print(f"  {col}:")
                    print(f"    范围: {values.min():.6f} 到 {values.max():.6f}")
                    print(f"    最新值: {values[-1]:.6f}")
                    print(f"    变化趋势: {'上升' if values[-1] > values[-2] else '下降'}")
                
                # 检测金叉死叉
                if 'macd_line' in macd_result.columns and 'signal_line' in macd_result.columns:
                    print(f"\n🔍 金叉死叉检测:")
                    macd_values = recent_macd['macd_line'].values
                    signal_values = recent_macd['signal_line'].values
                    
                    for i in range(1, len(macd_values)):
                        prev_diff = macd_values[i-1] - signal_values[i-1]
                        curr_diff = macd_values[i] - signal_values[i]
                        
                        if prev_diff <= 0 and curr_diff > 0:
                            print(f"    🟢 第{i}天发现金叉: MACD({macd_values[i]:.6f}) > Signal({signal_values[i]:.6f})")
                        elif prev_diff >= 0 and curr_diff < 0:
                            print(f"    🔴 第{i}天发现死叉: MACD({macd_values[i]:.6f}) < Signal({signal_values[i]:.6f})")
                    
                    # 检查零轴位置
                    print(f"\n📊 零轴位置分析:")
                    above_zero_macd = np.sum(macd_values > 0)
                    above_zero_signal = np.sum(signal_values > 0)
                    print(f"    MACD线在零轴上方的天数: {above_zero_macd}/{len(macd_values)}")
                    print(f"    信号线在零轴上方的天数: {above_zero_signal}/{len(signal_values)}")
                    
                    if above_zero_macd > 0 and above_zero_signal > 0:
                        print(f"    ✅ 存在零轴上方的数据，可能有零轴上金叉")
                    else:
                        print(f"    ⚠️ 数据主要在零轴下方")
            else:
                print("❌ 缺少关键的MACD列")
                print(f"可用列: {list(macd_result.columns)}")
        else:
            print("❌ MACD数据不足10天")
        
        # 检查数据类型
        print(f"\n🔧 数据类型检查:")
        for col in macd_result.columns:
            dtype = macd_result[col].dtype
            null_count = macd_result[col].isnull().sum()
            print(f"  {col}: {dtype}, 空值: {null_count}")
        
    except Exception as e:
        print(f"❌ 调试过程异常: {e}")
        import traceback
        traceback.print_exc()

def test_pattern_detection_logic():
    """测试形态检测逻辑"""
    
    print(f"\n🧪 测试形态检测逻辑")
    print("=" * 60)
    
    # 创建模拟的MACD数据来测试检测逻辑
    dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
    
    # 模拟一个金叉的情况
    macd_line = [-0.1, -0.08, -0.06, -0.04, -0.02, 0.01, 0.03, 0.05, 0.04, 0.03,
                 0.02, 0.01, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08]
    
    signal_line = [-0.05, -0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04,
                   0.03, 0.02, 0.01, 0.00, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06]
    
    test_macd_df = pd.DataFrame({
        'date': dates,
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': [m - s for m, s in zip(macd_line, signal_line)]
    })
    
    print("📊 模拟MACD数据:")
    print(test_macd_df[['date', 'macd_line', 'signal_line', 'histogram']].to_string(index=False))
    
    # 测试金叉检测
    print(f"\n🔍 测试金叉检测:")
    macd_values = test_macd_df['macd_line'].values
    signal_values = test_macd_df['signal_line'].values
    
    golden_crosses = []
    death_crosses = []
    
    for i in range(1, len(macd_values)):
        prev_diff = macd_values[i-1] - signal_values[i-1]
        curr_diff = macd_values[i] - signal_values[i]
        
        if prev_diff <= 0 and curr_diff > 0:
            golden_crosses.append(i)
            print(f"  🟢 第{i}天金叉: MACD({macd_values[i]:.3f}) > Signal({signal_values[i]:.3f})")
        elif prev_diff >= 0 and curr_diff < 0:
            death_crosses.append(i)
            print(f"  🔴 第{i}天死叉: MACD({macd_values[i]:.3f}) < Signal({signal_values[i]:.3f})")
    
    print(f"\n📊 检测结果:")
    print(f"  金叉次数: {len(golden_crosses)}")
    print(f"  死叉次数: {len(death_crosses)}")
    
    if len(golden_crosses) > 0:
        print(f"  ✅ 金叉检测逻辑正常工作")
    else:
        print(f"  ❌ 金叉检测逻辑可能有问题")

def main():
    """主函数"""
    debug_macd_calculation()
    test_pattern_detection_logic()

if __name__ == "__main__":
    main()
