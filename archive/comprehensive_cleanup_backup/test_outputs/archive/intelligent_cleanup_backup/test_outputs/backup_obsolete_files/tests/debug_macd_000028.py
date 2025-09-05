#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试000028股票MACD计算问题

用户反馈：
- 真实2025年5月12日数据：MACD=0.220, DIFF=-0.136, DEA=-0.246
- 系统计算结果与真实数据不符

需要检查：
1. 数据获取是否正确
2. MACD计算公式是否正确
3. 参数设置是否正确
4. 日期匹配是否正确
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

def debug_macd_000028():
    """调试000028股票的MACD计算"""
    
    print("🔍 调试000028股票MACD计算问题")
    print("=" * 80)
    print("📋 用户反馈的真实数据（2025年5月12日）:")
    print("  MACD = 0.220")
    print("  DIFF = -0.136") 
    print("  DEA = -0.246")
    print("=" * 80)
    
    try:
        # 初始化服务
        macd_indicator = MacdMacd()
        stock_data_service = get_stock_data_service()
        
        stock_code = "000028"
        target_date = "2025-05-12"
        
        print(f"\n📊 步骤1: 获取{stock_code}股票数据")
        
        # 获取股票数据
        df = stock_data_service.get_stock_data(stock_code, days=200)  # 获取更多历史数据
        
        if df is None or len(df) == 0:
            print(f"❌ 无法获取{stock_code}股票数据")
            return
        
        print(f"✅ 获取到{len(df)}天的数据")
        print(f"📅 数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
        
        # 检查目标日期是否存在
        target_date_obj = pd.to_datetime(target_date).date()
        target_rows = df[df['date'].dt.date == target_date_obj]
        
        if target_rows.empty:
            print(f"⚠️ 目标日期{target_date}不在数据范围内")
            print(f"📅 最接近的日期:")
            
            # 找最接近的日期
            df['date_diff'] = abs((df['date'].dt.date - target_date_obj).dt.days)
            closest_date = df.loc[df['date_diff'].idxmin()]
            print(f"  最接近日期: {closest_date['date'].strftime('%Y-%m-%d')}")
            print(f"  收盘价: {closest_date['close']:.3f}")
            
            # 使用最接近的日期
            target_date_obj = closest_date['date'].date()
            target_rows = df[df['date'].dt.date == target_date_obj]
        
        print(f"\n📈 目标日期{target_date_obj}的价格数据:")
        target_data = target_rows.iloc[0]
        print(f"  开盘价: {target_data['open']:.3f}")
        print(f"  最高价: {target_data['high']:.3f}")
        print(f"  最低价: {target_data['low']:.3f}")
        print(f"  收盘价: {target_data['close']:.3f}")
        print(f"  成交量: {target_data['volume']:,.0f}")
        
        print(f"\n🔄 步骤2: 计算MACD指标")
        
        # 显示MACD参数
        print(f"📋 MACD参数设置:")
        if hasattr(macd_indicator, '_parameters'):
            params = macd_indicator._parameters
            print(f"  快线周期: {params.get('fast_period', 12)}")
            print(f"  慢线周期: {params.get('slow_period', 26)}")
            print(f"  信号线周期: {params.get('signal_period', 9)}")
        
        # 计算MACD
        macd_result = macd_indicator.calculate(df)
        
        if macd_result is None or macd_result.empty:
            print("❌ MACD计算失败")
            return
        
        print(f"✅ MACD计算成功，共{len(macd_result)}条结果")
        print(f"📋 MACD结果列: {list(macd_result.columns)}")
        
        # 查找目标日期的MACD结果
        target_idx = target_rows.index[0]
        
        if target_idx < len(macd_result):
            macd_data = macd_result.iloc[target_idx]
            
            print(f"\n📊 系统计算的MACD结果（{target_date_obj}）:")
            
            # 显示所有MACD相关列
            for col in macd_result.columns:
                if 'macd' in col.lower() or 'diff' in col.lower() or 'dea' in col.lower() or 'signal' in col.lower():
                    value = macd_data[col]
                    print(f"  {col}: {value:.6f}")
            
            # 对比真实数据
            print(f"\n📋 数据对比:")
            print(f"{'指标':<15} {'真实值':<12} {'系统值':<12} {'差异':<12}")
            print("-" * 55)
            
            # 尝试匹配列名
            macd_col = None
            diff_col = None
            dea_col = None
            
            for col in macd_result.columns:
                if 'macd_line' in col.lower() or col.lower() == 'macd':
                    macd_col = col
                elif 'diff' in col.lower():
                    diff_col = col
                elif 'dea' in col.lower() or 'signal' in col.lower():
                    dea_col = col
            
            # MACD对比
            if macd_col:
                system_macd = macd_data[macd_col]
                real_macd = 0.220
                diff = abs(system_macd - real_macd)
                print(f"{'MACD':<15} {real_macd:<12.6f} {system_macd:<12.6f} {diff:<12.6f}")
            
            # DIFF对比
            if diff_col:
                system_diff = macd_data[diff_col]
                real_diff = -0.136
                diff_val = abs(system_diff - real_diff)
                print(f"{'DIFF':<15} {real_diff:<12.6f} {system_diff:<12.6f} {diff_val:<12.6f}")
            
            # DEA对比
            if dea_col:
                system_dea = macd_data[dea_col]
                real_dea = -0.246
                diff_val = abs(system_dea - real_dea)
                print(f"{'DEA':<15} {real_dea:<12.6f} {system_dea:<12.6f} {diff_val:<12.6f}")
            
            # 分析差异原因
            print(f"\n🔍 差异分析:")
            
            # 检查计算公式
            print(f"📐 MACD计算公式检查:")
            print(f"  标准公式: MACD = DIFF - DEA")
            
            if diff_col and dea_col:
                calculated_macd = macd_data[diff_col] - macd_data[dea_col]
                print(f"  计算验证: {macd_data[diff_col]:.6f} - {macd_data[dea_col]:.6f} = {calculated_macd:.6f}")
                
                if macd_col:
                    system_macd = macd_data[macd_col]
                    formula_diff = abs(calculated_macd - system_macd)
                    print(f"  公式一致性: {formula_diff:.6f} (应该接近0)")
            
            # 显示最近几天的数据用于趋势分析
            print(f"\n📈 最近5天的MACD趋势:")
            recent_start = max(0, target_idx - 4)
            recent_end = min(len(macd_result), target_idx + 1)
            
            recent_macd = macd_result.iloc[recent_start:recent_end]
            recent_price = df.iloc[recent_start:recent_end]
            
            print(f"{'日期':<12} {'收盘价':<10} {'DIFF':<12} {'DEA':<12} {'MACD':<12}")
            print("-" * 70)
            
            for i in range(len(recent_macd)):
                date_str = recent_price.iloc[i]['date'].strftime('%Y-%m-%d')
                close_price = recent_price.iloc[i]['close']
                
                diff_val = recent_macd.iloc[i][diff_col] if diff_col else 0
                dea_val = recent_macd.iloc[i][dea_col] if dea_col else 0
                macd_val = recent_macd.iloc[i][macd_col] if macd_col else 0
                
                marker = " ← 目标" if i == len(recent_macd) - 1 else ""
                print(f"{date_str:<12} {close_price:<10.3f} {diff_val:<12.6f} {dea_val:<12.6f} {macd_val:<12.6f}{marker}")
        
        else:
            print(f"❌ 目标日期索引{target_idx}超出MACD结果范围")
    
    except Exception as e:
        print(f"❌ 调试过程异常: {e}")
        import traceback
        traceback.print_exc()

def manual_macd_calculation():
    """手动计算MACD验证"""
    
    print(f"\n🧮 手动MACD计算验证")
    print("=" * 60)
    
    try:
        stock_data_service = get_stock_data_service()
        df = stock_data_service.get_stock_data("000028", days=200)
        
        if df is None or len(df) == 0:
            print("❌ 无法获取数据进行手动计算")
            return
        
        # 手动计算EMA
        def calculate_ema(prices, period):
            """计算指数移动平均线"""
            ema = np.zeros(len(prices))
            multiplier = 2 / (period + 1)
            
            # 第一个值使用SMA
            ema[0] = prices[0]
            
            for i in range(1, len(prices)):
                ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
            
            return ema
        
        # 获取收盘价
        close_prices = df['close'].values
        
        # 计算EMA12和EMA26
        ema12 = calculate_ema(close_prices, 12)
        ema26 = calculate_ema(close_prices, 26)
        
        # 计算DIFF (MACD线)
        diff = ema12 - ema26
        
        # 计算DEA (信号线) - DIFF的9日EMA
        dea = calculate_ema(diff, 9)
        
        # 计算MACD柱状图
        macd_histogram = diff - dea
        
        # 找到目标日期
        target_date = pd.to_datetime("2025-05-12").date()
        target_rows = df[df['date'].dt.date == target_date]
        
        if not target_rows.empty:
            target_idx = target_rows.index[0]
            
            print(f"📊 手动计算结果（索引{target_idx}）:")
            print(f"  EMA12: {ema12[target_idx]:.6f}")
            print(f"  EMA26: {ema26[target_idx]:.6f}")
            print(f"  DIFF: {diff[target_idx]:.6f}")
            print(f"  DEA: {dea[target_idx]:.6f}")
            print(f"  MACD: {macd_histogram[target_idx]:.6f}")
            
            print(f"\n📋 与真实数据对比:")
            print(f"  DIFF: 真实(-0.136) vs 手动({diff[target_idx]:.6f}) = 差异{abs(-0.136 - diff[target_idx]):.6f}")
            print(f"  DEA:  真实(-0.246) vs 手动({dea[target_idx]:.6f}) = 差异{abs(-0.246 - dea[target_idx]):.6f}")
            print(f"  MACD: 真实(0.220) vs 手动({macd_histogram[target_idx]:.6f}) = 差异{abs(0.220 - macd_histogram[target_idx]):.6f}")
        
    except Exception as e:
        print(f"❌ 手动计算异常: {e}")

def main():
    """主函数"""
    debug_macd_000028()
    manual_macd_calculation()
    
    print(f"\n💡 问题分析建议:")
    print(f"1. 检查MACD指标的计算公式是否正确")
    print(f"2. 验证EMA计算的准确性")
    print(f"3. 确认参数设置（12, 26, 9）是否正确")
    print(f"4. 检查数据源的一致性")
    print(f"5. 验证日期匹配的准确性")

if __name__ == "__main__":
    main()
