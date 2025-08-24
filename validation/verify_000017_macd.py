#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证000017股票MACD计算

用户反馈：
- 真实数据: MACD=0.037, DIFF=0.149, DEA=0.13
- 系统计算: DIFF=0.126218, DEA=0.130343, MACD=-0.008249

需要验证：
1. 日期是否正确匹配
2. MACD计算是否有误
3. 数据源是否一致
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

def verify_000017_macd():
    """验证000017股票MACD计算"""
    
    print("🔍 验证000017股票MACD计算")
    print("=" * 80)
    
    # 用户提供的真实数据
    real_data = {
        'date': '2025-05-14',  # 系统检测的日期
        'MACD': 0.037,
        'DIFF': 0.149,
        'DEA': 0.13
    }
    
    print(f"📊 用户提供的真实数据 (2025-05-14):")
    print(f"  DIFF: {real_data['DIFF']:.6f}")
    print(f"  DEA:  {real_data['DEA']:.6f}")
    print(f"  MACD: {real_data['MACD']:.6f}")
    
    try:
        macd_indicator = MacdMacd()
        stock_data_service = get_stock_data_service()
        
        # 获取000017股票数据
        print(f"\n📈 获取000017股票数据...")
        df = stock_data_service.get_stock_data('000017', days=200)
        
        if df is None or len(df) == 0:
            print(f"❌ 无法获取000017数据")
            return
        
        print(f"✅ 获取到{len(df)}天数据")
        print(f"📅 数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
        
        # 显示2025-05-14前后的价格数据
        target_date = pd.to_datetime('2025-05-14').date()
        
        # 查找目标日期前后的数据
        nearby_data = df[
            (df['date'].dt.date >= target_date - timedelta(days=3)) &
            (df['date'].dt.date <= target_date + timedelta(days=3))
        ].copy()
        
        print(f"\n📊 2025-05-14前后的价格数据:")
        if not nearby_data.empty:
            for _, row in nearby_data.iterrows():
                marker = " ← 目标日期" if row['date'].date() == target_date else ""
                print(f"  {row['date'].strftime('%Y-%m-%d')}: 开{row['open']:.2f} 高{row['high']:.2f} 低{row['low']:.2f} 收{row['close']:.2f}{marker}")
        else:
            print(f"  ❌ 未找到2025-05-14前后的数据")
            return
        
        # 查找精确日期
        target_rows = df[df['date'].dt.date == target_date]
        
        if target_rows.empty:
            print(f"❌ 未找到2025-05-14的精确数据")
            return
        
        target_idx = target_rows.index[0]
        price_data = df.iloc[target_idx]
        
        print(f"\n📊 2025-05-14价格数据:")
        print(f"  开盘: {price_data['open']:.3f}")
        print(f"  收盘: {price_data['close']:.3f}")
        print(f"  最高: {price_data['high']:.3f}")
        print(f"  最低: {price_data['low']:.3f}")
        print(f"  成交量: {price_data['volume']:,.0f}")
        
        # 计算MACD
        print(f"\n🔄 计算MACD指标...")
        macd_result = macd_indicator.calculate(df)
        
        if macd_result is None or macd_result.empty:
            print("❌ MACD计算失败")
            return
        
        if target_idx >= len(macd_result):
            print(f"❌ 目标索引{target_idx}超出MACD结果范围{len(macd_result)}")
            return
        
        # 获取系统计算结果
        system_data = macd_result.iloc[target_idx]
        
        print(f"\n📊 系统计算结果 (2025-05-14):")
        print(f"  DIFF: {system_data['macd_line']:.6f}")
        print(f"  DEA:  {system_data['macd_signal']:.6f}")
        print(f"  MACD: {system_data['macd_histogram']:.6f}")
        
        # 详细对比分析
        print(f"\n📋 详细对比分析:")
        print(f"{'指标':<8} {'真实值':<12} {'系统值':<12} {'绝对差异':<12} {'相对差异':<12} {'状态':<8}")
        print("-" * 75)
        
        diff_abs = abs(real_data['DIFF'] - system_data['macd_line'])
        diff_rel = diff_abs / abs(real_data['DIFF']) * 100 if real_data['DIFF'] != 0 else 0
        diff_status = "✅" if diff_abs < 0.01 else "⚠️" if diff_abs < 0.05 else "❌"
        print(f"{'DIFF':<8} {real_data['DIFF']:<12.6f} {system_data['macd_line']:<12.6f} {diff_abs:<12.6f} {diff_rel:<12.1f}% {diff_status:<8}")
        
        dea_abs = abs(real_data['DEA'] - system_data['macd_signal'])
        dea_rel = dea_abs / abs(real_data['DEA']) * 100 if real_data['DEA'] != 0 else 0
        dea_status = "✅" if dea_abs < 0.01 else "⚠️" if dea_abs < 0.05 else "❌"
        print(f"{'DEA':<8} {real_data['DEA']:<12.6f} {system_data['macd_signal']:<12.6f} {dea_abs:<12.6f} {dea_rel:<12.1f}% {dea_status:<8}")
        
        macd_abs = abs(real_data['MACD'] - system_data['macd_histogram'])
        macd_rel = macd_abs / abs(real_data['MACD']) * 100 if real_data['MACD'] != 0 else 0
        macd_status = "✅" if macd_abs < 0.01 else "⚠️" if macd_abs < 0.05 else "❌"
        print(f"{'MACD':<8} {real_data['MACD']:<12.6f} {system_data['macd_histogram']:<12.6f} {macd_abs:<12.6f} {macd_rel:<12.1f}% {macd_status:<8}")
        
        # 问题分析
        print(f"\n🔍 问题分析:")
        
        major_discrepancy = diff_abs > 0.02 or dea_abs > 0.02 or macd_abs > 0.04
        
        if major_discrepancy:
            print(f"❌ 发现重大差异！")
            print(f"🔍 可能原因:")
            
            # 1. 检查MACD公式验证
            manual_macd = 2 * (system_data['macd_line'] - system_data['macd_signal'])
            print(f"  1. MACD公式验证:")
            print(f"     计算: 2 × ({system_data['macd_line']:.6f} - {system_data['macd_signal']:.6f}) = {manual_macd:.6f}")
            print(f"     系统: {system_data['macd_histogram']:.6f}")
            print(f"     公式正确: {'✅' if abs(manual_macd - system_data['macd_histogram']) < 0.000001 else '❌'}")
            
            # 2. 检查数据源差异
            print(f"  2. 数据源分析:")
            print(f"     收盘价: {price_data['close']:.3f}")
            print(f"     建议: 确认真实数据使用的收盘价是否一致")
            
            # 3. 检查参数设置
            print(f"  3. MACD参数检查:")
            if hasattr(macd_indicator, '_parameters'):
                params = macd_indicator._parameters
                print(f"     快线周期: {params.get('fast_period', 12)}")
                print(f"     慢线周期: {params.get('slow_period', 26)}")
                print(f"     信号线周期: {params.get('signal_period', 9)}")
            else:
                print(f"     使用默认参数: EMA12, EMA26, EMA9")
            
            # 4. 显示前后几天的MACD趋势
            print(f"  4. MACD趋势分析:")
            print(f"     最近5天MACD数据:")
            
            start_idx = max(0, target_idx - 2)
            end_idx = min(len(macd_result), target_idx + 3)
            
            print(f"     {'日期':<12} {'DIFF':<12} {'DEA':<12} {'MACD':<12}")
            print(f"     {'-'*50}")
            
            for i in range(start_idx, end_idx):
                date_str = df.iloc[i]['date'].strftime('%Y-%m-%d')
                diff_val = macd_result.iloc[i]['macd_line']
                dea_val = macd_result.iloc[i]['macd_signal']
                macd_val = macd_result.iloc[i]['macd_histogram']
                marker = " ← 目标" if i == target_idx else ""
                print(f"     {date_str:<12} {diff_val:<12.6f} {dea_val:<12.6f} {macd_val:<12.6f}{marker}")
            
            # 5. 手动EMA计算验证
            print(f"  5. 手动EMA计算验证:")
            close_prices = df['close'].values
            
            def simple_ema(prices, period):
                ema = np.zeros(len(prices))
                multiplier = 2 / (period + 1)
                ema[0] = prices[0]
                for i in range(1, len(prices)):
                    ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
                return ema
            
            ema12 = simple_ema(close_prices, 12)
            ema26 = simple_ema(close_prices, 26)
            manual_diff = ema12[target_idx] - ema26[target_idx]
            
            print(f"     手动EMA12: {ema12[target_idx]:.6f}")
            print(f"     手动EMA26: {ema26[target_idx]:.6f}")
            print(f"     手动DIFF: {manual_diff:.6f}")
            print(f"     系统DIFF: {system_data['macd_line']:.6f}")
            print(f"     DIFF差异: {abs(manual_diff - system_data['macd_line']):.6f}")
            
        else:
            print(f"✅ 差异在可接受范围内")
        
        # 最终结论
        print(f"\n🏆 验证结论:")
        if major_discrepancy:
            print(f"❌ 000017股票MACD计算存在重大差异")
            print(f"💡 建议:")
            print(f"  1. 确认真实数据的具体来源和计算方法")
            print(f"  2. 检查数据源是否使用相同的价格数据")
            print(f"  3. 验证MACD参数设置是否一致")
            print(f"  4. 确认计算日期是否准确匹配")
        else:
            print(f"✅ 000017股票MACD计算基本准确")
            print(f"💡 小幅差异可能来源于:")
            print(f"  1. 数据精度差异")
            print(f"  2. 计算方法的细微差别")
            print(f"  3. 数据源的时间戳差异")
    
    except Exception as e:
        print(f"❌ 验证过程异常: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🔍 验证000017股票MACD计算准确性")
    print("用户反馈数据与系统计算存在差异，需要深入分析")
    
    verify_000017_macd()

if __name__ == "__main__":
    main()
