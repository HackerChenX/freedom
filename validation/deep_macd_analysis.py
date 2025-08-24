#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深层MACD计算问题分析

发现问题：
- 000001: 真实MACD=0.073, DIFF=-0.039, DEA=-0.076
- 系统计算: MACD线(-0.129323), 信号线(-0.146258)
- 差异巨大，需要深入分析根本原因

可能的问题：
1. 日期匹配错误
2. 数据源不同
3. MACD计算方法错误
4. 参数设置错误
5. 数据预处理问题
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

def deep_macd_analysis():
    """深层MACD计算问题分析"""
    
    print("🔍 深层MACD计算问题分析")
    print("=" * 80)
    
    # 用户提供的真实数据
    real_benchmarks = {
        '000001': {
            'date': '2025-05-12',
            'MACD': 0.073,
            'DIFF': -0.039,
            'DEA': -0.076
        },
        '000028': {
            'date': '2025-05-12', 
            'MACD': 0.220,
            'DIFF': -0.136,
            'DEA': -0.246
        }
    }
    
    try:
        macd_indicator = MacdMacd()
        stock_data_service = get_stock_data_service()
        
        for stock_code, benchmark in real_benchmarks.items():
            print(f"\n📊 分析{stock_code}股票")
            print("-" * 60)
            
            # 获取股票数据
            df = stock_data_service.get_stock_data(stock_code, days=200)
            
            if df is None or len(df) == 0:
                print(f"❌ 无法获取{stock_code}数据")
                continue
            
            print(f"✅ 获取到{len(df)}天数据")
            print(f"📅 数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
            
            # 显示最近几天的价格数据
            print(f"\n📈 最近10天价格数据:")
            recent_price = df.tail(10)[['date', 'open', 'high', 'low', 'close', 'volume']]
            for _, row in recent_price.iterrows():
                print(f"  {row['date'].strftime('%Y-%m-%d')}: 开{row['open']:.2f} 高{row['high']:.2f} 低{row['low']:.2f} 收{row['close']:.2f} 量{row['volume']:,.0f}")
            
            # 查找目标日期
            target_date = pd.to_datetime(benchmark['date']).date()
            target_rows = df[df['date'].dt.date == target_date]
            
            if target_rows.empty:
                print(f"⚠️ 未找到精确日期{target_date}")
                # 查找最接近的日期
                df['date_diff'] = abs((df['date'].dt.date - target_date).apply(lambda x: x.days))
                closest_idx = df['date_diff'].idxmin()
                actual_date = df.loc[closest_idx]['date'].date()
                target_idx = closest_idx
                print(f"📅 使用最接近日期: {actual_date}")
            else:
                target_idx = target_rows.index[0]
                actual_date = target_date
                print(f"✅ 找到精确日期: {actual_date}")
            
            # 显示目标日期的价格数据
            price_data = df.iloc[target_idx]
            print(f"\n📊 {actual_date}价格数据:")
            print(f"  开盘: {price_data['open']:.3f}")
            print(f"  收盘: {price_data['close']:.3f}")
            print(f"  最高: {price_data['high']:.3f}")
            print(f"  最低: {price_data['low']:.3f}")
            print(f"  成交量: {price_data['volume']:,.0f}")
            
            # 检查MACD参数
            print(f"\n⚙️ MACD参数检查:")
            if hasattr(macd_indicator, '_parameters'):
                params = macd_indicator._parameters
                print(f"  快线周期(EMA12): {params.get('fast_period', 12)}")
                print(f"  慢线周期(EMA26): {params.get('slow_period', 26)}")
                print(f"  信号线周期(EMA9): {params.get('signal_period', 9)}")
            else:
                print(f"  使用默认参数: EMA12, EMA26, EMA9")
            
            # 计算MACD
            print(f"\n🔄 计算MACD指标...")
            macd_result = macd_indicator.calculate(df)
            
            if macd_result is None or macd_result.empty:
                print("❌ MACD计算失败")
                continue
            
            print(f"✅ MACD计算成功，共{len(macd_result)}条结果")
            print(f"📋 MACD结果列: {list(macd_result.columns)}")
            
            if target_idx >= len(macd_result):
                print(f"❌ 目标索引{target_idx}超出MACD结果范围{len(macd_result)}")
                continue
            
            # 获取系统计算结果
            system_data = macd_result.iloc[target_idx]
            
            print(f"\n📊 系统计算结果:")
            print(f"  DIFF (macd_line): {system_data['macd_line']:.6f}")
            print(f"  DEA (macd_signal): {system_data['macd_signal']:.6f}")
            print(f"  MACD (macd_histogram): {system_data['macd_histogram']:.6f}")
            
            # 对比真实数据
            print(f"\n📋 与真实数据对比:")
            print(f"{'指标':<8} {'真实值':<12} {'系统值':<12} {'绝对差异':<12} {'相对差异':<12}")
            print("-" * 65)
            
            diff_abs = abs(benchmark['DIFF'] - system_data['macd_line'])
            diff_rel = diff_abs / abs(benchmark['DIFF']) * 100 if benchmark['DIFF'] != 0 else 0
            print(f"{'DIFF':<8} {benchmark['DIFF']:<12.6f} {system_data['macd_line']:<12.6f} {diff_abs:<12.6f} {diff_rel:<12.1f}%")
            
            dea_abs = abs(benchmark['DEA'] - system_data['macd_signal'])
            dea_rel = dea_abs / abs(benchmark['DEA']) * 100 if benchmark['DEA'] != 0 else 0
            print(f"{'DEA':<8} {benchmark['DEA']:<12.6f} {system_data['macd_signal']:<12.6f} {dea_abs:<12.6f} {dea_rel:<12.1f}%")
            
            macd_abs = abs(benchmark['MACD'] - system_data['macd_histogram'])
            macd_rel = macd_abs / abs(benchmark['MACD']) * 100 if benchmark['MACD'] != 0 else 0
            print(f"{'MACD':<8} {benchmark['MACD']:<12.6f} {system_data['macd_histogram']:<12.6f} {macd_abs:<12.6f} {macd_rel:<12.1f}%")
            
            # 深度分析差异原因
            print(f"\n🔍 深度差异分析:")
            
            if diff_abs > 0.05 or dea_abs > 0.05:
                print(f"  ❌ 发现重大差异 (>0.05)")
                print(f"  🔍 可能原因分析:")
                
                # 1. 检查数据源差异
                print(f"    1. 数据源差异检查:")
                print(f"       - 收盘价: {price_data['close']:.3f}")
                print(f"       - 建议: 确认真实数据使用的收盘价是否一致")
                
                # 2. 检查日期匹配
                print(f"    2. 日期匹配检查:")
                print(f"       - 目标日期: {benchmark['date']}")
                print(f"       - 实际日期: {actual_date}")
                print(f"       - 日期匹配: {'✅' if str(actual_date) == benchmark['date'] else '❌'}")
                
                # 3. 检查计算方法
                print(f"    3. 计算方法检查:")
                manual_macd = 2 * (system_data['macd_line'] - system_data['macd_signal'])
                print(f"       - 公式验证: 2×({system_data['macd_line']:.6f} - {system_data['macd_signal']:.6f}) = {manual_macd:.6f}")
                print(f"       - 系统MACD: {system_data['macd_histogram']:.6f}")
                print(f"       - 公式一致: {'✅' if abs(manual_macd - system_data['macd_histogram']) < 0.000001 else '❌'}")
                
                # 4. 手动计算EMA验证
                print(f"    4. 手动EMA计算验证:")
                close_prices = df['close'].values
                
                # 简单EMA计算
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
                
                print(f"       - 手动EMA12: {ema12[target_idx]:.6f}")
                print(f"       - 手动EMA26: {ema26[target_idx]:.6f}")
                print(f"       - 手动DIFF: {manual_diff:.6f}")
                print(f"       - 系统DIFF: {system_data['macd_line']:.6f}")
                print(f"       - DIFF差异: {abs(manual_diff - system_data['macd_line']):.6f}")
                
                # 5. 数据完整性检查
                print(f"    5. 数据完整性检查:")
                null_count = df['close'].isnull().sum()
                duplicate_count = df['date'].duplicated().sum()
                print(f"       - 空值数量: {null_count}")
                print(f"       - 重复日期: {duplicate_count}")
                print(f"       - 数据连续性: {'✅' if null_count == 0 and duplicate_count == 0 else '❌'}")
                
            elif diff_abs > 0.01 or dea_abs > 0.01:
                print(f"  ⚠️ 发现中等差异 (0.01-0.05)")
                print(f"  可能是数据源精度或计算精度差异")
            else:
                print(f"  ✅ 差异很小 (<0.01)，在可接受范围内")
            
            # 显示最近几天的MACD趋势
            print(f"\n📈 最近5天MACD趋势:")
            recent_start = max(0, target_idx - 4)
            recent_end = min(len(macd_result), target_idx + 1)
            recent_macd = macd_result.iloc[recent_start:recent_end]
            recent_dates = df.iloc[recent_start:recent_end]['date']
            
            print(f"{'日期':<12} {'DIFF':<12} {'DEA':<12} {'MACD':<12}")
            print("-" * 50)
            for i in range(len(recent_macd)):
                date_str = recent_dates.iloc[i].strftime('%Y-%m-%d')
                diff_val = recent_macd.iloc[i]['macd_line']
                dea_val = recent_macd.iloc[i]['macd_signal']
                macd_val = recent_macd.iloc[i]['macd_histogram']
                marker = " ← 目标" if i == len(recent_macd) - 1 else ""
                print(f"{date_str:<12} {diff_val:<12.6f} {dea_val:<12.6f} {macd_val:<12.6f}{marker}")
    
    except Exception as e:
        print(f"❌ 深度分析异常: {e}")
        import traceback
        traceback.print_exc()

def investigate_data_source():
    """调查数据源问题"""
    
    print(f"\n🔍 数据源调查")
    print("=" * 60)
    
    try:
        stock_data_service = get_stock_data_service()
        
        # 检查数据服务配置
        print(f"📊 数据服务信息:")
        print(f"  数据服务类型: {type(stock_data_service).__name__}")
        
        # 获取000001的数据并检查
        df = stock_data_service.get_stock_data('000001', days=10)
        
        if df is not None and not df.empty:
            print(f"  数据源状态: ✅ 正常")
            print(f"  数据条数: {len(df)}")
            print(f"  数据列: {list(df.columns)}")
            print(f"  最新日期: {df['date'].max()}")
            
            # 检查2025-05-12附近的数据
            target_date = pd.to_datetime('2025-05-12').date()
            nearby_data = df[df['date'].dt.date >= target_date - timedelta(days=3)]
            nearby_data = nearby_data[nearby_data['date'].dt.date <= target_date + timedelta(days=3)]
            
            print(f"\n📅 2025-05-12附近的数据:")
            if not nearby_data.empty:
                for _, row in nearby_data.iterrows():
                    print(f"  {row['date'].strftime('%Y-%m-%d')}: 收盘{row['close']:.3f}")
            else:
                print(f"  ❌ 未找到2025-05-12附近的数据")
                print(f"  💡 这可能是问题的根源！")
        else:
            print(f"  数据源状态: ❌ 异常")
    
    except Exception as e:
        print(f"❌ 数据源调查异常: {e}")

def main():
    """主函数"""
    print("🔍 深层MACD计算问题分析")
    print("挖掘000001等股票MACD计算差异的根本原因")
    
    deep_macd_analysis()
    investigate_data_source()
    
    print(f"\n💡 问题排查建议:")
    print(f"1. 确认真实数据的具体日期和数据源")
    print(f"2. 检查系统数据是否包含2025-05-12的数据")
    print(f"3. 验证MACD计算参数是否一致")
    print(f"4. 确认数据预处理方法是否正确")
    print(f"5. 检查是否存在数据时区或格式问题")

if __name__ == "__main__":
    main()
