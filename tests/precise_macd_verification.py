#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精确MACD验证

用户反馈000028股票2025年5月12日真实数据：
- MACD = 0.220
- DIFF = -0.136  
- DEA = -0.246

验证我们的计算是否正确，并找出差异原因
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

def precise_macd_verification():
    """精确MACD验证"""
    
    print("🔍 精确MACD验证 - 000028股票")
    print("=" * 80)
    
    # 用户提供的真实数据
    real_data = {
        'MACD': 0.220,
        'DIFF': -0.136,
        'DEA': -0.246,
        'date': '2025-05-12'
    }
    
    print(f"📋 用户提供的真实数据（{real_data['date']}）:")
    print(f"  MACD = {real_data['MACD']}")
    print(f"  DIFF = {real_data['DIFF']}")
    print(f"  DEA = {real_data['DEA']}")
    
    # 验证公式一致性
    calculated_macd = 2 * (real_data['DIFF'] - real_data['DEA'])
    print(f"\n🧮 公式验证:")
    print(f"  MACD = 2 × (DIFF - DEA)")
    print(f"  MACD = 2 × ({real_data['DIFF']} - ({real_data['DEA']}))")
    print(f"  MACD = 2 × {real_data['DIFF'] - real_data['DEA']}")
    print(f"  MACD = {calculated_macd}")
    print(f"  与真实值差异: {abs(calculated_macd - real_data['MACD']):.6f}")
    
    if abs(calculated_macd - real_data['MACD']) < 0.001:
        print(f"  ✅ 公式验证通过，我们的MACD计算公式正确")
    else:
        print(f"  ❌ 公式验证失败，可能使用了不同的计算方法")
    
    try:
        # 获取我们系统的数据
        macd_indicator = MacdMacd()
        stock_data_service = get_stock_data_service()
        
        stock_code = "000028"
        
        print(f"\n📊 获取系统数据进行对比...")
        df = stock_data_service.get_stock_data(stock_code, days=200)
        
        if df is None or len(df) == 0:
            print(f"❌ 无法获取{stock_code}数据")
            return
        
        print(f"✅ 获取到{len(df)}天数据")
        print(f"📅 数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
        
        # 计算MACD
        macd_result = macd_indicator.calculate(df)
        
        if macd_result is None or macd_result.empty:
            print("❌ MACD计算失败")
            return
        
        # 查找最接近的日期
        target_date = pd.to_datetime(real_data['date']).date()
        
        # 检查目标日期是否存在
        exact_match = df[df['date'].dt.date == target_date]
        
        if not exact_match.empty:
            print(f"✅ 找到精确匹配日期: {target_date}")
            target_idx = exact_match.index[0]
        else:
            print(f"⚠️ 未找到精确日期{target_date}，查找最接近的交易日")
            
            # 找最接近的交易日
            df['date_diff'] = abs((df['date'].dt.date - target_date).apply(lambda x: x.days))
            closest_idx = df['date_diff'].idxmin()
            target_idx = closest_idx
            closest_date = df.loc[closest_idx]['date'].date()
            
            print(f"📅 最接近的交易日: {closest_date}")
            print(f"📊 该日价格数据:")
            price_data = df.loc[closest_idx]
            print(f"  开盘: {price_data['open']:.3f}")
            print(f"  收盘: {price_data['close']:.3f}")
            print(f"  最高: {price_data['high']:.3f}")
            print(f"  最低: {price_data['low']:.3f}")
        
        # 获取系统计算的MACD数据
        if target_idx < len(macd_result):
            system_data = macd_result.iloc[target_idx]
            
            print(f"\n📊 系统计算结果:")
            print(f"  DIFF (macd_line): {system_data['macd_line']:.6f}")
            print(f"  DEA (macd_signal): {system_data['macd_signal']:.6f}")
            print(f"  MACD (macd_histogram): {system_data['macd_histogram']:.6f}")
            
            # 详细对比
            print(f"\n📋 详细对比分析:")
            print(f"{'指标':<8} {'真实值':<12} {'系统值':<12} {'绝对差异':<12} {'相对差异':<12}")
            print("-" * 60)
            
            diff_abs_diff = abs(real_data['DIFF'] - system_data['macd_line'])
            diff_rel_diff = diff_abs_diff / abs(real_data['DIFF']) * 100 if real_data['DIFF'] != 0 else 0
            print(f"{'DIFF':<8} {real_data['DIFF']:<12.6f} {system_data['macd_line']:<12.6f} {diff_abs_diff:<12.6f} {diff_rel_diff:<12.2f}%")
            
            dea_abs_diff = abs(real_data['DEA'] - system_data['macd_signal'])
            dea_rel_diff = dea_abs_diff / abs(real_data['DEA']) * 100 if real_data['DEA'] != 0 else 0
            print(f"{'DEA':<8} {real_data['DEA']:<12.6f} {system_data['macd_signal']:<12.6f} {dea_abs_diff:<12.6f} {dea_rel_diff:<12.2f}%")
            
            macd_abs_diff = abs(real_data['MACD'] - system_data['macd_histogram'])
            macd_rel_diff = macd_abs_diff / abs(real_data['MACD']) * 100 if real_data['MACD'] != 0 else 0
            print(f"{'MACD':<8} {real_data['MACD']:<12.6f} {system_data['macd_histogram']:<12.6f} {macd_abs_diff:<12.6f} {macd_rel_diff:<12.2f}%")
            
            # 分析差异原因
            print(f"\n🔍 差异原因分析:")
            
            # 1. 精度差异
            if diff_abs_diff < 0.001 and dea_abs_diff < 0.001:
                print(f"  ✅ DIFF和DEA计算精度很高（差异<0.001），计算方法正确")
            elif diff_abs_diff < 0.01 and dea_abs_diff < 0.01:
                print(f"  ⚠️ DIFF和DEA有小幅差异（差异<0.01），可能是精度或数据源差异")
            else:
                print(f"  ❌ DIFF和DEA有显著差异，可能是计算方法或数据源不同")
            
            # 2. MACD公式验证
            system_macd_calculated = 2 * (system_data['macd_line'] - system_data['macd_signal'])
            formula_diff = abs(system_macd_calculated - system_data['macd_histogram'])
            
            print(f"  📐 系统内部公式一致性检查:")
            print(f"    计算值: 2 × ({system_data['macd_line']:.6f} - {system_data['macd_signal']:.6f}) = {system_macd_calculated:.6f}")
            print(f"    系统值: {system_data['macd_histogram']:.6f}")
            print(f"    差异: {formula_diff:.6f}")
            
            if formula_diff < 0.000001:
                print(f"    ✅ 系统内部公式一致性完美")
            else:
                print(f"    ⚠️ 系统内部公式可能有问题")
            
            # 3. 总体评估
            print(f"\n🏆 总体评估:")
            
            if (diff_abs_diff < 0.001 and dea_abs_diff < 0.001 and 
                macd_abs_diff < 0.01):
                print(f"  ✅ 系统计算结果与真实数据高度一致")
                print(f"  ✅ MACD指标计算正确，可以通过人工验证")
            elif (diff_abs_diff < 0.01 and dea_abs_diff < 0.01 and 
                  macd_abs_diff < 0.1):
                print(f"  ⚠️ 系统计算结果与真实数据基本一致")
                print(f"  ⚠️ 存在小幅差异，建议进一步检查数据源")
            else:
                print(f"  ❌ 系统计算结果与真实数据差异较大")
                print(f"  ❌ 需要检查计算方法或数据源")
            
            # 4. 建议
            print(f"\n💡 建议:")
            if diff_abs_diff < 0.01 and dea_abs_diff < 0.01:
                print(f"  1. DIFF和DEA计算基本正确，差异可能来自:")
                print(f"     - 数据源的细微差异")
                print(f"     - EMA计算的精度差异")
                print(f"     - 历史数据的起始点不同")
                print(f"  2. 建议接受当前计算结果，差异在可接受范围内")
                print(f"  3. 可以继续进行人工验证，重点关注形态识别准确性")
            else:
                print(f"  1. 需要进一步检查数据源一致性")
                print(f"  2. 验证EMA计算方法是否标准")
                print(f"  3. 确认历史数据的完整性")
        
        else:
            print(f"❌ 目标索引超出MACD结果范围")
    
    except Exception as e:
        print(f"❌ 验证过程异常: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    precise_macd_verification()

if __name__ == "__main__":
    main()
