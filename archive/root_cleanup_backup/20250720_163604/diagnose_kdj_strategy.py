#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KDJ策略诊断脚本

分析为什么KDJ均上移策略没有选出股票

Author: AI Assistant
Date: 2025-07-20
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_kdj(data: pd.DataFrame, k_period: int = 9, d_period: int = 3) -> Dict[str, pd.Series]:
    """计算KDJ指标"""
    try:
        high = pd.to_numeric(data['high'], errors='coerce')
        low = pd.to_numeric(data['low'], errors='coerce')
        close = pd.to_numeric(data['close'], errors='coerce')
        
        # 计算RSV
        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()
        
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        
        # 计算K值
        k = rsv.ewm(alpha=1/d_period, adjust=False).mean()
        
        # 计算D值
        d = k.ewm(alpha=1/d_period, adjust=False).mean()
        
        # 计算J值
        j = 3 * k - 2 * d
        
        return {'k': k, 'd': d, 'j': j}
        
    except Exception as e:
        logger.error(f"计算KDJ失败: {e}")
        return {'k': pd.Series(), 'd': pd.Series(), 'j': pd.Series()}


def analyze_sample_stocks():
    """分析样本股票的KDJ情况"""
    from db.clickhouse_db import get_clickhouse_db
    
    target_date = '2025-05-12'
    start_date = '2025-03-01'  # 更长的历史数据
    
    # 选择几只代表性股票进行分析
    sample_stocks = ['000001', '000002', '000858', '002415', '300059']
    
    db = get_clickhouse_db()
    
    for stock_code in sample_stocks:
        print(f"\n{'='*60}")
        print(f"📊 分析股票: {stock_code}")
        print('='*60)
        
        try:
            # 获取股票数据
            query = f"""
            SELECT code, name, date, open, close, high, low, volume
            FROM stock.stock_info 
            WHERE code = '{stock_code}' 
            AND date >= '{start_date}' 
            AND date <= '{target_date}'
            ORDER BY date ASC
            """
            
            result = db.query(query)
            
            if result is None or result.empty:
                print(f"❌ 未找到 {stock_code} 的数据")
                continue
            
            print(f"✅ 获取到 {len(result)} 条数据")
            
            # 计算KDJ
            kdj_data = calculate_kdj(result)
            
            if kdj_data['k'].empty:
                print(f"❌ KDJ计算失败")
                continue
            
            # 找到目标日期的数据
            target_data = result[result['date'] == target_date]
            if target_data.empty:
                print(f"❌ 未找到目标日期 {target_date} 的数据")
                continue
            
            target_idx = target_data.index[0]
            data_idx = result.index.get_loc(target_idx)
            
            if data_idx < 1:
                print(f"❌ 数据不足，无法比较前一日")
                continue
            
            # 获取KDJ值
            k_current = kdj_data['k'].iloc[data_idx]
            k_previous = kdj_data['k'].iloc[data_idx - 1]
            d_current = kdj_data['d'].iloc[data_idx]
            d_previous = kdj_data['d'].iloc[data_idx - 1]
            j_current = kdj_data['j'].iloc[data_idx]
            j_previous = kdj_data['j'].iloc[data_idx - 1]
            
            print(f"📈 KDJ值分析:")
            print(f"   K值: {k_previous:.2f} → {k_current:.2f} ({'↑' if k_current > k_previous else '↓'})")
            print(f"   D值: {d_previous:.2f} → {d_current:.2f} ({'↑' if d_current > d_previous else '↓'})")
            print(f"   J值: {j_previous:.2f} → {j_current:.2f} ({'↑' if j_current > j_previous else '↓'})")
            
            # 检查条件
            k_upward = k_current > k_previous
            d_upward = d_current > d_previous
            j_upward = j_current > j_previous
            
            print(f"🎯 条件检查:")
            print(f"   K线上升: {'✅' if k_upward else '❌'}")
            print(f"   D线上升: {'✅' if d_upward else '❌'}")
            print(f"   J线上升: {'✅' if j_upward else '❌'}")
            print(f"   三线均上升: {'✅' if (k_upward and d_upward and j_upward) else '❌'}")
            
            # 检查范围过滤
            k_in_range = 20 <= k_current <= 80
            d_in_range = 20 <= d_current <= 80
            j_in_range = 0 <= j_current <= 100
            
            print(f"📏 范围检查:")
            print(f"   K值范围[20-80]: {'✅' if k_in_range else '❌'}")
            print(f"   D值范围[20-80]: {'✅' if d_in_range else '❌'}")
            print(f"   J值范围[0-100]: {'✅' if j_in_range else '❌'}")
            
            # 最终结果
            meets_all_conditions = (k_upward and d_upward and j_upward and 
                                  k_in_range and d_in_range and j_in_range)
            
            print(f"🏆 最终结果: {'✅ 符合条件' if meets_all_conditions else '❌ 不符合条件'}")
            
            # 显示最近几天的KDJ趋势
            print(f"📊 最近5天KDJ趋势:")
            recent_data = result.tail(5)
            recent_k = kdj_data['k'].tail(5)
            recent_d = kdj_data['d'].tail(5)
            recent_j = kdj_data['j'].tail(5)
            
            for i, (idx, row) in enumerate(recent_data.iterrows()):
                if i < len(recent_k):
                    print(f"   {row['date']}: K={recent_k.iloc[i]:.2f}, D={recent_d.iloc[i]:.2f}, J={recent_j.iloc[i]:.2f}")
            
        except Exception as e:
            print(f"❌ 分析股票 {stock_code} 失败: {e}")
            import traceback
            traceback.print_exc()


def analyze_strategy_conditions():
    """分析策略条件的合理性"""
    print(f"\n{'='*60}")
    print(f"🔧 策略条件分析")
    print('='*60)
    
    print("当前策略条件:")
    print("1. K线上升: K(today) > K(yesterday)")
    print("2. D线上升: D(today) > D(yesterday)")
    print("3. J线上升: J(today) > J(yesterday)")
    print("4. K值范围: 20-80")
    print("5. D值范围: 20-80")
    print("6. J值范围: 0-100")
    print()
    
    print("可能的问题:")
    print("1. 三线同时上升的条件可能过于严格")
    print("2. KDJ值范围限制可能过于严格")
    print("3. 目标日期可能不是最佳的技术分析时点")
    print()
    
    print("建议的调整:")
    print("1. 放宽范围限制，如K值范围改为10-90")
    print("2. 允许部分条件满足，如至少2条线上升")
    print("3. 增加KDJ金叉等其他技术条件")


def suggest_optimized_strategy():
    """建议优化后的策略"""
    print(f"\n{'='*60}")
    print(f"💡 优化策略建议")
    print('='*60)
    
    print("优化方案1: 放宽条件")
    print("- K、D、J至少2条线上升")
    print("- K值范围: 10-90")
    print("- D值范围: 10-90")
    print("- J值范围: -20-120")
    print()
    
    print("优化方案2: 增加技术条件")
    print("- KDJ金叉: K线上穿D线")
    print("- 超卖反弹: K值从20以下上升")
    print("- 成交量配合: 成交量放大")
    print()
    
    print("优化方案3: 多时间框架")
    print("- 日线KDJ上升")
    print("- 周线KDJ不在超买区")
    print("- 月线趋势向上")


if __name__ == "__main__":
    print("🔍 KDJ均上移策略诊断分析")
    print("="*80)
    
    # 分析样本股票
    analyze_sample_stocks()
    
    # 分析策略条件
    analyze_strategy_conditions()
    
    # 建议优化策略
    suggest_optimized_strategy()
    
    print(f"\n{'='*80}")
    print("🎯 诊断完成")
    print("="*80)
