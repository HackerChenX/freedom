#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
深度调试脚本：分析股票603359在2025-05-12的ZXM选股策略条件验证
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from db.unified_data_manager import get_unified_data_manager
from indicators.complete_indicator_registry import complete_registry
from strategy.strategy_condition_evaluator import Strategy_condition_evaluator
from strategy.strategy_parser import Strategy_parser
from utils.logger import get_logger, init_logging

# 初始化日志
init_logging(level="INFO")
logger = get_logger(__name__)

def deep_analyze_603359():
    """深度分析股票603359的ZXM选股策略条件验证"""
    stock_code = "603359"
    target_date = "2025-05-12"
    
    print(f"=== 深度分析股票 {stock_code} 在 {target_date} 的ZXM选股策略条件验证 ===\n")
    
    try:
        # 1. 获取数据管理器
        data_manager = get_unified_data_manager()
        
        # 2. 获取充足的历史数据
        print("1. 获取历史数据")
        print("-" * 60)
        
        # 获取日线数据（90天历史）
        daily_data = data_manager.get_stock_data(
            stock_code=stock_code,
            end_date=target_date,
            period='daily',
            lookback_days=90
        )
        
        print(f"✅ 日线数据: {len(daily_data)} 条记录")
        if not daily_data.empty:
            print(f"   日期范围: {daily_data['date'].min()} 到 {daily_data['date'].max()}")
            print(f"   目标日期数据: {'存在' if target_date in daily_data['date'].values else '不存在'}")
        
        # 获取30分钟数据（90天历史）
        min30_data = data_manager.get_stock_data(
            stock_code=stock_code,
            end_date=target_date,
            period='30min',
            lookback_days=90
        )
        
        print(f"✅ 30分钟数据: {len(min30_data)} 条记录")
        if not min30_data.empty:
            print(f"   日期范围: {min30_data['date'].min()} 到 {min30_data['date'].max()}")
            target_30min_count = len(min30_data[min30_data['date'] == target_date])
            print(f"   目标日期30分钟数据: {target_30min_count} 条")
        
        # 3. 深度分析ZXM_BS_ABSORB指标
        print(f"\n2. 深度分析ZXM_BS_ABSORB指标（30分钟）")
        print("-" * 60)
        
        zxm_absorb = complete_registry.create_indicator('ZXM_BS_ABSORB')
        if zxm_absorb and not min30_data.empty:
            print("2.1 指标创建成功，开始计算...")
            
            # 显示输入数据样本
            print("输入数据样本（最近5条30分钟数据）:")
            recent_30min = min30_data.tail(5)
            for idx, row in recent_30min.iterrows():
                print(f"  {row['date']} {row.get('time', '')}: O={row['open']:.2f}, H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f}, V={row['volume']}")
            
            # 计算指标
            absorb_result = zxm_absorb.calculate(min30_data)
            print(f"✅ ZXM_BS_ABSORB计算完成，结果长度: {len(absorb_result)}")
            
            if not absorb_result.empty:
                print(f"结果列名: {list(absorb_result.columns)}")
                
                # 查看最近的计算结果
                print("\n最近10个时段的ZXM_BS_ABSORB结果:")
                recent_results = absorb_result.tail(10)
                for idx, row in recent_results.iterrows():
                    date_str = row.get('date', str(idx))
                    time_str = row.get('time', '')
                    signal = row.get('signal', 'N/A')
                    # 显示更多调试信息
                    debug_info = []
                    for col in row.index:
                        if col not in ['date', 'time', 'signal'] and pd.notna(row[col]):
                            debug_info.append(f"{col}={row[col]}")
                    debug_str = ", ".join(debug_info[:3])  # 只显示前3个
                    print(f"  {date_str} {time_str}: signal={signal} ({debug_str})")
                
                # 重点分析目标日期
                target_absorb = absorb_result[absorb_result.get('date', pd.Series()) == target_date]
                if not target_absorb.empty:
                    print(f"\n🎯 {target_date} 的ZXM_BS_ABSORB详细分析:")
                    for idx, row in target_absorb.iterrows():
                        signal = row.get('signal', 'N/A')
                        time_str = row.get('time', '')
                        print(f"  时间 {time_str}: signal = {signal}")
                        
                        # 显示所有计算字段
                        for col in row.index:
                            if col not in ['date', 'time'] and pd.notna(row[col]):
                                print(f"    {col} = {row[col]}")
                    
                    # 统计BUY信号 - 修复：使用buy_signal列而不是signal列
                    buy_signals = target_absorb[target_absorb.get('buy_signal', False) == True]
                    print(f"  📊 BUY信号数量: {len(buy_signals)}")
                    print(f"  📊 条件1 (ZXM_BS_ABSORB=BUY): {'✅ 通过' if len(buy_signals) > 0 else '❌ 未通过'}")

                    # 分析为什么没有BUY信号
                    if len(buy_signals) == 0:
                        print("  🔍 分析未产生BUY信号的原因:")
                        for idx, row in target_absorb.iterrows():
                            time_str = row.get('time', '')
                            aa = row.get('AA', False)
                            bb = row.get('BB', False)
                            v11 = row.get('V11', 0)
                            ema_v11 = row.get('EMA_V11_3', 0)
                            v12 = row.get('V12', 0)
                            print(f"    {time_str}: AA={aa}, BB={bb}, V11={v11:.2f}, EMA_V11_3={ema_v11:.2f}, V12={v12:.2f}")

                            # 分析BUY信号条件
                            # 根据ZXM_BS_ABSORB逻辑，BUY信号通常需要：AA=True, BB=True, 且V11>EMA_V11_3
                            if aa and bb and v11 > ema_v11:
                                print(f"      ⚠️  满足基本条件但未产生BUY信号，可能需要额外条件")
                            elif not aa:
                                print(f"      ❌ AA条件不满足")
                            elif not bb:
                                print(f"      ❌ BB条件不满足")
                            elif v11 <= ema_v11:
                                print(f"      ❌ V11({v11:.2f}) <= EMA_V11_3({ema_v11:.2f})")
                else:
                    print(f"❌ {target_date} 没有ZXM_BS_ABSORB计算结果")
                    print("  📊 条件1 (ZXM_BS_ABSORB=BUY): ❌ 未通过（无数据）")
            else:
                print("❌ ZXM_BS_ABSORB计算结果为空")
                print("  📊 条件1 (ZXM_BS_ABSORB=BUY): ❌ 未通过（计算失败）")
        else:
            print("❌ ZXM_BS_ABSORB指标创建失败或30分钟数据为空")
            print("  📊 条件1 (ZXM_BS_ABSORB=BUY): ❌ 未通过（指标失败）")
        
        # 4. 深度分析ZXM_VOLUME_SHRINK指标
        print(f"\n3. 深度分析ZXM_VOLUME_SHRINK指标（日线）")
        print("-" * 60)
        
        zxm_volume = complete_registry.create_indicator('ZXM_VOLUME_SHRINK')
        if zxm_volume and not daily_data.empty:
            print("3.1 指标创建成功，开始计算...")
            
            # 显示输入数据样本
            print("输入数据样本（最近10个交易日）:")
            recent_daily = daily_data.tail(10)
            for idx, row in recent_daily.iterrows():
                print(f"  {row['date']}: O={row['open']:.2f}, H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f}, V={row['volume']}")
            
            # 计算指标
            volume_result = zxm_volume.calculate(daily_data)
            print(f"✅ ZXM_VOLUME_SHRINK计算完成，结果长度: {len(volume_result)}")
            
            if not volume_result.empty:
                print(f"结果列名: {list(volume_result.columns)}")
                
                # 查看最近的计算结果
                print("\n最近10个交易日的ZXM_VOLUME_SHRINK结果:")
                recent_results = volume_result.tail(10)
                for idx, row in recent_results.iterrows():
                    date_str = row.get('date', str(idx))
                    signal = row.get('signal', 'N/A')
                    # 显示更多调试信息
                    debug_info = []
                    for col in row.index:
                        if col not in ['date', 'signal'] and pd.notna(row[col]):
                            debug_info.append(f"{col}={row[col]}")
                    debug_str = ", ".join(debug_info[:3])  # 只显示前3个
                    print(f"  {date_str}: signal={signal} ({debug_str})")
                
                # 重点分析目标日期
                target_volume = volume_result[volume_result.get('date', pd.Series()) == target_date]
                if not target_volume.empty:
                    print(f"\n🎯 {target_date} 的ZXM_VOLUME_SHRINK详细分析:")
                    row = target_volume.iloc[0]
                    signal = row.get('signal', 'N/A')
                    print(f"  signal = {signal}")
                    
                    # 显示所有计算字段
                    for col in row.index:
                        if col not in ['date'] and pd.notna(row[col]):
                            print(f"    {col} = {row[col]}")
                    
                    # 判断条件
                    is_buy = signal == 'BUY'
                    print(f"  📊 条件2 (ZXM_VOLUME_SHRINK=BUY): {'✅ 通过' if is_buy else '❌ 未通过'}")
                else:
                    print(f"❌ {target_date} 没有ZXM_VOLUME_SHRINK计算结果")
                    print("  📊 条件2 (ZXM_VOLUME_SHRINK=BUY): ❌ 未通过（无数据）")
            else:
                print("❌ ZXM_VOLUME_SHRINK计算结果为空")
                print("  📊 条件2 (ZXM_VOLUME_SHRINK=BUY): ❌ 未通过（计算失败）")
        else:
            print("❌ ZXM_VOLUME_SHRINK指标创建失败或日线数据为空")
            print("  📊 条件2 (ZXM_VOLUME_SHRINK=BUY): ❌ 未通过（指标失败）")
        
        print(f"\n=== 深度分析完成 ===")
        
    except Exception as e:
        logger.error(f"深度分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    deep_analyze_603359()
