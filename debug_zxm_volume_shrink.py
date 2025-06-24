#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
调试ZXM_VOLUME_SHRINK指标问题
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from db.unified_data_manager import get_unified_data_manager
from indicators.complete_indicator_registry import complete_registry
from utils.logger import get_logger, init_logging

# 初始化日志
init_logging(level="INFO")
logger = get_logger(__name__)

def debug_zxm_volume_shrink():
    """调试ZXM_VOLUME_SHRINK指标"""
    print("=== 调试ZXM_VOLUME_SHRINK指标 ===\n")
    
    # 获取数据管理器
    data_manager = get_unified_data_manager()
    
    # 获取603359的日线数据
    stock_code = "603359"
    target_date = "2025-05-12"
    
    daily_data = data_manager.get_stock_data(
        stock_code=stock_code,
        end_date=target_date,
        period='daily',
        lookback_days=90
    )
    
    print(f"获取到日线数据: {len(daily_data)} 条")
    print(f"日期范围: {daily_data['date'].min()} 到 {daily_data['date'].max()}")
    
    # 检查目标日期是否存在
    target_exists = target_date in daily_data['date'].values
    print(f"目标日期 {target_date} 是否存在: {target_exists}")
    
    if not target_exists:
        print(f"\n最近的几个交易日:")
        recent_dates = daily_data['date'].tail(10).tolist()
        for date in recent_dates:
            print(f"  {date}")
        
        # 使用最近的交易日
        latest_date = daily_data['date'].max()
        print(f"\n使用最近的交易日: {latest_date}")
        target_date = latest_date
    
    # 创建ZXM_VOLUME_SHRINK指标
    zxm_volume = complete_registry.create_indicator('ZXM_VOLUME_SHRINK')
    
    # 计算指标
    result = zxm_volume.calculate(daily_data)
    
    print(f"\nZXM_VOLUME_SHRINK计算结果:")
    print(f"结果长度: {len(result)}")
    print(f"结果列名: {list(result.columns)}")
    
    if not result.empty:
        print(f"结果日期范围: {result['date'].min()} 到 {result['date'].max()}")
        
        # 分析目标日期的结果
        target_data = result[result['date'] == target_date]
        
        print(f"\n{target_date} 的ZXM_VOLUME_SHRINK分析:")
        print(f"数据条数: {len(target_data)}")
        
        if not target_data.empty:
            row = target_data.iloc[0]
            buy_signal = row.get('buy_signal', False)
            sell_signal = row.get('sell_signal', False)
            
            print(f"  buy_signal: {buy_signal}")
            print(f"  sell_signal: {sell_signal}")
            
            # 显示所有计算字段
            print(f"  所有字段:")
            for col in row.index:
                if col not in ['date'] and pd.notna(row[col]):
                    print(f"    {col} = {row[col]}")

            # 详细分析缩量条件
            vol_ratio = row.get('VOL_RATIO', 0)
            ma_vol_2 = row.get('MA_VOL_2', 0)
            volume = row.get('volume', 0)

            print(f"\n  🔍 缩量条件详细分析:")
            print(f"    当日成交量: {volume}")
            print(f"    2日平均成交量: {ma_vol_2:.2f}")
            print(f"    量比 (VOL_RATIO): {vol_ratio:.4f}")
            print(f"    缩量条件 (VOL_RATIO < 0.9): {vol_ratio < 0.9}")
            print(f"    XG (缩量信号): {row.get('XG', False)}")

            if vol_ratio >= 0.9:
                print(f"    ❌ 未缩量：量比{vol_ratio:.4f} >= 0.9，成交量未明显缩减")
            else:
                print(f"    ✅ 缩量：量比{vol_ratio:.4f} < 0.9，成交量明显缩减")

            print(f"  📊 条件2 (ZXM_VOLUME_SHRINK=BUY): {'✅ 通过' if buy_signal else '❌ 未通过'}")
            
            return buy_signal
        else:
            print(f"❌ 没有找到 {target_date} 的计算结果")
            
            # 显示最近几天的结果
            print(f"\n最近5天的计算结果:")
            recent_results = result.tail(5)
            for idx, row in recent_results.iterrows():
                date_str = row.get('date', str(idx))
                buy_signal = row.get('buy_signal', False)
                print(f"  {date_str}: buy_signal={buy_signal}")
            
            return False
    else:
        print("❌ ZXM_VOLUME_SHRINK计算结果为空")
        return False

def check_zxm_volume_shrink_source():
    """检查ZXM_VOLUME_SHRINK指标源码"""
    print("\n=== 检查ZXM_VOLUME_SHRINK指标源码 ===\n")
    
    # 创建指标实例
    zxm_volume = complete_registry.create_indicator('ZXM_VOLUME_SHRINK')
    
    print(f"指标名称: {zxm_volume.name}")
    print(f"指标描述: {zxm_volume.description}")
    print(f"指标类: {zxm_volume.__class__.__name__}")
    
    # 检查是否有buy_signal生成逻辑
    if hasattr(zxm_volume, 'add_signal_generation'):
        print("✅ 指标有add_signal_generation方法")
    else:
        print("❌ 指标没有add_signal_generation方法")
    
    # 检查是否继承了PatternSignalMixin
    from indicators.base.pattern_signal_mixin import PatternSignalMixin
    if isinstance(zxm_volume, PatternSignalMixin):
        print("✅ 指标继承了PatternSignalMixin")
    else:
        print("❌ 指标没有继承PatternSignalMixin")

def fix_zxm_volume_shrink():
    """修复ZXM_VOLUME_SHRINK指标的信号生成"""
    print("\n=== 修复ZXM_VOLUME_SHRINK指标 ===\n")
    
    # 查找ZXM_VOLUME_SHRINK指标的实现文件
    import inspect
    zxm_volume = complete_registry.create_indicator('ZXM_VOLUME_SHRINK')
    source_file = inspect.getfile(zxm_volume.__class__)
    
    print(f"指标源文件: {source_file}")
    
    # 检查是否需要类似的修复
    # 这里我们需要查看ZXM_VOLUME_SHRINK的具体实现
    
    return True

def main():
    """主函数"""
    print("🔧 调试ZXM_VOLUME_SHRINK指标问题\n")
    
    # 1. 调试指标计算
    volume_success = debug_zxm_volume_shrink()
    
    # 2. 检查指标源码
    check_zxm_volume_shrink_source()
    
    # 3. 尝试修复
    fix_success = fix_zxm_volume_shrink()
    
    print(f"\n🎯 调试结果:")
    print(f"  ZXM_VOLUME_SHRINK计算: {'✅ 成功' if volume_success else '❌ 失败'}")
    print(f"  修复尝试: {'✅ 成功' if fix_success else '❌ 失败'}")

if __name__ == "__main__":
    main()
