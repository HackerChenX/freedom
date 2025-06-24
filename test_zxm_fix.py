#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
测试ZXM_BS_ABSORB修复结果
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

def test_zxm_bs_absorb_fix():
    """测试ZXM_BS_ABSORB修复结果"""
    print("=== 测试ZXM_BS_ABSORB修复结果 ===\n")
    
    # 获取数据管理器
    data_manager = get_unified_data_manager()
    
    # 获取603359的30分钟数据
    stock_code = "603359"
    target_date = "2025-05-12"
    
    min30_data = data_manager.get_stock_data(
        stock_code=stock_code,
        end_date=target_date,
        period='30min',
        lookback_days=90
    )
    
    print(f"获取到30分钟数据: {len(min30_data)} 条")
    
    # 创建ZXM_BS_ABSORB指标
    zxm_absorb = complete_registry.create_indicator('ZXM_BS_ABSORB')
    
    # 计算指标
    result = zxm_absorb.calculate(min30_data)
    
    # 分析目标日期的结果
    target_data = result[result['date'] == target_date]
    
    print(f"\n{target_date} 的ZXM_BS_ABSORB修复后分析:")
    print(f"数据条数: {len(target_data)}")
    
    buy_signal_count = 0
    if not target_data.empty:
        print("\n详细数据:")
        for idx, row in target_data.iterrows():
            time_str = row.get('time', '')
            xg = row.get('XG', 0)
            buy_signal = row.get('buy_signal', False)
            aa = row.get('AA', False)
            bb = row.get('BB', False)
            v11 = row.get('V11', 0)
            ema_v11 = row.get('EMA_V11_3', 0)
            v12 = row.get('V12', 0)
            
            print(f"  {time_str}: XG={xg}, buy_signal={buy_signal}")
            print(f"    AA={aa}, BB={bb}, V11={v11:.2f}, EMA_V11_3={ema_v11:.2f}, V12={v12:.2f}")
            
            if buy_signal:
                buy_signal_count += 1
                print(f"    ✅ BUY信号！")
            
            # 验证修复逻辑
            expected_buy_signal = xg > 0
            if buy_signal == expected_buy_signal:
                print(f"    ✅ 修复正确：XG={xg} -> buy_signal={buy_signal}")
            else:
                print(f"    ❌ 修复错误：XG={xg} -> buy_signal={buy_signal}，期望={expected_buy_signal}")
    
    print(f"\n📊 总结:")
    print(f"  BUY信号数量: {buy_signal_count}")
    print(f"  条件1 (ZXM_BS_ABSORB=BUY): {'✅ 通过' if buy_signal_count > 0 else '❌ 未通过'}")
    
    return buy_signal_count > 0

def test_strategy_evaluation():
    """测试策略评估"""
    print("\n=== 测试策略评估 ===\n")
    
    # 模拟策略条件评估器的逻辑
    from strategy.strategy_condition_evaluator import StrategyConditionEvaluator
    
    # 获取数据管理器
    data_manager = get_unified_data_manager()
    
    # 获取603359的数据
    stock_code = "603359"
    target_date = "2025-05-12"
    
    # 获取30分钟数据
    min30_data = data_manager.get_stock_data(
        stock_code=stock_code,
        end_date=target_date,
        period='30min',
        lookback_days=90
    )
    
    # 获取日线数据
    daily_data = data_manager.get_stock_data(
        stock_code=stock_code,
        end_date=target_date,
        period='daily',
        lookback_days=90
    )
    
    print(f"30分钟数据: {len(min30_data)} 条")
    print(f"日线数据: {len(daily_data)} 条")
    
    # 创建条件评估器
    evaluator = StrategyConditionEvaluator()
    
    # 模拟策略条件
    conditions = [
        {
            'type': 'indicator',
            'indicator_id': 'ZXM_BS_ABSORB',
            'period': '30min',
            'signal_type': 'BUY',
            'parameter': 'zxm_bs_absorb_signal',
            'operator': '=',
            'value': 1,
            'description': '30分钟时间框架下出现ZXM主力吸筹信号'
        },
        {'logic': 'AND'},
        {
            'type': 'indicator',
            'indicator_id': 'ZXM_VOLUME_SHRINK',
            'period': 'DAILY',
            'signal_type': 'BUY',
            'parameter': 'zxm_volume_shrink_signal',
            'operator': '=',
            'value': 1,
            'description': '日线时间框架下出现缩量信号'
        }
    ]
    
    # 评估条件
    try:
        # 合并数据 - 策略评估器期望单一的stock_data
        # 这里我们使用日线数据作为主数据，30分钟数据通过其他方式传递
        result = evaluator.evaluate_conditions(
            conditions=conditions,
            stock_data=daily_data,
            date=target_date,
            logic="and"
        )
        
        print(f"策略条件评估结果: {result}")
        print(f"603359是否通过策略: {'✅ 通过' if result else '❌ 未通过'}")
        
        return result
        
    except Exception as e:
        print(f"❌ 策略条件评估出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_zxm_volume_shrink():
    """测试ZXM_VOLUME_SHRINK指标"""
    print("\n=== 测试ZXM_VOLUME_SHRINK指标 ===\n")
    
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
    
    # 创建ZXM_VOLUME_SHRINK指标
    zxm_volume = complete_registry.create_indicator('ZXM_VOLUME_SHRINK')
    
    # 计算指标
    result = zxm_volume.calculate(daily_data)
    
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
        for col in row.index:
            if col not in ['date'] and pd.notna(row[col]):
                print(f"    {col} = {row[col]}")
        
        print(f"  📊 条件2 (ZXM_VOLUME_SHRINK=BUY): {'✅ 通过' if buy_signal else '❌ 未通过'}")
        
        return buy_signal
    else:
        print("❌ 没有找到目标日期的数据")
        return False

def main():
    """主函数"""
    print("🔧 测试ZXM指标修复结果\n")
    
    # 1. 测试ZXM_BS_ABSORB修复
    absorb_success = test_zxm_bs_absorb_fix()
    
    # 2. 测试ZXM_VOLUME_SHRINK
    volume_success = test_zxm_volume_shrink()
    
    # 3. 测试策略评估
    strategy_success = test_strategy_evaluation()
    
    print(f"\n🎯 最终结果:")
    print(f"  ZXM_BS_ABSORB修复: {'✅ 成功' if absorb_success else '❌ 失败'}")
    print(f"  ZXM_VOLUME_SHRINK: {'✅ 通过' if volume_success else '❌ 未通过'}")
    print(f"  策略整体评估: {'✅ 通过' if strategy_success else '❌ 未通过'}")
    
    if absorb_success and volume_success and strategy_success:
        print("\n🎉 修复成功！603359现在应该能通过ZXM吸筹+缩量选股策略了。")
    else:
        print("\n⚠️  还有问题需要解决。")

if __name__ == "__main__":
    main()
