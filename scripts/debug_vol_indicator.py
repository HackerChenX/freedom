#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试VOL指标计算过程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.data_manager_adapter import Data_manager_adapter
from indicators.complete_indicator_registry import complete_registry
from strategy.strategy_condition_evaluator import Strategy_condition_evaluator
import pandas as pd
import numpy as np

def debug_vol_indicator():
    """调试VOL指标计算过程"""
    
    print("🔍 开始调试VOL指标计算过程")
    print("=" * 60)
    
    # 1. 获取603359的数据
    dm = Data_manager_adapter()
    stock_code = '603359'
    
    try:
        data = dm.get_stock_data(stock_code, start_date='2024-11-01', end_date='2024-12-01')
        if data is None or data.empty:
            print(f"❌ 无法获取{stock_code}的数据")
            return
            
        print(f"✅ 成功获取{stock_code}数据: {len(data)} 条记录")
        print(f"📅 数据时间范围: {data.index.min()} 到 {data.index.max()}")
        print(f"📊 数据列: {list(data.columns)}")
        print()
        
        # 显示最新几天的数据
        print("📊 最新5天的OHLCV数据:")
        print(data.tail(5)[['open', 'high', 'low', 'close', 'volume']])
        print()
        
        # 2. 计算VOL指标
        print("🔧 开始计算VOL指标...")
        vol_indicator = complete_registry.create_indicator('VOL')
        if vol_indicator is None:
            print("❌ 无法创建VOL指标实例")
            return
            
        vol_result = vol_indicator.calculate(data)
        print(f"✅ VOL指标计算完成，结果列: {list(vol_result.columns)}")
        print()
        
        # 显示VOL指标结果
        vol_columns = ['vol', 'vol_ma5', 'vol_ma10', 'vol_ma20', 'vol_ratio']
        available_cols = [col for col in vol_columns if col in vol_result.columns]
        
        print("📊 最新5天的VOL指标数据:")
        print(vol_result.tail(5)[available_cols])
        print()
        
        # 3. 检查"放量上涨"条件
        print("🔍 检查'放量上涨'条件...")
        
        # 检查最新一天的数据
        latest_data = vol_result.iloc[-1]
        latest_price_data = data.iloc[-1]
        prev_price_data = data.iloc[-2] if len(data) > 1 else None
        
        print(f"📅 最新交易日: {vol_result.index[-1]}")
        print(f"📊 当日成交量: {latest_data.get('vol', 'N/A'):,.0f}")
        print(f"📊 5日均量: {latest_data.get('vol_ma5', 'N/A'):,.0f}")
        print(f"📊 量比(vol_ratio): {latest_data.get('vol_ratio', 'N/A'):.2f}")
        
        if prev_price_data is not None:
            price_change = latest_price_data['close'] - prev_price_data['close']
            price_change_pct = (price_change / prev_price_data['close']) * 100
            print(f"📊 价格变化: {price_change:.2f} ({price_change_pct:+.2f}%)")
            
            # 判断是否为放量上涨
            is_volume_up = latest_data.get('vol_ratio', 0) > 1.5  # 放量标准
            is_price_up = price_change > 0  # 上涨标准
            
            print(f"🔍 是否放量: {'✅' if is_volume_up else '❌'} (量比 > 1.5)")
            print(f"🔍 是否上涨: {'✅' if is_price_up else '❌'} (价格上涨)")
            print(f"🔍 放量上涨: {'✅' if is_volume_up and is_price_up else '❌'}")
        print()
        
        # 4. 使用策略条件评估器测试
        print("🧪 使用策略条件评估器测试...")
        evaluator = Strategy_condition_evaluator()
        
        # 构造测试条件
        test_condition = {
            "type": "indicator",
            "period": "15min",
            "indicator": "VOL",
            "pattern": "放量上涨",
            "score_threshold": 0.5
        }
        
        # 评估条件
        result = evaluator.evaluate_condition(
            stock_code=stock_code,
            condition=test_condition,
            date='2024-12-01',
            stock_data=data
        )
        
        print(f"🔍 条件评估结果: {result}")
        print(f"🔍 是否满足条件: {'✅' if result else '❌'}")
        print()
        
        # 5. 检查VOL指标的形态识别
        print("🔍 检查VOL指标的形态识别...")
        if hasattr(vol_indicator, 'identify_patterns'):
            patterns = vol_indicator.identify_patterns(vol_result)
            print(f"📊 识别到的形态: {patterns}")
        
        # 检查是否有"放量上涨"相关的列
        pattern_cols = [col for col in vol_result.columns if '放量' in col or 'volume' in col.lower() or 'pattern' in col.lower()]
        if pattern_cols:
            print(f"📊 形态相关列: {pattern_cols}")
            print(vol_result.tail(3)[pattern_cols])
        else:
            print("⚠️ 未找到形态相关列")
        print()
        
        # 6. 分析为什么条件不满足
        print("🔍 分析为什么条件不满足...")
        
        # 检查数据完整性
        if len(data) < 20:
            print(f"⚠️ 数据量不足: 只有{len(data)}条记录，可能影响指标计算")
        
        # 检查成交量数据
        if 'volume' not in data.columns:
            print("❌ 缺少成交量数据")
        elif data['volume'].isna().any():
            print("⚠️ 成交量数据存在缺失值")
        elif (data['volume'] == 0).any():
            print("⚠️ 成交量数据存在零值")
        
        # 检查价格数据
        if any(col not in data.columns for col in ['open', 'high', 'low', 'close']):
            print("❌ 缺少价格数据")
        
        print("🔍 调试完成")
        
    except Exception as e:
        print(f"❌ 调试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_vol_indicator()
