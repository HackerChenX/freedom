#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
买点分析器调试脚本

专门用于调试为什么所有指标都显示0%准确率
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from utils.logger import getLogger
from components.buypoint_analyzer import BuypointAnalyzer
from components.stockinfo_compatible_data_generator import StockInfoCompatibleDataGenerator

logger = getLogger(__name__)

def debug_single_test():
    """调试单个测试用例"""
    print("🔍 开始调试买点分析器...")
    
    # 初始化组件
    analyzer = BuypointAnalyzer()
    data_generator = StockInfoCompatibleDataGenerator()
    
    # 生成测试数据
    print("📊 生成测试数据...")
    test_data = data_generator.generate_stockinfo_compatible_data(
        indicator_name='MACD',
        pattern_type='GOLDEN_CROSS',
        stock_code='TEST001',
        history_days=60
    )
    
    print(f"数据形状: {test_data.shape}")
    print(f"数据列: {list(test_data.columns)}")
    print(f"前5行数据:")
    print(test_data.head())
    print(f"后5行数据:")
    print(test_data.tail())
    
    # 测试指标计算
    print("\n🧮 测试指标计算...")
    indicator_values = analyzer._calculate_indicator(test_data, 'MACD')
    print(f"指标计算结果: {indicator_values}")
    
    if indicator_values:
        print("指标值详情:")
        for key, value in indicator_values.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                print(f"    最后5个值: {value.tail() if hasattr(value, 'tail') else value[-5:]}")
            else:
                print(f"  {key}: {type(value)} = {value}")
    
    # 测试形态检测
    print("\n🔍 测试形态检测...")
    if indicator_values:
        pattern_result = analyzer._detect_buypoint_pattern(
            indicator_values, test_data, 'MACD', 'GOLDEN_CROSS'
        )
        print(f"形态检测结果: {pattern_result}")
    
    # 测试完整流程
    print("\n🎯 测试完整流程...")
    full_result = analyzer.test_pattern_recognition(
        mock_data_pool=[test_data],
        pattern_key='MACD_GOLDEN_CROSS'
    )
    print(f"完整流程结果: {full_result}")
    
    # 测试其他指标
    indicators_to_test = ['RSI', 'KDJ', 'BOLL']
    for indicator in indicators_to_test:
        print(f"\n📈 测试指标: {indicator}")
        
        # 生成对应数据
        indicator_data = data_generator.generate_stockinfo_compatible_data(
            indicator_name=indicator,
            pattern_type='OVERSOLD' if indicator == 'RSI' else 'GOLDEN_CROSS',
            stock_code=f'TEST_{indicator}',
            history_days=50
        )
        
        # 计算指标
        indicator_values = analyzer._calculate_indicator(indicator_data, indicator)
        print(f"  指标计算结果: {indicator_values is not None}")
        
        if indicator_values:
            print(f"  指标值键: {list(indicator_values.keys())}")
        
        # 测试完整流程
        pattern_key = f"{indicator}_{'OVERSOLD' if indicator == 'RSI' else 'GOLDEN_CROSS'}"
        result = analyzer.test_pattern_recognition(
            mock_data_pool=[indicator_data],
            pattern_key=pattern_key
        )
        print(f"  测试结果准确率: {result.get('accuracy', 0)}%")

def debug_data_generation():
    """调试数据生成"""
    print("\n🏭 调试数据生成...")
    
    generator = StockInfoCompatibleDataGenerator()
    
    # 测试不同指标的数据生成
    indicators = ['MACD', 'RSI', 'KDJ', 'BOLL']
    
    for indicator in indicators:
        print(f"\n📊 测试 {indicator} 数据生成...")
        
        data = generator.generate_stockinfo_compatible_data(
            indicator_name=indicator,
            pattern_type='GOLDEN_CROSS',
            stock_code=f'DEBUG_{indicator}',
            history_days=30
        )
        
        print(f"  数据形状: {data.shape}")
        print(f"  数据类型: {data.dtypes.to_dict()}")
        print(f"  价格范围: close=[{data['close'].min():.2f}, {data['close'].max():.2f}]")
        print(f"  成交量范围: volume=[{data['volume'].min():.0f}, {data['volume'].max():.0f}]")
        
        # 验证数据兼容性
        compatibility = generator.validate_stockinfo_compatibility(data)
        print(f"  兼容性: {compatibility['is_compatible']}")
        if not compatibility['is_compatible']:
            print(f"  问题: {compatibility['data_quality_issues']}")

def debug_pattern_detection():
    """调试形态检测逻辑"""
    print("\n🎯 调试形态检测逻辑...")
    
    analyzer = BuypointAnalyzer()
    
    # 创建简单的测试数据
    test_data = pd.DataFrame({
        'close': [10, 10.5, 11, 10.8, 11.2, 11.5, 12],
        'high': [10.2, 10.7, 11.3, 11, 11.5, 11.8, 12.2],
        'low': [9.8, 10.3, 10.8, 10.5, 10.9, 11.2, 11.8],
        'open': [10, 10.5, 11, 10.8, 11.2, 11.5, 12],
        'volume': [1000000] * 7,
        'date': ['20240101', '20240102', '20240103', '20240104', '20240105', '20240108', '20240109']
    })
    
    print(f"测试数据:")
    print(test_data)
    
    # 手动计算MACD
    close = test_data['close']
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    
    print(f"\n手动MACD计算:")
    print(f"EMA12: {ema12.values}")
    print(f"EMA26: {ema26.values}")
    print(f"MACD: {macd.values}")
    print(f"Signal: {signal.values}")
    
    # 构造指标值格式
    indicator_values = {
        'macd_line': macd,
        'signal_line': signal,
        'histogram': macd - signal
    }
    
    # 测试形态检测
    pattern_result = analyzer._detect_macd_pattern(indicator_values, 'GOLDEN_CROSS')
    print(f"\n形态检测结果: {pattern_result}")

if __name__ == "__main__":
    print("=" * 80)
    print("🔧 买点分析器调试工具")
    print("=" * 80)
    
    try:
        debug_single_test()
        debug_data_generation()
        debug_pattern_detection()
        
        print("\n" + "=" * 80)
        print("✅ 调试完成")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 调试过程中发生错误: {e}")
        import traceback
        traceback.print_exc() 