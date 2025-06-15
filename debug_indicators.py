#!/usr/bin/env python3
"""
指标调试脚本 - 诊断新集成指标的计算问题
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.getcwd())

from indicators.indicator_registry import indicator_registry
from utils.logger import get_logger

logger = get_logger(__name__)

def create_test_data(periods=100):
    """创建测试股票数据"""
    dates = pd.date_range('2023-01-01', periods=periods, freq='D')
    
    # 生成模拟股票数据
    np.random.seed(42)  # 确保可重复性
    base_price = 100
    
    # 生成价格序列
    returns = np.random.normal(0.001, 0.02, periods)  # 日收益率
    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    # 生成OHLC数据
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = data['close'].shift(1).fillna(base_price)
    
    # 生成高低价（基于收盘价的波动）
    daily_range = np.random.uniform(0.01, 0.05, periods)  # 1-5%的日内波动
    data['high'] = data['close'] * (1 + daily_range/2)
    data['low'] = data['close'] * (1 - daily_range/2)
    
    # 确保开盘价在高低价之间
    data['open'] = np.clip(data['open'], data['low'], data['high'])
    
    # 生成成交量
    data['volume'] = np.random.randint(1000000, 10000000, periods)
    
    return data

def test_indicator_calculation(indicator_name, test_data):
    """测试单个指标的计算"""
    print(f"\n{'='*50}")
    print(f"测试指标: {indicator_name}")
    print(f"{'='*50}")
    
    try:
        # 创建指标实例
        indicator = indicator_registry.create_indicator(indicator_name)
        if not indicator:
            print(f"❌ 无法创建指标实例: {indicator_name}")
            return False
        
        print(f"✅ 成功创建指标实例")
        
        # 检查必需的列
        required_cols = getattr(indicator, 'required_columns', [])
        print(f"📋 必需列: {required_cols}")
        print(f"📋 可用列: {list(test_data.columns)}")
        
        missing_cols = [col for col in required_cols if col not in test_data.columns]
        if missing_cols:
            print(f"❌ 缺少必需列: {missing_cols}")
            return False
        
        # 尝试计算指标
        print("🔄 开始计算指标...")
        result = indicator.calculate(test_data)
        
        if result is None:
            print("❌ 计算返回None")
            return False
        
        if result.empty:
            print("❌ 计算返回空DataFrame")
            return False
        
        print(f"✅ 计算成功，返回 {len(result)} 行 x {len(result.columns)} 列")
        print(f"📊 新增列: {[col for col in result.columns if col not in test_data.columns]}")
        
        # 检查是否有NaN值
        new_cols = [col for col in result.columns if col not in test_data.columns]
        if new_cols:
            for col in new_cols:
                nan_count = result[col].isna().sum()
                valid_count = len(result) - nan_count
                print(f"   {col}: {valid_count}/{len(result)} 有效值")
        
        # 测试get_pattern_info方法
        if hasattr(indicator, 'get_pattern_info'):
            print("✅ 具有get_pattern_info方法")
            try:
                pattern_info = indicator.get_pattern_info('TEST_PATTERN')
                print(f"   返回格式: {type(pattern_info)}")
                if isinstance(pattern_info, dict):
                    print(f"   包含键: {list(pattern_info.keys())}")
            except Exception as e:
                print(f"⚠️  get_pattern_info方法调用出错: {e}")
        else:
            print("❌ 缺少get_pattern_info方法")
        
        # 测试get_patterns方法
        if hasattr(indicator, 'get_patterns'):
            print("🔍 测试形态识别...")
            try:
                patterns = indicator.get_patterns(test_data)
                if patterns is not None:
                    print(f"✅ 形态识别成功，返回 {len(patterns)} 行")
                    pattern_cols = [col for col in patterns.columns if col not in test_data.columns]
                    if pattern_cols:
                        print(f"   形态列: {pattern_cols}")
                else:
                    print("⚠️  形态识别返回None")
            except Exception as e:
                print(f"❌ 形态识别出错: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 计算出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 开始指标计算诊断...")
    
    # 创建测试数据
    test_data = create_test_data(100)
    print(f"📊 测试数据: {len(test_data)} 行 x {len(test_data.columns)} 列")
    print(f"📅 时间范围: {test_data.index[0]} 到 {test_data.index[-1]}")
    print(f"💰 价格范围: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
    
    # 测试新集成的指标
    test_indicators = ['BIAS', 'CCI', 'Chaikin', 'DMI', 'EMV']
    
    results = {}
    for indicator_name in test_indicators:
        success = test_indicator_calculation(indicator_name, test_data)
        results[indicator_name] = success
    
    # 总结结果
    print(f"\n{'='*50}")
    print("📋 测试结果总结")
    print(f"{'='*50}")
    
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    print(f"✅ 成功: {len(successful)}/{len(test_indicators)} 个指标")
    for name in successful:
        print(f"   - {name}")
    
    if failed:
        print(f"❌ 失败: {len(failed)} 个指标")
        for name in failed:
            print(f"   - {name}")
    
    return len(successful) == len(test_indicators)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
