#!/usr/bin/env python3
"""
调试RSI形态生成和检测
"""

import sys
import os
import pandas as pd
import numpy as np
import random

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import get_logger
from components.stockinfo_compatible_data_generator import StockInfoCompatibleDataGenerator
from indicators.rsi import RSI

def debug_rsi_pattern():
    logger = get_logger(__name__)
    print("🔍 调试RSI形态生成")
    
    # 初始化生成器和RSI计算器
    generator = StockInfoCompatibleDataGenerator()
    rsi_calculator = RSI()
    
    # Test OVERSOLD pattern
    print("\n📊 测试 OVERSOLD 形态:")
    oversold_data = generator.generate_stockinfo_compatible_data(
        indicator_name="RSI",
        pattern_type="OVERSOLD",
        stock_code="DEBUG_RSI_OVERSOLD",
        history_days=150
    )
    
    # Calculate actual RSI value
    rsi_result = rsi_calculator.calculate(oversold_data)
    # RSI calculation result can be DataFrame or dict
    if isinstance(rsi_result, pd.DataFrame):
        latest_rsi = rsi_result.iloc[-1]['rsi_14'] if 'rsi_14' in rsi_result.columns else None
    elif isinstance(rsi_result, dict):
        latest_rsi = rsi_result.get('rsi_14', rsi_result.get('RSI', None))
        if isinstance(latest_rsi, (list, np.ndarray)):
            latest_rsi = latest_rsi[-1] if len(latest_rsi) > 0 else None
        elif isinstance(latest_rsi, pd.Series):
            latest_rsi = latest_rsi.iloc[-1] if len(latest_rsi) > 0 else None
    else:
        latest_rsi = None

    print(f"Generated close price change: {oversold_data['close'].iloc[0]:.2f} -> {oversold_data['close'].iloc[-1]:.2f}")
    print(f"Price change magnitude: {((oversold_data['close'].iloc[-1] / oversold_data['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"Latest RSI value: {latest_rsi:.2f}" if latest_rsi is not None else "无法获取RSI值")
    print(f"预期: RSI < 35 (OVERSOLD)")
    print(f"实际结果: {'✅ 正确' if latest_rsi is not None and latest_rsi < 35 else '❌ 错误'}")

    # 显示价格变化趋势
    print("\n📈 OVERSOLD 价格变化趋势分析:")
    prices = oversold_data['close'].values
    for i in range(0, len(prices), len(prices)//10):
        if i + 1 < len(prices):
            change = ((prices[i+1] / prices[i]) - 1) * 100 if i > 0 else 0
            print(f"Day {i}: {prices[i]:.2f} ({change:+.2f}%)")
    
    # 检查最后10天的价格应该严格递减
    print("\n🔍 检查OVERSOLD最后10天价格是否严格递减:")
    last_10_prices = oversold_data['close'].values[-10:]
    for i in range(len(last_10_prices)-1):
        current_price = last_10_prices[i]
        next_price = last_10_prices[i+1]
        trend = "📉" if next_price < current_price else "📈" if next_price > current_price else "➡️"
        print(f"Day {140+i} -> {141+i}: {current_price:.2f} -> {next_price:.2f} {trend}")

    # Test OVERBOUGHT pattern
    print("\n📊 测试 OVERBOUGHT 形态:")
    overbought_data = generator.generate_stockinfo_compatible_data(
        indicator_name="RSI",
        pattern_type="OVERBOUGHT",
        stock_code="DEBUG_RSI_OVERBOUGHT",
        history_days=150
    )

    # Calculate actual RSI value
    rsi_result = rsi_calculator.calculate(overbought_data)
    # RSI calculation result can be DataFrame or dict
    if isinstance(rsi_result, pd.DataFrame):
        latest_rsi = rsi_result.iloc[-1]['rsi_14'] if 'rsi_14' in rsi_result.columns else None
    elif isinstance(rsi_result, dict):
        latest_rsi = rsi_result.get('rsi_14', rsi_result.get('RSI', None))
        if isinstance(latest_rsi, (list, np.ndarray)):
            latest_rsi = latest_rsi[-1] if len(latest_rsi) > 0 else None
        elif isinstance(latest_rsi, pd.Series):
            latest_rsi = latest_rsi.iloc[-1] if len(latest_rsi) > 0 else None
    else:
        latest_rsi = None

    print(f"Generated close price change: {overbought_data['close'].iloc[0]:.2f} -> {overbought_data['close'].iloc[-1]:.2f}")
    print(f"Price change magnitude: {((overbought_data['close'].iloc[-1] / overbought_data['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"Latest RSI value: {latest_rsi:.2f}" if latest_rsi is not None else "无法获取RSI值")
    print(f"预期: RSI > 65 (OVERBOUGHT)")
    print(f"实际结果: {'✅ 正确' if latest_rsi is not None and latest_rsi > 65 else '❌ 错误'}")

    # 显示价格变化趋势
    print("\n📈 OVERBOUGHT 价格变化趋势分析:")
    prices = overbought_data['close'].values
    for i in range(0, len(prices), len(prices)//10):
        if i + 1 < len(prices):
            change = ((prices[i+1] / prices[i]) - 1) * 100 if i > 0 else 0
            print(f"Day {i}: {prices[i]:.2f} ({change:+.2f}%)")

    # 详细分析最后几天的 RSI 变化
    print("\n🔍 详细分析最后10天RSI变化 (OVERBOUGHT):")
    rsi_series = rsi_result.get('rsi_14', rsi_result.get('RSI', None))
    if isinstance(rsi_series, pd.Series):
        for i in range(max(0, len(rsi_series) - 10), len(rsi_series)):
            price = overbought_data['close'].iloc[i]
            rsi_val = rsi_series.iloc[i]
            print(f"Day {i}: Price={price:.2f}, RSI={rsi_val:.2f}")
    
    # 检查最后10天的价格应该严格递增
    print("\n🔍 检查OVERBOUGHT最后10天价格是否严格递增:")
    last_10_prices = overbought_data['close'].values[-10:]
    for i in range(len(last_10_prices)-1):
        current_price = last_10_prices[i]
        next_price = last_10_prices[i+1]
        trend = "📈" if next_price > current_price else "📉" if next_price < current_price else "➡️"
        print(f"Day {140+i} -> {141+i}: {current_price:.2f} -> {next_price:.2f} {trend}")

if __name__ == "__main__":
    debug_rsi_pattern() 