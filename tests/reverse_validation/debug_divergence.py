#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
调试RSI背离形态生成

分析为什么背离形态无法正确识别，并优化数据生成
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from smart_pattern_generator import SmartPatternGenerator
from technical_indicators import TechnicalIndicators


def debug_rsi_divergence():
    """调试RSI背离形态"""
    print("调试RSI背离形态生成...")

    generator = SmartPatternGenerator()
    indicators = TechnicalIndicators()

    # 生成背离数据
    data = generator.generate_rsi_divergence_data_v2()
    rsi = indicators.calculate_rsi(data)
    close_prices = data['close']

    print(f"数据点数: {len(data)}")
    print(f"价格范围: {close_prices.min():.2f} - {close_prices.max():.2f}")
    print(f"RSI范围: {rsi.min():.2f} - {rsi.max():.2f}")

    # 分析前后半段
    mid_point = len(data) // 2
    first_half_price = close_prices.iloc[:mid_point]
    second_half_price = close_prices.iloc[mid_point:]
    first_half_rsi = rsi.iloc[:mid_point]
    second_half_rsi = rsi.iloc[mid_point:]

    print(f"\n前半段分析:")
    print(f"  价格: {first_half_price.iloc[0]:.2f} -> {first_half_price.iloc[-1]:.2f}")
    print(f"  RSI: {first_half_rsi.iloc[0]:.2f} -> {first_half_rsi.iloc[-1]:.2f}")
    print(f"  最高价格: {first_half_price.max():.2f}")
    print(f"  最高RSI: {first_half_rsi.max():.2f}")

    print(f"\n后半段分析:")
    print(f"  价格: {second_half_price.iloc[0]:.2f} -> {second_half_price.iloc[-1]:.2f}")
    print(f"  RSI: {second_half_rsi.iloc[0]:.2f} -> {second_half_rsi.iloc[-1]:.2f}")
    print(f"  最高价格: {second_half_price.max():.2f}")
    print(f"  最高RSI: {second_half_rsi.max():.2f}")

    # 检查背离条件
    price_higher = second_half_price.iloc[-1] > first_half_price.iloc[-1]
    rsi_lower = second_half_rsi.iloc[-1] < first_half_rsi.iloc[-1]

    # 更准确的背离检测：比较峰值
    first_peak_price = first_half_price.max()
    second_peak_price = second_half_price.max()
    first_peak_rsi = first_half_rsi.max()
    second_peak_rsi = second_half_rsi.max()

    peak_price_higher = second_peak_price > first_peak_price
    peak_rsi_lower = second_peak_rsi < first_peak_rsi

    print(f"\n背离检测结果:")
    print(f"  终点价格创新高: {price_higher}")
    print(f"  终点RSI未创新高: {rsi_lower}")
    print(f"  峰值价格创新高: {peak_price_higher}")
    print(f"  峰值RSI未创新高: {peak_rsi_lower}")

    print(f"\n当前验证逻辑结果: {'✅ 背离' if (price_higher and rsi_lower) else '❌ 无背离'}")
    print(f"峰值验证逻辑结果: {'✅ 背离' if (peak_price_higher and peak_rsi_lower) else '❌ 无背离'}")

    return data, rsi


if __name__ == '__main__':
    debug_rsi_divergence()