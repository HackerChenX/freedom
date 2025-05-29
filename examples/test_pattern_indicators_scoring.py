#!/usr/bin/env python3
"""
测试形态识别指标评分功能

测试K线形态识别、高级K线形态识别、斐波那契工具和艾略特波浪理论指标的评分功能
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.pattern.candlestick_patterns import CandlestickPatterns
from indicators.pattern.advanced_candlestick_patterns import AdvancedCandlestickPatterns
from indicators.fibonacci_tools import FibonacciTools
from indicators.elliott_wave import ElliottWave
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_test_data(days: int = 100) -> pd.DataFrame:
    """
    生成测试用的OHLCV数据
    
    Args:
        days: 生成的天数
        
    Returns:
        pd.DataFrame: 测试数据
    """
    # 生成日期索引
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # 生成基础价格走势（带趋势和波动）
    np.random.seed(42)  # 确保结果可重现
    
    # 生成价格走势
    base_price = 100.0
    price_changes = np.random.normal(0, 0.02, days)  # 2%的日波动
    
    # 添加趋势
    trend = np.linspace(0, 0.3, days)  # 30%的总体上涨趋势
    price_changes += trend / days
    
    # 计算累积价格
    prices = [base_price]
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = prices[1:]  # 移除初始价格
    
    # 生成OHLC数据
    data = []
    for i, price in enumerate(prices):
        # 生成日内波动
        daily_volatility = np.random.uniform(0.01, 0.03)  # 1-3%的日内波动
        
        # 开盘价（基于前一日收盘价）
        if i == 0:
            open_price = price
        else:
            gap = np.random.normal(0, 0.005)  # 0.5%的跳空
            open_price = prices[i-1] * (1 + gap)
        
        # 生成高低价
        high_price = max(open_price, price) * (1 + np.random.uniform(0, daily_volatility))
        low_price = min(open_price, price) * (1 - np.random.uniform(0, daily_volatility))
        
        # 确保价格逻辑正确
        high_price = max(high_price, open_price, price)
        low_price = min(low_price, open_price, price)
        
        # 生成成交量（与价格变化相关）
        price_change = abs(price - open_price) / open_price if open_price > 0 else 0
        base_volume = 1000000
        volume = base_volume * (1 + price_change * 5) * np.random.uniform(0.5, 2.0)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


def test_candlestick_patterns_scoring():
    """测试K线形态识别指标评分功能"""
    print("\n=== 测试K线形态识别指标评分功能 ===")
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建指标实例
    indicator = CandlestickPatterns()
    
    try:
        # 计算评分
        score_result = indicator.calculate_raw_score(data)
        print(f"✓ K线形态识别评分计算成功")
        print(f"  评分范围: {score_result['score'].min():.2f} - {score_result['score'].max():.2f}")
        print(f"  最新评分: {score_result['score'].iloc[-1]:.2f}")
        
        # 识别形态
        patterns = indicator.identify_patterns(data)
        print(f"✓ 识别到 {len(patterns)} 个形态:")
        for pattern in patterns[:5]:  # 显示前5个形态
            print(f"    - {pattern}")
        if len(patterns) > 5:
            print(f"    ... 还有 {len(patterns) - 5} 个形态")
        
        return True
        
    except Exception as e:
        print(f"✗ K线形态识别评分测试失败: {e}")
        return False


def test_advanced_candlestick_patterns_scoring():
    """测试高级K线形态识别指标评分功能"""
    print("\n=== 测试高级K线形态识别指标评分功能 ===")
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建指标实例
    indicator = AdvancedCandlestickPatterns()
    
    try:
        # 计算评分
        score_result = indicator.calculate_raw_score(data)
        print(f"✓ 高级K线形态识别评分计算成功")
        print(f"  评分范围: {score_result['score'].min():.2f} - {score_result['score'].max():.2f}")
        print(f"  最新评分: {score_result['score'].iloc[-1]:.2f}")
        
        # 识别形态
        patterns = indicator.identify_patterns(data)
        print(f"✓ 识别到 {len(patterns)} 个高级形态:")
        for pattern in patterns[:5]:  # 显示前5个形态
            print(f"    - {pattern}")
        if len(patterns) > 5:
            print(f"    ... 还有 {len(patterns) - 5} 个形态")
        
        return True
        
    except Exception as e:
        print(f"✗ 高级K线形态识别评分测试失败: {e}")
        return False


def test_fibonacci_tools_scoring():
    """测试斐波那契工具指标评分功能"""
    print("\n=== 测试斐波那契工具指标评分功能 ===")
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建指标实例
    indicator = FibonacciTools()
    
    try:
        # 计算评分
        score_result = indicator.calculate_raw_score(data)
        print(f"✓ 斐波那契工具评分计算成功")
        print(f"  评分范围: {score_result['score'].min():.2f} - {score_result['score'].max():.2f}")
        print(f"  最新评分: {score_result['score'].iloc[-1]:.2f}")
        
        # 识别形态
        patterns = indicator.identify_patterns(data)
        print(f"✓ 识别到 {len(patterns)} 个斐波那契形态:")
        for pattern in patterns[:5]:  # 显示前5个形态
            print(f"    - {pattern}")
        if len(patterns) > 5:
            print(f"    ... 还有 {len(patterns) - 5} 个形态")
        
        return True
        
    except Exception as e:
        print(f"✗ 斐波那契工具评分测试失败: {e}")
        return False


def test_elliott_wave_scoring():
    """测试艾略特波浪理论指标评分功能"""
    print("\n=== 测试艾略特波浪理论指标评分功能 ===")
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建指标实例
    indicator = ElliottWave()
    
    try:
        # 计算评分
        score_result = indicator.calculate_raw_score(data)
        print(f"✓ 艾略特波浪理论评分计算成功")
        print(f"  评分范围: {score_result['score'].min():.2f} - {score_result['score'].max():.2f}")
        print(f"  最新评分: {score_result['score'].iloc[-1]:.2f}")
        
        # 识别形态
        patterns = indicator.identify_patterns(data)
        print(f"✓ 识别到 {len(patterns)} 个波浪形态:")
        for pattern in patterns[:5]:  # 显示前5个形态
            print(f"    - {pattern}")
        if len(patterns) > 5:
            print(f"    ... 还有 {len(patterns) - 5} 个形态")
        
        return True
        
    except Exception as e:
        print(f"✗ 艾略特波浪理论评分测试失败: {e}")
        return False


def main():
    """主函数"""
    print("开始测试形态识别指标评分功能...")
    
    # 测试结果统计
    test_results = []
    
    # 测试各个指标
    test_results.append(test_candlestick_patterns_scoring())
    test_results.append(test_advanced_candlestick_patterns_scoring())
    test_results.append(test_fibonacci_tools_scoring())
    test_results.append(test_elliott_wave_scoring())
    
    # 输出测试总结
    print(f"\n=== 测试总结 ===")
    passed = sum(test_results)
    total = len(test_results)
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("✓ 所有形态识别指标评分功能测试通过！")
    else:
        print("✗ 部分测试失败，请检查相关指标实现")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 