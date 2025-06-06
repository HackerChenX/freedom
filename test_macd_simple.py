#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单MACD指标测试脚本

用于测试增强后的MACD指标功能，不包含图形展示
"""

import pandas as pd
import numpy as np
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from indicators.macd import MACD
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_test_data(size=200):
    """
    生成测试数据
    
    Args:
        size: 数据点数量
        
    Returns:
        pd.DataFrame: 测试数据
    """
    # 生成模拟股票价格
    np.random.seed(42)
    
    # 基础价格趋势
    trend = np.linspace(0, 1, size) * 10
    
    # 添加周期性波动
    cycles = np.sin(np.linspace(0, 4 * np.pi, size)) * 5
    
    # 添加一些波动
    noise = np.random.normal(0, 1, size)
    
    # 合成价格
    close = 100 + trend + cycles + noise
    
    # 生成OHLC数据
    high = close + np.random.uniform(0, 2, size)
    low = close - np.random.uniform(0, 2, size)
    open_price = close - np.random.uniform(-1, 1, size)
    
    # 生成交易量数据
    volume = np.random.uniform(1000, 5000, size)
    
    # 创建日期索引
    date_rng = pd.date_range(start='2020-01-01', periods=size, freq='D')
    
    # 创建DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=date_rng)
    
    return df


def test_enhanced_macd():
    """测试增强MACD指标"""
    # 生成测试数据
    df = generate_test_data(300)
    
    # 创建MACD实例
    macd = MACD(
        fast_period=12,
        slow_period=26,
        signal_period=9,
        histogram_threshold=0.01,
        divergence_window=15,
        divergence_threshold=0.03,
        zero_line_sensitivity=0.001
    )
    
    # 计算指标
    print("计算MACD指标...")
    result = macd.calculate(df)
    
    # 输出结果的列名
    print("\nMACD计算结果包含以下列:")
    for col in result.columns:
        print(f"- {col}")
    
    # 获取信号
    print("\n检测信号...")
    signals = macd.get_signals(df)
    
    print("\n检测到的信号:")
    active_signals = []
    for signal_name, is_active in signals.items():
        if is_active:
            active_signals.append(signal_name)
            print(f"- {signal_name}")
    
    if not active_signals:
        print("- 没有检测到信号")
    
    # 计算得分
    print("\n计算MACD评分...")
    score = macd.calculate_raw_score(df)
    
    print(f"\n最新评分: {score.iloc[-1]:.2f}")
    
    print("\n最近10个评分:")
    for date, score_value in score.iloc[-10:].items():
        print(f"- {date.strftime('%Y-%m-%d')}: {score_value:.2f}")


def test_parameter_tuning():
    """测试参数调整对MACD指标的影响"""
    # 生成测试数据
    df = generate_test_data(300)
    
    # 定义不同的参数组合
    parameter_sets = [
        {'fast_period': 8, 'slow_period': 17, 'signal_period': 9, 'name': 'Fast MACD (8,17,9)'},
        {'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'name': 'Standard MACD (12,26,9)'},
        {'fast_period': 16, 'slow_period': 35, 'signal_period': 9, 'name': 'Slow MACD (16,35,9)'}
    ]
    
    print("\n参数调整对MACD指标的影响:")
    
    # 为每组参数计算MACD
    for params in parameter_sets:
        print(f"\n测试 {params['name']}...")
        
        # 创建MACD实例
        macd_indicator = MACD(
            fast_period=params['fast_period'],
            slow_period=params['slow_period'],
            signal_period=params['signal_period']
        )
        
        # 计算指标
        result = macd_indicator.calculate(df)
        
        # 获取信号
        signals = macd_indicator.get_signals(df)
        
        # 计算得分
        score = macd_indicator.calculate_raw_score(df)
        
        # 输出信号统计
        active_signals = []
        for signal_name, is_active in signals.items():
            if is_active:
                active_signals.append(signal_name)
        
        print(f"- 信号数量: {len(active_signals)}")
        print(f"- 最新评分: {score.iloc[-1]:.2f}")
        
        # 计算交叉次数
        golden_crosses = result['golden_cross'].sum()
        death_crosses = result['death_cross'].sum()
        zero_up_crosses = result.get('macd_cross_zero_up', pd.Series([False] * len(result))).sum()
        zero_down_crosses = result.get('macd_cross_zero_down', pd.Series([False] * len(result))).sum()
        
        print(f"- 金叉次数: {golden_crosses}")
        print(f"- 死叉次数: {death_crosses}")
        print(f"- 零轴向上穿越次数: {zero_up_crosses}")
        print(f"- 零轴向下穿越次数: {zero_down_crosses}")
        
        # 计算背离和双顶双底次数
        bullish_divergences = result.get('bullish_divergence', pd.Series([False] * len(result))).sum()
        bearish_divergences = result.get('bearish_divergence', pd.Series([False] * len(result))).sum()
        double_bottoms = result.get('double_bottom', pd.Series([False] * len(result))).sum()
        double_tops = result.get('double_top', pd.Series([False] * len(result))).sum()
        
        print(f"- 底背离次数: {bullish_divergences}")
        print(f"- 顶背离次数: {bearish_divergences}")
        print(f"- 双底次数: {double_bottoms}")
        print(f"- 双顶次数: {double_tops}")


def main():
    """主函数"""
    print("测试增强MACD指标...")
    test_enhanced_macd()
    
    print("\n测试参数调整对MACD指标的影响...")
    test_parameter_tuning()
    
    print("\n测试完成！")


if __name__ == "__main__":
    main() 