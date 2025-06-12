#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强MACD指标测试脚本

用于测试增强后的MACD指标功能和性能
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 将项目根目录添加到sys.path
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

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
    """测试增强版MACD指标的计算、信号和得分"""
    # 准备测试数据
    df = generate_test_data()
    
    # 创建MACD实例
    macd = MACD()
    
    # 1. 测试计算
    result = macd.calculate(df)
    assert isinstance(result, pd.DataFrame), "计算结果应为DataFrame"
    assert 'macd' in result.columns, "结果中应包含 'macd' 列"
    assert 'signal' in result.columns, "结果中应包含 'signal' 列"
    assert 'hist' in result.columns, "结果中应包含 'hist' 列"
    
    # 2. 测试信号生成
    signals = macd.get_signals(result)
    assert isinstance(signals, dict), "信号应为字典"
    assert 'buy_signal' in signals, "信号中应包含 'buy_signal'"
    assert 'sell_signal' in signals, "信号中应包含 'sell_signal'"
    assert isinstance(signals['buy_signal'], pd.Series), "买入信号应为Series"
    assert isinstance(signals['sell_signal'], pd.Series), "卖出信号应为Series"

    print("\n--- 检测到的信号 ---")
    has_signals = False
    for signal_name, signal_series in signals.items():
        if isinstance(signal_series, pd.Series) and signal_series.any():
            print(f"- 检测到 '{signal_name}'")
            has_signals = True
    if not has_signals:
        print("- 未检测到任何信号")
    print("--------------------")

    # 3. 测试得分计算
    score = macd.calculate_raw_score(result)
    assert isinstance(score, pd.Series), "得分应为Series"
    assert not score.empty, "得分Series不应为空"


def test_parameter_tuning():
    """测试参数调整对MACD指标的影响"""
    # 生成测试数据
    df = generate_test_data(300)
    
    # 定义不同的参数组合
    parameter_sets = [
        {'fast_period': 8, 'slow_period': 17, 'signal_period': 9},
        {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        {'fast_period': 16, 'slow_period': 35, 'signal_period': 9}
    ]
    
    # 为每组参数计算MACD
    for params in parameter_sets:
        macd_indicator = MACD(**params)
        result_df = macd_indicator.calculate(df)
        assert isinstance(result_df, pd.DataFrame)
        assert 'macd' in result_df.columns
        assert 'signal' in result_df.columns
        assert 'hist' in result_df.columns


def main():
    logger.info("开始进行MACD指标计算测试...")
    test_enhanced_macd()
    test_parameter_tuning()
    logger.info("MACD指标计算测试完成。")


if __name__ == '__main__':
    main() 