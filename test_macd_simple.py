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
    """测试不同参数对MACD指标的影响"""
    # 生成测试数据
    df = generate_test_data(500)
    
    # 定义多组参数进行测试
    parameter_sets = [
        {'name': '默认参数', 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        {'name': '快速响应', 'fast_period': 5, 'slow_period': 15, 'signal_period': 5},
        {'name': '长周期', 'fast_period': 24, 'slow_period': 52, 'signal_period': 18},
    ]
    
    # 为每组参数计算MACD并验证
    for params in parameter_sets:
        name = params.pop('name')
        print(f"\n--- 测试参数组: {name} ---")
        
        # 创建MACD实例
        macd_indicator = MACD(**params)
        
        # 1. 测试计算
        result_df = macd_indicator.calculate(df)
        assert isinstance(result_df, pd.DataFrame), f"[{name}] 计算结果应为DataFrame"
        assert 'macd' in result_df.columns, f"[{name}] 结果中应包含 'macd' 列"
        assert 'signal' in result_df.columns, f"[{name}] 结果中应包含 'signal' 列"
        
        # 2. 测试信号
        signals = macd_indicator.get_signals(result_df)
        assert isinstance(signals, dict), f"[{name}] 信号应为字典"
        assert 'buy_signal' in signals, f"[{name}] 信号中应包含 'buy_signal'"
        assert isinstance(signals['buy_signal'], pd.Series), f"[{name}] 买入信号应为Series"
        
        # 打印检测到的信号
        has_buy_signal = signals['buy_signal'].any()
        print(f"[{name}] 是否检测到买入信号: {'是' if has_buy_signal else '否'}")


def main():
    """主函数"""
    print("测试增强MACD指标...")
    test_enhanced_macd()
    
    print("\n测试参数调整对MACD指标的影响...")
    test_parameter_tuning()
    
    print("\n测试完成！")


if __name__ == "__main__":
    main() 