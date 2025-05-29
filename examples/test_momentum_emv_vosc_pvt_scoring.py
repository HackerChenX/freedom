#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Momentum、EMV、VOSC、PVT指标评分功能

验证新增的4个指标的评分和形态识别功能是否正常工作
"""

import sys
import os
import numpy as np
import pandas as pd

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.momentum import Momentum
from indicators.emv import EMV
from indicators.vosc import VOSC
from indicators.pvt import PVT


def generate_test_data(length=100):
    """
    生成测试数据
    
    Args:
        length: 数据长度
        
    Returns:
        pd.DataFrame: 包含OHLCV数据的DataFrame
    """
    np.random.seed(42)  # 固定随机种子确保结果可重现
    
    # 生成基础价格序列
    base_price = 100
    price_changes = np.random.normal(0, 0.02, length)  # 2%的日波动
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # 确保价格为正
    
    # 生成OHLC数据
    close_prices = np.array(prices)
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, length)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, length)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # 生成成交量数据
    base_volume = 1000000
    volume_changes = np.random.normal(0, 0.3, length)
    volumes = []
    for i, change in enumerate(volume_changes):
        if i == 0:
            volumes.append(base_volume)
        else:
            new_volume = volumes[-1] * (1 + change)
            volumes.append(max(new_volume, 1000))  # 确保成交量为正
    
    # 创建DataFrame
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    return data


def test_momentum_scoring():
    """测试Momentum指标评分功能"""
    print("=" * 60)
    print("测试Momentum指标评分功能")
    print("=" * 60)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建Momentum指标实例
    momentum = Momentum(period=10, signal_period=6)
    
    # 计算指标
    result = momentum.calculate(data)
    print(f"Momentum指标计算完成，数据长度: {len(result)}")
    
    # 测试评分功能
    scores = momentum.calculate_raw_score(data)
    print(f"评分计算完成，评分范围: {scores.min():.2f} - {scores.max():.2f}")
    print(f"平均评分: {scores.mean():.2f}")
    print(f"评分标准差: {scores.std():.2f}")
    
    # 测试形态识别
    patterns = momentum.identify_patterns(data)
    print(f"识别到的形态数量: {len(patterns)}")
    print("识别到的形态:")
    for pattern in patterns:
        print(f"  - {pattern}")
    
    # 显示最近几个评分
    print("\n最近5个交易日的评分:")
    recent_scores = scores.tail(5)
    for date, score in recent_scores.items():
        print(f"  {date.strftime('%Y-%m-%d')}: {score:.2f}")
    
    return True


def test_emv_scoring():
    """测试EMV指标评分功能"""
    print("\n" + "=" * 60)
    print("测试EMV指标评分功能")
    print("=" * 60)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建EMV指标实例
    emv = EMV(period=14, volume_scale=10000)
    
    # 计算指标
    result = emv.calculate(data)
    print(f"EMV指标计算完成，数据长度: {len(result)}")
    
    # 测试评分功能
    scores = emv.calculate_raw_score(data)
    print(f"评分计算完成，评分范围: {scores.min():.2f} - {scores.max():.2f}")
    print(f"平均评分: {scores.mean():.2f}")
    print(f"评分标准差: {scores.std():.2f}")
    
    # 测试形态识别
    patterns = emv.identify_patterns(data)
    print(f"识别到的形态数量: {len(patterns)}")
    print("识别到的形态:")
    for pattern in patterns:
        print(f"  - {pattern}")
    
    # 显示最近几个评分
    print("\n最近5个交易日的评分:")
    recent_scores = scores.tail(5)
    for date, score in recent_scores.items():
        print(f"  {date.strftime('%Y-%m-%d')}: {score:.2f}")
    
    return True


def test_vosc_scoring():
    """测试VOSC指标评分功能"""
    print("\n" + "=" * 60)
    print("测试VOSC指标评分功能")
    print("=" * 60)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建VOSC指标实例
    vosc = VOSC(short_period=12, long_period=26)
    
    # 计算指标
    result = vosc.calculate(data)
    print(f"VOSC指标计算完成，数据长度: {len(result)}")
    
    # 测试评分功能
    scores = vosc.calculate_raw_score(data)
    print(f"评分计算完成，评分范围: {scores.min():.2f} - {scores.max():.2f}")
    print(f"平均评分: {scores.mean():.2f}")
    print(f"评分标准差: {scores.std():.2f}")
    
    # 测试形态识别
    patterns = vosc.identify_patterns(data)
    print(f"识别到的形态数量: {len(patterns)}")
    print("识别到的形态:")
    for pattern in patterns:
        print(f"  - {pattern}")
    
    # 显示最近几个评分
    print("\n最近5个交易日的评分:")
    recent_scores = scores.tail(5)
    for date, score in recent_scores.items():
        print(f"  {date.strftime('%Y-%m-%d')}: {score:.2f}")
    
    return True


def test_pvt_scoring():
    """测试PVT指标评分功能"""
    print("\n" + "=" * 60)
    print("测试PVT指标评分功能")
    print("=" * 60)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建PVT指标实例
    pvt = PVT(ma_period=12)
    
    # 计算指标
    result = pvt.calculate(data)
    print(f"PVT指标计算完成，数据长度: {len(result)}")
    
    # 测试评分功能
    scores = pvt.calculate_raw_score(data)
    print(f"评分计算完成，评分范围: {scores.min():.2f} - {scores.max():.2f}")
    print(f"平均评分: {scores.mean():.2f}")
    print(f"评分标准差: {scores.std():.2f}")
    
    # 测试形态识别
    patterns = pvt.identify_patterns(data)
    print(f"识别到的形态数量: {len(patterns)}")
    print("识别到的形态:")
    for pattern in patterns:
        print(f"  - {pattern}")
    
    # 显示最近几个评分
    print("\n最近5个交易日的评分:")
    recent_scores = scores.tail(5)
    for date, score in recent_scores.items():
        print(f"  {date.strftime('%Y-%m-%d')}: {score:.2f}")
    
    return True


def test_all_indicators_summary():
    """测试所有指标的综合评分"""
    print("\n" + "=" * 60)
    print("所有指标综合评分测试")
    print("=" * 60)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建所有指标实例
    indicators = {
        'Momentum': Momentum(period=10, signal_period=6),
        'EMV': EMV(period=14, volume_scale=10000),
        'VOSC': VOSC(short_period=12, long_period=26),
        'PVT': PVT(ma_period=12)
    }
    
    # 计算所有指标的评分
    all_scores = {}
    all_patterns = {}
    
    for name, indicator in indicators.items():
        try:
            # 计算指标
            indicator.calculate(data)
            
            # 计算评分
            scores = indicator.calculate_raw_score(data)
            all_scores[name] = scores
            
            # 识别形态
            patterns = indicator.identify_patterns(data)
            all_patterns[name] = patterns
            
            print(f"{name:10} - 平均评分: {scores.mean():.2f}, 形态数量: {len(patterns)}")
            
        except Exception as e:
            print(f"{name:10} - 计算失败: {str(e)}")
    
    # 计算综合评分
    if all_scores:
        # 计算所有指标的平均评分
        combined_scores = pd.DataFrame(all_scores)
        overall_score = combined_scores.mean(axis=1)
        
        print(f"\n综合评分统计:")
        print(f"  平均综合评分: {overall_score.mean():.2f}")
        print(f"  综合评分范围: {overall_score.min():.2f} - {overall_score.max():.2f}")
        print(f"  综合评分标准差: {overall_score.std():.2f}")
        
        # 显示最近5天的综合评分
        print(f"\n最近5个交易日的综合评分:")
        recent_overall = overall_score.tail(5)
        for date, score in recent_overall.items():
            print(f"  {date.strftime('%Y-%m-%d')}: {score:.2f}")
    
    # 统计所有形态
    all_pattern_list = []
    for patterns in all_patterns.values():
        all_pattern_list.extend(patterns)
    
    unique_patterns = list(set(all_pattern_list))
    print(f"\n所有指标识别到的唯一形态数量: {len(unique_patterns)}")
    
    return True


def main():
    """主函数"""
    print("开始测试Momentum、EMV、VOSC、PVT指标评分功能")
    print("测试时间:", pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    try:
        # 测试各个指标
        test_momentum_scoring()
        test_emv_scoring()
        test_vosc_scoring()
        test_pvt_scoring()
        
        # 综合测试
        test_all_indicators_summary()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 