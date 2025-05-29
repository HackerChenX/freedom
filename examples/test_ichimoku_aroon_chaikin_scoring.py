#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Ichimoku、Aroon、Chaikin指标评分功能

测试新实现的三个高级指标的评分和形态识别功能
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.ichimoku import Ichimoku
from indicators.aroon import Aroon
from indicators.chaikin import Chaikin


def generate_test_data(periods=100):
    """
    生成测试用的OHLCV数据
    
    Args:
        periods: 数据周期数
        
    Returns:
        pd.DataFrame: 包含OHLCV数据的DataFrame
    """
    np.random.seed(42)  # 固定随机种子以便复现
    
    # 生成日期索引
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    # 生成基础价格走势（带趋势的随机游走）
    base_price = 100
    price_changes = np.random.normal(0.001, 0.02, periods)  # 平均上涨0.1%，波动2%
    
    # 添加一些趋势性
    trend = np.linspace(0, 0.3, periods)  # 30%的总体上涨趋势
    price_changes += trend / periods
    
    # 计算累积价格
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # 生成OHLC数据
    opens = prices * (1 + np.random.normal(0, 0.005, periods))
    highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, periods)))
    lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, periods)))
    closes = prices
    
    # 生成成交量（与价格变化相关）
    volume_base = 1000000
    volume_changes = np.abs(price_changes) * 2 + np.random.normal(0, 0.3, periods)
    volumes = volume_base * (1 + volume_changes)
    volumes = np.maximum(volumes, volume_base * 0.1)  # 确保最小成交量
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes.astype(int)
    }, index=dates)
    
    return df


def test_ichimoku_scoring():
    """测试Ichimoku指标评分功能"""
    print("=" * 60)
    print("测试Ichimoku（一目均衡表）指标评分功能")
    print("=" * 60)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建Ichimoku指标实例
    ichimoku = Ichimoku()
    
    # 计算指标
    result_df = ichimoku.calculate(data)
    
    # 计算评分
    score_result = ichimoku.calculate_score(data)
    scores = score_result['final_score']
    patterns = score_result['patterns']
    
    # 输出结果
    print(f"数据周期: {len(data)}天")
    print(f"Ichimoku指标计算完成")
    print(f"评分范围: {scores.min():.2f} - {scores.max():.2f}")
    print(f"平均评分: {scores.mean():.2f}")
    print(f"最终评分: {scores.iloc[-1]:.2f}")
    print(f"识别形态数量: {len(patterns)}")
    print(f"识别的形态: {', '.join(patterns)}")
    print(f"市场环境: {score_result['market_environment'].value}")
    print(f"置信度: {score_result['confidence']:.2f}")
    
    # 显示最近几天的详细数据
    print("\n最近5天的指标数据:")
    recent_data = result_df[['close', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']].tail()
    print(recent_data.round(2))
    
    print(f"\n最近5天的评分:")
    recent_scores = scores.tail()
    for date, score in recent_scores.items():
        print(f"{date.strftime('%Y-%m-%d')}: {score:.2f}")
    
    return scores, patterns


def test_aroon_scoring():
    """测试Aroon指标评分功能"""
    print("\n" + "=" * 60)
    print("测试Aroon（阿隆指标）指标评分功能")
    print("=" * 60)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建Aroon指标实例
    aroon = Aroon()
    
    # 计算指标
    result_df = aroon.calculate(data)
    
    # 计算评分
    score_result = aroon.calculate_score(data)
    scores = score_result['final_score']
    patterns = score_result['patterns']
    
    # 输出结果
    print(f"数据周期: {len(data)}天")
    print(f"Aroon指标计算完成")
    print(f"评分范围: {scores.min():.2f} - {scores.max():.2f}")
    print(f"平均评分: {scores.mean():.2f}")
    print(f"最终评分: {scores.iloc[-1]:.2f}")
    print(f"识别形态数量: {len(patterns)}")
    print(f"识别的形态: {', '.join(patterns)}")
    print(f"市场环境: {score_result['market_environment'].value}")
    print(f"置信度: {score_result['confidence']:.2f}")
    
    # 显示最近几天的详细数据
    print("\n最近5天的指标数据:")
    recent_data = result_df[['close', 'aroon_up', 'aroon_down', 'aroon_oscillator']].tail()
    print(recent_data.round(2))
    
    print(f"\n最近5天的评分:")
    recent_scores = scores.tail()
    for date, score in recent_scores.items():
        print(f"{date.strftime('%Y-%m-%d')}: {score:.2f}")
    
    return scores, patterns


def test_chaikin_scoring():
    """测试Chaikin指标评分功能"""
    print("\n" + "=" * 60)
    print("测试Chaikin（蔡金指标）指标评分功能")
    print("=" * 60)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建Chaikin指标实例
    chaikin = Chaikin()
    
    # 计算指标
    result_df = chaikin.calculate(data)
    
    # 计算评分
    score_result = chaikin.calculate_score(data)
    scores = score_result['final_score']
    patterns = score_result['patterns']
    
    # 输出结果
    print(f"数据周期: {len(data)}天")
    print(f"Chaikin指标计算完成")
    print(f"评分范围: {scores.min():.2f} - {scores.max():.2f}")
    print(f"平均评分: {scores.mean():.2f}")
    print(f"最终评分: {scores.iloc[-1]:.2f}")
    print(f"识别形态数量: {len(patterns)}")
    print(f"识别的形态: {', '.join(patterns)}")
    print(f"市场环境: {score_result['market_environment'].value}")
    print(f"置信度: {score_result['confidence']:.2f}")
    
    # 显示最近几天的详细数据
    print("\n最近5天的指标数据:")
    recent_data = result_df[['close', 'ad_line', 'chaikin_oscillator', 'chaikin_signal']].tail()
    print(recent_data.round(2))
    
    print(f"\n最近5天的评分:")
    recent_scores = scores.tail()
    for date, score in recent_scores.items():
        print(f"{date.strftime('%Y-%m-%d')}: {score:.2f}")
    
    return scores, patterns


def test_comprehensive_scoring():
    """综合测试所有三个指标"""
    print("\n" + "=" * 60)
    print("综合测试Ichimoku、Aroon、Chaikin指标评分")
    print("=" * 60)
    
    # 生成测试数据
    data = generate_test_data(100)
    
    # 创建指标实例
    indicators = {
        'Ichimoku': Ichimoku(),
        'Aroon': Aroon(),
        'Chaikin': Chaikin()
    }
    
    # 计算所有指标的评分
    all_scores = {}
    all_patterns = {}
    all_score_results = {}
    
    for name, indicator in indicators.items():
        try:
            # 计算指标
            indicator.calculate(data)
            
            # 计算评分
            score_result = indicator.calculate_score(data)
            scores = score_result['final_score']
            patterns = score_result['patterns']
            
            all_scores[name] = scores
            all_patterns[name] = patterns
            all_score_results[name] = score_result
            
            print(f"{name}: 评分 {scores.iloc[-1]:.2f}, 形态 {len(patterns)}个, 置信度 {score_result['confidence']:.2f}")
            
        except Exception as e:
            print(f"{name}: 计算失败 - {str(e)}")
    
    # 计算综合评分
    if all_scores:
        # 简单平均
        combined_scores = pd.DataFrame(all_scores).mean(axis=1)
        
        print(f"\n综合评分统计:")
        print(f"综合评分范围: {combined_scores.min():.2f} - {combined_scores.max():.2f}")
        print(f"综合平均评分: {combined_scores.mean():.2f}")
        print(f"最终综合评分: {combined_scores.iloc[-1]:.2f}")
        
        # 统计所有形态
        total_patterns = []
        for patterns in all_patterns.values():
            total_patterns.extend(patterns)
        
        print(f"总识别形态数: {len(total_patterns)}")
        print(f"不重复形态数: {len(set(total_patterns))}")
        
        # 显示各指标最近评分对比
        print(f"\n最近5天各指标评分对比:")
        recent_scores_df = pd.DataFrame(all_scores).tail()
        recent_scores_df['综合'] = combined_scores.tail()
        print(recent_scores_df.round(2))
        
        # 显示市场环境分析
        print(f"\n市场环境分析:")
        for name, score_result in all_score_results.items():
            print(f"{name}: {score_result['market_environment'].value}")
        
        return combined_scores, total_patterns
    
    return None, []


def main():
    """主函数"""
    print("开始测试Ichimoku、Aroon、Chaikin指标评分功能")
    print("测试时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # 测试各个指标
        ichimoku_scores, ichimoku_patterns = test_ichimoku_scoring()
        aroon_scores, aroon_patterns = test_aroon_scoring()
        chaikin_scores, chaikin_patterns = test_chaikin_scoring()
        
        # 综合测试
        combined_scores, all_patterns = test_comprehensive_scoring()
        
        # 总结
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print(f"✓ Ichimoku指标: 平均评分 {ichimoku_scores.mean():.2f}, 识别 {len(ichimoku_patterns)} 种形态")
        print(f"✓ Aroon指标: 平均评分 {aroon_scores.mean():.2f}, 识别 {len(aroon_patterns)} 种形态")
        print(f"✓ Chaikin指标: 平均评分 {chaikin_scores.mean():.2f}, 识别 {len(chaikin_patterns)} 种形态")
        
        if combined_scores is not None:
            print(f"✓ 综合评分: 平均 {combined_scores.mean():.2f}, 总形态 {len(all_patterns)} 种")
        
        print("\n所有指标评分功能测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 