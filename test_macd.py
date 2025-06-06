#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强MACD指标测试脚本

用于测试增强后的MACD指标功能和性能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
    result = macd.calculate(df)
    
    # 获取信号
    signals = macd.get_signals(df)
    
    print("检测到的信号:")
    for signal_name, is_active in signals.items():
        if is_active:
            print(f"- {signal_name}")
    
    # 计算得分
    score = macd.calculate_raw_score(df)
    
    # 绘制结果
    plot_results(df, result, score)


def plot_results(data, result, score):
    """
    绘制结果
    
    Args:
        data: 原始数据
        result: MACD计算结果
        score: 得分数据
    """
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.1)
    
    # 绘制价格图
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data.index, data['close'], label='Close Price')
    ax1.set_title('Enhanced MACD Indicator Test')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    
    # 添加买入和卖出信号
    buy_signals = result[result['golden_cross'] | result['macd_cross_zero_up'] | 
                         result['bullish_divergence'] | result['double_bottom']].index
    sell_signals = result[result['death_cross'] | result['macd_cross_zero_down'] | 
                          result['bearish_divergence'] | result['double_top']].index
    
    ax1.scatter(buy_signals, data.loc[buy_signals, 'close'], marker='^', color='g', s=100, label='Buy Signal')
    ax1.scatter(sell_signals, data.loc[sell_signals, 'close'], marker='v', color='r', s=100, label='Sell Signal')
    
    # 标注底背离和顶背离
    bullish_div = result[result['bullish_divergence']].index
    bearish_div = result[result['bearish_divergence']].index
    
    for idx in bullish_div:
        ax1.annotate('底背离', (idx, data.loc[idx, 'close']), 
                    xytext=(0, 30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='g'))
    
    for idx in bearish_div:
        ax1.annotate('顶背离', (idx, data.loc[idx, 'close']), 
                    xytext=(0, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='r'))
    
    ax1.legend()
    
    # 绘制MACD线
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(result.index, result['macd'], label='MACD')
    ax2.plot(result.index, result['signal'], label='Signal')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax2.set_ylabel('MACD Lines')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 绘制MACD柱状图
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.bar(result.index, result['hist'], label='MACD Histogram')
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax3.set_ylabel('MACD Histogram')
    ax3.grid(True, alpha=0.3)
    
    # 标注双顶和双底
    double_tops = result[result['double_top']].index
    double_bottoms = result[result['double_bottom']].index
    
    for idx in double_tops:
        ax3.annotate('双顶', (idx, result.loc[idx, 'hist']), 
                    xytext=(0, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='r'))
    
    for idx in double_bottoms:
        ax3.annotate('双底', (idx, result.loc[idx, 'hist']), 
                    xytext=(0, -20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='g'))
    
    # 绘制得分
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(score.index, score, label='MACD Score')
    ax4.axhline(y=50, color='k', linestyle='-', alpha=0.2)
    ax4.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax4.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax4.set_ylim(0, 100)
    ax4.set_ylabel('Score (0-100)')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)
    
    # 隐藏上面图形的x轴标签
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join(root_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'enhanced_macd_test.png'), dpi=300)
    
    # 显示图表
    plt.show()


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
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # 绘制价格图
    axes[0].plot(df.index, df['close'], label='Close Price')
    axes[0].set_title('MACD Parameter Tuning Comparison')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    colors = ['r', 'g', 'b']
    
    # 为每组参数计算MACD并绘制
    for i, params in enumerate(parameter_sets):
        macd_indicator = MACD(**params)
        result = macd_indicator.calculate(df)
        
        # 获取信号
        signals = macd_indicator.get_signals(df)
        
        # 绘制MACD和Signal线
        axes[1].plot(result.index, result['macd'], color=colors[i], label=f"MACD - {params['name']}")
        axes[1].plot(result.index, result['signal'], color=colors[i], linestyle='--', label=f"Signal - {params['name']}")
        
        # 计算得分
        score = macd_indicator.calculate_raw_score(df)
        
        # 绘制得分
        axes[2].plot(score.index, score, color=colors[i], label=f"Score - {params['name']}")
        
        # 绘制买入和卖出信号
        buy_signals = result[result['golden_cross'] | result['macd_cross_zero_up']].index
        sell_signals = result[result['death_cross'] | result['macd_cross_zero_down']].index
        
        axes[0].scatter(buy_signals, df.loc[buy_signals, 'close'], marker='^', color=colors[i], s=50, 
                       label=f"Buy - {params['name']}")
        axes[0].scatter(sell_signals, df.loc[sell_signals, 'close'], marker='v', color=colors[i], s=50, 
                       label=f"Sell - {params['name']}")
    
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.2)
    axes[1].set_ylabel('MACD Lines')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].axhline(y=50, color='k', linestyle='-', alpha=0.2)
    axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
    axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel('Score (0-100)')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # 隐藏上面图形的x轴标签
    plt.setp(axes[0].get_xticklabels(), visible=False)
    plt.setp(axes[1].get_xticklabels(), visible=False)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join(root_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'enhanced_macd_parameter_tuning.png'), dpi=300)
    
    # 显示图表
    plt.show()


def main():
    """主函数"""
    print("测试增强MACD指标...")
    test_enhanced_macd()
    
    print("\n测试参数调整对MACD指标的影响...")
    test_parameter_tuning()
    
    print("\n测试完成！")


if __name__ == "__main__":
    main() 