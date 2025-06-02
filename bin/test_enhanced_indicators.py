#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试强化后的指标系统，包括形态识别和评分机制
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 将项目根目录添加到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.indicator_registry import get_indicator, create_indicator_score_manager
from indicators.pattern_registry import PatternRegistry
from indicators.scoring_framework import MarketEnvironment
from utils.logger import get_logger

logger = get_logger("test_enhanced_indicators")

def load_test_data(file_path=None):
    """
    加载测试数据，如果没有提供文件路径，则生成模拟数据
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        pd.DataFrame: 测试数据
    """
    if file_path and os.path.exists(file_path):
        try:
            # 尝试加载CSV数据
            data = pd.read_csv(file_path, parse_dates=['date'])
            logger.info(f"从文件加载数据: {file_path}")
            return data
        except Exception as e:
            logger.error(f"加载数据文件失败: {e}")
    
    # 生成模拟数据
    logger.info("生成模拟数据")
    
    # 创建日期范围
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    
    # 创建价格数据（模拟股票价格走势）
    np.random.seed(42)  # 确保结果可重现
    
    # 基础价格趋势（模拟长期趋势）
    base_trend = np.linspace(100, 150, len(dates))
    
    # 添加周期性波动（模拟季节性）
    seasonality = 10 * np.sin(np.linspace(0, 15 * np.pi, len(dates)))
    
    # 添加随机波动
    noise = np.random.normal(0, 5, len(dates))
    
    # 组合生成最终价格
    price = base_trend + seasonality + noise
    
    # 创建OHLC数据
    close = price
    high = close + np.random.uniform(0, 5, len(dates))
    low = close - np.random.uniform(0, 5, len(dates))
    open_price = low + np.random.uniform(0, high - low, len(dates))
    
    # 创建成交量数据
    volume = np.random.normal(1000000, 300000, len(dates))
    volume = np.abs(volume)  # 确保成交量为正
    
    # 根据价格趋势调整成交量（上涨时成交量通常更大）
    price_change = np.diff(close, prepend=close[0])
    volume_adjust = np.where(price_change > 0, 1.2, 0.8)
    volume = volume * volume_adjust
    
    # 创建DataFrame
    data = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # 只保留工作日
    data = data[data['date'].dt.dayofweek < 5].reset_index(drop=True)
    
    return data

def test_single_indicator(data, indicator_name, params=None):
    """
    测试单个指标的计算和评分
    
    Args:
        data: 输入数据
        indicator_name: 指标名称
        params: 指标参数
        
    Returns:
        tuple: (计算结果, 评分结果)
    """
    logger.info(f"测试指标: {indicator_name}")
    
    # 创建指标实例
    params = params or {}
    indicator = get_indicator(indicator_name, **params)
    
    if indicator is None:
        logger.error(f"指标 {indicator_name} 创建失败")
        return None, None
    
    # 计算指标
    result = indicator.calculate(data, price_col='close')
    
    # 计算评分
    score_result = indicator.calculate_score(data, price_col='close')
    
    # 获取形态
    patterns = indicator.get_patterns(data)
    
    # 输出统计信息
    logger.info(f"指标 {indicator_name} 计算完成，结果形状: {result.shape}")
    logger.info(f"检测到 {len(patterns)} 个形态")
    for pattern in patterns:
        logger.info(f"形态: {pattern.pattern_id}, 强度: {getattr(pattern, 'strength', 1.0)}")
    
    return result, score_result

def test_composite_scoring(data, indicator_names=None, weights=None):
    """
    测试组合评分
    
    Args:
        data: 输入数据
        indicator_names: 指标名称列表
        weights: 指标权重字典
        
    Returns:
        dict: 评分结果
    """
    logger.info("测试组合评分")
    
    # 创建评分管理器
    manager = create_indicator_score_manager(indicators=indicator_names, weights=weights)
    
    # 设置市场环境（可以根据实际情况动态调整）
    manager.set_market_environment(MarketEnvironment.BULL_MARKET)
    
    # 计算组合评分
    score_result = manager.calculate_combined_score(data, price_col='close')
    
    # 生成信号
    signals = manager.generate_signals(data, price_col='close')
    
    # 获取最后一个交易日的信号摘要
    summary = manager.get_signal_summary(data, index=-1, price_col='close')
    
    # 输出统计信息
    logger.info(f"组合评分计算完成，形状: {score_result['combined_score'].shape}")
    logger.info(f"检测到 {len(score_result['patterns'])} 个形态")
    logger.info(f"最后一个交易日信号: {summary['signal_type']}, 强度: {summary['signal_strength']}")
    
    return score_result, signals, summary

def visualize_results(data, indicator_results, score_results, signals=None, output_dir=None):
    """
    可视化结果
    
    Args:
        data: 输入数据
        indicator_results: 指标计算结果字典
        score_results: 评分结果字典
        signals: 信号结果
        output_dir: 输出目录
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 设置绘图环境
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 可视化价格和指标
    for indicator_name, result in indicator_results.items():
        plt.figure(figsize=(16, 12))
        
        # 创建三个子图
        ax1 = plt.subplot(3, 1, 1)  # 价格图
        ax2 = plt.subplot(3, 1, 2)  # 指标图
        ax3 = plt.subplot(3, 1, 3)  # 评分图
        
        # 绘制价格
        ax1.plot(data['close'], label='收盘价')
        ax1.set_title('价格走势')
        ax1.legend()
        
        # 绘制指标
        if indicator_name == 'MACD':
            # MACD特殊处理
            ax2.plot(result['macd'], label='MACD')
            ax2.plot(result['signal'], label='Signal')
            ax2.bar(range(len(result)), result['histogram'], label='Histogram', alpha=0.5)
        elif indicator_name == 'KDJ':
            # KDJ特殊处理
            ax2.plot(result['K'], label='K')
            ax2.plot(result['D'], label='D')
            ax2.plot(result['J'], label='J')
        elif indicator_name == 'RSI':
            # RSI特殊处理
            ax2.plot(result['RSI'], label='RSI')
            ax2.axhline(y=70, color='r', linestyle='--')
            ax2.axhline(y=30, color='g', linestyle='--')
        elif indicator_name == 'BOLL':
            # BOLL特殊处理
            ax2.plot(result['upper'], label='Upper')
            ax2.plot(result['middle'], label='Middle')
            ax2.plot(result['lower'], label='Lower')
            ax2.plot(data['close'], label='Close', alpha=0.5)
        else:
            # 通用处理
            for col in result.columns:
                if col not in ['date', 'open', 'high', 'low', 'close', 'volume']:
                    ax2.plot(result[col], label=col)
        
        ax2.set_title(f'{indicator_name} 指标')
        ax2.legend()
        
        # 绘制评分
        score_data = score_results[indicator_name]
        if 'score' in score_data:
            ax3.plot(score_data['score'], label='Score')
        elif 'final_score' in score_data:
            ax3.plot(score_data['final_score'], label='Final Score')
        
        # 添加阈值线
        ax3.axhline(y=80, color='g', linestyle='--')
        ax3.axhline(y=50, color='k', linestyle='--')
        ax3.axhline(y=20, color='r', linestyle='--')
        
        ax3.set_title(f'{indicator_name} 评分')
        ax3.legend()
        
        plt.tight_layout()
        
        # 保存图像
        if output_dir:
            file_path = os.path.join(output_dir, f"{indicator_name}_{timestamp}.png")
            plt.savefig(file_path)
            logger.info(f"保存图像: {file_path}")
        else:
            plt.show()
        
        plt.close()
    
    # 可视化组合评分
    if 'combined_score' in score_results:
        plt.figure(figsize=(16, 10))
        
        # 创建两个子图
        ax1 = plt.subplot(2, 1, 1)  # 价格图
        ax2 = plt.subplot(2, 1, 2)  # 评分图
        
        # 绘制价格
        ax1.plot(data['close'], label='收盘价')
        ax1.set_title('价格走势')
        ax1.legend()
        
        # 绘制组合评分
        ax2.plot(score_results['combined_score'], label='Combined Score')
        
        # 添加阈值线
        ax2.axhline(y=80, color='g', linestyle='--')
        ax2.axhline(y=50, color='k', linestyle='--')
        ax2.axhline(y=20, color='r', linestyle='--')
        
        # 如果有信号，绘制信号
        if signals and 'buy_signal' in signals:
            # 找出买入信号的位置
            buy_signal_indices = signals['buy_signal'][signals['buy_signal']].index
            ax1.scatter(buy_signal_indices, data.loc[buy_signal_indices, 'close'], 
                       color='g', marker='^', s=100, label='Buy Signal')
            
            # 找出卖出信号的位置
            sell_signal_indices = signals['sell_signal'][signals['sell_signal']].index
            ax1.scatter(sell_signal_indices, data.loc[sell_signal_indices, 'close'], 
                       color='r', marker='v', s=100, label='Sell Signal')
            
        ax2.set_title('组合评分')
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图像
        if output_dir:
            file_path = os.path.join(output_dir, f"combined_score_{timestamp}.png")
            plt.savefig(file_path)
            logger.info(f"保存图像: {file_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    """主函数"""
    logger.info("开始测试强化后的指标系统")
    
    # 加载测试数据
    data = load_test_data()
    
    # 测试的指标列表
    test_indicators = ['MACD', 'KDJ', 'RSI', 'BOLL']
    
    # 存储测试结果
    indicator_results = {}
    score_results = {}
    
    # 测试每个指标
    for indicator_name in test_indicators:
        result, score = test_single_indicator(data, indicator_name)
        if result is not None:
            indicator_results[indicator_name] = result
            score_results[indicator_name] = score
    
    # 测试组合评分
    weights = {
        'MACD': 1.2,
        'KDJ': 1.0,
        'RSI': 0.8,
        'BOLL': 1.0
    }
    
    composite_score, signals, summary = test_composite_scoring(
        data, indicator_names=test_indicators, weights=weights
    )
    
    # 可视化结果
    output_dir = os.path.join(root_dir, "data", "result", "indicator_test")
    visualize_results(data, indicator_results, score_results, signals, output_dir)
    
    # 输出组合评分摘要
    logger.info("\n组合评分摘要:")
    logger.info(f"信号类型: {summary['signal_type']}")
    logger.info(f"信号强度: {summary['signal_strength']}")
    logger.info(f"综合评分: {summary['combined_score']:.2f}")
    logger.info(f"市场环境: {summary['market_environment']}")
    
    logger.info("\n检测到的形态:")
    for pattern in summary['patterns']:
        logger.info(f"形态: {pattern['display_name']}, 影响: {pattern['score_impact']:.2f}, 信号类型: {pattern['signal_type']}")
    
    logger.info("\n各指标评分:")
    for indicator_name, info in summary['indicators'].items():
        logger.info(f"{indicator_name}: 评分 {info['score']:.2f}, 权重 {info['weight']:.2f}")
        
    logger.info("测试完成")

if __name__ == "__main__":
    main() 