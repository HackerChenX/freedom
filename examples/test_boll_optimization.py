#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
布林带(BOLL)指标优化测试脚本

用于验证布林带指标优化的效果
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from indicators.boll import BOLL
from indicators.common import boll
from utils.logger import get_logger

logger = get_logger(__name__)


def load_test_data(stock_code='000001.SZ', start_date='2022-01-01', end_date='2023-01-01'):
    """
    加载测试数据
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        pd.DataFrame: 测试数据
    """
    try:
        # 尝试从本地文件加载数据
        file_path = os.path.join(project_root, 'data', f'{stock_code}_{start_date}_{end_date}.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, parse_dates=['trade_date'])
            return data
    except Exception as e:
        logger.error(f"从本地加载数据失败: {e}")
    
    # 使用随机生成的测试数据
    logger.info("使用随机生成的测试数据")
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    # 生成模拟价格数据
    close = np.random.normal(100, 1, n).cumsum()
    
    # 添加趋势和波动
    t = np.arange(n)
    trend = 10 * np.sin(t / 30)
    volatility = np.abs(np.sin(t / 20)) * 5
    
    # 添加趋势和波动到价格
    close = close + trend
    high = close + np.random.rand(n) * volatility
    low = close - np.random.rand(n) * volatility
    open_price = close - volatility / 2 + np.random.rand(n) * volatility
    
    # 生成成交量
    volume = np.random.normal(1000, 100, n) * (1 + np.abs(np.sin(t / 25)))
    
    # 创建DataFrame
    data = pd.DataFrame({
        'trade_date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return data


def plot_boll_score_comparison(data, old_score, new_score, title="布林带评分对比"):
    """
    绘制BOLL指标优化前后评分对比图
    
    Args:
        data: 原始数据
        old_score: 优化前评分
        new_score: 优化后评分
        title: 图表标题
    """
    plt.figure(figsize=(15, 12))
    
    # 绘制子图1: 价格和布林带
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(data.index, data['close'], label='收盘价', color='black')
    ax1.plot(data.index, data['upper'], label='上轨', color='red', linestyle='--')
    ax1.plot(data.index, data['middle'], label='中轨', color='blue')
    ax1.plot(data.index, data['lower'], label='下轨', color='green', linestyle='--')
    ax1.set_title('价格与布林带')
    ax1.legend(loc='best')
    ax1.grid(True)
    
    # 绘制子图2: 带宽变化
    ax2 = plt.subplot(3, 1, 2)
    bandwidth = (data['upper'] - data['lower']) / data['middle']
    ax2.plot(data.index, bandwidth, label='带宽', color='purple')
    ax2.set_title('布林带带宽')
    ax2.legend(loc='best')
    ax2.grid(True)
    
    # 绘制子图3: 评分对比
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(data.index, old_score, label='优化前评分', color='blue', alpha=0.7)
    ax3.plot(data.index, new_score, label='优化后评分', color='red')
    ax3.set_title('BOLL评分对比')
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.6)  # 添加中性线
    ax3.legend(loc='best')
    ax3.grid(True)
    
    # 调整布局
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # 保存图表
    plt.savefig(os.path.join(project_root, 'data', 'result', 'boll_optimization.png'))
    plt.close()


def test_boll_optimization(data):
    """
    测试BOLL指标优化效果
    
    Args:
        data: 测试数据
        
    Returns:
        tuple: (原始评分, 优化后评分)
    """
    # 创建未优化的BOLL指标计算函数
    class SimpleBOLL(BOLL):
        """简化版BOLL，用于对比优化前的效果"""
        
        def calculate_raw_score(self, data):
            """
            计算BOLL原始评分（优化前）
            
            Args:
                data: 输入数据
                
            Returns:
                pd.Series: 原始评分序列（0-100分）
            """
            # 确保已计算BOLL
            if not self.has_result():
                self.calculate(data)
            
            if self._result is None:
                return pd.Series(50.0, index=data.index)
            
            score = pd.Series(50.0, index=data.index)  # 基础分50分
            
            close = self._result['close']
            upper = self._result['upper']
            middle = self._result['middle']
            lower = self._result['lower']
            
            # 1. 价格位置评分
            # 价格触及下轨（超卖）+20分
            price_at_lower = close <= lower
            score += price_at_lower * 20
            
            # 价格触及上轨（超买）-20分
            price_at_upper = close >= upper
            score -= price_at_upper * 20
            
            # 2. 价格突破评分
            # 价格突破下轨（强烈超卖）+25分
            price_break_lower = close < lower
            score += price_break_lower * 25
            
            # 价格突破上轨（强烈超买）-25分
            price_break_upper = close > upper
            score -= price_break_upper * 25
            
            # 3. 价格运行方向评分
            # 价格由下轨向中轨运行+15分
            price_from_lower_to_middle = self._detect_price_movement(close, lower, middle, direction='up')
            score += price_from_lower_to_middle * 15
            
            # 价格由上轨向中轨运行-15分
            price_from_upper_to_middle = self._detect_price_movement(close, upper, middle, direction='down')
            score -= price_from_upper_to_middle * 15
            
            # 4. 带宽变化评分
            bandwidth = (upper - lower) / middle
            bandwidth_expanding = bandwidth > bandwidth.shift(1)
            bandwidth_contracting = bandwidth < bandwidth.shift(1)
            
            # 带宽收缩（可能孕育突破）+15分
            score += bandwidth_contracting * 15
            
            # 带宽极低（即将突破）+20分
            bandwidth_percentile = bandwidth.rolling(window=60).rank(pct=True)
            extremely_low_bandwidth = bandwidth_percentile < 0.1
            score += extremely_low_bandwidth * 20
            
            # 5. 中轨穿越评分
            # 价格上穿中轨+10分
            price_cross_up_middle = self.crossover(close, middle)
            score += price_cross_up_middle * 10
            
            # 价格下穿中轨-10分
            price_cross_down_middle = self.crossunder(close, middle)
            score -= price_cross_down_middle * 10
            
            return np.clip(score, 0, 100)
    
    # 创建两个实例
    simple_boll = SimpleBOLL(periods=20, std_dev=2.0)
    enhanced_boll = BOLL(periods=20, std_dev=2.0)
    
    # 计算布林带
    simple_result = simple_boll.calculate(data)
    enhanced_result = enhanced_boll.calculate(data)
    
    # 计算评分
    simple_score = simple_boll.calculate_raw_score(data)
    enhanced_score = enhanced_boll.calculate_raw_score(data)
    
    # 计算评分差异
    score_diff = enhanced_score - simple_score
    
    # 输出统计信息
    print("===== 布林带指标优化效果分析 =====")
    print(f"数据周期: {data.index[0]} 至 {data.index[-1]}")
    print(f"样本数量: {len(data)}")
    print("\n评分差异统计:")
    print(f"平均差异: {score_diff.mean():.2f}")
    print(f"最大差异: {score_diff.max():.2f}")
    print(f"最小差异: {score_diff.min():.2f}")
    print(f"标准差: {score_diff.std():.2f}")
    
    # 计算买入卖出信号数量
    simple_buy_signal = simple_score > 70
    simple_sell_signal = simple_score < 30
    enhanced_buy_signal = enhanced_score > 70
    enhanced_sell_signal = enhanced_score < 30
    
    print("\n信号数量统计:")
    print(f"优化前买入信号: {simple_buy_signal.sum()}")
    print(f"优化后买入信号: {enhanced_buy_signal.sum()}")
    print(f"买入信号变化: {enhanced_buy_signal.sum() - simple_buy_signal.sum()}")
    print(f"优化前卖出信号: {simple_sell_signal.sum()}")
    print(f"优化后卖出信号: {enhanced_sell_signal.sum()}")
    print(f"卖出信号变化: {enhanced_sell_signal.sum() - simple_sell_signal.sum()}")
    
    # 形态识别统计
    simple_patterns = simple_boll.identify_patterns(data)
    enhanced_patterns = enhanced_boll.identify_patterns(data)
    
    print("\n形态识别:")
    print(f"优化前识别形态数量: {len(simple_patterns)}")
    print(f"优化后识别形态数量: {len(enhanced_patterns)}")
    
    if len(enhanced_patterns) > 0:
        print("\n优化后识别的形态:")
        for pattern in enhanced_patterns:
            print(f"- {pattern}")
    
    # 返回评分结果用于绘图
    return simple_score, enhanced_score


def main():
    """主函数"""
    try:
        # 加载测试数据
        data = load_test_data(start_date='2022-01-01', end_date='2022-12-31')
        
        # 测试BOLL指标优化效果
        old_score, new_score = test_boll_optimization(data)
        
        # 绘制对比图
        plot_boll_score_comparison(data, old_score, new_score)
        
        print("\n测试完成，对比图已保存到 'data/result/boll_optimization.png'")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 