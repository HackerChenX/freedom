#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KDJ指标优化测试脚本

用于验证KDJ指标优化的效果
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

from indicators.kdj import KDJ
from indicators.common import kdj
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


def plot_kdj_score_comparison(data, old_score, new_score, title="KDJ评分对比"):
    """
    绘制KDJ指标优化前后评分对比图
    
    Args:
        data: 原始数据
        old_score: 优化前评分
        new_score: 优化后评分
        title: 图表标题
    """
    plt.figure(figsize=(15, 12))
    
    # 绘制子图1: 价格和KDJ指标
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(data.index, data['close'], label='收盘价', color='black')
    ax1.set_title('价格')
    ax1.legend(loc='best')
    ax1.grid(True)
    
    # 绘制子图2: KDJ指标
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(data.index, data['K'], label='K', color='blue')
    ax2.plot(data.index, data['D'], label='D', color='orange')
    ax2.plot(data.index, data['J'], label='J', color='purple')
    ax2.axhline(y=80, color='red', linestyle='--', alpha=0.6)  # 超买线
    ax2.axhline(y=20, color='green', linestyle='--', alpha=0.6)  # 超卖线
    ax2.set_title('KDJ指标')
    ax2.legend(loc='best')
    ax2.grid(True)
    
    # 绘制子图3: 评分对比
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(data.index, old_score, label='优化前评分', color='blue', alpha=0.7)
    ax3.plot(data.index, new_score, label='优化后评分', color='red')
    ax3.set_title('KDJ评分对比')
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.6)  # 添加中性线
    ax3.legend(loc='best')
    ax3.grid(True)
    
    # 调整布局
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # 保存图表
    plt.savefig(os.path.join(project_root, 'data', 'result', 'kdj_optimization.png'))
    plt.close()


def test_kdj_optimization(data):
    """
    测试KDJ指标优化效果
    
    Args:
        data: 测试数据
        
    Returns:
        tuple: (原始评分, 优化后评分)
    """
    # 创建未优化的KDJ指标计算函数
    class SimpleKDJ(KDJ):
        """简化版KDJ，用于对比优化前的效果"""
        
        def calculate_raw_score(self, data):
            """
            计算KDJ原始评分（优化前）
            
            Args:
                data: 输入数据
                
            Returns:
                pd.Series: 原始评分序列（0-100分）
            """
            # 确保已计算KDJ
            if not self.has_result():
                self.calculate(data)
            
            if self._result is None:
                return pd.Series(50.0, index=data.index)
            
            score = pd.Series(50.0, index=data.index)  # 基础分50分
            
            k = self._result['K']
            d = self._result['D']
            j = self._result['J']
            
            # 1. K和D线交叉评分
            golden_cross = self.crossover(k, d)
            death_cross = self.crossunder(k, d)
            score += golden_cross * 20  # K上穿D+20分
            score -= death_cross * 20   # K下穿D-20分
            
            # 2. 超买超卖区域评分
            oversold_area = (k < 20) & (d < 20) & (j < 20)
            overbought_area = (k > 80) & (d > 80) & (j > 80)
            score += oversold_area * 25   # 三线超卖+25分
            score -= overbought_area * 25 # 三线超买-25分
            
            # 3. K线区域变化评分
            k_leaving_oversold = (k > 20) & (k.shift(1) <= 20)
            k_leaving_overbought = (k < 80) & (k.shift(1) >= 80)
            score += k_leaving_oversold * 15   # K线离开超卖区+15分
            score -= k_leaving_overbought * 15 # K线离开超买区-15分
            
            # 4. J线极端区域评分
            j_extreme_oversold = j < 0
            j_extreme_overbought = j > 100
            score += j_extreme_oversold * 30   # J线极端超卖+30分
            score -= j_extreme_overbought * 30 # J线极端超买-30分
            
            # 5. 钝化形态评分
            # 低位钝化（连续多个周期在超卖区）
            low_stagnation = self._detect_stagnation(k, d, j, low_threshold=20, periods=5)
            # 高位钝化（连续多个周期在超买区）
            high_stagnation = self._detect_stagnation(k, d, j, high_threshold=80, periods=5)
            
            score += low_stagnation * 20   # 低位钝化+20分
            score -= high_stagnation * 20  # 高位钝化-20分
            
            # 6. 背离评分
            if len(data) >= 20:
                divergence_score = self._calculate_kdj_divergence(data['close'], k, d)
                score += divergence_score
            
            return np.clip(score, 0, 100)
    
    # 创建两个实例
    simple_kdj = SimpleKDJ(n=9, m1=3, m2=3)
    enhanced_kdj = KDJ(n=9, m1=3, m2=3)
    
    # 计算KDJ
    simple_result = simple_kdj.calculate(data)
    enhanced_result = enhanced_kdj.calculate(data)
    
    # 确保有相同的K、D、J值用于绘图
    data_with_kdj = data.copy()
    data_with_kdj['K'] = simple_result['K']
    data_with_kdj['D'] = simple_result['D']
    data_with_kdj['J'] = simple_result['J']
    
    # 计算评分
    simple_score = simple_kdj.calculate_raw_score(data)
    enhanced_score = enhanced_kdj.calculate_raw_score(data)
    
    # 计算评分差异
    score_diff = enhanced_score - simple_score
    
    # 输出统计信息
    print("===== KDJ指标优化效果分析 =====")
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
    simple_patterns = simple_kdj.identify_patterns(data)
    enhanced_patterns = enhanced_kdj.identify_patterns(data)
    
    print("\n形态识别:")
    print(f"优化前识别形态数量: {len(simple_patterns)}")
    print(f"优化后识别形态数量: {len(enhanced_patterns)}")
    
    if len(enhanced_patterns) > 0:
        print("\n优化后识别的形态:")
        for pattern in enhanced_patterns:
            print(f"- {pattern}")
    
    # 返回评分结果用于绘图
    return simple_score, enhanced_score, data_with_kdj


def main():
    """主函数"""
    try:
        # 加载测试数据
        data = load_test_data(start_date='2022-01-01', end_date='2022-12-31')
        
        # 测试KDJ指标优化效果
        old_score, new_score, data_with_kdj = test_kdj_optimization(data)
        
        # 绘制对比图
        plot_kdj_score_comparison(data_with_kdj, old_score, new_score)
        
        print("\n测试完成，对比图已保存到 'data/result/kdj_optimization.png'")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 