#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
乖离率(BIAS)指标优化测试脚本

用于验证BIAS指标优化的效果
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

from indicators.bias import BIAS
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
    
    # 添加趋势和周期性波动
    t = np.arange(n)
    trend = 10 * np.sin(t / 45) + 5 * np.cos(t / 90)
    volatility = np.abs(np.sin(t / 30)) * 8
    
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


def plot_bias_score_comparison(data, old_score, new_score, title="BIAS评分对比"):
    """
    绘制BIAS指标优化前后评分对比图
    
    Args:
        data: 原始数据
        old_score: 优化前评分
        new_score: 优化后评分
        title: 图表标题
    """
    plt.figure(figsize=(15, 12))
    
    # 绘制子图1: 价格和BIAS
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(data.index, data['close'], label='收盘价', color='black')
    ax1.set_title('价格走势')
    ax1.legend(loc='best')
    ax1.grid(True)
    
    # 绘制子图2: BIAS指标
    ax2 = plt.subplot(3, 1, 2)
    bias_cols = [col for col in data.columns if col.startswith('BIAS')]
    colors = ['blue', 'red', 'green']
    
    for i, col in enumerate(bias_cols):
        ax2.plot(data.index, data[col], label=col, color=colors[i % len(colors)])
    
    ax2.axhline(y=6, color='r', linestyle='--', alpha=0.3)
    ax2.axhline(y=-6, color='g', linestyle='--', alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax2.set_title('BIAS指标')
    ax2.legend(loc='best')
    ax2.grid(True)
    
    # 绘制子图3: 评分对比
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(data.index, old_score, label='优化前评分', color='blue', alpha=0.7)
    ax3.plot(data.index, new_score, label='优化后评分', color='red')
    ax3.set_title('BIAS评分对比')
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.6)  # 添加中性线
    ax3.legend(loc='best')
    ax3.grid(True)
    
    # 调整布局
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # 保存图表
    result_dir = os.path.join(project_root, 'data', 'result')
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(os.path.join(result_dir, 'bias_optimization.png'))
    plt.close()


class SimpleBIAS(BIAS):
    """简化版BIAS，用于对比优化前的效果"""
    
    def _calculate_bias_overbought_oversold_score(self, data=None) -> pd.Series:
        """
        计算BIAS超买超卖评分（优化前版本）
        
        Returns:
            pd.Series: 超买超卖评分
        """
        overbought_oversold_score = pd.Series(0.0, index=self._result.index)
        
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in self._result.columns:
                bias_values = self._result[bias_col]
                
                # 超卖区域（BIAS < -6%）+20分
                oversold_condition = bias_values < -6
                oversold_intensity = np.abs(bias_values + 6) / 6  # 计算超卖强度
                oversold_score = oversold_condition * (20 + oversold_intensity * 10)  # 最多+30分
                overbought_oversold_score += oversold_score
                
                # 超买区域（BIAS > 6%）-20分
                overbought_condition = bias_values > 6
                overbought_intensity = (bias_values - 6) / 6  # 计算超买强度
                overbought_score = overbought_condition * (20 + overbought_intensity * 10)  # 最多-30分
                overbought_oversold_score -= overbought_score
                
                # 极度超卖（BIAS < -10%）额外+15分
                extreme_oversold = bias_values < -10
                overbought_oversold_score += extreme_oversold * 15
                
                # 极度超买（BIAS > 10%）额外-15分
                extreme_overbought = bias_values > 10
                overbought_oversold_score -= extreme_overbought * 15
        
        return overbought_oversold_score / len(self.periods)  # 平均化
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算BIAS原始评分（优化前版本）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算BIAS
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. BIAS超买超卖评分
        overbought_oversold_score = self._calculate_bias_overbought_oversold_score()
        score += overbought_oversold_score
        
        # 2. BIAS回归评分
        regression_score = self._calculate_bias_regression_score()
        score += regression_score
        
        # 3. BIAS极值评分
        extreme_score = self._calculate_bias_extreme_score()
        score += extreme_score
        
        # 4. BIAS趋势评分
        trend_score = self._calculate_bias_trend_score()
        score += trend_score
        
        # 5. BIAS零轴穿越评分
        zero_cross_score = self._calculate_bias_zero_cross_score()
        score += zero_cross_score
        
        return np.clip(score, 0, 100)


def test_bias_optimization(data):
    """
    测试BIAS指标优化效果
    
    Args:
        data: 测试数据
        
    Returns:
        tuple: (原始评分, 优化后评分)
    """
    # 创建未优化和优化后的BIAS实例
    simple_bias = SimpleBIAS(periods=[6, 12, 24])  # 短中长周期
    enhanced_bias = BIAS(periods=[6, 12, 24])  # 短中长周期
    
    # 计算BIAS指标
    simple_result = simple_bias.calculate(data)
    enhanced_result = enhanced_bias.calculate(data)
    
    # 计算评分
    simple_score = simple_bias.calculate_raw_score(data)
    enhanced_score = enhanced_bias.calculate_raw_score(data)
    
    # 计算评分差异
    score_diff = enhanced_score - simple_score
    
    # 输出统计信息
    print("===== BIAS指标优化效果分析 =====")
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
    simple_patterns = simple_bias.identify_patterns(data)
    enhanced_patterns = enhanced_bias.identify_patterns(data)
    
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
        
        # 测试BIAS指标优化效果
        old_score, new_score = test_bias_optimization(data)
        
        # 绘制对比图
        plot_bias_score_comparison(data, old_score, new_score)
        
        print("\n测试完成，对比图已保存到 'data/result/bias_optimization.png'")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 