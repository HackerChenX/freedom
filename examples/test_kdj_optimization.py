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


def find_best_kdj_params(data):
    """
    找到最佳的KDJ参数组合
    
    Args:
        data: 测试数据
        
    Returns:
        tuple: 最佳的KDJ参数组合
    """
    n_periods = [3, 5, 7, 9, 11]
    m_periods = [1, 2, 3, 4, 5]
    results = []

    for n in n_periods:
        for m1 in m_periods:
            for m2 in m_periods:
                kdj.set_parameters(n=n, m1=m1, m2=m2)
                signals = kdj.generate_signals(data)
                
                # 简单的评估逻辑：信号越多越好（仅为示例）
                score = signals['buy_signal'].sum() + signals['sell_signal'].sum()
                results.append(((n, m1, m2), score))

    # 找到最佳参数组合
    best_params, best_score = max(results, key=lambda item: item[1])
    
    logger.info(f"找到的最佳KDJ参数组合: n={best_params[0]}, m1={best_params[1]}, m2={best_params[2]}")
    logger.info(f"对应的评分为: {best_score}")

    return best_params


def main():
    """主函数"""
    try:
        # 加载测试数据
        data = load_test_data(start_date='2022-01-01', end_date='2022-12-31')
        
        # 测试KDJ指标优化效果
        old_score, new_score, data_with_kdj = test_kdj_optimization(data)
        
        # 找到最佳参数组合
        best_params = find_best_kdj_params(data)
        
        logger.info(f"KDJ指标优化后的最佳参数为: n={best_params[0]}, m1={best_params[1]}, m2={best_params[2]}")

        # 使用最佳参数重新计算并验证
        kdj = KDJ(n=best_params[0], m1=best_params[1], m2=best_params[2])
        final_result = kdj.calculate(data)
        final_signals = kdj.generate_signals(final_result)
        
        logger.info("使用最佳参数计算的最终买入信号数量: %d", final_signals['buy_signal'].sum())
        logger.info("使用最佳参数计算的最终卖出信号数量: %d", final_signals['sell_signal'].sum())
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 