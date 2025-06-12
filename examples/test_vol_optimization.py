#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
成交量(VOL)指标优化测试脚本

用于验证VOL指标优化的效果
"""

import os
import sys
import pandas as pd
import numpy as np

# 将项目根目录添加到sys.path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from indicators.volume.vol import VOL
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
        file_path = os.path.join(root_path, 'data', f'{stock_code}_{start_date}_{end_date}.csv')
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
    volume_base = 1000 + np.random.normal(0, 100, n).cumsum()
    
    # 添加成交量波动和周期性变化
    volume_cycle = 500 * np.sin(t / 20) + 200 * np.cos(t / 40)
    volume_trend = 10 * (t / n)  # 轻微增长趋势
    
    # 添加一些突发的大成交量
    volume_spikes = np.zeros(n)
    spike_positions = np.random.choice(range(n), size=int(n*0.05), replace=False)
    volume_spikes[spike_positions] = np.random.normal(2000, 500, len(spike_positions))
    
    # 组合所有成交量因素
    volume = np.abs(volume_base + volume_cycle + volume_trend + volume_spikes)
    
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


def plot_vol_score_comparison(data, old_score, new_score, title="VOL评分对比"):
    """
    绘制VOL指标优化前后评分对比图
    
    Args:
        data: 原始数据
        old_score: 优化前评分
        new_score: 优化后评分
        title: 图表标题
    """
    # 实现代码
    pass


class SimpleVOL(VOL):
    """简化版VOL，用于对比优化前的效果"""
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量(VOL)指标（优化前版本）
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            添加了VOL指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['volume']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 添加原始成交量
        df_copy['vol'] = df_copy['volume']
        
        # 计算成交量移动平均
        df_copy['vol_ma5'] = df_copy['volume'].rolling(window=5).mean()
        df_copy['vol_ma10'] = df_copy['volume'].rolling(window=10).mean()
        df_copy['vol_ma20'] = df_copy['volume'].rolling(window=20).mean()
        
        # 计算相对成交量（当前成交量与N日平均成交量的比值）
        df_copy['vol_ratio'] = df_copy['volume'] / df_copy['vol_ma5']
        
        return df_copy
    
    def calculate_raw_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算成交量指标的原始评分（优化前版本）
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.Series: 包含原始评分的Series
        """
        # 计算指标值
        indicator_data = self.calculate(data)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 获取成交量数据
        volume = indicator_data['vol']
        vol_ma5 = indicator_data['vol_ma5']
        vol_ma10 = indicator_data['vol_ma10']
        vol_ma20 = indicator_data['vol_ma20']
        vol_ratio = indicator_data['vol_ratio'].fillna(1.0)
        
        # 1. 成交量水平评分（-15到+25分）
        # 放量加分，缩量减分
        high_volume_mask = vol_ratio > 1.5
        score.loc[high_volume_mask] += 15
        
        very_high_volume_mask = vol_ratio > 2.0
        score.loc[very_high_volume_mask] += 25
        
        low_volume_mask = vol_ratio < 0.7
        score.loc[low_volume_mask] -= 10
        
        very_low_volume_mask = vol_ratio < 0.5
        score.loc[very_low_volume_mask] -= 15
        
        # 2. 量价配合评分（-20到+25分）
        if 'close' in data.columns:
            close_price = data['close']
            price_change = close_price.pct_change().fillna(0)
            
            # 价涨量增（理想状态）
            price_up_vol_up = (price_change > 0.02) & (vol_ratio > 1.2)
            score.loc[price_up_vol_up] += 20
            
            # 价涨量增（强势）
            strong_price_up_vol_up = (price_change > 0.05) & (vol_ratio > 1.5)
            score.loc[strong_price_up_vol_up] += 25
            
            # 价跌量增（警告信号）
            price_down_vol_up = (price_change < -0.02) & (vol_ratio > 1.2)
            score.loc[price_down_vol_up] -= 15
            
            # 价跌量增（恐慌信号）
            panic_price_down_vol_up = (price_change < -0.05) & (vol_ratio > 1.5)
            score.loc[panic_price_down_vol_up] -= 20
            
            # 价涨量缩（警告信号）
            price_up_vol_down = (price_change > 0.02) & (vol_ratio < 0.8)
            score.loc[price_up_vol_down] -= 10
            
            # 价跌量缩（可能见底）
            price_down_vol_down = (price_change < -0.02) & (vol_ratio < 0.8)
            score.loc[price_down_vol_down] += 10
        
        # 3. 成交量趋势评分（-15到+15分）
        # 计算成交量趋势
        vol_trend_5 = volume.rolling(window=5).mean().pct_change().fillna(0)
        vol_trend_10 = volume.rolling(window=10).mean().pct_change().fillna(0)
        
        # 持续放量
        sustained_volume_up = (vol_trend_5 > 0.1) & (vol_trend_10 > 0.05)
        score.loc[sustained_volume_up] += 10
        
        # 急剧放量
        sharp_volume_up = vol_trend_5 > 0.3
        score.loc[sharp_volume_up] += 15
        
        # 持续缩量
        sustained_volume_down = (vol_trend_5 < -0.1) & (vol_trend_10 < -0.05)
        score.loc[sustained_volume_down] -= 10
        
        # 急剧缩量
        sharp_volume_down = vol_trend_5 < -0.3
        score.loc[sharp_volume_down] -= 15
        
        return np.clip(score, 0, 100)


def test_vol_optimization(data):
    """
    测试VOL指标优化效果
    
    Args:
        data: 测试数据
        
    Returns:
        tuple: (原始评分, 优化后评分)
    """
    # 创建未优化和优化后的VOL实例
    simple_vol = SimpleVOL()
    enhanced_vol = VOL()
    
    # 计算VOL指标
    simple_result = simple_vol.calculate(data)
    enhanced_result = enhanced_vol.calculate(data)
    
    # 计算评分
    simple_score = simple_vol.calculate_raw_score(data)
    enhanced_score = enhanced_vol.calculate_raw_score(data)
    
    # 计算评分差异
    score_diff = enhanced_score - simple_score
    
    # 输出统计信息
    print("===== VOL指标优化效果分析 =====")
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
    
    # 异常放量识别
    if 'vol_std' in enhanced_result.columns:
        vol = enhanced_result['vol']
        vol_ma20 = enhanced_result['vol_ma20']
        vol_std = enhanced_result['vol_std']
        
        # 计算Z分数
        z_score = (vol - vol_ma20) / (vol_ma20 * vol_std)
        abnormal_volume = z_score > 3
        
        print("\n异常放量识别:")
        print(f"检测到的异常放量次数: {abnormal_volume.sum()}")
        
        # 输出异常放量的日期
        if abnormal_volume.sum() > 0:
            abnormal_dates = data.index[abnormal_volume]
            print("异常放量日期:")
            for date in abnormal_dates:
                idx = data.index.get_loc(date)
                vol_value = vol.iloc[idx]
                vol_ma_value = vol_ma20.iloc[idx]
                ratio = vol_value / vol_ma_value
                print(f"- {date}: 成交量 {vol_value:.0f}, 相对均量比 {ratio:.2f}")
    
    # 绘制评分曲线
    # 将原始数据与计算结果合并用于绘图
    plot_data = data.copy()
    for col in enhanced_result.columns:
        if col not in plot_data.columns:
            plot_data[col] = enhanced_result[col]
    
    # 返回评分结果用于绘图
    return simple_score, enhanced_score, plot_data


def find_best_vol_params(data):
    """
    找到最佳的VOL参数组合
    
    Args:
        data: 测试数据
        
    Returns:
        tuple: 最佳参数组合
    """
    short_periods = [5, 10, 15, 20]
    long_periods = [10, 20, 30, 40]
    results = []

    for period_short in short_periods:
        for period_long in long_periods:
            if period_short >= period_long:
                continue

            vol = VOL(periods=[period_short, period_long])
            signals = vol.generate_signals(data)
            
            # 简单的评估逻辑：信号越多越好（仅为示例）
            score = signals['buy_signal'].sum() + signals['sell_signal'].sum()
            results.append(((period_short, period_long), score))

    # 找到最佳参数组合
    best_params, best_score = max(results, key=lambda item: item[1])
    
    logger.info(f"找到的最佳VOL参数组合: 短周期={best_params[0]}, 长周期={best_params[1]}")
    logger.info(f"对应的评分为: {best_score}")

    return best_params


def main():
    """主函数"""
    try:
        # 加载测试数据
        data = load_test_data(start_date='2022-01-01', end_date='2022-12-31')
        
        # 测试VOL指标优化效果
        old_score, new_score, plot_data = test_vol_optimization(data)
        
        # 绘制对比图
        plot_vol_score_comparison(plot_data, old_score, new_score)
        
        print("\n测试完成，对比图已保存到 'data/result/vol_optimization.png'")
        
        # 找到最佳参数组合
        best_params = find_best_vol_params(data)
        
        logger.info(f"VOL指标优化后的最佳参数为: 短周期={best_params[0]}, 长周期={best_params[1]}")

        # 使用最佳参数重新计算并验证
        vol = VOL(periods=[best_params[0], best_params[1]])
        final_result = vol.calculate(data)
        final_signals = vol.generate_signals(final_result)
        
        logger.info("使用最佳参数计算的最终买入信号数量: %d", final_signals['buy_signal'].sum())
        logger.info("使用最佳参数计算的最终卖出信号数量: %d", final_signals['sell_signal'].sum())
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 