#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强指标测试脚本

测试增强型技术指标的计算和评分功能
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from db.clickhouse_db import get_clickhouse_db
from indicators.volume.enhanced_obv import EnhancedOBV
from indicators.enhanced_factory import EnhancedIndicatorFactory
from indicators.market_env import MarketDetector
from indicators.synergy import IndicatorSynergy
from utils.logger import get_logger

logger = get_logger(__name__)


def get_test_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取测试数据
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        pd.DataFrame: 测试数据
    """
    db = get_clickhouse_db()
    
    query = f"""
    SELECT 
        trade_date, 
        open, 
        high, 
        low, 
        close, 
        volume 
    FROM stock_kline_day 
    WHERE stock_code = '{stock_code}' 
      AND trade_date >= '{start_date}' 
      AND trade_date <= '{end_date}' 
    ORDER BY trade_date
    """
    
    data = db.query(query)
    if data is None or data.empty:
        return pd.DataFrame()
        
    data = pd.DataFrame(data, columns=['trade_date', 'open', 'high', 'low', 'close', 'volume'])
    data['trade_date'] = pd.to_datetime(data['trade_date'])
    data.set_index('trade_date', inplace=True)
    
    return data


def test_enhanced_obv():
    """测试增强型OBV指标"""
    # 获取测试数据
    stock_code = '000001.SZ'  # 平安银行
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    
    data = get_test_data(stock_code, start_date, end_date)
    if data.empty:
        logger.error("获取测试数据失败")
        return
    
    logger.info(f"获取 {stock_code} 从 {start_date} 到 {end_date} 的数据，共 {len(data)} 条记录")
    
    # 创建增强型OBV指标
    enhanced_obv = EnhancedOBV(
        ma_period=30, 
        sensitivity=1.2, 
        noise_filter=0.005, 
        multi_periods=[5, 10, 20, 60]
    )
    
    # 创建原始OBV指标（用于对比）
    from indicators.obv import OBV
    original_obv = OBV(ma_period=30)
    
    # 计算指标
    result_enhanced = enhanced_obv.calculate(data)
    result_original = original_obv.calculate(data)
    
    # 生成信号
    signals_enhanced = enhanced_obv.generate_signals(data)
    signals_original = original_obv.generate_signals(data)
    
    # 计算评分
    score_enhanced = enhanced_obv.calculate_score(data)
    score_original = original_obv.calculate_score(data)
    
    # 打印结果
    logger.info(f"增强型OBV评分范围: {score_enhanced['final_score'].min():.2f} - {score_enhanced['final_score'].max():.2f}, 平均: {score_enhanced['final_score'].mean():.2f}")
    logger.info(f"原始OBV评分范围: {score_original['final_score'].min():.2f} - {score_original['final_score'].max():.2f}, 平均: {score_original['final_score'].mean():.2f}")
    
    # 比较买入信号数量
    buy_signals_enhanced = signals_enhanced['buy_signal'].sum()
    buy_signals_original = signals_original['buy_signal'].sum()
    
    sell_signals_enhanced = signals_enhanced['sell_signal'].sum()
    sell_signals_original = signals_original['sell_signal'].sum()
    
    logger.info(f"增强型OBV买入信号数量: {buy_signals_enhanced}, 卖出信号数量: {sell_signals_enhanced}")
    logger.info(f"原始OBV买入信号数量: {buy_signals_original}, 卖出信号数量: {sell_signals_original}")
    
    # 绘制结果
    plt.figure(figsize=(14, 10))
    
    # 子图1：价格和OBV
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['close'], label='价格')
    plt.title(f'{stock_code} 价格走势')
    plt.legend()
    
    # 子图2：OBV对比
    plt.subplot(3, 1, 2)
    plt.plot(data.index, result_enhanced['obv'], label='增强型OBV')
    plt.plot(data.index, result_original['obv'], label='原始OBV')
    plt.title('OBV对比')
    plt.legend()
    
    # 子图3：评分对比
    plt.subplot(3, 1, 3)
    plt.plot(data.index, score_enhanced['final_score'], label='增强型OBV评分')
    plt.plot(data.index, score_original['final_score'], label='原始OBV评分')
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=30, color='g', linestyle='--')
    plt.title('评分对比')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'data', 'result', 'enhanced_obv_test.png'))
    plt.close()
    
    logger.info(f"结果图表已保存到 {os.path.join(root_dir, 'data', 'result', 'enhanced_obv_test.png')}")


def test_market_detector():
    """测试市场环境检测器"""
    # 获取测试数据 - 使用上证指数
    stock_code = '000001.SH'  # 上证指数
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    
    data = get_test_data(stock_code, start_date, end_date)
    if data.empty:
        logger.error("获取测试数据失败")
        return
    
    logger.info(f"获取 {stock_code} 从 {start_date} 到 {end_date} 的数据，共 {len(data)} 条记录")
    
    # 创建市场环境检测器
    detector = MarketDetector()
    
    # 使用滚动窗口检测市场环境
    window_size = 60  # 60个交易日窗口
    environments = []
    dates = []
    
    for i in range(window_size, len(data), 10):  # 每10天检测一次
        window_data = data.iloc[i-window_size:i]
        env = detector.detect_environment(window_data)
        environments.append(env.value)
        dates.append(data.index[i-1])
    
    # 统计各环境占比
    env_counts = {}
    for env in environments:
        env_counts[env] = env_counts.get(env, 0) + 1
    
    for env, count in env_counts.items():
        percentage = count / len(environments) * 100
        logger.info(f"市场环境 '{env}' 占比: {percentage:.2f}%")
    
    # 绘制结果
    plt.figure(figsize=(14, 8))
    
    # 子图1：价格走势
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'])
    plt.title(f'{stock_code} 价格走势')
    
    # 标记不同市场环境
    colors = {
        '牛市': 'red',
        '熊市': 'green',
        '震荡市': 'blue',
        '高波动市': 'purple',
        '突破市场': 'orange'
    }
    
    for i, date in enumerate(dates):
        plt.axvline(x=date, color=colors[environments[i]], alpha=0.3)
    
    # 子图2：市场环境分布
    plt.subplot(2, 1, 2)
    env_df = pd.DataFrame({'date': dates, 'environment': environments})
    
    # 将环境映射为数值以便绘图
    env_mapping = {
        '牛市': 4,
        '突破市场': 3,
        '震荡市': 2,
        '高波动市': 1,
        '熊市': 0
    }
    
    env_df['env_value'] = env_df['environment'].map(lambda x: env_mapping.get(x, 2))
    
    plt.scatter(env_df['date'], env_df['env_value'], c=env_df['environment'].map(colors))
    plt.yticks(list(env_mapping.values()), list(env_mapping.keys()))
    plt.title('市场环境分布')
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'data', 'result', 'market_environment_test.png'))
    plt.close()
    
    logger.info(f"结果图表已保存到 {os.path.join(root_dir, 'data', 'result', 'market_environment_test.png')}")


def test_indicator_synergy():
    """测试指标协同框架"""
    # 获取测试数据
    stock_code = '000001.SZ'  # 平安银行
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    
    data = get_test_data(stock_code, start_date, end_date)
    if data.empty:
        logger.error("获取测试数据失败")
        return
    
    logger.info(f"获取 {stock_code} 从 {start_date} 到 {end_date} 的数据，共 {len(data)} 条记录")
    
    # 创建市场环境检测器
    detector = MarketDetector()
    
    # 创建指标协同框架
    synergy = IndicatorSynergy(detector)
    
    # 添加增强型OBV指标
    enhanced_obv = EnhancedOBV(ma_period=30, sensitivity=1.2)
    synergy.add_indicator(enhanced_obv, weight=1.0)
    
    # 添加原始指标（用于对比）
    from indicators.obv import OBV
    from indicators.rsi import RSI
    from indicators.macd import MACD
    
    synergy.add_indicator(OBV(), weight=0.8)
    synergy.add_indicator(RSI(), weight=1.0)
    synergy.add_indicator(MACD(), weight=1.2)
    
    # 初始化指标
    synergy.initialize_indicators(data)
    
    # 计算相关性矩阵
    corr_matrix = synergy.calculate_correlation_matrix(data)
    logger.info(f"指标相关性矩阵:\n{corr_matrix}")
    
    # 调整权重
    synergy.adjust_weights_by_correlation()
    
    # 生成综合信号
    combined_signals = synergy.generate_combined_signals(data)
    
    # 检测冲突信号
    conflicts = synergy.find_conflicting_signals(data)
    if conflicts:
        logger.info(f"检测到冲突信号: {conflicts}")
    
    # 绘制结果
    plt.figure(figsize=(14, 10))
    
    # 子图1：价格走势
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'])
    plt.title(f'{stock_code} 价格走势')
    
    # 标记买入和卖出信号
    buy_signals = combined_signals[combined_signals['buy_signal']].index
    sell_signals = combined_signals[combined_signals['sell_signal']].index
    
    for date in buy_signals:
        plt.axvline(x=date, color='g', alpha=0.3)
    
    for date in sell_signals:
        plt.axvline(x=date, color='r', alpha=0.3)
    
    # 子图2：综合评分
    plt.subplot(2, 1, 2)
    plt.plot(combined_signals.index, combined_signals['score'])
    plt.axhline(y=70, color='g', linestyle='--')
    plt.axhline(y=30, color='r', linestyle='--')
    plt.title('综合评分')
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'data', 'result', 'indicator_synergy_test.png'))
    plt.close()
    
    logger.info(f"结果图表已保存到 {os.path.join(root_dir, 'data', 'result', 'indicator_synergy_test.png')}")
    
    # 输出信号统计
    buy_count = combined_signals['buy_signal'].sum()
    sell_count = combined_signals['sell_signal'].sum()
    bull_periods = combined_signals['bull_trend'].sum()
    bear_periods = combined_signals['bear_trend'].sum()
    
    logger.info(f"综合买入信号数量: {buy_count}")
    logger.info(f"综合卖出信号数量: {sell_count}")
    logger.info(f"看涨趋势天数: {bull_periods}, 占比: {bull_periods/len(combined_signals)*100:.2f}%")
    logger.info(f"看跌趋势天数: {bear_periods}, 占比: {bear_periods/len(combined_signals)*100:.2f}%")
    logger.info(f"平均置信度: {combined_signals['confidence'].mean():.2f}")


if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs(os.path.join(root_dir, 'data', 'result'), exist_ok=True)
    
    # 运行测试
    logger.info("开始测试增强型OBV指标...")
    test_enhanced_obv()
    
    logger.info("开始测试市场环境检测器...")
    test_market_detector()
    
    logger.info("开始测试指标协同框架...")
    test_indicator_synergy()
    
    logger.info("测试完成") 