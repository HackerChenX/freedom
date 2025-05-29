#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
简化版指标测试脚本

用于测试新实现的增强指标基本功能，不依赖图表绘制
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.unified_ma import UnifiedMA
from indicators.enhanced_macd import EnhancedMACD
from indicators.enhanced_rsi import EnhancedRSI
from indicators.factory import IndicatorFactory
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_mock_data(size: int = 200) -> pd.DataFrame:
    """
    生成模拟K线数据用于测试
    
    Args:
        size: 数据大小
        
    Returns:
        pd.DataFrame: 模拟K线数据
    """
    # 生成日期序列
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(size)]
    dates.reverse()
    
    # 生成模拟价格数据
    np.random.seed(42)  # 设置随机种子，保证可重复性
    
    # 生成基础价格走势（趋势+噪声）
    base_price = 100
    trend = np.cumsum(np.random.normal(0.001, 0.01, size))
    noise = np.random.normal(0, 0.02, size)
    price = base_price + trend + noise
    
    # 生成OHLC数据
    daily_volatility = 0.015
    
    high = price * (1 + np.random.uniform(0, daily_volatility, size))
    low = price * (1 - np.random.uniform(0, daily_volatility, size))
    open_price = low + np.random.uniform(0, 1, size) * (high - low)
    close = low + np.random.uniform(0, 1, size) * (high - low)
    
    # 生成成交量数据（与价格走势相关）
    volume_base = 100000
    volume = volume_base * (1 + np.abs(np.diff(np.append(0, price))) * 50) * (1 + np.random.uniform(-0.5, 0.5, size))
    
    # 生成成交额
    amount = volume * price
    
    # 生成换手率
    turnover = volume / 10000 * np.random.uniform(0.5, 1.5, size)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'code': ['000001.SZ'] * size,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume.astype(int),
        'amount': amount,
        'turnover': turnover
    }, index=pd.DatetimeIndex(dates, name='date'))
    
    return df


def test_unified_ma():
    """测试统一移动平均线指标"""
    logger.info("开始测试统一移动平均线指标")
    
    # 获取测试数据
    df = generate_mock_data()
    
    # 测试不同类型的MA
    ma_types = ['simple', 'ema', 'wma', 'ama', 'hma']
    periods = [5, 10, 20, 60]
    
    for ma_type in ma_types:
        # 创建指标实例
        ma = UnifiedMA(
            name=f"{ma_type.upper()}",
            periods=periods,
            ma_type=ma_type
        )
        
        # 计算指标
        ma_data = ma.compute(df)
        logger.info(f"{ma_type.upper()} 结果列: {ma_data.columns.tolist()}")
        
        # 获取信号
        signals = ma.generate_signals(df)
        
        # 统计买入卖出信号数量
        buy_count = signals['buy_signal'].sum()
        sell_count = signals['sell_signal'].sum()
        logger.info(f"{ma_type.upper()} 买入信号: {buy_count}, 卖出信号: {sell_count}")
        
        # 测试盘整判断
        if ma_type == 'simple':
            consolidation = ma.is_consolidation(period=20, threshold=0.01)
            consolidation_count = consolidation.sum()
            logger.info(f"盘整区域占比: {consolidation_count / len(df):.2%}")
    
    logger.info("统一移动平均线测试完成")


def test_enhanced_macd():
    """测试增强版MACD指标"""
    logger.info("开始测试增强版MACD指标")
    
    # 获取测试数据
    df = generate_mock_data()
    
    # 创建增强版MACD实例
    macd = EnhancedMACD(
        fast_period=12,
        slow_period=26,
        signal_period=9
    )
    
    # 计算指标
    macd_data = macd.compute(df)
    logger.info(f"MACD 结果列: {macd_data.columns.tolist()}")
    
    # 创建双MACD实例
    dual_macd = EnhancedMACD(
        fast_period=12,
        slow_period=26,
        signal_period=9,
        secondary_fast=5,
        secondary_slow=35,
        secondary_signal=5
    )
    
    # 设置使用双MACD
    dual_macd.set_parameters({'use_secondary_macd': True})
    
    # 计算指标
    dual_macd_data = dual_macd.compute(df)
    logger.info(f"双MACD 结果列: {dual_macd_data.columns.tolist()}")
    
    # 获取信号
    signals = dual_macd.generate_signals(df)
    
    # 统计买入卖出信号数量
    buy_count = signals['buy_signal'].sum()
    sell_count = signals['sell_signal'].sum()
    logger.info(f"双MACD 买入信号: {buy_count}, 卖出信号: {sell_count}")
    
    # 统计背离信号
    if 'bullish_divergence' in dual_macd_data.columns:
        bull_div_count = dual_macd_data['bullish_divergence'].sum()
        logger.info(f"MACD看涨背离数量: {bull_div_count}")
    
    if 'bearish_divergence' in dual_macd_data.columns:
        bear_div_count = dual_macd_data['bearish_divergence'].sum()
        logger.info(f"MACD看跌背离数量: {bear_div_count}")
    
    logger.info("增强版MACD测试完成")


def test_enhanced_rsi():
    """测试增强版RSI指标"""
    logger.info("开始测试增强版RSI指标")
    
    # 获取测试数据
    df = generate_mock_data()
    
    # 创建增强版RSI实例（单周期）
    rsi = EnhancedRSI(
        periods=[14],
        price_col='close'
    )
    
    # 设置参数
    rsi.set_parameters({'use_multi_period': False})
    
    # 计算指标
    rsi_data = rsi.compute(df)
    logger.info(f"单周期RSI 结果列: {rsi_data.columns.tolist()}")
    
    # 创建多周期RSI实例
    multi_rsi = EnhancedRSI(
        periods=[6, 14, 21],
        price_col='close'
    )
    
    # 计算指标
    multi_rsi_data = multi_rsi.compute(df)
    logger.info(f"多周期RSI 结果列: {multi_rsi_data.columns.tolist()}")
    
    # 获取信号
    signals = multi_rsi.generate_signals(df)
    
    # 统计买入卖出信号数量
    buy_count = signals['buy_signal'].sum()
    sell_count = signals['sell_signal'].sum()
    logger.info(f"多周期RSI 买入信号: {buy_count}, 卖出信号: {sell_count}")
    
    # 统计背离信号
    if 'bullish_divergence' in multi_rsi_data.columns:
        bull_div_count = multi_rsi_data['bullish_divergence'].sum()
        logger.info(f"RSI看涨背离数量: {bull_div_count}")
    
    if 'bearish_divergence' in multi_rsi_data.columns:
        bear_div_count = multi_rsi_data['bearish_divergence'].sum()
        logger.info(f"RSI看跌背离数量: {bear_div_count}")
    
    logger.info("增强版RSI测试完成")


def test_factory_creation():
    """测试通过工厂创建增强指标"""
    logger.info("开始测试通过工厂创建增强指标")
    
    # 获取测试数据
    df = generate_mock_data()
    
    # 通过工厂创建统一移动平均线
    unified_ma = IndicatorFactory.create("UNIFIED_MA", periods=[5, 10, 20], ma_type="ema")
    if unified_ma:
        logger.info("成功通过工厂创建统一移动平均线")
        ma_data = unified_ma.compute(df)
        logger.info(f"MA计算结果包含以下列: {ma_data.columns.tolist()}")
    else:
        logger.error("通过工厂创建统一移动平均线失败")
    
    # 通过工厂创建增强版MACD
    enhanced_macd = IndicatorFactory.create("ENHANCED_MACD", use_secondary_macd=True)
    if enhanced_macd:
        logger.info("成功通过工厂创建增强版MACD")
        macd_data = enhanced_macd.compute(df)
        logger.info(f"MACD计算结果包含以下列: {macd_data.columns.tolist()}")
    else:
        logger.error("通过工厂创建增强版MACD失败")
    
    # 通过工厂创建增强版RSI
    enhanced_rsi = IndicatorFactory.create("ENHANCED_RSI", periods=[6, 14, 21])
    if enhanced_rsi:
        logger.info("成功通过工厂创建增强版RSI")
        rsi_data = enhanced_rsi.compute(df)
        logger.info(f"RSI计算结果包含以下列: {rsi_data.columns.tolist()}")
    else:
        logger.error("通过工厂创建增强版RSI失败")
    
    logger.info("工厂创建测试完成")


if __name__ == "__main__":
    # 测试统一移动平均线
    test_unified_ma()
    
    # 测试增强版MACD
    test_enhanced_macd()
    
    # 测试增强版RSI
    test_enhanced_rsi()
    
    # 测试通过工厂创建
    test_factory_creation()
    
    logger.info("所有测试完成") 