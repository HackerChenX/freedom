#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标测试脚本

用于测试新增或修改的技术指标
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.factory import IndicatorFactory
from utils.logger import get_logger

logger = get_logger(__name__)


def create_test_data(n_rows=100):
    """
    创建测试数据
    
    Args:
        n_rows: 数据行数
        
    Returns:
        包含OHLCV数据的DataFrame
    """
    # 创建日期索引
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(n_rows)]
    dates.reverse()
    
    # 创建模拟价格数据
    close = np.random.normal(100, 10, n_rows).cumsum() + 1000
    high = close + np.random.uniform(0, 5, n_rows)
    low = close - np.random.uniform(0, 5, n_rows)
    open_price = close - np.random.uniform(-3, 3, n_rows)
    volume = np.random.uniform(1000, 10000, n_rows) * (1 + np.sin(np.arange(n_rows) / 10))
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    df.set_index('date', inplace=True)
    return df


def test_momentum():
    """测试Momentum指标"""
    logger.info("测试Momentum指标")
    
    # 创建测试数据
    df = create_test_data(50)
    
    # 使用指标工厂创建Momentum指标
    factory = IndicatorFactory()
    momentum = factory.create_indicator('Momentum', period=10)
    
    # 计算指标
    result = momentum.compute(df)
    
    # 检查结果
    assert 'mtm' in result.columns, "结果中没有mtm列"
    assert 'signal' in result.columns, "结果中没有signal列"
    
    # 转换DatetimeIndex为数值索引以便绘图
    x = np.arange(len(df))
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    
    # 价格图
    plt.subplot(2, 1, 1)
    plt.plot(x, df['close'].values)
    plt.title('价格')
    plt.grid(True)
    
    # Momentum指标图
    plt.subplot(2, 1, 2)
    plt.plot(x, result['mtm'].values, label='MTM(10)')
    plt.plot(x, result['signal'].values, label='Signal', linestyle='--')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title('Momentum指标')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('data/result/momentum_test.png')
    logger.info("Momentum指标测试结果已保存到data/result/momentum_test.png")
    
    return result


def test_rsima():
    """测试RSIMA指标"""
    logger.info("测试RSIMA指标")
    
    # 创建测试数据 - 只创建30行数据来测试对少量数据的处理能力
    df = create_test_data(30)
    
    # 使用指标工厂创建RSIMA指标
    factory = IndicatorFactory()
    rsima = factory.create_indicator('RSIMA', rsi_period=14, ma_periods=[3, 5, 10])
    
    # 计算指标
    result = rsima.compute(df)
    
    # 检查结果
    assert 'rsi' in result.columns, "结果中没有rsi列"
    
    # 检查可用的MA列
    ma_columns = [col for col in result.columns if col.startswith('rsi_ma')]
    logger.info(f"可用的RSI均线列: {ma_columns}")
    
    # 转换DatetimeIndex为数值索引以便绘图
    x = np.arange(len(df))
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    
    # 价格图
    plt.subplot(2, 1, 1)
    plt.plot(x, df['close'].values)
    plt.title('价格')
    plt.grid(True)
    
    # RSIMA指标图
    plt.subplot(2, 1, 2)
    plt.plot(x, result['rsi'].values, label='RSI(14)')
    for ma in ma_columns:
        if not result[ma].isna().all():  # 只绘制非空的MA列
            plt.plot(x, result[ma].values, linestyle='--', label=ma)
    
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
    plt.axhline(y=50, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
    plt.title('RSIMA指标')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('data/result/rsima_test.png')
    logger.info("RSIMA指标测试结果已保存到data/result/rsima_test.png")
    
    return result


def test_all_indicators():
    """测试所有指标"""
    # 创建测试数据
    df = create_test_data(100)
    
    # 获取工厂支持的所有指标类型
    factory = IndicatorFactory()
    supported_indicators = factory.get_supported_indicators()
    
    logger.info(f"支持的指标类型: {supported_indicators}")
    
    # 测试每个指标
    results = {}
    for indicator_type in supported_indicators:
        try:
            indicator = factory.create_indicator(indicator_type)
            result = indicator.compute(df)
            # 提取指标特有的列名
            indicator_columns = [col for col in result.columns if col not in df.columns]
            results[indicator_type] = indicator_columns
            logger.info(f"指标 {indicator_type} 计算成功，输出列: {indicator_columns}")
        except Exception as e:
            logger.error(f"指标 {indicator_type} 计算失败: {str(e)}")
    
    # 保存结果
    with open('data/result/indicators_test_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("所有指标测试结果已保存到data/result/indicators_test_result.json")


if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs('data/result', exist_ok=True)
    
    # 测试Momentum指标
    test_momentum()
    
    # 测试RSIMA指标
    test_rsima()
    
    # 测试所有指标
    # test_all_indicators()
    
    logger.info("指标测试完成") 