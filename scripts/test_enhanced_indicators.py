#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
增强指标测试脚本

用于测试新实现的增强指标功能
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.unified_ma import UnifiedMA
from indicators.enhanced_macd import EnhancedMACD
from indicators.enhanced_rsi import EnhancedRSI
from indicators.factory import IndicatorFactory
from indicators.indicator_registry import IndicatorEnum
from utils.logger import get_logger
from db.clickhouse_db import get_clickhouse_db

logger = get_logger(__name__)


def get_test_data(stock_code: str = '000001.SZ', start_date: str = None, end_date: str = None, limit: int = 200):
    """
    获取测试用的股票数据
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        limit: 限制记录数
        
    Returns:
        pd.DataFrame: 股票K线数据
    """
    try:
        # 获取数据库连接
        db = get_clickhouse_db()
        
        # 构建查询SQL
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        sql = f"""
        SELECT 
            trade_date as date, 
            ts_code as code, 
            open, 
            high, 
            low, 
            close, 
            vol as volume,
            amount,
            turnover_rate as turnover
        FROM stock_daily
        WHERE ts_code = '{stock_code}'
        AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY trade_date
        LIMIT {limit}
        """
        
        # 执行查询
        df = db.query(sql)
        
        # 检查结果
        if df is None or len(df) == 0:
            logger.warning(f"未查询到股票 {stock_code} 的数据")
            return generate_mock_data(limit)
            
        # 设置日期索引
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"获取股票数据失败: {e}")
        return generate_mock_data(limit)


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
    df = get_test_data()
    
    # 创建图表
    fig, axs = plt.subplots(5, 1, figsize=(15, 20), gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})
    
    # 绘制价格图
    axs[0].plot(df.index, df['close'], label='收盘价')
    axs[0].set_title('价格走势')
    axs[0].legend()
    
    # 测试不同类型的MA
    ma_types = ['simple', 'ema', 'wma', 'ama', 'hma']
    periods = [5, 10, 20, 60]
    
    for i, ma_type in enumerate(ma_types):
        # 创建指标实例
        ma = UnifiedMA(
            name=f"{ma_type.upper()}",
            periods=periods,
            ma_type=ma_type
        )
        
        # 计算指标
        ma_data = ma.compute(df)
        
        # 绘制指标
        for period in periods:
            axs[i].plot(df.index, ma_data[f'MA{period}'], label=f'{ma_type.upper()}{period}')
        
        # 绘制图例
        axs[i].set_title(f'{ma_type.upper()} 移动平均线')
        axs[i].legend()
        
        # 获取信号
        signals = ma.generate_signals(df)
        
        # 标记买入信号
        buy_signals = signals[signals['buy_signal']].index
        if len(buy_signals) > 0:
            axs[0].scatter(
                buy_signals, 
                df.loc[buy_signals, 'close'], 
                marker='^', 
                color='red', 
                s=100,
                label=f'{ma_type} 买入信号' if i == 0 else ""
            )
        
        # 标记卖出信号
        sell_signals = signals[signals['sell_signal']].index
        if len(sell_signals) > 0:
            axs[0].scatter(
                sell_signals, 
                df.loc[sell_signals, 'close'], 
                marker='v', 
                color='green', 
                s=100,
                label=f'{ma_type} 卖出信号' if i == 0 else ""
            )
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'data', 'result', 'unified_ma_test.png'))
    logger.info(f"统一移动平均线测试结果已保存到: {os.path.join(root_dir, 'data', 'result', 'unified_ma_test.png')}")


def test_enhanced_macd():
    """测试增强版MACD指标"""
    logger.info("开始测试增强版MACD指标")
    
    # 获取测试数据
    df = get_test_data()
    
    # 创建图表
    fig, axs = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 绘制价格图
    axs[0].plot(df.index, df['close'], label='收盘价')
    axs[0].set_title('价格走势')
    axs[0].grid(True, alpha=0.3)
    
    # 创建增强版MACD实例
    macd = EnhancedMACD(
        fast_period=12,
        slow_period=26,
        signal_period=9
    )
    
    # 计算指标
    macd_data = macd.compute(df)
    
    # 使用内置方法绘制MACD
    macd.plot_macd(df, ax=axs[1])
    
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
    
    # 绘制主MACD的DIF和DEA
    axs[2].plot(df.index, dual_macd_data['DIF'], 'b-', label='DIF(12,26,9)')
    axs[2].plot(df.index, dual_macd_data['DEA'], 'r-', label='DEA(12,26,9)')
    
    # 绘制第二组MACD的DIF和DEA
    axs[2].plot(df.index, dual_macd_data['DIF_2'], 'g-', label='DIF(5,35,5)')
    axs[2].plot(df.index, dual_macd_data['DEA_2'], 'y-', label='DEA(5,35,5)')
    
    # 绘制零轴
    axs[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[2].set_title('双MACD对比')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    # 获取信号
    signals = dual_macd.generate_signals(df)
    
    # 标记买入信号
    buy_signals = signals[signals['buy_signal']].index
    if len(buy_signals) > 0:
        axs[0].scatter(
            buy_signals, 
            df.loc[buy_signals, 'close'], 
            marker='^', 
            color='red', 
            s=100,
            label='MACD买入信号'
        )
    
    # 标记卖出信号
    sell_signals = signals[signals['sell_signal']].index
    if len(sell_signals) > 0:
        axs[0].scatter(
            sell_signals, 
            df.loc[sell_signals, 'close'], 
            marker='v', 
            color='green', 
            s=100,
            label='MACD卖出信号'
        )
    
    # 标记背离
    if 'bullish_divergence' in dual_macd_data.columns:
        bull_div = df.index[dual_macd_data['bullish_divergence']]
        if len(bull_div) > 0:
            axs[0].scatter(
                bull_div, 
                df.loc[bull_div, 'close'], 
                marker='*', 
                color='lime', 
                s=200,
                label='MACD看涨背离'
            )
    
    if 'bearish_divergence' in dual_macd_data.columns:
        bear_div = df.index[dual_macd_data['bearish_divergence']]
        if len(bear_div) > 0:
            axs[0].scatter(
                bear_div, 
                df.loc[bear_div, 'close'], 
                marker='*', 
                color='orangered', 
                s=200,
                label='MACD看跌背离'
            )
    
    axs[0].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'data', 'result', 'enhanced_macd_test.png'))
    logger.info(f"增强版MACD测试结果已保存到: {os.path.join(root_dir, 'data', 'result', 'enhanced_macd_test.png')}")


def test_enhanced_rsi():
    """测试增强版RSI指标"""
    logger.info("开始测试增强版RSI指标")
    
    # 获取测试数据
    df = get_test_data()
    
    # 创建图表
    fig, axs = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 绘制价格图
    axs[0].plot(df.index, df['close'], label='收盘价')
    axs[0].set_title('价格走势')
    axs[0].grid(True, alpha=0.3)
    
    # 创建增强版RSI实例（单周期）
    rsi = EnhancedRSI(
        periods=[14],
        price_col='close'
    )
    
    # 设置参数
    rsi.set_parameters({'use_multi_period': False})
    
    # 计算指标
    rsi_data = rsi.compute(df)
    
    # 使用内置方法绘制RSI
    rsi.plot_rsi(df, ax=axs[1])
    
    # 创建多周期RSI实例
    multi_rsi = EnhancedRSI(
        periods=[6, 14, 21],
        price_col='close'
    )
    
    # 计算指标
    multi_rsi_data = multi_rsi.compute(df)
    
    # 绘制多周期RSI
    axs[2].plot(df.index, multi_rsi_data['RSI6'], 'b-', label='RSI6')
    axs[2].plot(df.index, multi_rsi_data['RSI14'], 'r-', label='RSI14')
    axs[2].plot(df.index, multi_rsi_data['RSI21'], 'g-', label='RSI21')
    
    # 绘制RSI动量
    if 'RSI_momentum' in multi_rsi_data.columns:
        ax_twin = axs[2].twinx()
        ax_twin.plot(df.index, multi_rsi_data['RSI_momentum'], 'c--', label='RSI动量')
        ax_twin.set_ylabel('RSI动量')
        ax_twin.legend(loc='upper right')
    
    # 绘制超买超卖线
    axs[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
    axs[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
    axs[2].axhspan(40, 60, alpha=0.1, color='gray')
    
    axs[2].set_title('多周期RSI对比')
    axs[2].legend(loc='upper left')
    axs[2].set_ylim(0, 100)
    axs[2].grid(True, alpha=0.3)
    
    # 获取信号
    signals = multi_rsi.generate_signals(df)
    
    # 标记买入信号
    buy_signals = signals[signals['buy_signal']].index
    if len(buy_signals) > 0:
        axs[0].scatter(
            buy_signals, 
            df.loc[buy_signals, 'close'], 
            marker='^', 
            color='red', 
            s=100,
            label='RSI买入信号'
        )
    
    # 标记卖出信号
    sell_signals = signals[signals['sell_signal']].index
    if len(sell_signals) > 0:
        axs[0].scatter(
            sell_signals, 
            df.loc[sell_signals, 'close'], 
            marker='v', 
            color='green', 
            s=100,
            label='RSI卖出信号'
        )
    
    # 标记背离
    if 'bullish_divergence' in multi_rsi_data.columns:
        bull_div = df.index[multi_rsi_data['bullish_divergence']]
        if len(bull_div) > 0:
            axs[0].scatter(
                bull_div, 
                df.loc[bull_div, 'close'], 
                marker='*', 
                color='lime', 
                s=200,
                label='RSI看涨背离'
            )
    
    if 'bearish_divergence' in multi_rsi_data.columns:
        bear_div = df.index[multi_rsi_data['bearish_divergence']]
        if len(bear_div) > 0:
            axs[0].scatter(
                bear_div, 
                df.loc[bear_div, 'close'], 
                marker='*', 
                color='orangered', 
                s=200,
                label='RSI看跌背离'
            )
    
    axs[0].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'data', 'result', 'enhanced_rsi_test.png'))
    logger.info(f"增强版RSI测试结果已保存到: {os.path.join(root_dir, 'data', 'result', 'enhanced_rsi_test.png')}")


def test_factory_creation():
    """测试通过工厂创建增强指标"""
    logger.info("开始测试通过工厂创建增强指标")
    
    # 获取测试数据
    df = get_test_data()
    
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


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(os.path.join(root_dir, 'data', 'result'), exist_ok=True)
    
    # 测试统一移动平均线
    test_unified_ma()
    
    # 测试增强版MACD
    test_enhanced_macd()
    
    # 测试增强版RSI
    test_enhanced_rsi()
    
    # 测试通过工厂创建
    test_factory_creation()
    
    logger.info("所有测试完成") 