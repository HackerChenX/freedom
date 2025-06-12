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
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.unified_ma import UnifiedMA
from indicators.enhanced_macd import EnhancedMACD
from indicators.enhanced_rsi import EnhancedRSI
from indicators.factory import IndicatorFactory
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


def test_enhanced_macd():
    """测试增强版MACD指标"""
    pass


def test_enhanced_rsi():
    """测试增强版RSI指标"""
    logger.info("开始测试增强版RSI指标")
    
    # 获取测试数据
    df = get_test_data()
    
    # 创建增强版RSI实例（单周期）
    rsi = EnhancedRSI(
        period=14,
        multi_periods=None
    )
    
    # 计算指标
    rsi_data = rsi.calculate(df)
    
    # 创建增强版RSI实例（多周期）
    multi_rsi = EnhancedRSI(
        period=14,
        multi_periods=[9, 14, 21]
    )
    
    # 计算指标
    multi_rsi_data = multi_rsi.calculate(df)
    
    # 标记买卖信号
    signals = multi_rsi.generate_signals(multi_rsi_data)
    
    logger.info("增强版RSI测试完成")


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
    # test_unified_ma()
    
    # 测试增强版MACD
    # test_enhanced_macd()
    
    # 测试增强版RSI
    test_enhanced_rsi()
    
    # 测试通过工厂创建
    test_factory_creation()
    
    logger.info("所有测试完成") 