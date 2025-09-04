#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修复后的ZXM_MARKET_SENTIMENT指标
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger
from indicators.sentiment_analysis import MarketSentiment

logger = get_logger(__name__)


def create_test_data(length: int = 100) -> pd.DataFrame:
    """创建测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, length)  # 日收益率
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据
    close_prices = np.array(prices[1:])  # 去掉初始价格
    data_length = len(close_prices)
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, data_length)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, data_length)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price
    
    # 生成成交量数据
    volumes = np.random.lognormal(10, 0.5, data_length)
    
    data = pd.DataFrame({
        'date': dates[:data_length],  # 确保日期长度匹配
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return data


def test_fixed_market_sentiment():
    """测试修复后的ZXM_MARKET_SENTIMENT指标"""
    logger.info("🧪 开始测试修复后的ZXM_MARKET_SENTIMENT指标...")
    
    try:
        # 创建指标实例
        indicator = MarketSentiment()
        logger.info(f"✅ 指标实例创建成功: {indicator.name}")
        
        # 创建测试数据
        test_data = create_test_data(100)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 测试calculate方法
        logger.info("🧪 测试calculate方法...")
        result = indicator.calculate(test_data)
        
        if result is None:
            logger.error("❌ calculate方法返回None")
            return False
        elif result.empty:
            logger.error("❌ calculate方法返回空DataFrame")
            return False
        else:
            logger.info(f"✅ calculate方法成功，返回形状: {result.shape}")
            logger.info(f"  - 列名: {list(result.columns)}")
            
            # 检查情绪指标列
            sentiment_columns = ['FearGreedIndex', 'InvestorSentiment', 'MarketHeat', 'CompositeSentiment']
            for col in sentiment_columns:
                if col in result.columns:
                    values = result[col].dropna()
                    if len(values) > 0:
                        logger.info(f"  - {col}: 范围 [{values.min():.2f}, {values.max():.2f}]")
                    else:
                        logger.warning(f"  - {col}: 全部为NaN")
                else:
                    logger.error(f"  - 缺少列: {col}")
        
        # 测试get_patterns方法
        logger.info("🧪 测试get_patterns方法...")
        patterns = indicator.get_patterns()
        
        if patterns is None:
            logger.error("❌ get_patterns方法返回None")
            return False
        elif not patterns:
            logger.error("❌ get_patterns方法返回空字典")
            return False
        else:
            logger.info(f"✅ get_patterns方法成功，返回键: {list(patterns.keys())}")
            
            # 检查关键信息
            if 'indicator_type' in patterns:
                logger.info(f"  - 指标类型: {patterns['indicator_type']}")
            if 'metrics' in patterns:
                logger.info(f"  - 指标度量: {patterns['metrics']}")
            if 'signals' in patterns:
                logger.info(f"  - 信号类型: {patterns['signals']}")
        
        # 测试边界情况
        logger.info("🧪 测试边界情况...")
        
        # 测试空数据
        empty_result = indicator.calculate(pd.DataFrame())
        if isinstance(empty_result, pd.DataFrame):
            logger.info(f"✅ 空数据处理成功: {empty_result.shape}")
        else:
            logger.error("❌ 空数据处理失败")
            return False
        
        # 测试少量数据
        small_data = create_test_data(10)
        small_result = indicator.calculate(small_data)
        if isinstance(small_result, pd.DataFrame):
            logger.info(f"✅ 少量数据处理成功: {small_result.shape}")
        else:
            logger.error("❌ 少量数据处理失败")
            return False
        
        logger.info("🎉 ZXM_MARKET_SENTIMENT指标修复测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = test_fixed_market_sentiment()
    
    if success:
        logger.info("✅ ZXM_MARKET_SENTIMENT指标修复测试成功！")
    else:
        logger.error("❌ ZXM_MARKET_SENTIMENT指标修复测试失败！")
    
    return success


if __name__ == "__main__":
    main()
