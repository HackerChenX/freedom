#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断THREE_WHITE_SOLDIERS指标的实际状态
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

logger = get_logger(__name__)


def create_test_data_with_three_white_soldiers(length: int = 100) -> pd.DataFrame:
    """创建包含三个白兵形态的测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, length)
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据，特意插入三个白兵形态
    close_prices = np.array(prices[1:])
    data_length = len(close_prices)
    
    high_prices = []
    low_prices = []
    open_prices = []
    
    for i in range(data_length):
        close = close_prices[i]
        
        # 在第20-22位置插入三个白兵形态
        if 20 <= i <= 22:
            # 三个白兵：连续三根阳线，每根都收于接近最高点
            if i == 20:  # 第一根白兵
                open_price = close - 1.5  # 阳线：开盘价低于收盘价
                high_price = close + 0.2  # 上影线短，收盘价接近最高价
                low_price = open_price - 0.1  # 下影线短
            elif i == 21:  # 第二根白兵
                open_price = close_prices[i-1] - 0.5  # 在前一根收盘价附近开盘
                close = close_prices[i-1] + 1.2  # 收盘价高于前一根
                high_price = close + 0.15
                low_price = open_price - 0.1
                close_prices[i] = close  # 更新收盘价
            else:  # 第三根白兵
                open_price = close_prices[i-1] - 0.3
                close = close_prices[i-1] + 1.0  # 继续上涨
                high_price = close + 0.1
                low_price = open_price - 0.1
                close_prices[i] = close  # 更新收盘价
        else:
            # 正常K线
            open_price = close + np.random.normal(0, 0.5)
            high_price = max(open_price, close) + abs(np.random.normal(0, 0.3))
            low_price = min(open_price, close) - abs(np.random.normal(0, 0.3))
        
        open_prices.append(open_price)
        high_prices.append(high_price)
        low_prices.append(low_price)
    
    # 生成成交量数据
    volumes = np.random.lognormal(10, 0.5, data_length)
    
    data = pd.DataFrame({
        'date': dates[:data_length],
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return data


def diagnose_three_white_soldiers():
    """诊断THREE_WHITE_SOLDIERS指标的实际状态"""
    logger.info("🔍 诊断THREE_WHITE_SOLDIERS指标的实际状态...")
    
    try:
        # 创建测试数据
        test_data = create_test_data_with_three_white_soldiers(100)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 1. 测试CandlestickPatterns类
        logger.info("📦 测试CandlestickPatterns类...")
        try:
            from indicators.pattern.candlestick_patterns import CandlestickPatterns
            
            basic_indicator = CandlestickPatterns()
            basic_result = basic_indicator.calculate(test_data)
            
            logger.info(f"✅ CandlestickPatterns计算成功，形状: {basic_result.shape}")
            logger.info(f"  - 列名: {list(basic_result.columns)}")
            
            # 检查是否有three_white_soldiers列
            three_white_soldiers_columns = [col for col in basic_result.columns if 'three_white_soldiers' in col.lower() or 'white_soldiers' in col.lower()]
            logger.info(f"  - THREE_WHITE_SOLDIERS相关列: {three_white_soldiers_columns}")
            
            if not three_white_soldiers_columns:
                logger.warning("⚠️ CandlestickPatterns中未找到THREE_WHITE_SOLDIERS相关列")
            
        except Exception as e:
            logger.error(f"❌ CandlestickPatterns测试失败: {e}")
        
        # 2. 测试AdvancedCandlestickPatterns类
        logger.info("📦 测试AdvancedCandlestickPatterns类...")
        try:
            from indicators.pattern.advanced_candlestick_patterns import AdvancedCandlestickPatterns
            
            advanced_indicator = AdvancedCandlestickPatterns()
            advanced_result = advanced_indicator.calculate(test_data)
            
            logger.info(f"✅ AdvancedCandlestickPatterns计算成功，形状: {advanced_result.shape}")
            logger.info(f"  - 列名: {list(advanced_result.columns)}")
            
            # 检查是否有three_white_soldiers列
            three_white_soldiers_columns = [col for col in advanced_result.columns if 'three_white_soldiers' in col.lower() or '三白兵' in col or 'white_soldiers' in col.lower()]
            logger.info(f"  - THREE_WHITE_SOLDIERS相关列: {three_white_soldiers_columns}")
            
            if three_white_soldiers_columns:
                for col in three_white_soldiers_columns:
                    if col in advanced_result.columns:
                        pattern_values = advanced_result[col]
                        if pattern_values.dtype == bool:
                            pattern_count = pattern_values.sum()
                            logger.info(f"  - {col}: 检测到 {pattern_count} 个形态")
                        else:
                            logger.info(f"  - {col}: 列类型 {pattern_values.dtype}")
            else:
                logger.warning("⚠️ AdvancedCandlestickPatterns中也未找到THREE_WHITE_SOLDIERS相关列")
            
        except Exception as e:
            logger.error(f"❌ AdvancedCandlestickPatterns测试失败: {e}")
        
        # 3. 手动验证三个白兵形态
        logger.info("🧪 手动验证三个白兵形态...")
        try:
            open_prices = test_data['open']
            high_prices = test_data['high']
            low_prices = test_data['low']
            close_prices = test_data['close']
            
            # 手动识别三个白兵形态
            three_white_soldiers = pd.Series(False, index=test_data.index)
            
            for i in range(2, len(test_data)):
                # 检查连续三根K线
                if i >= 2:
                    # 三根都是阳线
                    is_bullish_1 = close_prices.iloc[i-2] > open_prices.iloc[i-2]
                    is_bullish_2 = close_prices.iloc[i-1] > open_prices.iloc[i-1]
                    is_bullish_3 = close_prices.iloc[i] > open_prices.iloc[i]
                    
                    # 每根都收于接近最高点
                    close_near_high_1 = (high_prices.iloc[i-2] - close_prices.iloc[i-2]) / (high_prices.iloc[i-2] - low_prices.iloc[i-2] + 1e-10) < 0.3
                    close_near_high_2 = (high_prices.iloc[i-1] - close_prices.iloc[i-1]) / (high_prices.iloc[i-1] - low_prices.iloc[i-1] + 1e-10) < 0.3
                    close_near_high_3 = (high_prices.iloc[i] - close_prices.iloc[i]) / (high_prices.iloc[i] - low_prices.iloc[i] + 1e-10) < 0.3
                    
                    # 连续上涨
                    rising = close_prices.iloc[i-1] > close_prices.iloc[i-2] and close_prices.iloc[i] > close_prices.iloc[i-1]
                    
                    if is_bullish_1 and is_bullish_2 and is_bullish_3 and close_near_high_1 and close_near_high_2 and close_near_high_3 and rising:
                        three_white_soldiers.iloc[i] = True
            
            manual_count = three_white_soldiers.sum()
            logger.info(f"  - 手动识别的THREE_WHITE_SOLDIERS数量: {manual_count}")
            
            if manual_count > 0:
                logger.info("✅ 测试数据中确实包含三个白兵形态")
            else:
                logger.warning("⚠️ 测试数据中未检测到三个白兵形态")
            
        except Exception as e:
            logger.error(f"❌ 手动验证失败: {e}")
        
        # 4. 测试get_patterns方法
        logger.info("🧪 测试get_patterns方法...")
        try:
            if hasattr(advanced_indicator, 'get_patterns'):
                patterns = advanced_indicator.get_patterns(test_data)
                logger.info(f"✅ get_patterns方法存在，返回类型: {type(patterns)}")
                
                if isinstance(patterns, pd.DataFrame):
                    logger.info(f"  - 形态DataFrame形状: {patterns.shape}")
                    logger.info(f"  - 形态列名: {list(patterns.columns)}")
                    
                    # 检查THREE_WHITE_SOLDIERS相关形态
                    white_soldiers_pattern_columns = [col for col in patterns.columns if 'white_soldiers' in col.lower() or '三白兵' in col]
                    logger.info(f"  - THREE_WHITE_SOLDIERS相关形态列: {white_soldiers_pattern_columns}")
                else:
                    logger.warning(f"⚠️ get_patterns返回未知类型: {type(patterns)}")
            else:
                logger.warning("⚠️ 缺少get_patterns方法")
                
        except Exception as e:
            logger.error(f"❌ get_patterns方法测试失败: {e}")
        
        logger.info("🎯 THREE_WHITE_SOLDIERS指标诊断完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 诊断过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = diagnose_three_white_soldiers()
    
    if success:
        logger.info("✅ THREE_WHITE_SOLDIERS指标诊断完成")
    else:
        logger.error("❌ THREE_WHITE_SOLDIERS指标诊断失败")
    
    return success


if __name__ == "__main__":
    main()
